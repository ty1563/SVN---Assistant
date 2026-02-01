import cv2
import time
import os
import numpy as np
from collections import Counter
from typing import Optional, Generator, Tuple, List, Dict
from core.detector import Detector, Detection
from core.classifier import SpeedClassifier
from config.settings import Settings
from config.constants import MODELS_DIR


class SignTracker:
    def __init__(self, sign_id: int, votes_needed: int = 5):
        self.sign_id = sign_id
        self.votes_needed = votes_needed
        self.votes: List[str] = []
        self.final_result: Optional[str] = None
        self.last_seen = time.time()
    
    def add_vote(self, label: str, instant_complete: bool = False):
        if self.final_result:
            return
        self.votes.append(label)
        self.last_seen = time.time()
        if instant_complete or len(self.votes) >= self.votes_needed:
            self.final_result = Counter(self.votes).most_common(1)[0][0]
    
    @property
    def is_complete(self) -> bool:
        return self.final_result is not None
    
    @property
    def progress(self) -> str:
        return f"{len(self.votes)}/{self.votes_needed}"


class MultiSignState:
    def __init__(self, votes_needed: int = 5, timeout: float = 2.0):
        self.votes_needed = votes_needed
        self.timeout = timeout
        self.trackers: Dict[int, SignTracker] = {}
        self._next_id = 0
    
    def _get_tracker_id(self, bbox: tuple) -> int:
        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        for tid, tracker in self.trackers.items():
            if hasattr(tracker, 'center'):
                dx, dy = abs(cx - tracker.center[0]), abs(cy - tracker.center[1])
                if dx < 100 and dy < 100:
                    tracker.center = (cx, cy)
                    return tid
        
        new_id = self._next_id
        self._next_id += 1
        self.trackers[new_id] = SignTracker(new_id, self.votes_needed)
        self.trackers[new_id].center = (cx, cy)
        return new_id
    
    def add_vote(self, bbox: tuple, label: str, instant_complete: bool = False):
        tid = self._get_tracker_id(bbox)
        self.trackers[tid].add_vote(label, instant_complete)
    
    def cleanup(self):
        now = time.time()
        expired = [tid for tid, t in self.trackers.items() if now - t.last_seen > self.timeout and not t.is_complete]
        for tid in expired:
            del self.trackers[tid]
    
    @property
    def results(self) -> List[str]:
        return [t.final_result for t in self.trackers.values() if t.is_complete]
    
    @property
    def progress_list(self) -> List[str]:
        return [t.progress for t in self.trackers.values() if not t.is_complete]
    
    def reset(self):
        self.trackers.clear()


class FrameProcessor:
    CLASSIFY_TRIGGER = "P.127"
    
    def __init__(self, detector: Detector, settings: Settings):
        self.detector = detector
        self.settings = settings
        self.classifier: Optional[SpeedClassifier] = None
        self.sign_state = MultiSignState(votes_needed=5)
        self._cap: Optional[cv2.VideoCapture] = None
        self._stats = {"total": [], "yolo": []}
        self._init_classifier()
    
    def _init_classifier(self):
        classifier_path = os.path.join(MODELS_DIR, self.settings.detection.classifier_model)
        if os.path.exists(classifier_path):
            self.classifier = SpeedClassifier()
            self.classifier.load(classifier_path)
    
    @property
    def fps(self) -> int:
        return self.settings.detection.frames_per_second
    
    @property
    def time_budget(self) -> float:
        return 1000 / self.fps
    
    def open_camera(self, camera_id: int = 0) -> bool:
        self._cap = cv2.VideoCapture(camera_id)
        return self._cap.isOpened()
    
    def open_video(self, path: str) -> bool:
        self._cap = cv2.VideoCapture(path)
        return self._cap.isOpened()
    
    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None
    
    def _get_skip_frames(self) -> int:
        if not self._cap:
            return 1
        video_fps = self._cap.get(cv2.CAP_PROP_FPS)
        return max(1, int(video_fps / self.fps))
    
    def _process_detections(self, frame: np.ndarray, detections: List[Detection]) -> List[Detection]:
        active_classes = self.settings.detection.target_classes
        filtered = [d for d in detections if d.label in active_classes] if active_classes else detections
        
        for det in filtered:
            if det.label == self.CLASSIFY_TRIGGER and self.classifier and self.classifier.is_loaded:
                sub_label, sub_conf = self.classifier.classify_crop(frame, det.bbox)
                if sub_label and sub_conf > 0.3:
                    det.label = sub_label
                    det.conf = sub_conf
                    if sub_conf > 0.9:
                        self.sign_state.add_vote(det.bbox, sub_label, instant_complete=True)
                    else:
                        self.sign_state.add_vote(det.bbox, sub_label)
            else:
                self.sign_state.add_vote(det.bbox, det.label)
        
        self.sign_state.cleanup()
        return filtered
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        t0 = time.perf_counter()
        detections = self.detector.detect(
            frame,
            conf=self.settings.detection.conf_threshold,
            imgsz=self.settings.detection.input_size
        )
        
        detections = self._process_detections(frame, detections)
        time_ms = (time.perf_counter() - t0) * 1000
        self._stats["total"].append(time_ms)
        return detections, time_ms
    
    def stream_camera(self) -> Generator[Tuple[np.ndarray, List[Detection], float], None, None]:
        if not self._cap:
            return
        
        last_time = time.perf_counter()
        interval = 1.0 / self.fps
        
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            
            now = time.perf_counter()
            if now - last_time < interval:
                continue
            last_time = now
            
            detections, time_ms = self.process_frame(frame)
            yield frame, detections, time_ms
    
    def stream_video(self) -> Generator[Tuple[np.ndarray, List[Detection], float], None, None]:
        if not self._cap:
            return
        
        skip = self._get_skip_frames()
        frame_idx = 0
        
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            
            frame_idx += 1
            if frame_idx % skip != 0:
                continue
            
            detections, time_ms = self.process_frame(frame)
            yield frame, detections, time_ms
    
    def get_avg_time(self) -> float:
        if not self._stats["total"]:
            return 0
        return sum(self._stats["total"]) / len(self._stats["total"])


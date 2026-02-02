import cv2
import numpy as np
from typing import List, Dict
from utils.logger import log


class Detection:
    def __init__(self, bbox: tuple, conf: float, label: str, class_id: int):
        self.bbox = bbox
        self.conf = conf
        self.label = label
        self.class_id = class_id
    
    def to_dict(self) -> Dict:
        return {
            "bbox": self.bbox,
            "conf": self.conf,
            "label": self.label,
            "class_id": self.class_id
        }


class ONNXDetector:
    def __init__(self):
        self._net = None
        self._model_path = None
        self._class_names = []
    
    @property
    def is_loaded(self) -> bool:
        return self._net is not None
    
    @property
    def model_path(self) -> str:
        return self._model_path
    
    @property
    def class_names(self) -> List[str]:
        return self._class_names
    
    def load(self, model_path: str, class_names: List[str] = None) -> bool:
        try:
            self._net = cv2.dnn.readNetFromONNX(model_path)
            self._model_path = model_path
            self._class_names = class_names or []
            log.info(f"ONNX model loaded: {model_path}")
            return True
        except Exception as e:
            log.error(f"Failed to load ONNX: {e}")
            return False
    
    def unload(self):
        self._net = None
        self._model_path = None
    
    def detect(self, image: np.ndarray, conf: float = 0.5, imgsz: int = 320) -> List[Detection]:
        if not self._net:
            return []
        
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (imgsz, imgsz), swapRB=True, crop=False)
        self._net.setInput(blob)
        outputs = self._net.forward()
        
        detections = []
        output = outputs[0].T
        
        for row in output:
            scores = row[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence < conf:
                continue
            
            cx, cy, bw, bh = row[:4]
            x1 = int((cx - bw/2) * w / imgsz)
            y1 = int((cy - bh/2) * h / imgsz)
            x2 = int((cx + bw/2) * w / imgsz)
            y2 = int((cy + bh/2) * h / imgsz)
            
            label = self._class_names[class_id] if class_id < len(self._class_names) else str(class_id)
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                conf=float(confidence),
                label=label,
                class_id=int(class_id)
            ))
        
        return self._nms(detections)
    
    def _nms(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        if not detections:
            return []
        
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.conf for d in detections])
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold
        )
        
        return [detections[i] for i in indices.flatten()] if len(indices) > 0 else []

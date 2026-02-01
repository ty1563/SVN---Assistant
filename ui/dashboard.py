import cv2
import numpy as np
from typing import List, Optional, Dict
from config.settings import Settings
from services.model_service import ModelService


class Dashboard:
    BG_COLOR = (40, 40, 40)
    PANEL_COLOR = (50, 50, 50)
    TEXT_COLOR = (220, 220, 220)
    ACCENT_COLOR = (0, 200, 150)
    ACTIVE_COLOR = (0, 255, 100)
    INACTIVE_COLOR = (100, 100, 100)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    def __init__(self, settings: Settings, model_service: ModelService, width: int = 1280, height: int = 720):
        self.settings = settings
        self.model_service = model_service
        self.width = width
        self.height = height
        self.camera_size = (400, 300)
        self.active_classes: Dict[str, bool] = {}
        self.selected_class_idx = 0
        self._init_classes()
    
    def _init_classes(self):
        names = self.model_service.detector.class_names
        for name in names:
            self.active_classes[name] = True
        self._sync_settings()
    
    def _sync_settings(self):
        self.settings.detection.target_classes = [k for k, v in self.active_classes.items() if v]
    
    def handle_key(self, key: int) -> bool:
        names = list(self.active_classes.keys())
        if not names:
            return False
        
        if key == ord('w') or key == 82:
            self.selected_class_idx = (self.selected_class_idx - 1) % len(names)
            return True
        elif key == ord('s') or key == 84:
            self.selected_class_idx = (self.selected_class_idx + 1) % len(names)
            return True
        elif key == ord(' ') or key == 13:
            name = names[self.selected_class_idx]
            self.active_classes[name] = not self.active_classes[name]
            self._sync_settings()
            return True
        return False
    
    def render(
        self,
        camera_frame: np.ndarray,
        detections: list,
        time_ms: float,
        sign_results: List[str],
        sign_progress: List[str]
    ) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), self.BG_COLOR, dtype=np.uint8)
        
        self._draw_camera_panel(canvas, camera_frame, detections)
        self._draw_class_panel(canvas)
        self._draw_result_panel(canvas, sign_results, sign_progress)
        self._draw_stats_panel(canvas, time_ms, len(detections))
        self._draw_help(canvas)
        
        return canvas
    
    def _draw_camera_panel(self, canvas: np.ndarray, frame: np.ndarray, detections: list):
        x, y = 20, 20
        w, h = self.camera_size
        
        cv2.rectangle(canvas, (x-2, y-2), (x+w+2, y+h+2), self.ACCENT_COLOR, 2)
        
        resized = cv2.resize(frame, (w, h))
        
        for det in detections:
            ox1, oy1, ox2, oy2 = det.bbox
            fh, fw = frame.shape[:2]
            sx, sy = w / fw, h / fh
            x1, y1 = int(ox1 * sx), int(oy1 * sy)
            x2, y2 = int(ox2 * sx), int(oy2 * sy)
            cv2.rectangle(resized, (x1, y1), (x2, y2), self.ACTIVE_COLOR, 2)
            cv2.putText(resized, f"{det.label}", (x1, y1 - 3), self.FONT, 0.4, self.ACTIVE_COLOR, 1)
        
        canvas[y:y+h, x:x+w] = resized
        
        cv2.putText(canvas, "CAMERA", (x, y + h + 20), self.FONT, 0.5, self.TEXT_COLOR, 1)
    
    def _draw_class_panel(self, canvas: np.ndarray):
        x, y = 450, 20
        w, h = 350, 400
        
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.PANEL_COLOR, -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.ACCENT_COLOR, 1)
        cv2.putText(canvas, "DETECT CLASSES", (x+10, y+25), self.FONT, 0.6, self.ACCENT_COLOR, 1)
        cv2.putText(canvas, "(Filter which classes to show)", (x+10, y+45), self.FONT, 0.35, self.INACTIVE_COLOR, 1)
        
        names = list(self.active_classes.keys())
        for i, name in enumerate(names):
            ty = y + 70 + i * 25
            if ty > y + h - 20:
                break
            
            active = self.active_classes[name]
            selected = i == self.selected_class_idx
            
            if selected:
                cv2.rectangle(canvas, (x+5, ty-15), (x+w-5, ty+8), (70, 70, 70), -1)
            
            indicator = "[X]" if active else "[ ]"
            color = self.ACTIVE_COLOR if active else self.INACTIVE_COLOR
            prefix = "> " if selected else "  "
            
            cv2.putText(canvas, f"{prefix}{indicator} {name}", (x+10, ty), self.FONT, 0.45, color, 1)
    
    def _draw_result_panel(self, canvas: np.ndarray, results: List[str], progress: List[str]):
        x, y = 450, 440
        w, h = 350, 250
        
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.PANEL_COLOR, -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.ACCENT_COLOR, 1)
        cv2.putText(canvas, "CLASSIFICATION RESULTS", (x+10, y+25), self.FONT, 0.5, self.ACCENT_COLOR, 1)
        
        ty = y + 55
        for res in results:
            cv2.putText(canvas, res, (x+20, ty), self.FONT, 0.8, self.ACTIVE_COLOR, 2)
            ty += 35
        
        if progress:
            cv2.putText(canvas, "Voting:", (x+20, ty), self.FONT, 0.5, self.TEXT_COLOR, 1)
            ty += 25
            for prog in progress:
                cv2.putText(canvas, f"  {prog}", (x+20, ty), self.FONT, 0.45, self.INACTIVE_COLOR, 1)
                ty += 20
    
    def _draw_stats_panel(self, canvas: np.ndarray, time_ms: float, det_count: int):
        x, y = 820, 20
        w, h = 200, 100
        
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.PANEL_COLOR, -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.ACCENT_COLOR, 1)
        cv2.putText(canvas, "STATS", (x+10, y+25), self.FONT, 0.5, self.ACCENT_COLOR, 1)
        cv2.putText(canvas, f"Time: {time_ms:.0f}ms", (x+10, y+50), self.FONT, 0.45, self.TEXT_COLOR, 1)
        cv2.putText(canvas, f"Detections: {det_count}", (x+10, y+75), self.FONT, 0.45, self.TEXT_COLOR, 1)
    
    def _draw_help(self, canvas: np.ndarray):
        x, y = 20, self.height - 40
        cv2.putText(canvas, "W/S: Navigate  SPACE: Toggle class  Q: Quit", (x, y), self.FONT, 0.45, self.INACTIVE_COLOR, 1)

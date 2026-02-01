import cv2
import numpy as np
from typing import List, Optional


class Visualizer:
    BOX_COLOR = (0, 255, 0)
    TEXT_COLOR = (0, 255, 0)
    WARNING_COLOR = (0, 0, 255)
    RESULT_COLOR = (255, 200, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    def __init__(self, time_budget: float = 200):
        self.time_budget = time_budget
    
    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        result = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            label = f"{det.label} {det.conf:.2f}"
            cv2.rectangle(result, (x1, y1), (x2, y2), self.BOX_COLOR, 2)
            cv2.putText(result, label, (x1, y1 - 5), self.FONT, 0.5, self.BOX_COLOR, 2)
        
        return result
    
    def draw_status(self, frame: np.ndarray, time_ms: float, det_count: int) -> np.ndarray:
        result = frame.copy()
        status = f"T:{time_ms:.0f}ms Det:{det_count}"
        color = self.TEXT_COLOR if time_ms <= self.time_budget else self.WARNING_COLOR
        cv2.putText(result, status, (10, 30), self.FONT, 0.7, color, 2)
        return result
    
    def draw_sign_result(
        self, 
        frame: np.ndarray, 
        final_result: Optional[str], 
        progress: str
    ) -> np.ndarray:
        result = frame.copy()
        h, w = result.shape[:2]
        
        if final_result:
            cv2.rectangle(result, (w - 200, 10), (w - 10, 60), (0, 0, 0), -1)
            cv2.putText(result, final_result, (w - 190, 45), self.FONT, 1.0, self.RESULT_COLOR, 2)
        else:
            cv2.putText(result, f"Vote: {progress}", (w - 120, 30), self.FONT, 0.6, self.TEXT_COLOR, 2)
        
        return result
    
    def render(
        self,
        frame: np.ndarray,
        detections: list,
        time_ms: float,
        sign_result: Optional[str] = None,
        sign_progress: str = "0/5"
    ) -> np.ndarray:
        result = self.draw_detections(frame, detections)
        result = self.draw_status(result, time_ms, len(detections))
        result = self.draw_sign_result(result, sign_result, sign_progress)
        return result

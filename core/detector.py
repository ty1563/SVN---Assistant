import numpy as np
from typing import List, Dict, Optional
from ultralytics import YOLO
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


class Detector:
    def __init__(self):
        self._model: Optional[YOLO] = None
        self._model_path: Optional[str] = None
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    @property
    def model_path(self) -> Optional[str]:
        return self._model_path
    
    @property
    def class_names(self) -> List[str]:
        if self._model:
            return list(self._model.names.values())
        return []
    
    def load(self, model_path: str) -> bool:
        try:
            self._model = YOLO(model_path)
            self._model_path = model_path
            log.info(f"Model loaded: {model_path}")
            return True
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            return False
    
    def unload(self):
        self._model = None
        self._model_path = None
    
    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.7,
        imgsz: int = 320
    ) -> List[Detection]:
        if not self._model:
            return []
        
        results = self._model(image, conf=conf, imgsz=imgsz, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    conf=float(box.conf[0]),
                    label=self._model.names[class_id],
                    class_id=class_id
                ))
        
        return detections

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Optional, Tuple
from utils.logger import log


class SpeedClassifier:
    CLASSES = [
        "P.127-5", "P.127-10", "P.127-15", "P.127-20", "P.127-25",
        "P.127-30", "P.127-35", "P.127-40", "P.127-45", "P.127-50",
        "P.127-55", "P.127-60", "P.127-65", "P.127-70", "P.127-75",
        "P.127-80", "P.127-85", "P.127-90", "P.127-95", "P.127-100",
        "P.127-110", "P.127-120"
    ]
    
    def __init__(self, input_size: int = 64):
        self._model: Optional[nn.Module] = None
        self._model_path: Optional[str] = None
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    def _build_model(self, num_classes: int) -> nn.Module:
        class SpeedNet(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return SpeedNet(num_classes)
    
    def load(self, model_path: str) -> bool:
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                if "classes" in checkpoint:
                    self.CLASSES = checkpoint["classes"]
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            self._model = self._build_model(len(self.CLASSES))
            self._model.load_state_dict(state_dict)
            self._model.to(self.device)
            self._model.eval()
            self._model_path = model_path
            log.info(f"Classifier loaded: {model_path} ({len(self.CLASSES)} classes)")
            return True
        except Exception as e:
            log.error(f"Failed to load classifier: {e}")
            return False
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        img = cv2.resize(image, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0).to(self.device)
    
    def classify(self, image: np.ndarray) -> Tuple[str, float]:
        if not self._model:
            return "", 0.0
        
        with torch.no_grad():
            tensor = self.preprocess(image)
            output = self._model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, dim=1)
            return self.CLASSES[idx.item()], conf.item()
    
    def classify_crop(self, frame: np.ndarray, bbox: tuple) -> Tuple[str, float]:
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "", 0.0
        return self.classify(crop)

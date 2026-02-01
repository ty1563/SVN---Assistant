import cv2
import numpy as np
from typing import List, Callable
from config.settings import Settings
from services.model_service import ModelService


class SettingsMenu:
    def __init__(self, settings: Settings, model_service: ModelService):
        self.settings = settings
        self.model_service = model_service
        self.visible = False
        self.selected = 0
        self.items = [
            ("Model", self._get_model, self._cycle_model),
            ("FPS", self._get_fps, self._cycle_fps),
            ("Conf", self._get_conf, self._cycle_conf),
            ("Target", self._get_target, self._cycle_target),
        ]
        self._all_classes = []
    
    def _get_model(self) -> str:
        return self.settings.detection.model_name
    
    def _cycle_model(self, delta: int):
        models = self.model_service.list_models()
        if not models:
            return
        names = [m.name for m in models]
        idx = names.index(self.settings.detection.model_name) if self.settings.detection.model_name in names else 0
        idx = (idx + delta) % len(names)
        self.settings.detection.model_name = names[idx]
    
    def _get_fps(self) -> str:
        return str(self.settings.detection.frames_per_second)
    
    def _cycle_fps(self, delta: int):
        options = [3, 5, 10, 15, 30]
        idx = options.index(self.settings.detection.frames_per_second) if self.settings.detection.frames_per_second in options else 1
        idx = (idx + delta) % len(options)
        self.settings.detection.frames_per_second = options[idx]
    
    def _get_conf(self) -> str:
        return f"{self.settings.detection.conf_threshold:.1f}"
    
    def _cycle_conf(self, delta: int):
        val = self.settings.detection.conf_threshold + delta * 0.1
        self.settings.detection.conf_threshold = max(0.1, min(0.9, val))
    
    def _get_target(self) -> str:
        targets = self.settings.detection.target_classes
        return ", ".join(targets) if targets else "All"
    
    def _cycle_target(self, delta: int):
        if not self._all_classes:
            self._all_classes = self.model_service.detector.class_names
        targets = self.settings.detection.target_classes
        if not targets:
            targets = self._all_classes[:1] if self._all_classes else []
        else:
            idx = self._all_classes.index(targets[0]) if targets[0] in self._all_classes else 0
            idx = (idx + delta) % len(self._all_classes)
            targets = [self._all_classes[idx]]
        self.settings.detection.target_classes = targets
    
    def toggle(self):
        self.visible = not self.visible
        if not self.visible:
            self.settings.save()
    
    def handle_key(self, key: int) -> bool:
        if key == ord('m'):
            self.toggle()
            return True
        if not self.visible:
            return False
        if key == ord('w') or key == 82:
            self.selected = (self.selected - 1) % len(self.items)
        elif key == ord('s') or key == 84:
            self.selected = (self.selected + 1) % len(self.items)
        elif key == ord('a') or key == 81:
            self.items[self.selected][2](-1)
        elif key == ord('d') or key == 83:
            self.items[self.selected][2](1)
        return True
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        if not self.visible:
            return frame
        
        result = frame.copy()
        overlay = result.copy()
        h, w = result.shape[:2]
        
        menu_w, menu_h = 300, 30 + len(self.items) * 35
        x, y = 20, 60
        
        cv2.rectangle(overlay, (x, y), (x + menu_w, y + menu_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, result, 0.15, 0, result)
        
        cv2.putText(result, "SETTINGS (M to close)", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        for i, (name, getter, _) in enumerate(self.items):
            ty = y + 55 + i * 35
            color = (0, 255, 255) if i == self.selected else (200, 200, 200)
            prefix = "> " if i == self.selected else "  "
            cv2.putText(result, f"{prefix}{name}: {getter()}", (x + 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.putText(result, "W/S: Select  A/D: Change", (x + 10, y + menu_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return result

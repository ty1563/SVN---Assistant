import os
import json
from dataclasses import dataclass
from typing import List, Optional
from config.constants import MODELS_DIR
from core.onnx_detector import ONNXDetector, Detection
from utils.logger import log
from utils.file_handler import FileHandler


@dataclass
class LocalModel:
    name: str
    path: str
    size: int
    md5: Optional[str] = None


class ModelService:
    SUPPORTED_EXTENSIONS = ('.onnx',)
    
    def __init__(self):
        self._detector: Optional[ONNXDetector] = None
        self._current_model: Optional[LocalModel] = None
        FileHandler.ensure_dir(MODELS_DIR)
    
    @property
    def detector(self) -> Optional[ONNXDetector]:
        return self._detector
    
    @property
    def current_model(self) -> Optional[LocalModel]:
        return self._current_model
    
    def list_models(self) -> List[LocalModel]:
        models = []
        for f in os.listdir(MODELS_DIR):
            if f.endswith(self.SUPPORTED_EXTENSIONS):
                path = os.path.join(MODELS_DIR, f)
                models.append(LocalModel(
                    name=f,
                    path=path,
                    size=os.path.getsize(path),
                    md5=FileHandler.md5(path)
                ))
        return models
    
    def _load_class_names(self, model_name: str) -> List[str]:
        base_name = os.path.splitext(model_name)[0]
        json_path = os.path.join(MODELS_DIR, f"{base_name}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('names', [])
        return []
    
    def load_model(self, model_name: str) -> bool:
        if not model_name.endswith(self.SUPPORTED_EXTENSIONS):
            for ext in self.SUPPORTED_EXTENSIONS:
                test_path = os.path.join(MODELS_DIR, model_name + ext)
                if os.path.exists(test_path):
                    model_name += ext
                    break
        
        path = os.path.join(MODELS_DIR, model_name)
        
        if not os.path.exists(path):
            log.error(f"Model not found: {path}")
            return False
        
        self._detector = ONNXDetector()
        class_names = self._load_class_names(model_name)
        success = self._detector.load(path, class_names)
        
        if success:
            self._current_model = LocalModel(
                name=model_name,
                path=path,
                size=os.path.getsize(path),
                md5=FileHandler.md5(path)
            )
            return True
        return False
    
    def switch_model(self, model_name: str) -> bool:
        self.unload()
        return self.load_model(model_name)
    
    def unload(self):
        if self._detector:
            self._detector.unload()
        self._detector = None
        self._current_model = None
    
    def delete_model(self, model_name: str) -> bool:
        if self._current_model and self._current_model.name == model_name:
            log.warning("Cannot delete currently loaded model")
            return False
        
        path = os.path.join(MODELS_DIR, model_name)
        return FileHandler.safe_delete(path)


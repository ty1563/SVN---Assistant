import os
from dataclasses import dataclass
from typing import List, Optional
from config.constants import MODELS_DIR, MODEL_EXTENSION
from core.detector import Detector
from utils.logger import log
from utils.file_handler import FileHandler


@dataclass
class LocalModel:
    name: str
    path: str
    size: int
    md5: Optional[str] = None


class ModelService:
    def __init__(self):
        self.detector = Detector()
        self._current_model: Optional[LocalModel] = None
        FileHandler.ensure_dir(MODELS_DIR)
    
    @property
    def current_model(self) -> Optional[LocalModel]:
        return self._current_model
    
    def list_models(self) -> List[LocalModel]:
        models = []
        for f in os.listdir(MODELS_DIR):
            if f.endswith(MODEL_EXTENSION):
                path = os.path.join(MODELS_DIR, f)
                models.append(LocalModel(
                    name=f,
                    path=path,
                    size=os.path.getsize(path),
                    md5=FileHandler.md5(path)
                ))
        return models
    
    def load_model(self, model_name: str) -> bool:
        if not model_name.endswith(MODEL_EXTENSION):
            model_name += MODEL_EXTENSION
        
        path = os.path.join(MODELS_DIR, model_name)
        
        if not os.path.exists(path):
            log.error(f"Model not found: {path}")
            return False
        
        if self.detector.load(path):
            self._current_model = LocalModel(
                name=model_name,
                path=path,
                size=os.path.getsize(path),
                md5=FileHandler.md5(path)
            )
            return True
        return False
    
    def switch_model(self, model_name: str) -> bool:
        self.detector.unload()
        return self.load_model(model_name)
    
    def unload(self):
        self.detector.unload()
        self._current_model = None
    
    def delete_model(self, model_name: str) -> bool:
        if self._current_model and self._current_model.name == model_name:
            log.warning("Cannot delete currently loaded model")
            return False
        
        path = os.path.join(MODELS_DIR, model_name)
        return FileHandler.safe_delete(path)

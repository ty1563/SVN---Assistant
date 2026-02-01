import os
from dataclasses import dataclass
from typing import List, Optional, Callable
from config.settings import Settings
from config.constants import MODELS_DIR, CACHE_DIR
from utils.network import HttpClient
from utils.file_handler import FileHandler
from utils.logger import log


@dataclass
class ModelInfo:
    id: str
    name: str
    version: str
    size: int
    md5: str
    download_url: str


class OTAService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = HttpClient(
            base_url=settings.ota.server_url,
            headers={"X-API-Key": settings.ota.api_key} if settings.ota.api_key else {}
        )
        FileHandler.ensure_dir(CACHE_DIR)
    
    def get_available_models(self) -> List[ModelInfo]:
        try:
            resp = self.client.get("/models")
            return [ModelInfo(**m) for m in resp.get("models", [])]
        except Exception as e:
            log.error(f"Failed to fetch models: {e}")
            return []
    
    def download_model(
        self,
        model: ModelInfo,
        progress_cb: Optional[Callable[[int], None]] = None
    ) -> Optional[str]:
        cache_path = os.path.join(CACHE_DIR, f"{model.id}.pt.download")
        final_path = os.path.join(MODELS_DIR, f"{model.id}.pt")
        
        log.info(f"Downloading model: {model.name} v{model.version}")
        
        if not self.client.download(model.download_url, cache_path, progress_cb):
            log.error("Download failed")
            FileHandler.safe_delete(cache_path)
            return None
        
        actual_md5 = FileHandler.md5(cache_path)
        if actual_md5 != model.md5:
            log.error(f"MD5 mismatch: expected {model.md5}, got {actual_md5}")
            FileHandler.safe_delete(cache_path)
            return None
        
        FileHandler.backup(final_path)
        os.replace(cache_path, final_path)
        
        log.info(f"Model installed: {final_path}")
        return final_path
    
    def verify_model(self, path: str, expected_md5: str) -> bool:
        actual = FileHandler.md5(path)
        return actual == expected_md5
    
    def rollback_model(self, model_id: str) -> bool:
        model_path = os.path.join(MODELS_DIR, f"{model_id}.pt")
        backup_path = f"{model_path}.bak"
        return FileHandler.restore(backup_path)

import os
import shutil
import hashlib
from typing import Optional


class FileHandler:
    @staticmethod
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def backup(src: str) -> Optional[str]:
        if not os.path.exists(src):
            return None
        backup_path = f"{src}.bak"
        shutil.copy2(src, backup_path)
        return backup_path
    
    @staticmethod
    def restore(backup_path: str) -> bool:
        if not os.path.exists(backup_path):
            return False
        original = backup_path.rsplit(".bak", 1)[0]
        shutil.copy2(backup_path, original)
        return True
    
    @staticmethod
    def md5(path: str) -> Optional[str]:
        if not os.path.exists(path):
            return None
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    @staticmethod
    def safe_delete(path: str) -> bool:
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
            return True
        except Exception:
            return False
    
    @staticmethod
    def atomic_write(path: str, content: bytes) -> bool:
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "wb") as f:
                f.write(content)
            os.replace(tmp_path, path)
            return True
        except Exception:
            FileHandler.safe_delete(tmp_path)
            return False

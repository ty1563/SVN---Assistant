import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable
from config.settings import Settings
from utils.network import HttpClient
from utils.logger import log
from version import __version__, __build__


@dataclass
class UpdateInfo:
    available: bool
    version: str = ""
    build: str = ""
    url: str = ""
    changelog: str = ""
    mandatory: bool = False


class UpdateChecker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = HttpClient(
            base_url=settings.ota.server_url,
            headers={"X-API-Key": settings.ota.api_key} if settings.ota.api_key else {}
        )
        self._timer: Optional[threading.Timer] = None
        self._on_update: Optional[Callable[[UpdateInfo], None]] = None
    
    def check(self) -> UpdateInfo:
        try:
            resp = self.client.get(f"/updates/check?version={__version__}&build={__build__}")
            
            if resp.get("update_available"):
                info = UpdateInfo(
                    available=True,
                    version=resp.get("version", ""),
                    build=resp.get("build", ""),
                    url=resp.get("download_url", ""),
                    changelog=resp.get("changelog", ""),
                    mandatory=resp.get("mandatory", False)
                )
                log.info(f"Update available: v{info.version}")
                return info
            
            log.info("App is up to date")
            return UpdateInfo(available=False)
            
        except Exception as e:
            log.warning(f"Update check failed: {e}")
            return UpdateInfo(available=False)
    
    def download(self, info: UpdateInfo, progress_cb: Optional[Callable[[int], None]] = None) -> bool:
        if not info.url:
            return False
        return self.client.download(info.url, "update.zip", progress_cb)
    
    def start_scheduled(
        self,
        interval_seconds: int,
        on_update: Optional[Callable[[UpdateInfo], None]] = None
    ):
        self._on_update = on_update
        self._schedule_next(interval_seconds)
        log.info(f"Update checker scheduled every {interval_seconds}s")
    
    def _schedule_next(self, interval: int):
        self._timer = threading.Timer(interval, self._run_check, args=[interval])
        self._timer.daemon = True
        self._timer.start()
    
    def _run_check(self, interval: int):
        info = self.check()
        if info.available and self._on_update:
            self._on_update(info)
        self._schedule_next(interval)
    
    def stop(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

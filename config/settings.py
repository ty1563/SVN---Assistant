import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from config.constants import BASE_DIR, MODELS_DIR, DEFAULT_MODEL

SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")


@dataclass
class OTAConfig:
    server_url: str = "https://api.svnagentic.site/ota"
    api_key: str = ""
    auto_update: bool = True
    check_interval: int = 3600


@dataclass
class DetectionConfig:
    model_name: str = DEFAULT_MODEL
    classifier_model: str = "speed_classifier.pth"
    frames_per_second: int = 5
    input_size: int = 320
    conf_threshold: float = 0.5
    target_classes: list = field(default_factory=lambda: ["P.127"])
    exclude_classes: list = field(default_factory=list)


@dataclass
class Settings:
    ota: OTAConfig = field(default_factory=OTAConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    
    @classmethod
    def load(cls) -> "Settings":
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                ota=OTAConfig(**data.get("ota", {})),
                detection=DetectionConfig(**data.get("detection", {}))
            )
        return cls()
    
    def save(self):
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "ota": asdict(self.ota),
                "detection": asdict(self.detection)
            }, f, indent=2)
    
    @property
    def model_path(self) -> str:
        return os.path.join(MODELS_DIR, self.detection.model_name)

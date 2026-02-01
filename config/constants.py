import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

DEFAULT_MODEL = "8-22k.pt"
MODEL_EXTENSION = ".pt"

API_TIMEOUT = 30
API_RETRY_COUNT = 3
UPDATE_CHECK_INTERVAL = 3600

WINDOW_NAME = "Assistant Detection"

import os
import logging
from datetime import datetime
from config.constants import LOGS_DIR


class Logger:
    _instances = {}
    
    def __new__(cls, name: str = "app"):
        if name not in cls._instances:
            instance = super().__new__(cls)
            instance._init_logger(name)
            cls._instances[name] = instance
        return cls._instances[name]
    
    def _init_logger(self, name: str):
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            log_file = os.path.join(LOGS_DIR, f"{datetime.now():%Y%m%d}.log")
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
            fh.setFormatter(fmt)
            ch.setFormatter(fmt)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)


log = Logger()

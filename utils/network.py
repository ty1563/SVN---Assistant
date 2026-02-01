import urllib.request
import urllib.error
import json
import time
from typing import Optional, Dict, Callable
from config.constants import API_TIMEOUT, API_RETRY_COUNT


class HttpClient:
    def __init__(self, base_url: str = "", headers: Optional[Dict] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = API_TIMEOUT
        self.retry_count = API_RETRY_COUNT
    
    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        headers = {**self.headers, "Content-Type": "application/json"}
        
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        
        for attempt in range(self.retry_count):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code < 500 or attempt == self.retry_count - 1:
                    raise
                time.sleep(2 ** attempt)
            except urllib.error.URLError:
                if attempt == self.retry_count - 1:
                    raise
                time.sleep(2 ** attempt)
        return {}
    
    def get(self, endpoint: str) -> Dict:
        return self._request("GET", endpoint)
    
    def post(self, endpoint: str, data: Dict) -> Dict:
        return self._request("POST", endpoint, data)
    
    def download(self, url: str, dest: str, progress_cb: Optional[Callable] = None) -> bool:
        try:
            def report(block_num, block_size, total_size):
                if progress_cb and total_size > 0:
                    progress_cb(min(100, (block_num * block_size * 100) // total_size))
            
            urllib.request.urlretrieve(url, dest, reporthook=report)
            return True
        except Exception:
            return False

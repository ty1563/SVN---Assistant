Assistant/
├── config/
│   ├── __init__.py
│   ├── constants.py      # Hằng số toàn cục
│   └── settings.py       # Cấu hình với persistence
├── core/
│   ├── __init__.py
│   ├── detector.py       # YOLO detection engine
│   ├── processor.py      # Frame processing
│   └── visualizer.py     # Drawing/overlay
├── services/
│   ├── __init__.py
│   ├── model_service.py  # Model management
│   ├── ota_service.py    # OTA download/verify
│   └── update_checker.py # Auto update check
├── utils/
│   ├── __init__.py
│   ├── file_handler.py   # File operations
│   ├── logger.py         # Logging system
│   └── network.py        # HTTP client
├── models/
│   ├── 8n-14k.pt
│   └── best.pt
├── main.py               # Entry point
├── requirements.txt
└── version.py
Tính Năng Chính
1. OTA Portal (
services/ota_service.py
)
Lấy danh sách models từ server
Download với progress callback
Verify MD5 checksum
Rollback nếu lỗi
2. Model Service (
services/model_service.py
)
List local models
Load/switch/unload models
Hot-reload tại runtime
3. Update Checker (
services/update_checker.py
)
Check version từ server
Scheduled background check
Callback khi có update
Cách Sử Dụng
# Chạy camera
python main.py --camera 0
# Chạy video
python main.py --video path/to/video.mp4
# Chọn model khác
python main.py --model 8n-14k.pt --camera 0
# Xem danh sách models
python main.py --list-models
# Kiểm tra update
python main.py --check-update
Kết Quả Test
[2026-01-30 11:23:35,867] INFO - Found 2 model(s):
[2026-01-30 11:23:35,867] INFO -   - 8n-14k.pt (6.6 MB)
[2026-01-30 11:23:35,868] INFO -   - best.pt (5.3 MB)
Cấu Hình OTA
Sửa file settings.json (tự động tạo):

{
  "ota": {
    "server_url": "https://your-ota-server.com/api",
    "api_key": "your-api-key",
    "auto_update": true,
    "check_interval": 3600
  },
  "detection": {
    "model_name": "best.pt",
    "frames_per_second": 5,
    "input_size": 320,
    "conf_threshold": 0.7
  }
}
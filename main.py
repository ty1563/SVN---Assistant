import sys
import cv2
import argparse
from config.settings import Settings
from config.constants import WINDOW_NAME
from core.processor import FrameProcessor
from services.model_service import ModelService
from services.update_checker import UpdateChecker
from ui.dashboard import Dashboard
from utils.logger import log
from version import __version__, __app_name__


class Application:
    def __init__(self):
        self.settings = Settings.load()
        self.model_service = ModelService()
        self.update_checker = UpdateChecker(self.settings)
        self.dashboard: Dashboard = None
    
    def init(self) -> bool:
        log.info(f"{__app_name__} v{__version__} starting...")
        
        if self.settings.ota.auto_update:
            info = self.update_checker.check()
            if info.available:
                log.info(f"New version available: {info.version}")
        
        model_name = self.settings.detection.model_name
        if not self.model_service.load_model(model_name):
            log.error(f"Failed to load model: {model_name}")
            return False
        
        self.dashboard = Dashboard(self.settings, self.model_service)
        log.info("Application initialized")
        return True
    
    def run_camera(self, camera_id: int = 0):
        processor = FrameProcessor(self.model_service.detector, self.settings)
        
        if not processor.open_camera(camera_id):
            log.error(f"Cannot open camera: {camera_id}")
            return
        
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        cv2.setMouseCallback(WINDOW_NAME, self.dashboard.handle_mouse)
        
        try:
            for frame, detections, time_ms in processor.stream_camera(self.dashboard.get_frame_roi):
                display = self.dashboard.render(
                    frame, detections, time_ms,
                    processor.sign_state.results,
                    processor.sign_state.progress_list,
                    processor.sign_state.active_trackers
                )
                cv2.imshow(WINDOW_NAME, display)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                self.dashboard.handle_key(key)
        finally:
            processor.close()
            cv2.destroyAllWindows()
        
        log.info(f"Average processing time: {processor.get_avg_time():.1f}ms")
    
    def run_video(self, video_path: str):
        processor = FrameProcessor(self.model_service.detector, self.settings)
        
        if not processor.open_video(video_path):
            log.error(f"Cannot open video: {video_path}")
            return
        
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        cv2.setMouseCallback(WINDOW_NAME, self.dashboard.handle_mouse)
        
        try:
            for frame, detections, time_ms in processor.stream_video(self.dashboard.get_frame_roi):
                display = self.dashboard.render(
                    frame, detections, time_ms,
                    processor.sign_state.results,
                    processor.sign_state.progress_list,
                    processor.sign_state.active_trackers
                )
                cv2.imshow(WINDOW_NAME, display)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                self.dashboard.handle_key(key)
        finally:
            processor.close()
            cv2.destroyAllWindows()
        
        log.info(f"Average processing time: {processor.get_avg_time():.1f}ms")
    
    def list_models(self):
        models = self.model_service.list_models()
        log.info(f"Found {len(models)} model(s):")
        for m in models:
            log.info(f"  - {m.name} ({m.size / 1024 / 1024:.1f} MB)")
    
    def check_update(self):
        info = self.update_checker.check()
        if info.available:
            log.info(f"Update available: v{info.version}")
            log.info(f"Changelog: {info.changelog}")
        else:
            log.info("No updates available")


def main():
    parser = argparse.ArgumentParser(description=f"{__app_name__} v{__version__}")
    parser.add_argument("--camera", type=int, metavar="ID", help="Camera ID")
    parser.add_argument("--video", type=str, metavar="PATH", help="Video file path")
    parser.add_argument("--model", type=str, metavar="NAME", help="Model name to use")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--check-update", action="store_true", help="Check for updates")
    
    args = parser.parse_args()
    
    app = Application()
    
    if args.model:
        app.settings.detection.model_name = args.model
    
    if args.list_models:
        app.list_models()
        return
    
    if args.check_update:
        if not app.init():
            return
        app.check_update()
        return
    
    if not app.init():
        sys.exit(1)
    
    if args.video:
        app.run_video(args.video)
    else:
        app.run_camera(args.camera if args.camera is not None else 0)


if __name__ == "__main__":
    main()

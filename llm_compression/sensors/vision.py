
"""
AI-OS Visual Cortex Implementation.

Interfaces with local camera devices (OpenCV) to capture visual stimuli.
"""

import time
import logging
import threading
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Try to import OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV (cv2) not found. Visual sense disabled.")

class VisionStartError(Exception):
    pass

class VisionModule:
    """
    Manages camera input stream.
    """
    
    def __init__(self, output_dir: str, device_index: int = 0):
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV is required for VisionModule")
            
        self.output_dir = Path(output_dir) / "vision_stream"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device_index = device_index
        self.cap = None
        self.running = False
        self.thread = None
        
        # Buffer
        self.last_frame = None
        self.last_capture_time = 0
        self.capture_interval = 5.0  # Seconds between frames
        
    def start(self):
        """Start capturing frames."""
        if self.running:
            return
            
        try:
            self.cap = cv2.VideoCapture(self.device_index)
            if not self.cap.isOpened():
                raise VisionStartError(f"Could not open camera index {self.device_index}")
                
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            logger.info(f"Vision Module started on device {self.device_index}")
        except Exception as e:
            logger.error(f"Failed to start vision: {e}")
            raise

    def stop(self):
        """Stop capturing."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Vision Module stopped")

    def _capture_loop(self):
        """Background loop to capture frames periodically."""
        while self.running and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret:
                    now = time.time()
                    self.last_frame = frame
                    
                    # Periodic snapshot
                    if now - self.last_capture_time > self.capture_interval:
                        self._save_snapshot(frame, now)
                        self.last_capture_time = now
                else:
                    logger.warning("Failed to read frame")
                    time.sleep(1)
                
                time.sleep(0.1) # 10 FPS poll
            except Exception as e:
                logger.error(f"Vision loop error: {e}")
                break
        
        self.running = False

    def _save_snapshot(self, frame, timestamp):
        """Save frame to disk."""
        filename = f"frame_{int(timestamp)}.jpg"
        filepath = self.output_dir / filename
        cv2.imwrite(str(filepath), frame)
        
        # Notify sensor manager via callback (if we had one)
        # For now, just log
        # logger.debug(f"Saved snapshot: {filepath}")
        return filepath

    def get_latest_frame(self):
        """Return latest captured frame (numpy array)."""
        return self.last_frame

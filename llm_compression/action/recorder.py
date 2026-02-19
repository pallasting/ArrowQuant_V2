
import time
import json
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from pynput import mouse, keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logger.warning("pynput not found. Recording disabled.")

class ActionRecorder:
    """
    Records user interactions (Mouse/Keyboard) to a JSONL trace file.
    Used for Imitation Learning (Task 12.4).
    """
    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.log_path = self.workspace / "action_trace.jsonl"
        self.recording = False
        self.mouse_listener = None
        self.key_listener = None
        self.start_time = 0.0

    def start(self):
        """Start recording user actions."""
        if not PYNPUT_AVAILABLE:
            logger.error("Cannot start recording: pynput missing.")
            return

        if self.recording:
            return

        logger.info("Starting Action Recorder...")
        self.recording = True
        self.start_time = time.time()

        # Initialize Listeners
        self.mouse_listener = mouse.Listener(
            on_move=None, # excessive data, maybe enable later if needed
            on_click=self._on_click,
            on_scroll=self._on_scroll
        )
        self.key_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=None
        )

        self.mouse_listener.start()
        self.key_listener.start()
        logger.info(f"Recording to {self.log_path}")

    def stop(self):
        """Stop recording."""
        if not self.recording:
            return

        logger.info("Stopping Action Recorder...")
        self.recording = False
        
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.key_listener:
            self.key_listener.stop()
            
        self.mouse_listener = None
        self.key_listener = None

    def _log_event(self, action_type: str, params: Dict[str, Any]):
        """Write event to log file."""
        if not self.recording:
            return
            
        record = {
            "timestamp": time.time(),
            "relative_time": time.time() - self.start_time,
            "action": action_type,
            "params": params,
            # We could capture screen state here too, but that might lag the recorder
            # "state": ... 
        }
        
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def _on_click(self, x, y, button, pressed):
        """Handle mouse click."""
        if pressed:
            # pynput button is like Button.left
            btn_str = str(button).replace("Button.", "")
            self._log_event("click", {"x": x, "y": y, "button": btn_str})

    def _on_scroll(self, x, y, dx, dy):
        """Handle mouse scroll."""
        self._log_event("scroll", {"x": x, "y": y, "dx": dx, "dy": dy})

    def _on_key_press(self, key):
        """Handle key press."""
        try:
            # Alphanumeric keys
            k_char = key.char
            self._log_event("type", {"text": k_char})
        except AttributeError:
            # Special keys (e.g. Key.space, Key.enter)
            k_name = str(key).replace("Key.", "")
            self._log_event("press", {"key": k_name})

if __name__ == "__main__":
    # Simple CLI for testing
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python recorder.py <workspace_dir>")
        sys.exit(1)
        
    recorder = ActionRecorder(sys.argv[1])
    try:
        recorder.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recorder.stop()

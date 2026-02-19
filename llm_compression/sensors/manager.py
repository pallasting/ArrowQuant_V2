
"""
AI-OS Sensory Manager.

Handles multimodal inputs (Vision, Audio, Proprioception) and converts them
into text-based memory logs for the Skill Factory to consolidate.

Architecture:
- Inputs: Raw sensory data (simulated via file paths or text descriptions).
- Processing: Mock VLM/ASR (Vision Language Model / Auto Speech Recognition).
- Output: 'sensory_log.jsonl' containing timestamped observations.
"""

import time
import json
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class SensorManager:
    """
    Central hub for sensory inputs.
    """

    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.log_path = self.workspace / "sensory_log.jsonl"

        # Hardware Modules
        self.vision = None
        self.audio = None

        # Proprioception collection
        self._proprioception_thread: Optional[threading.Thread] = None
        self._proprioception_stop = threading.Event()

    def start_hardware(self):
        """Initialize and start local sensors."""
        # Vision
        try:
            from llm_compression.sensors.vision import VisionModule
            self.vision = VisionModule(str(self.workspace))
            # Monkey-patch save callback
            original_save = self.vision._save_snapshot

            def on_visual_event(frame, timestamp):
                path = original_save(frame, timestamp)
                self.log_visual_event(f"Visual Snapshot at {timestamp}", str(path))
                return path

            self.vision._save_snapshot = on_visual_event
            self.vision.start()
        except ImportError:
            logger.warning("Vision module unavailable.")
        except Exception as e:
            logger.error(f"Vision start failed: {e}")

        # Audio
        try:
            from llm_compression.sensors.audio import AudioModule
            self.audio = AudioModule(str(self.workspace))
            # Monkey-patch save callback
            original_save = self.audio._save_recording

            def on_audio_event(buffer):
                path = original_save(buffer)
                if path:
                    self.log_auditory_event(f"Audio Recording at {time.time()}", str(path))
                return path

            self.audio._save_recording = on_audio_event
            self.audio.start()
        except ImportError:
            logger.warning("Audio module unavailable.")
        except Exception as e:
            logger.error(f"Audio start failed: {e}")

        # Proprioception (Task 11.4)
        self._proprioception_stop.clear()
        self._proprioception_thread = threading.Thread(
            target=self._proprioception_loop,
            daemon=True,
            name="proprioception"
        )
        self._proprioception_thread.start()
        logger.info("Proprioception collection started.")

    def stop_hardware(self):
        """Stop all sensor modules."""
        if self.vision:
            self.vision.stop()
        if self.audio:
            self.audio.stop()
        # Stop proprioception collection
        self._proprioception_stop.set()
        if self._proprioception_thread and self._proprioception_thread.is_alive():
            self._proprioception_thread.join(timeout=2.0)

    def _proprioception_loop(self, interval: float = 5.0):
        """
        Background loop that collects system stats every `interval` seconds.

        Collects: CPU usage, memory usage, battery level (if available),
        and system load average. These are logged as proprioceptive events
        so the CognitiveLoop can incorporate hardware state into memory.
        """
        import platform

        try:
            import psutil
            has_psutil = True
        except ImportError:
            logger.warning("psutil not installed; proprioception will use fallback stats.")
            has_psutil = False

        while not self._proprioception_stop.wait(timeout=interval):
            try:
                if has_psutil:
                    sensors: Dict[str, float] = {
                        "cpu_percent": psutil.cpu_percent(interval=None),
                        "memory_percent": psutil.virtual_memory().percent,
                        "load_1min": psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0.0,
                    }
                    battery = psutil.sensors_battery()
                    if battery is not None:
                        sensors["battery_percent"] = battery.percent
                        sensors["battery_plugged"] = float(battery.power_plugged)
                else:
                    # Minimal fallback using os.getloadavg (Unix only)
                    import os
                    load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
                    sensors = {"load_1min": load[0], "load_5min": load[1]}

                self.log_proprioception(sensors)
            except Exception as e:
                logger.error(f"Proprioception collection error: {e}")

    def log_visual_event(self, description: str, image_path: Optional[str] = None):
        """Log a visual observation."""
        event = {
            "timestamp": time.time(),
            "modality": "vision",
            "content": description,
            "source": image_path or "stream",
            "embedding_id": None  # Future: link to vector store
        }
        self._append_log(event)
        logger.info(f"Visual Sense: {description}")

    def log_auditory_event(self, transcript: str, audio_path: Optional[str] = None):
        """Log an auditory observation."""
        event = {
            "timestamp": time.time(),
            "modality": "audio",
            "content": transcript,
            "source": audio_path or "stream"
        }
        self._append_log(event)
        logger.info(f"Auditory Sense: {transcript}")

    def log_proprioception(self, sensors: Dict[str, float]):
        """Log internal state (temperature, battery, load)."""
        event = {
            "timestamp": time.time(),
            "modality": "proprioception",
            "data": sensors
        }
        self._append_log(event)

    def _append_log(self, event: Dict[str, Any]):
        """Append event to log file."""
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to log sensory event: {e}")

    def get_recent_events(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent sensory events (e.g., for Context Window)."""
        if not self.log_path.exists():
            return []

        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return [json.loads(line) for line in lines[-limit:]]
        except Exception:
            return []

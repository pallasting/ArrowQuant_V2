
"""
AI-OS System Proprioception (Body Awareness).

Allows the AI to introspect its hardware capabilities and adapt its behavior
(e.g., enable vision if camera found, optimize batch size for RAM).

Key Responsibilities:
1. Resource Monitoring (CPU/RAM/GPU).
2. Peripheral Discovery (Camera/Mic).
3. Dependency Adaptation (Install missing drivers).
"""

import os
import sys
import platform
import logging
import subprocess
import importlib.util
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SystemProprioceptor:
    """The AI's 'sense of self' regarding hardware."""
    
    def __init__(self):
        self.profile: Dict[str, Any] = {
            "os": platform.system(),
            "cpu_cores": os.cpu_count(),
            "ram_gb": 0,
            "gpu": None,
            "sensors": {
                "vision": False,
                "audio": False
            }
        }
        self.scan_resources()

    def scan_resources(self):
        """Scan available compute resources."""
        # 1. RAM (Try psutil, fallback to rough estimate)
        try:
            import psutil
            mem = psutil.virtual_memory()
            self.profile["ram_gb"] = round(mem.total / (1024**3), 1)
        except ImportError:
            # Fallback (Linux specific usually, Windows tricky without psutil)
            self.profile["ram_gb"] = "Unknown (Install psutil)"

        # 2. GPU (Torch CUDA/MPS)
        try:
            import torch
            if torch.cuda.is_available():
                self.profile["gpu"] = torch.cuda.get_device_name(0)
                self.profile["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            elif torch.backends.mps.is_available():
                self.profile["gpu"] = "Apple M-Series (MPS)"
            else:
                self.profile["gpu"] = "CPU Only"
        except ImportError:
            self.profile["gpu"] = "No Torch"

    def detect_sensors(self) -> Dict[str, bool]:
        """Check for physical sensors and required libraries."""
        status = {"vision": False, "audio": False}
        
        # Check Vision
        if importlib.util.find_spec("cv2"):
            import cv2
            # Quick check for camera 0-2
            for idx in range(3):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    status["vision"] = True
                    cap.release()
                    break
        else:
            status["vision_lib_missing"] = True

        # Check Audio (Prefer sounddevice -> pyaudio)
        audio_found = False
        
        # 1. Try sounddevice (Best for Windows/Modern Python)
        if importlib.util.find_spec("sounddevice"):
            try:
                import sounddevice as sd
                # Query devices (at least 1 input)
                if len(sd.query_devices(kind='input')) > 0:
                     status["audio"] = True
                     status["audio_driver"] = "sounddevice"
                     audio_found = True
            except Exception as e:
                logger.warning(f"sounddevice import failed: {e}")
                
        # 2. Try pyaudio (Legacy)
        if not audio_found and importlib.util.find_spec("pyaudio"):
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                if p.get_device_count() > 0:
                    status["audio"] = True
                    status["audio_driver"] = "pyaudio"
                    audio_found = True
                p.terminate()
            except Exception as e:
                 logger.warning(f"pyaudio import failed: {e}")

        # If neither found, mark missing
        if not audio_found:
            status["audio_lib_missing"] = True

        self.profile["sensors"] = status
        return status

    def adapt_environment(self, auto_install: bool = False):
        """Attempt to install drivers for detected hardware capabilities."""
        report = self.detect_sensors()
        
        needed = []
        if report.get("vision_lib_missing"):
            needed.append("opencv-python")
            
        if report.get("audio_lib_missing"):
            # Recommend sounddevice for stability
            needed.append("sounddevice")
            # needed.append("numpy") # Usually present
            
        if not needed:
            return "Environment nominal."
            
        msg = f"Missing sensory drivers: {', '.join(needed)}."
        
        if auto_install:
            logger.info(f"Auto-installing drivers: {needed}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + needed)
                return "Drivers installed successfully."
            except subprocess.CalledProcessError as e:
                return f"Installation failed: {e}"
        else:
            return msg + " Run with auto_install=True to fix."

    def suggest_optimizations(self) -> Dict[str, Any]:
        """Suggest runtime parameters based on hardware."""
        params = {
            "batch_size": 4, # Default (Safe)
            "lora_rank": 8,
            "quantization": "fp32"
        }
        
        # RAM Heuristics
        ram = self.profile["ram_gb"]
        if isinstance(ram, (int, float)):
            if ram > 16:
                params["batch_size"] = 16
            elif ram > 8:
                params["batch_size"] = 8
                
        # GPU Heuristics
        if self.profile["gpu"] != "CPU Only" and self.profile["gpu"] != "No Torch":
             params["device"] = "cuda" if "cuda" in str(self.profile["gpu"]).lower() else "mps"
             if self.profile.get("vram_gb", 0) > 8:
                 params["batch_size"] *= 2
        else:
             params["device"] = "cpu"
             params["quantization"] = "int8" # Recommended for CPU
             
        return params

if __name__ == "__main__":
    # Test run
    p = SystemProprioceptor()
    print("System Profile:", p.profile)
    print("Optimization Advice:", p.suggest_optimizations())

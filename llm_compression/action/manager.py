
"""
AI-OS Action Manager (The Muscles).

Responsible for executing GUI interactions (Mouse/Keyboard).
Designed with Reinforcement Learning (RL) in mind:
- Observations: Visual state before action.
- Actions: Low-level (click, type) or high-level (open_app).
- Feedback: Visual state after action.

Safety: Implements Fail-Safe (corner mouse abortion) and Sandbox boundaries.
"""

import time
import json
import logging
import platform
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Try to import PyAutoGUI
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
    # Fail-Safe: Moving mouse to top-left corner aborts action
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.5  # Human-like delay
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logger.warning("PyAutoGUI not found. Motor functions disabled.")


# Try to import Vision utils
try:
    import cv2
    import numpy as np
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

class ActionError(Exception):
    pass

class ActionManager:
    """
    Control interface for embodied actions.
    Records (State, Action, NextState) tuples for future learning.
    """
    
    
    def __init__(self, workspace_dir: str, sensor_manager=None, enable_logging: bool = True, tts_engine=None):
        self.workspace = Path(workspace_dir)
        self.log_path = self.workspace / "action_trace.jsonl"
        self.sensor_manager = sensor_manager # Link to Vision for feedback
        self.enable_logging = enable_logging
        self.tts_engine = tts_engine
        
        self.screen_size = (0, 0)
        self.safety = None
        
        if PYAUTOGUI_AVAILABLE:
            self.screen_size = pyautogui.size()
            try:
                from llm_compression.action.safety import SafetyMonitor
                self.safety = SafetyMonitor(self.screen_size)
            except ImportError:
                logger.warning("Safety module not found. Playing with fire.")
            
    def _capture_state(self, label: str) -> Optional[str]:
        """Request a visual snapshot from sensors."""
        if self.sensor_manager and self.sensor_manager.vision:
             # Force capture immediately? VisionModule is loop-based.
             # We can just get latest frame or save it now.
             frame = self.sensor_manager.vision.get_latest_frame()
             if frame is not None:
                # Save manually using vision module's helper
                timestamp = time.time()
                return str(self.sensor_manager.vision._save_snapshot(frame, timestamp))
        return None

    def find_element(self, template_path: str, confidence: float = 0.8) -> Optional[Tuple[int, int]]:
        """
        Locate a visual element on screen (Hand-Eye Coordination).
        Returns (center_x, center_y) or None if not found.
        """
        if not VISION_AVAILABLE:
            logger.warning("Vision lib missing. Cannot find element.")
            return None
            
        if not PYAUTOGUI_AVAILABLE:
            return None

        # 1. Take Screenshot (Uses pyautogui for full screen snap, faster than camera)
        # Note: Camera is "external view", PyAutoGUI is "internal view".
        # For precise clicking, internal screenshot is better.
        try:
            screenshot = pyautogui.screenshot()
            screen_np = np.array(screenshot)
            # Convert RGB to BGR for OpenCV
            screen_bgr = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
            
            # 2. Load Template
            template = cv2.imread(template_path)
            if template is None:
                logger.error(f"Template not found: {template_path}")
                return None
                
            # 3. Match
            result = cv2.matchTemplate(screen_bgr, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= confidence:
                h, w = template.shape[:2]
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                logger.info(f"Found {template_path} at ({center_x}, {center_y}) conf={max_val:.2f}")
                return (center_x, center_y)
            else:
                logger.info(f"Element {template_path} not found (best={max_val:.2f})")
                return None
        except Exception as e:
            logger.error(f"Visual search failed: {e}")
            return None

    def execute(self, action_type: str, **params) -> bool:
        """
        Execute an action and log the experience.
        """
        if not PYAUTOGUI_AVAILABLE:
            logger.error("Cannot execute: PyAutoGUI missing.")
            return False

        # 0. Safety Check
        if self.safety:
            try:
                self.safety.validate(action_type, params)
            except Exception as e:
                logger.warning(f"Safety Violation: {e}")
                return False

        # 1. Sense (Pre-Action)
        state_before = self._capture_state("pre_action")
        
        try:
            logger.info(f"Action: {action_type} {params}")
            
            # Announce Action verbally if TTS engine applies
            if self.tts_engine:
                self._narrate_action(action_type, params)
                
            # 2. Act
            if action_type == "move":
                pyautogui.moveTo(params["x"], params["y"], duration=game_like_randomness())
            elif action_type == "click":
                pyautogui.click(params.get("x"), params.get("y"), clicks=params.get("clicks", 1), button=params.get("button", "left"))
            elif action_type == "type":
                pyautogui.write(params["text"], interval=0.05)
            elif action_type == "hotkey":
                keys = params["keys"] # List e.g. ['ctrl', 'c']
                pyautogui.hotkey(*keys)
            elif action_type == "scroll":
                pyautogui.scroll(params["amount"])
            else:
                raise ActionError(f"Unknown action: {action_type}")
                
            # 3. Sense (Post-Action)
            time.sleep(0.5) # Wait for UI update
            state_after = self._capture_state("post_action")
            
            # 4. Log Experience (RL Tuple)
            self._log_experience({
                "timestamp": time.time(),
                "action": action_type,
                "params": params,
                "state_before": state_before,
                "state_after": state_after,
                "platform": platform.system()
            })
            return True
            
        except pyautogui.FailSafeException:
            logger.critical("Action aborted by Fail-Safe (Mouse Corner).")
            raise
        except Exception as e:
            logger.error(f"Action failed: {e}")
            return False

    def _log_experience(self, record: Dict[str, Any]):
        """Persist action trace for future training."""
        if not self.enable_logging:
            return
            
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _narrate_action(self, action_type: str, params: Dict[str, Any]):
        """Synthesize and play speech narrating the current action."""
        if not getattr(self, "tts_engine", None):
            return
            
        narrative = f"Executing {action_type}."
        if action_type == "type":
            text_short = params.get('text', '')[:10]
            narrative = f"Typing {text_short}."
        elif action_type == "hotkey":
            narrative = f"Pressing shortcut."
        elif action_type == "click":
            narrative = f"Clicking here."
            
        def _speak():
            try:
                import sounddevice as sd
                import numpy as np
                chunks = list(self.tts_engine.synthesize(narrative))
                if chunks:
                    audio_data = np.concatenate(chunks)
                    samplerate = self.tts_engine.config.sample_rate
                    sd.play(audio_data, samplerate=samplerate)
                    # We do not wait because we want to act while speaking!
            except Exception as e:
                logger.error(f"TTS Narration failed: {e}")
                
        import threading
        threading.Thread(target=_speak, daemon=True).start()

def game_like_randomness():
    """Add slight randomness to movement duration to mimic human."""
    import random
    return random.uniform(0.1, 0.3)

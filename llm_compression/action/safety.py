
"""
AI-OS Action Safety Sandbox.

Acts as a firewall between the AI's intentions and the OS input system.
Prevents rapid-fire actions, dangerous keystrokes, and out-of-bounds movement.
"""

import time
import logging
import re
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class SafetyViolation(Exception):
    """Raised when an action is blocked by safety policy."""
    pass

class SafetyMonitor:
    """
    enforces safety policies for embodied actions.
    """
    
    def __init__(self, screen_size: Tuple[int, int]):
        self.screen_size = screen_size
        self.last_action_time = 0
        self.min_interval = 0.1  # Max 10 actions per second
        
        # Banned text patterns (Regex)
        self.banned_text = [
            r"rm\s+-[a-zA-Z]*r",   # rm -rf / rm -r
            r"format\s+[a-z]:",    # format c:
            r"del\s+/s",           # del /s
            r">>\s*/dev/sd",       # destructive redirects
            r"shutdown",
            r"::\{[0-9A-F-]+\}"    # GUID hacks
        ]
        
    def validate(self, action_type: str, params: Dict[str, Any]):
        """
        Check if an action is safe to execute.
        Raises SafetyViolation if unsafe.
        """
        # 1. Rate Limiting
        now = time.time()
        if now - self.last_action_time < self.min_interval:
            # We don't raise error, just log warning and maybe sleep?
            # Better to block to prevent spam
            raise SafetyViolation("Rate limit exceeded (Speed kill-switch)")
        
        # 2. Coordinate Bounds
        if action_type in ["move", "click", "drag"]:
            x = params.get("x")
            y = params.get("y")
            if x is not None and y is not None:
                if not (0 <= x < self.screen_size[0] and 0 <= y < self.screen_size[1]):
                    raise SafetyViolation(f"Coordinates ({x},{y}) out of bounds {self.screen_size}")
                    
        # 3. Text Safety
        if action_type in ["type", "write"]:
            text = params.get("text", "")
            for pattern in self.banned_text:
                if re.search(pattern, text, re.IGNORECASE):
                    raise SafetyViolation(f"Blocked dangerous text input pattern: {pattern}")
        
        # 4. Key Safety
        if action_type == "hotkey":
            keys = params.get("keys", [])
            # Block Alt+F4 or dangerous combos?
            # For now, allow all, but maybe log warning
            pass

        self.last_action_time = now
        return True

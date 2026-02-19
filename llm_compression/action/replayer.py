
import time
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
from llm_compression.action.manager import ActionManager

logger = logging.getLogger(__name__)

class ActionReplayer:
    """
    Replays user interactions from a JSONL trace file.
    """
    def __init__(self, workspace_dir: str, trace_file: str = "action_trace.jsonl"):
        self.workspace = Path(workspace_dir)
        self.log_path = self.workspace / trace_file
        self.manager = ActionManager(workspace_dir, enable_logging=False)

    def load_trace(self, limit: int = None) -> List[Dict]:
        """Load actions from the log file."""
        actions = []
        if not self.log_path.exists():
            logger.error(f"Trace file not found: {self.log_path}")
            return []

        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        actions.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading trace: {e}")
            
        if limit:
            actions = actions[-limit:]
            
        return actions

    def replay(self, actions: List[Dict]):
        """Execute a sequence of actions."""
        logger.info(f"Replaying {len(actions)} actions...")
        
        start_time = time.time()
        
        # We need to respect relative timing based on the first action in the sequence
        if not actions:
            return

        # Normalize start time
        first_ref_time = actions[0].get("timestamp", 0)
        
        for i, action in enumerate(actions):
            action_type = action.get("action")
            params = action.get("params", {})
            timestamp = action.get("timestamp", 0)
            
            # Calculate delay
            # We want to wait until (now - start) >= (action_ts - first_ts)
            target_delay = timestamp - first_ref_time
            current_elapsed = time.time() - start_time
            
            wait_time = target_delay - current_elapsed
            if wait_time > 0:
                time.sleep(wait_time)
                
            logger.info(f"Executing [{i+1}/{len(actions)}]: {action_type} {params}")
            
            try:
                # Map recorded events to ActionManager Commands
                if action_type == "click":
                    # pynput: button='left' -> pyautogui: button='left' (compatible)
                    pass 
                elif action_type == "type":
                    # pynput char
                    pass
                elif action_type == "press":
                    # pynput key name 'space' -> pyautogui 'space'
                    # We need to convert this to 'click' or 'type' or 'hotkey'?
                    # ActionManager has 'type' and 'hotkey'.
                    # It doesn't have a single key press exposed directly as 'press'.
                    # But 'type' or 'hotkey' can handle it.
                    # Let's map 'press' to 'hotkey' with one key.
                    action_type = "hotkey"
                    params = {"keys": [params["key"]]}
                
                # Execute
                self.manager.execute(action_type, **params)
                
            except Exception as e:
                logger.error(f"Replay failed: {e}")

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python replayer.py <workspace_dir>")
        sys.exit(1)
        
    replayer = ActionReplayer(sys.argv[1])
    # Replay last 10 actions by default for safety in testing
    actions = replayer.load_trace(limit=10)
    replayer.replay(actions)

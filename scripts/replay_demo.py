
import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llm_compression.action.replayer import ActionReplayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/replay_demo.py <workspace_dir>")
        print("Example: python scripts/replay_demo.py m:/Documents/ai-os-memory")
        return

    workspace = sys.argv[1]
    replayer = ActionReplayer(workspace)
    
    print("Loading trace...")
    # Load last 50 actions to be safe
    actions = replayer.load_trace(limit=50)
    
    if not actions:
        print("No actions found in trace. Run record_demo.py first.")
        return

    print(f"Loaded {len(actions)} actions. Replaying in 3 seconds... HANDS OFF!")
    time.sleep(3)
    
    try:
        replayer.replay(actions)
        print("Replay finished.")
    except KeyboardInterrupt:
        print("\nReplay aborted.")
    except Exception as e:
        print(f"Replay error: {e}")

if __name__ == "__main__":
    main()

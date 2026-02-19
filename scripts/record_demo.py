
import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llm_compression.action.recorder import ActionRecorder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/record_demo.py <workspace_dir>")
        print("Example: python scripts/record_demo.py m:/Documents/ai-os-memory")
        return

    workspace = sys.argv[1]
    recorder = ActionRecorder(workspace)
    
    print(f"Starting recording in 3 seconds... Switch to the app you want to demonstrate.")
    time.sleep(3)
    
    print("reCording started! Perform your actions. Press Ctrl+C to stop.")
    try:
        recorder.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping recording...")
        recorder.stop()
        print(f"Recording saved to {recorder.log_path}")

if __name__ == "__main__":
    main()

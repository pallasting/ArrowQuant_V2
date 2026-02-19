import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from llm_compression.agent.autonomous import AutonomousAgent
from llm_compression.inference.arrow_engine import ArrowEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="AI-OS Autonomous Agent Runner")
    parser.add_argument("goal", type=str, help="High-level goal for the agent")
    parser.add_argument("--workspace", type=str, default="m:/Documents/ai-os-memory", help="Workspace directory")
    parser.add_argument("--steps", type=int, default=10, help="Max steps to run")
    parser.add_argument("--model", type=str, default="./models/base_llm", help="Path to ArrowEngine model (Parquet)")
    parser.add_argument("--vision", action="store_true", help="Enable Vision (Camera)")
    
    args = parser.parse_args()
    
    logger.info(f"Initializing Agent for goal: '{args.goal}'")
    
    # Initialize ArrowEngine - the zero-copy Core Inference Engine
    try:
        engine = ArrowEngine(args.model)
    except Exception as e:
        logger.warning(f"Failed to load ArrowEngine from {args.model}: {e}")
        logger.warning("Running without ArrowEngine (mock mode).")
        engine = None
        
    agent = AutonomousAgent(
        workspace_dir=args.workspace,
        engine=engine,
        enable_vision=args.vision 
    )
    
    try:
        agent.run_loop(args.goal, max_steps=args.steps)
    except KeyboardInterrupt:
        logger.info("Agent interrupted via keyboard.")
    finally:
        agent.stop()

if __name__ == "__main__":
    main()

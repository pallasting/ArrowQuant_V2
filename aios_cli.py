
#!/usr/bin/env python3
"""
AI-OS Command Line Interface (CLI).

The "Motor Cortex" of AI-OS. 
Exposes internal capabilities (Evolution, Federation, Factory) to:
1. Human Operators
2. System Automation Scripts
3. Other AI Agents

Usage:
    python aios_cli.py [command] [subcommand] [args]

Examples:
    python aios_cli.py skills list
    python aios_cli.py evolve "quantum computing"
    python aios_cli.py factory start --daemon
    python aios_cli.py factory status
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("aios-cli")

# Set paths
MEMORY_ROOT = Path(os.environ.get("AIOS_MEMORY_ROOT", "M:/Documents/ai-os-memory"))
MODEL_PATH = MEMORY_ROOT / "models/minilm"

def get_engine():
    """Lazy load ArrowEngine to avoid startup cost for simple commands."""
    try:
        from llm_compression.inference.arrow_engine import ArrowEngine
        return ArrowEngine(str(MODEL_PATH))
    except Exception as e:
        logger.error(f"Failed to load engine: {e}")
        sys.exit(1)

def get_factory(engine):
    """Get Skill Factory instance."""
    from llm_compression.evolution.skill_factory import SkillFactory
    return SkillFactory(engine, workspace_dir=str(MEMORY_ROOT / "skill_factory"))

# ──────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────

def cmd_skills_list(args):
    """List available LoRA skills."""
    lora_dir = MODEL_PATH / "lora_skills"
    if not lora_dir.exists():
        print("No skills found.")
        return

    print(f"{'SKILL NAME':<40} | {'SIZE (KB)':<10} | {'TYPE':<15}")
    print("-" * 70)
    
    for f in lora_dir.glob("*.lora.arrow"):
        size_kb = f.stat().st_size / 1024
        print(f"{f.stem:<40} | {size_kb:<10.1f} | {'Local'}")

def cmd_evolve(args):
    """Trigger immediate evolution on a topic."""
    logger.info(f"Triggering evolution for: '{args.topic}'")
    
    engine = get_engine()
    
    # Configure evolution
    # For CLI, we use CloudDistiller if key provided, else Engine Extraction
    providers = {}
    if args.openai_key:
        from llm_compression.evolution.cloud_distiller import OpenAIProvider
        providers["openai"] = OpenAIProvider(api_key=args.openai_key)
    else:
        # Use Mock if requested
        if args.mock:
            from llm_compression.evolution.cloud_distiller import MockCloudProvider
            providers["mock"] = MockCloudProvider()
            logger.info("Using Mock Provider")
    
    engine.enable_evolution(cloud_providers=providers, rank=args.rank)
    
    # Trigger
    # We use _trigger_evolution directly or simulate a query
    # Simulating a query is safer as it goes through the proper flow
    logger.info("Injecting cognitive dissonance trigger...")
    engine._trigger_evolution(args.topic, confidence=0.0)
    
    # Wait if not detached
    if not args.detach:
        logger.info("Waiting for evolution to complete...")
        if engine._evolution_thread:
            engine._evolution_thread.join()
        logger.info("Evolution finished.")

def cmd_factory_status(args):
    """Show Factory queue status."""
    # We don't need full engine to read status file
    factory_dir = MEMORY_ROOT / "skill_factory"
    tasks_file = factory_dir / "tasks.jsonl"
    
    if not tasks_file.exists():
        print("Factory queue is empty.")
        return

    print(f"{'ID':<10} | {'STATUS':<10} | {'TYPE':<15} | {'NAME'}")
    print("-" * 60)
    
    # Read latest status for each ID
    tasks = {}
    try:
        with open(tasks_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    tasks[data['id']] = data
        
        for t in tasks.values():
            print(f"{t['id']:<10} | {t['status']:<10} | {t['type']:<15} | {t['name']}")
            
    except Exception as e:
        logger.error(f"Error reading queue: {e}")

def cmd_factory_add(args):
    """Add a task to the factory queue."""
    # We can append to the file directly without loading engine if we follow the format
    # But better to use the class to ensure consistency
    # We'll use a lightweight import if possible, but SkillFactory needs engine logic
    # currently. For now, load engine.
    
    engine = get_engine()
    factory = get_factory(engine)
    
    task_id = factory.add_task(
        name=args.name,
        task_type=args.type,
        priority=args.priority,
        # Pass extra args as params
        dataset_path=args.dataset,
        topic=args.topic
    )
    print(f"Task added: {task_id}")

def cmd_factory_start(args):
    """Start the factory worker (Daemon mode)."""
    engine = get_engine()
    factory = get_factory(engine)
    
    factory.start_worker()
    print(f"Skill Factory Worker started. Workspace: {factory.workspace}")
    
    try:
        import time
        while True:
            time.sleep(1)
            # Autonomic Heartbeat
            factory.check_autonomic_triggers()
    except KeyboardInterrupt:
        print("Stopping worker...")
        factory.stop_worker()


def cmd_action_demo(args):
    """Demonstrate motor control."""
    try:
        import pyautogui
    except ImportError:
        logger.error("PyAutoGUI missing: pip install pyautogui")
        return

    engine = get_engine()
    engine.enable_sensors(start_hardware=False) # Optional linkage
    engine.enable_actions()
    
    am = engine.actions
    if not am:
        print("Action Manager failed to init.")
        return

    print("⚠️  WARNING: AI is taking control of mouse in 3 seconds!")
    print("⚠️  MOVE MOUSE TO CORNER TO ABORT (FAILSAFE)")
    time.sleep(3)
    
    try:
        w, h = pyautogui.size()
        center_x, center_y = w // 2, h // 2
        
        print(f"Screen: {w}x{h}. Moving to center...")
        am.execute("move", x=center_x, y=center_y)
        
        print("Drawing a square...")
        opts = [
            (center_x + 100, center_y),
            (center_x + 100, center_y + 100),
            (center_x, center_y + 100),
            (center_x, center_y)
        ]
        
        for tx, ty in opts:
            am.execute("move", x=tx, y=ty)
            time.sleep(0.5)
            
        print("Clicking center...")
        am.execute("click", x=center_x, y=center_y)
        
        print("✅ Demonstration complete. Hand-Eye Log saved.")
        
    except Exception as e:
        print(f"❌ Action failed: {e}")

def cmd_setup(args):
    """Run system diagnostic and setup wizard."""
    print("Initializing System Proprioception (Self-Check)...")
    
    # Check core util needed for check
    try:
        import psutil
    except ImportError:
        print("Installing core utility: psutil...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        
    try:
        # Load Proprioceptor
        sys.path.append(str(MEMORY_ROOT))
        from llm_compression.core.proprioception import SystemProprioceptor
        
        body = SystemProprioceptor()
        print(f"\n[System Profile]\n{json.dumps(body.profile, indent=2)}")
        
        advice = body.suggest_optimizations()
        print(f"\n[Optimization Strategy]\n{json.dumps(advice, indent=2)}")
        
        # Check drivers
        needed = []
        if body.profile["sensors"].get("vision_lib_missing"): needed.append("opencv-python")
        if body.profile["sensors"].get("audio_lib_missing"): needed.append("sounddevice")
        
        if needed:
            print(f"\n[Missing Drivers]: {', '.join(needed)}")
            if args.auto or input("Install these drivers now? [y/N] ").lower() == 'y':
                print("Installing drivers...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + needed)
                print("✅ Drivers installed.")
            else:
                print("Skipped driver installation.")
        else:
            print("\n✅ All sensory drivers present.")

        # Save Config
        config_path = MEMORY_ROOT / "aios_config.json"
        with open(config_path, 'w') as f:
            json.dump({"hardware_profile": body.profile, "runtime_params": advice}, f)
        print(f"Configuration saved to {config_path}")
            
    except Exception as e:
        logger.error(f"Setup failed: {e}")

def cmd_sensors_start(args):
    """Start Sensory Perception Loop."""
    try:
        import cv2
        import pyaudio
    except ImportError:
        logger.error("Missing hardware dependencies: pip install opencv-python pyaudio")
        return

    engine = get_engine()
    # Enable sensors with hardware ON
    engine.enable_sensors(start_hardware=True)
    
    print("Sensory System ACTIVE. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
            # Future: Real-time analysis loop
    except KeyboardInterrupt:
        print("Stopping sensors...")
        if engine.sensors:
            engine.sensors.stop_hardware()

def cmd_dashboard_start(args):
    """Start the Visual Cortex Dashboard."""
    from dashboard_server import DashboardServer
    
    engine = get_engine()
    factory = get_factory(engine)
    
    server = DashboardServer(engine, factory, port=args.port)
    server.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
            # Autonomic Heartbeat
            factory.check_autonomic_triggers()
    except KeyboardInterrupt:
        logger.info("Stopping Dashboard...")
        server.stop()

# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI-OS Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # --- Skills Command ---
    parser_skills = subparsers.add_parser("skills", help="Manage skills")
    parser_skills.add_argument("action", choices=["list"], help="Action to perform")
    
    # --- Dashboard Command ---
    parser_dash = subparsers.add_parser("dashboard", help="Visual Cortex Dashboard")
    dash_sub = parser_dash.add_subparsers(dest="subcommand")
    d_start = dash_sub.add_parser("start", help="Start dashboard server")
    d_start.add_argument("--port", type=int, default=8000, help="Port number")
    
    # --- Evolve Command ---
    parser_evolve = subparsers.add_parser("evolve", help="Trigger self-evolution")
    parser_evolve.add_argument("topic", help="Topic to learn")
    parser_evolve.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser_evolve.add_argument("--openai-key", help="OpenAI API Key")
    parser_evolve.add_argument("--mock", action="store_true", help="Use mock provider")
    parser_evolve.add_argument("--detach", action="store_true", help="Don't wait")

    # --- Factory Command ---
    parser_factory = subparsers.add_parser("factory", help="Manage Skill Factory")
    factory_sub = parser_factory.add_subparsers(dest="subcommand")
    
    # factory status
    factory_sub.add_parser("status", help="Show queue status")
    
    # factory start
    factory_sub.add_parser("start", help="Start factory daemon")
    
    # factory add
    f_add = factory_sub.add_parser("add", help="Add task")
    f_add.add_argument("--name", required=True)
    f_add.add_argument("--type", required=True, choices=["train_dataset", "distill_cloud"])
    f_add.add_argument("--dataset", help="Path to dataset (for train_dataset)")
    f_add.add_argument("--topic", help="Topic (for distill_cloud)")
    f_add.add_argument("--priority", type=int, default=5)

    # --- Setup Command ---
    parser_setup = subparsers.add_parser("setup", help="System Diagnostic & Driver Installation")
    parser_setup.add_argument("--auto", action="store_true", help="Auto-install drivers")

    # --- Action Command ---
    parser_action = subparsers.add_parser("action", help="Perform embodied actions")
    parser_action.add_argument("subcommand", choices=["demo"], help="Action to perform")

    # --- Sensors Command ---
    parser_sensors = subparsers.add_parser("sensors", help="Manage sensors")
    parser_sensors.add_argument("action", choices=["start"], help="Action to perform")

    args = parser.parse_args()

    if args.command == "skills":
        if args.action == "list":
            cmd_skills_list(args)
            
    elif args.command == "setup":
        cmd_setup(args)

    elif args.command == "action":
        if args.subcommand == "demo":
            cmd_action_demo(args)

    elif args.command == "sensors":
        if args.action == "start":
            cmd_sensors_start(args)

    elif args.command == "dashboard":
        if args.subcommand == "start":
            cmd_dashboard_start(args)
            
    elif args.command == "evolve":
        cmd_evolve(args)
        
    elif args.command == "factory":
        if args.subcommand == "status":
            cmd_factory_status(args)
        elif args.subcommand == "add":
            cmd_factory_add(args)
        elif args.subcommand == "start":
            cmd_factory_start(args)
        else:
            parser_factory.print_help()
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

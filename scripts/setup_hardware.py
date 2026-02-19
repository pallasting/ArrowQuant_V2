
"""
AI OS Memory: Hardware Acceleration Setup Tool.
Run this script during deployment to optimize for local hardware.
"""

import os
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from llm_compression.deployment.hardware_analyzer import HardwareAnalyzer

def update_config(backend, use_4bit=True):
    config_path = Path("config.yaml")
    if not config_path.exists():
        print(f"[!] Error: {config_path} not found.")
        return False
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        if "hardware" not in config:
            config["hardware"] = {}
            
        config["hardware"]["preferred_backend"] = backend
        config["hardware"]["enable_4bit_vector_compression"] = use_4bit
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
        return True
    except Exception as e:
        print(f"[!] Failed to update config: {e}")
        return False

def main():
    print("=" * 60)
    print("  AI OS Memory: Hardware Optimization & Deployment Setup")
    print("=" * 60)
    
    analyzer = HardwareAnalyzer()
    analyzer.analyze_system()
    analyzer.analyze_capabilities()
    
    print("\n[*] Initializing benchmarking engine...")
    try:
        analyzer.run_benchmark(quick=True)
    except Exception as e:
        print(f"[!] Benchmark engine error: {e}")
        
    analyzer.generate_recommendations()
    analyzer.save_profile()
    
    best_backend = analyzer.results["recommendations"]["best_backend"]
    print(f"\n[SUCCESS] Optimal Backend Detected: {best_backend.upper()}")
    
    # Auto-apply settings
    use_4bit = (best_backend == "cpu")
    if update_config(best_backend, use_4bit):
        print(f"[*] Configuration 'config.yaml' updated successfully.")
        print(f"    - Preferred Backend: {best_backend}")
        print(f"    - Enable 4-bit Support: {use_4bit}")
    
    # Installation Suggestions
    actions = analyzer.results["recommendations"]["actions"]
    if actions:
        print("\n" + "!"*10 + " RECOMMENDED INSTALLATIONS " + "!"*10)
        for action in actions:
            print(f"- {action['issue']}")
            print(f"  RUN COMMAND: {action['action']}")
            print(f"  WHY: {action['benefit']}")
        print("!"*47)
        
    print("\nHardware deployment analysis complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()

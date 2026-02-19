
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from llm_compression.embedding_provider import get_default_provider, reset_provider

def verify_hardware_config():
    print("Checking hardware configuration...")
    reset_provider()
    provider = get_default_provider()
    engine = provider.engine if hasattr(provider, 'engine') else provider
    print(f"Active Device: {engine.device}")
    
    # Check if it matches config.yaml
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        preferred = config.get("hardware", {}).get("preferred_backend")
        print(f"Preferred Backend in config.yaml: {preferred}")
        
    if preferred != "auto":
        assert engine.device == preferred or (preferred == "cuda" and engine.device == "cuda:0")
        print("✅ System correctly respects config.yaml")

if __name__ == "__main__":
    verify_hardware_config()

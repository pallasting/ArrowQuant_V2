#!/usr/bin/env python3
"""
Test GPU inference performance with Ollama.
Compares CPU vs GPU inference times.
"""

import time
import requests
import json
from typing import Dict, List, Tuple

def test_ollama_inference(prompt: str, max_tokens: int = 50) -> Tuple[float, str]:
    """Test Ollama inference and measure time."""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "qwen2.5:7b-instruct",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7
        }
    }
    
    start_time = time.time()
    response = requests.post(url, json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        return elapsed, result.get("response", "")
    else:
        return elapsed, f"Error: {response.status_code}"

def check_gpu_status() -> Dict[str, str]:
    """Check if Ollama is using GPU."""
    import subprocess
    
    # Check ollama ps
    result = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        text=True
    )
    
    processor = "Unknown"
    if "CPU" in result.stdout:
        processor = "CPU"
    elif "GPU" in result.stdout:
        processor = "GPU"
    
    # Check GPU usage
    gpu_result = subprocess.run(
        ["rocm-smi", "--showuse"],
        capture_output=True,
        text=True
    )
    
    gpu_usage = "0%"
    for line in gpu_result.stdout.split('\n'):
        if "GPU[0]" in line and "GPU use" in line:
            parts = line.split(':')
            if len(parts) > 1:
                gpu_usage = parts[1].strip()
    
    return {
        "processor": processor,
        "gpu_usage": gpu_usage,
        "ollama_output": result.stdout
    }

def main():
    print("=" * 60)
    print("Ollama GPU Inference Test")
    print("=" * 60)
    print()
    
    # Check current status
    print("1. Checking GPU Status...")
    status = check_gpu_status()
    print(f"   Processor: {status['processor']}")
    print(f"   GPU Usage: {status['gpu_usage']}")
    print()
    
    if status['processor'] == 'CPU':
        print("⚠️  WARNING: Ollama is using CPU, not GPU!")
        print("   Expected inference time: 6-11 seconds")
        print("   See GPU_CONFIGURATION_GUIDE.md for setup instructions")
    else:
        print("✓ Ollama is using GPU")
        print("   Expected inference time: 2-3 seconds")
    print()
    
    # Test prompts
    test_cases = [
        ("Summarize: The quick brown fox jumps over the lazy dog.", "Short prompt"),
        ("Explain the concept of machine learning in simple terms.", "Medium prompt"),
        ("Write a brief summary of the benefits of GPU acceleration.", "Technical prompt")
    ]
    
    print("2. Running Inference Tests...")
    print()
    
    results = []
    for prompt, description in test_cases:
        print(f"   Testing: {description}")
        print(f"   Prompt: {prompt[:50]}...")
        
        elapsed, response = test_ollama_inference(prompt, max_tokens=50)
        results.append((description, elapsed))
        
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Response: {response[:100]}...")
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    
    avg_time = sum(r[1] for r in results) / len(results)
    print(f"Average inference time: {avg_time:.2f}s")
    print()
    
    # Performance assessment
    if status['processor'] == 'CPU':
        print("Status: ❌ Using CPU")
        print(f"Performance: {avg_time:.2f}s (expected 6-11s on CPU)")
        print()
        print("Action Required:")
        print("1. Stop Ollama: sudo pkill ollama")
        print("2. Set environment variables:")
        print("   export HSA_OVERRIDE_GFX_VERSION=9.0.6")
        print("   export ROCR_VISIBLE_DEVICES=0")
        print("3. Restart Ollama with GPU:")
        print("   sudo -E HSA_OVERRIDE_GFX_VERSION=9.0.6 ROCR_VISIBLE_DEVICES=0 ollama serve &")
        print("4. Reload model: ollama stop qwen2.5:7b-instruct && ollama run qwen2.5:7b-instruct 'test'")
        print()
        print("See GPU_CONFIGURATION_GUIDE.md for detailed instructions")
    elif avg_time < 4.0:
        print("Status: ✓ Using GPU")
        print(f"Performance: {avg_time:.2f}s (excellent!)")
        print()
        print("GPU acceleration is working correctly!")
        print("You can now run Phase 1.1 acceptance tests:")
        print("  python scripts/phase_1_1_final_acceptance.py")
    else:
        print("Status: ⚠️  GPU enabled but slow")
        print(f"Performance: {avg_time:.2f}s (expected 2-3s)")
        print()
        print("GPU is enabled but performance is below expected.")
        print("Possible causes:")
        print("- GPU memory constraints")
        print("- Model not fully loaded on GPU")
        print("- ROCm configuration issues")
        print()
        print("Try:")
        print("1. Check GPU memory: rocm-smi --showmeminfo vram")
        print("2. Reload model: ollama stop qwen2.5:7b-instruct && ollama run qwen2.5:7b-instruct 'test'")
        print("3. Check ROCm: rocminfo | grep 'Marketing Name'")

if __name__ == "__main__":
    main()

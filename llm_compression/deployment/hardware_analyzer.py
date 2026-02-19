
"""
Hardware Analyzer and Benchmarking Tool for ArrowEngine-Native.

Automatically detects local hardware capabilities, benchmarks performance across 
available backends, and recommends the optimal configuration.
"""

import sys
import time
import json
import torch
import platform
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent dir to path to allow imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from llm_compression.inference.device_utils import get_best_device, get_device_info, is_intel_ipex_available
from llm_compression.embedding_provider import get_default_provider

class HardwareAnalyzer:
    def __init__(self):
        self.results = {
            "system": {},
            "capabilities": {},
            "benchmarks": {},
            "recommendations": {}
        }

    def analyze_system(self):
        """Gather general system info."""
        print("[*] Analyzing system hardware...")
        self.results["system"] = {
            "os": platform.system(),
            "os_version": platform.version(),
            "cpu": platform.processor(),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__
        }

    def analyze_capabilities(self):
        """Detect AI-specific hardware acceleration."""
        print("[*] Detecting AI acceleration capabilities...")
        
        # 1. CUDA (NVIDIA or AMD ROCm)
        cuda_avail = torch.cuda.is_available()
        cuda_info = get_device_info("cuda") if cuda_avail else None
        
        # 2. XPU (Intel GPU)
        xpu_avail = False
        xpu_info = None
        if is_intel_ipex_available():
            try:
                import intel_extension_for_pytorch as ipex
                # Note: Some torch versions need ipex to be imported to see XPU
                xpu_avail = torch.xpu.is_available()
                if xpu_avail:
                    xpu_info = get_device_info("xpu")
            except:
                pass

        # 3. MPS (Apple)
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        # 4. Vulkan (Cross-platform)
        vulkan_avail = False
        try:
            if hasattr(torch, "vulkan") and torch.vulkan.is_available():
                vulkan_avail = True
        except:
            pass

        # 5. AMX (Intel CPU)
        cpu_info = get_device_info("cpu")
        
        self.results["capabilities"] = {
            "cuda_available": cuda_avail,
            "cuda_details": cuda_info,
            "xpu_available": xpu_avail,
            "xpu_details": xpu_info,
            "mps_available": mps_avail,
            "vulkan_available": vulkan_avail,
            "is_amd_gpu": cuda_info.get("is_amd", False) if cuda_info else False,
            "ipex_installed": is_intel_ipex_available(),
            "intel_amx_available": cpu_info.get("has_amx", False)
        }

    def run_benchmark(self, quick=True):
        """Benchmark the primary embedding engine on available hardware."""
        print("[*] Running performance benchmarks...")
        
        test_texts = [
            "This is a sample sentence for benchmarking.",
            "ArrowEngine-Native provides hardware-agnostic acceleration for deep learning.",
            "The quick brown fox jumps over the lazy dog."
        ] * (1 if quick else 10)
        
        backends = ["cpu"]
        caps = self.results["capabilities"]
        if caps["cuda_available"]: backends.append("cuda")
        if caps["xpu_available"]: backends.append("xpu")
        if caps["mps_available"]: backends.append("mps")
        if caps["vulkan_available"]: backends.append("vulkan")
        
        provider = get_default_provider()
        engine = provider.engine if hasattr(provider, 'engine') else provider
        
        for backend in backends:
            print(f"    - Testing backend: {backend}...")
            try:
                # Move engine to backend if possible
                if hasattr(engine, "to"):
                    engine.to(backend)
                
                # Warmup
                engine.encode(test_texts[:1])
                
                # Measure
                start = time.perf_counter()
                for _ in range(5): # Multiple iterations
                    engine.encode(test_texts)
                duration = time.perf_counter() - start
                
                avg_latency = (duration / (len(test_texts) * 5)) * 1000 # ms
                throughput = (len(test_texts) * 5) / duration # seq/sec
                
                self.results["benchmarks"][backend] = {
                    "avg_latency_ms": round(avg_latency, 2),
                    "throughput_seq_sec": round(throughput, 2)
                }
            except Exception as e:
                print(f"      [!] Benchmark failed for {backend}: {e}")
                self.results["benchmarks"][backend] = {"error": str(e)}

    def generate_recommendations(self):
        """Generate setup recommendations."""
        print("[*] Generating hardware optimizations recommendations...")
        
        caps = self.results["capabilities"]
        recs = []
        
        # 1. Selection
        best_backend = "cpu"
        best_throughput = self.results["benchmarks"].get("cpu", {}).get("throughput_seq_sec", 0)
        
        for backend, data in self.results["benchmarks"].items():
            if data.get("throughput_seq_sec", 0) > best_throughput:
                best_throughput = data["throughput_seq_sec"]
                best_backend = backend
        
        self.results["recommendations"]["best_backend"] = best_backend
        
        # 2. Intel Optimizations
        if "Intel" in self.results["system"]["cpu"]:
            if not caps["ipex_installed"]:
                recs.append({
                    "issue": "Intel CPU detected but IPEX not installed",
                    "action": "pip install intel-extension-for-pytorch",
                    "benefit": "Significant AMX/AVX-512 acceleration on CPU"
                })
            elif caps["intel_amx_available"]:
                 recs.append({
                    "issue": "AMX Support found",
                    "action": "Ensure torch.bfloat16 is used for weights",
                    "benefit": "4-8x boost in matrix multiplication"
                })
        
        # 3. Quantization
        if best_backend == "cpu":
             recs.append({
                "issue": "Primary compute is CPU",
                "action": "Use 4-bit Vector Space Compression for long-term storage",
                "benefit": "Reduces memory bandwidth bottleneck"
            })
            
        self.results["recommendations"]["actions"] = recs

    def save_profile(self, path=None):
        """Save results to file."""
        if path is None:
            path = Path.cwd() / "hardware_profile.json"
        
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"[*] Hardware profile saved to: {path}")

def main():
    analyzer = HardwareAnalyzer()
    analyzer.analyze_system()
    analyzer.analyze_capabilities()
    
    # We only run benchmark if we can load the engine
    try:
        analyzer.run_benchmark(quick=True)
    except Exception as e:
        print(f"[!] Critical error during benchmark: {e}")
        
    analyzer.generate_recommendations()
    analyzer.save_profile()
    
    print("\n" + "="*40)
    print("  HARDWARE ANALYSIS SUMMARY")
    print("="*40)
    print(f"Best Engine Backend: {analyzer.results['recommendations']['best_backend'].upper()}")
    
    # Print actions
    if analyzer.results["recommendations"]["actions"]:
        print("\nRecommended Actions:")
        for action in analyzer.results["recommendations"]["actions"]:
            print(f"- {action['issue']}")
            print(f"  Action: {action['action']}")
            print(f"  Benefit: {action['benefit']}")
    
    print("\nDetailed benchmarks:")
    for b, d in analyzer.results["benchmarks"].items():
        if "error" not in d:
             print(f"- {b}: {d['throughput_seq_sec']} seq/sec ({d['avg_latency_ms']} ms/seq)")
    print("="*40)

if __name__ == "__main__":
    main()

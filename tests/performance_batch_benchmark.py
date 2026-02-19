
"""
Performance Benchmark for GPU-Accelerated Batch Compression.

Measures throughput (sequences/sec) of the localized memory system.
"""

import time
import torch
import numpy as np
from llm_compression.embedding_provider import get_default_provider
from llm_compression.compression.vector_compressor import VectorSpaceCompressor

def benchmark_batch_performance():
    print("=" * 60)
    print("  Production Performance Benchmark")
    print("=" * 60)
    
    # 1. Setup
    try:
        provider = get_default_provider()
        engine = provider.engine if hasattr(provider, 'engine') else provider
    except:
        print("❌ Failed to load provider.")
        return

    compressor = VectorSpaceCompressor(engine)
    print(f"  Device: {engine.device}")
    
    # 2. Prepare Data
    batch_sizes = [1, 8, 32, 64]
    num_total = 128
    text = "The quick brown fox jumps over the lazy dog and explores the world of artificial intelligence."
    
    print(f"\n[1] Benchmarking Batch Compression (Throughput)...")
    
    for bs in batch_sizes:
        try:
            # Warmup
            compressor.compress_batch([text] * bs, use_4bit=True)
            
            t0 = time.time()
            for _ in range(0, num_total, bs):
                compressor.compress_batch([text] * bs, use_4bit=True)
            elapsed = time.time() - t0
            
            throughput = num_total / elapsed
            latency = (elapsed / num_total) * 1000
            
            print(f"    Batch Size: {bs:2d} | Throughput: {throughput:7.2f} seq/s | Latency: {latency:6.2f} ms/seq")
        except Exception as e:
            print(f"    Batch Size: {bs:2d} | FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n[2] Device Statistics")
    from llm_compression.inference.device_utils import get_device_info
    info = get_device_info(engine.device)
    for k, v in info.items():
        print(f"    {k:12}: {v}")

    print("=" * 60)

if __name__ == "__main__":
    benchmark_batch_performance()

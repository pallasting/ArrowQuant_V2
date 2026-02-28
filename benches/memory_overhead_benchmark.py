#!/usr/bin/env python3
"""
Memory Overhead Benchmark for Thermodynamic Enhancement

Measures memory usage for each phase:
- Phase 1: Validation (<1% overhead expected)
- Phase 2: Smoothing (<10% overhead expected)
- Phase 3: Optimization (<15% overhead expected)
- Total: <15% overhead expected

Usage:
    python benches/memory_overhead_benchmark.py
"""

import sys
import os
import tracemalloc
import psutil
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import arrow_quant_v2_py as aq
except ImportError:
    print("Error: arrow_quant_v2_py module not found")
    print("Please build the Rust extension first:")
    print("  cd ai_os_diffusion/arrow_quant_v2")
    print("  maturin develop --release")
    sys.exit(1)


def get_memory_usage_mb():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def create_test_weights(shape=(1024, 1024)):
    """Create test weight tensor"""
    return np.random.randn(*shape).astype(np.float32)


def benchmark_baseline_memory():
    """Benchmark baseline quantization memory usage"""
    print("\n=== Baseline Memory Benchmark ===")
    
    weights = create_test_weights((2048, 2048))
    weight_size_mb = weights.nbytes / 1024 / 1024
    print(f"Weight tensor size: {weight_size_mb:.2f} MB")
    
    # Measure baseline memory
    tracemalloc.start()
    mem_before = get_memory_usage_mb()
    
    # Baseline quantization (no thermodynamic features)
    config = {
        "time_aware": {
            "enabled": True,
            "num_groups": 4
        },
        "thermodynamic": {
            "validation": {"enabled": False},
            "boundary_smoothing": {"enabled": False},
            "transition_optimization": {"enabled": False}
        }
    }
    
    quantizer = aq.TimeAwareQuantizer(config)
    _ = quantizer.quantize_layer(weights, bits=2)
    
    mem_after = get_memory_usage_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    baseline_overhead_mb = mem_after - mem_before
    baseline_overhead_pct = (baseline_overhead_mb / weight_size_mb) * 100
    
    print(f"Memory before: {mem_before:.2f} MB")
    print(f"Memory after: {mem_after:.2f} MB")
    print(f"Memory overhead: {baseline_overhead_mb:.2f} MB ({baseline_overhead_pct:.1f}%)")
    print(f"Peak traced memory: {peak / 1024 / 1024:.2f} MB")
    
    return baseline_overhead_mb, weight_size_mb


def benchmark_phase1_memory():
    """Benchmark Phase 1 (Validation) memory overhead"""
    print("\n=== Phase 1 (Validation) Memory Benchmark ===")
    
    weights = create_test_weights((2048, 2048))
    weight_size_mb = weights.nbytes / 1024 / 1024
    
    tracemalloc.start()
    mem_before = get_memory_usage_mb()
    
    # Phase 1: Validation enabled
    config = {
        "time_aware": {
            "enabled": True,
            "num_groups": 4
        },
        "thermodynamic": {
            "validation": {
                "enabled": True,
                "smoothness_threshold": 0.3,
                "log_violations": True
            },
            "boundary_smoothing": {"enabled": False},
            "transition_optimization": {"enabled": False}
        }
    }
    
    quantizer = aq.TimeAwareQuantizer(config)
    _ = quantizer.quantize_layer(weights, bits=2)
    
    mem_after = get_memory_usage_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    phase1_overhead_mb = mem_after - mem_before
    phase1_overhead_pct = (phase1_overhead_mb / weight_size_mb) * 100
    
    print(f"Memory overhead: {phase1_overhead_mb:.2f} MB ({phase1_overhead_pct:.1f}%)")
    print(f"Peak traced memory: {peak / 1024 / 1024:.2f} MB")
    print(f"Target: <1% overhead")
    
    if phase1_overhead_pct < 1.0:
        print("✓ Phase 1 memory target MET")
    else:
        print("✗ Phase 1 memory target EXCEEDED")
    
    return phase1_overhead_mb


def benchmark_phase2_memory():
    """Benchmark Phase 2 (Smoothing) memory overhead"""
    print("\n=== Phase 2 (Validation + Smoothing) Memory Benchmark ===")
    
    weights = create_test_weights((2048, 2048))
    weight_size_mb = weights.nbytes / 1024 / 1024
    
    tracemalloc.start()
    mem_before = get_memory_usage_mb()
    
    # Phase 2: Validation + Smoothing
    config = {
        "time_aware": {
            "enabled": True,
            "num_groups": 4
        },
        "thermodynamic": {
            "validation": {
                "enabled": True,
                "smoothness_threshold": 0.3,
                "log_violations": True
            },
            "boundary_smoothing": {
                "enabled": True,
                "window_size": 5,
                "interpolation": "linear"
            },
            "transition_optimization": {"enabled": False}
        }
    }
    
    quantizer = aq.TimeAwareQuantizer(config)
    _ = quantizer.quantize_layer(weights, bits=2)
    
    mem_after = get_memory_usage_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    phase2_overhead_mb = mem_after - mem_before
    phase2_overhead_pct = (phase2_overhead_mb / weight_size_mb) * 100
    
    print(f"Memory overhead: {phase2_overhead_mb:.2f} MB ({phase2_overhead_pct:.1f}%)")
    print(f"Peak traced memory: {peak / 1024 / 1024:.2f} MB")
    print(f"Target: <10% overhead")
    
    if phase2_overhead_pct < 10.0:
        print("✓ Phase 2 memory target MET")
    else:
        print("✗ Phase 2 memory target EXCEEDED")
    
    return phase2_overhead_mb


def benchmark_phase3_memory():
    """Benchmark Phase 3 (Full Pipeline) memory overhead"""
    print("\n=== Phase 3 (Full Pipeline) Memory Benchmark ===")
    
    weights = create_test_weights((2048, 2048))
    weight_size_mb = weights.nbytes / 1024 / 1024
    
    tracemalloc.start()
    mem_before = get_memory_usage_mb()
    
    # Phase 3: Full pipeline
    config = {
        "time_aware": {
            "enabled": True,
            "num_groups": 4
        },
        "thermodynamic": {
            "validation": {
                "enabled": True,
                "smoothness_threshold": 0.3,
                "log_violations": True
            },
            "boundary_smoothing": {
                "enabled": True,
                "window_size": 5,
                "interpolation": "linear"
            },
            "transition_optimization": {
                "enabled": True,
                "markov_weight": 0.1,
                "entropy_weight": 0.05,
                "learning_rate": 0.01,
                "max_iterations": 10,  # Reduced for benchmark
                "convergence_threshold": 1e-4,
                "beta_schedule": "linear"
            }
        }
    }
    
    quantizer = aq.TimeAwareQuantizer(config)
    _ = quantizer.quantize_layer(weights, bits=2)
    
    mem_after = get_memory_usage_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    phase3_overhead_mb = mem_after - mem_before
    phase3_overhead_pct = (phase3_overhead_mb / weight_size_mb) * 100
    
    print(f"Memory overhead: {phase3_overhead_mb:.2f} MB ({phase3_overhead_pct:.1f}%)")
    print(f"Peak traced memory: {peak / 1024 / 1024:.2f} MB")
    print(f"Target: <15% overhead")
    
    if phase3_overhead_pct < 15.0:
        print("✓ Phase 3 memory target MET")
    else:
        print("✗ Phase 3 memory target EXCEEDED")
    
    return phase3_overhead_mb


def main():
    """Run all memory benchmarks"""
    print("=" * 60)
    print("Thermodynamic Enhancement - Memory Overhead Benchmark")
    print("=" * 60)
    
    try:
        # Baseline
        baseline_mb, weight_size_mb = benchmark_baseline_memory()
        
        # Phase 1
        phase1_mb = benchmark_phase1_memory()
        phase1_vs_baseline = ((phase1_mb - baseline_mb) / weight_size_mb) * 100
        
        # Phase 2
        phase2_mb = benchmark_phase2_memory()
        phase2_vs_baseline = ((phase2_mb - baseline_mb) / weight_size_mb) * 100
        
        # Phase 3
        phase3_mb = benchmark_phase3_memory()
        phase3_vs_baseline = ((phase3_mb - baseline_mb) / weight_size_mb) * 100
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Weight size: {weight_size_mb:.2f} MB")
        print(f"Baseline overhead: {baseline_mb:.2f} MB")
        print(f"Phase 1 additional overhead: {phase1_vs_baseline:.2f}% (target: <1%)")
        print(f"Phase 2 additional overhead: {phase2_vs_baseline:.2f}% (target: <10%)")
        print(f"Phase 3 additional overhead: {phase3_vs_baseline:.2f}% (target: <15%)")
        
        # Overall assessment
        print("\n" + "=" * 60)
        print("ASSESSMENT")
        print("=" * 60)
        
        all_passed = True
        
        if phase1_vs_baseline < 1.0:
            print("✓ Phase 1 memory overhead target MET")
        else:
            print("✗ Phase 1 memory overhead target EXCEEDED")
            all_passed = False
        
        if phase2_vs_baseline < 10.0:
            print("✓ Phase 2 memory overhead target MET")
        else:
            print("✗ Phase 2 memory overhead target EXCEEDED")
            all_passed = False
        
        if phase3_vs_baseline < 15.0:
            print("✓ Phase 3 memory overhead target MET")
        else:
            print("✗ Phase 3 memory overhead target EXCEEDED")
            all_passed = False
        
        if all_passed:
            print("\n✓ ALL MEMORY OVERHEAD TARGETS MET")
            return 0
        else:
            print("\n✗ SOME MEMORY OVERHEAD TARGETS EXCEEDED")
            return 1
    
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

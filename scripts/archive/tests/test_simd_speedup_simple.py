#!/usr/bin/env python3
"""
Simple SIMD Speedup Test

**Validates: Requirements 3.5, 8.1**
**Property 7: SIMD Performance Improvement**

This script tests SIMD vs scalar quantization performance using the Python API.
"""

import time
import numpy as np
import sys

try:
    from arrow_quant_v2 import quantize_simd, dequantize_simd, is_simd_available
except ImportError:
    print("❌ Error: arrow_quant_v2 module not found")
    print("   Please build the module first: maturin develop --release")
    sys.exit(1)

def quantize_scalar_python(data, scale, zero_point):
    """Scalar quantization baseline in Python"""
    return np.clip(np.round(data / scale + zero_point), 0, 255).astype(np.uint8)

def test_simd_speedup(size, iterations, name):
    """Test SIMD vs scalar speedup for a given array size"""
    print(f"\n{name} Array ({size:,} elements):")
    print("-" * 60)
    
    # Generate test data
    np.random.seed(42)
    data = np.random.randn(size).astype(np.float32) * 10.0
    scale = 0.1
    zero_point = 128.0
    
    # Warm up
    for _ in range(5):
        _ = quantize_simd(data, scale, zero_point)
        _ = quantize_scalar_python(data, scale, zero_point)
    
    # Benchmark SIMD
    start = time.time()
    for _ in range(iterations):
        result_simd = quantize_simd(data, scale, zero_point)
    simd_time = time.time() - start
    
    # Benchmark scalar
    start = time.time()
    for _ in range(iterations):
        result_scalar = quantize_scalar_python(data, scale, zero_point)
    scalar_time = time.time() - start
    
    # Calculate speedup
    speedup = scalar_time / simd_time
    
    # Calculate throughput
    simd_throughput = (size * iterations) / simd_time / 1e6  # Million elements/sec
    scalar_throughput = (size * iterations) / scalar_time / 1e6
    
    print(f"  SIMD:   {simd_time:.4f}s ({simd_throughput:.2f} M elem/s)")
    print(f"  Scalar: {scalar_time:.4f}s ({scalar_throughput:.2f} M elem/s)")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Verify correctness
    max_diff = np.max(np.abs(result_simd.astype(np.int32) - result_scalar.astype(np.int32)))
    print(f"  Max difference: {max_diff} (should be ≤ 1 due to rounding)")
    
    return speedup

def main():
    print("=" * 80)
    print("SIMD Performance Speedup Test")
    print("=" * 80)
    
    # Check SIMD availability
    print("\nSIMD Availability:")
    if is_simd_available():
        print("  ✅ SIMD is available on this platform")
    else:
        print("  ⚠️  SIMD is NOT available - will use scalar fallback")
        print("     Speedup test will show minimal improvement")
    
    # Test different array sizes as specified in requirements
    test_cases = [
        (1_000, 1000, "1K"),
        (10_000, 100, "10K"),
        (100_000, 50, "100K"),
        (1_000_000, 10, "1M"),
    ]
    
    speedups = []
    for size, iterations, name in test_cases:
        speedup = test_simd_speedup(size, iterations, name)
        speedups.append((name, speedup))
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\n{'Size':<10} {'Speedup':<15} {'Target':<20} {'Status':<10}")
    print("-" * 60)
    
    targets = {
        "1K": (2.0, 3.0),
        "10K": (2.5, 4.0),
        "100K": (3.0, 5.0),
        "1M": (3.0, 6.0),
    }
    
    all_pass = True
    for name, speedup in speedups:
        min_target, max_target = targets[name]
        status = "✅ PASS" if speedup >= min_target else "❌ FAIL"
        if speedup < min_target:
            all_pass = False
        print(f"{name:<10} {speedup:.2f}x{'':<10} {min_target:.1f}x - {max_target:.1f}x{'':<8} {status}")
    
    print("\n" + "=" * 80)
    print("Overall Result:")
    print("=" * 80)
    
    if all_pass:
        print("\n✅ All SIMD speedup targets met!")
        print("   Requirements 3.5 and 8.1 validated successfully.")
        return 0
    else:
        print("\n⚠️  Some speedup targets not met.")
        print("   This may be due to:")
        print("   - SIMD not available on this platform")
        print("   - Python overhead dominating for small arrays")
        print("   - System load affecting benchmarks")
        return 1

if __name__ == "__main__":
    sys.exit(main())

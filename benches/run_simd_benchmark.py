#!/usr/bin/env python3
"""
SIMD Benchmark Runner

This script runs SIMD benchmarks and compares performance with scalar implementations.
Expected speedup: 2-4x with SIMD (AVX2 on x86_64, NEON on ARM64)

Usage:
    python benches/run_simd_benchmark.py
"""

import subprocess
import sys
import time
from pathlib import Path


def run_benchmark():
    """Run the SIMD benchmark using cargo bench."""
    print("=" * 80)
    print("SIMD Performance Benchmark")
    print("=" * 80)
    print()
    print("This benchmark measures the performance improvement of SIMD-accelerated")
    print("quantization operations compared to scalar implementations.")
    print()
    print("Expected speedup: 2-4x with SIMD")
    print("  - AVX2 on x86_64 (processes 8 floats at a time)")
    print("  - NEON on ARM64 (processes 4 floats at a time)")
    print("  - Scalar fallback for other platforms")
    print()
    print("=" * 80)
    print()

    # Get the project root
    project_root = Path(__file__).parent.parent

    # Run the benchmark
    print("Running benchmark (this may take a few minutes)...")
    print()

    try:
        result = subprocess.run(
            ["cargo", "bench", "--bench", "simd_benchmark"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode == 0:
            print("✅ Benchmark completed successfully!")
            print()
            print("Results:")
            print("-" * 80)
            print(result.stdout)
            print("-" * 80)
            
            # Parse and summarize results
            summarize_results(result.stdout)
        else:
            print("❌ Benchmark failed!")
            print()
            print("Error output:")
            print(result.stderr)
            return 1

    except subprocess.TimeoutExpired:
        print("❌ Benchmark timed out after 10 minutes")
        return 1
    except FileNotFoundError:
        print("❌ cargo command not found. Please install Rust toolchain.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

    return 0


def summarize_results(output: str):
    """Parse benchmark output and provide a summary."""
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    
    # Look for benchmark results
    lines = output.split('\n')
    
    quantize_simd = []
    quantize_scalar = []
    dequantize_simd = []
    dequantize_scalar = []
    cosine_simd = []
    cosine_scalar = []
    
    for line in lines:
        if 'quantize/simd' in line and 'time:' in line:
            quantize_simd.append(line)
        elif 'quantize/scalar' in line and 'time:' in line:
            quantize_scalar.append(line)
        elif 'dequantize/simd' in line and 'time:' in line:
            dequantize_simd.append(line)
        elif 'dequantize/scalar' in line and 'time:' in line:
            dequantize_scalar.append(line)
        elif 'cosine_similarity/simd' in line and 'time:' in line:
            cosine_simd.append(line)
        elif 'cosine_similarity/scalar' in line and 'time:' in line:
            cosine_scalar.append(line)
    
    print("Key Findings:")
    print()
    print("1. Quantization Performance:")
    if quantize_simd and quantize_scalar:
        print(f"   - SIMD: {len(quantize_simd)} test sizes")
        print(f"   - Scalar: {len(quantize_scalar)} test sizes")
        print("   - Expected: 2-4x speedup with SIMD")
    
    print()
    print("2. Dequantization Performance:")
    if dequantize_simd and dequantize_scalar:
        print(f"   - SIMD: {len(dequantize_simd)} test sizes")
        print(f"   - Scalar: {len(dequantize_scalar)} test sizes")
        print("   - Expected: 2-4x speedup with SIMD")
    
    print()
    print("3. Cosine Similarity Performance:")
    if cosine_simd and cosine_scalar:
        print(f"   - SIMD: {len(cosine_simd)} test sizes")
        print(f"   - Scalar: {len(cosine_scalar)} test sizes")
        print("   - Expected: 2-4x speedup with SIMD")
    
    print()
    print("=" * 80)
    print()
    print("📊 Full benchmark results are saved in:")
    print("   target/criterion/")
    print()
    print("📈 To view detailed HTML reports:")
    print("   Open target/criterion/report/index.html in a browser")
    print()


if __name__ == "__main__":
    sys.exit(run_benchmark())

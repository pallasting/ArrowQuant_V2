#!/usr/bin/env python3
"""
Runner script for accuracy benchmarks.

This script runs both Rust (criterion) and Python accuracy benchmarks and combines the results.

Usage:
    python benches/run_accuracy_benchmark.py
    python benches/run_accuracy_benchmark.py --rust-only
    python benches/run_accuracy_benchmark.py --python-only
    python benches/run_accuracy_benchmark.py --generate-charts
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_rust_benchmark():
    """Run Rust criterion benchmark for accuracy."""
    print("=" * 80)
    print("Running Rust Accuracy Benchmark (Criterion)")
    print("=" * 80)
    print()
    
    # Run criterion benchmark
    result = subprocess.run(
        ["cargo", "bench", "--bench", "accuracy_benchmark"],
        cwd=Path(__file__).parent.parent,
        capture_output=False,
    )
    
    if result.returncode != 0:
        print(f"Error: Rust benchmark failed with exit code {result.returncode}")
        return False
    
    print("\n✅ Rust benchmark completed successfully")
    print(f"Results saved to: target/criterion/")
    return True


def run_python_benchmark(generate_charts: bool = False):
    """Run Python accuracy benchmark."""
    print("\n" + "=" * 80)
    print("Running Python Accuracy Benchmark")
    print("=" * 80)
    print()
    
    # Build command
    cmd = [
        sys.executable,
        "benches/accuracy_benchmark.py",
        "--all-bit-widths",
        "--all-modalities",
        "--compare-baseline",
    ]
    
    if generate_charts:
        cmd.append("--generate-charts")
    
    # Run Python benchmark
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent,
        capture_output=False,
    )
    
    if result.returncode != 0:
        print(f"Error: Python benchmark failed with exit code {result.returncode}")
        return False
    
    print("\n✅ Python benchmark completed successfully")
    print(f"Results saved to: .benchmarks/accuracy/")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run accuracy benchmarks")
    parser.add_argument(
        "--rust-only",
        action="store_true",
        help="Run only Rust benchmarks",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Run only Python benchmarks",
    )
    parser.add_argument(
        "--generate-charts",
        action="store_true",
        help="Generate comparison charts (Python benchmark only)",
    )
    
    args = parser.parse_args()
    
    success = True
    
    # Run Rust benchmark
    if not args.python_only:
        if not run_rust_benchmark():
            success = False
    
    # Run Python benchmark
    if not args.rust_only:
        if not run_python_benchmark(args.generate_charts):
            success = False
    
    # Summary
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    
    if success:
        print("✅ All benchmarks completed successfully")
        print()
        print("Results:")
        if not args.python_only:
            print("  - Rust (Criterion): target/criterion/")
        if not args.rust_only:
            print("  - Python: .benchmarks/accuracy/")
            print("    - accuracy_benchmark_report.txt")
            print("    - accuracy_benchmark_results.json")
            if args.generate_charts:
                print("    - accuracy_by_bit_width.png")
                print("    - target_achievement.png")
                print("    - accuracy_compression_tradeoff.png")
        return 0
    else:
        print("❌ Some benchmarks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

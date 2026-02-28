#!/usr/bin/env python3
"""
Memory Benchmark Runner for ArrowQuant V2

This script runs both Rust and Python memory benchmarks and compares results.
It provides detailed memory profiling using psutil and tracemalloc.

Usage:
    python benches/run_memory_benchmark.py --all
    python benches/run_memory_benchmark.py --rust-only
    python benches/run_memory_benchmark.py --python-only --model-size 100M
    python benches/run_memory_benchmark.py --streaming-comparison
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_rust_benchmark():
    """Run Rust memory benchmarks using Criterion."""
    print("=" * 80)
    print("Running Rust Memory Benchmarks")
    print("=" * 80)
    print()
    
    try:
        result = subprocess.run(
            ["cargo", "bench", "--bench", "memory_benchmark"],
            cwd=Path(__file__).parent.parent,
            capture_output=False,
            text=True,
        )
        
        if result.returncode == 0:
            print("\n✅ Rust benchmarks completed successfully")
        else:
            print(f"\n❌ Rust benchmarks failed with exit code {result.returncode}")
            return False
            
    except FileNotFoundError:
        print("❌ Error: cargo not found. Please install Rust.")
        return False
    except Exception as e:
        print(f"❌ Error running Rust benchmarks: {e}")
        return False
    
    return True


def run_python_benchmark(args):
    """Run Python memory benchmarks."""
    print("\n" + "=" * 80)
    print("Running Python Memory Benchmarks")
    print("=" * 80)
    print()
    
    cmd = [sys.executable, "benches/memory_benchmark.py"]
    
    if args.all:
        cmd.append("--all")
    elif args.model_size:
        cmd.extend(["--model-size", args.model_size])
    
    if args.bit_width:
        cmd.extend(["--bit-width", str(args.bit_width)])
    
    if args.generate_charts:
        cmd.append("--generate-charts")
    
    if args.skip_python:
        cmd.append("--skip-python")
    
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir)])
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=False,
            text=True,
        )
        
        if result.returncode == 0:
            print("\n✅ Python benchmarks completed successfully")
        else:
            print(f"\n❌ Python benchmarks failed with exit code {result.returncode}")
            return False
            
    except FileNotFoundError:
        print("❌ Error: Python not found.")
        return False
    except Exception as e:
        print(f"❌ Error running Python benchmarks: {e}")
        return False
    
    return True


def run_streaming_comparison():
    """Run streaming vs batch comparison."""
    print("\n" + "=" * 80)
    print("Running Streaming vs Batch Comparison")
    print("=" * 80)
    print()
    
    # Run Rust benchmark for streaming comparison
    print("Running Rust streaming benchmark...")
    try:
        subprocess.run(
            ["cargo", "bench", "--bench", "memory_benchmark", "--", "streaming_vs_batch"],
            cwd=Path(__file__).parent.parent,
            capture_output=False,
            text=True,
        )
    except Exception as e:
        print(f"Warning: Rust benchmark failed: {e}")
    
    # Run Python benchmark for streaming comparison
    print("\nRunning Python streaming benchmark...")
    try:
        subprocess.run(
            [sys.executable, "benches/memory_benchmark.py", "--model-size", "600M", "--generate-charts"],
            cwd=Path(__file__).parent.parent,
            capture_output=False,
            text=True,
        )
    except Exception as e:
        print(f"Warning: Python benchmark failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run memory benchmarks for ArrowQuant V2")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks (Rust + Python)",
    )
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
        "--streaming-comparison",
        action="store_true",
        help="Run streaming vs batch comparison",
    )
    parser.add_argument(
        "--model-size",
        choices=["100M", "600M", "7B"],
        help="Model size for Python benchmarks",
    )
    parser.add_argument(
        "--bit-width",
        type=int,
        choices=[2, 4, 8],
        default=2,
        help="Bit width for Python benchmarks",
    )
    parser.add_argument(
        "--generate-charts",
        action="store_true",
        help="Generate comparison charts (Python only)",
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip Python implementation in Python benchmarks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for Python benchmark results",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.all, args.rust_only, args.python_only, args.streaming_comparison]):
        parser.error("Must specify one of: --all, --rust-only, --python-only, --streaming-comparison")
    
    success = True
    
    # Run benchmarks based on arguments
    if args.streaming_comparison:
        run_streaming_comparison()
    else:
        if args.all or args.rust_only:
            if not run_rust_benchmark():
                success = False
        
        if args.all or args.python_only:
            if not run_python_benchmark(args):
                success = False
    
    # Print summary
    print("\n" + "=" * 80)
    if success:
        print("✅ All benchmarks completed successfully")
        print("\nResults:")
        print("  - Rust results: target/criterion/report/index.html")
        print("  - Python results: .benchmarks/memory/memory_benchmark_report.txt")
        if args.generate_charts:
            print("  - Charts: .benchmarks/memory/*.png")
    else:
        print("❌ Some benchmarks failed")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()

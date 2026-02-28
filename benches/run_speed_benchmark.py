#!/usr/bin/env python3
"""
Runner script for quantization speed benchmarks.

This script provides a convenient interface to run both Rust and Python speed benchmarks
and generate comparison reports.

Usage:
    # Run all benchmarks
    python benches/run_speed_benchmark.py --all
    
    # Run specific model size
    python benches/run_speed_benchmark.py --model-size 100M
    
    # Run with charts
    python benches/run_speed_benchmark.py --all --generate-charts
    
    # Run Rust benchmarks only
    python benches/run_speed_benchmark.py --rust-only
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_rust_benchmarks():
    """Run Rust Criterion benchmarks."""
    print("\n" + "="*80)
    print("Running Rust Speed Benchmarks (Criterion)")
    print("="*80 + "\n")
    
    bench_dir = Path(__file__).parent.parent
    
    try:
        # Run Criterion benchmarks
        result = subprocess.run(
            ["cargo", "bench", "--bench", "speed_benchmark"],
            cwd=bench_dir,
            check=True,
        )
        
        print("\n✅ Rust benchmarks completed successfully")
        print(f"Results available at: {bench_dir}/target/criterion/report/index.html")
        
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Rust benchmarks failed: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ Cargo not found. Please install Rust toolchain.")
        return False


def run_python_benchmarks(args):
    """Run Python speed benchmarks."""
    print("\n" + "="*80)
    print("Running Python Speed Benchmarks")
    print("="*80 + "\n")
    
    bench_script = Path(__file__).parent / "speed_benchmark.py"
    
    cmd = [sys.executable, str(bench_script)]
    
    if args.all:
        cmd.append("--all")
    elif args.model_size:
        cmd.extend(["--model-size", args.model_size])
    
    if args.bit_width:
        cmd.extend(["--bit-width", str(args.bit_width)])
    
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir)])
    
    if args.generate_charts:
        cmd.append("--generate-charts")
    
    if args.skip_python:
        cmd.append("--skip-python")
    
    if args.num_runs:
        cmd.extend(["--num-runs", str(args.num_runs)])
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Python benchmarks completed successfully")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Python benchmarks failed: {e}")
        return False


def print_summary():
    """Print benchmark summary and next steps."""
    print("\n" + "="*80)
    print("Benchmark Summary")
    print("="*80 + "\n")
    
    print("Benchmark results have been generated. You can find:")
    print("  • Rust results: target/criterion/report/index.html")
    print("  • Python results: .benchmarks/speed/speed_benchmark_report.txt")
    print("  • JSON data: .benchmarks/speed/speed_benchmark_results.json")
    print("  • Charts: .benchmarks/speed/*.png (if --generate-charts was used)")
    print("\nTo view Rust results in browser:")
    print("  open target/criterion/report/index.html")
    print("\nTo view Python results:")
    print("  cat .benchmarks/speed/speed_benchmark_report.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Run quantization speed benchmarks for ArrowQuant V2"
    )
    
    # Benchmark selection
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
    
    # Python benchmark options
    parser.add_argument(
        "--model-size",
        choices=["100M", "600M", "7B"],
        help="Model size to benchmark (Python only)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all model sizes (Python only)",
    )
    parser.add_argument(
        "--bit-width",
        type=int,
        choices=[2, 4, 8],
        help="Target bit width (Python only, default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results (Python only)",
    )
    parser.add_argument(
        "--generate-charts",
        action="store_true",
        help="Generate comparison charts (Python only)",
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip Python implementation in comparison (Python only)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        help="Number of runs to average (Python only, default: 3)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.rust_only and args.python_only:
        parser.error("Cannot specify both --rust-only and --python-only")
    
    if not args.rust_only and not args.python_only:
        if not args.model_size and not args.all:
            parser.error("Either --model-size or --all must be specified for Python benchmarks")
    
    success = True
    
    # Run benchmarks
    if not args.python_only:
        rust_success = run_rust_benchmarks()
        success = success and rust_success
    
    if not args.rust_only:
        python_success = run_python_benchmarks(args)
        success = success and python_success
    
    # Print summary
    if success:
        print_summary()
        return 0
    else:
        print("\n❌ Some benchmarks failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

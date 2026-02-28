#!/usr/bin/env python3
"""
Parallelization Benchmark for ArrowQuant V2

This script benchmarks:
- Parallel vs sequential quantization performance
- Speedup on different core counts (1, 2, 4, 8, 16)
- Performance with different model sizes (100M, 600M, 7B)
- Streaming vs batch memory usage

Usage:
    python benches/parallel_benchmark.py --all
    python benches/parallel_benchmark.py --cores 1,2,4,8
    python benches/parallel_benchmark.py --model-size 600M
    python benches/parallel_benchmark.py --streaming-comparison
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: Rust ArrowQuantV2 not available. Run 'maturin develop --release' first.")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    model_size: str
    num_threads: int
    streaming: bool
    quantization_time_s: float
    peak_memory_mb: float
    baseline_memory_mb: float
    quantization_memory_mb: float
    speedup: Optional[float] = None
    efficiency: Optional[float] = None  # speedup / num_threads


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    results: List[BenchmarkResult] = field(default_factory=list)
    baseline_time: Optional[float] = None
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)
        
        # Calculate speedup relative to baseline (1 thread)
        if self.baseline_time is None and result.num_threads == 1:
            self.baseline_time = result.quantization_time_s
        
        if self.baseline_time is not None and result.num_threads > 1:
            result.speedup = self.baseline_time / result.quantization_time_s
            result.efficiency = result.speedup / result.num_threads
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "baseline_time_s": self.baseline_time,
            "results": [
                {
                    "name": r.name,
                    "model_size": r.model_size,
                    "num_threads": r.num_threads,
                    "streaming": r.streaming,
                    "quantization_time_s": r.quantization_time_s,
                    "peak_memory_mb": r.peak_memory_mb,
                    "baseline_memory_mb": r.baseline_memory_mb,
                    "quantization_memory_mb": r.quantization_memory_mb,
                    "speedup": r.speedup,
                    "efficiency": r.efficiency,
                }
                for r in self.results
            ]
        }


def create_synthetic_model(output_dir: Path, num_layers: int, layer_size: Tuple[int, int]):
    """Create a synthetic model for benchmarking."""
    import numpy as np
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata.json
    metadata = {
        "modality": "text",
        "num_layers": num_layers,
        "layer_size": layer_size,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create synthetic layer files (simplified)
    for i in range(num_layers):
        layer_path = output_dir / f"layer_{i}.parquet"
        # Create synthetic data
        weights = np.random.randn(*layer_size).astype(np.float32)
        # In real implementation, would write proper Parquet
        # For benchmarking, just create files
        with open(layer_path, "wb") as f:
            f.write(b"synthetic_layer_data")


def get_model_config(model_size: str) -> Tuple[int, Tuple[int, int]]:
    """Get model configuration (num_layers, layer_size) for a given size."""
    configs = {
        "100M": (50, (512, 4096)),      # ~2M params/layer
        "600M": (100, (1024, 6144)),    # ~6M params/layer
        "7B": (200, (2048, 8192)),      # ~16M params/layer
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    
    return configs[model_size]


def benchmark_quantization(
    model_path: Path,
    output_path: Path,
    num_threads: int,
    streaming: bool,
    bit_width: int = 4,
) -> Tuple[float, float, float]:
    """
    Benchmark quantization and return (time_s, peak_memory_mb, baseline_memory_mb).
    
    Args:
        model_path: Path to input model
        output_path: Path to output quantized model
        num_threads: Number of threads to use
        streaming: Whether to use streaming mode
        bit_width: Quantization bit width
        
    Returns:
        Tuple of (quantization_time_s, peak_memory_mb, baseline_memory_mb)
    """
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust ArrowQuantV2 not available")
    
    # Measure baseline memory
    process = psutil.Process()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create quantizer
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Configure quantization
    config = {
        "bit_width": bit_width,
        "num_threads": num_threads,
        "enable_streaming": streaming,
        "modality": "text",
    }
    
    # Measure peak memory during quantization
    peak_memory = baseline_memory
    
    def memory_monitor():
        nonlocal peak_memory
        while True:
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
            time.sleep(0.1)
    
    # Start memory monitoring in background
    import threading
    monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
    monitor_thread.start()
    
    # Benchmark quantization
    start_time = time.time()
    
    try:
        result = quantizer.quantize_diffusion_model(
            model_path=str(model_path),
            output_path=str(output_path),
            config=config,
        )
        quantization_time = time.time() - start_time
    except Exception as e:
        print(f"Error during quantization: {e}")
        quantization_time = -1.0
    
    # Stop memory monitoring
    time.sleep(0.2)  # Let monitor catch final peak
    
    return quantization_time, peak_memory, baseline_memory


def benchmark_parallel_scaling(
    model_size: str,
    thread_counts: List[int],
    output_dir: Path,
) -> BenchmarkReport:
    """
    Benchmark parallel scaling with different thread counts.
    
    Args:
        model_size: Model size (100M, 600M, 7B)
        thread_counts: List of thread counts to test
        output_dir: Output directory for results
        
    Returns:
        BenchmarkReport with results
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking Parallel Scaling: {model_size}")
    print(f"{'='*80}\n")
    
    report = BenchmarkReport()
    
    # Create synthetic model
    num_layers, layer_size = get_model_config(model_size)
    model_dir = output_dir / "models" / f"synthetic_{model_size}"
    
    print(f"Creating synthetic model: {num_layers} layers, {layer_size} shape")
    create_synthetic_model(model_dir, num_layers, layer_size)
    
    # Benchmark each thread count
    for num_threads in thread_counts:
        print(f"\nBenchmarking with {num_threads} thread(s)...")
        
        quant_dir = output_dir / "quantized" / f"{model_size}_threads_{num_threads}"
        quant_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            quant_time, peak_mem, baseline_mem = benchmark_quantization(
                model_path=model_dir,
                output_path=quant_dir,
                num_threads=num_threads,
                streaming=False,
            )
            
            result = BenchmarkResult(
                name=f"parallel_{num_threads}_threads",
                model_size=model_size,
                num_threads=num_threads,
                streaming=False,
                quantization_time_s=quant_time,
                peak_memory_mb=peak_mem,
                baseline_memory_mb=baseline_mem,
                quantization_memory_mb=peak_mem - baseline_mem,
            )
            
            report.add_result(result)
            
            print(f"  Time: {quant_time:.2f}s")
            print(f"  Peak Memory: {peak_mem:.2f} MB")
            print(f"  Quantization Memory: {result.quantization_memory_mb:.2f} MB")
            if result.speedup:
                print(f"  Speedup: {result.speedup:.2f}x")
                print(f"  Efficiency: {result.efficiency:.2%}")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    return report


def benchmark_model_sizes(
    model_sizes: List[str],
    num_threads: int,
    output_dir: Path,
) -> BenchmarkReport:
    """
    Benchmark different model sizes with fixed thread count.
    
    Args:
        model_sizes: List of model sizes (100M, 600M, 7B)
        num_threads: Number of threads to use
        output_dir: Output directory for results
        
    Returns:
        BenchmarkReport with results
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking Model Sizes (with {num_threads} threads)")
    print(f"{'='*80}\n")
    
    report = BenchmarkReport()
    
    for model_size in model_sizes:
        print(f"\nBenchmarking {model_size} model...")
        
        # Create synthetic model
        num_layers, layer_size = get_model_config(model_size)
        model_dir = output_dir / "models" / f"synthetic_{model_size}"
        
        print(f"Creating synthetic model: {num_layers} layers, {layer_size} shape")
        create_synthetic_model(model_dir, num_layers, layer_size)
        
        quant_dir = output_dir / "quantized" / f"{model_size}_size_test"
        quant_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            quant_time, peak_mem, baseline_mem = benchmark_quantization(
                model_path=model_dir,
                output_path=quant_dir,
                num_threads=num_threads,
                streaming=False,
            )
            
            result = BenchmarkResult(
                name=f"model_size_{model_size}",
                model_size=model_size,
                num_threads=num_threads,
                streaming=False,
                quantization_time_s=quant_time,
                peak_memory_mb=peak_mem,
                baseline_memory_mb=baseline_mem,
                quantization_memory_mb=peak_mem - baseline_mem,
            )
            
            report.add_result(result)
            
            print(f"  Time: {quant_time:.2f}s")
            print(f"  Peak Memory: {peak_mem:.2f} MB")
            print(f"  Quantization Memory: {result.quantization_memory_mb:.2f} MB")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    return report


def benchmark_streaming_vs_batch(
    model_size: str,
    num_threads: int,
    output_dir: Path,
) -> BenchmarkReport:
    """
    Compare streaming vs batch mode memory usage.
    
    Args:
        model_size: Model size (100M, 600M, 7B)
        num_threads: Number of threads to use
        output_dir: Output directory for results
        
    Returns:
        BenchmarkReport with results
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking Streaming vs Batch: {model_size}")
    print(f"{'='*80}\n")
    
    report = BenchmarkReport()
    
    # Create synthetic model
    num_layers, layer_size = get_model_config(model_size)
    model_dir = output_dir / "models" / f"synthetic_{model_size}"
    
    print(f"Creating synthetic model: {num_layers} layers, {layer_size} shape")
    create_synthetic_model(model_dir, num_layers, layer_size)
    
    # Benchmark batch mode
    print("\nBenchmarking Batch Mode...")
    quant_dir_batch = output_dir / "quantized" / f"{model_size}_batch"
    quant_dir_batch.mkdir(parents=True, exist_ok=True)
    
    try:
        quant_time_batch, peak_mem_batch, baseline_mem_batch = benchmark_quantization(
            model_path=model_dir,
            output_path=quant_dir_batch,
            num_threads=num_threads,
            streaming=False,
        )
        
        result_batch = BenchmarkResult(
            name="batch_mode",
            model_size=model_size,
            num_threads=num_threads,
            streaming=False,
            quantization_time_s=quant_time_batch,
            peak_memory_mb=peak_mem_batch,
            baseline_memory_mb=baseline_mem_batch,
            quantization_memory_mb=peak_mem_batch - baseline_mem_batch,
        )
        
        report.add_result(result_batch)
        
        print(f"  Time: {quant_time_batch:.2f}s")
        print(f"  Peak Memory: {peak_mem_batch:.2f} MB")
        print(f"  Quantization Memory: {result_batch.quantization_memory_mb:.2f} MB")
    
    except Exception as e:
        print(f"  Error: {e}")
        result_batch = None
    
    # Benchmark streaming mode
    print("\nBenchmarking Streaming Mode...")
    quant_dir_stream = output_dir / "quantized" / f"{model_size}_streaming"
    quant_dir_stream.mkdir(parents=True, exist_ok=True)
    
    try:
        quant_time_stream, peak_mem_stream, baseline_mem_stream = benchmark_quantization(
            model_path=model_dir,
            output_path=quant_dir_stream,
            num_threads=num_threads,
            streaming=True,
        )
        
        result_stream = BenchmarkResult(
            name="streaming_mode",
            model_size=model_size,
            num_threads=num_threads,
            streaming=True,
            quantization_time_s=quant_time_stream,
            peak_memory_mb=peak_mem_stream,
            baseline_memory_mb=baseline_mem_stream,
            quantization_memory_mb=peak_mem_stream - baseline_mem_stream,
        )
        
        report.add_result(result_stream)
        
        print(f"  Time: {quant_time_stream:.2f}s")
        print(f"  Peak Memory: {peak_mem_stream:.2f} MB")
        print(f"  Quantization Memory: {result_stream.quantization_memory_mb:.2f} MB")
        
        # Compare
        if result_batch:
            memory_ratio = result_stream.quantization_memory_mb / result_batch.quantization_memory_mb
            print(f"\n  Memory Ratio (Streaming/Batch): {memory_ratio:.2%}")
            if memory_ratio < 0.5:
                print(f"  ✅ Target achieved: <50% memory usage")
            else:
                print(f"  ⚠️  Target not met: {memory_ratio:.2%} > 50%")
    
    except Exception as e:
        print(f"  Error: {e}")
    
    return report


def generate_report(reports: List[BenchmarkReport], output_path: Path):
    """Generate comprehensive benchmark report."""
    print(f"\n{'='*80}")
    print("Parallelization Benchmark Report")
    print(f"{'='*80}\n")
    
    # Combine all results
    all_results = []
    for report in reports:
        all_results.extend(report.results)
    
    # Generate text report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("ArrowQuant V2 Parallelization Benchmark Report")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Parallel scaling results
    parallel_results = [r for r in all_results if "parallel" in r.name]
    if parallel_results:
        report_lines.append("Parallel Scaling Results:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Threads':<10} {'Time (s)':<12} {'Memory (MB)':<15} {'Speedup':<10} {'Efficiency':<12}")
        report_lines.append("-" * 80)
        
        for r in sorted(parallel_results, key=lambda x: x.num_threads):
            speedup_str = f"{r.speedup:.2f}x" if r.speedup else "N/A"
            efficiency_str = f"{r.efficiency:.1%}" if r.efficiency else "N/A"
            report_lines.append(
                f"{r.num_threads:<10} {r.quantization_time_s:<12.2f} "
                f"{r.quantization_memory_mb:<15.2f} {speedup_str:<10} {efficiency_str:<12}"
            )
        report_lines.append("")
        
        # Check if target achieved
        eight_core_result = next((r for r in parallel_results if r.num_threads == 8), None)
        if eight_core_result and eight_core_result.speedup:
            if eight_core_result.speedup >= 4.0:
                report_lines.append(f"✅ Target achieved: {eight_core_result.speedup:.2f}x speedup on 8 cores (target: 4-8x)")
            else:
                report_lines.append(f"⚠️  Target not met: {eight_core_result.speedup:.2f}x speedup on 8 cores (target: 4-8x)")
        report_lines.append("")
    
    # Model size results
    size_results = [r for r in all_results if "model_size" in r.name]
    if size_results:
        report_lines.append("Model Size Results:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Model Size':<15} {'Time (s)':<12} {'Memory (MB)':<15} {'Target':<15} {'Status':<10}")
        report_lines.append("-" * 80)
        
        targets = {"100M": 120, "600M": 600, "7B": 1800}  # seconds
        
        for r in size_results:
            target = targets.get(r.model_size, 0)
            status = "✅ Pass" if r.quantization_time_s < target else "⚠️  Slow"
            report_lines.append(
                f"{r.model_size:<15} {r.quantization_time_s:<12.2f} "
                f"{r.quantization_memory_mb:<15.2f} <{target}s{'':<10} {status:<10}"
            )
        report_lines.append("")
    
    # Streaming vs batch results
    streaming_results = [r for r in all_results if r.name in ["batch_mode", "streaming_mode"]]
    if len(streaming_results) == 2:
        batch = next(r for r in streaming_results if not r.streaming)
        stream = next(r for r in streaming_results if r.streaming)
        
        report_lines.append("Streaming vs Batch Comparison:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Mode':<15} {'Time (s)':<12} {'Memory (MB)':<15}")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Batch':<15} {batch.quantization_time_s:<12.2f} {batch.quantization_memory_mb:<15.2f}")
        report_lines.append(f"{'Streaming':<15} {stream.quantization_time_s:<12.2f} {stream.quantization_memory_mb:<15.2f}")
        report_lines.append("")
        
        memory_ratio = stream.quantization_memory_mb / batch.quantization_memory_mb
        report_lines.append(f"Memory Ratio (Streaming/Batch): {memory_ratio:.2%}")
        if memory_ratio < 0.5:
            report_lines.append(f"✅ Target achieved: <50% memory usage")
        else:
            report_lines.append(f"⚠️  Target not met: {memory_ratio:.2%} > 50%")
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save text report
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "parallel_benchmark_report.txt", "w") as f:
        f.write(report_text)
    
    # Save JSON results
    json_data = {
        "reports": [r.to_dict() for r in reports],
        "all_results": [
            {
                "name": r.name,
                "model_size": r.model_size,
                "num_threads": r.num_threads,
                "streaming": r.streaming,
                "quantization_time_s": r.quantization_time_s,
                "peak_memory_mb": r.peak_memory_mb,
                "quantization_memory_mb": r.quantization_memory_mb,
                "speedup": r.speedup,
                "efficiency": r.efficiency,
            }
            for r in all_results
        ]
    }
    
    with open(output_path / "parallel_benchmark_results.json", "w") as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - {output_path / 'parallel_benchmark_report.txt'}")
    print(f"  - {output_path / 'parallel_benchmark_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Parallelization benchmark for ArrowQuant V2")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--cores",
        type=str,
        default="1,2,4,8",
        help="Comma-separated list of core counts to test (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["100M", "600M", "7B"],
        help="Model size to benchmark",
    )
    parser.add_argument(
        "--model-sizes",
        type=str,
        default="100M,600M",
        help="Comma-separated list of model sizes (default: 100M,600M)",
    )
    parser.add_argument(
        "--streaming-comparison",
        action="store_true",
        help="Compare streaming vs batch mode",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".benchmarks/parallel",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    if not RUST_AVAILABLE:
        print("Error: Rust ArrowQuantV2 not available.")
        print("Please run: cd ai_os_diffusion/arrow_quant_v2 && maturin develop --release")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    reports = []
    
    # Parse thread counts
    thread_counts = [int(x.strip()) for x in args.cores.split(",")]
    
    # Parse model sizes
    model_sizes = [x.strip() for x in args.model_sizes.split(",")]
    
    if args.all:
        # Run all benchmarks
        print("Running all parallelization benchmarks...")
        
        # Parallel scaling
        report = benchmark_parallel_scaling("100M", thread_counts, output_dir)
        reports.append(report)
        
        # Model sizes
        report = benchmark_model_sizes(model_sizes, 8, output_dir)
        reports.append(report)
        
        # Streaming vs batch
        report = benchmark_streaming_vs_batch("600M", 8, output_dir)
        reports.append(report)
    
    elif args.model_size:
        # Benchmark specific model size with different thread counts
        report = benchmark_parallel_scaling(args.model_size, thread_counts, output_dir)
        reports.append(report)
    
    elif args.streaming_comparison:
        # Compare streaming vs batch
        model_size = args.model_size or "600M"
        report = benchmark_streaming_vs_batch(model_size, 8, output_dir)
        reports.append(report)
    
    else:
        # Default: parallel scaling with 100M model
        report = benchmark_parallel_scaling("100M", thread_counts, output_dir)
        reports.append(report)
    
    # Generate report
    if reports:
        generate_report(reports, output_dir)


if __name__ == "__main__":
    main()

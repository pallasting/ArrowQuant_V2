#!/usr/bin/env python3
"""
Memory Benchmark for ArrowQuant V2

Measures peak memory usage during quantization and compares Rust vs Python implementations.
Tests with models of different sizes (100M, 600M, 7B parameters).

Target: Rust implementation should use <50% memory vs Python implementation.

Usage:
    python benches/memory_benchmark.py --model-size 100M
    python benches/memory_benchmark.py --model-size 600M --output-dir ./benchmark_results
    python benches/memory_benchmark.py --all --generate-charts
"""

import argparse
import json
import sys
import time
import tracemalloc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: Rust ArrowQuantV2 not available. Install with: maturin develop")


@dataclass
class MemoryMetrics:
    """Memory usage metrics for a single quantization run."""
    implementation: str  # "rust" or "python"
    model_size: str  # "100M", "600M", "7B"
    bit_width: int
    peak_memory_mb: float
    baseline_memory_mb: float
    quantization_memory_mb: float
    quantization_time_s: float
    timestamp: str


@dataclass
class ModelConfig:
    """Configuration for synthetic model generation."""
    name: str
    num_params: int
    num_layers: int
    hidden_size: int
    intermediate_size: int


# Model configurations for different sizes
MODEL_CONFIGS = {
    "100M": ModelConfig(
        name="synthetic_100M",
        num_params=100_000_000,
        num_layers=12,
        hidden_size=768,
        intermediate_size=3072,
    ),
    "600M": ModelConfig(
        name="synthetic_600M",
        num_params=600_000_000,
        num_layers=24,
        hidden_size=1024,
        intermediate_size=4096,
    ),
    "7B": ModelConfig(
        name="synthetic_7B",
        num_params=7_000_000_000,
        num_layers=32,
        hidden_size=4096,
        intermediate_size=11008,
    ),
}


class MemoryProfiler:
    """Memory profiler using tracemalloc and psutil."""
    
    def __init__(self):
        self.baseline_memory = 0
        self.peak_memory = 0
        self.process = psutil.Process()
    
    def start(self):
        """Start memory profiling."""
        tracemalloc.start()
        self.baseline_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
    
    def stop(self) -> Tuple[float, float]:
        """
        Stop memory profiling and return metrics.
        
        Returns:
            Tuple of (peak_memory_mb, quantization_memory_mb)
        """
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        quantization_memory_mb = peak_memory_mb - self.baseline_memory
        
        return peak_memory_mb, quantization_memory_mb


def generate_synthetic_model(config: ModelConfig, output_dir: Path) -> Path:
    """
    Generate a synthetic model with specified size.
    
    Args:
        config: Model configuration
        output_dir: Output directory for model files
    
    Returns:
        Path to model directory
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    model_dir = output_dir / config.name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata.json
    metadata = {
        "modality": "text",
        "num_params": config.num_params,
        "num_layers": config.num_layers,
        "hidden_size": config.hidden_size,
    }
    
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    # Generate synthetic weights (Parquet V1 format)
    layers = []
    
    # Embedding layer
    layers.append({
        "layer_name": "model.embed_tokens.weight",
        "shape": [50000, config.hidden_size],
        "dtype": "float16",
        "data": np.random.randn(50000, config.hidden_size).astype(np.float16).tobytes(),
        "num_params": 50000 * config.hidden_size,
    })
    
    # Transformer layers
    for i in range(config.num_layers):
        # Self-attention
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            layers.append({
                "layer_name": f"model.layers.{i}.self_attn.{proj}.weight",
                "shape": [config.hidden_size, config.hidden_size],
                "dtype": "float16",
                "data": np.random.randn(config.hidden_size, config.hidden_size).astype(np.float16).tobytes(),
                "num_params": config.hidden_size * config.hidden_size,
            })
        
        # MLP
        layers.append({
            "layer_name": f"model.layers.{i}.mlp.gate_proj.weight",
            "shape": [config.intermediate_size, config.hidden_size],
            "dtype": "float16",
            "data": np.random.randn(config.intermediate_size, config.hidden_size).astype(np.float16).tobytes(),
            "num_params": config.intermediate_size * config.hidden_size,
        })
        
        layers.append({
            "layer_name": f"model.layers.{i}.mlp.up_proj.weight",
            "shape": [config.intermediate_size, config.hidden_size],
            "dtype": "float16",
            "data": np.random.randn(config.intermediate_size, config.hidden_size).astype(np.float16).tobytes(),
            "num_params": config.intermediate_size * config.hidden_size,
        })
        
        layers.append({
            "layer_name": f"model.layers.{i}.mlp.down_proj.weight",
            "shape": [config.hidden_size, config.intermediate_size],
            "dtype": "float16",
            "data": np.random.randn(config.hidden_size, config.intermediate_size).astype(np.float16).tobytes(),
            "num_params": config.hidden_size * config.intermediate_size,
        })
    
    # LM head
    layers.append({
        "layer_name": "lm_head.weight",
        "shape": [50000, config.hidden_size],
        "dtype": "float16",
        "data": np.random.randn(50000, config.hidden_size).astype(np.float16).tobytes(),
        "num_params": 50000 * config.hidden_size,
    })
    
    # Write to Parquet
    schema = pa.schema([
        ("layer_name", pa.string()),
        ("shape", pa.list_(pa.int64())),
        ("dtype", pa.string()),
        ("data", pa.binary()),
        ("num_params", pa.int64()),
    ])
    
    table = pa.Table.from_pylist(layers, schema=schema)
    pq.write_table(table, model_dir / "model.parquet")
    
    print(f"Generated synthetic model: {config.name} ({len(layers)} layers)")
    return model_dir


def benchmark_rust_quantization(
    model_dir: Path,
    config: ModelConfig,
    bit_width: int,
    output_dir: Path,
) -> MemoryMetrics:
    """
    Benchmark Rust quantization implementation.
    
    Args:
        model_dir: Path to model directory
        config: Model configuration
        bit_width: Target bit width (2, 4, or 8)
        output_dir: Output directory for quantized model
    
    Returns:
        Memory metrics
    """
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust ArrowQuantV2 not available")
    
    profiler = MemoryProfiler()
    profiler.start()
    
    # Create quantizer
    quant_config = DiffusionQuantConfig(
        bit_width=bit_width,
        num_time_groups=10,
        group_size=128,
        enable_time_aware=True,
        enable_spatial=True,
        min_accuracy=0.70,
        calibration_samples=32,
    )
    
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Quantize model
    start_time = time.time()
    
    result = quantizer.quantize_diffusion_model(
        model_path=str(model_dir),
        output_path=str(output_dir / f"{config.name}_int{bit_width}_rust"),
        config=quant_config.__dict__,
    )
    
    quantization_time = time.time() - start_time
    
    # Stop profiling
    peak_memory, quant_memory = profiler.stop()
    
    return MemoryMetrics(
        implementation="rust",
        model_size=config.name.split("_")[1],
        bit_width=bit_width,
        peak_memory_mb=peak_memory,
        baseline_memory_mb=profiler.baseline_memory,
        quantization_memory_mb=quant_memory,
        quantization_time_s=quantization_time,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def benchmark_python_quantization(
    model_dir: Path,
    config: ModelConfig,
    bit_width: int,
    output_dir: Path,
) -> MemoryMetrics:
    """
    Benchmark Python quantization implementation.
    
    Args:
        model_dir: Path to model directory
        config: Model configuration
        bit_width: Target bit width (2, 4, or 8)
        output_dir: Output directory for quantized model
    
    Returns:
        Memory metrics
    """
    # Import Python quantizer
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from llm_compression.inference.arrow_quantizer import ArrowQuantizer, QuantizationConfig
    
    profiler = MemoryProfiler()
    profiler.start()
    
    # Create quantizer
    quant_config = QuantizationConfig(
        quant_type=f"int{bit_width}",
        calibration_method="ptq",
        per_channel=False,
        symmetric=True,
        group_size=128,
    )
    
    quantizer = ArrowQuantizer(quant_config)
    
    # Quantize model
    start_time = time.time()
    
    quantizer.quantize_model(
        input_parquet=str(model_dir / "model.parquet"),
        output_parquet=str(output_dir / f"{config.name}_int{bit_width}_python.parquet"),
        show_progress=False,
    )
    
    quantization_time = time.time() - start_time
    
    # Stop profiling
    peak_memory, quant_memory = profiler.stop()
    
    return MemoryMetrics(
        implementation="python",
        model_size=config.name.split("_")[1],
        bit_width=bit_width,
        peak_memory_mb=peak_memory,
        baseline_memory_mb=profiler.baseline_memory,
        quantization_memory_mb=quant_memory,
        quantization_time_s=quantization_time,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def generate_comparison_report(
    results: List[MemoryMetrics],
    output_dir: Path,
):
    """
    Generate comparison report and charts.
    
    Args:
        results: List of memory metrics
        output_dir: Output directory for report
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group results by model size and bit width
    grouped = {}
    for metric in results:
        key = (metric.model_size, metric.bit_width)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][metric.implementation] = metric
    
    # Generate text report
    report_path = output_dir / "memory_benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ArrowQuant V2 Memory Benchmark Report\n")
        f.write("=" * 80 + "\n\n")
        
        for (model_size, bit_width), impls in sorted(grouped.items()):
            f.write(f"\nModel Size: {model_size}, Bit Width: INT{bit_width}\n")
            f.write("-" * 80 + "\n")
            
            rust_metrics = impls.get("rust")
            python_metrics = impls.get("python")
            
            if rust_metrics:
                f.write(f"Rust Implementation:\n")
                f.write(f"  Peak Memory: {rust_metrics.peak_memory_mb:.2f} MB\n")
                f.write(f"  Quantization Memory: {rust_metrics.quantization_memory_mb:.2f} MB\n")
                f.write(f"  Quantization Time: {rust_metrics.quantization_time_s:.2f}s\n")
            
            if python_metrics:
                f.write(f"\nPython Implementation:\n")
                f.write(f"  Peak Memory: {python_metrics.peak_memory_mb:.2f} MB\n")
                f.write(f"  Quantization Memory: {python_metrics.quantization_memory_mb:.2f} MB\n")
                f.write(f"  Quantization Time: {python_metrics.quantization_time_s:.2f}s\n")
            
            if rust_metrics and python_metrics:
                memory_ratio = rust_metrics.quantization_memory_mb / python_metrics.quantization_memory_mb
                speedup = python_metrics.quantization_time_s / rust_metrics.quantization_time_s
                
                f.write(f"\nComparison:\n")
                f.write(f"  Memory Ratio (Rust/Python): {memory_ratio:.2%}\n")
                f.write(f"  Speedup (Python/Rust): {speedup:.2f}x\n")
                
                if memory_ratio < 0.50:
                    f.write(f"  ✅ Target achieved: <50% memory usage\n")
                else:
                    f.write(f"  ❌ Target not met: {memory_ratio:.2%} > 50%\n")
            
            f.write("\n")
    
    print(f"Report generated: {report_path}")
    
    # Save JSON results
    json_path = output_dir / "memory_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump([asdict(m) for m in results], f, indent=2)
    
    print(f"JSON results saved: {json_path}")


def generate_charts(results: List[MemoryMetrics], output_dir: Path):
    """
    Generate memory usage charts.
    
    Args:
        results: List of memory metrics
        output_dir: Output directory for charts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by model size
    model_sizes = sorted(set(m.model_size for m in results))
    bit_widths = sorted(set(m.bit_width for m in results))
    
    # Chart 1: Memory usage comparison
    fig, axes = plt.subplots(1, len(bit_widths), figsize=(15, 5))
    if len(bit_widths) == 1:
        axes = [axes]
    
    for idx, bit_width in enumerate(bit_widths):
        ax = axes[idx]
        
        rust_memory = []
        python_memory = []
        labels = []
        
        for model_size in model_sizes:
            rust_metric = next((m for m in results if m.model_size == model_size and m.bit_width == bit_width and m.implementation == "rust"), None)
            python_metric = next((m for m in results if m.model_size == model_size and m.bit_width == bit_width and m.implementation == "python"), None)
            
            if rust_metric and python_metric:
                rust_memory.append(rust_metric.quantization_memory_mb)
                python_memory.append(python_metric.quantization_memory_mb)
                labels.append(model_size)
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, rust_memory, width, label='Rust', color='#1f77b4')
        ax.bar(x + width/2, python_memory, width, label='Python', color='#ff7f0e')
        
        ax.set_xlabel('Model Size')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title(f'INT{bit_width} Quantization Memory')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    chart_path = output_dir / "memory_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Chart generated: {chart_path}")
    plt.close()
    
    # Chart 2: Memory ratio (Rust/Python)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for bit_width in bit_widths:
        ratios = []
        labels = []
        
        for model_size in model_sizes:
            rust_metric = next((m for m in results if m.model_size == model_size and m.bit_width == bit_width and m.implementation == "rust"), None)
            python_metric = next((m for m in results if m.model_size == model_size and m.bit_width == bit_width and m.implementation == "python"), None)
            
            if rust_metric and python_metric:
                ratio = rust_metric.quantization_memory_mb / python_metric.quantization_memory_mb
                ratios.append(ratio * 100)  # Convert to percentage
                labels.append(model_size)
        
        x = np.arange(len(labels))
        ax.plot(x, ratios, marker='o', label=f'INT{bit_width}', linewidth=2)
    
    ax.axhline(y=50, color='r', linestyle='--', label='Target (50%)')
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Memory Ratio (Rust/Python %)')
    ax.set_title('Memory Efficiency: Rust vs Python')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    ratio_chart_path = output_dir / "memory_ratio.png"
    plt.savefig(ratio_chart_path, dpi=300, bbox_inches='tight')
    print(f"Chart generated: {ratio_chart_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Memory benchmark for ArrowQuant V2")
    parser.add_argument(
        "--model-size",
        choices=["100M", "600M", "7B"],
        help="Model size to benchmark",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all model sizes",
    )
    parser.add_argument(
        "--bit-width",
        type=int,
        choices=[2, 4, 8],
        default=2,
        help="Target bit width (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".benchmarks/memory"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--generate-charts",
        action="store_true",
        help="Generate comparison charts",
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip Python implementation benchmark",
    )
    
    args = parser.parse_args()
    
    if not args.model_size and not args.all:
        parser.error("Either --model-size or --all must be specified")
    
    # Determine which models to benchmark
    if args.all:
        model_sizes = ["100M", "600M", "7B"]
    else:
        model_sizes = [args.model_size]
    
    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    results = []
    
    for model_size in model_sizes:
        print(f"\n{'='*80}")
        print(f"Benchmarking {model_size} model")
        print(f"{'='*80}\n")
        
        config = MODEL_CONFIGS[model_size]
        
        # Generate synthetic model
        print(f"Generating synthetic model...")
        model_dir = generate_synthetic_model(config, models_dir)
        
        # Benchmark Rust implementation
        if RUST_AVAILABLE:
            print(f"\nBenchmarking Rust implementation (INT{args.bit_width})...")
            try:
                rust_metrics = benchmark_rust_quantization(
                    model_dir, config, args.bit_width, args.output_dir
                )
                results.append(rust_metrics)
                print(f"  Peak Memory: {rust_metrics.peak_memory_mb:.2f} MB")
                print(f"  Quantization Memory: {rust_metrics.quantization_memory_mb:.2f} MB")
                print(f"  Time: {rust_metrics.quantization_time_s:.2f}s")
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Rust implementation not available")
        
        # Benchmark Python implementation
        if not args.skip_python:
            print(f"\nBenchmarking Python implementation (INT{args.bit_width})...")
            try:
                python_metrics = benchmark_python_quantization(
                    model_dir, config, args.bit_width, args.output_dir
                )
                results.append(python_metrics)
                print(f"  Peak Memory: {python_metrics.peak_memory_mb:.2f} MB")
                print(f"  Quantization Memory: {python_metrics.quantization_memory_mb:.2f} MB")
                print(f"  Time: {python_metrics.quantization_time_s:.2f}s")
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
    
    # Generate report
    if results:
        print(f"\n{'='*80}")
        print("Generating comparison report...")
        print(f"{'='*80}\n")
        
        generate_comparison_report(results, args.output_dir)
        
        if args.generate_charts:
            try:
                generate_charts(results, args.output_dir)
            except Exception as e:
                print(f"Error generating charts: {e}")
    else:
        print("No results to report")


if __name__ == "__main__":
    main()

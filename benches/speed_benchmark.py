#!/usr/bin/env python3
"""
Quantization Speed Benchmark for ArrowQuant V2

Measures quantization time for different model sizes and compares Rust vs Python implementations.
Tests with models of different sizes (100M, 600M, 7B parameters).

Target: Rust implementation should achieve 5-10x speedup vs Python implementation.

Usage:
    python benches/speed_benchmark.py --model-size 100M
    python benches/speed_benchmark.py --model-size 600M --output-dir ./benchmark_results
    python benches/speed_benchmark.py --all --generate-charts
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: Rust ArrowQuantV2 not available. Install with: maturin develop --release")


@dataclass
class SpeedMetrics:
    """Speed metrics for a single quantization run."""
    implementation: str  # "rust" or "python"
    model_size: str  # "100M", "600M", "7B"
    bit_width: int
    quantization_time_s: float
    throughput_params_per_sec: float
    model_params: int
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
    num_runs: int = 3,
) -> SpeedMetrics:
    """
    Benchmark Rust quantization implementation.
    
    Args:
        model_dir: Path to model directory
        config: Model configuration
        bit_width: Target bit width (2, 4, or 8)
        output_dir: Output directory for quantized model
        num_runs: Number of runs to average (default: 3)
    
    Returns:
        Speed metrics
    """
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust ArrowQuantV2 not available")
    
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
    
    # Warm-up run
    print(f"  Warm-up run...")
    _ = quantizer.quantize_diffusion_model(
        model_path=str(model_dir),
        output_path=str(output_dir / f"{config.name}_int{bit_width}_rust_warmup"),
        config=quant_config.__dict__,
    )
    
    # Benchmark runs
    times = []
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...")
        
        start_time = time.time()
        
        result = quantizer.quantize_diffusion_model(
            model_path=str(model_dir),
            output_path=str(output_dir / f"{config.name}_int{bit_width}_rust_run{run}"),
            config=quant_config.__dict__,
        )
        
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
    
    # Calculate average time
    avg_time = sum(times) / len(times)
    throughput = config.num_params / avg_time
    
    return SpeedMetrics(
        implementation="rust",
        model_size=config.name.split("_")[1],
        bit_width=bit_width,
        quantization_time_s=avg_time,
        throughput_params_per_sec=throughput,
        model_params=config.num_params,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def benchmark_python_quantization(
    model_dir: Path,
    config: ModelConfig,
    bit_width: int,
    output_dir: Path,
    num_runs: int = 3,
) -> SpeedMetrics:
    """
    Benchmark Python quantization implementation.
    
    Args:
        model_dir: Path to model directory
        config: Model configuration
        bit_width: Target bit width (2, 4, or 8)
        output_dir: Output directory for quantized model
        num_runs: Number of runs to average (default: 3)
    
    Returns:
        Speed metrics
    """
    # Import Python quantizer
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from llm_compression.inference.arrow_quantizer import ArrowQuantizer, QuantizationConfig
    
    # Create quantizer
    quant_config = QuantizationConfig(
        quant_type=f"int{bit_width}",
        calibration_method="ptq",
        per_channel=False,
        symmetric=True,
        group_size=128,
    )
    
    quantizer = ArrowQuantizer(quant_config)
    
    # Warm-up run
    print(f"  Warm-up run...")
    quantizer.quantize_model(
        input_parquet=str(model_dir / "model.parquet"),
        output_parquet=str(output_dir / f"{config.name}_int{bit_width}_python_warmup.parquet"),
        show_progress=False,
    )
    
    # Benchmark runs
    times = []
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...")
        
        start_time = time.time()
        
        quantizer.quantize_model(
            input_parquet=str(model_dir / "model.parquet"),
            output_parquet=str(output_dir / f"{config.name}_int{bit_width}_python_run{run}.parquet"),
            show_progress=False,
        )
        
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
    
    # Calculate average time
    avg_time = sum(times) / len(times)
    throughput = config.num_params / avg_time
    
    return SpeedMetrics(
        implementation="python",
        model_size=config.name.split("_")[1],
        bit_width=bit_width,
        quantization_time_s=avg_time,
        throughput_params_per_sec=throughput,
        model_params=config.num_params,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def generate_comparison_report(
    results: List[SpeedMetrics],
    output_dir: Path,
):
    """
    Generate comparison report and charts.
    
    Args:
        results: List of speed metrics
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
    report_path = output_dir / "speed_benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ArrowQuant V2 Quantization Speed Benchmark Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary table
        f.write("Summary Table:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model Size':<12} {'Bit Width':<12} {'Rust Time':<15} {'Python Time':<15} {'Speedup':<10} {'Status':<10}\n")
        f.write("-" * 80 + "\n")
        
        for (model_size, bit_width), impls in sorted(grouped.items()):
            rust_metrics = impls.get("rust")
            python_metrics = impls.get("python")
            
            if rust_metrics and python_metrics:
                speedup = python_metrics.quantization_time_s / rust_metrics.quantization_time_s
                status = "✅ Pass" if speedup >= 5.0 else "⚠️  Warn"
                
                f.write(f"{model_size:<12} {'INT' + str(bit_width):<12} "
                       f"{rust_metrics.quantization_time_s:<15.2f} "
                       f"{python_metrics.quantization_time_s:<15.2f} "
                       f"{speedup:<10.2f}x {status:<10}\n")
        
        f.write("\n")
        
        # Detailed results
        for (model_size, bit_width), impls in sorted(grouped.items()):
            f.write(f"\nModel Size: {model_size}, Bit Width: INT{bit_width}\n")
            f.write("-" * 80 + "\n")
            
            rust_metrics = impls.get("rust")
            python_metrics = impls.get("python")
            
            if rust_metrics:
                f.write(f"Rust Implementation:\n")
                f.write(f"  Quantization Time: {rust_metrics.quantization_time_s:.2f}s\n")
                f.write(f"  Throughput: {rust_metrics.throughput_params_per_sec / 1e6:.2f}M params/s\n")
                f.write(f"  Model Parameters: {rust_metrics.model_params / 1e6:.0f}M\n")
            
            if python_metrics:
                f.write(f"\nPython Implementation:\n")
                f.write(f"  Quantization Time: {python_metrics.quantization_time_s:.2f}s\n")
                f.write(f"  Throughput: {python_metrics.throughput_params_per_sec / 1e6:.2f}M params/s\n")
                f.write(f"  Model Parameters: {python_metrics.model_params / 1e6:.0f}M\n")
            
            if rust_metrics and python_metrics:
                speedup = python_metrics.quantization_time_s / rust_metrics.quantization_time_s
                throughput_ratio = rust_metrics.throughput_params_per_sec / python_metrics.throughput_params_per_sec
                
                f.write(f"\nComparison:\n")
                f.write(f"  Speedup (Python/Rust): {speedup:.2f}x\n")
                f.write(f"  Throughput Ratio (Rust/Python): {throughput_ratio:.2f}x\n")
                
                if speedup >= 5.0:
                    f.write(f"  ✅ Target achieved: {speedup:.2f}x >= 5.0x\n")
                else:
                    f.write(f"  ⚠️  Target not met: {speedup:.2f}x < 5.0x\n")
            
            f.write("\n")
        
        # Performance targets validation
        f.write("\n" + "=" * 80 + "\n")
        f.write("Performance Targets Validation\n")
        f.write("=" * 80 + "\n\n")
        
        targets = {
            "100M": 120,  # <2 minutes
            "600M": 600,  # <10 minutes
            "7B": 300,    # <5 minutes (Dream 7B target)
        }
        
        for model_size, target_time in targets.items():
            rust_metric = next((m for m in results if m.model_size == model_size and m.implementation == "rust"), None)
            
            if rust_metric:
                status = "✅ Pass" if rust_metric.quantization_time_s < target_time else "❌ Fail"
                f.write(f"{model_size} Model:\n")
                f.write(f"  Target: <{target_time}s\n")
                f.write(f"  Actual: {rust_metric.quantization_time_s:.2f}s\n")
                f.write(f"  Status: {status}\n\n")
    
    print(f"Report generated: {report_path}")
    
    # Save JSON results
    json_path = output_dir / "speed_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump([asdict(m) for m in results], f, indent=2)
    
    print(f"JSON results saved: {json_path}")


def generate_charts(results: List[SpeedMetrics], output_dir: Path):
    """
    Generate speed comparison charts.
    
    Args:
        results: List of speed metrics
        output_dir: Output directory for charts
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available. Skipping chart generation.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by model size
    model_sizes = sorted(set(m.model_size for m in results))
    bit_widths = sorted(set(m.bit_width for m in results))
    
    # Chart 1: Quantization time comparison
    fig, axes = plt.subplots(1, len(bit_widths), figsize=(15, 5))
    if len(bit_widths) == 1:
        axes = [axes]
    
    for idx, bit_width in enumerate(bit_widths):
        ax = axes[idx]
        
        rust_times = []
        python_times = []
        labels = []
        
        for model_size in model_sizes:
            rust_metric = next((m for m in results if m.model_size == model_size and m.bit_width == bit_width and m.implementation == "rust"), None)
            python_metric = next((m for m in results if m.model_size == model_size and m.bit_width == bit_width and m.implementation == "python"), None)
            
            if rust_metric and python_metric:
                rust_times.append(rust_metric.quantization_time_s)
                python_times.append(python_metric.quantization_time_s)
                labels.append(model_size)
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, rust_times, width, label='Rust', color='#1f77b4')
        ax.bar(x + width/2, python_times, width, label='Python', color='#ff7f0e')
        
        ax.set_xlabel('Model Size')
        ax.set_ylabel('Quantization Time (seconds)')
        ax.set_title(f'INT{bit_width} Quantization Speed')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    chart_path = output_dir / "speed_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Chart generated: {chart_path}")
    plt.close()
    
    # Chart 2: Speedup (Python/Rust)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for bit_width in bit_widths:
        speedups = []
        labels = []
        
        for model_size in model_sizes:
            rust_metric = next((m for m in results if m.model_size == model_size and m.bit_width == bit_width and m.implementation == "rust"), None)
            python_metric = next((m for m in results if m.model_size == model_size and m.bit_width == bit_width and m.implementation == "python"), None)
            
            if rust_metric and python_metric:
                speedup = python_metric.quantization_time_s / rust_metric.quantization_time_s
                speedups.append(speedup)
                labels.append(model_size)
        
        x = np.arange(len(labels))
        ax.plot(x, speedups, marker='o', label=f'INT{bit_width}', linewidth=2, markersize=8)
    
    ax.axhline(y=5, color='g', linestyle='--', label='Target (5x)', linewidth=2)
    ax.axhline(y=10, color='r', linestyle='--', label='Stretch Goal (10x)', linewidth=2)
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Speedup (Python/Rust)')
    ax.set_title('Quantization Speedup: Rust vs Python')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    speedup_chart_path = output_dir / "speedup_comparison.png"
    plt.savefig(speedup_chart_path, dpi=300, bbox_inches='tight')
    print(f"Chart generated: {speedup_chart_path}")
    plt.close()
    
    # Chart 3: Throughput comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for bit_width in bit_widths:
        rust_throughputs = []
        python_throughputs = []
        labels = []
        
        for model_size in model_sizes:
            rust_metric = next((m for m in results if m.model_size == model_size and m.bit_width == bit_width and m.implementation == "rust"), None)
            python_metric = next((m for m in results if m.model_size == model_size and m.bit_width == bit_width and m.implementation == "python"), None)
            
            if rust_metric and python_metric:
                rust_throughputs.append(rust_metric.throughput_params_per_sec / 1e6)  # M params/s
                python_throughputs.append(python_metric.throughput_params_per_sec / 1e6)
                labels.append(model_size)
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, rust_throughputs, width, label=f'Rust INT{bit_width}', alpha=0.8)
        ax.bar(x + width/2, python_throughputs, width, label=f'Python INT{bit_width}', alpha=0.8)
    
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Throughput (M params/s)')
    ax.set_title('Quantization Throughput Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    throughput_chart_path = output_dir / "throughput_comparison.png"
    plt.savefig(throughput_chart_path, dpi=300, bbox_inches='tight')
    print(f"Chart generated: {throughput_chart_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Quantization speed benchmark for ArrowQuant V2")
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
        default=Path(".benchmarks/speed"),
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
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs to average (default: 3)",
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
                    model_dir, config, args.bit_width, args.output_dir, args.num_runs
                )
                results.append(rust_metrics)
                print(f"  Average Time: {rust_metrics.quantization_time_s:.2f}s")
                print(f"  Throughput: {rust_metrics.throughput_params_per_sec / 1e6:.2f}M params/s")
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
                    model_dir, config, args.bit_width, args.output_dir, args.num_runs
                )
                results.append(python_metrics)
                print(f"  Average Time: {python_metrics.quantization_time_s:.2f}s")
                print(f"  Throughput: {python_metrics.throughput_params_per_sec / 1e6:.2f}M params/s")
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
            generate_charts(results, args.output_dir)
    else:
        print("No results to report")


if __name__ == "__main__":
    main()

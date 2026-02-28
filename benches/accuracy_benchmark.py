#!/usr/bin/env python3
"""
Accuracy Benchmark for ArrowQuant V2

Measures cosine similarity for different bit widths (INT2/INT4/INT8) and validates
accuracy targets across all modalities (text, code, image, audio).

Accuracy Targets:
- INT2: cosine_similarity >= 0.70
- INT4: cosine_similarity >= 0.90
- INT8: cosine_similarity >= 0.95

Usage:
    python benches/accuracy_benchmark.py --bit-width 2
    python benches/accuracy_benchmark.py --all-bit-widths
    python benches/accuracy_benchmark.py --all-modalities --output-dir ./benchmark_results
    python benches/accuracy_benchmark.py --compare-baseline --generate-charts
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
class AccuracyMetrics:
    """Accuracy metrics for a single quantization run."""
    modality: str  # "text", "code", "image", "audio"
    bit_width: int
    cosine_similarity: float
    per_layer_min: float
    per_layer_max: float
    per_layer_mean: float
    per_layer_std: float
    compression_ratio: float
    model_size_mb: float
    target_threshold: float
    target_met: bool
    enable_time_aware: bool
    enable_spatial: bool
    timestamp: str


@dataclass
class ModelConfig:
    """Configuration for synthetic model generation."""
    name: str
    modality: str
    num_layers: int
    hidden_size: int
    intermediate_size: int


# Model configurations for different modalities
MODEL_CONFIGS = {
    "text": ModelConfig(
        name="synthetic_text",
        modality="text",
        num_layers=12,
        hidden_size=768,
        intermediate_size=3072,
    ),
    "code": ModelConfig(
        name="synthetic_code",
        modality="code",
        num_layers=12,
        hidden_size=768,
        intermediate_size=3072,
    ),
    "image": ModelConfig(
        name="synthetic_image",
        modality="image",
        num_layers=24,
        hidden_size=1024,
        intermediate_size=4096,
    ),
    "audio": ModelConfig(
        name="synthetic_audio",
        modality="audio",
        num_layers=16,
        hidden_size=512,
        intermediate_size=2048,
    ),
}

# Accuracy targets for each bit width
ACCURACY_TARGETS = {
    2: 0.70,  # INT2
    4: 0.90,  # INT4
    8: 0.95,  # INT8
}


def generate_synthetic_model(config: ModelConfig, output_dir: Path) -> Path:
    """
    Generate a synthetic model with specified configuration.
    
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
        "modality": config.modality,
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


def benchmark_accuracy(
    model_dir: Path,
    config: ModelConfig,
    bit_width: int,
    output_dir: Path,
    enable_time_aware: bool = True,
    enable_spatial: bool = True,
) -> AccuracyMetrics:
    """
    Benchmark accuracy for a specific configuration.
    
    Args:
        model_dir: Path to model directory
        config: Model configuration
        bit_width: Target bit width (2, 4, or 8)
        output_dir: Output directory for quantized model
        enable_time_aware: Enable time-aware quantization
        enable_spatial: Enable spatial quantization
    
    Returns:
        Accuracy metrics
    """
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust ArrowQuantV2 not available")
    
    # Get target threshold for this bit width
    target_threshold = ACCURACY_TARGETS.get(bit_width, 0.85)
    
    # Create quantizer
    quant_config = DiffusionQuantConfig(
        bit_width=bit_width,
        num_time_groups=10,
        group_size=128,
        enable_time_aware=enable_time_aware,
        enable_spatial=enable_spatial,
        min_accuracy=target_threshold,
        calibration_samples=32,
    )
    
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Quantize model
    print(f"  Quantizing {config.modality} model to INT{bit_width}...")
    result = quantizer.quantize_diffusion_model(
        model_path=str(model_dir),
        output_path=str(output_dir / f"{config.name}_int{bit_width}"),
        config=quant_config.__dict__,
    )
    
    # Validate quality
    print(f"  Validating quality...")
    validation = quantizer.validate_quality(
        original_path=str(model_dir),
        quantized_path=str(output_dir / f"{config.name}_int{bit_width}"),
    )
    
    # Extract per-layer statistics
    per_layer_accuracies = list(validation["per_layer_accuracy"].values())
    per_layer_min = min(per_layer_accuracies) if per_layer_accuracies else 0.0
    per_layer_max = max(per_layer_accuracies) if per_layer_accuracies else 0.0
    per_layer_mean = sum(per_layer_accuracies) / len(per_layer_accuracies) if per_layer_accuracies else 0.0
    per_layer_std = np.std(per_layer_accuracies) if per_layer_accuracies else 0.0
    
    # Check if target is met
    cosine_similarity = validation["cosine_similarity"]
    target_met = cosine_similarity >= target_threshold
    
    return AccuracyMetrics(
        modality=config.modality,
        bit_width=bit_width,
        cosine_similarity=cosine_similarity,
        per_layer_min=per_layer_min,
        per_layer_max=per_layer_max,
        per_layer_mean=per_layer_mean,
        per_layer_std=per_layer_std,
        compression_ratio=validation["compression_ratio"],
        model_size_mb=validation["model_size_mb"],
        target_threshold=target_threshold,
        target_met=target_met,
        enable_time_aware=enable_time_aware,
        enable_spatial=enable_spatial,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def generate_comparison_report(
    results: List[AccuracyMetrics],
    output_dir: Path,
):
    """
    Generate accuracy comparison report.
    
    Args:
        results: List of accuracy metrics
        output_dir: Output directory for report
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate text report
    report_path = output_dir / "accuracy_benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ArrowQuant V2 Accuracy Benchmark Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary table by bit width
        f.write("Accuracy by Bit Width:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Bit Width':<12} {'Modality':<12} {'Cosine Sim':<15} {'Target':<10} {'Status':<10} {'Compression':<12}\n")
        f.write("-" * 80 + "\n")
        
        for metric in sorted(results, key=lambda m: (m.bit_width, m.modality)):
            status = "✅ Pass" if metric.target_met else "❌ Fail"
            f.write(f"{'INT' + str(metric.bit_width):<12} {metric.modality:<12} "
                   f"{metric.cosine_similarity:<15.4f} {metric.target_threshold:<10.2f} "
                   f"{status:<10} {metric.compression_ratio:<12.2f}x\n")
        
        f.write("\n")
        
        # Detailed results by modality
        modalities = sorted(set(m.modality for m in results))
        for modality in modalities:
            f.write(f"\nModality: {modality.upper()}\n")
            f.write("-" * 80 + "\n")
            
            modality_results = [m for m in results if m.modality == modality]
            
            for metric in sorted(modality_results, key=lambda m: m.bit_width):
                f.write(f"\nINT{metric.bit_width} Quantization:\n")
                f.write(f"  Cosine Similarity: {metric.cosine_similarity:.4f}\n")
                f.write(f"  Target Threshold: {metric.target_threshold:.2f}\n")
                f.write(f"  Target Met: {'✅ Yes' if metric.target_met else '❌ No'}\n")
                f.write(f"  Per-Layer Statistics:\n")
                f.write(f"    Min: {metric.per_layer_min:.4f}\n")
                f.write(f"    Max: {metric.per_layer_max:.4f}\n")
                f.write(f"    Mean: {metric.per_layer_mean:.4f}\n")
                f.write(f"    Std: {metric.per_layer_std:.4f}\n")
                f.write(f"  Compression Ratio: {metric.compression_ratio:.2f}x\n")
                f.write(f"  Model Size: {metric.model_size_mb:.2f} MB\n")
                f.write(f"  Time-Aware: {'Enabled' if metric.enable_time_aware else 'Disabled'}\n")
                f.write(f"  Spatial: {'Enabled' if metric.enable_spatial else 'Disabled'}\n")
        
        # Accuracy targets validation
        f.write("\n" + "=" * 80 + "\n")
        f.write("Accuracy Targets Validation\n")
        f.write("=" * 80 + "\n\n")
        
        for bit_width, target in ACCURACY_TARGETS.items():
            f.write(f"INT{bit_width} Target: cosine_similarity >= {target:.2f}\n")
            
            bit_width_results = [m for m in results if m.bit_width == bit_width]
            
            if bit_width_results:
                passed = sum(1 for m in bit_width_results if m.target_met)
                total = len(bit_width_results)
                pass_rate = (passed / total) * 100 if total > 0 else 0
                
                f.write(f"  Passed: {passed}/{total} ({pass_rate:.1f}%)\n")
                
                for metric in bit_width_results:
                    status = "✅" if metric.target_met else "❌"
                    f.write(f"    {status} {metric.modality}: {metric.cosine_similarity:.4f}\n")
            
            f.write("\n")
        
        # Optimization impact
        f.write("\n" + "=" * 80 + "\n")
        f.write("Optimization Impact Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        # Group by optimization settings
        opt_groups = {}
        for metric in results:
            key = (metric.enable_time_aware, metric.enable_spatial)
            if key not in opt_groups:
                opt_groups[key] = []
            opt_groups[key].append(metric)
        
        for (time_aware, spatial), metrics in opt_groups.items():
            opt_name = []
            if time_aware:
                opt_name.append("Time-Aware")
            if spatial:
                opt_name.append("Spatial")
            if not opt_name:
                opt_name.append("Baseline (No Optimizations)")
            
            f.write(f"{' + '.join(opt_name)}:\n")
            
            avg_similarity = sum(m.cosine_similarity for m in metrics) / len(metrics)
            avg_compression = sum(m.compression_ratio for m in metrics) / len(metrics)
            
            f.write(f"  Average Cosine Similarity: {avg_similarity:.4f}\n")
            f.write(f"  Average Compression Ratio: {avg_compression:.2f}x\n")
            f.write(f"  Samples: {len(metrics)}\n\n")
    
    print(f"Report generated: {report_path}")
    
    # Save JSON results
    json_path = output_dir / "accuracy_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump([asdict(m) for m in results], f, indent=2)
    
    print(f"JSON results saved: {json_path}")


def generate_charts(results: List[AccuracyMetrics], output_dir: Path):
    """
    Generate accuracy comparison charts.
    
    Args:
        results: List of accuracy metrics
        output_dir: Output directory for charts
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available. Skipping chart generation.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Chart 1: Accuracy by bit width
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bit_widths = sorted(set(m.bit_width for m in results))
    modalities = sorted(set(m.modality for m in results))
    
    x = np.arange(len(bit_widths))
    width = 0.2
    
    for idx, modality in enumerate(modalities):
        accuracies = []
        for bw in bit_widths:
            metric = next((m for m in results if m.bit_width == bw and m.modality == modality), None)
            accuracies.append(metric.cosine_similarity if metric else 0.0)
        
        ax.bar(x + idx * width, accuracies, width, label=modality.capitalize())
    
    # Add target thresholds
    for idx, bw in enumerate(bit_widths):
        target = ACCURACY_TARGETS.get(bw, 0.85)
        ax.axhline(y=target, color='r', linestyle='--', alpha=0.3)
        ax.text(idx, target + 0.01, f'Target: {target:.2f}', ha='center', fontsize=8)
    
    ax.set_xlabel('Bit Width')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Accuracy by Bit Width and Modality')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'INT{bw}' for bw in bit_widths])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    chart_path = output_dir / "accuracy_by_bit_width.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Chart generated: {chart_path}")
    plt.close()
    
    # Chart 2: Target achievement
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for modality in modalities:
        bit_widths_list = []
        target_met_list = []
        
        for bw in bit_widths:
            metric = next((m for m in results if m.bit_width == bw and m.modality == modality), None)
            if metric:
                bit_widths_list.append(f'INT{bw}')
                target_met_list.append(1.0 if metric.target_met else 0.0)
        
        ax.plot(bit_widths_list, target_met_list, marker='o', label=modality.capitalize(), linewidth=2, markersize=8)
    
    ax.set_xlabel('Bit Width')
    ax.set_ylabel('Target Met (1=Yes, 0=No)')
    ax.set_title('Accuracy Target Achievement')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    target_chart_path = output_dir / "target_achievement.png"
    plt.savefig(target_chart_path, dpi=300, bbox_inches='tight')
    print(f"Chart generated: {target_chart_path}")
    plt.close()
    
    # Chart 3: Accuracy vs Compression tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for modality in modalities:
        modality_results = [m for m in results if m.modality == modality]
        
        compressions = [m.compression_ratio for m in modality_results]
        accuracies = [m.cosine_similarity for m in modality_results]
        
        ax.scatter(compressions, accuracies, label=modality.capitalize(), s=100, alpha=0.7)
        
        # Add bit width labels
        for m in modality_results:
            ax.annotate(f'INT{m.bit_width}', (m.compression_ratio, m.cosine_similarity),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Compression Ratio')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Accuracy vs Compression Tradeoff')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    tradeoff_chart_path = output_dir / "accuracy_compression_tradeoff.png"
    plt.savefig(tradeoff_chart_path, dpi=300, bbox_inches='tight')
    print(f"Chart generated: {tradeoff_chart_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Accuracy benchmark for ArrowQuant V2")
    parser.add_argument(
        "--bit-width",
        type=int,
        choices=[2, 4, 8],
        help="Bit width to benchmark",
    )
    parser.add_argument(
        "--all-bit-widths",
        action="store_true",
        help="Benchmark all bit widths (INT2, INT4, INT8)",
    )
    parser.add_argument(
        "--modality",
        choices=["text", "code", "image", "audio"],
        help="Modality to benchmark",
    )
    parser.add_argument(
        "--all-modalities",
        action="store_true",
        help="Benchmark all modalities",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare with baseline (no optimizations)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".benchmarks/accuracy"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--generate-charts",
        action="store_true",
        help="Generate comparison charts",
    )
    
    args = parser.parse_args()
    
    if not RUST_AVAILABLE:
        print("Error: Rust ArrowQuantV2 not available. Install with: maturin develop --release")
        return 1
    
    if not args.bit_width and not args.all_bit_widths:
        parser.error("Either --bit-width or --all-bit-widths must be specified")
    
    if not args.modality and not args.all_modalities:
        parser.error("Either --modality or --all-modalities must be specified")
    
    # Determine which bit widths to benchmark
    if args.all_bit_widths:
        bit_widths = [2, 4, 8]
    else:
        bit_widths = [args.bit_width]
    
    # Determine which modalities to benchmark
    if args.all_modalities:
        modalities = ["text", "code", "image", "audio"]
    else:
        modalities = [args.modality]
    
    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    results = []
    
    for modality in modalities:
        print(f"\n{'='*80}")
        print(f"Benchmarking {modality} model")
        print(f"{'='*80}\n")
        
        config = MODEL_CONFIGS[modality]
        
        # Generate synthetic model
        print(f"Generating synthetic model...")
        model_dir = generate_synthetic_model(config, models_dir)
        
        for bit_width in bit_widths:
            print(f"\nBenchmarking INT{bit_width}...")
            
            try:
                # Benchmark with optimizations
                metrics = benchmark_accuracy(
                    model_dir, config, bit_width, args.output_dir,
                    enable_time_aware=True, enable_spatial=True
                )
                results.append(metrics)
                
                print(f"  Cosine Similarity: {metrics.cosine_similarity:.4f}")
                print(f"  Target: {metrics.target_threshold:.2f}")
                print(f"  Target Met: {'✅ Yes' if metrics.target_met else '❌ No'}")
                print(f"  Compression Ratio: {metrics.compression_ratio:.2f}x")
                
                # Compare with baseline if requested
                if args.compare_baseline:
                    print(f"\nBenchmarking INT{bit_width} baseline (no optimizations)...")
                    baseline_metrics = benchmark_accuracy(
                        model_dir, config, bit_width, args.output_dir,
                        enable_time_aware=False, enable_spatial=False
                    )
                    results.append(baseline_metrics)
                    
                    improvement = metrics.cosine_similarity - baseline_metrics.cosine_similarity
                    print(f"  Baseline Cosine Similarity: {baseline_metrics.cosine_similarity:.4f}")
                    print(f"  Improvement: {improvement:+.4f} ({improvement/baseline_metrics.cosine_similarity*100:+.2f}%)")
                
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
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

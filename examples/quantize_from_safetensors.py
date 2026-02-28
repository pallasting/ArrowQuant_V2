"""
Example: Quantize a model from SafeTensors format

This example demonstrates how to load a model from SafeTensors format
(single-file or sharded) and quantize it using ArrowQuant V2.

Supports both single-file and sharded SafeTensors models.

Usage:
    # Single-file model
    python examples/quantize_from_safetensors.py \\
        --input model.safetensors \\
        --output model_int2/ \\
        --bit-width 2 \\
        --profile edge
    
    # Sharded model (auto-detected)
    python examples/quantize_from_safetensors.py \\
        --input model.safetensors.index.json \\
        --output model_int2/ \\
        --bit-width 2
    
    # Directory with sharded model
    python examples/quantize_from_safetensors.py \\
        --input model_directory/ \\
        --output model_int2/ \\
        --bit-width 2
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.safetensors_loader import (
    SafeTensorsLoader,
    ShardedSafeTensorsLoader,
    load_safetensors_model,
)
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a model from SafeTensors format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input .safetensors file, .safetensors.index.json, or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--bit-width",
        type=int,
        choices=[2, 4, 8],
        default=4,
        help="Target bit width (default: 4)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["edge", "local", "cloud"],
        default="local",
        help="Deployment profile (default: local)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["text", "code", "image", "audio"],
        default=None,
        help="Model modality (auto-detected if not specified)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate quantization quality",
    )
    
    args = parser.parse_args()
    
    # Load model info
    print("=" * 60)
    print("Loading SafeTensors model...")
    print("=" * 60)
    
    # Auto-detect single-file vs sharded
    input_path = Path(args.input)
    is_sharded = ShardedSafeTensorsLoader.is_sharded_model(str(input_path))
    
    if is_sharded:
        print(f"Detected sharded model at {args.input}")
        loader = ShardedSafeTensorsLoader(args.input)
    else:
        print(f"Detected single-file model at {args.input}")
        loader = SafeTensorsLoader(args.input)
    
    print(loader.summary())
    
    # Detect modality if not specified
    modality = args.modality or loader.detect_modality()
    if modality:
        print(f"\nDetected modality: {modality}")
    else:
        print("\nWarning: Could not detect modality, using default settings")
    
    # Create quantization config
    print("\n" + "=" * 60)
    print("Configuring quantization...")
    print("=" * 60)
    
    # Create config with correct bit_width directly
    # Cannot modify frozen config from from_profile, so create new one
    if args.profile == "edge":
        config = DiffusionQuantConfig(
            bit_width=args.bit_width,
            modality=modality,
            num_time_groups=5,
            group_size=256,
            min_accuracy=0.65,
            calibration_samples=32,
            deployment_profile="edge"
        )
    elif args.profile == "local":
        config = DiffusionQuantConfig(
            bit_width=args.bit_width,
            modality=modality,
            num_time_groups=10,
            group_size=128,
            min_accuracy=0.85,
            calibration_samples=128,
            deployment_profile="local"
        )
    elif args.profile == "cloud":
        config = DiffusionQuantConfig(
            bit_width=args.bit_width,
            modality=modality,
            num_time_groups=20,
            group_size=64,
            min_accuracy=0.95,
            calibration_samples=512,
            deployment_profile="cloud"
        )
    else:
        # Fallback to default
        config = DiffusionQuantConfig(
            bit_width=args.bit_width,
            modality=modality,
            deployment_profile=args.profile
        )
    
    print(f"Profile: {args.profile}")
    print(f"Bit width: {args.bit_width}")
    print(f"Modality: {modality or 'auto-detect'}")
    
    # Create quantizer
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Quantize model
    print("\n" + "=" * 60)
    print("Quantizing model...")
    print("=" * 60)
    
    def progress_callback(message: str, progress: float):
        """Progress callback for quantization."""
        print(f"[{progress:5.1f}%] {message}")
    
    result = quantizer.quantize_from_safetensors(
        safetensors_path=args.input,
        output_path=args.output,
        config=config,
        progress_callback=progress_callback,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Quantization complete!")
    print("=" * 60)
    
    print(f"\nResults:")
    print(f"  Output path: {result['quantized_path']}")
    print(f"  Compression ratio: {result['compression_ratio']:.2f}x")
    print(f"  Cosine similarity: {result['cosine_similarity']:.4f}")
    print(f"  Model size: {result['model_size_mb']:.2f} MB")
    print(f"  Quantization time: {result['quantization_time_s']:.2f}s")
    
    if modality:
        print(f"  Modality: {result['modality']}")
    
    # Validate if requested
    if args.validate:
        print("\n" + "=" * 60)
        print("Validating quantization quality...")
        print("=" * 60)
        
        validation = quantizer.validate_quality(
            original_path=args.input,
            quantized_path=args.output,
        )
        
        print(f"\nValidation results:")
        print(f"  Overall cosine similarity: {validation['cosine_similarity']:.4f}")
        print(f"  Compression ratio: {validation['compression_ratio']:.2f}x")
        
        # Show per-layer accuracy
        per_layer = validation.get("per_layer_accuracy", {})
        if per_layer:
            print(f"\n  Per-layer accuracy (top 5 worst):")
            sorted_layers = sorted(
                per_layer.items(),
                key=lambda x: x[1]
            )[:5]
            for layer_name, accuracy in sorted_layers:
                print(f"    {layer_name}: {accuracy:.4f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

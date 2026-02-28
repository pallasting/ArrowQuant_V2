#!/usr/bin/env python3
"""
Example: Automatic Granularity Allocation

This example demonstrates how to use the automatic granularity allocation
feature to optimize layer-wise group sizes for quantization.
"""

import json
from pathlib import Path

# Note: This is a conceptual example showing the workflow
# The actual Rust implementation would be called via PyO3 bindings


def example_basic_allocation():
    """Basic example: Analyze and allocate group sizes."""
    print("=" * 80)
    print("Example 1: Basic Granularity Allocation")
    print("=" * 80)
    
    # Configuration
    config = {
        "sensitivity_method": "gradient",
        "num_samples": 32,
        "target_compression_ratio": 10.0,
        "min_accuracy": 0.70,
        "available_group_sizes": [32, 64, 128, 256],
        "accuracy_weight": 0.7,
    }
    
    print("\nConfiguration:")
    print(f"  Sensitivity method: {config['sensitivity_method']}")
    print(f"  Minimum accuracy: {config['min_accuracy']}")
    print(f"  Target compression: {config['target_compression_ratio']}x")
    print(f"  Accuracy weight: {config['accuracy_weight']}")
    
    # Simulated layer names
    layer_names = [
        "transformer.layer.0.attention.q_proj",
        "transformer.layer.0.attention.k_proj",
        "transformer.layer.0.attention.v_proj",
        "transformer.layer.0.mlp.fc1",
        "transformer.layer.0.mlp.fc2",
        "transformer.layer.0.norm1",
        "transformer.layer.0.norm2",
    ]
    
    print(f"\nAnalyzing {len(layer_names)} layers...")
    
    # Simulated allocation results
    allocation = {
        "layers": {
            "transformer.layer.0.attention.q_proj": {
                "sensitivity": 0.85,
                "group_size": 64,
                "estimated_accuracy": 0.88,
                "estimated_compression": 11.2,
            },
            "transformer.layer.0.attention.k_proj": {
                "sensitivity": 0.82,
                "group_size": 64,
                "estimated_accuracy": 0.87,
                "estimated_compression": 11.2,
            },
            "transformer.layer.0.attention.v_proj": {
                "sensitivity": 0.80,
                "group_size": 128,
                "estimated_accuracy": 0.85,
                "estimated_compression": 12.5,
            },
            "transformer.layer.0.mlp.fc1": {
                "sensitivity": 0.55,
                "group_size": 128,
                "estimated_accuracy": 0.82,
                "estimated_compression": 12.5,
            },
            "transformer.layer.0.mlp.fc2": {
                "sensitivity": 0.52,
                "group_size": 256,
                "estimated_accuracy": 0.80,
                "estimated_compression": 14.8,
            },
            "transformer.layer.0.norm1": {
                "sensitivity": 0.92,
                "group_size": 32,
                "estimated_accuracy": 0.91,
                "estimated_compression": 9.5,
            },
            "transformer.layer.0.norm2": {
                "sensitivity": 0.90,
                "group_size": 32,
                "estimated_accuracy": 0.90,
                "estimated_compression": 9.5,
            },
        },
        "summary": {
            "num_layers": 7,
            "avg_accuracy": 0.86,
            "avg_compression": 11.6,
        },
    }
    
    print("\nAllocation Results:")
    print(f"  Average accuracy: {allocation['summary']['avg_accuracy']:.4f}")
    print(f"  Average compression: {allocation['summary']['avg_compression']:.2f}x")
    
    print("\nLayer-wise allocation:")
    print(f"{'Layer':<50} {'Sensitivity':>12} {'Group Size':>12}")
    print("-" * 80)
    for layer_name, data in allocation["layers"].items():
        print(f"{layer_name:<50} {data['sensitivity']:>12.4f} {data['group_size']:>12d}")
    
    # Group size distribution
    group_size_counts = {}
    for data in allocation["layers"].values():
        size = data["group_size"]
        group_size_counts[size] = group_size_counts.get(size, 0) + 1
    
    print("\nGroup size distribution:")
    for size in sorted(group_size_counts.keys()):
        count = group_size_counts[size]
        percentage = (count / len(allocation["layers"])) * 100
        print(f"  {size:3d}: {count:2d} layers ({percentage:5.1f}%)")


def example_accuracy_focused():
    """Example 2: Accuracy-focused allocation."""
    print("\n" + "=" * 80)
    print("Example 2: Accuracy-Focused Allocation")
    print("=" * 80)
    
    config = {
        "sensitivity_method": "gradient",
        "min_accuracy": 0.85,  # Higher minimum
        "accuracy_weight": 0.9,  # High accuracy weight
    }
    
    print("\nConfiguration:")
    print(f"  Minimum accuracy: {config['min_accuracy']}")
    print(f"  Accuracy weight: {config['accuracy_weight']} (accuracy-focused)")
    
    print("\nExpected behavior:")
    print("  - Smaller group sizes preferred (32, 64)")
    print("  - Higher accuracy maintained")
    print("  - Lower compression ratio")
    print("  - Suitable for: Production models, critical applications")


def example_compression_focused():
    """Example 3: Compression-focused allocation."""
    print("\n" + "=" * 80)
    print("Example 3: Compression-Focused Allocation")
    print("=" * 80)
    
    config = {
        "sensitivity_method": "gradient",
        "min_accuracy": 0.65,  # Lower minimum
        "accuracy_weight": 0.3,  # Low accuracy weight
        "target_compression_ratio": 15.0,  # Higher target
    }
    
    print("\nConfiguration:")
    print(f"  Minimum accuracy: {config['min_accuracy']}")
    print(f"  Accuracy weight: {config['accuracy_weight']} (compression-focused)")
    print(f"  Target compression: {config['target_compression_ratio']}x")
    
    print("\nExpected behavior:")
    print("  - Larger group sizes preferred (128, 256)")
    print("  - Higher compression ratio")
    print("  - Lower accuracy (but above minimum)")
    print("  - Suitable for: Edge devices, resource-constrained environments")


def example_sensitivity_methods():
    """Example 4: Different sensitivity methods."""
    print("\n" + "=" * 80)
    print("Example 4: Sensitivity Analysis Methods")
    print("=" * 80)
    
    methods = {
        "gradient": {
            "description": "L2 norm of gradients",
            "speed": "Fast",
            "accuracy": "Good",
            "use_case": "General purpose, default choice",
        },
        "hessian": {
            "description": "Diagonal of Hessian matrix",
            "speed": "Slower",
            "accuracy": "Better",
            "use_case": "High-accuracy requirements, sensitive layers",
        },
        "variance": {
            "description": "Weight variance",
            "speed": "Fastest",
            "accuracy": "Approximate",
            "use_case": "Quick analysis, initial exploration",
        },
    }
    
    print("\nAvailable methods:")
    for method, info in methods.items():
        print(f"\n{method.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Speed: {info['speed']}")
        print(f"  Accuracy: {info['accuracy']}")
        print(f"  Use case: {info['use_case']}")


def example_cli_usage():
    """Example 5: CLI usage."""
    print("\n" + "=" * 80)
    print("Example 5: Command-Line Usage")
    print("=" * 80)
    
    print("\nBasic usage:")
    print("  python scripts/granularity_allocation.py \\")
    print("      --model models/dream-7b \\")
    print("      --output allocation.json \\")
    print("      --method gradient \\")
    print("      --min-accuracy 0.70")
    
    print("\nAccuracy-focused:")
    print("  python scripts/granularity_allocation.py \\")
    print("      --model models/dream-7b \\")
    print("      --output accurate.json \\")
    print("      --accuracy-weight 0.9 \\")
    print("      --min-accuracy 0.85")
    
    print("\nCompression-focused:")
    print("  python scripts/granularity_allocation.py \\")
    print("      --model models/dream-7b \\")
    print("      --output compressed.json \\")
    print("      --accuracy-weight 0.3 \\")
    print("      --target-compression 15.0")
    
    print("\nCustom group sizes:")
    print("  python scripts/granularity_allocation.py \\")
    print("      --model models/dream-7b \\")
    print("      --output custom.json \\")
    print("      --available-sizes 64 128 256")


def example_integration():
    """Example 6: Integration with quantization."""
    print("\n" + "=" * 80)
    print("Example 6: Integration with Quantization")
    print("=" * 80)
    
    print("\nWorkflow:")
    print("  1. Analyze layer sensitivity")
    print("  2. Allocate optimal group sizes")
    print("  3. Apply allocation to quantization config")
    print("  4. Quantize model with optimized settings")
    
    print("\nPseudo-code:")
    print("""
    # Step 1: Allocate group sizes
    allocator = GranularityAllocator(config)
    allocation = allocator.allocate(model_path, base_config, layer_names)
    
    # Step 2: Create quantization config with allocation
    quant_config = DiffusionQuantConfig(
        bit_width=2,
        layer_group_sizes=allocation.layer_group_sizes,
    )
    
    # Step 3: Quantize with optimized config
    orchestrator = DiffusionOrchestrator(quant_config)
    result = orchestrator.quantize_model(model_path, output_path)
    
    # Step 4: Verify results
    print(f"Accuracy: {result.cosine_similarity:.4f}")
    print(f"Compression: {result.compression_ratio:.2f}x")
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("AUTOMATIC GRANULARITY ALLOCATION EXAMPLES")
    print("=" * 80)
    
    example_basic_allocation()
    example_accuracy_focused()
    example_compression_focused()
    example_sensitivity_methods()
    example_cli_usage()
    example_integration()
    
    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
    print("\nFor more information:")
    print("  - Documentation: docs/CONFIGURATION_GUIDE.md")
    print("  - CLI help: python scripts/granularity_allocation.py --help")
    print("  - Tests: cargo test --test test_granularity")
    print()


if __name__ == "__main__":
    main()

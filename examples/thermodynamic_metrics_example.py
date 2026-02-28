#!/usr/bin/env python3
"""
Example: Thermodynamic Metrics Collection

This example demonstrates how to enable thermodynamic validation and
retrieve Markov smoothness metrics from the quantization process.

Requirements:
    - REQ-1.1.3: Metrics collection and exposure via Python API
    - REQ-2.4.3: Observability through logging and metrics
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
except ImportError:
    print("Error: arrow_quant_v2 module not found.")
    print("Please build the module first:")
    print("  cd ai_os_diffusion/arrow_quant_v2")
    print("  maturin develop --release")
    sys.exit(1)


def main():
    """Demonstrate thermodynamic metrics collection."""
    
    print("=" * 70)
    print("Thermodynamic Metrics Collection Example")
    print("=" * 70)
    print()
    
    # Create quantizer instance
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Create configuration with thermodynamic validation enabled
    config = DiffusionQuantConfig(
        bit_width=2,  # INT2 quantization
        num_time_groups=4,
        enable_time_aware=True,
    )
    
    # Enable thermodynamic validation
    # Note: In the current implementation, thermodynamic config is set via YAML
    # or through the Rust API. This example shows the intended Python API.
    print("Configuration:")
    print(f"  Bit width: {config.bit_width}")
    print(f"  Time groups: {config.num_time_groups}")
    print(f"  Time-aware: {config.enable_time_aware}")
    print()
    
    # Note: The get_thermodynamic_metrics() method is available but requires
    # quantization to be performed first. In a real scenario:
    #
    # 1. Quantize a model:
    #    result = quantizer.quantize_diffusion_model(
    #        model_path="path/to/model",
    #        output_path="path/to/output",
    #        config=config
    #    )
    #
    # 2. Retrieve metrics:
    #    metrics = quantizer.get_thermodynamic_metrics()
    #
    # 3. Analyze metrics:
    #    if metrics:
    #        print(f"Smoothness score: {metrics['smoothness_score']:.3f}")
    #        print(f"Violations: {metrics['violation_count']}")
    #        print(f"Valid: {metrics['is_valid']}")
    #        
    #        for i, score in enumerate(metrics['boundary_scores']):
    #            print(f"  Boundary {i}: {score:.3f}")
    #        
    #        if metrics['violations']:
    #            print("\nViolations detected:")
    #            for v in metrics['violations']:
    #                print(f"  Boundary {v['boundary_idx']}: "
    #                      f"{v['scale_jump']*100:.1f}% jump "
    #                      f"(severity: {v['severity']})")
    
    print("Metrics API:")
    print("  quantizer.get_thermodynamic_metrics() -> Optional[Dict]")
    print()
    print("Metrics structure:")
    print("  {")
    print("    'smoothness_score': float,      # 0-1, higher is better")
    print("    'boundary_scores': List[float], # Per-boundary scores")
    print("    'violation_count': int,         # Number of violations")
    print("    'violations': List[Dict],       # Violation details")
    print("    'is_valid': bool                # True if no violations")
    print("  }")
    print()
    
    # Try to get metrics (will be None if no quantization performed)
    metrics = quantizer.get_thermodynamic_metrics()
    if metrics is None:
        print("Status: No metrics available (no quantization performed yet)")
        print()
        print("To collect metrics:")
        print("  1. Enable thermodynamic validation in config")
        print("  2. Perform quantization")
        print("  3. Call get_thermodynamic_metrics()")
    else:
        print("Metrics retrieved successfully!")
        print(f"  Smoothness score: {metrics['smoothness_score']:.3f}")
        print(f"  Violations: {metrics['violation_count']}")
        print(f"  Valid: {metrics['is_valid']}")
    
    print()
    print("=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

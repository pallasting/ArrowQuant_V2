"""
Test SafeTensors Integration Status

This script tests what parts of the SafeTensors integration are working
and provides clear diagnostics about what's missing.

Usage:
    python examples/test_safetensors_integration.py J:\\dream-7b
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_safetensors_loader(model_path: str):
    """Test if SafeTensors loader works."""
    print("=" * 60)
    print("TEST 1: SafeTensors Loader")
    print("=" * 60)
    
    try:
        from python.safetensors_loader import SafeTensorsLoader, ShardedSafeTensorsLoader
        
        # Check if it's a sharded model
        is_sharded = ShardedSafeTensorsLoader.is_sharded_model(model_path)
        
        if is_sharded:
            print(f"✓ Detected sharded model at {model_path}")
            loader = ShardedSafeTensorsLoader(model_path)
        else:
            print(f"✓ Detected single-file model at {model_path}")
            loader = SafeTensorsLoader(model_path)
        
        print("\nModel Summary:")
        print(loader.summary())
        
        print("\n✓ SafeTensors loader works correctly")
        return True, loader
        
    except Exception as e:
        print(f"\n✗ SafeTensors loader failed: {e}")
        return False, None


def test_arrow_quant_import():
    """Test if ArrowQuantV2 can be imported."""
    print("\n" + "=" * 60)
    print("TEST 2: ArrowQuantV2 Import")
    print("=" * 60)
    
    try:
        from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
        print("✓ ArrowQuantV2 imported successfully")
        
        # Create instance
        quantizer = ArrowQuantV2(mode="diffusion")
        print("✓ ArrowQuantV2 instance created")
        
        return True, quantizer
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("\nHint: You need to build the Rust library first:")
        print("  cd ai_os_diffusion/arrow_quant_v2")
        print("  maturin develop --release")
        return False, None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False, None


def test_quantize_from_safetensors_method(quantizer):
    """Test if quantize_from_safetensors method exists."""
    print("\n" + "=" * 60)
    print("TEST 3: quantize_from_safetensors Method")
    print("=" * 60)
    
    if quantizer is None:
        print("✗ Skipped (quantizer not available)")
        return False
    
    # Check if method exists
    if hasattr(quantizer, 'quantize_from_safetensors'):
        print("✓ quantize_from_safetensors method exists")
        return True
    else:
        print("✗ quantize_from_safetensors method NOT FOUND")
        print("\nAvailable methods:")
        methods = [m for m in dir(quantizer) if not m.startswith('_')]
        for method in methods:
            print(f"  - {method}")
        
        print("\n⚠️  INTEGRATION INCOMPLETE")
        print("The quantize_from_safetensors method needs to be implemented in:")
        print("  ai_os_diffusion/arrow_quant_v2/src/python.rs")
        return False


def test_quantize_diffusion_model_method(quantizer):
    """Test if quantize_diffusion_model method exists."""
    print("\n" + "=" * 60)
    print("TEST 4: quantize_diffusion_model Method")
    print("=" * 60)
    
    if quantizer is None:
        print("✗ Skipped (quantizer not available)")
        return False
    
    if hasattr(quantizer, 'quantize_diffusion_model'):
        print("✓ quantize_diffusion_model method exists")
        print("\nThis method requires Parquet format input:")
        print("  quantizer.quantize_diffusion_model(")
        print("      model_path='model_parquet/',  # Must be Parquet")
        print("      output_path='output/',")
        print("      config=config")
        print("  )")
        return True
    else:
        print("✗ quantize_diffusion_model method NOT FOUND")
        return False


def print_summary(results):
    """Print test summary and recommendations."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    loader_ok, quant_import_ok, from_safetensors_ok, from_parquet_ok = results
    
    print(f"\n1. SafeTensors Loader:        {'✓ WORKING' if loader_ok else '✗ FAILED'}")
    print(f"2. ArrowQuantV2 Import:       {'✓ WORKING' if quant_import_ok else '✗ FAILED'}")
    print(f"3. quantize_from_safetensors: {'✓ WORKING' if from_safetensors_ok else '✗ NOT IMPLEMENTED'}")
    print(f"4. quantize_diffusion_model:  {'✓ WORKING' if from_parquet_ok else '✗ FAILED'}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if not quant_import_ok:
        print("\n⚠️  Build the Rust library first:")
        print("  cd ai_os_diffusion/arrow_quant_v2")
        print("  maturin develop --release")
        
    elif not from_safetensors_ok:
        print("\n⚠️  SafeTensors → Quantization integration is INCOMPLETE")
        print("\nThe SafeTensors adapter exists but is not connected to the")
        print("quantization workflow. You have two options:")
        print("\nOption A: Implement the missing integration (recommended)")
        print("  - Add quantize_from_safetensors() to src/python.rs")
        print("  - Implement SafeTensors → Parquet conversion")
        print("  - Update orchestrator to support SafeTensors input")
        print("\nOption B: Manual workaround (temporary)")
        print("  1. Convert SafeTensors → Parquet format manually")
        print("  2. Use quantize_diffusion_model() on Parquet model")
        print("\nSee SAFETENSORS_INTEGRATION_STATUS.md for details")
        
    else:
        print("\n✓ All components working!")
        print("\nYou can now quantize SafeTensors models directly:")
        print("  python examples/quantize_from_safetensors.py \\")
        print("      --input J:\\dream-7b \\")
        print("      --output F:\\models\\dream-7b-int4 \\")
        print("      --bit-width 4")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_safetensors_integration.py <model_path>")
        print("\nExample:")
        print("  python test_safetensors_integration.py J:\\dream-7b")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print("SafeTensors Integration Test")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print()
    
    # Run tests
    loader_ok, loader = test_safetensors_loader(model_path)
    quant_import_ok, quantizer = test_arrow_quant_import()
    from_safetensors_ok = test_quantize_from_safetensors_method(quantizer)
    from_parquet_ok = test_quantize_diffusion_model_method(quantizer)
    
    # Print summary
    results = (loader_ok, quant_import_ok, from_safetensors_ok, from_parquet_ok)
    print_summary(results)


if __name__ == "__main__":
    main()

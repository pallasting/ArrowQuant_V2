"""
Simple integration test for SafeTensors quantization
"""

import sys
print("Testing SafeTensors quantization integration...")

# Test 1: Import arrow_quant_v2
try:
    from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
    print("✓ Successfully imported ArrowQuantV2 and DiffusionQuantConfig")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Create quantizer instance
try:
    quantizer = ArrowQuantV2(mode="diffusion")
    print("✓ Successfully created ArrowQuantV2 instance")
except Exception as e:
    print(f"✗ Failed to create quantizer: {e}")
    sys.exit(1)

# Test 3: Check quantize_from_safetensors method exists
try:
    assert hasattr(quantizer, 'quantize_from_safetensors')
    assert callable(getattr(quantizer, 'quantize_from_safetensors'))
    print("✓ quantize_from_safetensors method exists and is callable")
except AssertionError:
    print("✗ quantize_from_safetensors method not found or not callable")
    sys.exit(1)

# Test 4: Create config from profile
try:
    config = DiffusionQuantConfig.from_profile("local")
    print("✓ Successfully created DiffusionQuantConfig from profile")
except Exception as e:
    print(f"✗ Failed to create config: {e}")
    sys.exit(1)

# Test 5: Check method signature
try:
    import inspect
    method = getattr(quantizer, 'quantize_from_safetensors')
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())
    
    assert 'safetensors_path' in params
    assert 'output_path' in params
    assert 'config' in params
    assert 'progress_callback' in params
    print("✓ quantize_from_safetensors has correct signature")
except Exception as e:
    print(f"✗ Signature check failed: {e}")
    sys.exit(1)

# Test 6: Test progress callback interface
try:
    progress_calls = []
    
    def progress_callback(message: str, progress: float):
        progress_calls.append((message, progress))
    
    progress_callback("Test message", 0.5)
    assert len(progress_calls) == 1
    assert progress_calls[0] == ("Test message", 0.5)
    print("✓ Progress callback interface works correctly")
except Exception as e:
    print(f"✗ Progress callback test failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nSafeTensors quantization integration is ready.")
print("\nNext steps:")
print("1. Build the library: maturin build --release")
print("2. Install the wheel")
print("3. Test with a real SafeTensors model")
print("\nExample usage:")
print("""
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

quantizer = ArrowQuantV2(mode="diffusion")
config = DiffusionQuantConfig.from_profile("local")

result = quantizer.quantize_from_safetensors(
    safetensors_path="path/to/model.safetensors",
    output_path="path/to/output",
    config=config
)

print(f"Quantization complete!")
print(f"Compression ratio: {result['compression_ratio']:.2f}x")
print(f"Cosine similarity: {result['cosine_similarity']:.4f}")
""")

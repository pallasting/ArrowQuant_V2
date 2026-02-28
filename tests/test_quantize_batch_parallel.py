"""
Test parallel processing in quantize_batch API.

This test verifies that:
1. Batch API processes multiple layers correctly
2. Results are deterministic (same order as input)
3. Parallel processing produces same results as sequential
4. Thread-safe error handling works correctly
"""

import numpy as np
import pytest


def test_quantize_batch_basic():
    """Test basic batch quantization with multiple layers."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Create test data with multiple layers
    weights_dict = {
        "layer.0.weight": np.random.randn(1000).astype(np.float32),
        "layer.1.weight": np.random.randn(2000).astype(np.float32),
        "layer.2.weight": np.random.randn(1500).astype(np.float32),
    }
    
    # Quantize batch
    results = quantizer.quantize_batch(weights_dict, bit_width=4)
    
    # Verify all layers processed
    assert len(results) == 3
    assert "layer.0.weight" in results
    assert "layer.1.weight" in results
    assert "layer.2.weight" in results
    
    # Verify each result has expected fields
    for layer_name, result in results.items():
        assert "quantized_data" in result
        assert "scales" in result
        assert "zero_points" in result
        assert "shape" in result
        assert "bit_width" in result
        assert result["bit_width"] == 4


def test_quantize_batch_deterministic():
    """Test that batch quantization produces deterministic results."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Create test data
    np.random.seed(42)
    weights_dict = {
        f"layer.{i}.weight": np.random.randn(1000).astype(np.float32)
        for i in range(10)
    }
    
    # Quantize twice
    results1 = quantizer.quantize_batch(weights_dict, bit_width=4)
    results2 = quantizer.quantize_batch(weights_dict, bit_width=4)
    
    # Verify results are identical
    assert set(results1.keys()) == set(results2.keys())
    
    for layer_name in results1.keys():
        r1 = results1[layer_name]
        r2 = results2[layer_name]
        
        # Compare quantized data
        assert r1["quantized_data"] == r2["quantized_data"]
        
        # Compare scales (should be very close)
        scales1 = np.array(r1["scales"])
        scales2 = np.array(r2["scales"])
        np.testing.assert_allclose(scales1, scales2, rtol=1e-6)


def test_quantize_batch_empty():
    """Test batch quantization with empty dictionary."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Empty dictionary should return empty results
    results = quantizer.quantize_batch({}, bit_width=4)
    assert len(results) == 0


def test_quantize_batch_invalid_bit_width():
    """Test batch quantization with invalid bit width."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    weights_dict = {
        "layer.0.weight": np.random.randn(1000).astype(np.float32),
    }
    
    # Invalid bit width should raise error
    with pytest.raises(ValueError, match="Invalid bit_width"):
        quantizer.quantize_batch(weights_dict, bit_width=3)


def test_quantize_batch_invalid_array():
    """Test batch quantization with invalid array."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Non-contiguous array
    arr = np.random.randn(100, 100).astype(np.float32)
    non_contiguous = arr[::2, ::2]  # Create non-contiguous view
    
    weights_dict = {
        "layer.0.weight": non_contiguous,
    }
    
    # Should raise error with layer name
    with pytest.raises(ValueError, match="layer.0.weight.*contiguous"):
        quantizer.quantize_batch(weights_dict, bit_width=4)


def test_quantize_batch_large_scale():
    """Test batch quantization with many layers (stress test for parallel processing)."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Create 50 layers to test parallel processing
    np.random.seed(42)
    weights_dict = {
        f"layer.{i}.weight": np.random.randn(1000).astype(np.float32)
        for i in range(50)
    }
    
    # Quantize batch
    results = quantizer.quantize_batch(weights_dict, bit_width=4)
    
    # Verify all layers processed
    assert len(results) == 50
    
    # Verify each result is valid
    for i in range(50):
        layer_name = f"layer.{i}.weight"
        assert layer_name in results
        result = results[layer_name]
        assert len(result["quantized_data"]) > 0
        assert len(result["scales"]) > 0


def test_quantize_batch_vs_sequential():
    """Test that batch API produces same results as sequential calls."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Create test data
    np.random.seed(42)
    weights_dict = {
        "layer.0.weight": np.random.randn(1000).astype(np.float32),
        "layer.1.weight": np.random.randn(1000).astype(np.float32),
        "layer.2.weight": np.random.randn(1000).astype(np.float32),
    }
    
    # Quantize using batch API
    batch_results = quantizer.quantize_batch(weights_dict, bit_width=4)
    
    # Quantize using sequential single-layer batch calls
    sequential_results = {}
    for layer_name, weights in weights_dict.items():
        single_layer_dict = {layer_name: weights}
        single_result = quantizer.quantize_batch(single_layer_dict, bit_width=4)
        sequential_results[layer_name] = single_result[layer_name]
    
    # Compare results - they should be identical since same algorithm
    for layer_name in weights_dict.keys():
        batch_result = batch_results[layer_name]
        seq_result = sequential_results[layer_name]
        
        # Quantized data should be identical
        assert batch_result["quantized_data"] == seq_result["quantized_data"]
        
        # Scales should be identical
        batch_scales = np.array(batch_result["scales"])
        seq_scales = np.array(seq_result["scales"])
        np.testing.assert_array_equal(batch_scales, seq_scales)
        
        # Zero points should be identical
        batch_zp = np.array(batch_result["zero_points"])
        seq_zp = np.array(seq_result["zero_points"])
        np.testing.assert_array_equal(batch_zp, seq_zp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

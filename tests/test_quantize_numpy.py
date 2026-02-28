"""
Unit tests for quantize_numpy() method.

Tests the zero-copy numpy interface for quantization.
"""

import numpy as np
import pytest


def test_quantize_numpy_basic():
    """Test basic quantization with numpy array."""
    from arrow_quant_v2 import ArrowQuantV2
    
    # Create quantizer
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Create test data
    weights = np.random.randn(1000).astype(np.float32)
    
    # Quantize
    result = quantizer.quantize_numpy(weights, "test_layer", bit_width=4)
    
    # Verify result structure
    assert "quantized_data" in result
    assert "scales" in result
    assert "zero_points" in result
    assert "shape" in result
    assert "bit_width" in result
    assert "layer_name" in result
    
    # Verify types
    assert isinstance(result["quantized_data"], np.ndarray)
    assert isinstance(result["scales"], np.ndarray)
    assert isinstance(result["zero_points"], np.ndarray)
    assert result["quantized_data"].dtype == np.uint8
    assert result["scales"].dtype == np.float32
    assert result["zero_points"].dtype == np.float32
    
    # Verify shapes
    assert result["quantized_data"].shape[0] == 1000
    assert result["shape"] == (1000,)
    assert result["bit_width"] == 4
    assert result["layer_name"] == "test_layer"


def test_quantize_numpy_different_bit_widths():
    """Test quantization with different bit widths."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(500).astype(np.float32)
    
    for bit_width in [2, 4, 8]:
        result = quantizer.quantize_numpy(weights, f"layer_{bit_width}bit", bit_width=bit_width)
        assert result["bit_width"] == bit_width
        assert result["quantized_data"].shape[0] == 500


def test_quantize_numpy_invalid_bit_width():
    """Test that invalid bit width raises ValueError."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(100).astype(np.float32)
    
    with pytest.raises(ValueError, match="Invalid bit_width"):
        quantizer.quantize_numpy(weights, "test_layer", bit_width=3)


def test_quantize_numpy_wrong_dtype():
    """Test that wrong dtype raises ValueError."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(100).astype(np.float64)  # Wrong dtype
    
    # This should fail because numpy 0.22 will not accept float64 as PyArray1<f32>
    with pytest.raises((ValueError, TypeError)):
        quantizer.quantize_numpy(weights, "test_layer", bit_width=4)


def test_quantize_numpy_with_nan():
    """Test that NaN values raise ValueError."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(100).astype(np.float32)
    weights[50] = np.nan  # Insert NaN
    
    with pytest.raises(ValueError, match="NaN or Inf"):
        quantizer.quantize_numpy(weights, "test_layer", bit_width=4)


def test_quantize_numpy_with_inf():
    """Test that Inf values raise ValueError."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(100).astype(np.float32)
    weights[50] = np.inf  # Insert Inf
    
    with pytest.raises(ValueError, match="NaN or Inf"):
        quantizer.quantize_numpy(weights, "test_layer", bit_width=4)


def test_quantize_numpy_large_array():
    """Test quantization with large array (1M elements)."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(1000000).astype(np.float32)
    
    result = quantizer.quantize_numpy(weights, "large_layer", bit_width=4)
    
    assert result["quantized_data"].shape[0] == 1000000
    assert result["shape"] == (1000000,)


def test_quantize_numpy_zero_copy():
    """Test that quantization doesn't modify the original array."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(1000).astype(np.float32)
    weights_copy = weights.copy()
    
    # Quantize
    result = quantizer.quantize_numpy(weights, "test_layer", bit_width=4)
    
    # Verify original array is unchanged
    np.testing.assert_array_equal(weights, weights_copy)


def test_quantize_numpy_default_bit_width():
    """Test that default bit width is 4."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(100).astype(np.float32)
    
    # Don't specify bit_width
    result = quantizer.quantize_numpy(weights, "test_layer")
    
    assert result["bit_width"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# Tests for quantize_numpy_2d()

def test_quantize_numpy_2d_basic():
    """Test basic quantization with 2D numpy array."""
    from arrow_quant_v2 import ArrowQuantV2
    
    # Create quantizer
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Create test data (1024 x 768 weight matrix)
    weights = np.random.randn(1024, 768).astype(np.float32)
    
    # Quantize
    result = quantizer.quantize_numpy_2d(weights, "test_layer_2d", bit_width=4)
    
    # Verify result structure
    assert "quantized_data" in result
    assert "scales" in result
    assert "zero_points" in result
    assert "shape" in result
    assert "bit_width" in result
    assert "layer_name" in result
    
    # Verify types
    assert isinstance(result["quantized_data"], np.ndarray)
    assert isinstance(result["scales"], np.ndarray)
    assert isinstance(result["zero_points"], np.ndarray)
    assert result["quantized_data"].dtype == np.uint8
    assert result["scales"].dtype == np.float32
    assert result["zero_points"].dtype == np.float32
    
    # Verify shapes
    assert result["quantized_data"].shape[0] == 1024 * 768
    assert result["shape"] == (1024, 768)
    assert result["bit_width"] == 4
    assert result["layer_name"] == "test_layer_2d"


def test_quantize_numpy_2d_different_bit_widths():
    """Test 2D quantization with different bit widths."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(128, 256).astype(np.float32)
    
    for bit_width in [2, 4, 8]:
        result = quantizer.quantize_numpy_2d(weights, f"layer_2d_{bit_width}bit", bit_width=bit_width)
        assert result["bit_width"] == bit_width
        assert result["quantized_data"].shape[0] == 128 * 256
        assert result["shape"] == (128, 256)


def test_quantize_numpy_2d_invalid_bit_width():
    """Test that invalid bit width raises ValueError for 2D arrays."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(10, 20).astype(np.float32)
    
    with pytest.raises(ValueError, match="Invalid bit_width"):
        quantizer.quantize_numpy_2d(weights, "test_layer", bit_width=16)


def test_quantize_numpy_2d_wrong_dtype():
    """Test that wrong dtype raises error for 2D arrays."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(10, 20).astype(np.float64)  # Wrong dtype
    
    # This should fail because numpy 0.22 will not accept float64 as PyArray2<f32>
    with pytest.raises((ValueError, TypeError)):
        quantizer.quantize_numpy_2d(weights, "test_layer", bit_width=4)


def test_quantize_numpy_2d_with_nan():
    """Test that NaN values raise ValueError in 2D arrays."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(10, 20).astype(np.float32)
    weights[5, 10] = np.nan  # Insert NaN
    
    with pytest.raises(ValueError, match="NaN or Inf"):
        quantizer.quantize_numpy_2d(weights, "test_layer", bit_width=4)


def test_quantize_numpy_2d_with_inf():
    """Test that Inf values raise ValueError in 2D arrays."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(10, 20).astype(np.float32)
    weights[5, 10] = np.inf  # Insert Inf
    
    with pytest.raises(ValueError, match="NaN or Inf"):
        quantizer.quantize_numpy_2d(weights, "test_layer", bit_width=4)


def test_quantize_numpy_2d_non_contiguous():
    """Test that non-contiguous arrays raise ValueError."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(20, 30).astype(np.float32)
    
    # Create non-contiguous array by slicing
    non_contiguous = weights[::2, ::2]
    
    with pytest.raises(ValueError, match="contiguous"):
        quantizer.quantize_numpy_2d(non_contiguous, "test_layer", bit_width=4)


def test_quantize_numpy_2d_fortran_order():
    """Test that Fortran-order arrays work but emit warning."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(10, 20).astype(np.float32)
    
    # Convert to Fortran order
    weights_f = np.asfortranarray(weights)
    
    # Should work but may emit warning (captured by stderr)
    result = quantizer.quantize_numpy_2d(weights_f, "test_layer", bit_width=4)
    
    assert result["shape"] == (10, 20)
    assert result["quantized_data"].shape[0] == 10 * 20


def test_quantize_numpy_2d_wrong_dimensions():
    """Test that 1D or 3D arrays raise ValueError."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Test with 1D array
    weights_1d = np.random.randn(100).astype(np.float32)
    with pytest.raises((ValueError, TypeError)):
        quantizer.quantize_numpy_2d(weights_1d, "test_layer", bit_width=4)
    
    # Test with 3D array
    weights_3d = np.random.randn(10, 20, 30).astype(np.float32)
    with pytest.raises((ValueError, TypeError)):
        quantizer.quantize_numpy_2d(weights_3d, "test_layer", bit_width=4)


def test_quantize_numpy_2d_large_matrix():
    """Test quantization with large 2D matrix."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    # 4096 x 4096 = 16M elements = 64MB
    weights = np.random.randn(4096, 4096).astype(np.float32)
    
    result = quantizer.quantize_numpy_2d(weights, "large_layer_2d", bit_width=4)
    
    assert result["quantized_data"].shape[0] == 4096 * 4096
    assert result["shape"] == (4096, 4096)


def test_quantize_numpy_2d_zero_copy():
    """Test that 2D quantization doesn't modify the original array."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(100, 200).astype(np.float32)
    weights_copy = weights.copy()
    
    # Quantize
    result = quantizer.quantize_numpy_2d(weights, "test_layer", bit_width=4)
    
    # Verify original array is unchanged
    np.testing.assert_array_equal(weights, weights_copy)


def test_quantize_numpy_2d_default_bit_width():
    """Test that default bit width is 4 for 2D arrays."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    weights = np.random.randn(50, 100).astype(np.float32)
    
    # Don't specify bit_width
    result = quantizer.quantize_numpy_2d(weights, "test_layer")
    
    assert result["bit_width"] == 4


def test_quantize_numpy_2d_various_shapes():
    """Test 2D quantization with various matrix shapes."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2(mode="diffusion")
    
    shapes = [
        (1, 100),      # Row vector
        (100, 1),      # Column vector
        (64, 64),      # Square matrix
        (768, 3072),   # Typical transformer dimensions
        (4096, 1024),  # Large rectangular matrix
    ]
    
    for rows, cols in shapes:
        weights = np.random.randn(rows, cols).astype(np.float32)
        result = quantizer.quantize_numpy_2d(weights, f"layer_{rows}x{cols}", bit_width=4)
        
        assert result["shape"] == (rows, cols)
        assert result["quantized_data"].shape[0] == rows * cols

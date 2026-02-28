"""
Unit tests for quantize_batch() method.

Tests the batch API for reduced boundary crossings as specified in
pyo3-zero-copy-optimization spec Task 3.1.
"""

import numpy as np
import pytest


class TestQuantizeBatch:
    """Test suite for quantize_batch() method."""

    def test_quantize_batch_basic(self):
        """Test basic batch quantization with multiple layers."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        # Create test data - 3 layers with different sizes
        weights = {
            "layer.0.weight": np.random.randn(100, 100).astype(np.float32),
            "layer.1.weight": np.random.randn(200, 50).astype(np.float32),
            "layer.2.weight": np.random.randn(50, 200).astype(np.float32),
        }

        # Quantize batch
        results = quantizer.quantize_batch(weights, bit_width=4)

        # Verify results
        assert len(results) == 3
        assert "layer.0.weight" in results
        assert "layer.1.weight" in results
        assert "layer.2.weight" in results

        # Verify each result has required fields
        for layer_name, result in results.items():
            assert "quantized_data" in result
            assert "scales" in result
            assert "zero_points" in result
            assert "shape" in result
            assert "bit_width" in result
            assert result["bit_width"] == 4

    def test_quantize_batch_empty_dict(self):
        """Test batch quantization with empty dictionary."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        # Empty dictionary should return empty results
        results = quantizer.quantize_batch({}, bit_width=4)
        assert len(results) == 0

    def test_quantize_batch_single_layer(self):
        """Test batch quantization with single layer."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        weights = {
            "layer.0.weight": np.random.randn(1000).astype(np.float32),
        }

        results = quantizer.quantize_batch(weights, bit_width=4)

        assert len(results) == 1
        assert "layer.0.weight" in results
        result = results["layer.0.weight"]
        assert len(result["quantized_data"]) > 0
        assert len(result["scales"]) > 0
        assert len(result["zero_points"]) > 0

    def test_quantize_batch_different_bit_widths(self):
        """Test batch quantization with different bit widths."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        weights = {
            "layer.0.weight": np.random.randn(100, 100).astype(np.float32),
        }

        # Test bit_width=2
        results_2bit = quantizer.quantize_batch(weights, bit_width=2)
        assert results_2bit["layer.0.weight"]["bit_width"] == 2

        # Test bit_width=4
        results_4bit = quantizer.quantize_batch(weights, bit_width=4)
        assert results_4bit["layer.0.weight"]["bit_width"] == 4

        # Test bit_width=8
        results_8bit = quantizer.quantize_batch(weights, bit_width=8)
        assert results_8bit["layer.0.weight"]["bit_width"] == 8

    def test_quantize_batch_invalid_bit_width(self):
        """Test batch quantization with invalid bit width."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        weights = {
            "layer.0.weight": np.random.randn(100).astype(np.float32),
        }

        # Invalid bit width should raise ValueError
        with pytest.raises(ValueError, match="Invalid bit_width"):
            quantizer.quantize_batch(weights, bit_width=3)

        with pytest.raises(ValueError, match="Invalid bit_width"):
            quantizer.quantize_batch(weights, bit_width=16)

    def test_quantize_batch_non_contiguous_array(self):
        """Test batch quantization with non-contiguous array."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        # Create non-contiguous array
        arr = np.random.randn(100, 100).astype(np.float32)
        non_contiguous = arr[::2, ::2]  # Strided view

        weights = {
            "layer.0.weight": non_contiguous,
        }

        # Should raise ValueError with helpful message
        with pytest.raises(ValueError, match="not contiguous"):
            quantizer.quantize_batch(weights, bit_width=4)

    def test_quantize_batch_wrong_dtype(self):
        """Test batch quantization with wrong dtype."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        # Create array with wrong dtype
        weights = {
            "layer.0.weight": np.random.randn(100).astype(np.float64),  # float64 instead of float32
        }

        # Should raise ValueError with helpful message
        with pytest.raises(ValueError, match="expected 'float32'"):
            quantizer.quantize_batch(weights, bit_width=4)

    def test_quantize_batch_with_nan(self):
        """Test batch quantization with NaN values."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        # Create array with NaN
        arr = np.random.randn(100).astype(np.float32)
        arr[50] = np.nan

        weights = {
            "layer.0.weight": arr,
        }

        # Should raise ValueError with helpful message
        with pytest.raises(ValueError, match="NaN or Inf"):
            quantizer.quantize_batch(weights, bit_width=4)

    def test_quantize_batch_with_inf(self):
        """Test batch quantization with Inf values."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        # Create array with Inf
        arr = np.random.randn(100).astype(np.float32)
        arr[50] = np.inf

        weights = {
            "layer.0.weight": arr,
        }

        # Should raise ValueError with helpful message
        with pytest.raises(ValueError, match="NaN or Inf"):
            quantizer.quantize_batch(weights, bit_width=4)

    def test_quantize_batch_not_numpy_array(self):
        """Test batch quantization with non-numpy array."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        # Pass Python list instead of numpy array
        weights = {
            "layer.0.weight": [1.0, 2.0, 3.0],  # Python list
        }

        # Should raise ValueError with helpful message
        with pytest.raises(ValueError, match="Expected numpy array"):
            quantizer.quantize_batch(weights, bit_width=4)

    def test_quantize_batch_large_batch(self):
        """Test batch quantization with many layers (100 layers)."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        # Create 100 layers
        weights = {
            f"layer.{i}.weight": np.random.randn(100, 100).astype(np.float32)
            for i in range(100)
        }

        # Quantize batch
        results = quantizer.quantize_batch(weights, bit_width=4)

        # Verify all layers processed
        assert len(results) == 100
        for i in range(100):
            assert f"layer.{i}.weight" in results

    def test_quantize_batch_1d_arrays(self):
        """Test batch quantization with 1D arrays."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        weights = {
            "bias.0": np.random.randn(100).astype(np.float32),
            "bias.1": np.random.randn(200).astype(np.float32),
        }

        results = quantizer.quantize_batch(weights, bit_width=4)

        assert len(results) == 2
        # 1D arrays should have shape [n]
        assert results["bias.0"]["shape"] == [100]
        assert results["bias.1"]["shape"] == [200]

    def test_quantize_batch_2d_arrays(self):
        """Test batch quantization with 2D arrays."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        weights = {
            "weight.0": np.random.randn(100, 50).astype(np.float32),
            "weight.1": np.random.randn(200, 100).astype(np.float32),
        }

        results = quantizer.quantize_batch(weights, bit_width=4)

        assert len(results) == 2
        # 2D arrays should preserve shape
        assert results["weight.0"]["shape"] == [100, 50]
        assert results["weight.1"]["shape"] == [200, 100]

    def test_quantize_batch_mixed_shapes(self):
        """Test batch quantization with mixed 1D and 2D arrays."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        weights = {
            "weight": np.random.randn(100, 50).astype(np.float32),  # 2D
            "bias": np.random.randn(50).astype(np.float32),  # 1D
        }

        results = quantizer.quantize_batch(weights, bit_width=4)

        assert len(results) == 2
        assert results["weight"]["shape"] == [100, 50]
        assert results["bias"]["shape"] == [50]

    def test_quantize_batch_default_bit_width(self):
        """Test batch quantization with default bit width."""
        from arrow_quant_v2 import ArrowQuantV2

        quantizer = ArrowQuantV2()

        weights = {
            "layer.0.weight": np.random.randn(100).astype(np.float32),
        }

        # Don't specify bit_width, should default to 4
        results = quantizer.quantize_batch(weights)

        assert results["layer.0.weight"]["bit_width"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

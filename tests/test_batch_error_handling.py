"""
Test error handling for batch processing with partial success mode.

This test suite validates Requirements 2.3 and 6.4:
- Layer-specific error messages
- Partial success mode (continue on error)
- Error collection from failed layers
"""

import numpy as np
import pytest
from arrow_quant_v2 import ArrowQuantV2


class TestBatchErrorHandling:
    """Test error handling in batch quantization."""

    def test_fail_fast_mode_default(self):
        """Test that fail-fast is the default behavior."""
        quantizer = ArrowQuantV2()
        
        # Create batch with one invalid layer (contains NaN)
        weights = {
            "layer.0.weight": np.random.randn(1000).astype(np.float32),
            "layer.1.weight": np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32),
            "layer.2.weight": np.random.randn(1000).astype(np.float32),
        }
        
        # Should fail immediately with error identifying the problematic layer
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch(weights, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "layer.1.weight" in error_msg
        assert "NaN" in error_msg or "Inf" in error_msg

    def test_fail_fast_mode_explicit(self):
        """Test explicit fail-fast mode (continue_on_error=False)."""
        quantizer = ArrowQuantV2()
        
        # Create batch with invalid layer
        weights = {
            "layer.0.weight": np.random.randn(1000).astype(np.float32),
            "layer.1.weight": np.array([1.0, np.inf, 3.0], dtype=np.float32),
        }
        
        # Should fail with continue_on_error=False
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch(weights, bit_width=4, continue_on_error=False)
        
        error_msg = str(exc_info.value)
        assert "layer.1.weight" in error_msg

    def test_partial_success_mode(self):
        """Test partial success mode (continue_on_error=True)."""
        quantizer = ArrowQuantV2()
        
        # Create batch with one invalid layer
        weights = {
            "layer.0.weight": np.random.randn(1000).astype(np.float32),
            "layer.1.weight": np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32),
            "layer.2.weight": np.random.randn(1000).astype(np.float32),
        }
        
        # Should succeed with partial results
        results = quantizer.quantize_batch(weights, bit_width=4, continue_on_error=True)
        
        # Should have results for valid layers only
        assert "layer.0.weight" in results
        assert "layer.1.weight" not in results  # Failed layer excluded
        assert "layer.2.weight" in results
        
        # Verify valid results
        assert "quantized_data" in results["layer.0.weight"]
        assert "scales" in results["layer.0.weight"]
        assert "quantized_data" in results["layer.2.weight"]

    def test_partial_success_multiple_failures(self):
        """Test partial success with multiple failing layers."""
        quantizer = ArrowQuantV2()
        
        # Create batch with multiple invalid layers
        weights = {
            "layer.0.weight": np.random.randn(1000).astype(np.float32),
            "layer.1.weight": np.array([np.nan], dtype=np.float32),
            "layer.2.weight": np.random.randn(1000).astype(np.float32),
            "layer.3.weight": np.array([np.inf], dtype=np.float32),
            "layer.4.weight": np.random.randn(1000).astype(np.float32),
        }
        
        # Should succeed with partial results
        results = quantizer.quantize_batch(weights, bit_width=4, continue_on_error=True)
        
        # Should have results for valid layers only
        assert "layer.0.weight" in results
        assert "layer.1.weight" not in results  # Failed
        assert "layer.2.weight" in results
        assert "layer.3.weight" not in results  # Failed
        assert "layer.4.weight" in results
        
        # Should have 3 successful results
        assert len(results) == 3

    def test_partial_success_all_fail(self):
        """Test partial success when all layers fail."""
        quantizer = ArrowQuantV2()
        
        # Create batch where all layers are invalid
        weights = {
            "layer.0.weight": np.array([np.nan], dtype=np.float32),
            "layer.1.weight": np.array([np.inf], dtype=np.float32),
        }
        
        # Should return empty results
        results = quantizer.quantize_batch(weights, bit_width=4, continue_on_error=True)
        
        assert len(results) == 0

    def test_error_message_includes_layer_name(self):
        """Test that error messages identify the specific layer."""
        quantizer = ArrowQuantV2()
        
        # Create batch with invalid layer
        weights = {
            "my_special_layer": np.array([1.0, np.nan], dtype=np.float32),
        }
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch(weights, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "my_special_layer" in error_msg

    def test_non_contiguous_array_error(self):
        """Test error handling for non-contiguous arrays."""
        quantizer = ArrowQuantV2()
        
        # Create non-contiguous array
        arr = np.random.randn(100, 100).astype(np.float32)
        non_contiguous = arr[::2, ::2]  # Strided view
        
        weights = {
            "layer.0.weight": non_contiguous,
        }
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch(weights, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "layer.0.weight" in error_msg
        assert "contiguous" in error_msg.lower()

    def test_partial_success_with_non_contiguous(self):
        """Test partial success with non-contiguous array."""
        quantizer = ArrowQuantV2()
        
        # Create batch with non-contiguous array
        arr = np.random.randn(100, 100).astype(np.float32)
        non_contiguous = arr[::2, ::2]
        
        weights = {
            "layer.0.weight": np.random.randn(1000).astype(np.float32),
            "layer.1.weight": non_contiguous,
            "layer.2.weight": np.random.randn(1000).astype(np.float32),
        }
        
        # Should succeed with partial results
        results = quantizer.quantize_batch(weights, bit_width=4, continue_on_error=True)
        
        assert "layer.0.weight" in results
        assert "layer.1.weight" not in results  # Failed
        assert "layer.2.weight" in results

    def test_progress_callback_with_partial_success(self):
        """Test progress callback works with partial success mode."""
        quantizer = ArrowQuantV2()
        
        progress_calls = []
        
        def progress_fn(layer_name, progress):
            progress_calls.append((layer_name, progress))
        
        # Create batch with one invalid layer
        weights = {
            "layer.0.weight": np.random.randn(1000).astype(np.float32),
            "layer.1.weight": np.array([np.nan], dtype=np.float32),
            "layer.2.weight": np.random.randn(1000).astype(np.float32),
        }
        
        # Should succeed with partial results
        results = quantizer.quantize_batch_with_progress(
            weights,
            bit_width=4,
            progress_callback=progress_fn,
            continue_on_error=True
        )
        
        # Should have results for valid layers
        assert len(results) == 2
        
        # Should have progress callbacks for successful layers
        # Note: Failed layers may or may not trigger callbacks depending on when they fail
        assert len(progress_calls) >= 2

    def test_invalid_bit_width_fails_immediately(self):
        """Test that invalid bit_width fails before processing any layers."""
        quantizer = ArrowQuantV2()
        
        weights = {
            "layer.0.weight": np.random.randn(1000).astype(np.float32),
        }
        
        # Should fail immediately with invalid bit_width
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch(weights, bit_width=3, continue_on_error=True)
        
        error_msg = str(exc_info.value)
        assert "bit_width" in error_msg.lower()
        assert "3" in error_msg

    def test_empty_dict_returns_empty_results(self):
        """Test that empty dictionary returns empty results."""
        quantizer = ArrowQuantV2()
        
        results = quantizer.quantize_batch({}, bit_width=4, continue_on_error=True)
        
        assert len(results) == 0

    def test_partial_success_preserves_result_quality(self):
        """Test that partial success mode produces same quality results."""
        quantizer = ArrowQuantV2()
        
        # Create valid weights
        weights_valid = {
            "layer.0.weight": np.random.randn(1000).astype(np.float32),
        }
        
        # Quantize without errors
        results_normal = quantizer.quantize_batch(weights_valid, bit_width=4)
        
        # Quantize with partial success mode (but no errors)
        results_partial = quantizer.quantize_batch(weights_valid, bit_width=4, continue_on_error=True)
        
        # Results should be identical
        assert len(results_normal) == len(results_partial)
        assert "layer.0.weight" in results_normal
        assert "layer.0.weight" in results_partial
        
        # Quantized data should be the same
        assert results_normal["layer.0.weight"]["quantized_data"] == results_partial["layer.0.weight"]["quantized_data"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

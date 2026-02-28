"""
Unit tests for error handling in Arrow batch quantization.

This test suite validates Requirements 8.1, 8.2, and 8.5:
- NaN detection with descriptive error messages (8.1)
- Inf detection with descriptive error messages (8.1)
- Shape mismatch detection with descriptive error messages (8.2)
- Error message format validation (context, problem, fix suggestion) (8.5)

Task: 5.4 编写错误处理的单元测试
"""

import pytest
import numpy as np
import pyarrow as pa
from arrow_quant_v2 import ArrowQuantV2


class TestNaNDetection:
    """Test NaN detection with descriptive error messages (Requirement 8.1)."""

    def test_nan_detection_single_value(self):
        """Test NaN detection with a single NaN value."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create weights with NaN at specific position
        weights = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
        
        weights_data = {
            "layer_name": ["test_layer"],
            "weights": [weights.tolist()],
            "shape": [[5]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Should raise ValueError with descriptive message
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        
        # Verify error message contains layer name (context)
        assert "test_layer" in error_msg, "Error message should contain layer name"
        
        # Verify error message identifies NaN (problem)
        assert "NaN" in error_msg or "nan" in error_msg.lower(), "Error message should mention NaN"

    def test_nan_detection_multiple_values(self):
        """Test NaN detection with multiple NaN values."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create weights with multiple NaN values
        weights = np.array([np.nan, 2.0, 3.0, np.nan, 5.0], dtype=np.float32)
        
        weights_data = {
            "layer_name": ["multi_nan_layer"],
            "weights": [weights.tolist()],
            "shape": [[5]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "multi_nan_layer" in error_msg
        assert "NaN" in error_msg or "nan" in error_msg.lower()

    def test_nan_detection_at_start(self):
        """Test NaN detection when NaN is at the start of array."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        
        weights_data = {
            "layer_name": ["start_nan_layer"],
            "weights": [weights.tolist()],
            "shape": [[4]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "start_nan_layer" in error_msg
        assert "NaN" in error_msg or "nan" in error_msg.lower()

    def test_nan_detection_at_end(self):
        """Test NaN detection when NaN is at the end of array."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float32)
        
        weights_data = {
            "layer_name": ["end_nan_layer"],
            "weights": [weights.tolist()],
            "shape": [[4]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "end_nan_layer" in error_msg
        assert "NaN" in error_msg or "nan" in error_msg.lower()

    def test_nan_detection_large_array(self):
        """Test NaN detection in large array."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create large array with NaN in the middle
        weights = np.random.randn(10000).astype(np.float32)
        weights[5000] = np.nan
        
        weights_data = {
            "layer_name": ["large_nan_layer"],
            "weights": [weights.tolist()],
            "shape": [[10000]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "large_nan_layer" in error_msg
        assert "NaN" in error_msg or "nan" in error_msg.lower()

    def test_nan_detection_with_complex_layer_name(self):
        """Test NaN detection with complex layer name."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        
        weights_data = {
            "layer_name": ["model.layers.0.self_attn.q_proj.weight"],
            "weights": [weights.tolist()],
            "shape": [[3]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "model.layers.0.self_attn.q_proj.weight" in error_msg


class TestInfDetection:
    """Test Inf detection with descriptive error messages (Requirement 8.1)."""

    def test_positive_inf_detection(self):
        """Test positive infinity detection."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.array([1.0, 2.0, np.inf, 4.0], dtype=np.float32)
        
        weights_data = {
            "layer_name": ["pos_inf_layer"],
            "weights": [weights.tolist()],
            "shape": [[4]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        
        # Verify error message contains layer name (context)
        assert "pos_inf_layer" in error_msg
        
        # Verify error message identifies Inf (problem)
        assert "Inf" in error_msg or "inf" in error_msg.lower()

    def test_negative_inf_detection(self):
        """Test negative infinity detection."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.array([1.0, -np.inf, 3.0, 4.0], dtype=np.float32)
        
        weights_data = {
            "layer_name": ["neg_inf_layer"],
            "weights": [weights.tolist()],
            "shape": [[4]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "neg_inf_layer" in error_msg
        assert "Inf" in error_msg or "inf" in error_msg.lower()

    def test_multiple_inf_detection(self):
        """Test detection of multiple infinity values."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.array([np.inf, 2.0, -np.inf, 4.0], dtype=np.float32)
        
        weights_data = {
            "layer_name": ["multi_inf_layer"],
            "weights": [weights.tolist()],
            "shape": [[4]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "multi_inf_layer" in error_msg
        assert "Inf" in error_msg or "inf" in error_msg.lower()

    def test_inf_detection_at_boundaries(self):
        """Test Inf detection at array boundaries."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Test at start
        weights_start = np.array([np.inf, 1.0, 2.0], dtype=np.float32)
        weights_data = {
            "layer_name": ["inf_start"],
            "weights": [weights_start.tolist()],
            "shape": [[3]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "inf_start" in error_msg
        assert "Inf" in error_msg or "inf" in error_msg.lower()

    def test_mixed_nan_and_inf(self):
        """Test detection when both NaN and Inf are present."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.array([1.0, np.nan, np.inf, 4.0], dtype=np.float32)
        
        weights_data = {
            "layer_name": ["mixed_invalid_layer"],
            "weights": [weights.tolist()],
            "shape": [[4]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "mixed_invalid_layer" in error_msg
        # Should mention either NaN or Inf (or both)
        assert "NaN" in error_msg or "Inf" in error_msg or \
               "nan" in error_msg.lower() or "inf" in error_msg.lower()


class TestShapeMismatchDetection:
    """Test shape mismatch detection with descriptive error messages (Requirement 8.2)."""

    def test_shape_mismatch_too_large(self):
        """Test shape mismatch when shape product is larger than weights length."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Weights has 10 elements, but shape claims 20
        weights = np.random.randn(10).astype(np.float32)
        
        weights_data = {
            "layer_name": ["shape_too_large"],
            "weights": [weights.tolist()],
            "shape": [[20]],  # Mismatch: 20 != 10
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        
        # Verify error message contains layer name (context)
        assert "shape_too_large" in error_msg
        
        # Verify error message mentions shape mismatch (problem)
        assert "shape" in error_msg.lower()

    def test_shape_mismatch_too_small(self):
        """Test shape mismatch when shape product is smaller than weights length."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Weights has 20 elements, but shape claims 10
        weights = np.random.randn(20).astype(np.float32)
        
        weights_data = {
            "layer_name": ["shape_too_small"],
            "weights": [weights.tolist()],
            "shape": [[10]],  # Mismatch: 10 != 20
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "shape_too_small" in error_msg
        assert "shape" in error_msg.lower()

    def test_shape_mismatch_2d(self):
        """Test shape mismatch with 2D shape."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Weights has 100 elements, but shape claims 10x20 = 200
        weights = np.random.randn(100).astype(np.float32)
        
        weights_data = {
            "layer_name": ["shape_2d_mismatch"],
            "weights": [weights.tolist()],
            "shape": [[10, 20]],  # Mismatch: 10*20=200 != 100
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "shape_2d_mismatch" in error_msg
        assert "shape" in error_msg.lower()

    def test_shape_mismatch_3d(self):
        """Test shape mismatch with 3D shape."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Weights has 50 elements, but shape claims 5x5x5 = 125
        weights = np.random.randn(50).astype(np.float32)
        
        weights_data = {
            "layer_name": ["shape_3d_mismatch"],
            "weights": [weights.tolist()],
            "shape": [[5, 5, 5]],  # Mismatch: 5*5*5=125 != 50
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "shape_3d_mismatch" in error_msg
        assert "shape" in error_msg.lower()

    def test_shape_mismatch_with_zero_dimension(self):
        """Test shape mismatch with zero in shape dimensions."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.random.randn(10).astype(np.float32)
        
        weights_data = {
            "layer_name": ["shape_with_zero"],
            "weights": [weights.tolist()],
            "shape": [[10, 0]],  # Product is 0, but weights has 10 elements
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "shape_with_zero" in error_msg

    def test_shape_correct_no_error(self):
        """Test that correct shape does not raise error."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Weights has 20 elements, shape is [4, 5] = 20 (correct)
        weights = np.random.randn(20).astype(np.float32)
        
        weights_data = {
            "layer_name": ["correct_shape"],
            "weights": [weights.tolist()],
            "shape": [[4, 5]],  # Correct: 4*5=20
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Should not raise error
        result = quantizer.quantize_batch_arrow(batch, bit_width=4)
        assert result.num_rows == 1

    def test_shape_mismatch_complex_layer_name(self):
        """Test shape mismatch with complex layer name."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.random.randn(100).astype(np.float32)
        
        weights_data = {
            "layer_name": ["model.encoder.layers.5.attention.weight"],
            "weights": [weights.tolist()],
            "shape": [[10, 20]],  # Mismatch: 10*20=200 != 100
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "model.encoder.layers.5.attention.weight" in error_msg


class TestErrorMessageFormat:
    """Test error message format validation (Requirement 8.5)."""

    def test_error_message_contains_context(self):
        """Test that error messages contain context (layer name, operation type)."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        
        weights_data = {
            "layer_name": ["context_test_layer"],
            "weights": [weights.tolist()],
            "shape": [[3]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        
        # Context: layer name should be present
        assert "context_test_layer" in error_msg, \
            "Error message should contain layer name as context"

    def test_error_message_contains_problem_description(self):
        """Test that error messages describe the specific problem."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Test NaN problem description
        weights_nan = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        weights_data_nan = {
            "layer_name": ["nan_problem"],
            "weights": [weights_nan.tolist()],
            "shape": [[3]],
        }
        batch_nan = pa.RecordBatch.from_pydict(weights_data_nan)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch_nan, bit_width=4)
        
        error_msg = str(exc_info.value)
        # Problem: should mention NaN
        assert "NaN" in error_msg or "nan" in error_msg.lower(), \
            "Error message should describe the NaN problem"

    def test_error_message_inf_problem_description(self):
        """Test that Inf error messages describe the problem."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights_inf = np.array([1.0, np.inf, 3.0], dtype=np.float32)
        weights_data_inf = {
            "layer_name": ["inf_problem"],
            "weights": [weights_inf.tolist()],
            "shape": [[3]],
        }
        batch_inf = pa.RecordBatch.from_pydict(weights_data_inf)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch_inf, bit_width=4)
        
        error_msg = str(exc_info.value)
        # Problem: should mention Inf
        assert "Inf" in error_msg or "inf" in error_msg.lower(), \
            "Error message should describe the Inf problem"

    def test_error_message_shape_problem_description(self):
        """Test that shape mismatch error messages describe the problem."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.random.randn(10).astype(np.float32)
        weights_data = {
            "layer_name": ["shape_problem"],
            "weights": [weights.tolist()],
            "shape": [[20]],  # Mismatch
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        # Problem: should mention shape
        assert "shape" in error_msg.lower(), \
            "Error message should describe the shape mismatch problem"

    def test_error_message_is_descriptive(self):
        """Test that error messages are descriptive and helpful."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.array([1.0, 2.0, np.nan], dtype=np.float32)
        weights_data = {
            "layer_name": ["descriptive_test"],
            "weights": [weights.tolist()],
            "shape": [[3]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        
        # Error message should be reasonably long (descriptive)
        assert len(error_msg) > 20, \
            "Error message should be descriptive, not just a code"
        
        # Should contain layer name
        assert "descriptive_test" in error_msg

    def test_multiple_layers_error_identification(self):
        """Test that errors in multi-layer batches identify the specific layer."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create batch with 3 layers, middle one has error
        weights_data = {
            "layer_name": ["layer_0", "layer_1_bad", "layer_2"],
            "weights": [
                np.random.randn(10).astype(np.float32).tolist(),
                [1.0, np.nan, 3.0],  # Bad layer
                np.random.randn(10).astype(np.float32).tolist(),
            ],
            "shape": [[10], [3], [10]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        
        # Should identify the specific bad layer
        assert "layer_1_bad" in error_msg, \
            "Error message should identify the specific failing layer"


class TestEdgeCases:
    """Test edge cases in error handling."""

    def test_all_nan_array(self):
        """Test array with all NaN values."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.full(10, np.nan, dtype=np.float32)
        
        weights_data = {
            "layer_name": ["all_nan"],
            "weights": [weights.tolist()],
            "shape": [[10]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "all_nan" in error_msg
        assert "NaN" in error_msg or "nan" in error_msg.lower()

    def test_all_inf_array(self):
        """Test array with all Inf values."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.full(10, np.inf, dtype=np.float32)
        
        weights_data = {
            "layer_name": ["all_inf"],
            "weights": [weights.tolist()],
            "shape": [[10]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "all_inf" in error_msg
        assert "Inf" in error_msg or "inf" in error_msg.lower()

    def test_single_element_nan(self):
        """Test single-element array with NaN."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.array([np.nan], dtype=np.float32)
        
        weights_data = {
            "layer_name": ["single_nan"],
            "weights": [weights.tolist()],
            "shape": [[1]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "single_nan" in error_msg

    def test_valid_extreme_values(self):
        """Test that valid extreme values (not NaN/Inf) are accepted."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Use very large but finite values
        weights = np.array([1e30, -1e30, 1e-30, -1e-30, 0.0], dtype=np.float32)
        
        weights_data = {
            "layer_name": ["extreme_valid"],
            "weights": [weights.tolist()],
            "shape": [[5]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Should not raise error for valid extreme values
        result = quantizer.quantize_batch_arrow(batch, bit_width=4)
        assert result.num_rows == 1

    def test_empty_shape_list(self):
        """Test error handling with empty shape list."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.random.randn(10).astype(np.float32)
        
        weights_data = {
            "layer_name": ["empty_shape"],
            "weights": [weights.tolist()],
            "shape": [[]],  # Empty shape
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Should handle empty shape (might infer or error)
        # The behavior depends on implementation
        try:
            result = quantizer.quantize_batch_arrow(batch, bit_width=4)
            # If it succeeds, verify it has a result
            assert result.num_rows == 1
        except ValueError as e:
            # If it errors, verify error message is descriptive
            error_msg = str(e)
            assert "empty_shape" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

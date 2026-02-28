"""
Test Arrow schema validation for quantize_arrow() and quantize_arrow_batch().

This test suite verifies that schema validation correctly identifies:
- Missing required columns (layer_name, weights)
- Incorrect column types
- Invalid data types in list fields
- Proper handling of optional columns (shape)

Validates Requirements: 3.4, 6.3
"""

import numpy as np
import pyarrow as pa
import pytest

# Import will be available after maturin develop
try:
    from arrow_quant_v2 import ArrowQuantV2
    ARROW_QUANT_AVAILABLE = True
except ImportError:
    ARROW_QUANT_AVAILABLE = False


@pytest.mark.skipif(not ARROW_QUANT_AVAILABLE, reason="arrow_quant_v2 not built")
class TestArrowSchemaValidation:
    """Test Arrow schema validation for quantization methods."""

    def test_valid_schema_with_all_fields(self):
        """Test that a valid schema with all fields passes validation."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with all fields (layer_name, weights, shape)
        weights = np.random.randn(100).astype(np.float32)
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [weights.tolist()],
            "shape": [[100]],
        })
        
        # Should not raise any errors
        result = quantizer.quantize_arrow(table, bit_width=4)
        assert result is not None
        assert result.num_rows == 1

    def test_valid_schema_without_optional_shape(self):
        """Test that schema without optional shape field is valid."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table without shape field
        weights = np.random.randn(100).astype(np.float32)
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [weights.tolist()],
        })
        
        # Should not raise any errors
        result = quantizer.quantize_arrow(table, bit_width=4)
        assert result is not None
        assert result.num_rows == 1

    def test_missing_layer_name_column(self):
        """Test that missing layer_name column raises descriptive error."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table without layer_name
        weights = np.random.randn(100).astype(np.float32)
        table = pa.Table.from_pydict({
            "weights": [weights.tolist()],
            "shape": [[100]],
        })
        
        # Should raise ValueError with clear message
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "layer_name" in error_msg.lower()
        assert "missing" in error_msg.lower() or "required" in error_msg.lower()
        # Should include expected schema
        assert "expected schema" in error_msg.lower()

    def test_missing_weights_column(self):
        """Test that missing weights column raises descriptive error."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table without weights
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "shape": [[100]],
        })
        
        # Should raise ValueError with clear message
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "weights" in error_msg.lower()
        assert "missing" in error_msg.lower() or "required" in error_msg.lower()
        # Should include expected schema
        assert "expected schema" in error_msg.lower()

    def test_wrong_layer_name_type_int(self):
        """Test that integer layer_name column raises type error."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with integer layer_name (should be string)
        weights = np.random.randn(100).astype(np.float32)
        table = pa.Table.from_pydict({
            "layer_name": [1, 2, 3],  # Wrong type: int instead of string
            "weights": [weights.tolist(), weights.tolist(), weights.tolist()],
        })
        
        # Should raise ValueError about type mismatch
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "layer_name" in error_msg.lower()
        assert "type" in error_msg.lower() or "string" in error_msg.lower()

    def test_wrong_weights_type_not_list(self):
        """Test that non-list weights column raises type error."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with scalar weights (should be list)
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [1.0],  # Wrong: scalar instead of list
        })
        
        # Should raise ValueError about type mismatch
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "weights" in error_msg.lower()
        assert "list" in error_msg.lower() or "type" in error_msg.lower()

    def test_wrong_weights_inner_type_int(self):
        """Test that list<int> weights raises type error (should be list<float32>)."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with list<int64> instead of list<float32>
        weights_int = [1, 2, 3, 4, 5]
        table = pa.table({
            "layer_name": pa.array(["test_layer"], type=pa.string()),
            "weights": pa.array([weights_int], type=pa.list_(pa.int64())),
        })
        
        # Should raise ValueError about inner type mismatch
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "weights" in error_msg.lower()
        assert "float32" in error_msg.lower()

    def test_wrong_weights_inner_type_float64(self):
        """Test that list<float64> weights raises type error (should be list<float32>)."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with list<float64> instead of list<float32>
        weights_f64 = np.random.randn(100).astype(np.float64)
        table = pa.table({
            "layer_name": pa.array(["test_layer"], type=pa.string()),
            "weights": pa.array([weights_f64.tolist()], type=pa.list_(pa.float64())),
        })
        
        # Should raise ValueError about inner type mismatch
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "weights" in error_msg.lower()
        assert "float32" in error_msg.lower()

    def test_wrong_shape_type_not_list(self):
        """Test that non-list shape column raises type error."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with scalar shape (should be list)
        weights = np.random.randn(100).astype(np.float32)
        table = pa.table({
            "layer_name": pa.array(["test_layer"], type=pa.string()),
            "weights": pa.array([weights.tolist()], type=pa.list_(pa.float32())),
            "shape": pa.array([100], type=pa.int64()),  # Wrong: scalar instead of list
        })
        
        # Should raise ValueError about type mismatch
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "shape" in error_msg.lower()
        assert "list" in error_msg.lower() or "type" in error_msg.lower()

    def test_wrong_shape_inner_type_float(self):
        """Test that list<float> shape raises type error (should be list<int64>)."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with list<float32> shape instead of list<int64>
        weights = np.random.randn(100).astype(np.float32)
        table = pa.table({
            "layer_name": pa.array(["test_layer"], type=pa.string()),
            "weights": pa.array([weights.tolist()], type=pa.list_(pa.float32())),
            "shape": pa.array([[100.0]], type=pa.list_(pa.float32())),  # Wrong type
        })
        
        # Should raise ValueError about inner type mismatch
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "shape" in error_msg.lower()
        assert "int64" in error_msg.lower()

    def test_schema_validation_with_extra_columns(self):
        """Test that extra columns are allowed (forward compatibility)."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with extra columns
        weights = np.random.randn(100).astype(np.float32)
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [weights.tolist()],
            "shape": [[100]],
            "extra_field": ["extra_value"],  # Extra column
            "another_field": [42],  # Another extra column
        })
        
        # Should not raise errors (extra columns are ignored)
        result = quantizer.quantize_arrow(table, bit_width=4)
        assert result is not None
        assert result.num_rows == 1

    def test_error_message_includes_expected_schema(self):
        """Test that error messages include expected schema format."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with missing required field
        table = pa.Table.from_pydict({
            "wrong_field": ["value"],
        })
        
        # Error should include expected schema
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        # Should mention expected schema structure
        assert "expected schema" in error_msg.lower()
        assert "layer_name" in error_msg.lower()
        assert "weights" in error_msg.lower()

    def test_error_message_shows_actual_type(self):
        """Test that error messages show the actual incorrect type."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with wrong type
        table = pa.table({
            "layer_name": pa.array([123], type=pa.int32()),  # Wrong type
            "weights": pa.array([[1.0, 2.0]], type=pa.list_(pa.float32())),
        })
        
        # Error should show actual type
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        # Should mention the actual type found
        assert "type" in error_msg.lower()

    def test_quantize_arrow_batch_schema_validation(self):
        """Test that quantize_arrow_batch() also validates schema."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create RecordBatch with missing required field
        weights = pa.array([[1.0, 2.0]], type=pa.list_(pa.float32()))
        batch = pa.RecordBatch.from_arrays(
            [weights],
            names=["weights"]  # Missing layer_name
        )
        
        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow_batch(batch, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "layer_name" in error_msg.lower()

    def test_large_utf8_string_type_accepted(self):
        """Test that LargeUtf8 string type is also accepted for layer_name."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with LargeUtf8 type
        weights = np.random.randn(100).astype(np.float32)
        table = pa.table({
            "layer_name": pa.array(["test_layer"], type=pa.large_string()),
            "weights": pa.array([weights.tolist()], type=pa.list_(pa.float32())),
        })
        
        # Should not raise errors (LargeUtf8 is valid)
        result = quantizer.quantize_arrow(table, bit_width=4)
        assert result is not None
        assert result.num_rows == 1

    def test_large_list_type_accepted(self):
        """Test that LargeList types are also accepted."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with LargeList types
        weights = np.random.randn(100).astype(np.float32)
        table = pa.table({
            "layer_name": pa.array(["test_layer"], type=pa.string()),
            "weights": pa.array([weights.tolist()], type=pa.large_list(pa.float32())),
            "shape": pa.array([[100]], type=pa.large_list(pa.int64())),
        })
        
        # Should not raise errors (LargeList is valid)
        result = quantizer.quantize_arrow(table, bit_width=4)
        assert result is not None
        assert result.num_rows == 1

    def test_nullability_not_enforced(self):
        """Test that nullability is not strictly enforced in schema validation."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with nullable fields
        weights = np.random.randn(100).astype(np.float32)
        schema = pa.schema([
            pa.field("layer_name", pa.string(), nullable=True),
            pa.field("weights", pa.list_(pa.float32()), nullable=True),
        ])
        
        table = pa.Table.from_pydict(
            {
                "layer_name": ["test_layer"],
                "weights": [weights.tolist()],
            },
            schema=schema
        )
        
        # Should work (nullability is not enforced at schema level)
        result = quantizer.quantize_arrow(table, bit_width=4)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

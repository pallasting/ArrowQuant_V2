"""
Unit tests for FFI helper functions (import_pyarrow_table, export_recordbatch_to_pyarrow, validate_quantization_schema).

This test suite provides comprehensive unit testing for the Arrow FFI helper functions,
covering both success and failure cases.

Requirements tested:
- 1.1: Arrow Table import through C Data Interface
- 1.4: Schema validation for missing columns
- 1.5: Schema validation for incorrect types
- 4.1: RecordBatch export with correct schema
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
class TestImportPyArrowTable:
    """Test import_pyarrow_table() function - Requirement 1.1"""

    def test_import_valid_table_single_layer(self):
        """Test importing a valid Arrow Table with a single layer."""
        # Create valid table
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0.weight"],
            "weights": [np.random.randn(100).astype(np.float32)],
            "shape": [[100]],
        })

        # Import through quantize_arrow_batch (which uses import_pyarrow_table internally)
        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Verify successful import
        assert result is not None
        assert result.num_rows == 1

    def test_import_valid_table_multiple_layers(self):
        """Test importing a valid Arrow Table with multiple layers."""
        # Create table with 5 layers
        layer_names = [f"layer.{i}.weight" for i in range(5)]
        weights_data = [np.random.randn(100).astype(np.float32) for _ in range(5)]
        shapes = [[100] for _ in range(5)]

        table = pa.Table.from_pydict({
            "layer_name": layer_names,
            "weights": weights_data,
            "shape": shapes,
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Verify all layers imported
        assert result.num_rows == 5

    def test_import_table_without_optional_shape(self):
        """Test importing a table without the optional shape column."""
        # Create table without shape column
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0.weight"],
            "weights": [np.random.randn(100).astype(np.float32)],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Should succeed and infer shape
        assert result is not None
        assert result.num_rows == 1

    def test_import_empty_table(self):
        """Test importing an empty Arrow Table."""
        # Create empty table with correct schema
        table = pa.Table.from_pydict({
            "layer_name": pa.array([], type=pa.string()),
            "weights": pa.array([], type=pa.list_(pa.float32())),
            "shape": pa.array([], type=pa.list_(pa.int64())),
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Should succeed with empty result
        assert result is not None
        assert result.num_rows == 0

    def test_import_table_with_large_weights(self):
        """Test importing a table with large weight arrays."""
        # Create table with 1M weights
        large_weights = np.random.randn(1_000_000).astype(np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["large_layer"],
            "weights": [large_weights],
            "shape": [[1_000_000]],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Should handle large arrays
        assert result is not None
        assert result.num_rows == 1

    def test_import_table_with_multidimensional_shapes(self):
        """Test importing a table with multi-dimensional tensor shapes."""
        # Create 2D and 3D tensors
        weights_2d = np.random.randn(10, 20).astype(np.float32)
        weights_3d = np.random.randn(5, 4, 3).astype(np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["layer_2d", "layer_3d"],
            "weights": [weights_2d.flatten(), weights_3d.flatten()],
            "shape": [[10, 20], [5, 4, 3]],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Verify shapes preserved
        result_list = result.to_pylist()
        assert result_list[0]["shape"] == [10, 20]
        assert result_list[1]["shape"] == [5, 4, 3]


@pytest.mark.skipif(not ARROW_QUANT_AVAILABLE, reason="arrow_quant_v2 not built")
class TestExportRecordBatchToPyArrow:
    """Test export_recordbatch_to_pyarrow() function - Requirement 4.1"""

    def test_export_creates_valid_pyarrow_table(self):
        """Test that export creates a valid PyArrow Table."""
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [np.random.randn(100).astype(np.float32)],
            "shape": [[100]],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Verify result is a PyArrow Table
        assert isinstance(result, pa.Table)

    def test_export_preserves_layer_names(self):
        """Test that export preserves layer names correctly."""
        layer_names = ["layer.0", "layer.1", "layer.2"]
        table = pa.Table.from_pydict({
            "layer_name": layer_names,
            "weights": [np.random.randn(100).astype(np.float32) for _ in range(3)],
            "shape": [[100] for _ in range(3)],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Verify layer names preserved
        result_layer_names = result.column("layer_name").to_pylist()
        assert set(result_layer_names) == set(layer_names)

    def test_export_includes_all_required_columns(self):
        """Test that export includes all required columns."""
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [np.random.randn(100).astype(np.float32)],
            "shape": [[100]],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Verify all required columns exist
        required_columns = [
            "layer_name",
            "quantized_data",
            "scales",
            "zero_points",
            "shape",
            "bit_width"
        ]
        for col in required_columns:
            assert col in result.column_names, f"Missing column: {col}"

    def test_export_correct_column_types(self):
        """Test that export creates columns with correct types."""
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [np.random.randn(100).astype(np.float32)],
            "shape": [[100]],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=8)

        schema = result.schema

        # Verify types
        assert schema.field("layer_name").type == pa.string()
        assert schema.field("quantized_data").type == pa.binary()
        
        scales_field = schema.field("scales")
        assert isinstance(scales_field.type, pa.ListType)
        assert scales_field.type.value_type == pa.float32()
        
        zp_field = schema.field("zero_points")
        assert isinstance(zp_field.type, pa.ListType)
        assert zp_field.type.value_type == pa.float32()
        
        shape_field = schema.field("shape")
        assert isinstance(shape_field.type, pa.ListType)
        assert shape_field.type.value_type == pa.int64()
        
        assert schema.field("bit_width").type == pa.uint8()

    def test_export_bit_width_values(self):
        """Test that export correctly sets bit_width values."""
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [np.random.randn(100).astype(np.float32)],
            "shape": [[100]],
        })

        quantizer = ArrowQuantV2(mode="diffusion")

        for bit_width in [2, 4, 8]:
            result = quantizer.quantize_arrow_batch(table, bit_width=bit_width)
            bit_widths = result.column("bit_width").to_pylist()
            assert all(bw == bit_width for bw in bit_widths)


@pytest.mark.skipif(not ARROW_QUANT_AVAILABLE, reason="arrow_quant_v2 not built")
class TestValidateQuantizationSchema:
    """Test validate_quantization_schema() function - Requirements 1.4, 1.5"""

    def test_validate_accepts_valid_schema(self):
        """Test that validation accepts a valid schema."""
        # Create valid table
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [np.random.randn(100).astype(np.float32)],
            "shape": [[100]],
        })

        # Should not raise error
        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)
        assert result is not None

    def test_validate_accepts_schema_without_optional_shape(self):
        """Test that validation accepts schema without optional shape column."""
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [np.random.randn(100).astype(np.float32)],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)
        assert result is not None

    def test_validate_rejects_missing_layer_name(self):
        """Test that validation rejects table missing layer_name column."""
        # Create table without layer_name
        table = pa.Table.from_pydict({
            "weights": [np.random.randn(100).astype(np.float32)],
            "shape": [[100]],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow_batch(table, bit_width=4)
        
        assert "layer_name" in str(exc_info.value).lower()

    def test_validate_rejects_missing_weights(self):
        """Test that validation rejects table missing weights column."""
        # Create table without weights
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "shape": [[100]],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow_batch(table, bit_width=4)
        
        assert "weights" in str(exc_info.value).lower()

    def test_validate_rejects_wrong_layer_name_type(self):
        """Test that validation rejects incorrect layer_name type."""
        # Create table with int32 layer_name instead of string
        table = pa.Table.from_pydict({
            "layer_name": pa.array([1, 2, 3], type=pa.int32()),
            "weights": [
                np.random.randn(100).astype(np.float32),
                np.random.randn(100).astype(np.float32),
                np.random.randn(100).astype(np.float32),
            ],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow_batch(table, bit_width=4)
        
        error_msg = str(exc_info.value).lower()
        assert "layer_name" in error_msg or "type" in error_msg

    def test_validate_rejects_wrong_weights_type(self):
        """Test that validation rejects incorrect weights type."""
        # Create table with int64 weights instead of float32
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": pa.array([[1, 2, 3]], type=pa.list_(pa.int64())),
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow_batch(table, bit_width=4)
        
        error_msg = str(exc_info.value).lower()
        assert "weights" in error_msg or "float32" in error_msg

    def test_validate_rejects_wrong_shape_type(self):
        """Test that validation rejects incorrect shape type."""
        # Create table with float32 shape instead of int64
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [np.random.randn(100).astype(np.float32)],
            "shape": pa.array([[1.0, 2.0]], type=pa.list_(pa.float32())),
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_arrow_batch(table, bit_width=4)
        
        error_msg = str(exc_info.value).lower()
        assert "shape" in error_msg or "int64" in error_msg

    def test_validate_accepts_large_utf8(self):
        """Test that validation accepts LargeUtf8 for layer_name."""
        # Create table with LargeUtf8
        table = pa.Table.from_pydict({
            "layer_name": pa.array(["test"], type=pa.large_string()),
            "weights": [np.random.randn(100).astype(np.float32)],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)
        assert result is not None

    def test_validate_accepts_large_list(self):
        """Test that validation accepts LargeList for weights and shape."""
        # Create table with LargeList
        weights_array = pa.array([np.random.randn(100).astype(np.float32).tolist()], 
                                 type=pa.large_list(pa.float32()))
        shape_array = pa.array([[100]], type=pa.large_list(pa.int64()))
        
        table = pa.Table.from_arrays(
            [pa.array(["test"]), weights_array, shape_array],
            names=["layer_name", "weights", "shape"]
        )

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)
        assert result is not None


@pytest.mark.skipif(not ARROW_QUANT_AVAILABLE, reason="arrow_quant_v2 not built")
class TestPyCapsuleDestructors:
    """Test PyCapsule destructor functionality - Requirement 4.1"""

    def test_pycapsule_memory_safety(self):
        """Test that PyCapsule destructors properly clean up memory."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create and destroy multiple results to test destructors
        for i in range(20):
            table = pa.Table.from_pydict({
                "layer_name": [f"layer.{i}"],
                "weights": [np.random.randn(1000).astype(np.float32)],
                "shape": [[1000]],
            })
            
            result = quantizer.quantize_arrow_batch(table, bit_width=4)
            
            # Access data to ensure it's valid
            assert result.num_rows == 1
            layer_name = result.column("layer_name").to_pylist()[0]
            assert layer_name == f"layer.{i}"
            
            # Result goes out of scope - destructors should clean up
            del result

        # If we reach here without segfault, memory management is working

    def test_pycapsule_with_large_data(self):
        """Test PyCapsule destructors with large data."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create large result
        large_weights = np.random.randn(10_000_000).astype(np.float32)
        table = pa.Table.from_pydict({
            "layer_name": ["large_layer"],
            "weights": [large_weights],
            "shape": [[10_000_000]],
        })
        
        result = quantizer.quantize_arrow_batch(table, bit_width=4)
        assert result is not None
        
        # Delete and ensure cleanup
        del result
        del large_weights

    def test_pycapsule_multiple_references(self):
        """Test PyCapsule destructors with multiple Python references."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [np.random.randn(1000).astype(np.float32)],
            "shape": [[1000]],
        })
        
        result1 = quantizer.quantize_arrow_batch(table, bit_width=4)
        result2 = result1  # Create another reference
        result3 = result1  # And another
        
        # All should be valid
        assert result1.num_rows == 1
        assert result2.num_rows == 1
        assert result3.num_rows == 1
        
        # Delete references
        del result1
        del result2
        del result3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

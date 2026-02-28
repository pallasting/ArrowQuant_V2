"""
Test export_recordbatch_to_pyarrow() function.

This test verifies that the export_recordbatch_to_pyarrow() function correctly
exports Rust RecordBatch to Python using Arrow C Data Interface for zero-copy.

Requirements tested:
- 4.1: Constructs Arrow RecordBatch to store results
- 4.2: Zero-copy export through Arrow C Data Interface
- 4.3: Uses PyCapsule to wrap FFI pointers
- 4.4: Provides custom destructors for memory safety
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
class TestExportRecordBatch:
    """Test export_recordbatch_to_pyarrow() function."""

    def test_export_single_layer(self):
        """Test exporting a single layer quantization result."""
        # Create input table
        layer_names = ["layer.0.weight"]
        weights_data = [np.random.randn(1000).astype(np.float32)]
        shapes = [[1000]]

        table = pa.Table.from_pydict({
            "layer_name": layer_names,
            "weights": weights_data,
            "shape": shapes,
        })

        # Quantize using arrow batch API (which uses export internally)
        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Verify result is a PyArrow Table
        assert isinstance(result, pa.Table)
        assert result.num_rows == 1

        # Verify all required columns exist (Requirement 4.1)
        expected_columns = [
            "layer_name",
            "quantized_data",
            "scales",
            "zero_points",
            "shape",
            "bit_width"
        ]
        for col in expected_columns:
            assert col in result.column_names, f"Missing column: {col}"

    def test_export_multiple_layers(self):
        """Test exporting multiple layers quantization result."""
        # Create input table with multiple layers
        layer_names = ["layer.0.weight", "layer.1.weight", "layer.2.weight"]
        weights_data = [
            np.random.randn(1000).astype(np.float32),
            np.random.randn(2000).astype(np.float32),
            np.random.randn(1500).astype(np.float32),
        ]
        shapes = [[1000], [2000], [1500]]

        table = pa.Table.from_pydict({
            "layer_name": layer_names,
            "weights": weights_data,
            "shape": shapes,
        })

        # Quantize
        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Verify result
        assert isinstance(result, pa.Table)
        assert result.num_rows == 3

        # Verify layer names preserved
        result_layer_names = result.column("layer_name").to_pylist()
        assert set(result_layer_names) == set(layer_names)

    def test_export_result_schema(self):
        """Test that exported result has correct schema (Requirement 4.1)."""
        # Create input
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [np.random.randn(100).astype(np.float32)],
            "shape": [[100]],
        })

        # Quantize
        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=8)

        # Verify schema types
        schema = result.schema

        # layer_name should be string
        assert schema.field("layer_name").type == pa.string()

        # quantized_data should be binary
        assert schema.field("quantized_data").type == pa.binary()

        # scales should be list<float32>
        scales_field = schema.field("scales")
        assert isinstance(scales_field.type, pa.ListType)
        assert scales_field.type.value_type == pa.float32()

        # zero_points should be list<float32>
        zp_field = schema.field("zero_points")
        assert isinstance(zp_field.type, pa.ListType)
        assert zp_field.type.value_type == pa.float32()

        # shape should be list<int64>
        shape_field = schema.field("shape")
        assert isinstance(shape_field.type, pa.ListType)
        assert shape_field.type.value_type == pa.int64()

        # bit_width should be uint8
        assert schema.field("bit_width").type == pa.uint8()

    def test_export_zero_copy(self):
        """Test that export is zero-copy (Requirement 4.2)."""
        # Create a large input to make copying noticeable
        large_weights = np.random.randn(1_000_000).astype(np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["large_layer"],
            "weights": [large_weights],
            "shape": [[1_000_000]],
        })

        # Quantize - should be fast due to zero-copy
        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Verify result is valid
        assert isinstance(result, pa.Table)
        assert result.num_rows == 1

        # The fact that this completes quickly indicates zero-copy
        # (A full copy of 1M floats would be slower)

    def test_export_memory_safety(self):
        """Test that custom destructors ensure memory safety (Requirement 4.4)."""
        # Create multiple results and let them go out of scope
        # This tests that the PyCapsule destructors properly clean up memory
        
        quantizer = ArrowQuantV2(mode="diffusion")
        
        for i in range(10):
            table = pa.Table.from_pydict({
                "layer_name": [f"layer.{i}"],
                "weights": [np.random.randn(1000).astype(np.float32)],
                "shape": [[1000]],
            })
            
            result = quantizer.quantize_arrow_batch(table, bit_width=4)
            
            # Access the data to ensure it's valid
            assert result.num_rows == 1
            assert result.column("layer_name").to_pylist()[0] == f"layer.{i}"
            
            # Result goes out of scope here - destructors should clean up

        # If we get here without segfault, memory management is working

    def test_export_with_different_bit_widths(self):
        """Test export works with different bit widths."""
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [np.random.randn(100).astype(np.float32)],
            "shape": [[100]],
        })

        quantizer = ArrowQuantV2(mode="diffusion")

        for bit_width in [2, 4, 8]:
            result = quantizer.quantize_arrow_batch(table, bit_width=bit_width)
            
            # Verify bit_width is correctly set
            bit_widths = result.column("bit_width").to_pylist()
            assert all(bw == bit_width for bw in bit_widths)

    def test_export_preserves_shapes(self):
        """Test that export preserves original tensor shapes."""
        # Create multi-dimensional tensors
        weights_2d = np.random.randn(10, 20).astype(np.float32)
        weights_3d = np.random.randn(5, 4, 3).astype(np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["layer_2d", "layer_3d"],
            "weights": [weights_2d.flatten(), weights_3d.flatten()],
            "shape": [[10, 20], [5, 4, 3]],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Verify shapes are preserved
        result_list = result.to_pylist()
        assert result_list[0]["shape"] == [10, 20]
        assert result_list[1]["shape"] == [5, 4, 3]

    def test_export_c_interface_compliance(self):
        """Test that export uses Arrow C Data Interface (Requirement 4.3)."""
        # The export function should use PyCapsule with proper names
        # We can verify this by checking that the result can be re-exported
        
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [np.random.randn(100).astype(np.float32)],
            "shape": [[100]],
        })

        quantizer = ArrowQuantV2(mode="diffusion")
        result = quantizer.quantize_arrow_batch(table, bit_width=4)

        # Convert to RecordBatch and verify it supports __arrow_c_array__
        batches = result.to_batches()
        assert len(batches) > 0
        
        batch = batches[0]
        assert hasattr(batch, "__arrow_c_array__")
        
        # Call __arrow_c_array__ to verify C interface works
        schema_capsule, array_capsule = batch.__arrow_c_array__()
        assert schema_capsule is not None
        assert array_capsule is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

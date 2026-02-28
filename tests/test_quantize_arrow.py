"""
Test quantize_arrow() method for zero-copy Arrow IPC quantization.

This test verifies that the quantize_arrow() method correctly processes
PyArrow Tables using the Arrow C Data Interface for zero-copy data transfer.
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
class TestQuantizeArrow:
    """Test quantize_arrow() method for zero-copy quantization."""

    def test_quantize_arrow_single_layer(self):
        """Test quantizing a single layer using Arrow IPC."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create input table with single layer
        weights = np.random.randn(1000).astype(np.float32)
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0.weight"],
            "weights": [weights.tolist()],
            "shape": [[1000]],
        })
        
        # Quantize using Arrow IPC
        result_table = quantizer.quantize_arrow(table, bit_width=4)
        
        # Verify result is a PyArrow Table
        assert isinstance(result_table, pa.Table)
        
        # Verify result schema
        assert "layer_name" in result_table.schema.names
        assert "quantized_data" in result_table.schema.names
        assert "scales" in result_table.schema.names
        assert "zero_points" in result_table.schema.names
        assert "shape" in result_table.schema.names
        assert "bit_width" in result_table.schema.names
        
        # Verify result has correct number of rows
        assert result_table.num_rows == 1
        
        # Verify layer name
        layer_names = result_table.column("layer_name").to_pylist()
        assert layer_names[0] == "layer.0.weight"
        
        # Verify bit width
        bit_widths = result_table.column("bit_width").to_pylist()
        assert bit_widths[0] == 4
        
        # Verify shape
        shapes = result_table.column("shape").to_pylist()
        assert shapes[0] == [1000]
        
        # Verify quantized data is not empty
        quantized_data = result_table.column("quantized_data").to_pylist()
        assert len(quantized_data[0]) > 0
        
        # Verify scales and zero_points are not empty
        scales = result_table.column("scales").to_pylist()
        assert len(scales[0]) > 0
        
        zero_points = result_table.column("zero_points").to_pylist()
        assert len(zero_points[0]) > 0

    def test_quantize_arrow_multiple_layers(self):
        """Test quantizing multiple layers in a single Arrow table."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create input table with multiple layers
        layer_names = ["layer.0.weight", "layer.1.weight", "layer.2.weight"]
        weights_list = [
            np.random.randn(1000).astype(np.float32).tolist(),
            np.random.randn(2000).astype(np.float32).tolist(),
            np.random.randn(500).astype(np.float32).tolist(),
        ]
        shapes = [[1000], [2000], [500]]
        
        table = pa.Table.from_pydict({
            "layer_name": layer_names,
            "weights": weights_list,
            "shape": shapes,
        })
        
        # Quantize using Arrow IPC
        result_table = quantizer.quantize_arrow(table, bit_width=4)
        
        # Verify result has correct number of rows
        assert result_table.num_rows == 3
        
        # Verify all layer names are preserved
        result_layer_names = result_table.column("layer_name").to_pylist()
        assert result_layer_names == layer_names
        
        # Verify all layers have quantized data
        quantized_data = result_table.column("quantized_data").to_pylist()
        assert all(len(data) > 0 for data in quantized_data)
        
        # Verify shapes are preserved
        result_shapes = result_table.column("shape").to_pylist()
        assert result_shapes == shapes

    def test_quantize_arrow_different_bit_widths(self):
        """Test quantizing with different bit widths."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.random.randn(1000).astype(np.float32)
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [weights.tolist()],
            "shape": [[1000]],
        })
        
        # Test bit_width=2
        result_2bit = quantizer.quantize_arrow(table, bit_width=2)
        assert result_2bit.column("bit_width").to_pylist()[0] == 2
        
        # Test bit_width=4
        result_4bit = quantizer.quantize_arrow(table, bit_width=4)
        assert result_4bit.column("bit_width").to_pylist()[0] == 4
        
        # Test bit_width=8
        result_8bit = quantizer.quantize_arrow(table, bit_width=8)
        assert result_8bit.column("bit_width").to_pylist()[0] == 8

    def test_quantize_arrow_without_shape_column(self):
        """Test quantizing when shape column is not provided."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table without shape column
        weights = np.random.randn(1000).astype(np.float32)
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0.weight"],
            "weights": [weights.tolist()],
        })
        
        # Should still work, inferring shape from weights
        result_table = quantizer.quantize_arrow(table, bit_width=4)
        
        assert result_table.num_rows == 1
        
        # Verify shape was inferred
        shapes = result_table.column("shape").to_pylist()
        assert shapes[0] == [1000]

    def test_quantize_arrow_2d_weights(self):
        """Test quantizing 2D weight matrices."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create 2D weights (e.g., 32x64 matrix)
        rows, cols = 32, 64
        weights = np.random.randn(rows * cols).astype(np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0.weight"],
            "weights": [weights.tolist()],
            "shape": [[rows, cols]],
        })
        
        # Quantize
        result_table = quantizer.quantize_arrow(table, bit_width=4)
        
        # Verify shape is preserved
        shapes = result_table.column("shape").to_pylist()
        assert shapes[0] == [rows, cols]

    def test_quantize_arrow_invalid_bit_width(self):
        """Test that invalid bit widths raise appropriate errors."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.random.randn(100).astype(np.float32)
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [weights.tolist()],
        })
        
        # Test invalid bit width
        with pytest.raises(ValueError, match="Invalid bit_width"):
            quantizer.quantize_arrow(table, bit_width=3)
        
        with pytest.raises(ValueError, match="Invalid bit_width"):
            quantizer.quantize_arrow(table, bit_width=16)

    def test_quantize_arrow_missing_layer_name(self):
        """Test that missing layer_name column raises error."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table without layer_name column
        weights = np.random.randn(100).astype(np.float32)
        table = pa.Table.from_pydict({
            "weights": [weights.tolist()],
        })
        
        # Should raise error about missing layer_name
        with pytest.raises(ValueError, match="layer_name"):
            quantizer.quantize_arrow(table, bit_width=4)

    def test_quantize_arrow_missing_weights(self):
        """Test that missing weights column raises error."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table without weights column
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
        })
        
        # Should raise error about missing weights
        with pytest.raises(ValueError, match="weights"):
            quantizer.quantize_arrow(table, bit_width=4)

    def test_quantize_arrow_nan_values(self):
        """Test that NaN values in weights raise appropriate error."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create weights with NaN
        weights = np.random.randn(100).astype(np.float32)
        weights[50] = np.nan
        
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [weights.tolist()],
        })
        
        # Should raise error about NaN values
        with pytest.raises(ValueError, match="NaN or Inf"):
            quantizer.quantize_arrow(table, bit_width=4)

    def test_quantize_arrow_inf_values(self):
        """Test that Inf values in weights raise appropriate error."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create weights with Inf
        weights = np.random.randn(100).astype(np.float32)
        weights[25] = np.inf
        
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [weights.tolist()],
        })
        
        # Should raise error about Inf values
        with pytest.raises(ValueError, match="NaN or Inf"):
            quantizer.quantize_arrow(table, bit_width=4)

    def test_quantize_arrow_empty_table(self):
        """Test handling of empty tables."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create empty table with correct schema
        table = pa.Table.from_pydict({
            "layer_name": [],
            "weights": [],
            "shape": [],
        })
        
        # Should handle empty table gracefully
        result_table = quantizer.quantize_arrow(table, bit_width=4)
        assert result_table.num_rows == 0

    def test_quantize_arrow_large_batch(self):
        """Test quantizing a large batch of layers."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create a large batch (100 layers)
        num_layers = 100
        layer_names = [f"layer.{i}.weight" for i in range(num_layers)]
        weights_list = [
            np.random.randn(1000).astype(np.float32).tolist()
            for _ in range(num_layers)
        ]
        shapes = [[1000] for _ in range(num_layers)]
        
        table = pa.Table.from_pydict({
            "layer_name": layer_names,
            "weights": weights_list,
            "shape": shapes,
        })
        
        # Quantize
        result_table = quantizer.quantize_arrow(table, bit_width=4)
        
        # Verify all layers were processed
        assert result_table.num_rows == num_layers
        
        # Verify all layer names are preserved
        result_layer_names = result_table.column("layer_name").to_pylist()
        assert result_layer_names == layer_names

    def test_quantize_arrow_result_schema_types(self):
        """Test that result table has correct schema types."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights = np.random.randn(100).astype(np.float32)
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [weights.tolist()],
            "shape": [[100]],
        })
        
        result_table = quantizer.quantize_arrow(table, bit_width=4)
        
        # Verify schema types
        schema = result_table.schema
        
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

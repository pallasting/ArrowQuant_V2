"""
Unit tests for Task 2.4: Result Building Phase

Tests the result building phase of quantize_batch_arrow that constructs
the output RecordBatch with all required columns.

**Validates: Requirements 4.1, 4.2, 4.5**
"""

import pytest
import numpy as np
import pyarrow as pa
from arrow_quant_v2 import ArrowQuantV2


def create_arrow_table(layers_dict):
    """Helper to create Arrow Table from layer dictionary"""
    layer_names = []
    weights_lists = []
    shapes_lists = []
    
    for layer_name, weights_array in layers_dict.items():
        flat_weights = weights_array.flatten()
        shape = list(weights_array.shape)
        
        layer_names.append(layer_name)
        weights_lists.append(flat_weights)
        shapes_lists.append(shape)
    
    table = pa.Table.from_arrays(
        [
            pa.array(layer_names),
            pa.array(weights_lists, type=pa.list_(pa.float32())),
            pa.array(shapes_lists, type=pa.list_(pa.int64())),
        ],
        names=["layer_name", "weights", "shape"]
    )
    
    return table


class TestResultBuilding:
    """Test suite for result building phase (Task 2.4)"""
    
    def test_result_schema_completeness(self):
        """Test that result RecordBatch contains all required columns"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create simple test data
        layers = {
            "layer.0": np.random.randn(128).astype(np.float32),
            "layer.1": np.random.randn(256).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Verify result is a PyArrow Table
        assert isinstance(result, pa.Table)
        
        # Verify all required columns exist
        expected_columns = [
            "layer_name",
            "quantized_data",
            "scales",
            "zero_points",
            "shape",
            "bit_width"
        ]
        
        for col in expected_columns:
            assert col in result.column_names, f"Missing required column: {col}"
    
    def test_result_column_types(self):
        """Test that result columns have correct types"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "layer.0": np.random.randn(128).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        schema = result.schema
        
        # Verify column types
        assert schema.field("layer_name").type == pa.string()
        assert schema.field("quantized_data").type == pa.binary()
        assert schema.field("scales").type == pa.list_(pa.float32())
        assert schema.field("zero_points").type == pa.list_(pa.float32())
        assert schema.field("shape").type == pa.list_(pa.int64())
        assert schema.field("bit_width").type == pa.uint8()
    
    def test_result_row_count(self):
        """Test that result has correct number of rows"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "layer.0": np.random.randn(128).astype(np.float32),
            "layer.1": np.random.randn(256).astype(np.float32),
            "layer.2": np.random.randn(512).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Should have one row per layer
        assert result.num_rows == len(layers)
    
    def test_result_layer_names_preserved(self):
        """Test that layer names are preserved in result"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "model.layer.0.weight": np.random.randn(128).astype(np.float32),
            "model.layer.1.bias": np.random.randn(64).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Extract layer names from result
        result_layer_names = set(result.column("layer_name").to_pylist())
        
        # Should match input layer names
        assert result_layer_names == set(layers.keys())
    
    def test_result_shapes_preserved(self):
        """Test that shapes are preserved in result"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "layer.0": np.random.randn(4, 8).astype(np.float32),
            "layer.1": np.random.randn(16, 32).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Convert to list of dicts for easier access
        result_list = result.to_pylist()
        
        # Verify shapes are preserved
        for row in result_list:
            layer_name = row["layer_name"]
            original_shape = list(layers[layer_name].shape)
            result_shape = row["shape"]
            assert result_shape == original_shape, \
                f"Shape mismatch for {layer_name}: {result_shape} != {original_shape}"
    
    def test_result_bit_width_correct(self):
        """Test that bit_width is correctly set in result"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "layer.0": np.random.randn(128).astype(np.float32),
        }
        
        for bit_width in [2, 4, 8]:
            table = create_arrow_table(layers)
            result = quantizer.quantize_batch_arrow(table, bit_width=bit_width)
            
            # All rows should have the specified bit_width
            bit_widths = result.column("bit_width").to_pylist()
            assert all(bw == bit_width for bw in bit_widths), \
                f"Expected all bit_widths to be {bit_width}, got {bit_widths}"
    
    def test_result_quantized_data_not_empty(self):
        """Test that quantized_data is not empty"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "layer.0": np.random.randn(128).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Quantized data should not be empty
        quantized_data = result.column("quantized_data").to_pylist()
        assert all(len(data) > 0 for data in quantized_data), \
            "Quantized data should not be empty"
    
    def test_result_scales_not_empty(self):
        """Test that scales list is not empty"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "layer.0": np.random.randn(128).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Scales should not be empty
        scales = result.column("scales").to_pylist()
        assert all(len(s) > 0 for s in scales), \
            "Scales should not be empty"
        
        # All scales should be finite
        for scale_list in scales:
            assert all(np.isfinite(s) for s in scale_list), \
                "All scales should be finite"
    
    def test_result_zero_points_not_empty(self):
        """Test that zero_points list is not empty"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "layer.0": np.random.randn(128).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Zero points should not be empty
        zero_points = result.column("zero_points").to_pylist()
        assert all(len(zp) > 0 for zp in zero_points), \
            "Zero points should not be empty"
        
        # All zero points should be finite
        for zp_list in zero_points:
            assert all(np.isfinite(zp) for zp in zp_list), \
                "All zero points should be finite"
    
    def test_result_scales_zero_points_same_length(self):
        """Test that scales and zero_points have same length"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "layer.0": np.random.randn(256).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        result_list = result.to_pylist()
        
        for row in result_list:
            scales = row["scales"]
            zero_points = row["zero_points"]
            assert len(scales) == len(zero_points), \
                f"Scales and zero_points should have same length: {len(scales)} != {len(zero_points)}"
    
    def test_result_multiple_layers(self):
        """Test result building with multiple layers"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create multiple layers with different sizes
        layers = {
            f"layer.{i}": np.random.randn(128 * (i + 1)).astype(np.float32)
            for i in range(5)
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Should have one row per layer
        assert result.num_rows == len(layers)
        
        # All required columns should be present
        assert "layer_name" in result.column_names
        assert "quantized_data" in result.column_names
        assert "scales" in result.column_names
        assert "zero_points" in result.column_names
        assert "shape" in result.column_names
        assert "bit_width" in result.column_names
    
    def test_result_empty_table(self):
        """Test result building with empty table"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create empty table with correct schema
        table = pa.Table.from_arrays(
            [
                pa.array([], type=pa.string()),
                pa.array([], type=pa.list_(pa.float32())),
                pa.array([], type=pa.list_(pa.int64())),
            ],
            names=["layer_name", "weights", "shape"]
        )
        
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Should return empty table with correct schema
        assert result.num_rows == 0
        assert "layer_name" in result.column_names
        assert "quantized_data" in result.column_names
    
    def test_result_single_layer(self):
        """Test result building with single layer"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "single.layer": np.random.randn(128).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Should have exactly one row
        assert result.num_rows == 1
        
        # Verify the single row has all required data
        row = result.to_pylist()[0]
        assert row["layer_name"] == "single.layer"
        assert len(row["quantized_data"]) > 0
        assert len(row["scales"]) > 0
        assert len(row["zero_points"]) > 0
        assert row["shape"] == [128]
        assert row["bit_width"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

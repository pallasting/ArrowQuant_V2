"""
Unit tests for Task 2.5: Core Quantization Logic

This test file validates the complete quantize_batch_arrow() implementation end-to-end,
covering Tasks 2.2-2.4 integration:
- Single layer quantization
- Multi-layer parallel quantization
- Empty table handling
- Error collection mechanism
- Continue-on-error mode

**Validates: Requirements 1.1, 3.1, 3.5, 8.4**
"""

import pytest
import numpy as np
import pyarrow as pa
from arrow_quant_v2 import ArrowQuantV2


def create_arrow_table(layers_dict):
    """Helper to create Arrow Table from layer dictionary."""
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


class TestSingleLayerQuantization:
    """Test single layer quantization end-to-end."""
    
    def test_single_layer_basic(self):
        """Test basic single layer quantization."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create single layer
        layers = {
            "layer.0": np.random.randn(128).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Verify result structure
        assert isinstance(result, pa.Table)
        assert result.num_rows == 1
        
        # Verify all columns present
        assert "layer_name" in result.column_names
        assert "quantized_data" in result.column_names
        assert "scales" in result.column_names
        assert "zero_points" in result.column_names
        assert "shape" in result.column_names
        assert "bit_width" in result.column_names
        
        # Verify data
        row = result.to_pylist()[0]
        assert row["layer_name"] == "layer.0"
        assert len(row["quantized_data"]) > 0
        assert len(row["scales"]) > 0
        assert len(row["zero_points"]) > 0
        assert row["shape"] == [128]
        assert row["bit_width"] == 4
    
    def test_single_layer_2d_array(self):
        """Test single layer with 2D array."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "layer.0": np.random.randn(16, 32).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        assert result.num_rows == 1
        row = result.to_pylist()[0]
        assert row["shape"] == [16, 32]
    
    def test_single_layer_different_bit_widths(self):
        """Test single layer with different bit widths."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "layer.0": np.random.randn(256).astype(np.float32),
        }
        
        for bit_width in [2, 4, 8]:
            table = create_arrow_table(layers)
            result = quantizer.quantize_batch_arrow(table, bit_width=bit_width)
            
            assert result.num_rows == 1
            row = result.to_pylist()[0]
            assert row["bit_width"] == bit_width
    
    def test_single_layer_large_array(self):
        """Test single layer with large array (4MB)."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # 4MB = 1M float32 values
        layers = {
            "large.layer": np.random.randn(1024 * 1024).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        assert result.num_rows == 1
        row = result.to_pylist()[0]
        assert row["layer_name"] == "large.layer"
        assert len(row["quantized_data"]) > 0


class TestMultiLayerParallelQuantization:
    """Test multi-layer parallel quantization (Requirement 3.1)."""
    
    def test_multi_layer_basic(self):
        """Test basic multi-layer quantization."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create multiple layers
        layers = {
            "layer.0": np.random.randn(128).astype(np.float32),
            "layer.1": np.random.randn(256).astype(np.float32),
            "layer.2": np.random.randn(512).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Verify all layers processed
        assert result.num_rows == len(layers)
        
        # Verify all layer names present
        result_layer_names = set(result.column("layer_name").to_pylist())
        assert result_layer_names == set(layers.keys())
    
    def test_multi_layer_10_layers(self):
        """Test with 10 layers to verify parallel processing."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            f"layer.{i}": np.random.randn(128 * (i + 1)).astype(np.float32)
            for i in range(10)
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        assert result.num_rows == 10
        
        # Verify each layer has valid quantization results
        result_list = result.to_pylist()
        for row in result_list:
            assert len(row["quantized_data"]) > 0
            assert len(row["scales"]) > 0
            assert len(row["zero_points"]) > 0
            assert all(np.isfinite(s) for s in row["scales"])
            assert all(np.isfinite(zp) for zp in row["zero_points"])
    
    def test_multi_layer_different_shapes(self):
        """Test multi-layer with different shapes."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "layer.1d": np.random.randn(100).astype(np.float32),
            "layer.2d": np.random.randn(10, 20).astype(np.float32),
            "layer.3d": np.random.randn(5, 4, 10).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        assert result.num_rows == 3
        
        # Verify shapes preserved
        result_dict = {row["layer_name"]: row for row in result.to_pylist()}
        assert result_dict["layer.1d"]["shape"] == [100]
        assert result_dict["layer.2d"]["shape"] == [10, 20]
        assert result_dict["layer.3d"]["shape"] == [5, 4, 10]
    
    def test_multi_layer_100_layers(self):
        """Test with 100 layers to verify scalability."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create 100 layers with 4KB each (1024 float32 values)
        layers = {
            f"layer.{i:03d}": np.random.randn(1024).astype(np.float32)
            for i in range(100)
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Verify all 100 layers processed
        assert result.num_rows == 100
        
        # Verify all layer names present
        result_layer_names = set(result.column("layer_name").to_pylist())
        assert result_layer_names == set(layers.keys())
    
    def test_multi_layer_deterministic_ordering(self):
        """Test that layer ordering is deterministic (sorted by name)."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create layers with non-alphabetical names
        layers = {
            "layer.z": np.random.randn(100).astype(np.float32),
            "layer.a": np.random.randn(100).astype(np.float32),
            "layer.m": np.random.randn(100).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Verify layers are sorted alphabetically
        result_layer_names = result.column("layer_name").to_pylist()
        expected_order = sorted(layers.keys())
        assert result_layer_names == expected_order


class TestEmptyTableHandling:
    """Test empty table handling."""
    
    def test_empty_table(self):
        """Test quantization with empty table."""
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
        assert isinstance(result, pa.Table)
        assert result.num_rows == 0
        
        # Verify schema is correct
        assert "layer_name" in result.column_names
        assert "quantized_data" in result.column_names
        assert "scales" in result.column_names
        assert "zero_points" in result.column_names
        assert "shape" in result.column_names
        assert "bit_width" in result.column_names
    
    def test_empty_table_different_bit_widths(self):
        """Test empty table with different bit widths."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        table = pa.Table.from_arrays(
            [
                pa.array([], type=pa.string()),
                pa.array([], type=pa.list_(pa.float32())),
                pa.array([], type=pa.list_(pa.int64())),
            ],
            names=["layer_name", "weights", "shape"]
        )
        
        for bit_width in [2, 4, 8]:
            result = quantizer.quantize_batch_arrow(table, bit_width=bit_width)
            assert result.num_rows == 0


class TestErrorCollectionMechanism:
    """Test error collection mechanism (Requirement 3.5, 8.4)."""
    
    def test_error_collection_nan_fail_fast(self):
        """Test error collection with NaN in fail-fast mode."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with NaN in one layer
        weights_with_nan = np.random.randn(100).astype(np.float32)
        weights_with_nan[50] = np.nan
        
        layers = {
            "layer.0": np.random.randn(100).astype(np.float32),
            "layer.1": weights_with_nan,
            "layer.2": np.random.randn(100).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        
        # Should fail with error about NaN
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(table, bit_width=4, continue_on_error=False)
        
        error_msg = str(exc_info.value)
        assert "layer.1" in error_msg
        assert "NaN" in error_msg
    
    def test_error_collection_inf_fail_fast(self):
        """Test error collection with Inf in fail-fast mode."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with Inf in one layer
        weights_with_inf = np.random.randn(100).astype(np.float32)
        weights_with_inf[25] = np.inf
        
        layers = {
            "layer.0": np.random.randn(100).astype(np.float32),
            "layer.1": weights_with_inf,
        }
        
        table = create_arrow_table(layers)
        
        # Should fail with error about Inf
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(table, bit_width=4, continue_on_error=False)
        
        error_msg = str(exc_info.value)
        assert "layer.1" in error_msg
        assert "Inf" in error_msg
    
    def test_error_collection_shape_mismatch_fail_fast(self):
        """Test error collection with shape mismatch in fail-fast mode."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with shape mismatch
        layer_names = ["layer.0", "layer.1"]
        weights_lists = [
            np.random.randn(100).astype(np.float32),
            np.random.randn(100).astype(np.float32),
        ]
        shapes_lists = [
            [100],  # Correct
            [10, 20],  # Mismatch: product is 200, but weights has 100 elements
        ]
        
        table = pa.Table.from_arrays(
            [
                pa.array(layer_names),
                pa.array(weights_lists, type=pa.list_(pa.float32())),
                pa.array(shapes_lists, type=pa.list_(pa.int64())),
            ],
            names=["layer_name", "weights", "shape"]
        )
        
        # Should fail with shape mismatch error
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(table, bit_width=4, continue_on_error=False)
        
        error_msg = str(exc_info.value)
        assert "Shape mismatch" in error_msg
        assert "layer.1" in error_msg


class TestContinueOnErrorMode:
    """Test continue-on-error mode (Requirement 8.4)."""
    
    def test_continue_on_error_default_false(self):
        """Test that continue_on_error defaults to False."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with NaN
        weights_with_nan = np.random.randn(100).astype(np.float32)
        weights_with_nan[50] = np.nan
        
        layers = {
            "layer.0": np.random.randn(100).astype(np.float32),
            "layer.1": weights_with_nan,
        }
        
        table = create_arrow_table(layers)
        
        # Should fail (default is continue_on_error=False)
        with pytest.raises(ValueError):
            quantizer.quantize_batch_arrow(table, bit_width=4)
    
    def test_continue_on_error_explicit_false(self):
        """Test continue_on_error=False explicitly."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights_with_nan = np.random.randn(100).astype(np.float32)
        weights_with_nan[50] = np.nan
        
        layers = {
            "layer.0": np.random.randn(100).astype(np.float32),
            "layer.1": weights_with_nan,
        }
        
        table = create_arrow_table(layers)
        
        # Should fail
        with pytest.raises(ValueError):
            quantizer.quantize_batch_arrow(table, bit_width=4, continue_on_error=False)


class TestEndToEndIntegration:
    """Test end-to-end integration of all phases."""
    
    def test_complete_workflow_single_layer(self):
        """Test complete workflow with single layer."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create test data
        original_weights = np.random.randn(256).astype(np.float32)
        layers = {"test.layer": original_weights}
        
        # Convert to Arrow Table
        table = create_arrow_table(layers)
        
        # Quantize
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Verify result
        assert result.num_rows == 1
        row = result.to_pylist()[0]
        
        # Verify all fields
        assert row["layer_name"] == "test.layer"
        assert len(row["quantized_data"]) > 0
        assert len(row["scales"]) > 0
        assert len(row["zero_points"]) > 0
        assert row["shape"] == [256]
        assert row["bit_width"] == 4
        
        # Verify quantization parameters are valid
        assert all(np.isfinite(s) for s in row["scales"])
        assert all(np.isfinite(zp) for zp in row["zero_points"])
    
    def test_complete_workflow_multiple_layers(self):
        """Test complete workflow with multiple layers."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create test data with different sizes
        layers = {
            "layer.0": np.random.randn(128).astype(np.float32),
            "layer.1": np.random.randn(256).astype(np.float32),
            "layer.2": np.random.randn(512).astype(np.float32),
            "layer.3": np.random.randn(1024).astype(np.float32),
        }
        
        # Convert to Arrow Table
        table = create_arrow_table(layers)
        
        # Quantize
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Verify all layers processed
        assert result.num_rows == len(layers)
        
        # Verify each layer
        result_dict = {row["layer_name"]: row for row in result.to_pylist()}
        
        for layer_name, original_weights in layers.items():
            assert layer_name in result_dict
            row = result_dict[layer_name]
            
            # Verify quantization results
            assert len(row["quantized_data"]) > 0
            assert len(row["scales"]) > 0
            assert len(row["zero_points"]) > 0
            assert row["shape"] == list(original_weights.shape)
            assert row["bit_width"] == 4
            
            # Verify parameters are valid
            assert all(np.isfinite(s) for s in row["scales"])
            assert all(np.isfinite(zp) for zp in row["zero_points"])
    
    def test_complete_workflow_all_bit_widths(self):
        """Test complete workflow with all bit widths."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        layers = {
            "layer.0": np.random.randn(256).astype(np.float32),
        }
        
        for bit_width in [2, 4, 8]:
            table = create_arrow_table(layers)
            result = quantizer.quantize_batch_arrow(table, bit_width=bit_width)
            
            assert result.num_rows == 1
            row = result.to_pylist()[0]
            assert row["bit_width"] == bit_width
            assert len(row["quantized_data"]) > 0
    
    def test_complete_workflow_realistic_model(self):
        """Test complete workflow with realistic model-like data."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Simulate a small model with various layer types
        layers = {
            "model.encoder.layer.0.attention.q.weight": np.random.randn(768, 768).astype(np.float32),
            "model.encoder.layer.0.attention.k.weight": np.random.randn(768, 768).astype(np.float32),
            "model.encoder.layer.0.attention.v.weight": np.random.randn(768, 768).astype(np.float32),
            "model.encoder.layer.0.ffn.dense1.weight": np.random.randn(768, 3072).astype(np.float32),
            "model.encoder.layer.0.ffn.dense2.weight": np.random.randn(3072, 768).astype(np.float32),
            "model.encoder.layer.0.attention.q.bias": np.random.randn(768).astype(np.float32),
            "model.encoder.layer.0.attention.k.bias": np.random.randn(768).astype(np.float32),
            "model.encoder.layer.0.attention.v.bias": np.random.randn(768).astype(np.float32),
        }
        
        table = create_arrow_table(layers)
        result = quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Verify all layers processed
        assert result.num_rows == len(layers)
        
        # Verify all layer names present
        result_layer_names = set(result.column("layer_name").to_pylist())
        assert result_layer_names == set(layers.keys())
        
        # Verify shapes preserved
        result_dict = {row["layer_name"]: row for row in result.to_pylist()}
        for layer_name, original_weights in layers.items():
            assert result_dict[layer_name]["shape"] == list(original_weights.shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

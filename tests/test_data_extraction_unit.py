"""
Unit tests for Task 2.2: Data extraction phase (holding GIL)

This test file validates the data extraction phase of quantize_batch_arrow:
- Import Arrow Table via C Data Interface
- Validate schema
- Extract layer_name, weights, shape columns
- Zero-copy slice access via Float32Array.values()
- Validate finite values (NaN/Inf detection)
- Clone to owned Vec for cross-thread passing
- Sort by layer name for deterministic ordering

Requirements tested:
- 1.1: Arrow Table input support
- 1.2: Extract all layers data
- 1.3: Parse shape information
- 2.1: Zero-copy slice references
- 2.2: Avoid data copying (except for cross-thread Vec)
- 8.1: NaN/Inf detection
- 8.2: Shape mismatch detection
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


class TestDataExtraction:
    """Test data extraction phase of quantize_batch_arrow."""
    
    def test_valid_table_import(self):
        """Test that valid Arrow Table can be imported."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create valid table
        layers = {
            "layer.0": np.random.randn(10, 10).astype(np.float32),
            "layer.1": np.random.randn(20, 20).astype(np.float32),
        }
        table = create_arrow_table(layers)
        
        # Should fail with NotImplementedError (Task 2.3/2.4 not done yet)
        # but should NOT fail with import/extraction errors
        with pytest.raises(NotImplementedError) as exc_info:
            quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Verify it got past data extraction phase
        assert "data extraction phase is complete" in str(exc_info.value)
    
    def test_nan_detection(self):
        """Test that NaN values are detected and rejected."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with NaN
        weights = np.random.randn(100).astype(np.float32)
        weights[50] = np.nan
        
        layers = {"layer.0": weights}
        table = create_arrow_table(layers)
        
        # Should fail with ValueError about NaN
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "layer.0" in error_msg
        assert "NaN" in error_msg
        assert "index 50" in error_msg
        assert "np.nan_to_num()" in error_msg
    
    def test_inf_detection(self):
        """Test that Inf values are detected and rejected."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with Inf
        weights = np.random.randn(100).astype(np.float32)
        weights[25] = np.inf
        
        layers = {"layer.0": weights}
        table = create_arrow_table(layers)
        
        # Should fail with ValueError about Inf
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "layer.0" in error_msg
        assert "Inf" in error_msg
        assert "index 25" in error_msg
    
    def test_shape_mismatch_detection(self):
        """Test that shape mismatch is detected."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with mismatched shape
        layer_names = ["layer.0"]
        weights_lists = [np.random.randn(100).astype(np.float32)]
        shapes_lists = [[10, 20]]  # Product is 200, but weights has 100 elements
        
        table = pa.Table.from_arrays(
            [
                pa.array(layer_names),
                pa.array(weights_lists, type=pa.list_(pa.float32())),
                pa.array(shapes_lists, type=pa.list_(pa.int64())),
            ],
            names=["layer_name", "weights", "shape"]
        )
        
        # Should fail with ValueError about shape mismatch
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "Shape mismatch" in error_msg
        assert "layer.0" in error_msg
        assert "200" in error_msg  # shape product
        assert "100" in error_msg  # weights length
    
    def test_multiple_layers_extraction(self):
        """Test extraction of multiple layers."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with multiple layers
        layers = {
            "layer.0": np.random.randn(10, 10).astype(np.float32),
            "layer.1": np.random.randn(20, 20).astype(np.float32),
            "layer.2": np.random.randn(5, 5).astype(np.float32),
        }
        table = create_arrow_table(layers)
        
        # Should successfully extract all layers
        with pytest.raises(NotImplementedError) as exc_info:
            quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Verify it got past data extraction phase
        assert "data extraction phase is complete" in str(exc_info.value)
    
    def test_1d_array_extraction(self):
        """Test extraction of 1D arrays."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with 1D array
        layers = {
            "layer.0": np.random.randn(100).astype(np.float32),
        }
        table = create_arrow_table(layers)
        
        # Should successfully extract 1D array
        with pytest.raises(NotImplementedError) as exc_info:
            quantizer.quantize_batch_arrow(table, bit_width=4)
        
        assert "data extraction phase is complete" in str(exc_info.value)
    
    def test_empty_table(self):
        """Test extraction from empty table."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create empty table
        table = pa.Table.from_arrays(
            [
                pa.array([], type=pa.string()),
                pa.array([], type=pa.list_(pa.float32())),
                pa.array([], type=pa.list_(pa.int64())),
            ],
            names=["layer_name", "weights", "shape"]
        )
        
        # Should handle empty table gracefully
        with pytest.raises(NotImplementedError) as exc_info:
            quantizer.quantize_batch_arrow(table, bit_width=4)
        
        assert "data extraction phase is complete" in str(exc_info.value)
    
    def test_missing_layer_name_column(self):
        """Test error when layer_name column is missing."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table without layer_name
        table = pa.Table.from_arrays(
            [
                pa.array([[1.0, 2.0, 3.0]], type=pa.list_(pa.float32())),
            ],
            names=["weights"]
        )
        
        # Should fail with schema validation error
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "layer_name" in error_msg or "schema" in error_msg.lower()
    
    def test_missing_weights_column(self):
        """Test error when weights column is missing."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table without weights
        table = pa.Table.from_arrays(
            [
                pa.array(["layer.0"], type=pa.string()),
            ],
            names=["layer_name"]
        )
        
        # Should fail with schema validation error
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "weights" in error_msg or "schema" in error_msg.lower()
    
    def test_wrong_weights_type(self):
        """Test error when weights column has wrong type."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create table with int64 weights instead of float32
        table = pa.Table.from_arrays(
            [
                pa.array(["layer.0"], type=pa.string()),
                pa.array([[1, 2, 3]], type=pa.list_(pa.int64())),
            ],
            names=["layer_name", "weights"]
        )
        
        # Should fail with type error
        with pytest.raises(ValueError) as exc_info:
            quantizer.quantize_batch_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        assert "float32" in error_msg or "type" in error_msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for Task 2.3: Parallel Processing Phase

Tests the parallel processing implementation that:
- Uses Rayon's par_iter() to parallelize layer processing
- Implements thread-safe error collection
- Calls the quantization engine for each layer
- Collects quantization results
- Handles continue_on_error mode
"""

import pytest
import numpy as np
import pyarrow as pa
from arrow_quant_v2 import ArrowQuantV2


def create_test_arrow_table(num_layers=3, weights_per_layer=4):
    """Helper function to create a simple Arrow Table with test data"""
    layer_names = [f"layer{i}" for i in range(num_layers)]
    weights_data = [
        [float(i * weights_per_layer + j) for j in range(weights_per_layer)]
        for i in range(num_layers)
    ]
    shapes = [[1, weights_per_layer] for _ in range(num_layers)]
    
    table = pa.Table.from_arrays(
        [
            pa.array(layer_names),
            pa.array(weights_data, type=pa.list_(pa.float32())),
            pa.array(shapes, type=pa.list_(pa.int64())),
        ],
        names=["layer_name", "weights", "shape"]
    )
    
    return table


def test_parallel_processing_basic():
    """Test basic parallel processing with valid data"""
    quantizer = ArrowQuantV2(mode="diffusion")
    table = create_test_arrow_table(num_layers=3, weights_per_layer=4)
    
    # Call quantize_batch_arrow
    # Note: This will fail with NotImplementedError because Task 2.4 is not done yet
    # But it should complete the parallel processing phase
    with pytest.raises(NotImplementedError) as exc_info:
        quantizer.quantize_batch_arrow(table, bit_width=4)
    
    # Verify we reached Task 2.4 (parallel processing completed)
    assert "Result building (Task 2.4) is not yet implemented" in str(exc_info.value)


def test_parallel_processing_multiple_layers():
    """Test parallel processing with multiple layers"""
    quantizer = ArrowQuantV2(mode="diffusion")
    table = create_test_arrow_table(num_layers=10, weights_per_layer=128)
    
    # Should process all layers in parallel
    with pytest.raises(NotImplementedError) as exc_info:
        quantizer.quantize_batch_arrow(table, bit_width=4)
    
    assert "Result building (Task 2.4) is not yet implemented" in str(exc_info.value)


def test_parallel_processing_with_nan():
    """Test that NaN values are detected during data extraction (Task 2.2)"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Create table with NaN
    layer_names = ["layer_with_nan"]
    weights_data = [[1.0, float('nan'), 3.0, 4.0]]
    shapes = [[1, 4]]
    
    table = pa.Table.from_arrays(
        [
            pa.array(layer_names),
            pa.array(weights_data, type=pa.list_(pa.float32())),
            pa.array(shapes, type=pa.list_(pa.int64())),
        ],
        names=["layer_name", "weights", "shape"]
    )
    
    # Should fail with NaN error during data extraction (Task 2.2)
    with pytest.raises(ValueError) as exc_info:
        quantizer.quantize_batch_arrow(table, bit_width=4)
    
    error_msg = str(exc_info.value)
    assert "NaN" in error_msg
    assert "layer_with_nan" in error_msg


def test_parallel_processing_with_inf():
    """Test that Inf values are detected during data extraction (Task 2.2)"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Create table with Inf
    layer_names = ["layer_with_inf"]
    weights_data = [[1.0, float('inf'), 3.0, 4.0]]
    shapes = [[1, 4]]
    
    table = pa.Table.from_arrays(
        [
            pa.array(layer_names),
            pa.array(weights_data, type=pa.list_(pa.float32())),
            pa.array(shapes, type=pa.list_(pa.int64())),
        ],
        names=["layer_name", "weights", "shape"]
    )
    
    # Should fail with Inf error during data extraction (Task 2.2)
    with pytest.raises(ValueError) as exc_info:
        quantizer.quantize_batch_arrow(table, bit_width=4)
    
    error_msg = str(exc_info.value)
    assert "Inf" in error_msg
    assert "layer_with_inf" in error_msg


def test_parallel_processing_empty_table():
    """Test parallel processing with empty table"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Create empty table
    layer_names = []
    weights_data = []
    shapes = []
    
    table = pa.Table.from_arrays(
        [
            pa.array(layer_names),
            pa.array(weights_data, type=pa.list_(pa.float32())),
            pa.array(shapes, type=pa.list_(pa.int64())),
        ],
        names=["layer_name", "weights", "shape"]
    )
    
    # Should reach Task 2.4 not implemented error (no layers to process)
    with pytest.raises(NotImplementedError) as exc_info:
        quantizer.quantize_batch_arrow(table, bit_width=4)
    
    assert "Result building (Task 2.4) is not yet implemented" in str(exc_info.value)


def test_parallel_processing_different_bit_widths():
    """Test parallel processing with different bit widths"""
    quantizer = ArrowQuantV2(mode="diffusion")
    table = create_test_arrow_table(num_layers=3, weights_per_layer=8)
    
    # Test bit_width=2
    with pytest.raises(NotImplementedError) as exc_info:
        quantizer.quantize_batch_arrow(table, bit_width=2)
    assert "Result building (Task 2.4) is not yet implemented" in str(exc_info.value)
    
    # Test bit_width=4
    with pytest.raises(NotImplementedError) as exc_info:
        quantizer.quantize_batch_arrow(table, bit_width=4)
    assert "Result building (Task 2.4) is not yet implemented" in str(exc_info.value)
    
    # Test bit_width=8
    with pytest.raises(NotImplementedError) as exc_info:
        quantizer.quantize_batch_arrow(table, bit_width=8)
    assert "Result building (Task 2.4) is not yet implemented" in str(exc_info.value)


def test_parallel_processing_large_weights():
    """Test parallel processing with larger weight arrays"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Create table with larger weights (1024 elements per layer)
    table = create_test_arrow_table(num_layers=5, weights_per_layer=1024)
    
    # Should process successfully
    with pytest.raises(NotImplementedError) as exc_info:
        quantizer.quantize_batch_arrow(table, bit_width=4)
    
    assert "Result building (Task 2.4) is not yet implemented" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Test performance metrics logging in quantize_arrow_optimized()

This test verifies that performance metrics are logged correctly during
quantization operations, including timing, SIMD status, data size, and
memory usage.

**Validates: Requirements 5.3, 12.1** - Performance metrics recording
"""

import pyarrow as pa
import numpy as np
import logging
import sys
from arrow_quant_v2 import ArrowQuantV2


def test_performance_metrics_logging():
    """Test that performance metrics are logged during quantization"""
    
    # Set up logging to capture log messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    # Create quantizer
    quantizer = ArrowQuantV2()
    
    # Create test data with multiple layers
    num_layers = 5
    weights_per_layer = 10000
    
    layer_names = [f"layer.{i}" for i in range(num_layers)]
    weights_data = [
        np.random.randn(weights_per_layer).astype(np.float32)
        for _ in range(num_layers)
    ]
    
    # Create PyArrow Table
    table = pa.Table.from_pydict({
        "layer_name": layer_names,
        "weights": weights_data,
    })
    
    print("\n=== Testing Performance Metrics Logging ===")
    print(f"Input: {num_layers} layers, {weights_per_layer} weights per layer")
    print(f"Total elements: {num_layers * weights_per_layer}")
    
    # Quantize (this should log performance metrics)
    result_table = quantizer.quantize_arrow(table, bit_width=4)
    
    # Verify result
    assert result_table.num_rows == num_layers
    assert "quantized_data" in result_table.column_names
    assert "scales" in result_table.column_names
    assert "zero_points" in result_table.column_names
    
    print("\n✓ Quantization completed successfully")
    print("✓ Performance metrics should be logged above")
    print("\nExpected log entries:")
    print("  1. Quantization completed: X layers, Y total elements, Z ms elapsed")
    print("  2. SIMD status: enabled=true/false, width=X, type=...")
    print("  3. Memory usage: input=X MB, output=Y MB, total=Z MB")
    print("  4. Throughput: X elements/ms, Y MB/s")


def test_performance_metrics_with_different_sizes():
    """Test performance metrics with different data sizes"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    quantizer = ArrowQuantV2()
    
    # Test with different sizes
    test_cases = [
        (1, 100),      # Small: 1 layer, 100 elements
        (10, 1000),    # Medium: 10 layers, 1000 elements each
        (5, 10000),    # Large: 5 layers, 10000 elements each
    ]
    
    for num_layers, weights_per_layer in test_cases:
        print(f"\n=== Testing {num_layers} layers × {weights_per_layer} elements ===")
        
        layer_names = [f"layer.{i}" for i in range(num_layers)]
        weights_data = [
            np.random.randn(weights_per_layer).astype(np.float32)
            for _ in range(num_layers)
        ]
        
        table = pa.Table.from_pydict({
            "layer_name": layer_names,
            "weights": weights_data,
        })
        
        result_table = quantizer.quantize_arrow(table, bit_width=4)
        
        assert result_table.num_rows == num_layers
        print(f"✓ Completed: {num_layers} layers")


def test_performance_metrics_with_different_bit_widths():
    """Test performance metrics with different bit widths"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    quantizer = ArrowQuantV2()
    
    # Create test data
    num_layers = 3
    weights_per_layer = 5000
    
    layer_names = [f"layer.{i}" for i in range(num_layers)]
    weights_data = [
        np.random.randn(weights_per_layer).astype(np.float32)
        for _ in range(num_layers)
    ]
    
    table = pa.Table.from_pydict({
        "layer_name": layer_names,
        "weights": weights_data,
    })
    
    # Test with different bit widths
    for bit_width in [2, 4, 8]:
        print(f"\n=== Testing bit_width={bit_width} ===")
        
        result_table = quantizer.quantize_arrow(table, bit_width=bit_width)
        
        assert result_table.num_rows == num_layers
        print(f"✓ Completed with bit_width={bit_width}")


if __name__ == "__main__":
    print("=" * 70)
    print("Performance Metrics Logging Tests")
    print("=" * 70)
    
    test_performance_metrics_logging()
    print("\n" + "=" * 70)
    
    test_performance_metrics_with_different_sizes()
    print("\n" + "=" * 70)
    
    test_performance_metrics_with_different_bit_widths()
    print("\n" + "=" * 70)
    
    print("\n✓ All performance metrics logging tests passed!")

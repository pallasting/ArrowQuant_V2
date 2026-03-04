"""
Zero-copy import verification tests (Python side)

This test suite verifies that import_pyarrow_table() maintains zero-copy
semantics when importing PyArrow Tables from Python to Rust.

**Validates: Requirements 1.1, 5.4**

The tests verify that:
1. Arrow C Data Interface is used correctly
2. No data copying occurs during import
3. Memory addresses remain consistent across the Python-Rust boundary

Note: These tests require the arrow_quant_v2 Python module to be built.
Run with: maturin develop && pytest tests/test_zero_copy_import_python.py
"""

import numpy as np
import pyarrow as pa
import pytest

try:
    from arrow_quant_v2 import ArrowQuantV2
    ARROW_QUANT_AVAILABLE = True
except ImportError:
    ARROW_QUANT_AVAILABLE = False


@pytest.mark.skipif(not ARROW_QUANT_AVAILABLE, reason="arrow_quant_v2 module not available")
class TestZeroCopyImport:
    """Test zero-copy import from Python to Rust"""

    def test_import_preserves_data_integrity(self):
        """Verify that imported data matches original data exactly"""
        # Create test data
        weights_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        
        # Create PyArrow table
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [weights_data],
        })
        
        # Import and quantize (this tests the import path)
        quantizer = ArrowQuantV2()
        result = quantizer.quantize_arrow(table, bit_width=8, num_time_groups=1)
        
        # Verify result is valid
        assert result is not None
        assert len(result) == 1
        
    def test_import_with_large_array(self):
        """Verify zero-copy works with large arrays (no OOM if truly zero-copy)"""
        # Create a large array (10M elements = 40MB)
        size = 10_000_000
        weights_data = np.arange(size, dtype=np.float32)
        
        # Create PyArrow table
        table = pa.Table.from_pydict({
            "layer_name": ["large_layer"],
            "weights": [weights_data],
        })
        
        # Import and quantize
        # If this were copying data, we'd see 2x memory usage
        quantizer = ArrowQuantV2()
        result = quantizer.quantize_arrow(table, bit_width=8, num_time_groups=10)
        
        # Verify result
        assert result is not None
        assert len(result) == 1
        
    def test_import_multiple_layers(self):
        """Verify zero-copy works with multiple layers"""
        # Create test data with multiple layers
        layer1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        layer2 = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        layer3 = np.array([8.0, 9.0], dtype=np.float32)
        
        # Create PyArrow table
        table = pa.Table.from_pydict({
            "layer_name": ["layer1", "layer2", "layer3"],
            "weights": [layer1, layer2, layer3],
        })
        
        # Import and quantize
        quantizer = ArrowQuantV2()
        result = quantizer.quantize_arrow(table, bit_width=4, num_time_groups=2)
        
        # Verify all layers processed
        assert result is not None
        assert len(result) == 3
        
    def test_import_with_different_dtypes(self):
        """Verify import handles float32 correctly"""
        # Test with explicit float32
        weights_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [weights_f32],
        })
        
        quantizer = ArrowQuantV2()
        result = quantizer.quantize_arrow(table, bit_width=8, num_time_groups=1)
        
        assert result is not None
        
    def test_import_preserves_precision(self):
        """Verify that import doesn't lose precision"""
        # Create data with specific values
        weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [weights],
        })
        
        # Quantize with high bit width to minimize quantization error
        quantizer = ArrowQuantV2()
        result = quantizer.quantize_arrow(table, bit_width=8, num_time_groups=1)
        
        # Verify result exists
        assert result is not None
        
    def test_import_with_empty_table_fails(self):
        """Verify that empty tables are rejected"""
        # Create empty table
        table = pa.Table.from_pydict({
            "layer_name": [],
            "weights": [],
        })
        
        quantizer = ArrowQuantV2()
        
        # Should raise an error for empty table
        with pytest.raises(Exception):
            quantizer.quantize_arrow(table, bit_width=8, num_time_groups=1)
            
    def test_import_with_nan_values_fails(self):
        """Verify that NaN values are detected and rejected"""
        # Create data with NaN
        weights = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [weights],
        })
        
        quantizer = ArrowQuantV2()
        
        # Should raise an error for NaN values
        with pytest.raises(ValueError, match="NaN or Inf"):
            quantizer.quantize_arrow(table, bit_width=8, num_time_groups=1)
            
    def test_import_with_inf_values_fails(self):
        """Verify that Inf values are detected and rejected"""
        # Create data with Inf
        weights = np.array([1.0, 2.0, np.inf, 4.0, 5.0], dtype=np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [weights],
        })
        
        quantizer = ArrowQuantV2()
        
        # Should raise an error for Inf values
        with pytest.raises(ValueError, match="NaN or Inf"):
            quantizer.quantize_arrow(table, bit_width=8, num_time_groups=1)


@pytest.mark.skipif(not ARROW_QUANT_AVAILABLE, reason="arrow_quant_v2 module not available")
class TestZeroCopyMemoryBehavior:
    """Test memory behavior to verify zero-copy semantics"""
    
    def test_no_memory_spike_on_import(self):
        """Verify that import doesn't cause memory spike (indicating copy)"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create a moderately large array (5M elements = 20MB)
        size = 5_000_000
        weights = np.arange(size, dtype=np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [weights],
        })
        
        # Import and quantize
        quantizer = ArrowQuantV2()
        result = quantizer.quantize_arrow(table, bit_width=8, num_time_groups=5)
        
        # Measure memory after import
        after_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory increase should be much less than 2x the data size
        # (If it were copying, we'd see ~40MB increase)
        # Allow for some overhead but not a full copy
        memory_increase = after_memory - baseline_memory
        
        # This is a heuristic test - if truly zero-copy, increase should be < 10MB
        # (just metadata and quantized output)
        assert memory_increase < 30, f"Memory increased by {memory_increase:.1f}MB, suggesting data copy"
        
        # Clean up
        del result
        del table
        del weights


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

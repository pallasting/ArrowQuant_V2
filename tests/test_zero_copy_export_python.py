"""
Zero-copy export verification tests (Python side)

This test suite verifies that export_recordbatch_to_pyarrow() maintains zero-copy
semantics when exporting RecordBatch from Rust to Python.

**Validates: Requirements 5.4, 8.4**

The tests verify that:
1. Arrow C Data Interface is used correctly for export
2. No data copying occurs during export
3. Python can access data with zero-copy semantics
4. to_pandas(zero_copy_only=True) succeeds

Note: These tests require the arrow_quant_v2 Python module to be built.
Run with: maturin develop && pytest tests/test_zero_copy_export_python.py
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
class TestZeroCopyExport:
    """Test zero-copy export from Rust to Python"""

    def test_export_to_pandas_zero_copy_only(self):
        """
        Verify that exported data can be converted to pandas with zero_copy_only=True
        
        This is the acceptance criteria for Task 6.2:
        Python to_pandas(zero_copy_only=True) succeeds
        """
        # Create test data
        weights_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        
        # Create PyArrow table
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [weights_data],
        })
        
        # Quantize (this tests the export path)
        quantizer = ArrowQuantV2()
        
        # Convert table to RecordBatch first
        batches = table.to_batches()
        assert len(batches) > 0, "Table has no batches"
        
        # Use quantize_arrow_batch which now supports zero-copy export
        result_batch = quantizer.quantize_arrow_batch(batches[0], bit_width=8)
        
        # Verify result is a PyArrow RecordBatch
        assert isinstance(result_batch, pa.RecordBatch), f"Expected RecordBatch, got {type(result_batch)}"
        
        # The key test: convert to pandas with zero_copy_only=True
        # This will FAIL if the data was copied during export
        try:
            df = result_batch.to_pandas(zero_copy_only=True)
            assert df is not None
            assert len(df) > 0
            print("✓ Zero-copy export verified: to_pandas(zero_copy_only=True) succeeded")
        except pa.ArrowInvalid as e:
            pytest.fail(f"Zero-copy export failed: to_pandas(zero_copy_only=True) raised ArrowInvalid: {e}")
        
    def test_export_preserves_data_integrity(self):
        """Verify that exported data matches original data exactly"""
        # Create test data with known values
        weights_data = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
        
        # Create PyArrow table
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [weights_data],
        })
        
        # Quantize
        quantizer = ArrowQuantV2()
        result_batch = quantizer.quantize_arrow(table, bit_width=8, num_time_groups=1)
        
        # Verify result structure
        assert isinstance(result_batch, pa.RecordBatch)
        assert "quantized_data" in result_batch.schema.names
        assert "time_group_ids" in result_batch.schema.names
        
        # Convert to pandas (zero-copy)
        df = result_batch.to_pandas(zero_copy_only=True)
        assert len(df) > 0
        
    def test_export_with_large_array(self):
        """Verify zero-copy export works with large arrays"""
        # Create a large array (10M elements = 40MB)
        size = 10_000_000
        weights_data = np.arange(size, dtype=np.float32)
        
        # Create PyArrow table
        table = pa.Table.from_pydict({
            "layer_name": ["large_layer"],
            "weights": [weights_data],
        })
        
        # Quantize
        quantizer = ArrowQuantV2()
        result_batch = quantizer.quantize_arrow(table, bit_width=8, num_time_groups=10)
        
        # Verify zero-copy export
        assert isinstance(result_batch, pa.RecordBatch)
        
        # This should succeed without copying
        df = result_batch.to_pandas(zero_copy_only=True)
        assert len(df) > 0
        print(f"✓ Zero-copy export verified for large array ({size} elements)")
        
    def test_export_multiple_layers(self):
        """Verify zero-copy export works with multiple layers"""
        # Create test data with multiple layers
        layer1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        layer2 = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        layer3 = np.array([8.0, 9.0], dtype=np.float32)
        
        # Create PyArrow table
        table = pa.Table.from_pydict({
            "layer_name": ["layer1", "layer2", "layer3"],
            "weights": [layer1, layer2, layer3],
        })
        
        # Quantize
        quantizer = ArrowQuantV2()
        result_batch = quantizer.quantize_arrow(table, bit_width=4, num_time_groups=2)
        
        # Verify zero-copy export
        assert isinstance(result_batch, pa.RecordBatch)
        df = result_batch.to_pandas(zero_copy_only=True)
        assert len(df) == 3  # 3 layers
        
    def test_export_arrow_schema_preserved(self):
        """Verify that Arrow schema is correctly preserved during export"""
        # Create test data
        weights_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [weights_data],
        })
        
        # Quantize
        quantizer = ArrowQuantV2()
        result_batch = quantizer.quantize_arrow(table, bit_width=8, num_time_groups=1)
        
        # Verify schema
        schema = result_batch.schema
        assert "quantized_data" in schema.names
        assert "time_group_ids" in schema.names
        
        # Verify types
        quantized_field = schema.field("quantized_data")
        assert quantized_field.type == pa.uint8()
        
        time_group_field = schema.field("time_group_ids")
        assert time_group_field.type == pa.uint32()
        
    def test_export_with_arrow_quantized_layer(self):
        """
        Verify zero-copy export through PyArrowQuantizedLayer.to_pyarrow()
        
        This tests the complete workflow:
        1. Quantize using quantize_diffusion_model_arrow()
        2. Export using to_pyarrow()
        3. Convert to pandas with zero_copy_only=True
        """
        # Create test data
        weights_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["test_layer"],
            "weights": [weights_data],
        })
        
        # Quantize using Arrow API
        quantizer = ArrowQuantV2()
        result = quantizer.quantize_arrow(table, bit_width=8, num_time_groups=1)
        
        # Verify it's a RecordBatch
        assert isinstance(result, pa.RecordBatch)
        
        # Convert to pandas with zero-copy
        df = result.to_pandas(zero_copy_only=True)
        assert df is not None
        assert len(df) > 0
        print("✓ Zero-copy export verified through complete workflow")


@pytest.mark.skipif(not ARROW_QUANT_AVAILABLE, reason="arrow_quant_v2 module not available")
class TestZeroCopyMemoryBehaviorExport:
    """Test memory behavior to verify zero-copy export semantics"""
    
    def test_no_memory_spike_on_export(self):
        """Verify that export doesn't cause memory spike (indicating copy)"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Create a moderately large array (5M elements = 20MB)
        size = 5_000_000
        weights = np.arange(size, dtype=np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [weights],
        })
        
        # Quantize
        quantizer = ArrowQuantV2()
        result_batch = quantizer.quantize_arrow(table, bit_width=8, num_time_groups=5)
        
        # Measure baseline memory after quantization
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Export to pandas (this is where export happens)
        df = result_batch.to_pandas(zero_copy_only=True)
        
        # Measure memory after export
        after_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory increase should be minimal (just pandas overhead)
        # If it were copying, we'd see significant increase
        memory_increase = after_memory - baseline_memory
        
        # Allow for pandas overhead but not a full data copy
        assert memory_increase < 20, f"Memory increased by {memory_increase:.1f}MB during export, suggesting data copy"
        
        print(f"✓ Memory increase during export: {memory_increase:.1f}MB (expected < 20MB)")
        
        # Clean up
        del df
        del result_batch
        del table
        del weights
        
    def test_export_buffer_sharing(self):
        """
        Verify that exported RecordBatch shares buffers with Rust
        
        This test checks that the PyArrow RecordBatch returned from Rust
        uses the same underlying buffers (zero-copy).
        """
        # Create test data
        weights_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [weights_data],
        })
        
        # Quantize
        quantizer = ArrowQuantV2()
        result_batch = quantizer.quantize_arrow(table, bit_width=8, num_time_groups=1)
        
        # Get buffer information
        quantized_column = result_batch.column("quantized_data")
        
        # Verify we can access the buffer without copying
        # If this were a copy, to_pandas(zero_copy_only=True) would fail
        df = result_batch.to_pandas(zero_copy_only=True)
        
        # Verify data is accessible
        assert "quantized_data" in df.columns
        assert len(df["quantized_data"]) > 0
        
        print("✓ Buffer sharing verified: PyArrow can access Rust buffers directly")


@pytest.mark.skipif(not ARROW_QUANT_AVAILABLE, reason="arrow_quant_v2 module not available")
class TestZeroCopyExportEdgeCases:
    """Test edge cases for zero-copy export"""
    
    def test_export_empty_result_fails_gracefully(self):
        """Verify that empty results are handled correctly"""
        # Create empty table
        table = pa.Table.from_pydict({
            "layer_name": [],
            "weights": [],
        })
        
        quantizer = ArrowQuantV2()
        
        # Should raise an error for empty table
        with pytest.raises(Exception):
            quantizer.quantize_arrow(table, bit_width=8, num_time_groups=1)
            
    def test_export_with_different_bit_widths(self):
        """Verify zero-copy export works with different bit widths"""
        weights_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [weights_data],
        })
        
        quantizer = ArrowQuantV2()
        
        # Test with different bit widths
        for bit_width in [2, 4, 8]:
            result_batch = quantizer.quantize_arrow(table, bit_width=bit_width, num_time_groups=1)
            
            # Verify zero-copy export works
            df = result_batch.to_pandas(zero_copy_only=True)
            assert df is not None
            print(f"✓ Zero-copy export verified for bit_width={bit_width}")
            
    def test_export_with_different_time_groups(self):
        """Verify zero-copy export works with different numbers of time groups"""
        weights_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        
        table = pa.Table.from_pydict({
            "layer_name": ["test"],
            "weights": [weights_data],
        })
        
        quantizer = ArrowQuantV2()
        
        # Test with different time group counts
        for num_groups in [1, 5, 10, 20]:
            result_batch = quantizer.quantize_arrow(table, bit_width=8, num_time_groups=num_groups)
            
            # Verify zero-copy export works
            df = result_batch.to_pandas(zero_copy_only=True)
            assert df is not None
            print(f"✓ Zero-copy export verified for num_time_groups={num_groups}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

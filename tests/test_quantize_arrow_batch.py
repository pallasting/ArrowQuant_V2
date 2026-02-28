"""
Tests for quantize_arrow_batch() method - RecordBatch-level API.

This test suite validates the lower-level RecordBatch API for Arrow IPC quantization.
"""

import pytest
import numpy as np
import pyarrow as pa
from arrow_quant_v2 import ArrowQuantV2


class TestQuantizeArrowBatch:
    """Test suite for quantize_arrow_batch() method."""

    def test_basic_recordbatch_quantization(self):
        """Test basic RecordBatch quantization with single layer."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create RecordBatch with single layer
        weights_data = {
            "layer_name": ["layer.0.weight"],
            "weights": [np.random.randn(1000).astype(np.float32).tolist()],
            "shape": [[1000]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Quantize
        result_batch = quantizer.quantize_arrow_batch(batch, bit_width=4)
        
        # Verify result is RecordBatch
        assert isinstance(result_batch, pa.RecordBatch)
        assert result_batch.num_rows == 1
        
        # Verify schema
        expected_columns = {"layer_name", "quantized_data", "scales", "zero_points", "shape", "bit_width"}
        actual_columns = set(result_batch.schema.names)
        assert expected_columns == actual_columns
        
        # Verify layer name preserved
        assert result_batch.column("layer_name")[0].as_py() == "layer.0.weight"
        
        # Verify bit width
        assert result_batch.column("bit_width")[0].as_py() == 4

    def test_multi_layer_recordbatch(self):
        """Test RecordBatch quantization with multiple layers."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create RecordBatch with 5 layers
        num_layers = 5
        weights_data = {
            "layer_name": [f"layer.{i}.weight" for i in range(num_layers)],
            "weights": [np.random.randn(1000).astype(np.float32).tolist() for _ in range(num_layers)],
            "shape": [[1000] for _ in range(num_layers)],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Quantize
        result_batch = quantizer.quantize_arrow_batch(batch, bit_width=4)
        
        # Verify all layers processed
        assert result_batch.num_rows == num_layers
        
        # Verify all layer names preserved
        layer_names = result_batch.column("layer_name").to_pylist()
        assert layer_names == [f"layer.{i}.weight" for i in range(num_layers)]

    def test_recordbatch_2d_weights(self):
        """Test RecordBatch quantization with 2D weight matrices."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create RecordBatch with 2D weights
        rows, cols = 100, 200
        weights_2d = np.random.randn(rows, cols).astype(np.float32)
        
        weights_data = {
            "layer_name": ["layer.0.weight"],
            "weights": [weights_2d.flatten().tolist()],
            "shape": [[rows, cols]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Quantize
        result_batch = quantizer.quantize_arrow_batch(batch, bit_width=4)
        
        # Verify shape preserved
        result_shape = result_batch.column("shape")[0].as_py()
        assert result_shape == [rows, cols]

    def test_recordbatch_different_bit_widths(self):
        """Test RecordBatch quantization with different bit widths."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights_data = {
            "layer_name": ["layer.0.weight"],
            "weights": [np.random.randn(1000).astype(np.float32).tolist()],
            "shape": [[1000]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Test each bit width
        for bit_width in [2, 4, 8]:
            result_batch = quantizer.quantize_arrow_batch(batch, bit_width=bit_width)
            assert result_batch.column("bit_width")[0].as_py() == bit_width

    def test_recordbatch_invalid_bit_width(self):
        """Test RecordBatch quantization with invalid bit width."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights_data = {
            "layer_name": ["layer.0.weight"],
            "weights": [np.random.randn(1000).astype(np.float32).tolist()],
            "shape": [[1000]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Should raise ValueError for invalid bit width
        with pytest.raises(ValueError, match="Invalid bit_width"):
            quantizer.quantize_arrow_batch(batch, bit_width=3)

    def test_recordbatch_missing_layer_name_column(self):
        """Test RecordBatch quantization with missing layer_name column."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create RecordBatch without layer_name
        weights_data = {
            "weights": [np.random.randn(1000).astype(np.float32).tolist()],
            "shape": [[1000]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Missing required column 'layer_name'"):
            quantizer.quantize_arrow_batch(batch, bit_width=4)

    def test_recordbatch_missing_weights_column(self):
        """Test RecordBatch quantization with missing weights column."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create RecordBatch without weights
        weights_data = {
            "layer_name": ["layer.0.weight"],
            "shape": [[1000]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Missing required column 'weights'"):
            quantizer.quantize_arrow_batch(batch, bit_width=4)

    def test_recordbatch_nan_values(self):
        """Test RecordBatch quantization with NaN values."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create weights with NaN
        weights = np.random.randn(1000).astype(np.float32)
        weights[500] = np.nan
        
        weights_data = {
            "layer_name": ["layer.0.weight"],
            "weights": [weights.tolist()],
            "shape": [[1000]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            quantizer.quantize_arrow_batch(batch, bit_width=4)

    def test_recordbatch_inf_values(self):
        """Test RecordBatch quantization with Inf values."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create weights with Inf
        weights = np.random.randn(1000).astype(np.float32)
        weights[500] = np.inf
        
        weights_data = {
            "layer_name": ["layer.0.weight"],
            "weights": [weights.tolist()],
            "shape": [[1000]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            quantizer.quantize_arrow_batch(batch, bit_width=4)

    def test_recordbatch_without_shape_column(self):
        """Test RecordBatch quantization without optional shape column."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create RecordBatch without shape column
        weights_data = {
            "layer_name": ["layer.0.weight"],
            "weights": [np.random.randn(1000).astype(np.float32).tolist()],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Should work and infer shape
        result_batch = quantizer.quantize_arrow_batch(batch, bit_width=4)
        
        # Verify shape inferred from weights length
        result_shape = result_batch.column("shape")[0].as_py()
        assert result_shape == [1000]

    def test_recordbatch_large_weights(self):
        """Test RecordBatch quantization with large weight tensors."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create large weights (1M elements = 4MB)
        large_weights = np.random.randn(1_000_000).astype(np.float32)
        
        weights_data = {
            "layer_name": ["large_layer"],
            "weights": [large_weights.tolist()],
            "shape": [[1_000_000]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Should handle large tensors
        result_batch = quantizer.quantize_arrow_batch(batch, bit_width=4)
        assert result_batch.num_rows == 1

    def test_recordbatch_vs_table_equivalence(self):
        """Test that quantize_arrow_batch produces same results as quantize_arrow."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create test data
        np.random.seed(42)
        weights = np.random.randn(1000).astype(np.float32)
        
        weights_data = {
            "layer_name": ["layer.0.weight"],
            "weights": [weights.tolist()],
            "shape": [[1000]],
        }
        
        # Quantize using RecordBatch API
        batch = pa.RecordBatch.from_pydict(weights_data)
        result_batch = quantizer.quantize_arrow_batch(batch, bit_width=4)
        
        # Quantize using Table API
        table = pa.Table.from_pydict(weights_data)
        result_table = quantizer.quantize_arrow(table, bit_width=4)
        
        # Convert table to batch for comparison
        result_table_batch = result_table.to_batches()[0]
        
        # Results should be equivalent
        assert result_batch.num_rows == result_table_batch.num_rows
        assert result_batch.schema.equals(result_table_batch.schema)
        
        # Compare quantized data (should be identical with same seed)
        batch_data = result_batch.column("quantized_data")[0].as_py()
        table_data = result_table_batch.column("quantized_data")[0].as_py()
        assert batch_data == table_data

    def test_recordbatch_empty_batch(self):
        """Test RecordBatch quantization with empty batch."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create empty RecordBatch
        schema = pa.schema([
            ("layer_name", pa.string()),
            ("weights", pa.list_(pa.float32())),
            ("shape", pa.list_(pa.int64())),
        ])
        batch = pa.RecordBatch.from_arrays(
            [pa.array([], type=pa.string()),
             pa.array([], type=pa.list_(pa.float32())),
             pa.array([], type=pa.list_(pa.int64()))],
            schema=schema
        )
        
        # Should handle empty batch
        result_batch = quantizer.quantize_arrow_batch(batch, bit_width=4)
        assert result_batch.num_rows == 0

    def test_recordbatch_default_bit_width(self):
        """Test RecordBatch quantization with default bit width."""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        weights_data = {
            "layer_name": ["layer.0.weight"],
            "weights": [np.random.randn(1000).astype(np.float32).tolist()],
            "shape": [[1000]],
        }
        batch = pa.RecordBatch.from_pydict(weights_data)
        
        # Quantize without specifying bit_width (should default to 4)
        result_batch = quantizer.quantize_arrow_batch(batch)
        assert result_batch.column("bit_width")[0].as_py() == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

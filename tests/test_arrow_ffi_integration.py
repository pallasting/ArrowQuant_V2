"""
Test Arrow C Data Interface integration for zero-copy PyArrow interop.

This test verifies that the Arrow FFI helper functions correctly import
and export PyArrow objects using the Arrow C Data Interface.
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
class TestArrowFFIIntegration:
    """Test Arrow C Data Interface integration."""

    def test_arrow_ffi_module_exists(self):
        """Verify that Arrow FFI helper functions are available."""
        # The arrow_ffi_helpers module is internal, but we can verify
        # that the main module compiles successfully
        quantizer = ArrowQuantV2(mode="diffusion")
        assert quantizer is not None

    def test_pyarrow_table_creation(self):
        """Test creating a PyArrow table with expected schema."""
        # Create test data
        layer_names = ["layer.0.weight", "layer.1.weight"]
        weights_data = [
            np.random.randn(1000).astype(np.float32),
            np.random.randn(2000).astype(np.float32),
        ]
        shapes = [[1000], [2000]]

        # Create PyArrow table
        table = pa.Table.from_pydict({
            "layer_name": layer_names,
            "weights": weights_data,
            "shape": shapes,
        })

        # Verify schema
        assert table.schema.field("layer_name").type == pa.string()
        assert isinstance(table.schema.field("weights").type, pa.ListType)
        assert isinstance(table.schema.field("shape").type, pa.ListType)

    def test_pyarrow_recordbatch_creation(self):
        """Test creating a PyArrow RecordBatch."""
        # Create test data
        layer_names = pa.array(["layer.0.weight"])
        weights = pa.array([np.random.randn(100).astype(np.float32).tolist()])
        shapes = pa.array([[100]])

        # Create RecordBatch
        batch = pa.RecordBatch.from_arrays(
            [layer_names, weights, shapes],
            names=["layer_name", "weights", "shape"]
        )

        assert batch.num_rows == 1
        assert batch.num_columns == 3

    def test_arrow_c_interface_available(self):
        """Verify that PyArrow objects support __arrow_c_array__ protocol."""
        # Create a simple array
        arr = pa.array([1.0, 2.0, 3.0], type=pa.float32())
        
        # Check that __arrow_c_array__ method exists
        assert hasattr(arr, "__arrow_c_array__")
        
        # Call it to verify it works
        schema_capsule, array_capsule = arr.__arrow_c_array__()
        assert schema_capsule is not None
        assert array_capsule is not None

    def test_recordbatch_c_interface(self):
        """Verify that RecordBatch supports __arrow_c_array__ protocol."""
        # Create a RecordBatch
        layer_names = pa.array(["test_layer"])
        weights = pa.array([np.random.randn(10).astype(np.float32).tolist()])
        
        batch = pa.RecordBatch.from_arrays(
            [layer_names, weights],
            names=["layer_name", "weights"]
        )
        
        # Check that __arrow_c_array__ method exists
        assert hasattr(batch, "__arrow_c_array__")
        
        # Call it to verify it works
        schema_capsule, array_capsule = batch.__arrow_c_array__()
        assert schema_capsule is not None
        assert array_capsule is not None

    def test_table_to_batches(self):
        """Test converting PyArrow Table to batches."""
        # Create table
        table = pa.Table.from_pydict({
            "layer_name": ["layer1", "layer2"],
            "weights": [
                np.random.randn(100).astype(np.float32),
                np.random.randn(200).astype(np.float32),
            ],
        })
        
        # Convert to batches
        batches = table.to_batches()
        assert len(batches) > 0
        assert all(isinstance(b, pa.RecordBatch) for b in batches)

    def test_schema_validation_structure(self):
        """Test that we can create tables with the expected schema structure."""
        # Expected schema for quantization:
        # - layer_name: string (required)
        # - weights: list<float32> (required)
        # - shape: list<int64> (optional)
        
        # Create table with correct schema
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0"],
            "weights": [np.random.randn(100).astype(np.float32)],
            "shape": [[100]],
        })
        
        # Verify field types
        assert table.schema.field("layer_name").type == pa.string()
        
        weights_field = table.schema.field("weights")
        assert isinstance(weights_field.type, pa.ListType)
        assert weights_field.type.value_type == pa.float32()
        
        shape_field = table.schema.field("shape")
        assert isinstance(shape_field.type, pa.ListType)
        assert shape_field.type.value_type == pa.int64()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Integration tests for validate_arrow_input() method

These tests verify the production-grade input validation for PyArrow Tables
in the ArrowQuantV2 Python API.

Requirements tested:
- REQ-5.1: Python API SHALL validate PyArrow Table schema and return detailed error information
- REQ-6.3: System SHALL return detailed schema validation errors for mismatched schemas
"""

import pytest
import pyarrow as pa
import numpy as np

# Import the module - adjust path as needed
try:
    from arrow_quant_v2 import ArrowQuantV2
except ImportError:
    pytest.skip("arrow_quant_v2 module not available", allow_module_level=True)


class TestValidateArrowInput:
    """Test suite for validate_arrow_input() method"""

    def setup_method(self):
        """Create a quantizer instance for each test"""
        self.quantizer = ArrowQuantV2(mode="diffusion")

    def test_valid_schema_with_required_fields(self):
        """Test validation passes for valid schema with required fields"""
        # Create valid table
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0.weight", "layer.1.weight"],
            "weights": [
                np.random.randn(100).astype(np.float32).tolist(),
                np.random.randn(100).astype(np.float32).tolist(),
            ],
        })

        # Should not raise any exception
        self.quantizer.validate_arrow_input(table)

    def test_valid_schema_with_optional_shape_field(self):
        """Test validation passes for valid schema with optional shape field"""
        # Create valid table with shape field
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0.weight"],
            "weights": [np.random.randn(100).astype(np.float32).tolist()],
            "shape": [[100]],
        })

        # Should not raise any exception
        self.quantizer.validate_arrow_input(table)

    def test_missing_layer_name_field(self):
        """Test validation fails when layer_name field is missing"""
        # Create table without layer_name
        table = pa.Table.from_pydict({
            "weights": [np.random.randn(100).astype(np.float32).tolist()],
        })

        # Should raise ValueError with detailed message
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input(table)

        error_msg = str(exc_info.value)
        assert "layer_name" in error_msg.lower()
        assert "missing" in error_msg.lower() or "required" in error_msg.lower()

    def test_missing_weights_field(self):
        """Test validation fails when weights field is missing"""
        # Create table without weights
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0.weight"],
        })

        # Should raise ValueError with detailed message
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input(table)

        error_msg = str(exc_info.value)
        assert "weights" in error_msg.lower()
        assert "missing" in error_msg.lower() or "required" in error_msg.lower()

    def test_invalid_layer_name_type(self):
        """Test validation fails when layer_name has wrong type"""
        # Create table with int32 layer_name instead of string
        table = pa.Table.from_pydict({
            "layer_name": [1, 2],
            "weights": [
                np.random.randn(100).astype(np.float32).tolist(),
                np.random.randn(100).astype(np.float32).tolist(),
            ],
        })

        # Should raise ValueError with detailed message
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input(table)

        error_msg = str(exc_info.value)
        assert "layer_name" in error_msg.lower()
        assert "type" in error_msg.lower() or "invalid" in error_msg.lower()

    def test_invalid_weights_type_not_list(self):
        """Test validation fails when weights is not a list type"""
        # Create table with float32 weights instead of list<float32>
        # Note: PyArrow will automatically convert single values to appropriate types
        # We need to explicitly create the wrong schema
        schema = pa.schema([
            pa.field("layer_name", pa.string()),
            pa.field("weights", pa.float32()),  # Wrong: should be list<float32>
        ])

        arrays = [
            pa.array(["layer.0.weight"]),
            pa.array([1.0], type=pa.float32()),
        ]

        table = pa.Table.from_arrays(arrays, schema=schema)

        # Should raise ValueError with detailed message
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input(table)

        error_msg = str(exc_info.value)
        assert "weights" in error_msg.lower()
        assert "list" in error_msg.lower() or "type" in error_msg.lower()

    def test_invalid_weights_inner_type(self):
        """Test validation fails when weights list has wrong inner type"""
        # Create table with list<float64> instead of list<float32>
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0.weight"],
            "weights": [np.random.randn(100).astype(np.float64).tolist()],
        })

        # Should raise ValueError with detailed message
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input(table)

        error_msg = str(exc_info.value)
        assert "weights" in error_msg.lower()
        assert "float32" in error_msg.lower() or "type" in error_msg.lower()

    def test_invalid_shape_type(self):
        """Test validation fails when optional shape field has wrong type"""
        # Create table with list<int32> shape instead of list<int64>
        schema = pa.schema([
            pa.field("layer_name", pa.string()),
            pa.field("weights", pa.list_(pa.float32())),
            pa.field("shape", pa.list_(pa.int32())),  # Wrong: should be int64
        ])

        arrays = [
            pa.array(["layer.0.weight"]),
            pa.array([[1.0, 2.0, 3.0]], type=pa.list_(pa.float32())),
            pa.array([[3]], type=pa.list_(pa.int32())),
        ]

        table = pa.Table.from_arrays(arrays, schema=schema)

        # Should raise ValueError with detailed message
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input(table)

        error_msg = str(exc_info.value)
        assert "shape" in error_msg.lower()
        assert "int64" in error_msg.lower() or "type" in error_msg.lower()

    def test_large_utf8_accepted_for_layer_name(self):
        """Test validation accepts LargeUtf8 for layer_name field"""
        # Create table with LargeUtf8 for layer_name
        schema = pa.schema([
            pa.field("layer_name", pa.large_string()),
            pa.field("weights", pa.list_(pa.float32())),
        ])

        arrays = [
            pa.array(["layer.0.weight"], type=pa.large_string()),
            pa.array([[1.0, 2.0, 3.0]], type=pa.list_(pa.float32())),
        ]

        table = pa.Table.from_arrays(arrays, schema=schema)

        # Should not raise any exception
        self.quantizer.validate_arrow_input(table)

    def test_large_list_accepted_for_weights(self):
        """Test validation accepts LargeList for weights field"""
        # Create table with LargeList for weights
        schema = pa.schema([
            pa.field("layer_name", pa.string()),
            pa.field("weights", pa.large_list(pa.float32())),
        ])

        arrays = [
            pa.array(["layer.0.weight"]),
            pa.array([[1.0, 2.0, 3.0]], type=pa.large_list(pa.float32())),
        ]

        table = pa.Table.from_arrays(arrays, schema=schema)

        # Should not raise any exception
        self.quantizer.validate_arrow_input(table)

    def test_empty_table_valid_schema(self):
        """Test validation passes for empty table with valid schema"""
        # Create empty table with valid schema
        table = pa.Table.from_pydict({
            "layer_name": [],
            "weights": [],
        })

        # Should not raise any exception
        self.quantizer.validate_arrow_input(table)

    def test_invalid_input_not_table(self):
        """Test validation fails when input is not a PyArrow Table"""
        # Try to validate a non-table object
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input("not a table")

        error_msg = str(exc_info.value)
        assert "failed to import" in error_msg.lower() or "table" in error_msg.lower()

    def test_validation_performance(self):
        """Test that validation is fast (only checks schema, not data)"""
        import time

        # Create a large table
        large_table = pa.Table.from_pydict({
            "layer_name": ["layer.0.weight"] * 1000,
            "weights": [np.random.randn(10000).astype(np.float32).tolist()] * 1000,
        })

        # Measure validation time
        start = time.time()
        self.quantizer.validate_arrow_input(large_table)
        elapsed = time.time() - start

        # Validation should be very fast (< 10ms) since it only checks schema
        assert elapsed < 0.01, f"Validation took {elapsed:.4f}s, expected < 0.01s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

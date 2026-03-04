"""
Unit tests for Python API input validation

This test suite provides comprehensive unit testing for the validate_arrow_input()
and validate_parameters() methods in the ArrowQuantV2 Python API. These tests
verify production-grade input validation with various invalid schema and parameter
combinations.

Requirements tested:
- REQ-5.1: Python API SHALL validate PyArrow Table schema and return detailed error information
- REQ-5.2: Python API SHALL return ValueError for invalid parameters with specific constraints
- REQ-6.3: System SHALL return detailed schema validation errors for mismatched schemas
- REQ-6.4: Time_Group_Allocator SHALL return InvalidParameterError for invalid parameters
- REQ-9.6: System SHALL ensure scale > 0.0 and zero_point ∈ [0, 255]

Test Coverage:
- Schema validation: missing fields, wrong types, invalid combinations
- Parameter validation: bit_width, num_time_groups, scale, zero_point
- Error message quality: clear, actionable, detailed
- Edge cases: boundary values, special floats (NaN, Inf)
- Performance: validation should be fast (metadata-only)
"""

import pytest
import pyarrow as pa
import numpy as np
import time

# Import the module - adjust path as needed
try:
    from arrow_quant_v2 import ArrowQuantV2
except ImportError:
    pytest.skip("arrow_quant_v2 module not available", allow_module_level=True)


class TestValidateArrowInputUnit:
    """Unit tests for validate_arrow_input() method"""

    def setup_method(self):
        """Create a quantizer instance for each test"""
        self.quantizer = ArrowQuantV2(mode="diffusion")

    # ========== Valid Schema Tests ==========

    def test_minimal_valid_schema(self):
        """Test validation passes for minimal valid schema"""
        schema = pa.schema([
            pa.field("layer_name", pa.string()),
            pa.field("weights", pa.list_(pa.float32())),
        ])
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0"],
            "weights": [np.array([1.0, 2.0, 3.0], dtype=np.float32).tolist()],
        }, schema=schema)
        self.quantizer.validate_arrow_input(table)

    def test_schema_with_optional_shape(self):
        """Test validation passes with optional shape field"""
        schema = pa.schema([
            pa.field("layer_name", pa.string()),
            pa.field("weights", pa.list_(pa.float32())),
            pa.field("shape", pa.list_(pa.int64())),
        ])
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0"],
            "weights": [np.random.randn(100).astype(np.float32).tolist()],
            "shape": [[10, 10]],
        }, schema=schema)
        self.quantizer.validate_arrow_input(table)

    def test_schema_with_large_types(self):
        """Test validation accepts LargeUtf8 and LargeList types"""
        schema = pa.schema([
            pa.field("layer_name", pa.large_string()),
            pa.field("weights", pa.large_list(pa.float32())),
        ])
        arrays = [
            pa.array(["layer.0"], type=pa.large_string()),
            pa.array([[1.0, 2.0]], type=pa.large_list(pa.float32())),
        ]
        table = pa.Table.from_arrays(arrays, schema=schema)
        self.quantizer.validate_arrow_input(table)

    # ========== Missing Field Tests ==========

    def test_missing_layer_name(self):
        """Test validation fails when layer_name is missing"""
        table = pa.Table.from_pydict({
            "weights": [np.random.randn(10).astype(np.float32).tolist()],
        })
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input(table)
        error_msg = str(exc_info.value)
        assert "layer_name" in error_msg.lower()
        assert "missing" in error_msg.lower() or "required" in error_msg.lower()

    def test_missing_weights(self):
        """Test validation fails when weights is missing"""
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0"],
        })
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input(table)
        error_msg = str(exc_info.value)
        assert "weights" in error_msg.lower()
        assert "missing" in error_msg.lower() or "required" in error_msg.lower()

    # ========== Invalid Type Tests ==========

    def test_layer_name_wrong_type(self):
        """Test validation fails when layer_name is not string type"""
        table = pa.Table.from_pydict({
            "layer_name": [1, 2, 3],
            "weights": [
                np.random.randn(10).astype(np.float32).tolist(),
                np.random.randn(10).astype(np.float32).tolist(),
                np.random.randn(10).astype(np.float32).tolist(),
            ],
        })
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input(table)
        error_msg = str(exc_info.value)
        assert "layer_name" in error_msg.lower()
        assert "type" in error_msg.lower() or "invalid" in error_msg.lower()

    def test_weights_not_list_type(self):
        """Test validation fails when weights is not a list"""
        schema = pa.schema([
            pa.field("layer_name", pa.string()),
            pa.field("weights", pa.float32()),
        ])
        arrays = [
            pa.array(["layer.0"]),
            pa.array([1.0], type=pa.float32()),
        ]
        table = pa.Table.from_arrays(arrays, schema=schema)
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input(table)
        error_msg = str(exc_info.value)
        assert "weights" in error_msg.lower()
        assert "list" in error_msg.lower() or "type" in error_msg.lower()

    def test_weights_wrong_inner_type(self):
        """Test validation fails when weights is not list<float32>"""
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0"],
            "weights": [np.random.randn(10).astype(np.float64).tolist()],
        })
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input(table)
        error_msg = str(exc_info.value)
        assert "weights" in error_msg.lower()
        assert "float32" in error_msg.lower() or "float64" in error_msg.lower()

    def test_shape_wrong_type(self):
        """Test validation fails when shape is not list<int64>"""
        schema = pa.schema([
            pa.field("layer_name", pa.string()),
            pa.field("weights", pa.list_(pa.float32())),
            pa.field("shape", pa.list_(pa.int32())),
        ])
        arrays = [
            pa.array(["layer.0"]),
            pa.array([[1.0, 2.0]], type=pa.list_(pa.float32())),
            pa.array([[10, 10]], type=pa.list_(pa.int32())),
        ]
        table = pa.Table.from_arrays(arrays, schema=schema)
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input(table)
        error_msg = str(exc_info.value)
        assert "shape" in error_msg.lower()
        assert "int64" in error_msg.lower() or "int32" in error_msg.lower()

    # ========== Invalid Input Tests ==========

    def test_invalid_input_not_table(self):
        """Test validation fails when input is not a PyArrow Table"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_arrow_input("not a table")
        error_msg = str(exc_info.value)
        assert "failed to import" in error_msg.lower() or "table" in error_msg.lower()

    # ========== Complex Schema Tests ==========

    def test_schema_with_extra_fields(self):
        """Test validation passes when extra fields are present"""
        schema = pa.schema([
            pa.field("layer_name", pa.string()),
            pa.field("weights", pa.list_(pa.float32())),
            pa.field("extra_field", pa.int64()),
        ])
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0"],
            "weights": [np.random.randn(10).astype(np.float32).tolist()],
            "extra_field": [42],
        }, schema=schema)
        # Should pass - extra fields are allowed
        self.quantizer.validate_arrow_input(table)

    # ========== Performance Tests ==========

    def test_validation_is_fast_for_large_table(self):
        """Test that validation is fast (only checks schema, not data)"""
        # Create a large table with 1000 layers and 10K weights each
        schema = pa.schema([
            pa.field("layer_name", pa.string()),
            pa.field("weights", pa.list_(pa.float32())),
        ])
        large_table = pa.Table.from_pydict({
            "layer_name": [f"layer.{i}" for i in range(1000)],
            "weights": [np.random.randn(10000).astype(np.float32).tolist() for _ in range(1000)],
        }, schema=schema)

        start = time.time()
        self.quantizer.validate_arrow_input(large_table)
        elapsed = time.time() - start

        # Validation should be very fast (< 10ms) since it only checks schema
        assert elapsed < 0.01, f"Validation took {elapsed:.4f}s, expected < 0.01s"

    def test_validation_does_not_access_data(self):
        """Test that validation only checks schema, not data values"""
        # Create table with invalid data values (NaN, Inf) but valid schema
        schema = pa.schema([
            pa.field("layer_name", pa.string()),
            pa.field("weights", pa.list_(pa.float32())),
        ])
        table = pa.Table.from_pydict({
            "layer_name": ["layer.0"],
            "weights": [np.array([float('nan'), float('inf'), -float('inf')], dtype=np.float32).tolist()],
        }, schema=schema)
        # Should pass - validation only checks schema, not data values
        self.quantizer.validate_arrow_input(table)



class TestValidateParametersUnit:
    """Unit tests for validate_parameters() method"""

    def setup_method(self):
        """Create a quantizer instance for each test"""
        self.quantizer = ArrowQuantV2(mode="diffusion")

    # ========== Valid Parameter Tests ==========

    def test_minimal_valid_parameters(self):
        """Test validation passes for minimal valid parameters"""
        self.quantizer.validate_parameters(bit_width=4, num_time_groups=10)

    def test_all_valid_bit_widths(self):
        """Test validation passes for all valid bit_width values (2, 4, 8)"""
        for bit_width in [2, 4, 8]:
            self.quantizer.validate_parameters(bit_width=bit_width, num_time_groups=10)

    def test_valid_num_time_groups_range(self):
        """Test validation passes for various num_time_groups values"""
        for num_groups in [1, 10, 100, 10000]:
            self.quantizer.validate_parameters(bit_width=4, num_time_groups=num_groups)

    def test_valid_scale_range(self):
        """Test validation passes for various scale values"""
        for scale in [0.0001, 1.0, 100.0, 10000.0]:
            self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, scale=scale)

    def test_valid_zero_point_range(self):
        """Test validation passes for zero_point boundary values"""
        for zero_point in [0.0, 127.5, 255.0]:
            self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, zero_point=zero_point)

    # ========== Invalid bit_width Tests ==========

    def test_invalid_bit_width_values(self):
        """Test validation fails for invalid bit_width values"""
        for invalid_width in [0, 1, 3, 5, 6, 7, 16, 32]:
            with pytest.raises(ValueError) as exc_info:
                self.quantizer.validate_parameters(bit_width=invalid_width, num_time_groups=10)
            error_msg = str(exc_info.value)
            assert "bit_width" in error_msg.lower()
            assert str(invalid_width) in error_msg

    # ========== Invalid num_time_groups Tests ==========

    def test_invalid_num_time_groups_0(self):
        """Test validation fails for num_time_groups=0"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(bit_width=4, num_time_groups=0)
        error_msg = str(exc_info.value)
        assert "num_time_groups" in error_msg.lower()
        assert "0" in error_msg
        assert "greater than 0" in error_msg.lower() or "must be" in error_msg.lower()

    # ========== Invalid scale Tests ==========

    def test_invalid_scale_values(self):
        """Test validation fails for invalid scale values (zero, negative, NaN, Inf)"""
        invalid_scales = [0.0, -0.1, -100.0, float('nan'), float('inf'), float('-inf')]
        for scale in invalid_scales:
            with pytest.raises(ValueError) as exc_info:
                self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, scale=scale)
            error_msg = str(exc_info.value)
            assert "scale" in error_msg.lower()

    # ========== Invalid zero_point Tests ==========

    def test_invalid_zero_point_out_of_range(self):
        """Test validation fails for zero_point outside [0, 255] range"""
        invalid_values = [-0.1, -100.0, 255.1, 1000.0]
        for zero_point in invalid_values:
            with pytest.raises(ValueError) as exc_info:
                self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, zero_point=zero_point)
            error_msg = str(exc_info.value)
            assert "zero_point" in error_msg.lower()
            assert "[0, 255]" in error_msg or "range" in error_msg.lower()

    def test_invalid_zero_point_special_floats(self):
        """Test validation fails for zero_point with NaN or Inf"""
        for zero_point in [float('nan'), float('inf'), float('-inf')]:
            with pytest.raises(ValueError) as exc_info:
                self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, zero_point=zero_point)
            error_msg = str(exc_info.value)
            assert "zero_point" in error_msg.lower()

    # ========== Multiple Invalid Parameters Tests ==========

    def test_multiple_invalid_parameters(self):
        """Test validation reports first invalid parameter when multiple are invalid"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(bit_width=3, num_time_groups=0)
        error_msg = str(exc_info.value)
        # Should report the first validation error (bit_width)
        assert "bit_width" in error_msg.lower()

    # ========== Edge Cases ==========

    def test_edge_case_extreme_scale_values(self):
        """Test validation passes for extreme but valid scale values"""
        for scale in [1e-10, 1e10]:
            self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, scale=scale)

    def test_edge_case_zero_point_boundaries(self):
        """Test validation at zero_point boundaries"""
        # Valid boundaries
        self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, zero_point=0.0)
        self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, zero_point=255.0)
        self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, zero_point=254.999)
        self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, zero_point=0.001)

    # ========== Error Message Quality Tests ==========

    def test_error_message_quality_bit_width(self):
        """Test that error messages for bit_width are clear and actionable"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(bit_width=7, num_time_groups=10)
        error_msg = str(exc_info.value)
        # Should contain parameter name, invalid value, and valid values
        assert "bit_width" in error_msg.lower()
        assert "7" in error_msg
        assert "2" in error_msg and "4" in error_msg and "8" in error_msg

    def test_error_message_quality_scale(self):
        """Test that error messages for scale are clear"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, scale=0.0)
        error_msg = str(exc_info.value)
        assert "scale" in error_msg.lower()
        assert "0" in error_msg or "0.0" in error_msg
        assert "greater than 0" in error_msg.lower() or "> 0" in error_msg

    def test_error_message_quality_zero_point(self):
        """Test that error messages for zero_point are clear"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, zero_point=300.0)
        error_msg = str(exc_info.value)
        assert "zero_point" in error_msg.lower()
        assert "300" in error_msg
        assert "[0, 255]" in error_msg or ("0" in error_msg and "255" in error_msg)

    # ========== Optional Parameter Tests ==========

    def test_optional_parameters(self):
        """Test validation passes when optional parameters are not provided"""
        # Both optional parameters not provided
        self.quantizer.validate_parameters(bit_width=4, num_time_groups=10)
        # Only scale provided
        self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, scale=1.0)
        # Only zero_point provided
        self.quantizer.validate_parameters(bit_width=4, num_time_groups=10, zero_point=128.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

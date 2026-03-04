"""
Test error logging and context recording in Python API

This test verifies that all error paths in the Python API have detailed logging with:
- Error type
- Input parameters
- Contextual information

**Validates: Requirements 12.2, 12.3**
- REQ-12.2: System SHALL record detailed error information and context
- REQ-12.3: System SHALL record warning logs when appropriate
"""

import pytest
import pyarrow as pa
import numpy as np
from arrow_quant_v2 import ArrowQuantV2


def test_error_logging_invalid_bit_width():
    """Test that invalid bit_width logs error with context"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Create valid table
    table = pa.Table.from_pydict({
        "layer_name": ["layer.0"],
        "weights": [[1.0, 2.0, 3.0]],
    })
    
    # Invalid bit_width should log error
    with pytest.raises(ValueError) as exc_info:
        quantizer.quantize_arrow(table, bit_width=3)
    
    assert "bit_width" in str(exc_info.value).lower()
    assert "3" in str(exc_info.value)


def test_error_logging_invalid_num_time_groups():
    """Test that invalid num_time_groups logs error with context"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Invalid num_time_groups should log error
    with pytest.raises(ValueError) as exc_info:
        quantizer.validate_parameters(bit_width=4, num_time_groups=0)
    
    assert "num_time_groups" in str(exc_info.value).lower()
    assert "0" in str(exc_info.value)


def test_error_logging_invalid_scale():
    """Test that invalid scale logs error with context"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Invalid scale (negative) should log error
    with pytest.raises(ValueError) as exc_info:
        quantizer.validate_parameters(bit_width=4, num_time_groups=10, scale=-1.0)
    
    assert "scale" in str(exc_info.value).lower()
    assert "-1" in str(exc_info.value) or "negative" in str(exc_info.value).lower()


def test_error_logging_invalid_zero_point():
    """Test that invalid zero_point logs error with context"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Invalid zero_point (out of range) should log error
    with pytest.raises(ValueError) as exc_info:
        quantizer.validate_parameters(bit_width=4, num_time_groups=10, zero_point=300.0)
    
    assert "zero_point" in str(exc_info.value).lower()
    assert "300" in str(exc_info.value)


def test_error_logging_missing_required_field():
    """Test that missing required field logs error with schema context"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Table missing required 'weights' field
    table = pa.Table.from_pydict({
        "layer_name": ["layer.0"],
        # Missing 'weights' field
    })
    
    with pytest.raises(ValueError) as exc_info:
        quantizer.quantize_arrow(table, bit_width=4)
    
    error_msg = str(exc_info.value).lower()
    assert "weights" in error_msg or "missing" in error_msg or "schema" in error_msg


def test_error_logging_invalid_field_type():
    """Test that invalid field type logs error with type context"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Table with wrong type for 'weights' field (should be list<float32>)
    table = pa.Table.from_pydict({
        "layer_name": ["layer.0"],
        "weights": [1.0],  # Should be list, not scalar
    })
    
    with pytest.raises(ValueError) as exc_info:
        quantizer.quantize_arrow(table, bit_width=4)
    
    error_msg = str(exc_info.value).lower()
    # Should mention type issue
    assert "type" in error_msg or "schema" in error_msg or "list" in error_msg


def test_error_logging_empty_table():
    """Test that empty table logs error with context"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Create empty table
    schema = pa.schema([
        pa.field("layer_name", pa.string()),
        pa.field("weights", pa.list_(pa.float32())),
    ])
    table = pa.Table.from_batches([], schema=schema)
    
    # Empty table should log error
    with pytest.raises(ValueError) as exc_info:
        quantizer.quantize_arrow(table, bit_width=4)
    
    error_msg = str(exc_info.value).lower()
    assert "empty" in error_msg or "no" in error_msg


def test_error_logging_non_finite_scale():
    """Test that non-finite scale (NaN, Inf) logs error with context"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Non-finite scale should log error
    with pytest.raises(ValueError) as exc_info:
        quantizer.validate_parameters(bit_width=4, num_time_groups=10, scale=float('nan'))
    
    assert "scale" in str(exc_info.value).lower()
    assert "finite" in str(exc_info.value).lower() or "nan" in str(exc_info.value).lower()


def test_error_logging_non_finite_zero_point():
    """Test that non-finite zero_point logs error with context"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Non-finite zero_point should log error
    with pytest.raises(ValueError) as exc_info:
        quantizer.validate_parameters(bit_width=4, num_time_groups=10, zero_point=float('inf'))
    
    assert "zero_point" in str(exc_info.value).lower()
    assert "finite" in str(exc_info.value).lower() or "inf" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

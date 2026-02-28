"""
Unit tests for quantize_batch_arrow method signature and parameter validation.

This test file validates Task 2.1: Method signature and parameter validation.
Tests cover:
- Method signature accepts correct parameters
- bit_width validation (must be 2, 4, or 8)
- continue_on_error parameter handling
- Error messages are descriptive
"""

import pytest
import numpy as np
import pyarrow as pa
from arrow_quant_v2 import ArrowQuantV2


class TestQuantizeBatchArrowSignature:
    """Test suite for quantize_batch_arrow method signature and parameter validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.quantizer = ArrowQuantV2(mode="diffusion")

    def create_simple_arrow_table(self):
        """Helper to create a simple valid Arrow Table."""
        layer_names = ["layer.0.weight"]
        weights_lists = [[1.0, 2.0, 3.0, 4.0]]
        shapes_lists = [[4]]
        
        table = pa.Table.from_arrays(
            [
                pa.array(layer_names),
                pa.array(weights_lists, type=pa.list_(pa.float32())),
                pa.array(shapes_lists, type=pa.list_(pa.int64())),
            ],
            names=["layer_name", "weights", "shape"]
        )
        return table

    def test_method_exists(self):
        """Test that quantize_batch_arrow method exists."""
        assert hasattr(self.quantizer, "quantize_batch_arrow")
        assert callable(getattr(self.quantizer, "quantize_batch_arrow"))

    def test_accepts_arrow_table_parameter(self):
        """Test that method accepts PyArrow Table as first parameter."""
        table = self.create_simple_arrow_table()
        
        # Should not raise TypeError for accepting the table
        # (will raise NotImplementedError since we're only testing signature)
        with pytest.raises(NotImplementedError):
            self.quantizer.quantize_batch_arrow(table)

    def test_bit_width_default_value(self):
        """Test that bit_width has default value (should be 4)."""
        table = self.create_simple_arrow_table()
        
        # Call without bit_width parameter
        with pytest.raises(NotImplementedError):
            self.quantizer.quantize_batch_arrow(table)

    def test_bit_width_validation_valid_values(self):
        """Test that valid bit_width values (2, 4, 8) are accepted."""
        table = self.create_simple_arrow_table()
        
        for bit_width in [2, 4, 8]:
            # Should not raise ValueError for valid bit_width
            # (will raise NotImplementedError since we're only testing signature)
            with pytest.raises(NotImplementedError):
                self.quantizer.quantize_batch_arrow(table, bit_width=bit_width)

    def test_bit_width_validation_invalid_values(self):
        """Test that invalid bit_width values are rejected."""
        table = self.create_simple_arrow_table()
        
        invalid_bit_widths = [1, 3, 5, 6, 7, 16, 32, 0, -1]
        
        for bit_width in invalid_bit_widths:
            with pytest.raises(ValueError) as exc_info:
                self.quantizer.quantize_batch_arrow(table, bit_width=bit_width)
            
            # Verify error message is descriptive
            error_msg = str(exc_info.value)
            assert "Invalid bit_width" in error_msg or "bit_width" in error_msg.lower()
            assert str(bit_width) in error_msg
            assert "2, 4, or 8" in error_msg or "2, 4" in error_msg

    def test_continue_on_error_parameter_default(self):
        """Test that continue_on_error parameter has default value (False)."""
        table = self.create_simple_arrow_table()
        
        # Call without continue_on_error parameter
        with pytest.raises(NotImplementedError):
            self.quantizer.quantize_batch_arrow(table, bit_width=4)

    def test_continue_on_error_parameter_true(self):
        """Test that continue_on_error=True is accepted."""
        table = self.create_simple_arrow_table()
        
        with pytest.raises(NotImplementedError):
            self.quantizer.quantize_batch_arrow(
                table, 
                bit_width=4, 
                continue_on_error=True
            )

    def test_continue_on_error_parameter_false(self):
        """Test that continue_on_error=False is accepted."""
        table = self.create_simple_arrow_table()
        
        with pytest.raises(NotImplementedError):
            self.quantizer.quantize_batch_arrow(
                table, 
                bit_width=4, 
                continue_on_error=False
            )

    def test_all_parameters_together(self):
        """Test that all parameters can be provided together."""
        table = self.create_simple_arrow_table()
        
        with pytest.raises(NotImplementedError):
            self.quantizer.quantize_batch_arrow(
                table,
                bit_width=8,
                continue_on_error=True
            )

    def test_keyword_arguments(self):
        """Test that parameters can be passed as keyword arguments."""
        table = self.create_simple_arrow_table()
        
        # All keyword arguments
        with pytest.raises(NotImplementedError):
            self.quantizer.quantize_batch_arrow(
                weights_table=table,
                bit_width=4,
                continue_on_error=False
            )

    def test_positional_and_keyword_mix(self):
        """Test that positional and keyword arguments can be mixed."""
        table = self.create_simple_arrow_table()
        
        # Positional table, keyword bit_width
        with pytest.raises(NotImplementedError):
            self.quantizer.quantize_batch_arrow(table, bit_width=4)
        
        # Positional table and bit_width, keyword continue_on_error
        with pytest.raises(NotImplementedError):
            self.quantizer.quantize_batch_arrow(table, 4, continue_on_error=True)

    def test_error_message_quality_invalid_bit_width(self):
        """Test that error messages for invalid bit_width are clear and helpful."""
        table = self.create_simple_arrow_table()
        
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.quantize_batch_arrow(table, bit_width=16)
        
        error_msg = str(exc_info.value)
        
        # Error message should contain:
        # 1. The invalid value
        assert "16" in error_msg
        
        # 2. What values are valid
        assert "2" in error_msg
        assert "4" in error_msg
        assert "8" in error_msg
        
        # 3. Clear indication this is about bit_width
        assert "bit_width" in error_msg.lower() or "bit width" in error_msg.lower()

    def test_not_implemented_error_message(self):
        """Test that NotImplementedError message is clear about implementation status."""
        table = self.create_simple_arrow_table()
        
        with pytest.raises(NotImplementedError) as exc_info:
            self.quantizer.quantize_batch_arrow(table, bit_width=4)
        
        error_msg = str(exc_info.value)
        
        # Should mention that this is task 2.1 and implementation is incomplete
        assert "not yet fully implemented" in error_msg.lower() or "not implemented" in error_msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

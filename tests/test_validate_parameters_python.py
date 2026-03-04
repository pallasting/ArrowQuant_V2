"""
Integration tests for validate_parameters() method

These tests verify the production-grade parameter validation for quantization
parameters in the ArrowQuantV2 Python API.

Requirements tested:
- REQ-5.2: Python API SHALL return ValueError for invalid parameters with specific constraints
- REQ-6.4: Time_Group_Allocator SHALL return InvalidParameterError for invalid parameters
- REQ-9.6: System SHALL ensure scale > 0.0 and zero_point ∈ [0, 255]
"""

import pytest
import math

# Import the module - adjust path as needed
try:
    from arrow_quant_v2 import ArrowQuantV2
except ImportError:
    pytest.skip("arrow_quant_v2 module not available", allow_module_level=True)


class TestValidateParameters:
    """Test suite for validate_parameters() method"""

    def setup_method(self):
        """Create a quantizer instance for each test"""
        self.quantizer = ArrowQuantV2(mode="diffusion")

    # ========== Valid Parameter Tests ==========

    def test_valid_parameters_minimal(self):
        """Test validation passes for valid minimal parameters"""
        # Should not raise any exception
        self.quantizer.validate_parameters(bit_width=4, num_time_groups=10)

    def test_valid_parameters_with_scale(self):
        """Test validation passes for valid parameters with scale"""
        # Should not raise any exception
        self.quantizer.validate_parameters(
            bit_width=4, num_time_groups=10, scale=0.5
        )

    def test_valid_parameters_with_zero_point(self):
        """Test validation passes for valid parameters with zero_point"""
        # Should not raise any exception
        self.quantizer.validate_parameters(
            bit_width=4, num_time_groups=10, zero_point=128.0
        )

    def test_valid_parameters_complete(self):
        """Test validation passes for all valid parameters"""
        # Should not raise any exception
        self.quantizer.validate_parameters(
            bit_width=8, num_time_groups=20, scale=1.5, zero_point=64.0
        )

    def test_valid_bit_width_2(self):
        """Test validation passes for bit_width=2"""
        self.quantizer.validate_parameters(bit_width=2, num_time_groups=10)

    def test_valid_bit_width_4(self):
        """Test validation passes for bit_width=4"""
        self.quantizer.validate_parameters(bit_width=4, num_time_groups=10)

    def test_valid_bit_width_8(self):
        """Test validation passes for bit_width=8"""
        self.quantizer.validate_parameters(bit_width=8, num_time_groups=10)

    def test_valid_num_time_groups_1(self):
        """Test validation passes for num_time_groups=1 (minimum valid)"""
        self.quantizer.validate_parameters(bit_width=4, num_time_groups=1)

    def test_valid_num_time_groups_large(self):
        """Test validation passes for large num_time_groups"""
        self.quantizer.validate_parameters(bit_width=4, num_time_groups=1000)

    def test_valid_scale_small(self):
        """Test validation passes for small positive scale"""
        self.quantizer.validate_parameters(
            bit_width=4, num_time_groups=10, scale=0.001
        )

    def test_valid_scale_large(self):
        """Test validation passes for large scale"""
        self.quantizer.validate_parameters(
            bit_width=4, num_time_groups=10, scale=1000.0
        )

    def test_valid_zero_point_min(self):
        """Test validation passes for zero_point=0 (minimum)"""
        self.quantizer.validate_parameters(
            bit_width=4, num_time_groups=10, zero_point=0.0
        )

    def test_valid_zero_point_max(self):
        """Test validation passes for zero_point=255 (maximum)"""
        self.quantizer.validate_parameters(
            bit_width=4, num_time_groups=10, zero_point=255.0
        )

    def test_valid_zero_point_mid(self):
        """Test validation passes for zero_point in middle of range"""
        self.quantizer.validate_parameters(
            bit_width=4, num_time_groups=10, zero_point=127.5
        )

    # ========== Invalid bit_width Tests ==========

    def test_invalid_bit_width_0(self):
        """Test validation fails for bit_width=0"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(bit_width=0, num_time_groups=10)

        error_msg = str(exc_info.value)
        assert "bit_width" in error_msg.lower()
        assert "0" in error_msg
        assert "2, 4, or 8" in error_msg or "must be" in error_msg.lower()

    def test_invalid_bit_width_1(self):
        """Test validation fails for bit_width=1"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(bit_width=1, num_time_groups=10)

        error_msg = str(exc_info.value)
        assert "bit_width" in error_msg.lower()
        assert "1" in error_msg

    def test_invalid_bit_width_3(self):
        """Test validation fails for bit_width=3"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(bit_width=3, num_time_groups=10)

        error_msg = str(exc_info.value)
        assert "bit_width" in error_msg.lower()
        assert "3" in error_msg

    def test_invalid_bit_width_5(self):
        """Test validation fails for bit_width=5"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(bit_width=5, num_time_groups=10)

        error_msg = str(exc_info.value)
        assert "bit_width" in error_msg.lower()
        assert "5" in error_msg

    def test_invalid_bit_width_16(self):
        """Test validation fails for bit_width=16"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(bit_width=16, num_time_groups=10)

        error_msg = str(exc_info.value)
        assert "bit_width" in error_msg.lower()
        assert "16" in error_msg

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

    def test_invalid_scale_zero(self):
        """Test validation fails for scale=0.0"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(
                bit_width=4, num_time_groups=10, scale=0.0
            )

        error_msg = str(exc_info.value)
        assert "scale" in error_msg.lower()
        assert "0" in error_msg or "0.0" in error_msg
        assert "greater than 0" in error_msg.lower() or "must be" in error_msg.lower()

    def test_invalid_scale_negative(self):
        """Test validation fails for negative scale"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(
                bit_width=4, num_time_groups=10, scale=-1.0
            )

        error_msg = str(exc_info.value)
        assert "scale" in error_msg.lower()
        assert "-1" in error_msg or "negative" in error_msg.lower()
        assert "greater than 0" in error_msg.lower() or "must be" in error_msg.lower()

    def test_invalid_scale_nan(self):
        """Test validation fails for scale=NaN"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(
                bit_width=4, num_time_groups=10, scale=float('nan')
            )

        error_msg = str(exc_info.value)
        assert "scale" in error_msg.lower()
        assert "nan" in error_msg.lower() or "finite" in error_msg.lower()

    def test_invalid_scale_inf(self):
        """Test validation fails for scale=Infinity"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(
                bit_width=4, num_time_groups=10, scale=float('inf')
            )

        error_msg = str(exc_info.value)
        assert "scale" in error_msg.lower()
        assert "inf" in error_msg.lower() or "finite" in error_msg.lower()

    def test_invalid_scale_neg_inf(self):
        """Test validation fails for scale=-Infinity"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(
                bit_width=4, num_time_groups=10, scale=float('-inf')
            )

        error_msg = str(exc_info.value)
        assert "scale" in error_msg.lower()
        # Should fail for either being negative or non-finite
        assert "inf" in error_msg.lower() or "finite" in error_msg.lower() or "greater than 0" in error_msg.lower()

    # ========== Invalid zero_point Tests ==========

    def test_invalid_zero_point_negative(self):
        """Test validation fails for negative zero_point"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(
                bit_width=4, num_time_groups=10, zero_point=-1.0
            )

        error_msg = str(exc_info.value)
        assert "zero_point" in error_msg.lower()
        assert "-1" in error_msg or "negative" in error_msg.lower()
        assert "[0, 255]" in error_msg or "range" in error_msg.lower()

    def test_invalid_zero_point_above_max(self):
        """Test validation fails for zero_point > 255"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(
                bit_width=4, num_time_groups=10, zero_point=256.0
            )

        error_msg = str(exc_info.value)
        assert "zero_point" in error_msg.lower()
        assert "256" in error_msg
        assert "[0, 255]" in error_msg or "range" in error_msg.lower()

    def test_invalid_zero_point_large(self):
        """Test validation fails for very large zero_point"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(
                bit_width=4, num_time_groups=10, zero_point=1000.0
            )

        error_msg = str(exc_info.value)
        assert "zero_point" in error_msg.lower()
        assert "1000" in error_msg
        assert "[0, 255]" in error_msg or "range" in error_msg.lower()

    def test_invalid_zero_point_nan(self):
        """Test validation fails for zero_point=NaN"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(
                bit_width=4, num_time_groups=10, zero_point=float('nan')
            )

        error_msg = str(exc_info.value)
        assert "zero_point" in error_msg.lower()
        assert "nan" in error_msg.lower() or "finite" in error_msg.lower()

    def test_invalid_zero_point_inf(self):
        """Test validation fails for zero_point=Infinity"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(
                bit_width=4, num_time_groups=10, zero_point=float('inf')
            )

        error_msg = str(exc_info.value)
        assert "zero_point" in error_msg.lower()
        assert "inf" in error_msg.lower() or "finite" in error_msg.lower()

    # ========== Multiple Invalid Parameters Tests ==========

    def test_multiple_invalid_parameters_bit_width_and_num_time_groups(self):
        """Test validation fails for multiple invalid parameters (reports first error)"""
        # Both bit_width and num_time_groups are invalid
        # Should report the first validation error (bit_width)
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(bit_width=3, num_time_groups=0)

        error_msg = str(exc_info.value)
        # Should report bit_width error first
        assert "bit_width" in error_msg.lower()

    def test_multiple_invalid_parameters_scale_and_zero_point(self):
        """Test validation fails when both scale and zero_point are invalid"""
        # Both scale and zero_point are invalid
        # Should report the first validation error (scale)
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(
                bit_width=4, num_time_groups=10, scale=0.0, zero_point=300.0
            )

        error_msg = str(exc_info.value)
        # Should report scale error first
        assert "scale" in error_msg.lower()

    # ========== Edge Cases ==========

    def test_edge_case_very_small_scale(self):
        """Test validation passes for very small but positive scale"""
        # Should not raise - any positive finite value is valid
        self.quantizer.validate_parameters(
            bit_width=4, num_time_groups=10, scale=1e-10
        )

    def test_edge_case_very_large_scale(self):
        """Test validation passes for very large scale"""
        # Should not raise - any positive finite value is valid
        self.quantizer.validate_parameters(
            bit_width=4, num_time_groups=10, scale=1e10
        )

    def test_edge_case_fractional_zero_point(self):
        """Test validation passes for fractional zero_point within range"""
        # Should not raise - fractional values in [0, 255] are valid
        self.quantizer.validate_parameters(
            bit_width=4, num_time_groups=10, zero_point=127.5
        )

    def test_error_message_quality_bit_width(self):
        """Test that error messages for bit_width are clear and actionable"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(bit_width=7, num_time_groups=10)

        error_msg = str(exc_info.value)
        # Error message should contain:
        # 1. The parameter name
        assert "bit_width" in error_msg.lower()
        # 2. The invalid value
        assert "7" in error_msg
        # 3. The valid values
        assert "2" in error_msg and "4" in error_msg and "8" in error_msg

    def test_error_message_quality_scale(self):
        """Test that error messages for scale are clear and actionable"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(
                bit_width=4, num_time_groups=10, scale=-0.5
            )

        error_msg = str(exc_info.value)
        # Error message should contain:
        # 1. The parameter name
        assert "scale" in error_msg.lower()
        # 2. The invalid value
        assert "-0.5" in error_msg or "0.5" in error_msg
        # 3. The constraint
        assert "greater than 0" in error_msg.lower() or "> 0" in error_msg

    def test_error_message_quality_zero_point(self):
        """Test that error messages for zero_point are clear and actionable"""
        with pytest.raises(ValueError) as exc_info:
            self.quantizer.validate_parameters(
                bit_width=4, num_time_groups=10, zero_point=300.0
            )

        error_msg = str(exc_info.value)
        # Error message should contain:
        # 1. The parameter name
        assert "zero_point" in error_msg.lower()
        # 2. The invalid value
        assert "300" in error_msg
        # 3. The valid range
        assert "[0, 255]" in error_msg or ("0" in error_msg and "255" in error_msg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

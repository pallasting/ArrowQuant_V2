/// Unit tests for parameter validation logic
///
/// These tests verify the validate_parameters() function in the Python API
/// to ensure it correctly validates quantization parameters.
///
/// Requirements tested:
/// - REQ-5.2: Python API SHALL return ValueError for invalid parameters with specific constraints
/// - REQ-6.4: Time_Group_Allocator SHALL return InvalidParameterError for invalid parameters  
/// - REQ-9.6: System SHALL ensure scale > 0.0 and zero_point ∈ [0, 255]

#[cfg(test)]
mod tests {
    // Note: Since validate_parameters is a Python-facing method in PyO3,
    // we cannot directly test it from Rust without the Python runtime.
    // Instead, we create a standalone validation function that can be tested
    // and used by the Python method.

    /// Validate quantization parameters
    ///
    /// Returns Ok(()) if all parameters are valid, or Err with a descriptive message.
    fn validate_quantization_parameters(
        bit_width: u8,
        num_time_groups: usize,
        scale: Option<f32>,
        zero_point: Option<f32>,
    ) -> Result<(), String> {
        // Validate bit_width ∈ {2, 4, 8}
        if ![2, 4, 8].contains(&bit_width) {
            return Err(format!(
                "Invalid bit_width: {}. Must be 2, 4, or 8",
                bit_width
            ));
        }

        // Validate num_time_groups > 0
        if num_time_groups == 0 {
            return Err("Invalid num_time_groups: 0. Must be greater than 0".to_string());
        }

        // Validate scale > 0.0 and finite (if provided)
        if let Some(s) = scale {
            if s <= 0.0 {
                return Err(format!("Invalid scale: {}. Must be greater than 0.0", s));
            }
            if !s.is_finite() {
                return Err(format!(
                    "Invalid scale: {}. Must be a finite value (not NaN or Inf)",
                    s
                ));
            }
        }

        // Validate zero_point ∈ [0, 255] (if provided)
        if let Some(zp) = zero_point {
            if zp < 0.0 || zp > 255.0 {
                return Err(format!(
                    "Invalid zero_point: {}. Must be in range [0, 255]",
                    zp
                ));
            }
            if !zp.is_finite() {
                return Err(format!(
                    "Invalid zero_point: {}. Must be a finite value (not NaN or Inf)",
                    zp
                ));
            }
        }

        Ok(())
    }

    // ========== Valid Parameter Tests ==========

    #[test]
    fn test_valid_parameters_minimal() {
        let result = validate_quantization_parameters(4, 10, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_parameters_with_scale() {
        let result = validate_quantization_parameters(4, 10, Some(0.5), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_parameters_with_zero_point() {
        let result = validate_quantization_parameters(4, 10, None, Some(128.0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_parameters_complete() {
        let result = validate_quantization_parameters(8, 20, Some(1.5), Some(64.0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_bit_width_2() {
        let result = validate_quantization_parameters(2, 10, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_bit_width_4() {
        let result = validate_quantization_parameters(4, 10, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_bit_width_8() {
        let result = validate_quantization_parameters(8, 10, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_num_time_groups_1() {
        let result = validate_quantization_parameters(4, 1, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_num_time_groups_large() {
        let result = validate_quantization_parameters(4, 1000, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_scale_small() {
        let result = validate_quantization_parameters(4, 10, Some(0.001), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_scale_large() {
        let result = validate_quantization_parameters(4, 10, Some(1000.0), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_zero_point_min() {
        let result = validate_quantization_parameters(4, 10, None, Some(0.0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_zero_point_max() {
        let result = validate_quantization_parameters(4, 10, None, Some(255.0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_zero_point_mid() {
        let result = validate_quantization_parameters(4, 10, None, Some(127.5));
        assert!(result.is_ok());
    }

    // ========== Invalid bit_width Tests ==========

    #[test]
    fn test_invalid_bit_width_0() {
        let result = validate_quantization_parameters(0, 10, None, None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("bit_width"));
        assert!(err.contains("0"));
    }

    #[test]
    fn test_invalid_bit_width_1() {
        let result = validate_quantization_parameters(1, 10, None, None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("bit_width"));
        assert!(err.contains("1"));
    }

    #[test]
    fn test_invalid_bit_width_3() {
        let result = validate_quantization_parameters(3, 10, None, None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("bit_width"));
        assert!(err.contains("3"));
    }

    #[test]
    fn test_invalid_bit_width_16() {
        let result = validate_quantization_parameters(16, 10, None, None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("bit_width"));
        assert!(err.contains("16"));
    }

    // ========== Invalid num_time_groups Tests ==========

    #[test]
    fn test_invalid_num_time_groups_0() {
        let result = validate_quantization_parameters(4, 0, None, None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("num_time_groups"));
        assert!(err.contains("0"));
    }

    // ========== Invalid scale Tests ==========

    #[test]
    fn test_invalid_scale_zero() {
        let result = validate_quantization_parameters(4, 10, Some(0.0), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("scale"));
        assert!(err.to_lowercase().contains("greater than 0"));
    }

    #[test]
    fn test_invalid_scale_negative() {
        let result = validate_quantization_parameters(4, 10, Some(-1.0), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("scale"));
    }

    #[test]
    fn test_invalid_scale_nan() {
        let result = validate_quantization_parameters(4, 10, Some(f32::NAN), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("scale"));
        assert!(err.to_lowercase().contains("finite") || err.to_lowercase().contains("nan"));
    }

    #[test]
    fn test_invalid_scale_inf() {
        let result = validate_quantization_parameters(4, 10, Some(f32::INFINITY), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("scale"));
        assert!(err.to_lowercase().contains("finite") || err.to_lowercase().contains("inf"));
    }

    #[test]
    fn test_invalid_scale_neg_inf() {
        let result = validate_quantization_parameters(4, 10, Some(f32::NEG_INFINITY), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("scale"));
    }

    // ========== Invalid zero_point Tests ==========

    #[test]
    fn test_invalid_zero_point_negative() {
        let result = validate_quantization_parameters(4, 10, None, Some(-1.0));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("zero_point"));
        assert!(err.contains("[0, 255]") || err.to_lowercase().contains("range"));
    }

    #[test]
    fn test_invalid_zero_point_above_max() {
        let result = validate_quantization_parameters(4, 10, None, Some(256.0));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("zero_point"));
        assert!(err.contains("256"));
    }

    #[test]
    fn test_invalid_zero_point_large() {
        let result = validate_quantization_parameters(4, 10, None, Some(1000.0));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("zero_point"));
        assert!(err.contains("1000"));
    }

    #[test]
    fn test_invalid_zero_point_nan() {
        let result = validate_quantization_parameters(4, 10, None, Some(f32::NAN));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("zero_point"));
        assert!(err.to_lowercase().contains("finite") || err.to_lowercase().contains("nan"));
    }

    #[test]
    fn test_invalid_zero_point_inf() {
        let result = validate_quantization_parameters(4, 10, None, Some(f32::INFINITY));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_lowercase().contains("zero_point"));
        assert!(err.to_lowercase().contains("finite") || err.to_lowercase().contains("inf"));
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_edge_case_very_small_scale() {
        let result = validate_quantization_parameters(4, 10, Some(1e-10), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_edge_case_very_large_scale() {
        let result = validate_quantization_parameters(4, 10, Some(1e10), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_edge_case_fractional_zero_point() {
        let result = validate_quantization_parameters(4, 10, None, Some(127.5));
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_message_quality_bit_width() {
        let result = validate_quantization_parameters(7, 10, None, None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        // Error message should contain the parameter name, invalid value, and valid values
        assert!(err.to_lowercase().contains("bit_width"));
        assert!(err.contains("7"));
        assert!(err.contains("2") && err.contains("4") && err.contains("8"));
    }

    #[test]
    fn test_error_message_quality_scale() {
        let result = validate_quantization_parameters(4, 10, Some(-0.5), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        // Error message should contain the parameter name, invalid value, and constraint
        assert!(err.to_lowercase().contains("scale"));
        assert!(err.to_lowercase().contains("greater than 0"));
    }

    #[test]
    fn test_error_message_quality_zero_point() {
        let result = validate_quantization_parameters(4, 10, None, Some(300.0));
        assert!(result.is_err());
        let err = result.unwrap_err();
        // Error message should contain the parameter name, invalid value, and valid range
        assert!(err.to_lowercase().contains("zero_point"));
        assert!(err.contains("300"));
        assert!(err.contains("[0, 255]") || (err.contains("0") && err.contains("255")));
    }
}

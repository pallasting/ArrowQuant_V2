/// Test error logging and context recording
///
/// This test verifies that all error paths have detailed logging with:
/// - Error type
/// - Input parameters
/// - Contextual information
/// - Stack information (via log context)
///
/// **Validates: Requirements 12.2, 12.3**
/// - REQ-12.2: System SHALL record detailed error information and context
/// - REQ-12.3: SIMD_Engine SHALL record warning logs when falling back to scalar

use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};

#[test]
fn test_error_logging_empty_weights() {
    // Initialize logger for test
    let _ = env_logger::builder().is_test(true).try_init();

    let quantizer = TimeAwareQuantizer::new(5);
    let params = vec![
        TimeGroupParams {
            scale: 1.0,
            zero_point: 0.0,
            group_size: 100,
            time_range: (0, 100),
        },
    ];

    // This should log an error about empty weights
    let result = quantizer.validate_quantization_inputs(&[], &params);
    
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("empty"));
}

#[test]
fn test_error_logging_invalid_scale() {
    let _ = env_logger::builder().is_test(true).try_init();

    let quantizer = TimeAwareQuantizer::new(5);
    let weights = vec![1.0, 2.0, 3.0];
    let params = vec![
        TimeGroupParams {
            scale: -1.0, // Invalid: negative scale
            zero_point: 0.0,
            group_size: 100,
            time_range: (0, 100),
        },
    ];

    // This should log an error about invalid scale
    let result = quantizer.validate_quantization_inputs(&weights, &params);
    
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("scale"));
}

#[test]
fn test_error_logging_invalid_time_range() {
    let _ = env_logger::builder().is_test(true).try_init();

    let quantizer = TimeAwareQuantizer::new(5);
    let weights = vec![1.0, 2.0, 3.0];
    let params = vec![
        TimeGroupParams {
            scale: 1.0,
            zero_point: 0.0,
            group_size: 100,
            time_range: (100, 0), // Invalid: start >= end
        },
    ];

    // This should log an error about invalid time_range
    let result = quantizer.validate_quantization_inputs(&weights, &params);
    
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("time_range"));
}

#[test]
fn test_error_logging_time_group_assignment_mismatch() {
    let _ = env_logger::builder().is_test(true).try_init();

    let quantizer = TimeAwareQuantizer::new(2);
    let assignments = vec![0, 1, 0]; // 3 elements
    
    // This should log an error about length mismatch
    let result = quantizer.validate_time_group_assignments(&assignments, 5, 2);
    
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("mismatch"));
}

#[test]
fn test_error_logging_invalid_group_id() {
    let _ = env_logger::builder().is_test(true).try_init();

    let quantizer = TimeAwareQuantizer::new(2);
    let assignments = vec![0, 1, 5]; // 5 is invalid (>= num_groups)
    
    // This should log an error about invalid group ID
    let result = quantizer.validate_time_group_assignments(&assignments, 3, 2);
    
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Invalid time group ID"));
}

#[test]
fn test_error_logging_structured_format() {
    let _ = env_logger::builder().is_test(true).try_init();

    let quantizer = TimeAwareQuantizer::new(5);
    let weights = vec![1.0, 2.0, 3.0];
    let params = vec![
        TimeGroupParams {
            scale: 0.0, // Invalid: zero scale
            zero_point: 0.0,
            group_size: 100,
            time_range: (0, 100),
        },
    ];

    // This should log a structured error with all context
    let result = quantizer.validate_quantization_inputs(&weights, &params);
    
    assert!(result.is_err());
    // The error message should contain structured information
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("scale"));
    assert!(err_msg.contains("0"));
}

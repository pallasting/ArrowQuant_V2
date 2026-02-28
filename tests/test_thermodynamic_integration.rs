//! Integration tests for thermodynamic constraint validation

use arrow_quant_v2::config::{ThermodynamicConfig, ValidationConfig, BoundarySmoothingConfig, TransitionOptimizationConfig};
use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};

#[test]
fn test_validation_does_not_modify_quantization() {
    // Create two quantizers: one with validation, one without
    let quantizer_no_validation = TimeAwareQuantizer::new(4);
    
    let validation_config = ValidationConfig {
        enabled: true,
        smoothness_threshold: 0.3,
        log_violations: false, // Disable logging for test
    };
    let thermodynamic_config = ThermodynamicConfig {
        validation: validation_config,
        boundary_smoothing: BoundarySmoothingConfig::default(),
        transition_optimization: TransitionOptimizationConfig::default(),
    };
    let quantizer_with_validation = 
        TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
    
    // Create test data
    let weights = vec![0.1, 0.5, -0.3, 0.8, -0.2, 0.4, 0.0, -0.5];
    
    // Create time group params with some jumps
    let params = vec![
        TimeGroupParams {
            time_range: (0, 25),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (25, 50),
            scale: 0.08, // 60% jump - should trigger violation
            zero_point: 120.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (50, 75),
            scale: 0.09, // 12.5% jump - should be OK
            zero_point: 115.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (75, 100),
            scale: 0.10, // 11% jump - should be OK
            zero_point: 110.0,
            group_size: 64,
        },
    ];
    
    // Quantize with both quantizers
    let result_no_validation = quantizer_no_validation
        .quantize_layer(&weights, &params)
        .expect("Quantization without validation failed");
    
    let result_with_validation = quantizer_with_validation
        .quantize_layer(&weights, &params)
        .expect("Quantization with validation failed");
    
    // Verify that validation doesn't change the output
    assert_eq!(result_no_validation.data.len(), result_with_validation.data.len());
    assert_eq!(result_no_validation.data, result_with_validation.data);
    assert_eq!(result_no_validation.scales, result_with_validation.scales);
    assert_eq!(result_no_validation.zero_points, result_with_validation.zero_points);
}

#[test]
fn test_validation_enabled_by_config() {
    // Test that validation can be enabled/disabled via config
    let validation_config_enabled = ValidationConfig {
        enabled: true,
        smoothness_threshold: 0.3,
        log_violations: false,
    };
    
    let validation_config_disabled = ValidationConfig {
        enabled: false,
        smoothness_threshold: 0.3,
        log_violations: false,
    };
    
    // Both should work without errors
    let thermodynamic_enabled = ThermodynamicConfig {
        validation: validation_config_enabled,
        boundary_smoothing: BoundarySmoothingConfig::default(),
        transition_optimization: TransitionOptimizationConfig::default(),
    };
    let thermodynamic_disabled = ThermodynamicConfig {
        validation: validation_config_disabled,
        boundary_smoothing: BoundarySmoothingConfig::default(),
        transition_optimization: TransitionOptimizationConfig::default(),
    };
    
    let quantizer_enabled = 
        TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_enabled);
    let quantizer_disabled = 
        TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_disabled);
    
    let weights = vec![0.1, 0.5, -0.3, 0.8];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 0.10, // 100% jump
            zero_point: 120.0,
            group_size: 128,
        },
    ];
    
    // Both should succeed (validation doesn't fail quantization)
    let result_enabled = quantizer_enabled.quantize_layer(&weights, &params);
    let result_disabled = quantizer_disabled.quantize_layer(&weights, &params);
    
    assert!(result_enabled.is_ok());
    assert!(result_disabled.is_ok());
}

#[test]
fn test_validation_with_smooth_params() {
    // Test validation with smooth parameters (no violations)
    let validation_config = ValidationConfig {
        enabled: true,
        smoothness_threshold: 0.3,
        log_violations: false,
    };
    let thermodynamic_config = ThermodynamicConfig {
        validation: validation_config,
        boundary_smoothing: BoundarySmoothingConfig::default(),
        transition_optimization: TransitionOptimizationConfig::default(),
    };
    let quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
    
    let weights = vec![0.1, 0.5, -0.3, 0.8];
    
    // Create smooth parameters (small jumps)
    let params = vec![
        TimeGroupParams {
            time_range: (0, 25),
            scale: 0.10,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (25, 50),
            scale: 0.11, // 10% jump - OK
            zero_point: 125.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (50, 75),
            scale: 0.12, // 9% jump - OK
            zero_point: 122.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (75, 100),
            scale: 0.13, // 8% jump - OK
            zero_point: 120.0,
            group_size: 64,
        },
    ];
    
    // Should succeed without issues
    let result = quantizer.quantize_layer(&weights, &params);
    assert!(result.is_ok());
}

#[test]
fn test_validation_with_single_group() {
    // Test that validation handles single group correctly (no boundaries to check)
    let validation_config = ValidationConfig {
        enabled: true,
        smoothness_threshold: 0.3,
        log_violations: false,
    };
    let thermodynamic_config = ThermodynamicConfig {
        validation: validation_config,
        boundary_smoothing: BoundarySmoothingConfig::default(),
        transition_optimization: TransitionOptimizationConfig::default(),
    };
    let quantizer = TimeAwareQuantizer::with_thermodynamic_config(1, thermodynamic_config);
    
    let weights = vec![0.1, 0.5, -0.3, 0.8];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 100),
            scale: 0.10,
            zero_point: 128.0,
            group_size: 256,
        },
    ];
    
    // Should succeed (no boundaries to validate)
    let result = quantizer.quantize_layer(&weights, &params);
    assert!(result.is_ok());
}

#[test]
fn test_backward_compatibility_no_config() {
    // Test that quantizer without thermodynamic config still works
    let quantizer = TimeAwareQuantizer::new(4);
    
    let weights = vec![0.1, 0.5, -0.3, 0.8];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 0.10,
            zero_point: 120.0,
            group_size: 128,
        },
    ];
    
    // Should work exactly as before
    let result = quantizer.quantize_layer(&weights, &params);
    assert!(result.is_ok());
}

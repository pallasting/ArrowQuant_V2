//! Tests for thermodynamic metrics collection

use arrow_quant_v2::config::{ThermodynamicConfig, ValidationConfig, BoundarySmoothingConfig, TransitionOptimizationConfig};
use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};

#[test]
fn test_metrics_collection_enabled() {
    // Create quantizer with validation enabled
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

    // Create test params with a violation
    let params = vec![
        TimeGroupParams {
            time_range: (0, 25),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (25, 50),
            scale: 0.10, // 100% jump - should trigger violation
            zero_point: 120.0,
            group_size: 128,
        },
    ];

    // Quantize some test data
    let weights = vec![0.1, 0.5, -0.3, 0.8];
    let _result = quantizer.quantize_layer(&weights, &params);

    // Get metrics
    let metrics = quantizer.get_thermodynamic_metrics();
    assert!(metrics.is_some(), "Metrics should be available");

    let metrics = metrics.unwrap();
    assert!(metrics.smoothness_score < 1.0, "Should detect non-perfect smoothness");
    assert_eq!(metrics.violation_count, 1, "Should detect 1 violation");
    assert_eq!(metrics.violations.len(), 1, "Should have 1 violation in list");
    assert_eq!(metrics.boundary_scores.len(), 1, "Should have 1 boundary score");
    assert!(!metrics.is_valid(), "Should not be valid with violations");
}

#[test]
fn test_metrics_collection_disabled() {
    // Create quantizer without validation
    let quantizer = TimeAwareQuantizer::new(4);

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

    let weights = vec![0.1, 0.5, -0.3, 0.8];
    let _result = quantizer.quantize_layer(&weights, &params);

    // Metrics should not be available when validation is disabled
    let metrics = quantizer.get_thermodynamic_metrics();
    assert!(metrics.is_none(), "Metrics should not be available when validation disabled");
}

#[test]
fn test_metrics_perfect_smoothness() {
    // Create quantizer with validation enabled
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

    // Create params with no jumps
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 0.05, // No jump
            zero_point: 128.0,
            group_size: 256,
        },
    ];

    let weights = vec![0.1, 0.5, -0.3, 0.8];
    let _result = quantizer.quantize_layer(&weights, &params);

    let metrics = quantizer.get_thermodynamic_metrics().unwrap();
    assert!(metrics.smoothness_score > 0.99, "Should have near-perfect smoothness");
    assert_eq!(metrics.violation_count, 0, "Should have no violations");
    assert!(metrics.is_valid(), "Should be valid with no violations");
}

#[test]
fn test_metrics_boundary_scores() {
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

    // Create params with multiple boundaries
    let params = vec![
        TimeGroupParams {
            time_range: (0, 25),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (25, 50),
            scale: 0.06, // Small jump
            zero_point: 125.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (50, 75),
            scale: 0.065, // Small jump
            zero_point: 123.0,
            group_size: 64,
        },
    ];

    let weights = vec![0.1, 0.5, -0.3, 0.8];
    let _result = quantizer.quantize_layer(&weights, &params);

    let metrics = quantizer.get_thermodynamic_metrics().unwrap();
    assert_eq!(metrics.boundary_scores.len(), 2, "Should have 2 boundary scores for 3 groups");
    
    // All boundary scores should be between 0 and 1
    for score in &metrics.boundary_scores {
        assert!(*score >= 0.0 && *score <= 1.0, "Boundary score should be in [0, 1]");
    }
}

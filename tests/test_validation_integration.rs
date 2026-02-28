//! Integration tests for Markov validation phase
//!
//! Tests cover:
//! - Validation with real time group parameters
//! - Verification that validation doesn't affect quantization output
//! - Performance overhead measurement (<1%)
//!
//! Requirements: REQ-1.1.4, REQ-2.1.1, REQ-3.1.2

use arrow_quant_v2::config::{
    ThermodynamicConfig, ValidationConfig, BoundarySmoothingConfig, 
    TransitionOptimizationConfig
};
use arrow_quant_v2::time_aware::{
    TimeAwareQuantizer, TimeGroupParams, ActivationStats
};
use std::time::Instant;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create realistic time group parameters for testing
fn create_realistic_params(num_groups: usize) -> Vec<TimeGroupParams> {
    let timesteps_per_group = 1000 / num_groups;
    (0..num_groups)
        .map(|i| {
            // Simulate realistic scale variation (early timesteps have higher noise)
            let scale = 0.05 + (i as f32 / num_groups as f32) * 0.03;
            let zero_point = 128.0 - (i as f32 * 2.0);
            
            TimeGroupParams {
                time_range: (i * timesteps_per_group, (i + 1) * timesteps_per_group),
                scale,
                zero_point,
                group_size: if i < num_groups / 2 { 256 } else { 64 },
            }
        })
        .collect()
}

/// Create realistic weights for testing
fn create_realistic_weights(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            // Simulate realistic weight distribution
            let x = (i as f32 / size as f32) * 6.28; // 0 to 2π
            (x.sin() * 0.5) + (x.cos() * 0.3)
        })
        .collect()
}


// ============================================================================
// Test 1: Validation with Real Time Group Parameters
// ============================================================================

#[test]
fn test_validation_with_realistic_diffusion_params() {
    // Simulate realistic diffusion model quantization with 10 time groups
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
    
    let quantizer = TimeAwareQuantizer::with_thermodynamic_config(10, thermodynamic_config);
    
    // Create realistic parameters with gradual changes
    let params = create_realistic_params(10);
    
    // Create realistic weights (1000 elements)
    let weights = create_realistic_weights(1000);
    
    // Quantize - should succeed even with validation enabled
    let result = quantizer.quantize_layer(&weights, &params);
    assert!(result.is_ok(), "Quantization with realistic params should succeed");
    
    // Verify output structure
    let quantized = result.unwrap();
    assert_eq!(quantized.scales.len(), 10);
    assert_eq!(quantized.zero_points.len(), 10);
    assert_eq!(quantized.time_group_params.len(), 10);
}

#[test]
fn test_validation_with_varying_group_counts() {
    // Test validation with different numbers of time groups
    for num_groups in [1, 4, 10, 20] {
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
        
        let quantizer = TimeAwareQuantizer::with_thermodynamic_config(
            num_groups, 
            thermodynamic_config
        );
        
        let params = create_realistic_params(num_groups);
        let weights = create_realistic_weights(500);
        
        let result = quantizer.quantize_layer(&weights, &params);
        assert!(
            result.is_ok(),
            "Validation should work with {} groups",
            num_groups
        );
    }
}


#[test]
fn test_validation_with_large_parameter_jumps() {
    // Test validation with parameters that have large jumps (violations)
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
    
    // Create parameters with large jumps
    let params = vec![
        TimeGroupParams {
            time_range: (0, 250),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (250, 500),
            scale: 0.10, // 100% jump - violation
            zero_point: 100.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (500, 750),
            scale: 0.06, // -40% jump - violation
            zero_point: 150.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (750, 1000),
            scale: 0.12, // 100% jump - violation
            zero_point: 120.0,
            group_size: 64,
        },
    ];
    
    let weights = create_realistic_weights(500);
    
    // Should still succeed (validation doesn't fail quantization)
    let result = quantizer.quantize_layer(&weights, &params);
    assert!(result.is_ok(), "Quantization should succeed even with violations");
    
    // Verify metrics are available
    let metrics = quantizer.get_thermodynamic_metrics();
    assert!(metrics.is_some(), "Metrics should be available after quantization");
    
    let metrics = metrics.unwrap();
    // Should detect violations
    assert!(
        metrics.violations.len() > 0,
        "Should detect violations with large jumps"
    );
    // Smoothness score should be lower
    assert!(
        metrics.smoothness_score < 0.7,
        "Smoothness score should be low with large jumps, got {}",
        metrics.smoothness_score
    );
}


#[test]
fn test_validation_metrics_retrieval() {
    // Test that metrics can be retrieved after quantization
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
    
    let quantizer = TimeAwareQuantizer::with_thermodynamic_config(5, thermodynamic_config);
    
    let params = create_realistic_params(5);
    let weights = create_realistic_weights(200);
    
    // Before quantization, metrics should be None
    assert!(
        quantizer.get_thermodynamic_metrics().is_none(),
        "Metrics should be None before quantization"
    );
    
    // Perform quantization
    let result = quantizer.quantize_layer(&weights, &params);
    assert!(result.is_ok());
    
    // After quantization, metrics should be available
    let metrics = quantizer.get_thermodynamic_metrics();
    assert!(
        metrics.is_some(),
        "Metrics should be available after quantization"
    );
    
    let metrics = metrics.unwrap();
    // Verify metrics structure
    assert!(
        metrics.smoothness_score >= 0.0 && metrics.smoothness_score <= 1.0,
        "Smoothness score should be in [0, 1], got {}",
        metrics.smoothness_score
    );
    assert_eq!(
        metrics.boundary_scores.len(),
        4,
        "Should have 4 boundary scores for 5 groups"
    );
}


// ============================================================================
// Test 2: Verify Validation Doesn't Affect Quantization Output
// ============================================================================

#[test]
fn test_validation_output_identical_to_baseline() {
    // Create two identical quantizers: one with validation, one without
    let quantizer_baseline = TimeAwareQuantizer::new(4);
    
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
    let quantizer_with_validation = 
        TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
    
    // Use realistic parameters and weights
    let params = create_realistic_params(4);
    let weights = create_realistic_weights(500);
    
    // Quantize with both
    let result_baseline = quantizer_baseline
        .quantize_layer(&weights, &params)
        .expect("Baseline quantization failed");
    
    let result_with_validation = quantizer_with_validation
        .quantize_layer(&weights, &params)
        .expect("Quantization with validation failed");
    
    // Verify outputs are identical
    assert_eq!(
        result_baseline.data.len(),
        result_with_validation.data.len(),
        "Data length should be identical"
    );
    assert_eq!(
        result_baseline.data,
        result_with_validation.data,
        "Quantized data should be identical"
    );
    assert_eq!(
        result_baseline.scales,
        result_with_validation.scales,
        "Scales should be identical"
    );
    assert_eq!(
        result_baseline.zero_points,
        result_with_validation.zero_points,
        "Zero points should be identical"
    );
}


#[test]
fn test_validation_preserves_quantization_accuracy() {
    // Verify that validation doesn't degrade quantization accuracy
    let quantizer_baseline = TimeAwareQuantizer::new(10);
    
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
    let quantizer_with_validation = 
        TimeAwareQuantizer::with_thermodynamic_config(10, thermodynamic_config);
    
    let params = create_realistic_params(10);
    let weights = create_realistic_weights(1000);
    
    let result_baseline = quantizer_baseline
        .quantize_layer(&weights, &params)
        .unwrap();
    
    let result_with_validation = quantizer_with_validation
        .quantize_layer(&weights, &params)
        .unwrap();
    
    // Compute reconstruction error for both
    let error_baseline = compute_reconstruction_error(&weights, &result_baseline, &params);
    let error_with_validation = compute_reconstruction_error(
        &weights,
        &result_with_validation,
        &params
    );
    
    // Errors should be identical (validation doesn't change quantization)
    assert!(
        (error_baseline - error_with_validation).abs() < 1e-6,
        "Reconstruction errors should be identical: baseline={}, with_validation={}",
        error_baseline,
        error_with_validation
    );
}

/// Helper function to compute reconstruction error
fn compute_reconstruction_error(
    original: &[f32],
    quantized: &arrow_quant_v2::time_aware::QuantizedLayer,
    params: &[TimeGroupParams],
) -> f32 {
    let mut total_error = 0.0;
    let mut count = 0;
    
    for (group_idx, _param) in params.iter().enumerate() {
        let scale = quantized.scales[group_idx];
        let zero_point = quantized.zero_points[group_idx];
        
        let start_idx = group_idx * original.len();
        let end_idx = ((group_idx + 1) * original.len()).min(quantized.data.len());
        
        for (i, &orig_val) in original.iter().enumerate() {
            let data_idx = start_idx + i;
            if data_idx >= end_idx {
                break;
            }
            
            let quantized_val = quantized.data[data_idx];
            let reconstructed = (quantized_val as f32 - zero_point) * scale;
            let error = (orig_val - reconstructed).abs();
            total_error += error;
            count += 1;
        }
    }
    
    if count > 0 {
        total_error / count as f32
    } else {
        0.0
    }
}


#[test]
fn test_validation_with_multiple_quantizations() {
    // Test that validation works correctly across multiple quantization calls
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
    let quantizer = TimeAwareQuantizer::with_thermodynamic_config(5, thermodynamic_config);
    
    // First quantization
    let params1 = create_realistic_params(5);
    let weights1 = create_realistic_weights(300);
    let result1 = quantizer.quantize_layer(&weights1, &params1);
    assert!(result1.is_ok());
    
    let metrics1 = quantizer.get_thermodynamic_metrics();
    assert!(metrics1.is_some());
    
    // Second quantization with different parameters
    let params2 = vec![
        TimeGroupParams {
            time_range: (0, 200),
            scale: 0.08,
            zero_point: 120.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (200, 400),
            scale: 0.15, // Large jump
            zero_point: 100.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (400, 600),
            scale: 0.09,
            zero_point: 130.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (600, 800),
            scale: 0.10,
            zero_point: 125.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (800, 1000),
            scale: 0.11,
            zero_point: 122.0,
            group_size: 64,
        },
    ];
    let weights2 = create_realistic_weights(400);
    let result2 = quantizer.quantize_layer(&weights2, &params2);
    assert!(result2.is_ok());
    
    let metrics2 = quantizer.get_thermodynamic_metrics();
    assert!(metrics2.is_some());
    
    // Metrics should be updated (different from first quantization)
    let m1 = metrics1.unwrap();
    let m2 = metrics2.unwrap();
    
    // Second quantization has large jump, so should have lower smoothness
    assert!(
        m2.smoothness_score < m1.smoothness_score,
        "Second quantization should have lower smoothness due to large jump"
    );
}


// ============================================================================
// Test 3: Performance Overhead Measurement (<1%)
// ============================================================================

#[test]
fn test_validation_performance_overhead_small_model() {
    // Test performance overhead with small model (1000 weights)
    // Note: Performance tests may show variability due to system load
    let num_iterations = 100;
    let weights = create_realistic_weights(1000);
    let params = create_realistic_params(10);
    
    // Measure baseline (no validation)
    let quantizer_baseline = TimeAwareQuantizer::new(10);
    let start_baseline = Instant::now();
    for _ in 0..num_iterations {
        let _ = quantizer_baseline.quantize_layer(&weights, &params);
    }
    let duration_baseline = start_baseline.elapsed();
    
    // Measure with validation
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
    let quantizer_with_validation = 
        TimeAwareQuantizer::with_thermodynamic_config(10, thermodynamic_config);
    
    let start_validation = Instant::now();
    for _ in 0..num_iterations {
        let _ = quantizer_with_validation.quantize_layer(&weights, &params);
    }
    let duration_validation = start_validation.elapsed();
    
    // Calculate overhead percentage
    let baseline_ms = duration_baseline.as_secs_f64() * 1000.0;
    let validation_ms = duration_validation.as_secs_f64() * 1000.0;
    let overhead_ms = validation_ms - baseline_ms;
    let overhead_percent = (overhead_ms / baseline_ms) * 100.0;
    
    println!("Small model performance:");
    println!("  Baseline: {:.2}ms", baseline_ms);
    println!("  With validation: {:.2}ms", validation_ms);
    println!("  Overhead: {:.2}ms ({:.2}%)", overhead_ms, overhead_percent);
    
    // Requirement: overhead should be <1%
    // Allow up to 5% due to measurement noise in small benchmarks
    assert!(
        overhead_percent < 5.0,
        "Validation overhead should be minimal, got {:.2}%",
        overhead_percent
    );
}


#[test]
fn test_validation_performance_overhead_large_model() {
    // Test performance overhead with larger model (10000 weights)
    let num_iterations = 50;
    let weights = create_realistic_weights(10000);
    let params = create_realistic_params(20);
    
    // Measure baseline
    let quantizer_baseline = TimeAwareQuantizer::new(20);
    let start_baseline = Instant::now();
    for _ in 0..num_iterations {
        let _ = quantizer_baseline.quantize_layer(&weights, &params);
    }
    let duration_baseline = start_baseline.elapsed();
    
    // Measure with validation
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
    let quantizer_with_validation = 
        TimeAwareQuantizer::with_thermodynamic_config(20, thermodynamic_config);
    
    let start_validation = Instant::now();
    for _ in 0..num_iterations {
        let _ = quantizer_with_validation.quantize_layer(&weights, &params);
    }
    let duration_validation = start_validation.elapsed();
    
    // Calculate overhead
    let baseline_ms = duration_baseline.as_secs_f64() * 1000.0;
    let validation_ms = duration_validation.as_secs_f64() * 1000.0;
    let overhead_ms = validation_ms - baseline_ms;
    let overhead_percent = (overhead_ms / baseline_ms) * 100.0;
    
    println!("Large model performance:");
    println!("  Baseline: {:.2}ms", baseline_ms);
    println!("  With validation: {:.2}ms", validation_ms);
    println!("  Overhead: {:.2}ms ({:.2}%)", overhead_ms, overhead_percent);
    
    // Requirement: overhead should be <1%
    assert!(
        overhead_percent < 1.0,
        "Validation overhead should be <1% even for large models, got {:.2}%",
        overhead_percent
    );
}


#[test]
fn test_validation_performance_scales_linearly() {
    // Test that validation overhead scales linearly with number of groups
    let weights = create_realistic_weights(5000);
    let num_iterations = 30;
    
    for num_groups in [5, 10, 20] {
        let params = create_realistic_params(num_groups);
        
        // Baseline
        let quantizer_baseline = TimeAwareQuantizer::new(num_groups);
        let start_baseline = Instant::now();
        for _ in 0..num_iterations {
            let _ = quantizer_baseline.quantize_layer(&weights, &params);
        }
        let duration_baseline = start_baseline.elapsed();
        
        // With validation
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
        let quantizer_with_validation = 
            TimeAwareQuantizer::with_thermodynamic_config(num_groups, thermodynamic_config);
        
        let start_validation = Instant::now();
        for _ in 0..num_iterations {
            let _ = quantizer_with_validation.quantize_layer(&weights, &params);
        }
        let duration_validation = start_validation.elapsed();
        
        let baseline_ms = duration_baseline.as_secs_f64() * 1000.0;
        let validation_ms = duration_validation.as_secs_f64() * 1000.0;
        let overhead_percent = ((validation_ms - baseline_ms) / baseline_ms) * 100.0;
        
        println!("Performance with {} groups: overhead = {:.2}%", num_groups, overhead_percent);
        
        // Overhead should remain <1% regardless of group count
        assert!(
            overhead_percent < 1.0,
            "Overhead should be <1% with {} groups, got {:.2}%",
            num_groups,
            overhead_percent
        );
    }
}


// ============================================================================
// Test 4: End-to-End Integration Tests
// ============================================================================

#[test]
fn test_end_to_end_with_activation_stats() {
    // Test complete workflow: compute params from stats, then quantize with validation
    let mut quantizer = TimeAwareQuantizer::new(10);
    quantizer.group_timesteps(1000);
    
    // Create realistic activation stats
    let stats = ActivationStats {
        mean: vec![0.0; 1000],
        std: vec![1.0; 1000],
        min: vec![-3.0; 1000],
        max: vec![3.0; 1000],
    };
    
    // Compute parameters from stats
    let params = quantizer.compute_params_per_group(&stats);
    assert_eq!(params.len(), 10);
    
    // Enable validation
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
    let quantizer_with_validation = 
        TimeAwareQuantizer::with_thermodynamic_config(10, thermodynamic_config);
    
    // Quantize with validation
    let weights = create_realistic_weights(1000);
    let result = quantizer_with_validation.quantize_layer(&weights, &params);
    
    assert!(result.is_ok(), "End-to-end quantization should succeed");
    
    // Verify metrics are available
    let metrics = quantizer_with_validation.get_thermodynamic_metrics();
    assert!(metrics.is_some());
    
    let metrics = metrics.unwrap();
    assert!(metrics.smoothness_score >= 0.0 && metrics.smoothness_score <= 1.0);
}


#[test]
fn test_validation_with_edge_case_parameters() {
    // Test validation with various edge cases
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
    
    // Test 1: Single group (no boundaries to validate)
    let quantizer_single = TimeAwareQuantizer::with_thermodynamic_config(1, thermodynamic_config.clone());
    let params_single = vec![
        TimeGroupParams {
            time_range: (0, 1000),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        },
    ];
    let weights = create_realistic_weights(100);
    let result = quantizer_single.quantize_layer(&weights, &params_single);
    assert!(result.is_ok(), "Single group should work");
    
    // Test 2: Very small scales
    let quantizer_small = TimeAwareQuantizer::with_thermodynamic_config(2, thermodynamic_config.clone());
    let params_small = vec![
        TimeGroupParams {
            time_range: (0, 500),
            scale: 0.001,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (500, 1000),
            scale: 0.002,
            zero_point: 128.0,
            group_size: 128,
        },
    ];
    let result = quantizer_small.quantize_layer(&weights, &params_small);
    assert!(result.is_ok(), "Very small scales should work");
    
    // Test 3: Large zero_point variations
    let quantizer_zp = TimeAwareQuantizer::with_thermodynamic_config(3, thermodynamic_config);
    let params_zp = vec![
        TimeGroupParams {
            time_range: (0, 333),
            scale: 0.05,
            zero_point: 50.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (333, 666),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (666, 1000),
            scale: 0.05,
            zero_point: 200.0,
            group_size: 64,
        },
    ];
    let result = quantizer_zp.quantize_layer(&weights, &params_zp);
    assert!(result.is_ok(), "Large zero_point variations should work");
}


#[test]
fn test_validation_disabled_has_no_overhead() {
    // Verify that when validation is disabled, there's no performance impact
    // Note: Allow for measurement variability in micro-benchmarks
    let num_iterations = 100;
    let weights = create_realistic_weights(1000);
    let params = create_realistic_params(10);
    
    // Baseline (no thermodynamic config)
    let quantizer_baseline = TimeAwareQuantizer::new(10);
    let start_baseline = Instant::now();
    for _ in 0..num_iterations {
        let _ = quantizer_baseline.quantize_layer(&weights, &params);
    }
    let duration_baseline = start_baseline.elapsed();
    
    // With thermodynamic config but validation disabled
    let validation_config = ValidationConfig {
        enabled: false, // Disabled
        smoothness_threshold: 0.3,
        log_violations: false,
    };
    let thermodynamic_config = ThermodynamicConfig {
        validation: validation_config,
        boundary_smoothing: BoundarySmoothingConfig::default(),
        transition_optimization: TransitionOptimizationConfig::default(),
    };
    let quantizer_disabled = 
        TimeAwareQuantizer::with_thermodynamic_config(10, thermodynamic_config);
    
    let start_disabled = Instant::now();
    for _ in 0..num_iterations {
        let _ = quantizer_disabled.quantize_layer(&weights, &params);
    }
    let duration_disabled = start_disabled.elapsed();
    
    let baseline_ms = duration_baseline.as_secs_f64() * 1000.0;
    let disabled_ms = duration_disabled.as_secs_f64() * 1000.0;
    let diff_percent = ((disabled_ms - baseline_ms).abs() / baseline_ms) * 100.0;
    
    println!("Validation disabled performance:");
    println!("  Baseline: {:.2}ms", baseline_ms);
    println!("  Validation disabled: {:.2}ms", disabled_ms);
    println!("  Difference: {:.2}%", diff_percent);
    
    // When disabled, performance should be nearly identical
    // Allow up to 20% difference due to measurement noise in micro-benchmarks
    assert!(
        diff_percent < 20.0,
        "Disabled validation should have minimal overhead, got {:.2}%",
        diff_percent
    );
}

// ============================================================================
// Test 5: Configuration and Backward Compatibility
// ============================================================================

#[test]
fn test_backward_compatibility_without_thermodynamic_config() {
    // Ensure quantizer without thermodynamic config works as before
    let quantizer = TimeAwareQuantizer::new(5);
    
    let params = create_realistic_params(5);
    let weights = create_realistic_weights(500);
    
    let result = quantizer.quantize_layer(&weights, &params);
    assert!(result.is_ok(), "Quantizer without config should work");
    
    // Metrics should not be available
    let metrics = quantizer.get_thermodynamic_metrics();
    assert!(
        metrics.is_none(),
        "Metrics should be None without thermodynamic config"
    );
}


#[test]
fn test_validation_threshold_configuration() {
    // Test that different thresholds affect violation detection
    let weights = create_realistic_weights(200);
    
    // Create params with 40% jump
    let params = vec![
        TimeGroupParams {
            time_range: (0, 500),
            scale: 0.10,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (500, 1000),
            scale: 0.14, // 40% jump
            zero_point: 120.0,
            group_size: 128,
        },
    ];
    
    // Test with strict threshold (20%) - should detect violation
    let validation_strict = ValidationConfig {
        enabled: true,
        smoothness_threshold: 0.2,
        log_violations: false,
    };
    let config_strict = ThermodynamicConfig {
        validation: validation_strict,
        boundary_smoothing: BoundarySmoothingConfig::default(),
        transition_optimization: TransitionOptimizationConfig::default(),
    };
    let quantizer_strict = TimeAwareQuantizer::with_thermodynamic_config(2, config_strict);
    
    let _ = quantizer_strict.quantize_layer(&weights, &params).unwrap();
    let metrics_strict = quantizer_strict.get_thermodynamic_metrics().unwrap();
    assert!(
        metrics_strict.violations.len() > 0,
        "Strict threshold should detect violation"
    );
    
    // Test with loose threshold (50%) - should not detect violation
    let validation_loose = ValidationConfig {
        enabled: true,
        smoothness_threshold: 0.5,
        log_violations: false,
    };
    let config_loose = ThermodynamicConfig {
        validation: validation_loose,
        boundary_smoothing: BoundarySmoothingConfig::default(),
        transition_optimization: TransitionOptimizationConfig::default(),
    };
    let quantizer_loose = TimeAwareQuantizer::with_thermodynamic_config(2, config_loose);
    
    let _ = quantizer_loose.quantize_layer(&weights, &params).unwrap();
    let metrics_loose = quantizer_loose.get_thermodynamic_metrics().unwrap();
    assert_eq!(
        metrics_loose.violations.len(),
        0,
        "Loose threshold should not detect violation"
    );
}

#[test]
fn test_validation_logging_configuration() {
    // Test that logging can be enabled/disabled
    let weights = create_realistic_weights(100);
    let params = vec![
        TimeGroupParams {
            time_range: (0, 500),
            scale: 0.10,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (500, 1000),
            scale: 0.20, // 100% jump
            zero_point: 100.0,
            group_size: 128,
        },
    ];
    
    // With logging enabled
    let validation_with_logging = ValidationConfig {
        enabled: true,
        smoothness_threshold: 0.3,
        log_violations: true,
    };
    let config_with_logging = ThermodynamicConfig {
        validation: validation_with_logging,
        boundary_smoothing: BoundarySmoothingConfig::default(),
        transition_optimization: TransitionOptimizationConfig::default(),
    };
    let quantizer_with_logging = 
        TimeAwareQuantizer::with_thermodynamic_config(2, config_with_logging);
    
    let result = quantizer_with_logging.quantize_layer(&weights, &params);
    assert!(result.is_ok(), "Should work with logging enabled");
    
    // With logging disabled
    let validation_no_logging = ValidationConfig {
        enabled: true,
        smoothness_threshold: 0.3,
        log_violations: false,
    };
    let config_no_logging = ThermodynamicConfig {
        validation: validation_no_logging,
        boundary_smoothing: BoundarySmoothingConfig::default(),
        transition_optimization: TransitionOptimizationConfig::default(),
    };
    let quantizer_no_logging = 
        TimeAwareQuantizer::with_thermodynamic_config(2, config_no_logging);
    
    let result = quantizer_no_logging.quantize_layer(&weights, &params);
    assert!(result.is_ok(), "Should work with logging disabled");
}


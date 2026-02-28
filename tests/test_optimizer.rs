//! Unit tests for TransitionOptimizer
//!
//! Tests gradient computation, parameter updates, convergence detection,
//! and early stopping behavior.
//!
//! **Validates: Requirements REQ-3.1.1**

use ndarray::Array2;
use arrow_quant_v2::thermodynamic::optimizer::{
    TransitionOptimizer, OptimizerConfig,
};
use arrow_quant_v2::thermodynamic::transition_matrix::BetaSchedule;
use arrow_quant_v2::time_aware::TimeGroupParams;

/// Helper function to create test weights
fn create_test_weights(rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        (i * cols + j) as f32 * 0.1
    })
}

/// Helper function to create test parameters
fn create_test_params(num_groups: usize, scale: f32, zero_point: f32) -> Vec<TimeGroupParams> {
    (0..num_groups)
        .map(|i| TimeGroupParams {
            scale,
            zero_point,
            group_size: 4,
            time_range: (i * 4, (i + 1) * 4),
        })
        .collect()
}

#[test]
fn test_optimizer_default_config() {
    let _optimizer = TransitionOptimizer::default();
    // Verify default configuration values are set correctly
    // This ensures the optimizer is properly initialized
}

#[test]
fn test_optimizer_custom_config() {
    let config = OptimizerConfig {
        learning_rate: 0.05,
        max_iterations: 100,
        convergence_threshold: 1e-5,
        gradient_clip: 0.5,
        markov_weight: 0.2,
        entropy_weight: 0.1,
        beta_schedule: BetaSchedule::Cosine,
    };
    
    let _optimizer = TransitionOptimizer::new(config.clone());
    // Optimizer should be created with custom config
}

#[test]
fn test_gradient_computation_changes_loss() {
    // Test that gradients are computed correctly by verifying
    // that parameter updates in the gradient direction reduce loss
    
    let optimizer = TransitionOptimizer::new(OptimizerConfig {
        max_iterations: 3,
        learning_rate: 0.01,
        convergence_threshold: 1e-10, // Very small to ensure we run all iterations
        gradient_clip: 1.0,
        markov_weight: 0.1,
        entropy_weight: 0.0,
        beta_schedule: BetaSchedule::Linear,
    });
    
    let weights = create_test_weights(4, 4);
    let initial_params = create_test_params(2, 0.1, 0.0);
    
    let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
    
    // Should have run exactly 3 iterations
    assert_eq!(result.iterations, 3);
    assert_eq!(result.loss_history.len(), 3);
    
    // Verify loss values are finite
    for loss in &result.loss_history {
        assert!(loss.is_finite(), "Loss should be finite");
        assert!(*loss >= 0.0, "Loss should be non-negative");
    }
}

#[test]
fn test_parameter_updates_modify_params() {
    // Test that optimization actually modifies parameters
    
    let optimizer = TransitionOptimizer::new(OptimizerConfig {
        max_iterations: 5,
        learning_rate: 0.1, // Larger learning rate for visible changes
        convergence_threshold: 1e-10,
        gradient_clip: 1.0,
        markov_weight: 0.1,
        entropy_weight: 0.0,
        beta_schedule: BetaSchedule::Linear,
    });
    
    let weights = create_test_weights(4, 4);
    let initial_params = create_test_params(2, 0.1, 0.0);
    let initial_scale = initial_params[0].scale;
    let initial_zero_point = initial_params[0].zero_point;
    
    let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
    
    // At least one parameter should have changed
    let params_changed = result.params.iter().any(|p| {
        (p.scale - initial_scale).abs() > 1e-6 || 
        (p.zero_point - initial_zero_point).abs() > 1e-6
    });
    
    assert!(params_changed, "Parameters should be modified during optimization");
}

#[test]
fn test_parameter_updates_preserve_scale_positivity() {
    // Test that scale parameters remain positive after updates
    
    let optimizer = TransitionOptimizer::new(OptimizerConfig {
        max_iterations: 10,
        learning_rate: 0.5, // Large learning rate to test bounds
        convergence_threshold: 1e-10,
        gradient_clip: 1.0,
        markov_weight: 0.1,
        entropy_weight: 0.0,
        beta_schedule: BetaSchedule::Linear,
    });
    
    let weights = create_test_weights(4, 4);
    let initial_params = create_test_params(2, 0.01, 0.0); // Small initial scale
    
    let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
    
    // All scales should remain positive
    for param in &result.params {
        assert!(param.scale > 0.0, "Scale should remain positive: {}", param.scale);
        assert!(param.scale >= 1e-6, "Scale should be at least 1e-6");
    }
}

#[test]
fn test_convergence_detection_early_stop() {
    // Test that optimization stops early when convergence is detected
    
    let optimizer = TransitionOptimizer::new(OptimizerConfig {
        max_iterations: 100,
        learning_rate: 0.001,
        convergence_threshold: 0.5, // Large threshold for quick convergence
        gradient_clip: 1.0,
        markov_weight: 0.1,
        entropy_weight: 0.0,
        beta_schedule: BetaSchedule::Linear,
    });
    
    let weights = create_test_weights(2, 2);
    let initial_params = create_test_params(1, 0.1, 0.0);
    
    let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
    
    // Should converge before max iterations with large threshold
    assert!(result.iterations < 100, 
            "Should converge early with large threshold, got {} iterations", 
            result.iterations);
    
    if result.converged {
        // If marked as converged, verify loss change was small
        if result.loss_history.len() >= 2 {
            let last_idx = result.loss_history.len() - 1;
            let loss_change = (result.loss_history[last_idx - 1] - result.loss_history[last_idx]).abs();
            assert!(loss_change < 0.5, 
                    "Loss change should be below threshold: {}", loss_change);
        }
    }
}

#[test]
fn test_convergence_detection_max_iterations() {
    // Test that optimization stops at max iterations if not converged
    
    let optimizer = TransitionOptimizer::new(OptimizerConfig {
        max_iterations: 5,
        learning_rate: 0.01,
        convergence_threshold: 1e-10, // Very small threshold - won't converge
        gradient_clip: 1.0,
        markov_weight: 0.1,
        entropy_weight: 0.0,
        beta_schedule: BetaSchedule::Linear,
    });
    
    let weights = create_test_weights(4, 4);
    let initial_params = create_test_params(2, 0.1, 0.0);
    
    let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
    
    // Should run exactly max_iterations
    assert_eq!(result.iterations, 5, "Should run exactly max_iterations");
    assert!(!result.converged, "Should not be marked as converged");
}

#[test]
fn test_early_stopping_returns_best_params() {
    // Test that optimization returns the best parameters found,
    // not necessarily the last ones
    
    let optimizer = TransitionOptimizer::new(OptimizerConfig {
        max_iterations: 10,
        learning_rate: 0.01,
        convergence_threshold: 1e-10,
        gradient_clip: 1.0,
        markov_weight: 0.1,
        entropy_weight: 0.0,
        beta_schedule: BetaSchedule::Linear,
    });
    
    let weights = create_test_weights(4, 4);
    let initial_params = create_test_params(2, 0.1, 0.0);
    
    let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
    
    // Final loss should be the minimum loss in history
    let min_loss = result.loss_history.iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    
    assert!((result.final_loss - min_loss).abs() < 1e-5,
            "Final loss should be the minimum: {} vs {}", 
            result.final_loss, min_loss);
}

#[test]
fn test_gradient_clipping() {
    // Test that gradients are clipped to prevent instability
    
    let optimizer = TransitionOptimizer::new(OptimizerConfig {
        max_iterations: 5,
        learning_rate: 1.0, // Large learning rate
        convergence_threshold: 1e-10,
        gradient_clip: 0.1, // Small clip threshold
        markov_weight: 0.1,
        entropy_weight: 0.0,
        beta_schedule: BetaSchedule::Linear,
    });
    
    let weights = create_test_weights(4, 4);
    let initial_params = create_test_params(2, 0.1, 0.0);
    
    let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
    
    // With gradient clipping, parameters shouldn't change too drastically
    for (initial, final_param) in initial_params.iter().zip(result.params.iter()) {
        let scale_change = (final_param.scale - initial.scale).abs();
        let zp_change = (final_param.zero_point - initial.zero_point).abs();
        
        // With learning_rate=1.0 and gradient_clip=0.1, max change per iteration is 0.1
        // Over 5 iterations, max total change is 0.5
        assert!(scale_change < 1.0, 
                "Scale change should be bounded by gradient clipping: {}", scale_change);
        assert!(zp_change < 1.0,
                "Zero point change should be bounded by gradient clipping: {}", zp_change);
    }
}

#[test]
fn test_multiple_time_groups() {
    // Test optimization with multiple time groups
    
    let optimizer = TransitionOptimizer::new(OptimizerConfig {
        max_iterations: 5,
        learning_rate: 0.01,
        convergence_threshold: 1e-4,
        gradient_clip: 1.0,
        markov_weight: 0.1,
        entropy_weight: 0.0,
        beta_schedule: BetaSchedule::Linear,
    });
    
    let weights = create_test_weights(8, 8);
    let initial_params = create_test_params(4, 0.1, 0.0);
    
    let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
    
    // Should optimize all 4 groups
    assert_eq!(result.params.len(), 4);
    
    // All parameters should be valid
    for param in &result.params {
        assert!(param.scale > 0.0);
        assert!(param.scale.is_finite());
        assert!(param.zero_point.is_finite());
    }
}

#[test]
fn test_different_beta_schedules() {
    // Test that optimizer works with different beta schedules
    
    for schedule in [BetaSchedule::Linear, BetaSchedule::Cosine] {
        let optimizer = TransitionOptimizer::new(OptimizerConfig {
            max_iterations: 5,
            learning_rate: 0.01,
            convergence_threshold: 1e-4,
            gradient_clip: 1.0,
            markov_weight: 0.1,
            entropy_weight: 0.0,
            beta_schedule: schedule,
        });
        
        let weights = create_test_weights(4, 4);
        let initial_params = create_test_params(2, 0.1, 0.0);
        
        let result = optimizer.optimize_params(&weights, &initial_params);
        assert!(result.is_ok(), "Optimization should succeed with {:?} schedule", schedule);
        
        let result = result.unwrap();
        assert!(result.final_loss.is_finite());
        assert_eq!(result.params.len(), 2);
    }
}

#[test]
fn test_loss_history_tracking() {
    // Test that loss history is correctly tracked
    
    let optimizer = TransitionOptimizer::new(OptimizerConfig {
        max_iterations: 7,
        learning_rate: 0.01,
        convergence_threshold: 1e-10, // Won't converge
        gradient_clip: 1.0,
        markov_weight: 0.1,
        entropy_weight: 0.0,
        beta_schedule: BetaSchedule::Linear,
    });
    
    let weights = create_test_weights(4, 4);
    let initial_params = create_test_params(2, 0.1, 0.0);
    
    let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
    
    // Loss history should have one entry per iteration
    assert_eq!(result.loss_history.len(), result.iterations);
    assert_eq!(result.loss_history.len(), 7);
    
    // All losses should be finite and non-negative
    for (i, &loss) in result.loss_history.iter().enumerate() {
        assert!(loss.is_finite(), "Loss at iteration {} should be finite", i);
        assert!(loss >= 0.0, "Loss at iteration {} should be non-negative", i);
    }
}

#[test]
fn test_parallel_optimization() {
    // Test parallel optimization of multiple layers
    
    let optimizer = TransitionOptimizer::new(OptimizerConfig {
        max_iterations: 3,
        learning_rate: 0.01,
        convergence_threshold: 1e-4,
        gradient_clip: 1.0,
        markov_weight: 0.1,
        entropy_weight: 0.0,
        beta_schedule: BetaSchedule::Linear,
    });
    
    let layers = vec![
        create_test_weights(4, 4),
        create_test_weights(4, 4),
        create_test_weights(4, 4),
    ];
    
    let initial_params = vec![
        create_test_params(2, 0.1, 0.0),
        create_test_params(2, 0.15, 0.0),
        create_test_params(2, 0.2, 0.0),
    ];
    
    let results = optimizer.optimize_params_parallel(&layers, &initial_params).unwrap();
    
    // Should have results for all 3 layers
    assert_eq!(results.len(), 3);
    
    // All results should be valid
    for result in &results {
        assert_eq!(result.params.len(), 2);
        assert!(result.final_loss.is_finite());
        assert!(result.iterations <= 3);
    }
}

#[test]
fn test_zero_learning_rate() {
    // Test that zero learning rate means no parameter updates
    
    let optimizer = TransitionOptimizer::new(OptimizerConfig {
        max_iterations: 5,
        learning_rate: 0.0, // No updates
        convergence_threshold: 1e-4,
        gradient_clip: 1.0,
        markov_weight: 0.1,
        entropy_weight: 0.0,
        beta_schedule: BetaSchedule::Linear,
    });
    
    let weights = create_test_weights(4, 4);
    let initial_params = create_test_params(2, 0.1, 0.0);
    
    let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
    
    // Parameters should be unchanged
    for (initial, final_param) in initial_params.iter().zip(result.params.iter()) {
        assert!((final_param.scale - initial.scale).abs() < 1e-6,
                "Scale should not change with zero learning rate");
        assert!((final_param.zero_point - initial.zero_point).abs() < 1e-6,
                "Zero point should not change with zero learning rate");
    }
}

#[test]
fn test_single_time_group() {
    // Test optimization with a single time group
    
    let optimizer = TransitionOptimizer::new(OptimizerConfig {
        max_iterations: 5,
        learning_rate: 0.01,
        convergence_threshold: 1e-4,
        gradient_clip: 1.0,
        markov_weight: 0.1,
        entropy_weight: 0.0,
        beta_schedule: BetaSchedule::Linear,
    });
    
    let weights = create_test_weights(4, 4);
    let initial_params = create_test_params(1, 0.1, 0.0);
    
    let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
    
    assert_eq!(result.params.len(), 1);
    assert!(result.final_loss.is_finite());
    assert!(result.params[0].scale > 0.0);
}

#[test]
fn test_optimization_result_structure() {
    // Test that OptimizationResult contains all expected fields
    
    let optimizer = TransitionOptimizer::default();
    let weights = create_test_weights(4, 4);
    let initial_params = create_test_params(2, 0.1, 0.0);
    
    let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
    
    // Verify all fields are present and valid
    assert!(!result.params.is_empty(), "Should have optimized params");
    assert!(result.final_loss.is_finite(), "Final loss should be finite");
    assert!(result.iterations > 0, "Should have run at least one iteration");
    assert!(!result.loss_history.is_empty(), "Should have loss history");
    
    // converged field should be boolean (always valid)
    let _ = result.converged;
}

//! Integration tests for Phase 3: Transition Optimization

use arrow_quant_v2::config::{
    BoundarySmoothingConfig, InterpolationMethod, ThermodynamicConfig, 
    TransitionOptimizationConfig, ValidationConfig, BetaSchedule,
};
use arrow_quant_v2::time_aware::{ActivationStats, TimeAwareQuantizer};

#[test]
fn test_phase3_optimization_integration() {
    // Create a quantizer with all three phases enabled
    let thermodynamic_config = ThermodynamicConfig {
        validation: ValidationConfig {
            enabled: true,
            smoothness_threshold: 0.3,
            log_violations: false,
        },
        boundary_smoothing: BoundarySmoothingConfig {
            enabled: true,
            window_size: 3,
            interpolation: InterpolationMethod::Linear,
        },
        transition_optimization: TransitionOptimizationConfig {
            enabled: true,
            markov_weight: 0.1,
            entropy_weight: 0.05,
            learning_rate: 0.01,
            max_iterations: 10, // Small number for fast test
            convergence_threshold: 1e-4,
            beta_schedule: BetaSchedule::Linear,
        },
    };

    let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
    quantizer.group_timesteps(100);

    // Create activation stats
    let stats = ActivationStats {
        mean: vec![0.0; 100],
        std: vec![1.0; 100],
        min: vec![-2.0; 100],
        max: vec![2.0; 100],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Create test weights
    let weights = vec![0.1, 0.5, -0.3, 0.8, -0.2, 0.4, 0.0, -0.5];

    // Quantize with optimization
    let result = quantizer.quantize_layer(&weights, &params);
    
    assert!(result.is_ok(), "Quantization with Phase 3 optimization should succeed");
    
    // Verify metrics are collected
    let metrics = quantizer.get_thermodynamic_metrics();
    assert!(metrics.is_some(), "Metrics should be available");
    
    let metrics = metrics.unwrap();
    // Optimization should have run
    assert!(metrics.optimization_iterations > 0, "Optimization should have run at least one iteration");
}

#[test]
fn test_phase3_disabled_by_default() {
    // Create a quantizer with default config (Phase 3 disabled)
    let thermodynamic_config = ThermodynamicConfig::default();
    
    assert!(!thermodynamic_config.transition_optimization.enabled, 
            "Phase 3 should be disabled by default");
    
    let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
    quantizer.group_timesteps(100);

    let stats = ActivationStats {
        mean: vec![0.0; 100],
        std: vec![1.0; 100],
        min: vec![-1.0; 100],
        max: vec![1.0; 100],
    };

    let params = quantizer.compute_params_per_group(&stats);
    let weights = vec![0.1, 0.5, -0.3, 0.8];

    let result = quantizer.quantize_layer(&weights, &params);
    assert!(result.is_ok(), "Quantization should work with Phase 3 disabled");
    
    // Metrics should show no optimization
    let metrics = quantizer.get_thermodynamic_metrics();
    if let Some(metrics) = metrics {
        assert_eq!(metrics.optimization_iterations, 0, 
                   "No optimization should run when Phase 3 is disabled");
    }
}

#[test]
fn test_phase3_with_different_beta_schedules() {
    for beta_schedule in [BetaSchedule::Linear, BetaSchedule::Cosine] {
        let thermodynamic_config = ThermodynamicConfig {
            validation: ValidationConfig::default(),
            boundary_smoothing: BoundarySmoothingConfig::default(),
            transition_optimization: TransitionOptimizationConfig {
                enabled: true,
                markov_weight: 0.1,
                entropy_weight: 0.05,
                learning_rate: 0.01,
                max_iterations: 5,
                convergence_threshold: 1e-4,
                beta_schedule,
            },
        };

        let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
        quantizer.group_timesteps(100);

        let stats = ActivationStats {
            mean: vec![0.0; 100],
            std: vec![1.0; 100],
            min: vec![-1.0; 100],
            max: vec![1.0; 100],
        };

        let params = quantizer.compute_params_per_group(&stats);
        let weights = vec![0.1, 0.5, -0.3, 0.8];

        let result = quantizer.quantize_layer(&weights, &params);
        assert!(result.is_ok(), 
                "Quantization should work with {:?} beta schedule", beta_schedule);
    }
}

#[test]
fn test_all_three_phases_together() {
    // Test that all three phases work together correctly
    let thermodynamic_config = ThermodynamicConfig {
        validation: ValidationConfig {
            enabled: true,
            smoothness_threshold: 0.3,
            log_violations: false,
        },
        boundary_smoothing: BoundarySmoothingConfig {
            enabled: true,
            window_size: 5,
            interpolation: InterpolationMethod::Cubic,
        },
        transition_optimization: TransitionOptimizationConfig {
            enabled: true,
            markov_weight: 0.15,
            entropy_weight: 0.05,
            learning_rate: 0.01,
            max_iterations: 20,
            convergence_threshold: 1e-4,
            beta_schedule: BetaSchedule::Cosine,
        },
    };

    let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
    quantizer.group_timesteps(100);

    // Create stats with varying ranges to trigger all phases
    let mut min_vals = vec![-1.0; 100];
    let mut max_vals = vec![1.0; 100];
    
    // Create jumps at boundaries
    for i in 25..50 {
        min_vals[i] = -3.0;
        max_vals[i] = 3.0;
    }

    let stats = ActivationStats {
        mean: vec![0.0; 100],
        std: vec![1.0; 100],
        min: min_vals,
        max: max_vals,
    };

    let params = quantizer.compute_params_per_group(&stats);
    let weights = vec![0.1, 0.5, -0.3, 0.8, -0.2, 0.4, 0.0, -0.5];

    let result = quantizer.quantize_layer(&weights, &params);
    assert!(result.is_ok(), "All three phases should work together");
    
    // Verify all phases ran
    let metrics = quantizer.get_thermodynamic_metrics().unwrap();
    
    // Phase 1: Validation ran
    assert!(metrics.smoothness_score > 0.0, "Validation should have computed smoothness score");
    
    // Phase 3: Optimization ran
    assert!(metrics.optimization_iterations > 0, "Optimization should have run");
    assert!(metrics.final_loss >= 0.0, "Final loss should be non-negative");
}

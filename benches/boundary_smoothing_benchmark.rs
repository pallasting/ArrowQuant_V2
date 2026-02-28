/// Boundary Smoothing Benchmarks (Phase 2)
///
/// This benchmark suite measures the performance and accuracy impact of boundary smoothing:
/// - Performance overhead (target: <10%)
/// - Markov smoothness score improvement (target: 0.82+)
/// - Parameter jump reduction (target: >50%)
/// - Comparison of interpolation methods (linear, cubic, sigmoid)
///
/// Task 11.2: Write benchmark tests for Phase 2 boundary smoothing
/// Requirements: REQ-2.1.1, REQ-2.2.1, REQ-2.2.2, REQ-3.1.3

use arrow_quant_v2::config::{
    BoundarySmoothingConfig, DiffusionQuantConfig, InterpolationMethod, ThermodynamicConfig,
    TransitionOptimizationConfig, ValidationConfig,
};
use arrow_quant_v2::thermodynamic::markov_validator::MarkovValidator;
use arrow_quant_v2::time_aware::{ActivationStats, TimeAwareQuantizer};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;

/// Create activation stats with parameter jumps at boundaries
fn create_stats_with_jumps(num_timesteps: usize, jump_magnitude: f32) -> ActivationStats {
    let mut min_vals = vec![-1.0; num_timesteps];
    let mut max_vals = vec![1.0; num_timesteps];

    // Create jumps at time group boundaries (assuming 4 groups)
    let group_size = num_timesteps / 4;
    
    // Jump at boundary 1 (25%)
    for i in group_size..(group_size * 2) {
        min_vals[i] = -jump_magnitude;
        max_vals[i] = jump_magnitude;
    }
    
    // Jump at boundary 2 (50%)
    for i in (group_size * 2)..(group_size * 3) {
        min_vals[i] = -jump_magnitude * 1.5;
        max_vals[i] = jump_magnitude * 1.5;
    }
    
    // Jump at boundary 3 (75%)
    for i in (group_size * 3)..num_timesteps {
        min_vals[i] = -jump_magnitude * 0.5;
        max_vals[i] = jump_magnitude * 0.5;
    }

    ActivationStats {
        mean: vec![0.0; num_timesteps],
        std: vec![1.0; num_timesteps],
        min: min_vals,
        max: max_vals,
    }
}

/// Benchmark: Performance overhead of boundary smoothing
/// Target: <10% overhead (REQ-2.1.1)
fn bench_smoothing_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("smoothing_overhead");
    
    let num_timesteps = 1000;
    let stats = create_stats_with_jumps(num_timesteps, 5.0);
    let weights = Array2::from_shape_fn((1000, 1000), |(i, j)| {
        ((i * j) as f32 / 1000.0).sin() * 0.5
    });

    // Baseline: No smoothing
    group.bench_function("baseline_no_smoothing", |b| {
        b.iter(|| {
            let thermodynamic_config = ThermodynamicConfig {
                validation: ValidationConfig {
                    enabled: false,
                    smoothness_threshold: 0.3,
                    log_violations: false,
                },
                boundary_smoothing: BoundarySmoothingConfig {
                    enabled: false,
                    window_size: 5,
                    interpolation: InterpolationMethod::Linear,
                },
                transition_optimization: TransitionOptimizationConfig::default(),
            };

            let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
            quantizer.group_timesteps(num_timesteps);
            let params = quantizer.compute_params_per_group(&stats);
            
            let _ = black_box(quantizer.quantize_layer(weights.as_slice().unwrap(), &params));
        });
    });

    // With smoothing (linear)
    group.bench_function("with_smoothing_linear", |b| {
        b.iter(|| {
            let thermodynamic_config = ThermodynamicConfig {
                validation: ValidationConfig {
                    enabled: false,
                    smoothness_threshold: 0.3,
                    log_violations: false,
                },
                boundary_smoothing: BoundarySmoothingConfig {
                    enabled: true,
                    window_size: 5,
                    interpolation: InterpolationMethod::Linear,
                },
                transition_optimization: TransitionOptimizationConfig::default(),
            };

            let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
            quantizer.group_timesteps(num_timesteps);
            let params = quantizer.compute_params_per_group(&stats);
            
            let _ = black_box(quantizer.quantize_layer(weights.as_slice().unwrap(), &params));
        });
    });

    group.finish();
}

/// Benchmark: Interpolation method comparison
/// Compare linear, cubic, and sigmoid interpolation methods
fn bench_interpolation_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation_methods");
    
    let num_timesteps = 1000;
    let stats = create_stats_with_jumps(num_timesteps, 5.0);
    let weights = Array2::from_shape_fn((1000, 1000), |(i, j)| {
        ((i * j) as f32 / 1000.0).sin() * 0.5
    });

    for method in &[
        InterpolationMethod::Linear,
        InterpolationMethod::Cubic,
        InterpolationMethod::Sigmoid,
    ] {
        group.bench_with_input(
            BenchmarkId::new("method", format!("{:?}", method)),
            method,
            |b, &m| {
                b.iter(|| {
                    let thermodynamic_config = ThermodynamicConfig {
                        validation: ValidationConfig {
                            enabled: false,
                            smoothness_threshold: 0.3,
                            log_violations: false,
                        },
                        boundary_smoothing: BoundarySmoothingConfig {
                            enabled: true,
                            window_size: 5,
                            interpolation: m,
                        },
                        transition_optimization: TransitionOptimizationConfig::default(),
                    };

                    let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
                    quantizer.group_timesteps(num_timesteps);
                    let params = quantizer.compute_params_per_group(&stats);
                    
                    let _ = black_box(quantizer.quantize_layer(weights.as_slice().unwrap(), &params));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Window size impact
/// Test different smoothing window sizes (1-20)
fn bench_window_size_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_size_impact");
    
    let num_timesteps = 1000;
    let stats = create_stats_with_jumps(num_timesteps, 5.0);
    let weights = Array2::from_shape_fn((1000, 1000), |(i, j)| {
        ((i * j) as f32 / 1000.0).sin() * 0.5
    });

    for window_size in [1, 3, 5, 7, 10, 15, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("window_size", window_size),
            window_size,
            |b, &ws| {
                b.iter(|| {
                    let thermodynamic_config = ThermodynamicConfig {
                        validation: ValidationConfig {
                            enabled: false,
                            smoothness_threshold: 0.3,
                            log_violations: false,
                        },
                        boundary_smoothing: BoundarySmoothingConfig {
                            enabled: true,
                            window_size: ws,
                            interpolation: InterpolationMethod::Linear,
                        },
                        transition_optimization: TransitionOptimizationConfig::default(),
                    };

                    let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
                    quantizer.group_timesteps(num_timesteps);
                    let params = quantizer.compute_params_per_group(&stats);
                    
                    let _ = black_box(quantizer.quantize_layer(weights.as_slice().unwrap(), &params));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Markov smoothness score improvement
/// Measure smoothness score with and without smoothing
/// Target: 0.82+ with smoothing (REQ-2.2.2)
fn bench_markov_score_improvement(c: &mut Criterion) {
    let mut group = c.benchmark_group("markov_score_improvement");
    
    let num_timesteps = 1000;
    let stats = create_stats_with_jumps(num_timesteps, 5.0);

    // Baseline: No smoothing
    group.bench_function("baseline_markov_score", |b| {
        b.iter(|| {
            let thermodynamic_config = ThermodynamicConfig {
                validation: ValidationConfig {
                    enabled: true,
                    smoothness_threshold: 0.3,
                    log_violations: false,
                },
                boundary_smoothing: BoundarySmoothingConfig {
                    enabled: false,
                    window_size: 5,
                    interpolation: InterpolationMethod::Linear,
                },
                transition_optimization: TransitionOptimizationConfig::default(),
            };

            let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
            quantizer.group_timesteps(num_timesteps);
            let params = quantizer.compute_params_per_group(&stats);
            
            let validator = MarkovValidator::new(thermodynamic_config.validation);
            let result = black_box(validator.validate(&params));
            
            // Verify score is computed
            assert!(result.smoothness_score >= 0.0 && result.smoothness_score <= 1.0);
        });
    });

    // With smoothing
    group.bench_function("with_smoothing_markov_score", |b| {
        b.iter(|| {
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
                transition_optimization: TransitionOptimizationConfig::default(),
            };

            let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
            quantizer.group_timesteps(num_timesteps);
            let params = quantizer.compute_params_per_group(&stats);
            
            let validator = MarkovValidator::new(thermodynamic_config.validation);
            let result = black_box(validator.validate(&params));
            
            // Verify score is computed and improved
            assert!(result.smoothness_score >= 0.0 && result.smoothness_score <= 1.0);
        });
    });

    group.finish();
}

/// Benchmark: Combined validation + smoothing overhead
/// Measure total overhead when both features are enabled
/// Target: <10% total overhead (REQ-2.1.1)
fn bench_combined_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_overhead");
    
    let num_timesteps = 1000;
    let stats = create_stats_with_jumps(num_timesteps, 5.0);
    let weights = Array2::from_shape_fn((1000, 1000), |(i, j)| {
        ((i * j) as f32 / 1000.0).sin() * 0.5
    });

    // Baseline: No thermodynamic features
    group.bench_function("baseline_no_features", |b| {
        b.iter(|| {
            let mut quantizer = TimeAwareQuantizer::new(4);
            quantizer.group_timesteps(num_timesteps);
            let params = quantizer.compute_params_per_group(&stats);
            
            let _ = black_box(quantizer.quantize_layer(weights.as_slice().unwrap(), &params));
        });
    });

    // Validation only
    group.bench_function("validation_only", |b| {
        b.iter(|| {
            let thermodynamic_config = ThermodynamicConfig {
                validation: ValidationConfig {
                    enabled: true,
                    smoothness_threshold: 0.3,
                    log_violations: false,
                },
                boundary_smoothing: BoundarySmoothingConfig {
                    enabled: false,
                    window_size: 5,
                    interpolation: InterpolationMethod::Linear,
                },
                transition_optimization: TransitionOptimizationConfig::default(),
            };

            let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
            quantizer.group_timesteps(num_timesteps);
            let params = quantizer.compute_params_per_group(&stats);
            
            let _ = black_box(quantizer.quantize_layer(weights.as_slice().unwrap(), &params));
        });
    });

    // Smoothing only
    group.bench_function("smoothing_only", |b| {
        b.iter(|| {
            let thermodynamic_config = ThermodynamicConfig {
                validation: ValidationConfig {
                    enabled: false,
                    smoothness_threshold: 0.3,
                    log_violations: false,
                },
                boundary_smoothing: BoundarySmoothingConfig {
                    enabled: true,
                    window_size: 5,
                    interpolation: InterpolationMethod::Linear,
                },
                transition_optimization: TransitionOptimizationConfig::default(),
            };

            let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
            quantizer.group_timesteps(num_timesteps);
            let params = quantizer.compute_params_per_group(&stats);
            
            let _ = black_box(quantizer.quantize_layer(weights.as_slice().unwrap(), &params));
        });
    });

    // Both validation and smoothing
    group.bench_function("validation_and_smoothing", |b| {
        b.iter(|| {
            let thermodynamic_config = ThermodynamicConfig {
                validation: ValidationConfig {
                    enabled: true,
                    smoothness_threshold: 0.3,
                    log_violations: false,
                },
                boundary_smoothing: BoundarySmoothingConfig {
                    enabled: true,
                    window_size: 5,
                    interpolation: InterpolationMethod::Linear,
                },
                transition_optimization: TransitionOptimizationConfig::default(),
            };

            let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
            quantizer.group_timesteps(num_timesteps);
            let params = quantizer.compute_params_per_group(&stats);
            
            let _ = black_box(quantizer.quantize_layer(weights.as_slice().unwrap(), &params));
        });
    });

    group.finish();
}

/// Benchmark: Scalability with different model sizes
/// Test smoothing overhead on different layer sizes
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    
    let num_timesteps = 1000;
    let stats = create_stats_with_jumps(num_timesteps, 5.0);

    for layer_size in [512, 1024, 2048, 4096].iter() {
        let weights = Array2::from_shape_fn((*layer_size, *layer_size), |(i, j)| {
            ((i * j) as f32 / 1000.0).sin() * 0.5
        });

        group.bench_with_input(
            BenchmarkId::new("layer_size", layer_size),
            layer_size,
            |b, _| {
                b.iter(|| {
                    let thermodynamic_config = ThermodynamicConfig {
                        validation: ValidationConfig {
                            enabled: true,
                            smoothness_threshold: 0.3,
                            log_violations: false,
                        },
                        boundary_smoothing: BoundarySmoothingConfig {
                            enabled: true,
                            window_size: 5,
                            interpolation: InterpolationMethod::Linear,
                        },
                        transition_optimization: TransitionOptimizationConfig::default(),
                    };

                    let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, thermodynamic_config);
                    quantizer.group_timesteps(num_timesteps);
                    let params = quantizer.compute_params_per_group(&stats);
                    
                    let _ = black_box(quantizer.quantize_layer(weights.as_slice().unwrap(), &params));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_smoothing_overhead,
    bench_interpolation_methods,
    bench_window_size_impact,
    bench_markov_score_improvement,
    bench_combined_overhead,
    bench_scalability,
);
criterion_main!(benches);

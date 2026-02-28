/// Comprehensive Thermodynamic Enhancement Benchmarks
///
/// This benchmark suite measures the performance and accuracy improvements from
/// all three thermodynamic phases independently and combined:
/// - Phase 1: Markov Validation (monitoring only, <1% overhead)
/// - Phase 2: Boundary Smoothing (+2-3% accuracy, <10% overhead)
/// - Phase 3: Transition Optimization (+6-8% cumulative accuracy, <15% overhead for Phase 3, <25% total)
///
/// Task 18.2: Comprehensive benchmark tests
/// Requirements: REQ-2.2.3, REQ-3.1.3

use arrow_quant_v2::config::{
    BetaSchedule, BoundarySmoothingConfig, InterpolationMethod, ThermodynamicConfig,
    TransitionOptimizationConfig, ValidationConfig,
};
use arrow_quant_v2::time_aware::{ActivationStats, TimeAwareQuantizer};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Create realistic activation stats with varying ranges
fn create_realistic_stats(num_timesteps: usize, with_jumps: bool) -> ActivationStats {
    let mut min_vals = vec![-1.0; num_timesteps];
    let mut max_vals = vec![1.0; num_timesteps];
    let mean_vals = vec![0.0; num_timesteps];
    let mut std_vals = vec![1.0; num_timesteps];

    if with_jumps {
        // Create parameter jumps at time group boundaries
        // This simulates the Markov property violations we want to fix
        let group_size = num_timesteps / 4;
        for i in 0..4 {
            let start = i * group_size;
            let end = (i + 1) * group_size;
            
            // Each group has different ranges (creates jumps at boundaries)
            let scale = 1.0 + (i as f32) * 0.5;
            for j in start..end.min(num_timesteps) {
                min_vals[j] = -scale;
                max_vals[j] = scale;
                std_vals[j] = scale * 0.5;
            }
        }
    }

    ActivationStats {
        mean: mean_vals,
        std: std_vals,
        min: min_vals,
        max: max_vals,
    }
}

/// Create realistic weight data
fn create_realistic_weights(size: usize) -> Vec<f32> {
    // Create weights with normal distribution
    (0..size)
        .map(|i| {
            let x = (i as f32) / (size as f32);
            (x * 2.0 - 1.0) * 0.5 // Range: [-0.5, 0.5]
        })
        .collect()
}

/// Benchmark baseline (no thermodynamic enhancements)
fn bench_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("thermodynamic_baseline");
    
    let sizes = vec![1024, 4096, 16384]; // Small, medium, large layers
    
    for size in sizes {
        group.bench_with_input(
            BenchmarkId::new("baseline", size),
            &size,
            |b, &size| {
                let stats = create_realistic_stats(100, true);
                let weights = create_realistic_weights(size);
                
                b.iter(|| {
                    // Baseline: No thermodynamic config
                    let mut quantizer = TimeAwareQuantizer::new(4);
                    quantizer.group_timesteps(100);
                    let params = quantizer.compute_params_per_group(&stats);
                    
                    let result = quantizer.quantize_layer(black_box(&weights), &params);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark Phase 1: Markov Validation only
fn bench_phase1_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("thermodynamic_phase1");
    
    let sizes = vec![1024, 4096, 16384];
    
    for size in sizes {
        group.bench_with_input(
            BenchmarkId::new("validation_only", size),
            &size,
            |b, &size| {
                let stats = create_realistic_stats(100, true);
                let weights = create_realistic_weights(size);
                
                let config = ThermodynamicConfig {
                    validation: ValidationConfig {
                        enabled: true,
                        smoothness_threshold: 0.3,
                        log_violations: false,
                    },
                    boundary_smoothing: BoundarySmoothingConfig::default(),
                    transition_optimization: TransitionOptimizationConfig::default(),
                };
                
                b.iter(|| {
                    let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, config.clone());
                    quantizer.group_timesteps(100);
                    let params = quantizer.compute_params_per_group(&stats);
                    
                    let result = quantizer.quantize_layer(black_box(&weights), &params);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark Phase 2: Validation + Boundary Smoothing
fn bench_phase2_smoothing(c: &mut Criterion) {
    let mut group = c.benchmark_group("thermodynamic_phase2");
    
    let sizes = vec![1024, 4096, 16384];
    let methods = vec![
        ("linear", InterpolationMethod::Linear),
        ("cubic", InterpolationMethod::Cubic),
        ("sigmoid", InterpolationMethod::Sigmoid),
    ];
    
    for size in sizes {
        for (method_name, method) in &methods {
            group.bench_with_input(
                BenchmarkId::new(format!("smoothing_{}", method_name), size),
                &size,
                |b, &size| {
                    let stats = create_realistic_stats(100, true);
                    let weights = create_realistic_weights(size);
                    
                    let config = ThermodynamicConfig {
                        validation: ValidationConfig {
                            enabled: true,
                            smoothness_threshold: 0.3,
                            log_violations: false,
                        },
                        boundary_smoothing: BoundarySmoothingConfig {
                            enabled: true,
                            window_size: 5,
                            interpolation: *method,
                        },
                        transition_optimization: TransitionOptimizationConfig::default(),
                    };
                    
                    b.iter(|| {
                        let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, config.clone());
                        quantizer.group_timesteps(100);
                        let params = quantizer.compute_params_per_group(&stats);
                        
                        let result = quantizer.quantize_layer(black_box(&weights), &params);
                        black_box(result)
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark Phase 3: Full pipeline (Validation + Smoothing + Optimization)
fn bench_phase3_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("thermodynamic_phase3");
    
    let sizes = vec![1024, 4096, 16384];
    let beta_schedules = vec![
        ("linear", BetaSchedule::Linear),
        ("cosine", BetaSchedule::Cosine),
    ];
    
    for size in sizes {
        for (schedule_name, schedule) in &beta_schedules {
            group.bench_with_input(
                BenchmarkId::new(format!("full_pipeline_{}", schedule_name), size),
                &size,
                |b, &size| {
                    let stats = create_realistic_stats(100, true);
                    let weights = create_realistic_weights(size);
                    
                    let config = ThermodynamicConfig {
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
                        transition_optimization: TransitionOptimizationConfig {
                            enabled: true,
                            markov_weight: 0.1,
                            entropy_weight: 0.05,
                            learning_rate: 0.01,
                            max_iterations: 20,
                            convergence_threshold: 1e-4,
                            beta_schedule: *schedule,
                        },
                    };
                    
                    b.iter(|| {
                        let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, config.clone());
                        quantizer.group_timesteps(100);
                        let params = quantizer.compute_params_per_group(&stats);
                        
                        let result = quantizer.quantize_layer(black_box(&weights), &params);
                        black_box(result)
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark overhead comparison: Baseline vs Phase 1 vs Phase 2 vs Phase 3
fn bench_overhead_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("thermodynamic_overhead");
    
    let size = 4096; // Medium layer size
    let stats = create_realistic_stats(100, true);
    let weights = create_realistic_weights(size);
    
    // Baseline
    group.bench_function("baseline", |b| {
        b.iter(|| {
            let mut quantizer = TimeAwareQuantizer::new(4);
            quantizer.group_timesteps(100);
            let params = quantizer.compute_params_per_group(&stats);
            let result = quantizer.quantize_layer(black_box(&weights), &params);
            black_box(result)
        });
    });
    
    // Phase 1: Validation only (target: <1% overhead)
    group.bench_function("phase1_validation", |b| {
        let config = ThermodynamicConfig {
            validation: ValidationConfig {
                enabled: true,
                smoothness_threshold: 0.3,
                log_violations: false,
            },
            ..Default::default()
        };
        
        b.iter(|| {
            let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, config.clone());
            quantizer.group_timesteps(100);
            let params = quantizer.compute_params_per_group(&stats);
            let result = quantizer.quantize_layer(black_box(&weights), &params);
            black_box(result)
        });
    });
    
    // Phase 2: Validation + Smoothing (target: <10% overhead)
    group.bench_function("phase2_smoothing", |b| {
        let config = ThermodynamicConfig {
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
            ..Default::default()
        };
        
        b.iter(|| {
            let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, config.clone());
            quantizer.group_timesteps(100);
            let params = quantizer.compute_params_per_group(&stats);
            let result = quantizer.quantize_layer(black_box(&weights), &params);
            black_box(result)
        });
    });
    
    // Phase 3: Full pipeline (target: <25% total overhead)
    group.bench_function("phase3_full", |b| {
        let config = ThermodynamicConfig {
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
            transition_optimization: TransitionOptimizationConfig {
                enabled: true,
                markov_weight: 0.1,
                entropy_weight: 0.05,
                learning_rate: 0.01,
                max_iterations: 20,
                convergence_threshold: 1e-4,
                beta_schedule: BetaSchedule::Linear,
            },
        };
        
        b.iter(|| {
            let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, config.clone());
            quantizer.group_timesteps(100);
            let params = quantizer.compute_params_per_group(&stats);
            let result = quantizer.quantize_layer(black_box(&weights), &params);
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark quantization quality (measures how well parameters are optimized)
fn bench_quantization_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("thermodynamic_quantization_quality");
    
    let size = 4096;
    let stats = create_realistic_stats(100, true);
    let weights = create_realistic_weights(size);
    
    // Helper function to measure quantization quality via loss
    let measure_quality = |config: ThermodynamicConfig| -> f32 {
        let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, config);
        quantizer.group_timesteps(100);
        let params = quantizer.compute_params_per_group(&stats);
        
        // Quantize
        let _ = quantizer.quantize_layer(&weights, &params).unwrap();
        
        // Get final loss from metrics (lower is better)
        if let Some(metrics) = quantizer.get_thermodynamic_metrics() {
            metrics.final_loss
        } else {
            f32::MAX // No optimization, return high loss
        }
    };
    
    // Baseline quality
    group.bench_function("baseline_quality", |b| {
        b.iter(|| {
            let config = ThermodynamicConfig::default();
            black_box(measure_quality(config))
        });
    });
    
    // Phase 2 quality (should have lower loss due to smoothing)
    group.bench_function("phase2_quality", |b| {
        let config = ThermodynamicConfig {
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
            ..Default::default()
        };
        
        b.iter(|| black_box(measure_quality(config.clone())));
    });
    
    // Phase 3 quality (should have lowest loss due to optimization)
    group.bench_function("phase3_quality", |b| {
        let config = ThermodynamicConfig {
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
            transition_optimization: TransitionOptimizationConfig {
                enabled: true,
                markov_weight: 0.1,
                entropy_weight: 0.05,
                learning_rate: 0.01,
                max_iterations: 20,
                convergence_threshold: 1e-4,
                beta_schedule: BetaSchedule::Linear,
            },
        };
        
        b.iter(|| black_box(measure_quality(config.clone())));
    });
    
    group.finish();
}

/// Benchmark Markov smoothness score improvement
fn bench_markov_smoothness(c: &mut Criterion) {
    let mut group = c.benchmark_group("thermodynamic_markov_smoothness");
    
    let size = 4096;
    let stats = create_realistic_stats(100, true);
    let weights = create_realistic_weights(size);
    
    // Helper function to get Markov score
    let get_markov_score = |config: ThermodynamicConfig| -> f32 {
        let mut quantizer = TimeAwareQuantizer::with_thermodynamic_config(4, config);
        quantizer.group_timesteps(100);
        let params = quantizer.compute_params_per_group(&stats);
        
        let _ = quantizer.quantize_layer(&weights, &params);
        
        if let Some(metrics) = quantizer.get_thermodynamic_metrics() {
            metrics.smoothness_score
        } else {
            0.0
        }
    };
    
    // Baseline (expected: ~0.65-0.78)
    group.bench_function("baseline_score", |b| {
        let config = ThermodynamicConfig {
            validation: ValidationConfig {
                enabled: true,
                smoothness_threshold: 0.3,
                log_violations: false,
            },
            ..Default::default()
        };
        
        b.iter(|| black_box(get_markov_score(config.clone())));
    });
    
    // Phase 2 (target: 0.82+)
    group.bench_function("phase2_score", |b| {
        let config = ThermodynamicConfig {
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
            ..Default::default()
        };
        
        b.iter(|| black_box(get_markov_score(config.clone())));
    });
    
    // Phase 3 (target: 0.90+)
    group.bench_function("phase3_score", |b| {
        let config = ThermodynamicConfig {
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
            transition_optimization: TransitionOptimizationConfig {
                enabled: true,
                markov_weight: 0.1,
                entropy_weight: 0.05,
                learning_rate: 0.01,
                max_iterations: 20,
                convergence_threshold: 1e-4,
                beta_schedule: BetaSchedule::Linear,
            },
        };
        
        b.iter(|| black_box(get_markov_score(config.clone())));
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_baseline,
    bench_phase1_validation,
    bench_phase2_smoothing,
    bench_phase3_full_pipeline,
    bench_overhead_comparison,
    bench_quantization_quality,
    bench_markov_smoothness,
);
criterion_main!(benches);

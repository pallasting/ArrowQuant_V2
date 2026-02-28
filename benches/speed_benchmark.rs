/// Quantization Speed Benchmarks
///
/// This benchmark suite measures quantization speed for different model sizes:
/// - 100M parameter model (target: <2 minutes = 120s)
/// - 600M parameter model (target: <10 minutes = 600s)
/// - 7B parameter model (Dream 7B target: <5 minutes = 300s)
///
/// Target: 5-10x speedup vs Python implementation

use arrow_quant_v2::{DiffusionOrchestrator, DiffusionQuantConfig};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use std::fs;
use tempfile::TempDir;

/// Model configuration for benchmarking
#[derive(Clone)]
struct ModelConfig {
    name: &'static str,
    num_params: usize,
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
}

/// Model configurations for different sizes
const MODEL_CONFIGS: &[ModelConfig] = &[
    ModelConfig {
        name: "100M",
        num_params: 100_000_000,
        num_layers: 12,
        hidden_size: 768,
        intermediate_size: 3072,
    },
    ModelConfig {
        name: "600M",
        num_params: 600_000_000,
        num_layers: 24,
        hidden_size: 1024,
        intermediate_size: 4096,
    },
    ModelConfig {
        name: "7B",
        num_params: 7_000_000_000,
        num_layers: 32,
        hidden_size: 4096,
        intermediate_size: 11008,
    },
];

/// Create a synthetic model with specified configuration
fn create_synthetic_model(config: &ModelConfig) -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json
    let metadata = serde_json::json!({
        "modality": "text",
        "num_params": config.num_params,
        "num_layers": config.num_layers,
        "hidden_size": config.hidden_size,
    });
    fs::write(
        model_path.join("metadata.json"),
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .unwrap();

    // Create synthetic layer files
    // Embedding layer
    let _embed_weights = Array2::from_shape_fn((50000, config.hidden_size), |(i, j)| {
        ((i * j) as f32 / 1000.0).sin() * 0.5
    });

    // Transformer layers
    for i in 0..config.num_layers {
        // Self-attention projections
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let layer_name = format!("layer_{}_attn_{}.parquet", i, proj);
            let layer_path = model_path.join(&layer_name);
            
            let _weights = Array2::from_shape_fn(
                (config.hidden_size, config.hidden_size),
                |(i, j)| ((i * j) as f32 / 1000.0).sin() * 0.5,
            );
            
            // Write placeholder (in real implementation would use proper Parquet schema)
            fs::write(&layer_path, b"synthetic_layer_data").unwrap();
        }

        // MLP layers
        for proj in &["gate_proj", "up_proj"] {
            let layer_name = format!("layer_{}_mlp_{}.parquet", i, proj);
            let layer_path = model_path.join(&layer_name);
            
            let _weights = Array2::from_shape_fn(
                (config.intermediate_size, config.hidden_size),
                |(i, j)| ((i * j) as f32 / 1000.0).sin() * 0.5,
            );
            
            fs::write(&layer_path, b"synthetic_layer_data").unwrap();
        }

        let layer_name = format!("layer_{}_mlp_down_proj.parquet", i);
        let layer_path = model_path.join(&layer_name);
        
        let _weights = Array2::from_shape_fn(
            (config.hidden_size, config.intermediate_size),
            |(i, j)| ((i * j) as f32 / 1000.0).sin() * 0.5,
        );
        
        fs::write(&layer_path, b"synthetic_layer_data").unwrap();
    }

    // LM head
    let _lm_head_weights = Array2::from_shape_fn((50000, config.hidden_size), |(i, j)| {
        ((i * j) as f32 / 1000.0).sin() * 0.5
    });

    temp_dir
}

/// Benchmark quantization speed for different model sizes
fn bench_quantization_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_speed");
    
    // Set longer measurement time for large models
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));

    for config in MODEL_CONFIGS.iter() {
        group.throughput(Throughput::Elements(config.num_params as u64));

        group.bench_with_input(
            BenchmarkId::new("model_size", config.name),
            config,
            |b, cfg| {
                b.iter(|| {
                    let model_dir = create_synthetic_model(cfg);
                    let output_dir = TempDir::new().unwrap();

                    let quant_config = DiffusionQuantConfig {
                        bit_width: 2,
                        num_time_groups: 10,
                        group_size: 128,
                        enable_time_aware: true,
                        enable_spatial: true,
                        min_accuracy: 0.70,
                        calibration_samples: 32,
                        ..Default::default()
                    };

                    let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

                    let _ = black_box(orchestrator.quantize_model(
                        model_dir.path(),
                        output_dir.path(),
                    ));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark quantization speed with different bit widths
fn bench_bit_width_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_width_comparison");
    
    // Use 100M model for bit width comparison
    let config = &MODEL_CONFIGS[0];
    
    group.throughput(Throughput::Elements(config.num_params as u64));

    for bit_width in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("bit_width", format!("INT{}", bit_width)),
            bit_width,
            |b, &bw| {
                b.iter(|| {
                    let model_dir = create_synthetic_model(config);
                    let output_dir = TempDir::new().unwrap();

                    let quant_config = DiffusionQuantConfig {
                        bit_width: bw,
                        num_time_groups: 10,
                        group_size: 128,
                        enable_time_aware: true,
                        enable_spatial: true,
                        min_accuracy: 0.70,
                        calibration_samples: 32,
                        ..Default::default()
                    };

                    let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

                    let _ = black_box(orchestrator.quantize_model(
                        model_dir.path(),
                        output_dir.path(),
                    ));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Dream 7B quantization (specific target: <5 minutes)
fn bench_dream_7b_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("dream_7b_target");
    
    // Dream 7B configuration
    let config = &MODEL_CONFIGS[2]; // 7B model
    
    group.throughput(Throughput::Elements(config.num_params as u64));
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(120));

    group.bench_function("dream_7b_int2", |b| {
        b.iter(|| {
            let model_dir = create_synthetic_model(config);
            let output_dir = TempDir::new().unwrap();

            let quant_config = DiffusionQuantConfig {
                bit_width: 2,
                num_time_groups: 10,
                group_size: 128,
                enable_time_aware: true,
                enable_spatial: true,
                min_accuracy: 0.70,
                calibration_samples: 32,
                ..Default::default()
            };

            let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

            let _ = black_box(orchestrator.quantize_model(
                model_dir.path(),
                output_dir.path(),
            ));
        });
    });

    group.finish();
}

/// Benchmark with different optimization levels
fn bench_optimization_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_levels");
    
    // Use 100M model for optimization comparison
    let config = &MODEL_CONFIGS[0];
    
    group.throughput(Throughput::Elements(config.num_params as u64));

    // Baseline: No optimizations
    group.bench_function("baseline_no_opt", |b| {
        b.iter(|| {
            let model_dir = create_synthetic_model(config);
            let output_dir = TempDir::new().unwrap();

            let quant_config = DiffusionQuantConfig {
                bit_width: 2,
                num_time_groups: 10,
                group_size: 128,
                enable_time_aware: false,
                enable_spatial: false,
                min_accuracy: 0.70,
                calibration_samples: 32,
                ..Default::default()
            };

            let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

            let _ = black_box(orchestrator.quantize_model(
                model_dir.path(),
                output_dir.path(),
            ));
        });
    });

    // With time-aware optimization
    group.bench_function("with_time_aware", |b| {
        b.iter(|| {
            let model_dir = create_synthetic_model(config);
            let output_dir = TempDir::new().unwrap();

            let quant_config = DiffusionQuantConfig {
                bit_width: 2,
                num_time_groups: 10,
                group_size: 128,
                enable_time_aware: true,
                enable_spatial: false,
                min_accuracy: 0.70,
                calibration_samples: 32,
                ..Default::default()
            };

            let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

            let _ = black_box(orchestrator.quantize_model(
                model_dir.path(),
                output_dir.path(),
            ));
        });
    });

    // With spatial optimization
    group.bench_function("with_spatial", |b| {
        b.iter(|| {
            let model_dir = create_synthetic_model(config);
            let output_dir = TempDir::new().unwrap();

            let quant_config = DiffusionQuantConfig {
                bit_width: 2,
                num_time_groups: 10,
                group_size: 128,
                enable_time_aware: false,
                enable_spatial: true,
                min_accuracy: 0.70,
                calibration_samples: 32,
                ..Default::default()
            };

            let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

            let _ = black_box(orchestrator.quantize_model(
                model_dir.path(),
                output_dir.path(),
            ));
        });
    });

    // With all optimizations
    group.bench_function("with_all_opt", |b| {
        b.iter(|| {
            let model_dir = create_synthetic_model(config);
            let output_dir = TempDir::new().unwrap();

            let quant_config = DiffusionQuantConfig {
                bit_width: 2,
                num_time_groups: 10,
                group_size: 128,
                enable_time_aware: true,
                enable_spatial: true,
                min_accuracy: 0.70,
                calibration_samples: 32,
                ..Default::default()
            };

            let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

            let _ = black_box(orchestrator.quantize_model(
                model_dir.path(),
                output_dir.path(),
            ));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_quantization_speed,
    bench_bit_width_comparison,
    bench_dream_7b_target,
    bench_optimization_levels,
);
criterion_main!(benches);

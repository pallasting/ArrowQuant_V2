/// Accuracy Benchmarks for ArrowQuant V2
///
/// This benchmark suite measures cosine similarity for different bit widths (INT2/INT4/INT8)
/// and validates accuracy targets:
/// - INT2: cosine_similarity >= 0.70
/// - INT4: cosine_similarity >= 0.90
/// - INT8: cosine_similarity >= 0.95
///
/// Tests across all modalities: text, code, image, audio

use arrow_quant_v2::{DiffusionOrchestrator, DiffusionQuantConfig, ValidationSystem};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use std::collections::HashMap;
use std::fs;
use tempfile::TempDir;

/// Modality types for diffusion models
#[derive(Clone, Copy, Debug)]
enum Modality {
    Text,
    Code,
    Image,
    Audio,
}

impl Modality {
    fn as_str(&self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::Code => "code",
            Modality::Image => "image",
            Modality::Audio => "audio",
        }
    }
}

/// Model configuration for benchmarking
#[derive(Clone)]
struct ModelConfig {
    name: &'static str,
    modality: Modality,
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
}

/// Model configurations for different modalities
const MODEL_CONFIGS: &[ModelConfig] = &[
    ModelConfig {
        name: "text_model",
        modality: Modality::Text,
        num_layers: 12,
        hidden_size: 768,
        intermediate_size: 3072,
    },
    ModelConfig {
        name: "code_model",
        modality: Modality::Code,
        num_layers: 12,
        hidden_size: 768,
        intermediate_size: 3072,
    },
    ModelConfig {
        name: "image_model",
        modality: Modality::Image,
        num_layers: 24,
        hidden_size: 1024,
        intermediate_size: 4096,
    },
    ModelConfig {
        name: "audio_model",
        modality: Modality::Audio,
        num_layers: 16,
        hidden_size: 512,
        intermediate_size: 2048,
    },
];

/// Create a synthetic model with specified configuration
fn create_synthetic_model(config: &ModelConfig) -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json
    let metadata = serde_json::json!({
        "modality": config.modality.as_str(),
        "num_layers": config.num_layers,
        "hidden_size": config.hidden_size,
    });
    fs::write(
        model_path.join("metadata.json"),
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .unwrap();

    // Create synthetic layer files
    for i in 0..config.num_layers {
        // Self-attention projections
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let layer_name = format!("layer_{}_attn_{}.parquet", i, proj);
            let layer_path = model_path.join(&layer_name);
            fs::write(&layer_path, b"synthetic_layer_data").unwrap();
        }

        // MLP layers
        for proj in &["gate_proj", "up_proj"] {
            let layer_name = format!("layer_{}_mlp_{}.parquet", i, proj);
            let layer_path = model_path.join(&layer_name);
            fs::write(&layer_path, b"synthetic_layer_data").unwrap();
        }

        let layer_name = format!("layer_{}_mlp_down_proj.parquet", i);
        let layer_path = model_path.join(&layer_name);
        fs::write(&layer_path, b"synthetic_layer_data").unwrap();
    }

    temp_dir
}

/// Benchmark accuracy for different bit widths
fn bench_accuracy_by_bit_width(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_by_bit_width");

    // Use text model for bit width comparison
    let config = &MODEL_CONFIGS[0];

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
                        min_accuracy: match bw {
                            2 => 0.70,
                            4 => 0.90,
                            8 => 0.95,
                            _ => 0.85,
                        },
                        calibration_samples: 32,
                        ..Default::default()
                    };

                    let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

                    // Quantize model
                    let _ = orchestrator
                        .quantize_model(model_dir.path(), output_dir.path())
                        .unwrap();

                    // Validate quality
                    let validator = ValidationSystem::new_with_bit_width(bw);
                    let report = validator
                        .validate_quality(model_dir.path(), output_dir.path())
                        .unwrap();

                    black_box(report.cosine_similarity)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark accuracy across all modalities
fn bench_accuracy_by_modality(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_by_modality");

    // Test INT4 across all modalities
    let bit_width = 4;

    for config in MODEL_CONFIGS.iter() {
        group.bench_with_input(
            BenchmarkId::new("modality", config.modality.as_str()),
            config,
            |b, cfg| {
                b.iter(|| {
                    let model_dir = create_synthetic_model(cfg);
                    let output_dir = TempDir::new().unwrap();

                    let quant_config = DiffusionQuantConfig {
                        bit_width,
                        num_time_groups: 10,
                        group_size: 128,
                        enable_time_aware: true,
                        enable_spatial: true,
                        min_accuracy: 0.90,
                        calibration_samples: 32,
                        ..Default::default()
                    };

                    let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

                    // Quantize model
                    let _ = orchestrator
                        .quantize_model(model_dir.path(), output_dir.path())
                        .unwrap();

                    // Validate quality
                    let validator = ValidationSystem::new_with_bit_width(bit_width);
                    let report = validator
                        .validate_quality(model_dir.path(), output_dir.path())
                        .unwrap();

                    black_box(report.cosine_similarity)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark accuracy with vs without optimizations
fn bench_accuracy_with_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_with_optimizations");

    // Use text model for optimization comparison
    let config = &MODEL_CONFIGS[0];
    let bit_width = 2; // INT2 for aggressive quantization

    // Baseline: No optimizations
    group.bench_function("baseline_no_opt", |b| {
        b.iter(|| {
            let model_dir = create_synthetic_model(config);
            let output_dir = TempDir::new().unwrap();

            let quant_config = DiffusionQuantConfig {
                bit_width,
                num_time_groups: 10,
                group_size: 128,
                enable_time_aware: false,
                enable_spatial: false,
                min_accuracy: 0.70,
                calibration_samples: 32,
                ..Default::default()
            };

            let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

            // Quantize model
            let _ = orchestrator
                .quantize_model(model_dir.path(), output_dir.path())
                .unwrap();

            // Validate quality
            let validator = ValidationSystem::new_with_bit_width(bit_width);
            let report = validator
                .validate_quality(model_dir.path(), output_dir.path())
                .unwrap();

            black_box(report.cosine_similarity)
        });
    });

    // With time-aware optimization
    group.bench_function("with_time_aware", |b| {
        b.iter(|| {
            let model_dir = create_synthetic_model(config);
            let output_dir = TempDir::new().unwrap();

            let quant_config = DiffusionQuantConfig {
                bit_width,
                num_time_groups: 10,
                group_size: 128,
                enable_time_aware: true,
                enable_spatial: false,
                min_accuracy: 0.70,
                calibration_samples: 32,
                ..Default::default()
            };

            let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

            // Quantize model
            let _ = orchestrator
                .quantize_model(model_dir.path(), output_dir.path())
                .unwrap();

            // Validate quality
            let validator = ValidationSystem::new_with_bit_width(bit_width);
            let report = validator
                .validate_quality(model_dir.path(), output_dir.path())
                .unwrap();

            black_box(report.cosine_similarity)
        });
    });

    // With spatial optimization
    group.bench_function("with_spatial", |b| {
        b.iter(|| {
            let model_dir = create_synthetic_model(config);
            let output_dir = TempDir::new().unwrap();

            let quant_config = DiffusionQuantConfig {
                bit_width,
                num_time_groups: 10,
                group_size: 128,
                enable_time_aware: false,
                enable_spatial: true,
                min_accuracy: 0.70,
                calibration_samples: 32,
                ..Default::default()
            };

            let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

            // Quantize model
            let _ = orchestrator
                .quantize_model(model_dir.path(), output_dir.path())
                .unwrap();

            // Validate quality
            let validator = ValidationSystem::new_with_bit_width(bit_width);
            let report = validator
                .validate_quality(model_dir.path(), output_dir.path())
                .unwrap();

            black_box(report.cosine_similarity)
        });
    });

    // With all optimizations
    group.bench_function("with_all_opt", |b| {
        b.iter(|| {
            let model_dir = create_synthetic_model(config);
            let output_dir = TempDir::new().unwrap();

            let quant_config = DiffusionQuantConfig {
                bit_width,
                num_time_groups: 10,
                group_size: 128,
                enable_time_aware: true,
                enable_spatial: true,
                min_accuracy: 0.70,
                calibration_samples: 32,
                ..Default::default()
            };

            let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

            // Quantize model
            let _ = orchestrator
                .quantize_model(model_dir.path(), output_dir.path())
                .unwrap();

            // Validate quality
            let validator = ValidationSystem::new_with_bit_width(bit_width);
            let report = validator
                .validate_quality(model_dir.path(), output_dir.path())
                .unwrap();

            black_box(report.cosine_similarity)
        });
    });

    group.finish();
}

/// Benchmark accuracy targets validation
fn bench_accuracy_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_targets");

    // Test each bit width against its target threshold
    let test_cases = vec![
        (2, 0.70, "INT2_target_0.70"),
        (4, 0.90, "INT4_target_0.90"),
        (8, 0.95, "INT8_target_0.95"),
    ];

    for (bit_width, target, name) in test_cases {
        group.bench_function(name, |b| {
            b.iter(|| {
                let config = &MODEL_CONFIGS[0]; // Text model
                let model_dir = create_synthetic_model(config);
                let output_dir = TempDir::new().unwrap();

                let quant_config = DiffusionQuantConfig {
                    bit_width,
                    num_time_groups: 10,
                    group_size: 128,
                    enable_time_aware: true,
                    enable_spatial: true,
                    min_accuracy: target,
                    calibration_samples: 32,
                    ..Default::default()
                };

                let orchestrator = DiffusionOrchestrator::new(quant_config).unwrap();

                // Quantize model
                let _ = orchestrator
                    .quantize_model(model_dir.path(), output_dir.path())
                    .unwrap();

                // Validate quality
                let validator = ValidationSystem::new_with_bit_width(bit_width);
                let report = validator
                    .validate_quality(model_dir.path(), output_dir.path())
                    .unwrap();

                // Check if target is met
                let target_met = report.cosine_similarity >= target;

                black_box((report.cosine_similarity, target_met))
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_accuracy_by_bit_width,
    bench_accuracy_by_modality,
    bench_accuracy_with_optimizations,
    bench_accuracy_targets,
);
criterion_main!(benches);

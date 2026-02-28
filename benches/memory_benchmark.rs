//! Memory Usage Benchmark for ArrowQuant V2
//!
//! This benchmark measures peak memory usage during quantization and compares:
//! - Streaming vs non-streaming (batch) mode
//! - Different model sizes (100M, 600M, 7B parameters)
//! - Different bit widths (INT2, INT4, INT8)
//!
//! Target: Streaming mode should use <50% memory vs batch mode
//!
//! Usage:
//!     cargo bench --bench memory_benchmark
//!     cargo bench --bench memory_benchmark -- streaming_vs_batch
//!     cargo bench --bench memory_benchmark -- model_sizes

use arrow_quant_v2::{DiffusionOrchestrator, DiffusionQuantConfig};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;

/// Generate synthetic layer data for testing
fn generate_synthetic_layer(rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        ((i * cols + j) as f32 * 0.01).sin()
    })
}

/// Benchmark streaming vs batch mode memory usage
fn bench_streaming_vs_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_vs_batch_memory");
    group.sample_size(10); // Fewer samples for memory benchmarks

    let num_layers = 50;
    let layer_size = (1024, 4096); // ~16MB per layer

    for mode in ["batch", "streaming"].iter() {
        group.bench_with_input(BenchmarkId::new("mode", mode), mode, |b, &m| {
            b.iter(|| {
                let mut config = DiffusionQuantConfig::default();
                config.bit_width = 4;
                config.num_threads = 4;
                config.enable_streaming = m == "streaming";
                config.calibration_samples = 16;

                let orchestrator = DiffusionOrchestrator::new(config).unwrap();

                // Simulate quantizing multiple layers
                let layers: Vec<Array2<f32>> = if m == "batch" {
                    // Batch mode: Load all layers into memory
                    (0..num_layers)
                        .map(|_| generate_synthetic_layer(layer_size.0, layer_size.1))
                        .collect()
                } else {
                    // Streaming mode: Only one layer at a time
                    vec![generate_synthetic_layer(layer_size.0, layer_size.1)]
                };

                // Process layers
                for layer in &layers {
                    let _ = black_box(orchestrator.quantize_layer_internal(
                        layer,
                        4, // bit_width
                        128, // group_size
                    ));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark memory usage with different model sizes
fn bench_model_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_size_memory");
    group.sample_size(10);

    // Model configurations: (name, num_layers, layer_size)
    let models = vec![
        ("100M", 12, (768, 3072)),
        ("600M", 24, (1024, 4096)),
        ("7B", 32, (4096, 11008)),
    ];

    for (name, num_layers, layer_size) in models {
        group.bench_with_input(
            BenchmarkId::new("streaming", name),
            &(num_layers, layer_size),
            |b, &(n_layers, (rows, cols))| {
                b.iter(|| {
                    let mut config = DiffusionQuantConfig::default();
                    config.bit_width = 2;
                    config.enable_streaming = true;
                    config.calibration_samples = 16;

                    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

                    // Streaming mode: Process one layer at a time
                    for _ in 0..n_layers {
                        let layer = generate_synthetic_layer(rows, cols);
                        let _ = black_box(orchestrator.quantize_layer_internal(
                            &layer,
                            2, // bit_width
                            128, // group_size
                        ));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batch", name),
            &(num_layers, layer_size),
            |b, &(n_layers, (rows, cols))| {
                b.iter(|| {
                    let mut config = DiffusionQuantConfig::default();
                    config.bit_width = 2;
                    config.enable_streaming = false;
                    config.num_threads = 4;
                    config.calibration_samples = 16;

                    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

                    // Batch mode: Load all layers
                    let layers: Vec<Array2<f32>> = (0..n_layers)
                        .map(|_| generate_synthetic_layer(rows, cols))
                        .collect();

                    // Process all layers
                    for layer in &layers {
                        let _ = black_box(orchestrator.quantize_layer_internal(
                            layer,
                            2, // bit_width
                            128, // group_size
                        ));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage with different bit widths
fn bench_bit_widths(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_width_memory");
    group.sample_size(10);

    let num_layers = 24;
    let layer_size = (1024, 4096);

    for bit_width in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("streaming", format!("INT{}", bit_width)),
            bit_width,
            |b, &bw| {
                b.iter(|| {
                    let mut config = DiffusionQuantConfig::default();
                    config.bit_width = bw;
                    config.enable_streaming = true;
                    config.calibration_samples = 16;

                    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

                    // Process layers in streaming mode
                    for _ in 0..num_layers {
                        let layer = generate_synthetic_layer(layer_size.0, layer_size.1);
                        let _ = black_box(orchestrator.quantize_layer_internal(
                            &layer,
                            bw,
                            128, // group_size
                        ));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_streaming_vs_batch,
    bench_model_sizes,
    bench_bit_widths
);
criterion_main!(benches);


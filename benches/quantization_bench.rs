//! Performance benchmarks for ArrowQuant V2

use arrow_quant_v2::{DiffusionQuantConfig, SpatialQuantizer, TimeAwareQuantizer};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;

fn bench_time_aware_grouping(c: &mut Criterion) {
    c.bench_function("time_aware_grouping", |b| {
        b.iter(|| {
            let mut quantizer = TimeAwareQuantizer::new(10);
            quantizer.group_timesteps(black_box(1000));
        });
    });
}

fn bench_spatial_quantization(c: &mut Criterion) {
    let quantizer = SpatialQuantizer::new(128);
    let weights = Array2::from_shape_fn((256, 512), |(i, j)| (i + j) as f32 / 1000.0);

    c.bench_function("spatial_per_group_quantize", |b| {
        b.iter(|| {
            let _ = quantizer.per_group_quantize(black_box(&weights));
        });
    });
}

fn bench_channel_equalization(c: &mut Criterion) {
    let quantizer = SpatialQuantizer::new(128);
    let weights = Array2::from_shape_fn((256, 512), |(i, j)| (i + j) as f32 / 1000.0);
    let activations = Array2::from_shape_fn((256, 512), |(i, j)| (i * j) as f32 / 1000.0);

    c.bench_function("channel_equalization", |b| {
        b.iter(|| {
            let _ = quantizer.channel_equalization(black_box(&weights), black_box(&activations));
        });
    });
}

criterion_group!(
    benches,
    bench_time_aware_grouping,
    bench_spatial_quantization,
    bench_channel_equalization
);
criterion_main!(benches);

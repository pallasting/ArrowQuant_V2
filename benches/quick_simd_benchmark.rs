//! Quick SIMD Performance Benchmark
//!
//! Simplified benchmark for faster execution while measuring key metrics

use arrow_quant_v2::simd::{quantize_simd, dequantize_simd, cosine_similarity_simd};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_quantize_quick(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_quick");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));

    // Test representative sizes
    for size in [256, 1024, 4096].iter() {
        let data: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.01).collect();
        let scale = 0.1;
        let zero_point = 0.0;

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                let _ = quantize_simd(black_box(&data), black_box(scale), black_box(zero_point));
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            b.iter(|| {
                let result: Vec<u8> = data
                    .iter()
                    .map(|&value| {
                        ((value / scale) + zero_point).round().clamp(0.0, 255.0) as u8
                    })
                    .collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_dequantize_quick(c: &mut Criterion) {
    let mut group = c.benchmark_group("dequantize_quick");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));

    for size in [256, 1024, 4096].iter() {
        let data: Vec<u8> = (0..*size).map(|i| (i % 256) as u8).collect();
        let scale = 0.1;
        let zero_point = 0.0;

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                let _ = dequantize_simd(black_box(&data), black_box(scale), black_box(zero_point));
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            b.iter(|| {
                let result: Vec<f32> = data
                    .iter()
                    .map(|&value| (value as f32 - zero_point) * scale)
                    .collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_cosine_similarity_quick(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity_quick");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));

    for size in [256, 1024, 4096].iter() {
        let a: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.02).collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bencher, _| {
            bencher.iter(|| {
                let _ = cosine_similarity_simd(black_box(&a), black_box(&b));
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bencher, _| {
            bencher.iter(|| {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let result = if norm_a == 0.0 || norm_b == 0.0 {
                    0.0
                } else {
                    dot / (norm_a * norm_b)
                };
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    quick_benches,
    bench_quantize_quick,
    bench_dequantize_quick,
    bench_cosine_similarity_quick,
);
criterion_main!(quick_benches);

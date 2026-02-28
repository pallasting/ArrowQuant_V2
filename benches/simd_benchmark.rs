//! SIMD Performance Benchmarks
//!
//! This benchmark suite measures the performance improvement of SIMD-accelerated
//! quantization operations compared to scalar implementations.
//!
//! Expected speedup: 2-4x with SIMD (AVX2 on x86_64, NEON on ARM64)

use arrow_quant_v2::simd::{
    cosine_similarity_simd, dequantize_simd, quantize_simd,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// ============================================================================
// Quantization Benchmarks
// ============================================================================

fn bench_quantize_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize");

    // Test different array sizes to show SIMD scaling
    for size in [64, 256, 1024, 4096, 16384].iter() {
        let data: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.01).collect();
        let scale = 0.1;
        let zero_point = 0.0;

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                let _ = quantize_simd(black_box(&data), black_box(scale), black_box(zero_point));
            });
        });

        // Scalar baseline for comparison
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

// ============================================================================
// Dequantization Benchmarks
// ============================================================================

fn bench_dequantize_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("dequantize");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        let data: Vec<u8> = (0..*size).map(|i| (i % 256) as u8).collect();
        let scale = 0.1;
        let zero_point = 0.0;

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                let _ =
                    dequantize_simd(black_box(&data), black_box(scale), black_box(zero_point));
            });
        });

        // Scalar baseline for comparison
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

// ============================================================================
// Cosine Similarity Benchmarks
// ============================================================================

fn bench_cosine_similarity_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        let a: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.02).collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bencher, _| {
            bencher.iter(|| {
                let _ = cosine_similarity_simd(black_box(&a), black_box(&b));
            });
        });

        // Scalar baseline for comparison
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bencher, _| {
            bencher.iter(|| {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
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

// ============================================================================
// Roundtrip Benchmarks (Quantize + Dequantize)
// ============================================================================

fn bench_roundtrip_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");

    for size in [256, 1024, 4096].iter() {
        let data: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.01).collect();
        let scale = 0.1;
        let zero_point = 0.0;

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                let quantized =
                    quantize_simd(black_box(&data), black_box(scale), black_box(zero_point));
                let dequantized =
                    dequantize_simd(black_box(&quantized), black_box(scale), black_box(zero_point));
                black_box(dequantized);
            });
        });

        // Scalar baseline for comparison
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            b.iter(|| {
                let quantized: Vec<u8> = data
                    .iter()
                    .map(|&value| {
                        ((value / scale) + zero_point).round().clamp(0.0, 255.0) as u8
                    })
                    .collect();
                let dequantized: Vec<f32> = quantized
                    .iter()
                    .map(|&value| (value as f32 - zero_point) * scale)
                    .collect();
                black_box(dequantized);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Layer Quantization Benchmark (Realistic Workload)
// ============================================================================

fn bench_layer_quantization_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_quantization");

    // Simulate realistic layer sizes for diffusion models
    // - Small: 768 (embedding dimension)
    // - Medium: 3072 (MLP hidden dimension)
    // - Large: 12288 (large model MLP)
    for (name, rows, cols) in [
        ("small_768x768", 768, 768),
        ("medium_3072x768", 3072, 768),
        ("large_12288x3072", 12288, 3072),
    ]
    .iter()
    {
        let size = rows * cols;
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.0001).collect();
        let scale = 0.1;
        let zero_point = 0.0;

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("simd", name), name, |b, _| {
            b.iter(|| {
                let _ = quantize_simd(black_box(&data), black_box(scale), black_box(zero_point));
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", name), name, |b, _| {
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

criterion_group!(
    simd_benches,
    bench_quantize_simd_vs_scalar,
    bench_dequantize_simd_vs_scalar,
    bench_cosine_similarity_simd_vs_scalar,
    bench_roundtrip_simd_vs_scalar,
    bench_layer_quantization_realistic,
);
criterion_main!(simd_benches);

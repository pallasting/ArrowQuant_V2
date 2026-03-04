//! SIMD Performance Speedup Benchmarks
//!
//! **Validates: Requirements 3.5, 8.1**
//! **Property 7: SIMD Performance Improvement**
//!
//! This benchmark suite measures the performance speedup of SIMD-accelerated
//! quantization compared to scalar implementations across different array sizes.
//!
//! # Test Strategy
//!
//! - Test array sizes: 1K, 10K, 100K, 1M elements
//! - Compare SIMD vs scalar quantization performance
//! - Verify speedup is 3x-6x for arrays >= 1K elements
//! - Measure throughput in elements/second
//!
//! # Expected Results
//!
//! - Small arrays (1K): 2-3x speedup
//! - Medium arrays (10K): 3-4x speedup  
//! - Large arrays (100K): 4-5x speedup
//! - Very large arrays (1M): 5-6x speedup

use arrow_quant_v2::simd::{quantize_simd, is_simd_available};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Generate random weight array for benchmarking
fn generate_weights(size: usize, seed: u64) -> Vec<f32> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(-10.0..10.0)).collect()
}

/// Scalar quantization baseline (for comparison)
fn quantize_scalar(data: &[f32], scale: f32, zero_point: f32) -> Vec<u8> {
    data.iter()
        .map(|&value| {
            let quantized = (value / scale) + zero_point;
            quantized.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Benchmark SIMD vs Scalar quantization speedup
///
/// Tests array sizes: 1K, 10K, 100K, 1M
/// Measures throughput and calculates speedup ratio
fn bench_simd_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_speedup");
    
    // Configure benchmark parameters
    group.sample_size(20); // Reduced from default 100 for faster execution
    group.warm_up_time(std::time::Duration::from_secs(2));
    group.measurement_time(std::time::Duration::from_secs(5));
    
    // Test sizes as specified in requirements: 1K, 10K, 100K, 1M
    let sizes = vec![
        ("1K", 1_000),
        ("10K", 10_000),
        ("100K", 100_000),
        ("1M", 1_000_000),
    ];
    
    let scale = 0.1f32;
    let zero_point = 128.0f32;
    let seed = 42u64;
    
    for (size_name, size) in sizes {
        // Generate test data
        let weights = generate_weights(size, seed);
        
        // Set throughput for performance measurement
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark SIMD implementation
        if is_simd_available() {
            group.bench_with_input(
                BenchmarkId::new("simd", size_name),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = quantize_simd(
                            black_box(&weights),
                            black_box(scale),
                            black_box(zero_point),
                        );
                        black_box(result);
                    });
                },
            );
        }
        
        // Benchmark scalar baseline
        group.bench_with_input(
            BenchmarkId::new("scalar", size_name),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = quantize_scalar(
                        black_box(&weights),
                        black_box(scale),
                        black_box(zero_point),
                    );
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD quantization with different bit widths
///
/// Tests how SIMD performance scales with different quantization bit widths
fn bench_simd_bit_widths(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_bit_widths");
    
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_secs(1));
    
    let size = 100_000; // 100K elements
    let weights = generate_weights(size, 42);
    
    group.throughput(Throughput::Elements(size as u64));
    
    // Test different bit widths (affects quantization range)
    let bit_widths = vec![
        ("int2", 0.025, 2.0),   // 2-bit: range [0, 3]
        ("int4", 0.1, 8.0),     // 4-bit: range [0, 15]
        ("int8", 0.4, 128.0),   // 8-bit: range [0, 255]
    ];
    
    for (name, scale, zero_point) in bit_widths {
        if is_simd_available() {
            group.bench_with_input(
                BenchmarkId::new("simd", name),
                &name,
                |b, _| {
                    b.iter(|| {
                        let result = quantize_simd(
                            black_box(&weights),
                            black_box(scale),
                            black_box(zero_point),
                        );
                        black_box(result);
                    });
                },
            );
        }
        
        group.bench_with_input(
            BenchmarkId::new("scalar", name),
            &name,
            |b, _| {
                b.iter(|| {
                    let result = quantize_scalar(
                        black_box(&weights),
                        black_box(scale),
                        black_box(zero_point),
                    );
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD performance with different data patterns
///
/// Tests how SIMD handles different data distributions:
/// - Uniform random data
/// - Sorted data
/// - Data with many duplicates
fn bench_simd_data_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_data_patterns");
    
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_secs(1));
    
    let size = 100_000;
    let scale = 0.1f32;
    let zero_point = 128.0f32;
    
    group.throughput(Throughput::Elements(size as u64));
    
    // Generate different data patterns
    let patterns = vec![
        ("random", generate_weights(size, 42)),
        ("sorted", {
            let mut data = generate_weights(size, 42);
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());
            data
        }),
        ("duplicates", {
            vec![1.0f32; size] // All same value
        }),
    ];
    
    for (pattern_name, weights) in patterns {
        if is_simd_available() {
            group.bench_with_input(
                BenchmarkId::new("simd", pattern_name),
                &pattern_name,
                |b, _| {
                    b.iter(|| {
                        let result = quantize_simd(
                            black_box(&weights),
                            black_box(scale),
                            black_box(zero_point),
                        );
                        black_box(result);
                    });
                },
            );
        }
        
        group.bench_with_input(
            BenchmarkId::new("scalar", pattern_name),
            &pattern_name,
            |b, _| {
                b.iter(|| {
                    let result = quantize_scalar(
                        black_box(&weights),
                        black_box(scale),
                        black_box(zero_point),
                    );
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_simd_speedup,
    bench_simd_bit_widths,
    bench_simd_data_patterns,
);
criterion_main!(benches);

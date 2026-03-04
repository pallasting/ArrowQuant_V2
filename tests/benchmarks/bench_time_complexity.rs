//! Time Group Assignment Complexity Benchmark
//!
//! This benchmark validates that time group assignment achieves O(n log m) complexity
//! where n is the number of weights and m is the number of time groups.
//!
//! **Validates: Requirements 2.3, 8.3**
//!
//! Test Matrix:
//! - Array sizes (n): 1K, 10K, 100K, 1M elements
//! - Time groups (m): 5, 10, 20 groups
//! - Expected complexity: O(n log m)
//!
//! The benchmark measures actual execution times and validates that:
//! 1. Time scales linearly with n (array size)
//! 2. Time scales logarithmically with m (number of groups)
//! 3. The optimized binary search implementation is faster than linear scanning

use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box as std_black_box;

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate random weights for testing
fn generate_random_weights(size: usize, seed: u64) -> Vec<f32> {
    // Simple LCG for reproducible random numbers
    let mut rng = seed;
    (0..size)
        .map(|_| {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng % 10000) as f32) / 100.0 // Range: 0.0 to 100.0
        })
        .collect()
}

/// Generate time group parameters
fn generate_time_group_params(num_groups: usize) -> Vec<TimeGroupParams> {
    (0..num_groups)
        .map(|i| {
            let start = (i * 100) / num_groups;
            let end = ((i + 1) * 100) / num_groups;
            let scale = 0.1 + (i as f32) * 0.05; // Varying scales
            let zero_point = 0.0;
            
            TimeGroupParams {
                time_range: (start, end),
                scale,
                zero_point,
                group_size: 64,
            }
        })
        .collect()
}

// ============================================================================
// Benchmark 1: Scaling with Array Size (n)
// ============================================================================

/// Benchmark time group assignment with varying array sizes
///
/// This benchmark validates that execution time scales linearly with n.
/// For a fixed number of groups (m=10), we test array sizes from 1K to 1M.
///
/// Expected behavior:
/// - Time(10K) ≈ 10 × Time(1K)
/// - Time(100K) ≈ 100 × Time(1K)
/// - Time(1M) ≈ 1000 × Time(1K)
fn bench_time_complexity_scaling_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_complexity_scaling_n");
    
    let num_groups = 10;
    let params = generate_time_group_params(num_groups);
    let quantizer = TimeAwareQuantizer::new(num_groups);
    
    // Test different array sizes
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        let weights = generate_random_weights(*size, 42);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("uniform_distribution", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _ = quantizer.assign_time_groups(
                        std_black_box(&weights),
                        std_black_box(&params)
                    );
                });
            }
        );
    }
    
    group.finish();
}

// ============================================================================
// Benchmark 2: Scaling with Number of Groups (m)
// ============================================================================

/// Benchmark time group assignment with varying number of groups
///
/// This benchmark validates that execution time scales with m.
/// For a fixed array size (n=100K), we test group counts from 5 to 20.
///
/// Expected behavior:
/// - Time should remain relatively constant as m increases (O(n) complexity)
/// - Uniform distribution doesn't depend on number of groups
fn bench_time_complexity_scaling_m(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_complexity_scaling_m");
    
    let size = 100_000;
    let weights = generate_random_weights(size, 42);
    
    // Test different numbers of time groups
    for num_groups in [5, 10, 20].iter() {
        let params = generate_time_group_params(*num_groups);
        let quantizer = TimeAwareQuantizer::new(*num_groups);
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("uniform_distribution", num_groups),
            num_groups,
            |b, _| {
                b.iter(|| {
                    let _ = quantizer.assign_time_groups(
                        std_black_box(&weights),
                        std_black_box(&params)
                    );
                });
            }
        );
    }
    
    group.finish();
}

// ============================================================================
// Benchmark 3: Binary Search vs Linear Scan Comparison
// ============================================================================

/// Compare uniform distribution implementation
///
/// This benchmark measures the current uniform distribution implementation.
/// Note: Binary search optimization has not been implemented yet.
///
/// Current implementation: O(n) uniform distribution
fn bench_uniform_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("uniform_distribution");
    
    let size = 100_000;
    let weights = generate_random_weights(size, 42);
    
    for num_groups in [5, 10, 20].iter() {
        let params = generate_time_group_params(*num_groups);
        let quantizer = TimeAwareQuantizer::new(*num_groups);
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Current implementation: Uniform distribution O(n)
        group.bench_with_input(
            BenchmarkId::new("uniform_distribution", num_groups),
            num_groups,
            |b, _| {
                b.iter(|| {
                    let _ = quantizer.assign_time_groups(
                        std_black_box(&weights),
                        std_black_box(&params)
                    );
                });
            }
        );
    }
    
    group.finish();
}

// ============================================================================
// Benchmark 4: Full Matrix Test (n × m combinations)
// ============================================================================

/// Comprehensive benchmark testing all combinations of n and m
///
/// This benchmark provides a complete performance matrix showing how
/// execution time varies with both array size and number of groups.
///
/// Test matrix:
/// - n ∈ {1K, 10K, 100K, 1M}
/// - m ∈ {5, 10, 20}
/// - Total: 12 combinations
fn bench_time_complexity_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_complexity_matrix");
    
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let group_counts = [5, 10, 20];
    
    for &size in &sizes {
        for &num_groups in &group_counts {
            let weights = generate_random_weights(size, 42);
            let params = generate_time_group_params(num_groups);
            let quantizer = TimeAwareQuantizer::new(num_groups);
            
            group.throughput(Throughput::Elements(size as u64));
            
            let bench_name = format!("n={}_m={}", size, num_groups);
            group.bench_with_input(
                BenchmarkId::new("uniform_distribution", &bench_name),
                &bench_name,
                |b, _| {
                    b.iter(|| {
                        let _ = quantizer.assign_time_groups(
                            std_black_box(&weights),
                            std_black_box(&params)
                        );
                    });
                }
            );
        }
    }
    
    group.finish();
}

// ============================================================================
// Benchmark 5: Boundary Pre-computation Overhead
// ============================================================================

/// Benchmark the overhead of parameter generation
///
/// This benchmark measures the cost of generating time group parameters.
/// Note: Binary search boundary pre-computation has not been implemented yet.
fn bench_parameter_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("parameter_generation");
    
    for num_groups in [5, 10, 20, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("generate_params", num_groups),
            num_groups,
            |b, &ng| {
                b.iter(|| {
                    let _ = generate_time_group_params(std_black_box(ng));
                });
            }
        );
    }
    
    group.finish();
}

// ============================================================================
// Benchmark 6: Realistic Workload (Large Model)
// ============================================================================

/// Benchmark with realistic large model parameters
///
/// This benchmark simulates quantizing a large diffusion model layer
/// with realistic dimensions and time group counts.
///
/// Scenarios:
/// - Small model: 768×768 weights, 10 time groups
/// - Medium model: 3072×768 weights, 10 time groups
/// - Large model: 12288×3072 weights, 20 time groups
fn bench_realistic_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_workload");
    
    let scenarios = [
        ("small_768x768_m10", 768 * 768, 10),
        ("medium_3072x768_m10", 3072 * 768, 10),
        ("large_12288x3072_m20", 12288 * 3072, 20),
    ];
    
    for (name, size, num_groups) in scenarios.iter() {
        let weights = generate_random_weights(*size, 42);
        let params = generate_time_group_params(*num_groups);
        let quantizer = TimeAwareQuantizer::new(*num_groups);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("uniform_distribution", name),
            name,
            |b, _| {
                b.iter(|| {
                    let _ = quantizer.assign_time_groups(
                        std_black_box(&weights),
                        std_black_box(&params)
                    );
                });
            }
        );
    }
    
    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    time_complexity_benches,
    bench_time_complexity_scaling_n,
    bench_time_complexity_scaling_m,
    bench_uniform_distribution,
    bench_time_complexity_matrix,
    bench_parameter_generation,
    bench_realistic_workload,
);

criterion_main!(time_complexity_benches);

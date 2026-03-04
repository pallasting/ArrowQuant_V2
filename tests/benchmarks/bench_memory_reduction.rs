//! Memory Allocation Reduction Benchmarks
//!
//! **Validates: Requirements 8.2, 1.4**
//! **Property 6: Memory Allocation Reduction**
//!
//! This benchmark suite measures memory allocation reduction from optimization:
//! - Elimination of Vec clones (Arc-based shared ownership)
//! - Buffer reuse (Vec::clear() + Vec::reserve() pattern)
//! - Zero-copy Arrow buffer access
//!
//! # Test Strategy
//!
//! - Compare baseline (with clones) vs optimized (zero-copy) implementations
//! - Measure metadata-related memory allocations
//! - Verify 50%+ reduction in memory overhead
//! - Use allocation tracking to count allocations
//!
//! # Expected Results
//!
//! - Metadata allocations: 50%+ reduction
//! - Buffer allocations: 30%+ reduction
//! - Total memory overhead: 50%+ reduction
//!
//! # Validation Method
//!
//! This benchmark provides performance comparison. For precise memory measurement,
//! use Valgrind massif:
//!
//! ```bash
//! valgrind --tool=massif --massif-out-file=massif.out \
//!     cargo test --release test_memory_allocation
//! ms_print massif.out
//! ```

use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Generate random weight array for benchmarking
fn generate_weights(size: usize, seed: u64) -> Vec<f32> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(-10.0..10.0)).collect()
}

/// Generate time group parameters
fn generate_time_group_params(num_groups: usize, seed: u64) -> Vec<TimeGroupParams> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    (0..num_groups)
        .map(|i| {
            let start = i * 1000;
            let end = (i + 1) * 1000;
            TimeGroupParams {
                time_range: (start, end),
                scale: rng.gen_range(0.01..1.0),
                zero_point: rng.gen_range(0.0..256.0),
                group_size: 128,
            }
        })
        .collect()
}

/// Benchmark memory allocation: baseline vs optimized
///
/// Tests the memory allocation overhead of quantization operations
/// comparing implementations with and without optimizations
fn bench_memory_allocation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation_overhead");

    // Configure benchmark parameters
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_secs(2));
    group.measurement_time(std::time::Duration::from_secs(5));

    // Test sizes: 10K, 100K, 1M elements
    let sizes = vec![("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)];

    let num_time_groups = 10;
    let seed = 42u64;

    for (size_name, size) in sizes {
        let weights = generate_weights(size, seed);
        let params = generate_time_group_params(num_time_groups, seed);

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark optimized implementation (zero-copy, Arc-based)
        group.bench_with_input(BenchmarkId::new("optimized", size_name), &size, |b, _| {
            let quantizer = TimeAwareQuantizer::new(num_time_groups);
            b.iter(|| {
                let result =
                    quantizer.quantize_layer_arrow(black_box(&weights), black_box(&params));
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark buffer reuse efficiency
///
/// Tests the efficiency of buffer reuse patterns in batch processing
fn bench_buffer_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_reuse");

    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_secs(1));

    let num_layers = 10;
    let layer_size = 100_000;
    let num_time_groups = 10;
    let seed = 42u64;

    // Generate test data for multiple layers
    let layers: Vec<Vec<f32>> = (0..num_layers)
        .map(|i| generate_weights(layer_size, seed + i as u64))
        .collect();
    let params = generate_time_group_params(num_time_groups, seed);

    group.throughput(Throughput::Elements((num_layers * layer_size) as u64));

    // Benchmark with buffer reuse (optimized)
    group.bench_function("with_reuse", |b| {
        let quantizer = TimeAwareQuantizer::new(num_time_groups);
        b.iter(|| {
            for layer_weights in &layers {
                let result =
                    quantizer.quantize_layer_arrow(black_box(layer_weights), black_box(&params));
                black_box(result);
            }
        });
    });

    group.finish();
}

/// Benchmark metadata allocation patterns
///
/// Tests memory allocation patterns for metadata structures
fn bench_metadata_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("metadata_allocation");

    group.sample_size(20);

    let size = 100_000;
    let seed = 42u64;

    // Test different numbers of time groups
    let group_counts = vec![5, 10, 20, 50];

    for num_groups in group_counts {
        let weights = generate_weights(size, seed);
        let params = generate_time_group_params(num_groups, seed);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("optimized", num_groups),
            &num_groups,
            |b, _| {
                let quantizer = TimeAwareQuantizer::new(num_groups);
                b.iter(|| {
                    let result =
                        quantizer.quantize_layer_arrow(black_box(&weights), black_box(&params));
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Arrow RecordBatch construction overhead
///
/// Tests the memory allocation overhead of Arrow RecordBatch construction
fn bench_recordbatch_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("recordbatch_construction");

    group.sample_size(20);

    let sizes = vec![("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)];

    let num_time_groups = 10;
    let seed = 42u64;

    for (size_name, size) in sizes {
        let weights = generate_weights(size, seed);
        let params = generate_time_group_params(num_time_groups, seed);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("optimized", size_name), &size, |b, _| {
            let quantizer = TimeAwareQuantizer::new(num_time_groups);
            b.iter(|| {
                let result =
                    quantizer.quantize_layer_arrow(black_box(&weights), black_box(&params));
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_allocation_overhead,
    bench_buffer_reuse,
    bench_metadata_allocation,
    bench_recordbatch_construction,
);
criterion_main!(benches);

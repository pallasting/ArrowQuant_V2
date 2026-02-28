# PyO3 Zero-Copy Optimization - Benchmarks

This directory contains performance benchmarks for the PyO3 zero-copy optimization feature.

## Benchmark Suites

### 1. Numpy vs Legacy API Benchmark
**File**: `bench_numpy_vs_legacy.rs`
- Compares data transfer time between legacy and numpy APIs
- Tests with 4MB, 40MB, 400MB tensors
- Target: 15x speedup (150ms → 10ms for 4MB)
- Validates: Requirements 1.3, 4.1

### 2. Batch API Overhead Benchmark
**File**: `bench_batch_overhead.rs`
- Measures call overhead for batch processing
- Tests with 10, 100, 1000 layer batches
- Target: 100x reduction (200ms → 2ms for 100 layers)
- Validates: Requirements 2.2, 4.2

### 3. Arrow IPC Performance Benchmark
**File**: `bench_arrow_ipc.rs`
- Measures Arrow IPC data transfer performance
- Tests with 4MB, 40MB, 400MB tensors
- Target: 30x speedup (150ms → 5ms for 4MB)
- Validates: Requirements 3.3, 4.3

### 4. End-to-End Performance Benchmark
**File**: `bench_end_to_end.rs`
- Measures complete quantization workflow performance
- Tests online quantization scenarios (LoRA, ControlNet)
- Target: 2.3-2.4x overall speedup
- Target: PyO3 overhead reduction from 68% to <10%
- Validates: Requirements 4.3, 4.4

## Running Benchmarks

Run all PyO3 optimization benchmarks:
```bash
cargo bench --bench pyo3_numpy_benchmark
cargo bench --bench pyo3_batch_benchmark
cargo bench --bench pyo3_arrow_ipc_benchmark
```

Run specific benchmark:
```bash
cargo bench --bench pyo3_numpy_benchmark -- numpy_4mb
```

Generate HTML report:
```bash
cargo bench --bench pyo3_numpy_benchmark
# Report available at: target/criterion/report/index.html
```

## Performance Targets

| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| Data transfer (4MB) | 150ms | 5-10ms | 15-30x |
| Batch call overhead (100 layers) | 200ms | 2ms | 100x |
| PyO3 overhead | 68% | <10% | 6.8x reduction |
| Overall speedup | 1x | 2.3-2.4x | 2.3-2.4x |

## Benchmark Configuration

All benchmarks use:
- Sample size: 100 iterations (configurable)
- Warm-up time: 3 seconds
- Measurement time: 5 seconds
- Confidence level: 95%

## Interpreting Results

Criterion outputs:
- **time**: Mean execution time with confidence interval
- **thrpt**: Throughput (elements/second)
- **change**: Performance change vs previous run

Look for:
- ✅ Green: Performance improved
- ⚠️ Yellow: No significant change
- ❌ Red: Performance regressed

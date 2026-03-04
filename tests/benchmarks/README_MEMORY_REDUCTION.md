# Memory Allocation Reduction Benchmarks

**Validates: Requirements 8.2, 1.4**  
**Property 6: Memory Allocation Reduction**

## Overview

This benchmark suite measures memory allocation reduction from the Arrow performance optimizations:

1. **Elimination of Vec clones**: Using Arc-based shared ownership for metadata
2. **Buffer reuse**: Vec::clear() + Vec::reserve() pattern for repeated allocations
3. **Zero-copy Arrow buffer access**: Direct buffer access without copying

## Target

According to Requirements 8.2 and 1.4:
- **50%+ reduction** in metadata-related memory allocations
- Efficient buffer reuse in batch processing scenarios
- Zero-copy data access patterns

## Running the Benchmarks

### Option 1: Criterion Benchmarks (Performance Comparison)

```bash
# Run all memory reduction benchmarks
cargo bench --bench bench_memory_reduction

# Run specific benchmark group
cargo bench --bench bench_memory_reduction -- memory_allocation_overhead
cargo bench --bench bench_memory_reduction -- buffer_reuse
cargo bench --bench bench_memory_reduction -- metadata_allocation
cargo bench --bench bench_memory_reduction -- recordbatch_construction
```

### Option 2: Valgrind Massif (Precise Memory Measurement)

**Note**: Valgrind is required for precise memory allocation measurement.

```bash
# Install Valgrind (if not already installed)
# Ubuntu/Debian: sudo apt-get install valgrind
# macOS: brew install valgrind
# Fedora: sudo dnf install valgrind

# Run the memory analysis script
./tests/benchmarks/run_memory_reduction_analysis.sh
```

This will:
1. Build the memory allocation tests in release mode
2. Run each test with Valgrind massif
3. Generate memory profiles and summaries
4. Create a comparison report

Results will be saved in `target/memory_analysis/`:
- `massif_*.out`: Raw Valgrind output
- `massif_*_summary.txt`: Human-readable summaries
- `MEMORY_ANALYSIS_REPORT.md`: Comparison report

### Option 3: Simple Memory Tests

```bash
# Run memory allocation tests (without Valgrind)
cargo test --release test_memory_allocation

# These tests verify correctness but don't measure memory precisely
```

## Benchmark Groups

### 1. Memory Allocation Overhead

Compares the optimized implementation's memory allocation patterns across different array sizes:
- 10K elements
- 100K elements
- 1M elements

**Expected**: Consistent low overhead regardless of array size due to zero-copy patterns.

### 2. Buffer Reuse

Tests buffer reuse efficiency when processing multiple layers in batch:
- 10 layers of 100K elements each
- Measures allocation patterns across iterations

**Expected**: Minimal allocations after first iteration due to buffer reuse.

### 3. Metadata Allocation

Tests memory allocation for metadata structures with varying numbers of time groups:
- 5, 10, 20, 50 time groups
- 100K element arrays

**Expected**: Linear scaling with number of groups, but with Arc-based sharing reducing per-operation overhead.

### 4. RecordBatch Construction

Tests Arrow RecordBatch construction overhead:
- 10K, 100K, 1M element arrays
- Measures Arrow buffer allocation patterns

**Expected**: Efficient buffer pool usage, minimal redundant allocations.

## Interpreting Results

### Criterion Output

Criterion benchmarks show:
- **Time**: Execution time (lower is better)
- **Throughput**: Elements processed per second (higher is better)
- **Change**: Comparison with previous runs

Look for:
- Consistent performance across array sizes (good scaling)
- High throughput values
- Stable measurements (low variance)

### Valgrind Massif Output

Massif profiles show:
- **Peak heap memory**: Maximum memory used
- **Allocation patterns**: When and where memory is allocated
- **Allocation counts**: Number of allocation calls

To verify 50%+ reduction:
1. Compare peak memory with baseline (if available)
2. Check allocation counts (fewer is better)
3. Look for flat memory usage in batch scenarios (indicates reuse)

Key metrics:
```
mem_heap_B=<bytes>     # Heap memory at this snapshot
mem_heap_extra_B=<bytes>  # Extra heap memory (overhead)
mem_stacks_B=<bytes>   # Stack memory
```

## Validation Checklist

- [ ] Criterion benchmarks run successfully
- [ ] Performance is consistent across array sizes
- [ ] Buffer reuse shows minimal allocations after warmup
- [ ] Metadata allocation scales linearly with groups
- [ ] RecordBatch construction is efficient
- [ ] Valgrind massif analysis completed (if available)
- [ ] Memory reduction target (50%+) verified

## Troubleshooting

### Valgrind Not Available

If Valgrind is not available on your system:
1. Use Criterion benchmarks for performance comparison
2. Use system monitoring tools (htop, Activity Monitor) during test runs
3. Compare execution time as a proxy for memory efficiency

### Compilation Issues

If benchmarks fail to compile:
```bash
# Clean and rebuild
cargo clean
cargo build --release --bench bench_memory_reduction
```

### Permission Errors

If you encounter permission errors:
```bash
# Clean the target directory
cargo clean

# Or remove the lock file
rm -f target/release/.cargo-lock
```

## Expected Results Summary

Based on the optimization work in Tasks 1.1-1.3:

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Metadata allocations | High (Vec clones) | Low (Arc sharing) | 50%+ reduction |
| Buffer allocations | Per-operation | Reused | 30%+ reduction |
| Total overhead | 100% | <50% | 50%+ reduction |
| Batch processing | Linear growth | Flat after warmup | Significant |

## References

- **Requirements**: 8.2, 1.4
- **Design**: Memory Optimization Layer (Component 1)
- **Tasks**: 1.1, 1.2, 1.3 (Memory optimization tasks)
- **Property**: Property 6 (Memory Allocation Reduction)

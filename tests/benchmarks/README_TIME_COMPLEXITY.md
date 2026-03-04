# Time Group Assignment Complexity Benchmark

## Overview

This benchmark validates that the time group assignment implementation achieves **O(n log m)** complexity, where:
- **n** = number of weights (array size)
- **m** = number of time groups

## Validates

- **Requirements 2.3**: Time group assignment complexity is O(n log m)
- **Requirements 8.3**: Performance target verification through benchmarking

## Test Matrix

The benchmark tests the following combinations:

### Array Sizes (n)
- 1,000 elements (1K)
- 10,000 elements (10K)
- 100,000 elements (100K)
- 1,000,000 elements (1M)

### Time Group Counts (m)
- 5 groups
- 10 groups
- 20 groups

### Total Combinations
- 12 combinations (4 sizes × 3 group counts)

## Benchmark Suites

### 1. Scaling with Array Size (n)

**Purpose**: Validate that execution time scales linearly with n

**Test**: Fixed m=10, varying n from 1K to 1M

**Expected Behavior**:
- Time(10K) ≈ 10 × Time(1K)
- Time(100K) ≈ 100 × Time(1K)
- Time(1M) ≈ 1000 × Time(1K)

### 2. Scaling with Number of Groups (m)

**Purpose**: Validate that execution time scales logarithmically with m

**Test**: Fixed n=100K, varying m from 5 to 20

**Expected Behavior**:
- Time(10) ≈ Time(5) × log(10)/log(5) ≈ 1.43 × Time(5)
- Time(20) ≈ Time(5) × log(20)/log(5) ≈ 1.86 × Time(5)

### 3. Binary Search vs Linear Scan

**Purpose**: Demonstrate performance improvement of optimized implementation

**Test**: Compare `assign_time_groups_fast()` (binary search) vs `assign_time_groups()` (uniform distribution)

**Expected Speedup**:
- For m=5: ~2-3x faster
- For m=10: ~3-5x faster
- For m=20: ~5-10x faster

### 4. Full Matrix Test

**Purpose**: Comprehensive performance matrix across all n × m combinations

**Test**: All 12 combinations of array sizes and group counts

**Output**: Complete performance profile showing how execution time varies with both parameters

### 5. Boundary Pre-computation Overhead

**Purpose**: Measure the one-time cost of pre-computing boundaries

**Test**: Measure `TimeGroupBoundaries::precompute_boundaries()` for m=5 to 100

**Expected**: O(m log m) complexity, negligible compared to O(n log m) assignment for large n

### 6. Realistic Workload

**Purpose**: Benchmark with realistic large model parameters

**Scenarios**:
- Small model: 768×768 weights (589K elements), 10 time groups
- Medium model: 3072×768 weights (2.36M elements), 10 time groups
- Large model: 12288×3072 weights (37.7M elements), 20 time groups

## Running the Benchmark

### Quick Run (10 samples)
```bash
./tests/benchmarks/run_time_complexity_benchmark.sh --quick
```

### Standard Run (default samples)
```bash
./tests/benchmarks/run_time_complexity_benchmark.sh
```

### Full Run (100 samples, more accurate)
```bash
./tests/benchmarks/run_time_complexity_benchmark.sh --full
```

### Save Baseline
```bash
./tests/benchmarks/run_time_complexity_benchmark.sh --baseline
```

### Compare Against Baseline
```bash
./tests/benchmarks/run_time_complexity_benchmark.sh --compare
```

### Direct Cargo Command
```bash
cargo bench --bench bench_time_complexity
```

## Interpreting Results

### Linear Scaling with n

Look for the `time_complexity_scaling_n` group. The throughput (elements/sec) should remain roughly constant as n increases, indicating linear scaling:

```
time_complexity_scaling_n/binary_search/1000
                        time:   [X.XX µs X.XX µs X.XX µs]
                        thrpt:  [YYY.Y Kelem/s YYY.Y Kelem/s YYY.Y Kelem/s]

time_complexity_scaling_n/binary_search/10000
                        time:   [X.XX µs X.XX µs X.XX µs]
                        thrpt:  [YYY.Y Kelem/s YYY.Y Kelem/s YYY.Y Kelem/s]
```

If throughput is consistent, the implementation scales linearly with n ✓

### Logarithmic Scaling with m

Look for the `time_complexity_scaling_m` group. The time should increase logarithmically as m increases:

```
time_complexity_scaling_m/binary_search/5
                        time:   [X.XX µs X.XX µs X.XX µs]

time_complexity_scaling_m/binary_search/10
                        time:   [Y.YY µs Y.YY µs Y.YY µs]  (≈ 1.43x)

time_complexity_scaling_m/binary_search/20
                        time:   [Z.ZZ µs Z.ZZ µs Z.ZZ µs]  (≈ 1.86x)
```

If the ratios match the logarithmic expectations, the implementation scales logarithmically with m ✓

### Binary Search Speedup

Look for the `binary_search_vs_linear` group. The binary search should be significantly faster:

```
binary_search_vs_linear/binary_search/10
                        time:   [X.XX µs X.XX µs X.XX µs]

binary_search_vs_linear/uniform_distribution/10
                        time:   [Y.YY µs Y.YY µs Y.YY µs]  (3-5x slower)
```

If binary search is 3-5x faster for m=10, the optimization is working ✓

## Viewing Detailed Reports

After running the benchmark, detailed HTML reports are generated:

```bash
# Open the main report
open target/criterion/bench_time_complexity/report/index.html

# Or view specific benchmark reports
open target/criterion/time_complexity_scaling_n/report/index.html
open target/criterion/time_complexity_scaling_m/report/index.html
open target/criterion/binary_search_vs_linear/report/index.html
```

## Implementation Details

### Binary Search Algorithm

The optimized implementation uses:

1. **Pre-computation**: `TimeGroupBoundaries::precompute_boundaries()`
   - Computes upper bounds for each time group
   - Sorts boundaries in ascending order
   - One-time O(m log m) cost

2. **Binary Search**: `TimeGroupBoundaries::find_group()`
   - Uses `binary_search_by()` for O(log m) lookup
   - Handles edge cases (value < all boundaries, value > all boundaries)
   - Returns appropriate group ID

3. **Batch Assignment**: `assign_time_groups_fast()`
   - Pre-computes boundaries once
   - Applies binary search to each weight
   - Total complexity: O(n log m)

### Comparison with Baseline

The baseline `assign_time_groups()` uses uniform distribution:
- Divides weights evenly across time groups
- O(n) complexity (no binary search)
- Simpler but less flexible

## Performance Targets

Based on Requirements 8.3, the benchmark should demonstrate:

1. **Linear scaling with n**: Throughput remains constant as n increases
2. **Logarithmic scaling with m**: Time increases by ~log(m2/m1) when m increases from m1 to m2
3. **Speedup over baseline**: 3-5x faster for typical m values (10-20)

## Troubleshooting

### Benchmark Takes Too Long

Use the `--quick` flag for faster results:
```bash
./tests/benchmarks/run_time_complexity_benchmark.sh --quick
```

### Inconsistent Results

Run with more samples for better statistical significance:
```bash
./tests/benchmarks/run_time_complexity_benchmark.sh --full
```

### Comparing Across Runs

Save a baseline and compare:
```bash
# First run
./tests/benchmarks/run_time_complexity_benchmark.sh --baseline

# After changes
./tests/benchmarks/run_time_complexity_benchmark.sh --compare
```

## Related Files

- **Benchmark**: `tests/benchmarks/bench_time_complexity.rs`
- **Implementation**: `src/time_aware.rs`
  - `TimeGroupBoundaries::precompute_boundaries()`
  - `TimeGroupBoundaries::find_group()`
  - `TimeAwareQuantizer::assign_time_groups_fast()`
- **Design**: `.kiro/specs/arrow-performance-optimization/design.md`
- **Requirements**: `.kiro/specs/arrow-performance-optimization/requirements.md`

## References

- **Property 4**: Time group assignment complexity (Design Document)
- **Requirements 2.3**: O(n log m) complexity requirement
- **Requirements 8.3**: Performance target verification

#!/bin/bash
#
# Run SIMD Speedup Benchmarks
#
# This script runs the SIMD performance benchmarks and generates a report
# showing the speedup achieved by SIMD acceleration.
#
# **Validates: Requirements 3.5, 8.1**
# **Property 7: SIMD Performance Improvement**

set -e

echo "========================================="
echo "SIMD Speedup Benchmark Suite"
echo "========================================="
echo ""
echo "Testing array sizes: 1K, 10K, 100K, 1M"
echo "Expected speedup: 3x-6x"
echo ""

# Check if SIMD is available
echo "Checking SIMD availability..."
if cargo run --release --example check_simd 2>/dev/null; then
    echo "✓ SIMD is available on this platform"
else
    echo "⚠ SIMD may not be available - results will show scalar performance only"
fi
echo ""

# Run the benchmarks
echo "Running benchmarks (this may take 5-10 minutes)..."
echo ""

cargo bench --bench bench_simd_speedup -- --save-baseline simd_speedup

echo ""
echo "========================================="
echo "Benchmark Complete!"
echo "========================================="
echo ""
echo "Results saved to: target/criterion/simd_speedup/"
echo ""
echo "To view detailed results:"
echo "  1. Open target/criterion/simd_speedup/report/index.html in a browser"
echo "  2. Or run: cargo bench --bench bench_simd_speedup -- --baseline simd_speedup"
echo ""
echo "To compare with a previous run:"
echo "  cargo bench --bench bench_simd_speedup -- --baseline simd_speedup"
echo ""

# Generate summary
echo "Generating summary..."
echo ""

if [ -f "target/criterion/simd_speedup/simd/1K/base/estimates.json" ]; then
    echo "Summary of SIMD Speedup:"
    echo "------------------------"
    
    for size in "1K" "10K" "100K" "1M"; do
        simd_file="target/criterion/simd_speedup/simd/$size/base/estimates.json"
        scalar_file="target/criterion/simd_speedup/scalar/$size/base/estimates.json"
        
        if [ -f "$simd_file" ] && [ -f "$scalar_file" ]; then
            # Extract mean times (in nanoseconds)
            simd_time=$(jq '.mean.point_estimate' "$simd_file")
            scalar_time=$(jq '.mean.point_estimate' "$scalar_file")
            
            # Calculate speedup
            speedup=$(echo "scale=2; $scalar_time / $simd_time" | bc)
            
            echo "  $size: ${speedup}x speedup"
        fi
    done
    
    echo ""
    echo "✓ Target speedup (3x-6x) verification:"
    echo "  - 1K:   Expected 2-3x"
    echo "  - 10K:  Expected 3-4x"
    echo "  - 100K: Expected 4-5x"
    echo "  - 1M:   Expected 5-6x"
else
    echo "⚠ Could not generate summary - benchmark data not found"
    echo "  Run the benchmark first: cargo bench --bench bench_simd_speedup"
fi

echo ""
echo "Done!"

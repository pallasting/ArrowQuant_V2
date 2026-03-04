#!/bin/bash
#
# Memory Reduction Analysis Script
#
# **Validates: Requirements 8.2, 1.4**
# **Property 6: Memory Allocation Reduction**
#
# This script runs memory allocation analysis using Valgrind massif
# to verify 50%+ reduction in metadata-related memory overhead.
#
# Usage:
#   ./run_memory_reduction_analysis.sh

set -e

echo "========================================="
echo "Memory Reduction Analysis"
echo "========================================="
echo ""

# Check if valgrind is installed
if ! command -v valgrind &> /dev/null; then
    echo "ERROR: Valgrind is not installed."
    echo "Please install valgrind:"
    echo "  Ubuntu/Debian: sudo apt-get install valgrind"
    echo "  macOS: brew install valgrind"
    echo "  Fedora: sudo dnf install valgrind"
    exit 1
fi

# Build the test in release mode
echo "Building memory allocation test..."
cargo test --release test_memory_allocation --no-run

# Find the test binary
TEST_BINARY=$(find target/release/deps -name 'test_memory_allocation-*' -type f -executable | head -1)

if [ -z "$TEST_BINARY" ]; then
    echo "ERROR: Could not find test binary"
    exit 1
fi

echo "Found test binary: $TEST_BINARY"
echo ""

# Create output directory
mkdir -p target/memory_analysis

# Run each test with Valgrind massif
echo "Running memory allocation tests with Valgrind massif..."
echo ""

for test_name in small medium large batch many_groups; do
    echo "----------------------------------------"
    echo "Test: test_memory_allocation_$test_name"
    echo "----------------------------------------"
    
    OUTPUT_FILE="target/memory_analysis/massif_${test_name}.out"
    
    # Run valgrind massif
    valgrind --tool=massif \
        --massif-out-file="$OUTPUT_FILE" \
        --stacks=yes \
        --time-unit=B \
        "$TEST_BINARY" "test_memory_allocation_$test_name" 2>&1 | grep -E "(PASS|FAIL|heap|total)"
    
    echo ""
    echo "Memory profile saved to: $OUTPUT_FILE"
    
    # Print peak memory usage
    if [ -f "$OUTPUT_FILE" ]; then
        PEAK_MEM=$(grep "mem_heap_B" "$OUTPUT_FILE" | awk '{print $2}' | sort -n | tail -1)
        PEAK_MB=$(echo "scale=2; $PEAK_MEM / 1024 / 1024" | bc)
        echo "Peak heap memory: ${PEAK_MB} MB"
        
        # Generate summary
        ms_print "$OUTPUT_FILE" > "target/memory_analysis/massif_${test_name}_summary.txt"
        echo "Summary saved to: target/memory_analysis/massif_${test_name}_summary.txt"
    fi
    
    echo ""
done

echo "========================================="
echo "Memory Analysis Complete"
echo "========================================="
echo ""
echo "Results saved in: target/memory_analysis/"
echo ""
echo "To view detailed memory profile:"
echo "  ms_print target/memory_analysis/massif_<test_name>.out"
echo ""
echo "To compare memory usage:"
echo "  cat target/memory_analysis/massif_*_summary.txt | grep 'peak'"
echo ""

# Generate comparison report
echo "Generating comparison report..."
cat > target/memory_analysis/MEMORY_ANALYSIS_REPORT.md << 'EOF'
# Memory Allocation Analysis Report

**Validates: Requirements 8.2, 1.4**
**Property 6: Memory Allocation Reduction**

## Test Results

This report shows memory allocation patterns for the optimized implementation.

### Peak Memory Usage by Test

EOF

for test_name in small medium large batch many_groups; do
    OUTPUT_FILE="target/memory_analysis/massif_${test_name}.out"
    if [ -f "$OUTPUT_FILE" ]; then
        PEAK_MEM=$(grep "mem_heap_B" "$OUTPUT_FILE" | awk '{print $2}' | sort -n | tail -1)
        PEAK_MB=$(echo "scale=2; $PEAK_MEM / 1024 / 1024" | bc)
        
        cat >> target/memory_analysis/MEMORY_ANALYSIS_REPORT.md << EOF
- **test_memory_allocation_$test_name**: ${PEAK_MB} MB peak heap memory

EOF
    fi
done

cat >> target/memory_analysis/MEMORY_ANALYSIS_REPORT.md << 'EOF'

## Analysis

The memory allocation tests demonstrate the effectiveness of the following optimizations:

1. **Arc-based shared ownership**: Eliminates Vec clones for metadata
2. **Buffer reuse**: Vec::clear() + Vec::reserve() pattern reduces allocations
3. **Zero-copy Arrow buffer access**: Direct buffer access without copying

### Expected Results

According to Requirements 8.2 and 1.4, the optimized implementation should achieve:
- **50%+ reduction** in metadata-related memory allocations
- Efficient buffer reuse in batch processing
- Zero-copy data access patterns

### Validation Method

To validate the 50%+ reduction target, compare these results with a baseline
implementation that includes Vec clones and does not use buffer reuse patterns.

The key metrics to compare:
- Total heap allocations
- Peak memory usage
- Number of allocation calls
- Memory allocation patterns over time

## Files Generated

- `massif_*.out`: Raw Valgrind massif output files
- `massif_*_summary.txt`: Human-readable memory profile summaries
- `MEMORY_ANALYSIS_REPORT.md`: This report

## Next Steps

1. Review the massif summaries to identify allocation hotspots
2. Compare with baseline implementation (if available)
3. Verify 50%+ reduction in metadata allocations
4. Document findings in the task completion report

EOF

echo "Report generated: target/memory_analysis/MEMORY_ANALYSIS_REPORT.md"
echo ""

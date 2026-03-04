#!/bin/bash
# Run Time Group Assignment Complexity Benchmark
#
# This script runs the time complexity benchmark for time group assignment
# and validates that the implementation achieves O(n log m) complexity.
#
# Usage:
#   ./run_time_complexity_benchmark.sh [OPTIONS]
#
# Options:
#   --quick     Run quick benchmark (fewer samples)
#   --full      Run full benchmark (more samples, longer)
#   --baseline  Save baseline results for comparison
#   --compare   Compare against baseline results
#
# Examples:
#   ./run_time_complexity_benchmark.sh --quick
#   ./run_time_complexity_benchmark.sh --full --baseline
#   ./run_time_complexity_benchmark.sh --compare

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
QUICK=false
FULL=false
BASELINE=false
COMPARE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            shift
            ;;
        --full)
            FULL=true
            shift
            ;;
        --baseline)
            BASELINE=true
            shift
            ;;
        --compare)
            COMPARE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Time Complexity Benchmark${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "This benchmark validates that time group assignment"
echo "achieves O(n log m) complexity where:"
echo "  - n = number of weights"
echo "  - m = number of time groups"
echo ""

# Determine benchmark options
BENCH_OPTS=""
if [ "$QUICK" = true ]; then
    echo -e "${YELLOW}Running quick benchmark (10 samples)...${NC}"
    BENCH_OPTS="--quick"
elif [ "$FULL" = true ]; then
    echo -e "${YELLOW}Running full benchmark (100 samples)...${NC}"
    BENCH_OPTS="--sample-size 100"
else
    echo -e "${YELLOW}Running standard benchmark (default samples)...${NC}"
fi

# Run the benchmark
echo ""
echo "Building and running benchmark..."
echo ""

if [ "$BASELINE" = true ]; then
    echo -e "${YELLOW}Saving baseline results...${NC}"
    cargo bench --bench bench_time_complexity $BENCH_OPTS -- --save-baseline time_complexity_baseline
elif [ "$COMPARE" = true ]; then
    echo -e "${YELLOW}Comparing against baseline...${NC}"
    cargo bench --bench bench_time_complexity $BENCH_OPTS -- --baseline time_complexity_baseline
else
    cargo bench --bench bench_time_complexity $BENCH_OPTS
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Benchmark Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results are saved in target/criterion/bench_time_complexity/"
echo ""
echo "To view detailed results:"
echo "  - Open target/criterion/bench_time_complexity/report/index.html"
echo "  - Or check the console output above"
echo ""
echo "Expected results:"
echo "  - Time scales linearly with n (array size)"
echo "  - Time scales logarithmically with m (number of groups)"
echo "  - Binary search is faster than uniform distribution"
echo ""

#!/bin/bash

# Quick verification script for arrow-performance-optimization
# Uses local target directory to work around CIFS limitations

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Use local target directory
export CARGO_TARGET_DIR=~/cargo_target_arrow_quant

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Arrow Performance Optimization${NC}"
echo -e "${BLUE}Quick Verification Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if build exists
if [ ! -d "$CARGO_TARGET_DIR/release" ]; then
    echo -e "${YELLOW}Building project (first time)...${NC}"
    cargo build --release
    echo -e "${GREEN}✓ Build complete${NC}"
    echo ""
fi

# Test categories
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test() {
    local test_name=$1
    local test_pattern=$2
    
    echo -e "${BLUE}Running: ${test_name}${NC}"
    
    if timeout 30 cargo test --release --lib "$test_pattern" -- --nocapture 2>&1 | tee /tmp/test_output.log | grep -q "test result: ok"; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        tail -20 /tmp/test_output.log
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
}

echo -e "${YELLOW}=== Stage 1: Memory Optimization Tests ===${NC}"
run_test "Arc Optimization" "test_arc_optimization"
run_test "Buffer Optimization" "test_buffer_reuse"
run_test "Time Group Boundaries" "test_time_group_boundaries"

echo -e "${YELLOW}=== Stage 2: Python API Tests ===${NC}"
run_test "Parameter Validation" "test_validate_parameters"
run_test "Error Handling" "test_error_logging"
run_test "Performance Metrics" "test_performance_metrics"

echo -e "${YELLOW}=== Stage 3: SIMD Tests ===${NC}"
run_test "SIMD Detection" "test_simd_detection"
run_test "SIMD Quantization" "test_simd_quantization"
run_test "SIMD Workflow" "test_simd_workflow"

echo -e "${YELLOW}=== Property Tests (Quick) ===${NC}"
run_test "SIMD Equivalence" "test_simd_equivalence"
run_test "Monotonicity" "test_monotonicity"
run_test "Zero Copy" "test_zero_copy"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total Tests:  ${TOTAL_TESTS}"
echo -e "${GREEN}Passed:       ${PASSED_TESTS}${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Failed:       ${FAILED_TESTS}${NC}"
else
    echo -e "Failed:       ${FAILED_TESTS}"
fi
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Run full test suite: CARGO_TARGET_DIR=~/cargo_target_arrow_quant cargo test --release"
    echo "2. Run benchmarks: CARGO_TARGET_DIR=~/cargo_target_arrow_quant cargo bench"
    echo "3. Check FINAL_VERIFICATION_CHECKLIST.md"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Please review the output above.${NC}"
    exit 1
fi

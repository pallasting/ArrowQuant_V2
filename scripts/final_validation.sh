#!/bin/bash
# Final Validation Script for Arrow Zero-Copy Implementation
# This script runs comprehensive validation checks

set -e  # Exit on error

echo "=========================================="
echo "Arrow Zero-Copy Final Validation"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Function to run a check
run_check() {
    local name="$1"
    local command="$2"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    echo -n "[$TOTAL_CHECKS] $name... "
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

echo "Phase 1: Code Quality Checks"
echo "----------------------------------------"

run_check "Code formatting" "cargo fmt -- --check"
run_check "Clippy lints" "cargo clippy -- -D warnings"
run_check "Documentation build" "cargo doc --no-deps"

echo ""
echo "Phase 2: Core Functionality Tests"
echo "----------------------------------------"

run_check "All Rust tests" "cargo test --lib --release"
run_check "Arrow schema tests" "cargo test --lib --release test_create_time_aware_schema"
run_check "Arrow quantization tests" "cargo test --lib --release test_quantize_layer_arrow"
run_check "Arrow dequantization tests" "cargo test --lib --release test_dequantize_group"
run_check "Parallel dequantization" "cargo test --lib --release test_parallel_dequantization"
run_check "Integration tests" "cargo test --lib --release test_apply_time_aware_quantization"

echo ""
echo "Phase 3: Performance Validation"
echo "----------------------------------------"

run_check "Performance benchmarks" "cargo bench --bench performance_validation --no-run"
run_check "Memory efficiency tests" "cargo test --lib --release test_memory_usage_comparison"
run_check "Zero-copy validation" "cargo test --lib --release test_zero_copy"

echo ""
echo "Phase 4: Python Integration"
echo "----------------------------------------"

# Check if Python environment is available
if command -v python3 &> /dev/null; then
    run_check "Build Python extension" "maturin develop --release"
    run_check "Python tests" "pytest tests/ -v"
    run_check "Arrow Python tests" "pytest tests/test_py_arrow_quantized_layer.py -v"
else
    echo -e "${YELLOW}⚠ Python not available, skipping Python tests${NC}"
fi

echo ""
echo "Phase 5: Documentation Validation"
echo "----------------------------------------"

run_check "README exists" "test -f README.md"
run_check "API docs exist" "test -f docs/api_documentation.md"
run_check "Usage guide exists" "test -f docs/arrow_zero_copy_guide.md"
run_check "Migration guide exists" "test -f docs/migration_guide.md"

echo ""
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo ""
echo "Total checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
echo ""

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}✓ All validation checks passed!${NC}"
    echo ""
    echo "The Arrow zero-copy implementation is ready for release."
    exit 0
else
    echo -e "${RED}✗ Some validation checks failed.${NC}"
    echo ""
    echo "Please review the failed checks above and fix the issues."
    exit 1
fi

#!/bin/bash
#
# 运行完整测试套件
#
# 此脚本运行所有测试以验证 Arrow 性能优化项目的正确性
# **验证需求**: 7.1, 7.3, 11.6
#
# 使用方法:
#   ./run_all_tests.sh [--quick]
#
# 选项:
#   --quick    快速模式，跳过长时间运行的测试

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
QUICK_MODE=false
if [ "$1" == "--quick" ]; then
    QUICK_MODE=true
fi

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Arrow 性能优化 - 完整测试套件${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 检查环境
echo -e "${YELLOW}检查环境...${NC}"
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}✗ cargo 未找到，请安装 Rust${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Rust 环境正常${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}⚠ python3 未找到，将跳过 Python 测试${NC}"
    SKIP_PYTHON=true
else
    echo -e "${GREEN}✓ Python 环境正常${NC}"
    SKIP_PYTHON=false
fi

echo ""

# 统计变量
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 函数：运行测试并记录结果
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "${BLUE}运行: $test_name${NC}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if eval "$test_command" > /tmp/test_output_$$.log 2>&1; then
        echo -e "${GREEN}✓ $test_name 通过${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ $test_name 失败${NC}"
        echo -e "${YELLOW}查看详细输出: /tmp/test_output_$$.log${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# ============================================================================
# 任务 14.1: 运行所有 374+ 现有测试用例
# ============================================================================

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}任务 14.1: 运行所有现有测试用例${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

echo -e "${YELLOW}编译项目...${NC}"
if cargo build --release 2>&1 | tee /tmp/build_output_$$.log; then
    echo -e "${GREEN}✓ 编译成功${NC}"
else
    echo -e "${RED}✗ 编译失败${NC}"
    echo -e "${YELLOW}查看详细输出: /tmp/build_output_$$.log${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}运行所有单元测试和集成测试...${NC}"
run_test "所有 Rust 测试" "cargo test --release --lib"
echo ""

# ============================================================================
# 任务 14.2: 运行所有属性测试
# ============================================================================

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}任务 14.2: 运行所有属性测试${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

if [ "$QUICK_MODE" = true ]; then
    echo -e "${YELLOW}快速模式：跳过属性测试${NC}"
else
    echo -e "${YELLOW}运行属性测试（每个测试 20 个用例）...${NC}"
    
    # SIMD 等价性测试
    run_test "SIMD 等价性属性测试" "cargo test --release test_simd_equivalence"
    
    # 单调性测试
    run_test "时间组分配单调性测试" "cargo test --release test_monotonicity"
    
    # 零拷贝测试
    run_test "零拷贝属性测试" "cargo test --release test_zero_copy"
    
    # 量化往返测试
    run_test "量化往返属性测试" "cargo test --release test_quantization_roundtrip"
    
    # 验证属性测试
    run_test "验证属性测试" "cargo test --release test_validation_property"
    
    # 精度测试
    run_test "Arrow Kernels 精度测试" "cargo test --release test_precision"
    
    echo ""
fi

# ============================================================================
# 任务 14.3: 运行跨平台测试
# ============================================================================

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}任务 14.3: 运行跨平台测试${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

echo -e "${YELLOW}检测平台...${NC}"
PLATFORM=$(uname -s)
ARCH=$(uname -m)
echo -e "${GREEN}平台: $PLATFORM${NC}"
echo -e "${GREEN}架构: $ARCH${NC}"
echo ""

# SIMD 检测测试
run_test "SIMD 检测测试" "cargo test --release test_simd_detection"

# 平台特定测试
if [ "$ARCH" == "x86_64" ]; then
    echo -e "${YELLOW}运行 x86_64 特定测试...${NC}"
    run_test "x86_64 SIMD 测试" "cargo test --release test_x86_64"
elif [ "$ARCH" == "aarch64" ] || [ "$ARCH" == "arm64" ]; then
    echo -e "${YELLOW}运行 ARM64 特定测试...${NC}"
    run_test "ARM64 NEON 测试" "cargo test --release test_arm64"
fi

echo ""

# ============================================================================
# Python 测试
# ============================================================================

if [ "$SKIP_PYTHON" = false ]; then
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}Python 集成测试${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo ""
    
    echo -e "${YELLOW}检查 Python 模块...${NC}"
    if python3 -c "import arrow_quant_v2" 2>/dev/null; then
        echo -e "${GREEN}✓ arrow_quant_v2 模块已安装${NC}"
        
        if command -v pytest &> /dev/null; then
            echo -e "${YELLOW}运行 Python 测试...${NC}"
            run_test "Python 绑定测试" "python3 -m pytest tests/test_python_bindings.py -v"
        else
            echo -e "${YELLOW}⚠ pytest 未安装，跳过 Python 测试${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ arrow_quant_v2 模块未安装，跳过 Python 测试${NC}"
        echo -e "${YELLOW}  提示: 运行 'maturin develop --release' 安装模块${NC}"
    fi
    echo ""
fi

# ============================================================================
# 测试总结
# ============================================================================

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}测试总结${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

echo -e "总测试数: ${BLUE}$TOTAL_TESTS${NC}"
echo -e "通过: ${GREEN}$PASSED_TESTS${NC}"
echo -e "失败: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}✓ 所有测试通过！${NC}"
    echo -e "${GREEN}=========================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}✗ 有 $FAILED_TESTS 个测试失败${NC}"
    echo -e "${RED}=========================================${NC}"
    echo ""
    echo -e "${YELLOW}请检查失败的测试并修复问题${NC}"
    exit 1
fi

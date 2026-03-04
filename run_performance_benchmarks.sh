#!/bin/bash
#
# 运行性能基准测试
#
# 此脚本运行所有性能基准测试以验证优化目标
# **验证需求**: 8.1, 8.2, 8.3, 3.5
#
# 使用方法:
#   ./run_performance_benchmarks.sh [--quick]

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置
QUICK_MODE=false
if [ "$1" == "--quick" ]; then
    QUICK_MODE=true
fi

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Arrow 性能优化 - 性能基准测试${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 检查环境
echo -e "${YELLOW}检查环境...${NC}"
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}✗ cargo 未找到${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Rust 环境正常${NC}"
echo ""

# ============================================================================
# 任务 15.1: 运行量化速度基准测试
# ============================================================================

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}任务 15.1: SIMD 量化速度基准测试${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

echo -e "${YELLOW}目标: SIMD 速度提升 3x-6x${NC}"
echo -e "${YELLOW}测试规模: 1K, 10K, 100K, 1M 元素${NC}"
echo ""

if [ "$QUICK_MODE" = true ]; then
    echo -e "${YELLOW}快速模式：使用较少样本${NC}"
    export CRITERION_SAMPLE_SIZE=10
else
    export CRITERION_SAMPLE_SIZE=20
fi

echo -e "${YELLOW}运行 SIMD 加速比基准测试...${NC}"
if cargo bench --bench bench_simd_speedup 2>&1 | tee /tmp/bench_simd_$$.log; then
    echo -e "${GREEN}✓ SIMD 基准测试完成${NC}"
    echo -e "${YELLOW}结果保存在: target/criterion/simd_speedup/${NC}"
else
    echo -e "${RED}✗ SIMD 基准测试失败${NC}"
fi
echo ""

# ============================================================================
# 任务 15.2: 运行内存分配基准测试
# ============================================================================

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}任务 15.2: 内存分配基准测试${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

echo -e "${YELLOW}目标: 内存分配减少 50%+${NC}"
echo ""

if command -v valgrind &> /dev/null; then
    echo -e "${YELLOW}使用 Valgrind massif 分析内存...${NC}"
    
    # 运行内存分析
    valgrind --tool=massif \
        --massif-out-file=/tmp/massif_$$.out \
        cargo test --release test_buffer_reuse -- --nocapture 2>&1 | tee /tmp/memory_test_$$.log
    
    echo -e "${GREEN}✓ 内存分析完成${NC}"
    echo -e "${YELLOW}Massif 输出: /tmp/massif_$$.out${NC}"
    
    # 生成报告
    if command -v ms_print &> /dev/null; then
        ms_print /tmp/massif_$$.out > /tmp/massif_report_$$.txt
        echo -e "${YELLOW}Massif 报告: /tmp/massif_report_$$.txt${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Valgrind 未安装，跳过内存分析${NC}"
    echo -e "${YELLOW}  提示: 安装 valgrind 以运行内存分析${NC}"
fi
echo ""

# ============================================================================
# 任务 15.3: 运行时间复杂度基准测试
# ============================================================================

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}任务 15.3: 时间复杂度基准测试${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

echo -e "${YELLOW}目标: 时间复杂度 O(n log m)${NC}"
echo -e "${YELLOW}测试规模: n=1K-1M, m=5-20${NC}"
echo ""

echo -e "${YELLOW}运行时间复杂度基准测试...${NC}"
if [ -f "tests/benchmarks/bench_time_complexity.rs" ]; then
    if cargo bench --bench bench_time_complexity 2>&1 | tee /tmp/bench_complexity_$$.log; then
        echo -e "${GREEN}✓ 时间复杂度基准测试完成${NC}"
        echo -e "${YELLOW}结果保存在: target/criterion/time_complexity/${NC}"
    else
        echo -e "${RED}✗ 时间复杂度基准测试失败${NC}"
    fi
else
    echo -e "${YELLOW}⚠ 时间复杂度基准测试文件不存在${NC}"
fi
echo ""

# ============================================================================
# 生成性能报告
# ============================================================================

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}性能报告生成${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

REPORT_FILE="PERFORMANCE_REPORT_$(date +%Y%m%d_%H%M%S).md"

cat > "$REPORT_FILE" << 'EOF'
# Arrow 性能优化 - 性能基准测试报告

## 测试环境

EOF

echo "- **日期**: $(date)" >> "$REPORT_FILE"
echo "- **平台**: $(uname -s)" >> "$REPORT_FILE"
echo "- **架构**: $(uname -m)" >> "$REPORT_FILE"
echo "- **CPU**: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo 'Unknown')" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << 'EOF'
## 性能目标验证

### 1. SIMD 量化速度提升

**目标**: 3x-6x 加速比

| 数组大小 | SIMD 时间 | 标量时间 | 加速比 | 状态 |
|---------|----------|---------|--------|------|
| 1K      | -        | -       | -      | 待测量 |
| 10K     | -        | -       | -      | 待测量 |
| 100K    | -        | -       | -      | 待测量 |
| 1M      | -        | -       | -      | 待测量 |

**结论**: 待运行基准测试后更新

### 2. 内存分配减少

**目标**: 50%+ 减少

- **基线内存分配**: 待测量
- **优化后内存分配**: 待测量
- **减少百分比**: 待测量

**结论**: 待运行内存分析后更新

### 3. 时间复杂度优化

**目标**: O(n log m)

- **基线复杂度**: O(n × m)
- **优化后复杂度**: O(n log m)
- **理论加速比**: ~100x (m=10, n=1M)

**结论**: 算法分析已验证，待基准测试确认

## 详细结果

### SIMD 基准测试

详细结果见: `target/criterion/simd_speedup/report/index.html`

### 内存分析

详细结果见: `/tmp/massif_report_*.txt`

### 时间复杂度基准测试

详细结果见: `target/criterion/time_complexity/report/index.html`

## 结论

- [ ] SIMD 加速比达到 3x-6x
- [ ] 内存分配减少 50%+
- [ ] 时间复杂度优化到 O(n log m)

**总体状态**: 待验证

---

**生成时间**: $(date)
**生成脚本**: run_performance_benchmarks.sh
EOF

echo -e "${GREEN}✓ 性能报告已生成: $REPORT_FILE${NC}"
echo ""

# ============================================================================
# 总结
# ============================================================================

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}基准测试完成${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

echo -e "${YELLOW}查看结果:${NC}"
echo -e "  1. SIMD 基准: ${BLUE}target/criterion/simd_speedup/report/index.html${NC}"
echo -e "  2. 时间复杂度: ${BLUE}target/criterion/time_complexity/report/index.html${NC}"
echo -e "  3. 性能报告: ${BLUE}$REPORT_FILE${NC}"
echo ""

echo -e "${YELLOW}下一步:${NC}"
echo -e "  1. 在浏览器中打开 HTML 报告查看详细结果"
echo -e "  2. 更新性能报告中的实际测量值"
echo -e "  3. 验证所有性能目标是否达成"
echo ""

echo -e "${GREEN}完成！${NC}"

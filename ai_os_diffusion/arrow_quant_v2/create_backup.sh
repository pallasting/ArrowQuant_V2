#!/bin/bash
# 完整代码备份脚本
# 排除编译临时文件和缓存

set -e

echo "=========================================="
echo "ArrowQuant V2 完整代码备份"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 获取项目名称和时间戳
PROJECT_NAME="arrow_quant_v2"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="${PROJECT_NAME}_backup_${TIMESTAMP}"
BACKUP_DIR="../${BACKUP_NAME}"

echo -e "${YELLOW}备份配置:${NC}"
echo "  项目名称: $PROJECT_NAME"
echo "  备份时间: $TIMESTAMP"
echo "  备份目录: $BACKUP_DIR"
echo ""

# 创建备份目录
echo -e "${YELLOW}步骤 1: 创建备份目录${NC}"
mkdir -p "$BACKUP_DIR"
echo -e "${GREEN}✓ 备份目录已创建: $BACKUP_DIR${NC}"
echo ""

# 复制文件（排除编译产物和缓存）
echo -e "${YELLOW}步骤 2: 复制项目文件${NC}"
echo "排除以下目录和文件:"
echo "  - target/ (Rust 编译产物)"
echo "  - .venv/ (Python 虚拟环境)"
echo "  - dist/ (构建产物)"
echo "  - __pycache__/ (Python 缓存)"
echo "  - .pytest_cache/ (测试缓存)"
echo "  - .hypothesis/ (Hypothesis 缓存)"
echo "  - .benchmarks/ (基准测试缓存)"
echo "  - proptest-regressions/ (Proptest 数据)"
echo "  - *.pyc, *.pyo, *.so (编译文件)"
echo ""

rsync -av --progress \
    --exclude='target/' \
    --exclude='.venv/' \
    --exclude='dist/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.so' \
    --exclude='*.dylib' \
    --exclude='*.dll' \
    --exclude='.pytest_cache/' \
    --exclude='.hypothesis/' \
    --exclude='.benchmarks/' \
    --exclude='proptest-regressions/' \
    --exclude='.git/' \
    --exclude='*.egg-info/' \
    --exclude='build/' \
    ./ "$BACKUP_DIR/"

echo ""
echo -e "${GREEN}✓ 文件复制完成${NC}"
echo ""

# 统计备份信息
echo -e "${YELLOW}步骤 3: 统计备份信息${NC}"
TOTAL_FILES=$(find "$BACKUP_DIR" -type f | wc -l)
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "  总文件数: $TOTAL_FILES"
echo "  总大小: $TOTAL_SIZE"
echo ""

# 创建备份说明文件
echo -e "${YELLOW}步骤 4: 创建备份说明${NC}"
cat > "$BACKUP_DIR/BACKUP_INFO.txt" << EOF
ArrowQuant V2 代码备份
=====================

备份时间: $(date +"%Y-%m-%d %H:%M:%S")
备份来源: $(pwd)
Git 提交: $(git rev-parse HEAD 2>/dev/null || echo "N/A")
Git 分支: $(git branch --show-current 2>/dev/null || echo "N/A")

备份内容:
- 源代码 (src/, python/, llm_compression/)
- 测试代码 (tests/, benches/)
- 文档 (docs/, *.md)
- 配置文件 (Cargo.toml, pyproject.toml, etc.)
- 脚本 (scripts/, examples/)
- 所有临时文件和调试脚本

排除内容:
- target/ (Rust 编译产物)
- .venv/ (Python 虚拟环境)
- dist/ (构建产物)
- __pycache__/ (Python 缓存)
- .pytest_cache/ (测试缓存)
- .hypothesis/ (Hypothesis 缓存)
- .benchmarks/ (基准测试缓存)
- proptest-regressions/ (Proptest 数据)
- *.pyc, *.pyo, *.so (编译文件)

备份统计:
- 总文件数: $TOTAL_FILES
- 总大小: $TOTAL_SIZE

恢复方法:
1. 解压或复制备份目录
2. 进入项目目录
3. 重新构建: cargo build --release
4. 安装 Python 依赖: pip install -e .
5. 运行测试: cargo test && pytest

注意事项:
- 此备份包含所有源代码和临时文件
- 不包含编译产物，需要重新构建
- 不包含 .git 目录，如需 Git 历史请单独备份
EOF

echo -e "${GREEN}✓ 备份说明已创建: $BACKUP_DIR/BACKUP_INFO.txt${NC}"
echo ""

# 创建压缩包（可选）
echo -e "${YELLOW}是否创建压缩包？ (y/n)${NC}"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}步骤 5: 创建压缩包${NC}"
    ARCHIVE_NAME="${BACKUP_NAME}.tar.gz"
    tar -czf "../${ARCHIVE_NAME}" -C .. "$BACKUP_NAME"
    ARCHIVE_SIZE=$(du -sh "../${ARCHIVE_NAME}" | cut -f1)
    echo -e "${GREEN}✓ 压缩包已创建: ../${ARCHIVE_NAME} ($ARCHIVE_SIZE)${NC}"
    echo ""
    
    # 询问是否删除未压缩的备份目录
    echo -e "${YELLOW}是否删除未压缩的备份目录？ (y/n)${NC}"
    read -r response2
    if [[ "$response2" =~ ^[Yy]$ ]]; then
        rm -rf "$BACKUP_DIR"
        echo -e "${GREEN}✓ 未压缩的备份目录已删除${NC}"
        echo ""
    fi
fi

# 完成
echo "=========================================="
echo -e "${GREEN}备份完成！${NC}"
echo "=========================================="
echo ""
echo "备份位置:"
if [ -d "$BACKUP_DIR" ]; then
    echo "  目录: $BACKUP_DIR"
fi
if [ -f "../${ARCHIVE_NAME}" ]; then
    echo "  压缩包: ../${ARCHIVE_NAME}"
fi
echo ""
echo "查看备份说明:"
if [ -d "$BACKUP_DIR" ]; then
    echo "  cat $BACKUP_DIR/BACKUP_INFO.txt"
else
    echo "  tar -xzf ../${ARCHIVE_NAME} && cat ${BACKUP_NAME}/BACKUP_INFO.txt"
fi
echo ""
echo -e "${GREEN}现在可以安全地执行清理脚本了！${NC}"
echo "  ./cleanup_and_backup.sh"

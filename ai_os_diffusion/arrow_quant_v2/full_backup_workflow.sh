#!/bin/bash
# 完整备份工作流程 - 一键执行
# 包含: 本地备份 + Git 清理 + 远程推送

set -e

echo "=========================================="
echo "ArrowQuant V2 完整备份工作流程"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}此脚本将执行以下操作:${NC}"
echo "  1. 创建完整代码备份（本地）"
echo "  2. 清理临时文件到 .archive/"
echo "  3. 提交更改到 Git"
echo "  4. 推送到远程仓库"
echo "  5. 创建备份标签"
echo ""
echo -e "${YELLOW}预计时间: 15-25 分钟${NC}"
echo ""
echo -e "${RED}警告: 此操作将修改 Git 历史并推送到远程${NC}"
echo -e "${YELLOW}是否继续？ (y/n)${NC}"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo -e "${RED}操作已取消${NC}"
    exit 0
fi
echo ""

# ============================================
# 阶段 1: 完整代码备份
# ============================================
echo "=========================================="
echo -e "${BLUE}阶段 1: 完整代码备份${NC}"
echo "=========================================="
echo ""

# 检查 create_backup.sh 是否存在
if [ ! -f "create_backup.sh" ]; then
    echo -e "${RED}错误: create_backup.sh 不存在${NC}"
    exit 1
fi

# 执行备份（自动选择创建压缩包并删除未压缩目录）
echo -e "${YELLOW}开始创建本地备份...${NC}"
echo ""

# 获取项目名称和时间戳
PROJECT_NAME="arrow_quant_v2"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="${PROJECT_NAME}_backup_${TIMESTAMP}"
BACKUP_DIR="../${BACKUP_NAME}"

# 创建备份目录
mkdir -p "$BACKUP_DIR"
echo -e "${GREEN}✓ 备份目录已创建: $BACKUP_DIR${NC}"

# 复制文件
echo "正在复制文件..."
rsync -a --quiet \
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

echo -e "${GREEN}✓ 文件复制完成${NC}"

# 统计信息
TOTAL_FILES=$(find "$BACKUP_DIR" -type f | wc -l)
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "  总文件数: $TOTAL_FILES"
echo "  总大小: $TOTAL_SIZE"

# 创建备份说明
cat > "$BACKUP_DIR/BACKUP_INFO.txt" << EOF
ArrowQuant V2 代码备份
=====================

备份时间: $(date +"%Y-%m-%d %H:%M:%S")
备份来源: $(pwd)
Git 提交: $(git rev-parse HEAD 2>/dev/null || echo "N/A")
Git 分支: $(git branch --show-current 2>/dev/null || echo "N/A")

备份统计:
- 总文件数: $TOTAL_FILES
- 总大小: $TOTAL_SIZE

恢复方法:
1. 解压或复制备份目录
2. 进入项目目录
3. 重新构建: cargo build --release
4. 安装 Python 依赖: pip install -e .
5. 运行测试: cargo test && pytest
EOF

echo -e "${GREEN}✓ 备份说明已创建${NC}"

# 创建压缩包
echo "正在创建压缩包..."
ARCHIVE_NAME="${BACKUP_NAME}.tar.gz"
tar -czf "../${ARCHIVE_NAME}" -C .. "$BACKUP_NAME" 2>/dev/null
ARCHIVE_SIZE=$(du -sh "../${ARCHIVE_NAME}" | cut -f1)
echo -e "${GREEN}✓ 压缩包已创建: ../${ARCHIVE_NAME} ($ARCHIVE_SIZE)${NC}"

# 删除未压缩目录
rm -rf "$BACKUP_DIR"
echo -e "${GREEN}✓ 未压缩目录已删除（节省空间）${NC}"

echo ""
echo -e "${GREEN}阶段 1 完成！本地备份已创建${NC}"
echo "  位置: ../${ARCHIVE_NAME}"
echo "  大小: $ARCHIVE_SIZE"
echo ""

# 暂停 3 秒
sleep 3

# ============================================
# 阶段 2: 清理和 Git 备份
# ============================================
echo "=========================================="
echo -e "${BLUE}阶段 2: 清理和 Git 备份${NC}"
echo "=========================================="
echo ""

# 创建归档目录
echo -e "${YELLOW}创建归档目录...${NC}"
mkdir -p .archive/{debug-scripts,analysis-results,temp-docs}
echo -e "${GREEN}✓ 归档目录创建完成${NC}"

# 移动调试脚本
echo -e "${YELLOW}归档调试脚本...${NC}"
moved_scripts=0
for file in analyze_pymethods_detailed.py check_exports.py diagnose_pymethods.py verify_pymethods.py test_arrow_import_debug.py test_integration_simple.py test_methods.py; do
    if [ -f "$file" ]; then
        mv "$file" .archive/debug-scripts/
        ((moved_scripts++))
    fi
done
echo -e "${GREEN}✓ 已归档 $moved_scripts 个调试脚本${NC}"

# 移动分析结果
echo -e "${YELLOW}归档分析结果...${NC}"
moved_results=0
for file in *_results.json *_analysis.json; do
    if [ -f "$file" ]; then
        mv "$file" .archive/analysis-results/
        ((moved_results++))
    fi
done
echo -e "${GREEN}✓ 已归档 $moved_results 个分析结果文件${NC}"

# 移动临时文档和日志
echo -e "${YELLOW}归档临时文档和日志...${NC}"
moved_temp=0
for file in test_export_hypothesis.md *.log; do
    if [ -f "$file" ]; then
        mv "$file" .archive/temp-docs/
        ((moved_temp++))
    fi
done
echo -e "${GREEN}✓ 已归档 $moved_temp 个临时文件${NC}"

echo ""
echo -e "${YELLOW}Git 状态:${NC}"
git status --short
echo ""

# Git 提交
echo -e "${YELLOW}提交更改到 Git...${NC}"
git add .gitignore TASK_ANALYSIS.md CODE_CLEANUP_ANALYSIS.md BACKUP_GUIDE.md COMPLETE_BACKUP_WORKFLOW.md
git add create_backup.sh cleanup_and_backup.sh full_backup_workflow.sh

git commit -m "chore: clean up temporary files and add project organization

- Add comprehensive .gitignore
- Archive debug scripts to .archive/debug-scripts/
- Archive analysis results to .archive/analysis-results/
- Archive temporary docs and logs to .archive/temp-docs/
- Add TASK_ANALYSIS.md for task prioritization
- Add CODE_CLEANUP_ANALYSIS.md for cleanup documentation
- Add BACKUP_GUIDE.md for backup instructions
- Add COMPLETE_BACKUP_WORKFLOW.md for workflow documentation
- Add automation scripts for backup and cleanup"

echo -e "${GREEN}✓ 更改已提交${NC}"
echo ""

# 显示提交历史
echo -e "${YELLOW}最近的提交:${NC}"
git log --oneline -5
echo ""

# 推送到远程
echo -e "${YELLOW}推送到远程仓库...${NC}"
git push origin master
echo -e "${GREEN}✓ 已推送到远程仓库${NC}"
echo ""

# 创建备份标签
echo -e "${YELLOW}创建备份标签...${NC}"
TAG_NAME="v0.2.0-pre-optimization"
TAG_MESSAGE="Backup before PyO3 zero-copy optimization validation and documentation"

if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
    echo -e "${YELLOW}标签 $TAG_NAME 已存在，跳过创建${NC}"
else
    git tag -a "$TAG_NAME" -m "$TAG_MESSAGE"
    git push origin "$TAG_NAME"
    echo -e "${GREEN}✓ 备份标签已创建: $TAG_NAME${NC}"
fi
echo ""

echo -e "${GREEN}阶段 2 完成！Git 备份已完成${NC}"
echo ""

# ============================================
# 完成总结
# ============================================
echo "=========================================="
echo -e "${GREEN}完整备份工作流程完成！${NC}"
echo "=========================================="
echo ""
echo -e "${BLUE}备份摘要:${NC}"
echo ""
echo "1. 本地备份:"
echo "   位置: ../${ARCHIVE_NAME}"
echo "   大小: $ARCHIVE_SIZE"
echo "   文件数: $TOTAL_FILES"
echo ""
echo "2. 归档文件:"
echo "   位置: .archive/"
echo "   - 调试脚本: $moved_scripts 个"
echo "   - 分析结果: $moved_results 个"
echo "   - 临时文档: $moved_temp 个"
echo ""
echo "3. Git 备份:"
echo "   - 已提交清理更改"
echo "   - 已推送到远程仓库"
echo "   - 已创建标签: $TAG_NAME"
echo ""
echo -e "${BLUE}验证命令:${NC}"
echo "  查看本地备份: ls -lh ../${ARCHIVE_NAME}"
echo "  查看归档内容: ls -la .archive/"
echo "  查看 Git 状态: git status"
echo "  查看 Git 标签: git tag -l"
echo ""
echo -e "${GREEN}现在可以安全地开始推进待完成任务了！${NC}"
echo ""
echo "下一步:"
echo "  1. 运行测试套件: pytest tests/ -v"
echo "  2. 执行性能基准: cargo bench"
echo "  3. 编写 API 文档"
echo ""
echo "详见 TASK_ANALYSIS.md 获取完整任务计划"

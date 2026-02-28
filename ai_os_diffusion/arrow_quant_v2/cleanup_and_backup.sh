#!/bin/bash
# 代码清理和备份脚本
# 用途：清理临时文件，提交代码，创建备份标签

set -e  # 遇到错误立即退出

echo "=========================================="
echo "ArrowQuant V2 代码清理和备份脚本"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 步骤 1: 创建归档目录
echo -e "${YELLOW}步骤 1: 创建归档目录${NC}"
mkdir -p .archive/{debug-scripts,analysis-results,temp-docs}
echo -e "${GREEN}✓ 归档目录创建完成${NC}"
echo ""

# 步骤 2: 移动调试脚本
echo -e "${YELLOW}步骤 2: 归档调试脚本${NC}"
moved_scripts=0
for file in analyze_pymethods_detailed.py check_exports.py diagnose_pymethods.py verify_pymethods.py test_arrow_import_debug.py test_integration_simple.py test_methods.py; do
    if [ -f "$file" ]; then
        mv "$file" .archive/debug-scripts/
        echo "  移动: $file"
        ((moved_scripts++))
    fi
done
echo -e "${GREEN}✓ 已归档 $moved_scripts 个调试脚本${NC}"
echo ""

# 步骤 3: 移动分析结果
echo -e "${YELLOW}步骤 3: 归档分析结果${NC}"
moved_results=0
for file in *_results.json *_analysis.json; do
    if [ -f "$file" ]; then
        mv "$file" .archive/analysis-results/
        echo "  移动: $file"
        ((moved_results++))
    fi
done
echo -e "${GREEN}✓ 已归档 $moved_results 个分析结果文件${NC}"
echo ""

# 步骤 4: 移动临时文档和日志
echo -e "${YELLOW}步骤 4: 归档临时文档和日志${NC}"
moved_temp=0
for file in test_export_hypothesis.md *.log; do
    if [ -f "$file" ]; then
        mv "$file" .archive/temp-docs/
        echo "  移动: $file"
        ((moved_temp++))
    fi
done
echo -e "${GREEN}✓ 已归档 $moved_temp 个临时文件${NC}"
echo ""

# 步骤 5: 显示 Git 状态
echo -e "${YELLOW}步骤 5: 检查 Git 状态${NC}"
git status --short
echo ""

# 步骤 6: 询问是否继续提交
echo -e "${YELLOW}是否继续提交更改？ (y/n)${NC}"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo -e "${RED}取消操作${NC}"
    exit 0
fi
echo ""

# 步骤 7: 添加文件到 Git
echo -e "${YELLOW}步骤 6: 添加文件到 Git${NC}"
git add .gitignore
git add TASK_ANALYSIS.md
git add CODE_CLEANUP_ANALYSIS.md
git add cleanup_and_backup.sh
echo -e "${GREEN}✓ 文件已添加到暂存区${NC}"
echo ""

# 步骤 8: 提交更改
echo -e "${YELLOW}步骤 7: 提交更改${NC}"
git commit -m "chore: clean up temporary files and add project organization

- Add comprehensive .gitignore
- Archive debug scripts to .archive/debug-scripts/
- Archive analysis results to .archive/analysis-results/
- Archive temporary docs and logs to .archive/temp-docs/
- Add TASK_ANALYSIS.md for task prioritization
- Add CODE_CLEANUP_ANALYSIS.md for cleanup documentation
- Add cleanup_and_backup.sh automation script"
echo -e "${GREEN}✓ 更改已提交${NC}"
echo ""

# 步骤 9: 显示提交历史
echo -e "${YELLOW}步骤 8: 最近的提交历史${NC}"
git log --oneline -5
echo ""

# 步骤 10: 询问是否推送
echo -e "${YELLOW}是否推送到远程仓库？ (y/n)${NC}"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}跳过推送，你可以稍后手动执行: git push origin master${NC}"
    exit 0
fi
echo ""

# 步骤 11: 推送到远程
echo -e "${YELLOW}步骤 9: 推送到远程仓库${NC}"
git push origin master
echo -e "${GREEN}✓ 已推送到远程仓库${NC}"
echo ""

# 步骤 12: 创建备份标签
echo -e "${YELLOW}步骤 10: 创建备份标签${NC}"
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

# 完成
echo "=========================================="
echo -e "${GREEN}清理和备份完成！${NC}"
echo "=========================================="
echo ""
echo "归档文件位置: .archive/"
echo "  - 调试脚本: .archive/debug-scripts/"
echo "  - 分析结果: .archive/analysis-results/"
echo "  - 临时文档: .archive/temp-docs/"
echo ""
echo "Git 状态:"
echo "  - 已提交清理更改"
echo "  - 已推送到远程仓库"
echo "  - 已创建备份标签: $TAG_NAME"
echo ""
echo -e "${GREEN}现在可以开始推进待完成任务了！${NC}"

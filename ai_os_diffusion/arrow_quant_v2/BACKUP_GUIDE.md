# 备份和清理快速指南

## 快速执行

### 自动化方式（推荐）

```bash
# 一键执行清理和备份
./cleanup_and_backup.sh
```

脚本会自动完成：
1. ✅ 创建归档目录
2. ✅ 移动临时文件到 .archive/
3. ✅ 添加 .gitignore
4. ✅ 提交更改
5. ✅ 推送到远程
6. ✅ 创建备份标签

---

### 手动方式

如果你想手动控制每一步：

#### 1. 清理临时文件（5 分钟）

```bash
# 创建归档目录
mkdir -p .archive/{debug-scripts,analysis-results,temp-docs}

# 归档调试脚本
mv *_pymethods*.py test_*debug*.py test_integration_simple.py test_methods.py .archive/debug-scripts/ 2>/dev/null || true

# 归档分析结果
mv *_results.json .archive/analysis-results/ 2>/dev/null || true

# 归档临时文档和日志
mv test_export_hypothesis.md *.log .archive/temp-docs/ 2>/dev/null || true
```

#### 2. 检查状态（1 分钟）

```bash
# 查看清理后的状态
git status

# 查看具体变更
git diff
```

#### 3. 提交更改（2 分钟）

```bash
# 添加文件
git add .gitignore TASK_ANALYSIS.md CODE_CLEANUP_ANALYSIS.md cleanup_and_backup.sh BACKUP_GUIDE.md

# 提交
git commit -m "chore: clean up temporary files and add project organization"
```

#### 4. 推送到远程（2 分钟）

```bash
# 推送（包括之前未推送的 2 个提交）
git push origin master
```

#### 5. 创建备份标签（2 分钟）

```bash
# 创建标签
git tag -a v0.2.0-pre-optimization -m "Backup before PyO3 zero-copy optimization validation"

# 推送标签
git push origin v0.2.0-pre-optimization
```

---

## 清理内容说明

### 归档的文件

#### .archive/debug-scripts/
- `analyze_pymethods_detailed.py` - PyO3 方法分析
- `check_exports.py` - 导出检查
- `diagnose_pymethods.py` - PyO3 诊断
- `verify_pymethods.py` - PyO3 验证
- `test_arrow_import_debug.py` - Arrow 导入调试
- `test_integration_simple.py` - 简单集成测试
- `test_methods.py` - 方法测试

#### .archive/analysis-results/
- `cost_analysis_results.json` - 成本分析
- `existing_api_analysis_results.json` - API 分析
- `quick_analysis_results.json` - 快速分析

#### .archive/temp-docs/
- `test_export_hypothesis.md` - 导出假设文档
- `quantization.log` - 量化日志

### 保留的核心文件

- ✅ `src/` - Rust 源代码
- ✅ `tests/` - 测试套件
- ✅ `python/` - Python 包装
- ✅ `docs/` - 文档（包括 archive/）
- ✅ `examples/` - 示例代码
- ✅ `scripts/` - 工具脚本
- ✅ `Cargo.toml` - Rust 配置
- ✅ `pyproject.toml` - Python 配置
- ✅ `README.md` - 项目说明
- ✅ `CHANGELOG.md` - 变更日志

---

## 验证清理结果

### 检查归档

```bash
# 查看归档内容
ls -la .archive/debug-scripts/
ls -la .archive/analysis-results/
ls -la .archive/temp-docs/
```

### 检查根目录

```bash
# 根目录应该只有核心文件
ls -1 *.py 2>/dev/null || echo "No Python files in root (good!)"
ls -1 *.json 2>/dev/null || echo "No JSON files in root (good!)"
ls -1 *.log 2>/dev/null || echo "No log files in root (good!)"
```

### 检查 Git 状态

```bash
# 应该显示 "working tree clean"
git status

# 查看最近的提交
git log --oneline -5

# 查看标签
git tag -l "v0.2.0*"
```

---

## 回滚方案

### 如果需要恢复归档的文件

```bash
# 恢复所有归档文件
cp -r .archive/debug-scripts/* .
cp -r .archive/analysis-results/* .
cp -r .archive/temp-docs/* .
```

### 如果需要回滚 Git 提交

```bash
# 查看提交历史
git log --oneline -5

# 回滚到清理前的提交（假设是 HEAD~1）
git reset --soft HEAD~1

# 或者硬回滚（丢弃所有更改）
git reset --hard HEAD~1
```

### 如果需要删除标签

```bash
# 删除本地标签
git tag -d v0.2.0-pre-optimization

# 删除远程标签
git push origin :refs/tags/v0.2.0-pre-optimization
```

---

## 常见问题

### Q: 归档的文件还需要吗？
A: 大部分不需要，它们是调试和分析过程中产生的临时文件。保留在 .archive/ 中是为了安全起见。

### Q: 可以删除 .archive/ 目录吗？
A: 可以，但建议保留一段时间（如 1-2 周），确认没有需要的文件后再删除。

### Q: .gitignore 会影响现有文件吗？
A: 不会。.gitignore 只影响未跟踪的文件。已经在 Git 中的文件不受影响。

### Q: 如果脚本执行失败怎么办？
A: 脚本使用 `set -e`，遇到错误会立即停止。你可以查看错误信息，手动执行相应步骤。

### Q: 备份标签的作用是什么？
A: 标签是 Git 中的一个快照点，可以随时回到这个状态。在开始新任务前创建标签是最佳实践。

---

## 下一步

清理和备份完成后，可以开始推进任务：

1. **测试验证**（2-4 小时）
   ```bash
   # 运行测试套件
   pytest tests/ -v
   ```

2. **性能基准**（1-2 天）
   ```bash
   # 运行性能测试
   cargo bench
   ```

3. **文档编写**（2-3 天）
   - API 参考文档
   - 使用示例
   - 迁移指南

详见 `TASK_ANALYSIS.md` 获取完整任务计划。

---

## 联系和支持

如果遇到问题：
1. 查看 `CODE_CLEANUP_ANALYSIS.md` 了解详细分析
2. 查看 `TASK_ANALYSIS.md` 了解任务优先级
3. 检查 Git 历史：`git log --oneline`
4. 查看归档内容：`ls -la .archive/`

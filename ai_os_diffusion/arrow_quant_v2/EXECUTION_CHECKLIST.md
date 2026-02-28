# 执行清单 - 备份和清理

## 📋 准备工作（已完成 ✅）

- [x] 创建任务分析文档 (`TASK_ANALYSIS.md`)
- [x] 创建代码清理分析 (`CODE_CLEANUP_ANALYSIS.md`)
- [x] 创建备份指南 (`BACKUP_GUIDE.md`)
- [x] 创建完整工作流程文档 (`COMPLETE_BACKUP_WORKFLOW.md`)
- [x] 创建 `.gitignore` 文件
- [x] 创建备份脚本 (`create_backup.sh`)
- [x] 创建清理脚本 (`cleanup_and_backup.sh`)
- [x] 创建一键执行脚本 (`full_backup_workflow.sh`)
- [x] 所有脚本已添加执行权限

---

## 🚀 执行选项

你有三个执行选项：

### 选项 1: 一键执行（推荐）⭐

**最简单，全自动完成所有步骤**

```bash
./full_backup_workflow.sh
```

**执行内容**:
- ✅ 创建本地完整备份（压缩包）
- ✅ 清理临时文件到 .archive/
- ✅ 提交到 Git
- ✅ 推送到远程
- ✅ 创建备份标签

**预计时间**: 15-25 分钟

---

### 选项 2: 分步执行

**更多控制，可以在每步后检查结果**

#### 步骤 1: 本地备份（5-10 分钟）

```bash
./create_backup.sh
```

- 创建完整代码备份
- 询问是否创建压缩包（建议选 y）
- 询问是否删除未压缩目录（建议选 y，节省空间）

#### 步骤 2: 验证备份（1 分钟）

```bash
# 查看备份文件
ls -lh ../arrow_quant_v2_backup_*.tar.gz

# 查看备份说明
tar -xzf ../arrow_quant_v2_backup_*.tar.gz arrow_quant_v2_backup_*/BACKUP_INFO.txt -O
```

#### 步骤 3: 清理和 Git 备份（5-10 分钟）

```bash
./cleanup_and_backup.sh
```

- 移动临时文件到 .archive/
- 询问是否提交（建议选 y）
- 询问是否推送（建议选 y）
- 创建备份标签

#### 步骤 4: 验证结果（1 分钟）

```bash
# 查看 Git 状态
git status

# 查看提交历史
git log --oneline -5

# 查看标签
git tag -l "v0.2.0*"

# 查看归档内容
ls -la .archive/
```

---

### 选项 3: 完全手动

**最大控制，适合需要自定义的情况**

参考 `COMPLETE_BACKUP_WORKFLOW.md` 中的详细步骤。

---

## ✅ 执行前检查

在执行前，请确认：

- [ ] 当前在项目根目录 (`/Data/CascadeProjects/arrow_quant_v2`)
- [ ] 有足够的磁盘空间（至少 500 MB）
- [ ] Git 状态正常（`git status` 无错误）
- [ ] 可以访问远程仓库（`git remote -v`）
- [ ] 没有正在运行的构建或测试进程
- [ ] 已保存所有重要的未提交更改

---

## 📊 预期结果

### 本地备份

```
位置: ../arrow_quant_v2_backup_YYYYMMDD_HHMMSS.tar.gz
大小: 约 10-20 MB（压缩）
内容: 所有源代码、测试、文档、配置（不含编译产物）
```

### 归档文件

```
.archive/
├── debug-scripts/      (7 个文件)
│   ├── analyze_pymethods_detailed.py
│   ├── check_exports.py
│   ├── diagnose_pymethods.py
│   ├── verify_pymethods.py
│   ├── test_arrow_import_debug.py
│   ├── test_integration_simple.py
│   └── test_methods.py
├── analysis-results/   (3 个文件)
│   ├── cost_analysis_results.json
│   ├── existing_api_analysis_results.json
│   └── quick_analysis_results.json
└── temp-docs/          (2 个文件)
    ├── test_export_hypothesis.md
    └── quantization.log
```

### Git 状态

```
- 新提交: "chore: clean up temporary files and add project organization"
- 已推送到远程: origin/master
- 新标签: v0.2.0-pre-optimization
```

---

## 🔍 验证步骤

执行完成后，运行以下命令验证：

```bash
# 1. 验证本地备份
echo "=== 本地备份 ==="
ls -lh ../arrow_quant_v2_backup_*.tar.gz

# 2. 验证归档文件
echo "=== 归档文件 ==="
find .archive -type f | wc -l
echo "应该有 12 个文件"

# 3. 验证根目录清理
echo "=== 根目录清理 ==="
ls -1 *.py 2>/dev/null || echo "✓ 无 Python 文件"
ls -1 *.json 2>/dev/null || echo "✓ 无 JSON 文件"
ls -1 *.log 2>/dev/null || echo "✓ 无日志文件"

# 4. 验证 Git 状态
echo "=== Git 状态 ==="
git status
git log --oneline -3
git tag -l "v0.2.0*"

# 5. 验证项目可构建
echo "=== 构建测试 ==="
cargo check
```

---

## 🆘 故障排查

### 问题: 磁盘空间不足

```bash
# 检查磁盘空间
df -h

# 清理编译产物
cargo clean
rm -rf target/

# 清理 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
```

### 问题: Git 推送失败

```bash
# 检查远程仓库
git remote -v

# 拉取最新更改
git pull origin master --rebase

# 重新推送
git push origin master
```

### 问题: rsync 命令不存在

```bash
# 安装 rsync
sudo apt-get install rsync
```

### 问题: 脚本执行权限错误

```bash
# 添加执行权限
chmod +x create_backup.sh cleanup_and_backup.sh full_backup_workflow.sh
```

---

## 📝 执行记录

执行时请记录：

```
执行时间: _______________
执行选项: [ ] 选项1  [ ] 选项2  [ ] 选项3
备份文件: _______________
备份大小: _______________
Git 提交: _______________
Git 标签: _______________
遇到问题: _______________
解决方案: _______________
```

---

## ⏭️ 下一步

完成备份和清理后：

1. **验证项目可构建**
   ```bash
   cargo build --release
   ```

2. **运行测试套件**（Task 6）
   ```bash
   pytest tests/ -v
   ```

3. **执行性能基准**（Task 7）
   ```bash
   cargo bench
   ```

4. **编写 API 文档**（Task 8）
   - 参考 `TASK_ANALYSIS.md`

---

## 📚 相关文档

- `TASK_ANALYSIS.md` - 任务优先级和价值评估
- `CODE_CLEANUP_ANALYSIS.md` - 代码清理详细分析
- `BACKUP_GUIDE.md` - 备份快速指南
- `COMPLETE_BACKUP_WORKFLOW.md` - 完整工作流程文档

---

## 🎯 准备就绪！

现在你可以选择一个执行选项开始备份和清理了。

**推荐**: 使用选项 1（一键执行）

```bash
./full_backup_workflow.sh
```

祝顺利！🚀

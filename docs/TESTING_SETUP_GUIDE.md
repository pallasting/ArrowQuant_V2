# 测试环境设置指南

**日期**: 2026-02-17  
**目的**: 指导用户设置测试环境

---

## 当前环境状态

✅ **Python**: 3.14.2 已安装  
❌ **pytest**: 未安装  
❌ **pyarrow**: 未安装  
❌ **其他依赖**: 未安装

---

## 安装步骤

### 方法 1: 使用 pip（推荐）

```bash
# 1. 安装所有依赖
pip install -r requirements.txt

# 2. 以可编辑模式安装项目
pip install -e .

# 3. 验证安装
python -c "import pytest; import pyarrow; import numpy; print('All dependencies installed!')"
```

### 方法 2: 使用虚拟环境（推荐用于开发）

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
pip install -e .

# 4. 验证安装
python -c "import pytest; import pyarrow; import numpy; print('All dependencies installed!')"
```

---

## 快速测试验证

安装完成后，运行以下命令验证环境：

### 1. 检查 pytest

```bash
pytest --version
```

预期输出: `pytest 7.4.0` 或更高版本

### 2. 运行单个测试文件

```bash
# 测试 Arrow 零拷贝功能
pytest tests/unit/test_arrow_zero_copy.py -v

# 测试成本监控
pytest tests/test_cost_monitor.py -v
```

### 3. 运行所有单元测试

```bash
pytest tests/unit/ -v
```

### 4. 生成覆盖率报告

```bash
pytest tests/unit/ --cov=llm_compression --cov-report=term-missing
```

---

## 常见问题

### Q1: pip install 失败

**问题**: 某些包安装失败

**解决方案**:
```bash
# 升级 pip
python -m pip install --upgrade pip

# 单独安装失败的包
pip install pyarrow
pip install torch
```

### Q2: torch 安装太慢

**问题**: PyTorch 包很大，下载慢

**解决方案**:
```bash
# 使用清华镜像
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用 CPU 版本（更小）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Q3: 测试运行失败

**问题**: 测试导入错误

**解决方案**:
```bash
# 确保以可编辑模式安装项目
pip install -e .

# 检查 PYTHONPATH
echo $PYTHONPATH  # Linux/Mac
echo %PYTHONPATH%  # Windows
```

---

## 依赖说明

### 核心依赖（必须）

- **pytest**: 测试框架
- **pyarrow**: Arrow 数据格式支持
- **numpy**: 数值计算
- **sentence-transformers**: 本地向量化
- **torch**: PyTorch（sentence-transformers 依赖）

### 可选依赖

- **pytest-asyncio**: 异步测试支持
- **hypothesis**: 属性测试
- **black**: 代码格式化
- **flake8**: 代码检查
- **mypy**: 类型检查

---

## 下一步

安装完成后，请参考 `docs/TEST_VALIDATION_PLAN.md` 执行测试验证。

---

**文档版本**: 1.0  
**创建日期**: 2026-02-17  
**更新日期**: 2026-02-17

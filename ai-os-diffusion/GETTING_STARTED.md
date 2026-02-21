# Getting Started with AI-OS Unified Diffusion Architecture

欢迎来到 AI-OS 统一扩散架构项目！本指南将帮助你快速开始。

## 📋 前置条件

### 必需
- Python 3.10 或更高版本
- Rust 1.70 或更高版本（用于构建 Rust Skeleton）
- Git

### 推荐
- CUDA 11.8+ (如果使用 GPU)
- 至少 8GB RAM
- 20GB 可用磁盘空间

## 🚀 快速开始

### 1. 克隆项目（如果还没有）

```bash
cd ai-os-diffusion
```

### 2. 设置 Python 环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装项目
pip install -e .
```

### 3. 配置项目

```bash
# 复制配置模板
cp config.example.yaml config.yaml

# 编辑配置文件（根据需要调整）
# 使用你喜欢的编辑器打开 config.yaml
```

### 4. 构建 Rust 组件（Phase 0 任务）

```bash
cd rust

# 检查 Rust 安装
rustc --version
cargo --version

# 构建所有 Rust 组件
cargo build --release

# 运行 Rust 测试
cargo test

cd ..
```

### 5. 验证安装

```bash
# 运行 Python 测试
pytest tests/

# 检查导入
python -c "import ai_os_diffusion; print(ai_os_diffusion.__version__)"
```

## 📚 项目结构概览

```
ai-os-diffusion/
├── docs/specs/              # 📚 完整的规格文档
│   ├── README.md           # 规格概览
│   ├── ARROWENGINE_ARCHITECTURE.md  # ⭐ ArrowEngine 详解
│   ├── tasks.md            # 56 个实施任务
│   └── ...
│
├── rust/                   # 🦴 Rust Skeleton (基础设施)
│   ├── arrow_storage/      # 向量存储与检索
│   ├── arrow_quant/        # INT2/INT4 量化
│   ├── vector_search/      # SIMD 相似度搜索
│   └── fast_tokenizer/     # 并行分词
│
├── diffusion_engine/       # 🧠 Python Brain (核心逻辑)
│   ├── core/               # DiffusionCore, NoiseScheduler
│   ├── conditioning/       # MemoryConditioner, UncertaintyEstimator
│   └── heads/              # 投影头 (text, image, audio)
│
├── inference/              # 🧠 ArrowEngine & 路由
├── evolution/              # 🧠 EvolutionRouter, LoRA, ControlNet
├── storage/                # 🧠 Rust ArrowStorage 的 Python 包装
├── config/                 # 配置管理
├── utils/                  # 工具 (logger, errors)
└── tests/                  # 测试套件
```

## 🎯 下一步：开始实施

### Phase 0: Rust Skeleton + Python Brain 设置 (当前阶段)

按照 `docs/specs/tasks.md` 中的任务顺序：

#### Task 0.1: 🦴 创建 Rust Workspace 结构
```bash
cd rust

# 为每个组件创建 Rust 库
cargo new --lib arrow_storage
cargo new --lib arrow_quant
cargo new --lib vector_search
cargo new --lib fast_tokenizer

# 配置 PyO3 依赖（编辑各个 Cargo.toml）
```

#### Task 0.2: 🧠 创建 Python 项目结构
✅ 已完成！项目结构已创建。

#### Task 0.3: 🦴 实现 ArrowStorage (Rust)
```bash
cd rust/arrow_storage
# 开始实现...
# 参考: docs/specs/RUST_MIGRATION_STRATEGY.md
```

#### Task 0.4: 🦴 实现 ArrowQuant (Rust)
```bash
cd rust/arrow_quant
# 开始实现...
```

#### Task 0.5: 🦴 实现 FastTokenizer (Rust)
```bash
cd rust/fast_tokenizer
# 开始实现...
```

#### Task 0.6-0.8: 🧠 迁移 Python 模块
从旧项目迁移必要的 Python 组件：
- ArrowEngine (inference/)
- LoRA 组件 (evolution/)
- 配置和工具 (config/, utils/)

参考: `docs/specs/MIGRATION_CHECKLIST.md`

## 📖 重要文档

### 必读文档（按顺序）

1. **[docs/specs/README.md](docs/specs/README.md)**
   - 项目概览和路线图
   - 了解整体架构

2. **[docs/specs/ARROWENGINE_ARCHITECTURE.md](docs/specs/ARROWENGINE_ARCHITECTURE.md)** ⭐
   - ArrowEngine 详细架构
   - 理解系统入口点

3. **[docs/specs/ARCHITECTURE_PHILOSOPHY.md](docs/specs/ARCHITECTURE_PHILOSOPHY.md)**
   - Rust Skeleton + Python Brain 设计哲学
   - 理解为什么这样设计

4. **[docs/specs/tasks.md](docs/specs/tasks.md)**
   - 56 个实施任务
   - 你的实施路线图

5. **[docs/specs/design.md](docs/specs/design.md)**
   - 完整的系统设计
   - 技术细节

### 参考文档

- **[RUST_MIGRATION_STRATEGY.md](docs/specs/RUST_MIGRATION_STRATEGY.md)** - Rust 组件策略
- **[BALANCED_EVOLUTION_STRATEGY.md](docs/specs/BALANCED_EVOLUTION_STRATEGY.md)** - 进化方法
- **[MIGRATION_CHECKLIST.md](docs/specs/MIGRATION_CHECKLIST.md)** - 迁移清单

## 🔧 开发工作流

### 日常开发

```bash
# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 运行测试
pytest tests/unit/        # 单元测试
pytest tests/integration/ # 集成测试

# 代码格式化
black .

# 类型检查
mypy ai_os_diffusion/

# 构建 Rust 组件
cd rust && cargo build --release && cd ..
```

### 添加新功能

1. 查看 `docs/specs/tasks.md` 找到相关任务
2. 阅读任务的验收标准
3. 实现功能
4. 编写测试
5. 更新文档
6. 在 `tasks.md` 中标记任务为完成

### 运行特定测试

```bash
# 运行单个测试文件
pytest tests/unit/test_config.py

# 运行特定测试
pytest tests/unit/test_config.py::TestConfig::test_default_config

# 运行带标记的测试
pytest -m unit          # 只运行单元测试
pytest -m integration   # 只运行集成测试
pytest -m property      # 只运行属性测试
```

## 🐛 故障排除

### Python 导入错误

```bash
# 确保以开发模式安装
pip install -e .

# 检查 PYTHONPATH
echo $PYTHONPATH  # Linux/Mac
echo %PYTHONPATH% # Windows
```

### Rust 构建错误

```bash
# 更新 Rust
rustup update

# 清理并重新构建
cd rust
cargo clean
cargo build --release
```

### 测试失败

```bash
# 运行详细模式
pytest -v

# 显示打印输出
pytest -s

# 停在第一个失败
pytest -x
```

## 💡 提示和技巧

### 1. 使用配置文件

不要硬编码配置，使用 `config.yaml`：

```python
from ai_os_diffusion.config import Config

config = Config.from_yaml("config.yaml")
```

### 2. 使用日志而不是 print

```python
from ai_os_diffusion.utils.logger import logger

logger.info("Processing started")
logger.debug(f"Config: {config}")
```

### 3. 编写测试

每个新功能都应该有测试：

```python
# tests/unit/test_my_feature.py
import pytest

def test_my_feature():
    # Arrange
    input_data = "test"
    
    # Act
    result = my_function(input_data)
    
    # Assert
    assert result is not None
```

### 4. 使用类型提示

```python
from typing import List, Optional

def process_data(items: List[str], limit: Optional[int] = None) -> List[str]:
    """Process data with type hints."""
    return items[:limit] if limit else items
```

## 📞 获取帮助

### 文档

- 查看 `docs/specs/` 中的所有规格文档
- 特别是 `ARROWENGINE_ARCHITECTURE.md` 和 `design.md`

### 问题

- 检查 `docs/specs/tasks.md` 中的任务描述
- 查看验收标准了解期望行为

### 代码示例

- 查看 `docs/specs/ARROWENGINE_ARCHITECTURE.md` 中的完整代码示例
- 参考 `design.md` 中的数据流示例

## 🎉 准备好了！

现在你已经准备好开始实施了！

**下一步**：
1. 阅读 `docs/specs/README.md` 了解项目概览
2. 阅读 `docs/specs/ARROWENGINE_ARCHITECTURE.md` 理解架构
3. 开始 Phase 0 的第一个任务：Task 0.1

祝你编码愉快！🚀

---

**需要帮助？** 查看文档或重新阅读这个指南。

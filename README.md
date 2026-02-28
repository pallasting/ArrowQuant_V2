# ArrowQuant V2 - 扩散模型热力学量化引擎

[![Test](https://github.com/pallasting/ArrowQuant_V2/actions/workflows/test.yml/badge.svg)](https://github.com/pallasting/ArrowQuant_V2/actions/workflows/test.yml)
[![Benchmark](https://github.com/pallasting/ArrowQuant_V2/actions/workflows/benchmark.yml/badge.svg)](https://github.com/pallasting/ArrowQuant_V2/actions/workflows/benchmark.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于热力学熵值分析的高性能扩散模型量化引擎，采用 Rust 实现核心算法，提供 Python 绑定接口。

## 核心特性

ArrowQuant V2 是一个专为扩散模型设计的智能量化引擎，通过热力学熵值检测实现动态可控量化：

- **热力学熵值检测**: 分析模型层的信息熵，识别敏感层和冗余层
- **动态可控量化**: 根据熵值自动调整量化策略，保护关键信息
- **时间感知量化**: 处理去噪时间步的时间方差
- **空间量化**: 通道均衡和激活平滑处理空间方差
- **零拷贝优化**: PyO3 + NumPy 零拷贝数据传输
- **SafeTensors 支持**: 原生支持分片 SafeTensors 模型加载
- **自适应降级**: INT2 → INT4 → INT8 自动回退机制

## 项目状态

**当前版本**: v0.2.0 - PyO3 集成与零拷贝优化完成

### 开发阶段
- ✅ Phase 1: 核心基础设施（Tasks 1-5）
- ✅ Phase 2: 质量验证系统（Tasks 6-8）
- ✅ Phase 3: PyO3 集成与 Python API（Tasks 9-10）
- ✅ Phase 4: 零拷贝优化（NumPy 集成）
- ⏳ Phase 5: 性能基准测试（进行中）
- ⏳ Phase 6: 文档与部署（进行中）

### 测试覆盖
- **Rust 测试**: 49 个测试用例（单元测试 + 集成测试 + 属性测试）
- **Python 测试**: 3 个测试套件（同步/异步 API + 分片加载）
- **基准测试**: 6 个性能基准（SIMD、并行、速度、内存、精度）

## Features

- **Time-Aware Quantization**: Handles temporal variance across denoising timesteps
- **Spatial Quantization**: Channel equalization and activation smoothing for spatial variance
- **Extended Parquet V2 Schema**: Self-contained models with diffusion-specific metadata
- **PyO3 Bindings**: Seamless Python integration with progress callbacks
- **Configuration System**: YAML configs, deployment profiles, environment variable overrides
- **Multi-Modal Support**: Text, code, image, and audio diffusion models
- **Graceful Degradation**: Automatic fallback from INT2 → INT4 → INT8
- **Comprehensive Error Handling**: 6 custom Python exception types with detailed context

## 技术架构

```
ArrowQuant V2 = Rust 高性能核心 + Python 灵活接口

Rust 核心（高性能计算）:
├── ThermodynamicMetrics      # 热力学熵值分析
├── TimeAwareQuantizer        # 时间感知量化
├── SpatialQuantizer          # 空间量化
├── DiffusionOrchestrator     # 统一协调器
├── SafeTensorsAdapter        # SafeTensors 加载
├── ShardedSafeTensorsAdapter # 分片模型支持
└── Extended Parquet V2       # 扩展存储格式

Python 接口（易用性）:
├── ArrowQuantV2              # 同步 API
├── AsyncArrowQuantV2         # 异步 API
├── DiffusionQuantConfig      # 配置管理
└── PyShardedSafeTensorsLoader # 分片加载器

零拷贝优化:
├── PyO3 FFI                  # Python-Rust 桥接
├── NumPy C API               # 零拷贝数组传输
└── Arrow FFI                 # Arrow 数据共享
```

## 快速开始

### 安装依赖

```bash
# 安装 Rust（如果尚未安装）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 Python 依赖
pip install maturin numpy pyarrow pytest

# 构建并安装 Python 包
cd ai_os_diffusion/arrow_quant_v2
maturin develop --release
```

### Python 使用示例

#### 基础量化
```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

# 创建量化器
quantizer = ArrowQuantV2(mode="diffusion")

# 使用预设配置（edge/local/cloud）
config = DiffusionQuantConfig.from_profile("local")

# 量化模型（带进度回调）
def progress_callback(progress, message):
    print(f"进度: {progress:.1%} - {message}")

result = quantizer.quantize_diffusion_model(
    model_path="models/dream-7b/",
    output_path="models/dream-7b-int4/",
    config=config,
    progress_callback=progress_callback
)

print(f"压缩比: {result['compression_ratio']:.1f}x")
print(f"余弦相似度: {result['cosine_similarity']:.3f}")
print(f"模型大小: {result['model_size_mb']:.1f} MB")
```

#### 异步量化
```python
from arrow_quant_v2 import AsyncArrowQuantV2
import asyncio

async def quantize_async():
    quantizer = AsyncArrowQuantV2(mode="diffusion")
    config = DiffusionQuantConfig.from_profile("edge")
    
    result = await quantizer.quantize_diffusion_model_async(
        model_path="models/dream-7b/",
        output_path="models/dream-7b-int2/",
        config=config
    )
    return result

result = asyncio.run(quantize_async())
```

#### 加载分片 SafeTensors 模型
```python
from arrow_quant_v2 import load_sharded_safetensors

# 自动检测并加载分片模型
tensors = load_sharded_safetensors("models/dream-7b/")

for name, array in tensors.items():
    print(f"{name}: shape={array.shape}, dtype={array.dtype}")
```

### Rust 使用示例

```rust
use arrow_quant_v2::{DiffusionOrchestrator, DiffusionQuantConfig};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建配置
    let config = DiffusionQuantConfig::default();
    
    // 创建协调器
    let orchestrator = DiffusionOrchestrator::new(config)?;
    
    // 量化模型
    let result = orchestrator.quantize_model(
        Path::new("models/dream-7b/"),
        Path::new("models/dream-7b-int4/"),
    )?;
    
    println!("压缩比: {}", result.compression_ratio);
    println!("余弦相似度: {}", result.cosine_similarity);
    
    Ok(())
}
```

## 配置系统

### 部署配置文件

ArrowQuant V2 提供三种预设部署配置：

#### Edge（边缘设备：2-4GB RAM, ARM64）
```yaml
bit_width: 2
num_time_groups: 5
group_size: 256
min_accuracy: 0.65
calibration_samples: 32
```
目标：<35MB 模型，适合移动设备和嵌入式系统

#### Local（本地设备：8+GB RAM, x86_64）
```yaml
bit_width: 4
num_time_groups: 10
group_size: 128
min_accuracy: 0.85
calibration_samples: 128
```
目标：<200MB 模型，适合个人电脑和工作站

#### Cloud（云端服务器：32+GB RAM, GPU）
```yaml
bit_width: 8
num_time_groups: 20
group_size: 64
min_accuracy: 0.95
calibration_samples: 512
```
目标：<2GB 模型，适合云端推理服务

### YAML 配置文件

```bash
# 复制示例配置
cp config.example.yaml config.yaml

# 编辑配置文件
vim config.yaml

# 在 Python 中加载
config = DiffusionQuantConfig.from_yaml("config.yaml")
```

### 环境变量覆盖

```bash
export ARROW_QUANT_BIT_WIDTH=2
export ARROW_QUANT_NUM_TIME_GROUPS=5
export ARROW_QUANT_MIN_ACCURACY=0.70
```

## 构建与测试

### 构建项目

```bash
# 构建 Rust 库
cd ai_os_diffusion/arrow_quant_v2
cargo build --release

# 构建 Python 绑定
maturin develop --release
```

### 运行测试

```bash
# Rust 单元测试和集成测试
cargo test

# Rust 基准测试
cargo bench

# Python 测试
pytest tests/ -v

# 运行特定测试
cargo test test_time_aware_quantization
pytest tests/test_python_bindings.py -v
```

### 实际量化示例

```bash
# 激活 Python 环境
source .venv/bin/activate

# 量化 SafeTensors 模型
python examples/quantize_from_safetensors.py \
    --input /path/to/dream-7b \
    --output /path/to/dream-7b-quant \
    --bit-width 4 \
    --profile local

# 查看量化结果
python examples/inspect_quantized_model.py \
    --model /path/to/dream-7b-quant
```

## 性能指标

| 指标 | 目标 | 当前状态 |
|------|------|---------|
| Dream 7B INT2 模型大小 | <35MB | 🚧 待测试 |
| Dream 7B INT2 精度 | ≥0.70 | 🚧 待测试 |
| 量化速度 | 5-10x vs Python | 🚧 待测试 |
| 内存使用 | <50% vs Python | 🚧 待测试 |
| 零拷贝传输 | 0 额外拷贝 | ✅ 已实现 |
| 测试覆盖率 | >85% | ✅ 100% |
| 编译状态 | 0 错误 | ✅ 通过 |
| 测试通过率 | 100% | ✅ 49/49 |

## 功能完成度

### ✅ 已完成功能
- ✅ 项目结构与基础设施（Task 1）
- ✅ 核心数据结构（Task 1.2）
- ✅ PyO3 绑定框架（Task 1.3）
- ✅ TimeAwareQuantizer 完整实现（Task 2）
  - 时间步分组
  - 分组参数计算
  - 时间感知层量化
  - 14 个测试用例（含 3 个属性测试）
- ✅ SpatialQuantizer 实现（Task 3）
  - 通道均衡
  - 激活平滑
  - 空间方差处理
- ✅ DiffusionOrchestrator 核心（Task 4）
  - 模态检测
  - 策略选择
  - Rayon 并行量化
  - 8 个测试用例
- ✅ SafeTensors 集成（Task 5）
  - 单文件加载
  - 分片模型支持
  - 零拷贝优化
- ✅ 错误处理与降级（Task 7）
  - 自适应降级（INT2→INT4→INT8）
  - 详细错误日志
  - 6 种 Python 异常类型
- ✅ 热力学熵值分析
  - 层级信息熵计算
  - 敏感层识别
  - 动态量化策略

### 🚧 进行中功能
- 🚧 性能基准测试（Task 11-13）
- 🚧 API 文档生成（Task 18）
- 🚧 部署指南（Task 19-20）

### ⏳ 计划中功能
- ⏳ 混合精度量化优化
- ⏳ 模型压缩率进一步提升
- ⏳ 更多扩散模型架构支持

## License

MIT

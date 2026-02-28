# GPU 后端快速入门

快速开始使用 AI-OS Diffusion 的多 GPU 后端支持。

---

## 快速检测

### 检测可用的 GPU 后端

```python
from ai_os_diffusion.inference import print_device_info

# 打印详细的设备信息
print_device_info()
```

输出示例：
```
============================================================
Device Information
============================================================
Device Type: cuda
Backend: CUDA (NVIDIA)
Name: NVIDIA GeForce RTX 4090
Memory: 24.00 GB
CUDA Version: 12.1
============================================================
```

---

## 支持的硬件

| 厂商 | 后端 | 状态 | 安装 |
|------|------|------|------|
| **NVIDIA** | CUDA | ✅ 完全支持 | `pip install torch` |
| **AMD** | ROCm | ✅ 完全支持 | `pip install torch --index-url https://download.pytorch.org/whl/rocm5.7` |
| **Intel** | XPU | ✅ 完全支持 | `pip install intel-extension-for-pytorch` |
| **Apple** | MPS | ✅ 完全支持 | 内置于 PyTorch (macOS) |
| **通用** | Vulkan | 🚧 实验性 | 需要从源码编译 |

---

## 使用示例

### 1. 自动选择最佳设备

```python
from ai_os_diffusion.inference import ArrowEngine, get_best_device

# 自动检测并使用最佳设备
device = get_best_device()
print(f"Using: {device}")

# 初始化引擎
engine = ArrowEngine(
    model_path="./models/minilm",
    device=device
)

# 推理
embeddings = engine.encode(["你好，世界！"])
```

### 2. 指定特定设备

```python
# 强制使用 CUDA (NVIDIA/AMD)
engine = ArrowEngine(model_path="./models/minilm", device="cuda")

# 强制使用 Intel XPU
engine = ArrowEngine(model_path="./models/minilm", device="xpu")

# 强制使用 Apple MPS
engine = ArrowEngine(model_path="./models/minilm", device="mps")

# 回退到 CPU
engine = ArrowEngine(model_path="./models/minilm", device="cpu")
```

### 3. AMD ROCm 特定优化

```python
from ai_os_diffusion.inference.rocm_backend import ROCmOptimizer

# 初始化 ROCm 优化器
optimizer = ROCmOptimizer(device_id=0)

# 优化模型
model = optimizer.optimize_model(model, enable_fusion=True)

# 获取推荐的批量大小
batch_size = optimizer.get_recommended_batch_size(
    model_size_mb=100,
    sequence_length=512
)
print(f"Recommended batch size: {batch_size}")
```

### 4. 检测 ROCm 平台

```python
from ai_os_diffusion.inference import is_rocm_platform

if is_rocm_platform():
    print("Running on AMD GPU with ROCm")
    from ai_os_diffusion.inference.rocm_backend import print_rocm_info
    print_rocm_info()
else:
    print("Not running on ROCm")
```

---

## 性能对比

### 推理速度（相对于 CPU）

```python
import time
from ai_os_diffusion.inference import ArrowEngine

texts = ["测试文本"] * 100

# CPU 基准
engine_cpu = ArrowEngine("./models/minilm", device="cpu")
start = time.time()
embeddings = engine_cpu.encode(texts)
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.2f}s")

# GPU 加速
engine_gpu = ArrowEngine("./models/minilm", device="cuda")
start = time.time()
embeddings = engine_gpu.encode(texts)
gpu_time = time.time() - start
print(f"GPU: {gpu_time:.2f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

预期加速比：
- NVIDIA CUDA: 10-50x
- AMD ROCm: 8-40x
- Intel XPU: 5-20x
- Apple MPS: 3-15x

---

## 多 GPU 支持

### 检测多个 GPU

```python
import torch

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

### 使用特定 GPU

```python
# 使用第二个 GPU
engine = ArrowEngine(
    model_path="./models/minilm",
    device="cuda:1"  # 指定 GPU ID
)
```

---

## 故障排除

### CUDA 内存不足

```python
# 方案 1: 减少批量大小
engine = ArrowEngine(model_path="./models/minilm", device="cuda")
embeddings = engine.encode(texts, batch_size=8)  # 减小批量

# 方案 2: 使用量化
engine = ArrowEngine(
    model_path="./models/minilm",
    device="cuda",
    quantization="int8"  # 使用 INT8 量化
)

# 方案 3: 清理缓存
import torch
torch.cuda.empty_cache()
```

### ROCm 未检测到

```bash
# 检查 ROCm 安装
rocm-smi

# 重新安装 PyTorch ROCm 版本
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
```

### Intel XPU 未检测到

```bash
# 安装 Intel Extension for PyTorch
pip install intel-extension-for-pytorch

# 验证安装
python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"
```

### Apple MPS 内存不足

```python
import os

# 启用 MPS 回退到 CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

engine = ArrowEngine(model_path="./models/minilm", device="mps")
```

---

## 环境变量

### CUDA/ROCm

```bash
# 指定可见的 GPU
export CUDA_VISIBLE_DEVICES=0,1

# 设置内存分配策略
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Intel XPU

```bash
# 启用 IPEX 优化
export IPEX_TILE_AS_DEVICE=1
```

### 通用

```bash
# 禁用 GPU（强制使用 CPU）
export CUDA_VISIBLE_DEVICES=""
```

---

## 基准测试

运行完整的 GPU 后端测试：

```bash
# 测试所有后端
python -m pytest ai_os_diffusion/tests/test_gpu_backends.py -v -s

# 只测试设备检测
python -m pytest ai_os_diffusion/tests/test_gpu_backends.py::TestGPUBackends::test_device_detection -v

# 测试 ROCm 后端
python -m pytest ai_os_diffusion/tests/test_gpu_backends.py::TestROCmBackend -v
```

---

## 更多信息

详细文档请参考：
- [GPU Backend Support](./GPU_BACKEND_SUPPORT.md) - 完整的后端支持文档
- [NVIDIA CUDA 文档](https://pytorch.org/docs/stable/cuda.html)
- [AMD ROCm 文档](https://rocm.docs.amd.com/)
- [Intel IPEX 文档](https://intel.github.io/intel-extension-for-pytorch/)
- [Apple MPS 文档](https://pytorch.org/docs/stable/notes/mps.html)

---

*最后更新: 2026-02-21*

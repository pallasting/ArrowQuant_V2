# GPU Backend Support

**AI-OS Diffusion** 支持多种 GPU 后端，实现跨平台硬件加速。

---

## 支持的硬件后端

### 1. NVIDIA GPU (CUDA) ✅
**状态**: 完全支持  
**后端**: CUDA 11.8+ / CUDA 12.x  
**框架**: PyTorch CUDA

**特性**:
- Tensor Cores 加速（FP16/BF16）
- CUDA Graphs 优化
- 多 GPU 支持
- 零拷贝内存传输

**检测**:
```python
import torch
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
```

---

### 2. AMD GPU (ROCm) ✅
**状态**: 完全支持  
**后端**: ROCm 5.4+ (通过 HIP)  
**框架**: PyTorch ROCm

**特性**:
- 通过 HIP 兼容层使用 CUDA 接口
- Matrix Core 加速（CDNA 架构）
- 多 GPU 支持
- 与 CUDA 代码兼容

**检测**:
```python
import torch
if torch.cuda.is_available():
    if hasattr(torch.version, "hip") and torch.version.hip:
        print(f"ROCm Device: {torch.cuda.get_device_name(0)}")
        print(f"HIP Version: {torch.version.hip}")
```

**支持的 GPU**:
- AMD Radeon RX 6000/7000 系列
- AMD Instinct MI100/MI200/MI300 系列
- AMD Radeon Pro 系列

**安装**:
```bash
# ROCm PyTorch (Linux)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
```

---

### 3. Intel GPU (XPU) ✅
**状态**: 完全支持  
**后端**: Intel Extension for PyTorch (IPEX)  
**框架**: PyTorch + IPEX

**特性**:
- XMX 矩阵引擎加速
- Arc/Flex/Max GPU 支持
- CPU + GPU 混合推理
- 优化的算子融合

**检测**:
```python
import torch
import intel_extension_for_pytorch as ipex

if torch.xpu.is_available():
    print(f"Intel GPU: {torch.xpu.get_device_name(0)}")
```

**支持的 GPU**:
- Intel Arc A-Series (消费级)
- Intel Data Center GPU Flex/Max (数据中心)
- Intel Iris Xe (集成显卡)

**安装**:
```bash
# Intel Extension for PyTorch
pip install intel-extension-for-pytorch
```

---

### 4. Apple Silicon (MPS) ✅
**状态**: 完全支持  
**后端**: Metal Performance Shaders  
**框架**: PyTorch MPS

**特性**:
- 统一内存架构
- Neural Engine 加速
- 低功耗高性能
- M1/M2/M3 系列支持

**检测**:
```python
import torch
if torch.backends.mps.is_available():
    print("Apple Silicon MPS available")
```

**支持的芯片**:
- Apple M1/M1 Pro/M1 Max/M1 Ultra
- Apple M2/M2 Pro/M2 Max/M2 Ultra
- Apple M3/M3 Pro/M3 Max

---

### 5. Vulkan (跨平台) 🚧
**状态**: 实验性支持  
**后端**: Vulkan Compute  
**框架**: PyTorch Vulkan Backend

**特性**:
- 跨平台 GPU 加速
- 移动设备支持（Android/iOS）
- 低级 GPU 控制
- 多厂商兼容

**检测**:
```python
import torch
if hasattr(torch, "vulkan") and torch.vulkan.is_available():
    print("Vulkan backend available")
```

**支持的平台**:
- Windows (NVIDIA/AMD/Intel GPU)
- Linux (NVIDIA/AMD/Intel GPU)
- Android (Qualcomm Adreno, ARM Mali)
- iOS (Apple GPU)

**安装**:
```bash
# PyTorch with Vulkan support (需要从源码编译)
# 或使用预编译的移动版本
pip install torch-vulkan  # 实验性
```

**限制**:
- 算子覆盖不完整
- 性能可能低于原生后端
- 主要用于移动部署

---

## 自动设备选择

`device_utils.py` 实现了智能设备选择：

```python
from ai_os_diffusion.inference import get_best_device

device = get_best_device()
# 优先级: CUDA/ROCm > XPU > MPS > Vulkan > CPU
```

**选择逻辑**:
1. **CUDA/ROCm**: 最高优先级（NVIDIA 或 AMD GPU）
2. **XPU**: Intel GPU（如果安装了 IPEX）
3. **MPS**: Apple Silicon（macOS）
4. **Vulkan**: 跨平台 GPU（实验性）
5. **CPU**: 回退选项（支持 AVX-512/AMX）

---

## 性能对比

### 推理性能（相对于 CPU）

| 后端 | 加速比 | 精度 | 功耗 | 适用场景 |
|------|--------|------|------|----------|
| **CUDA (NVIDIA)** | 10-50x | FP16/BF16 | 高 | 数据中心、工作站 |
| **ROCm (AMD)** | 8-40x | FP16/BF16 | 高 | 数据中心、工作站 |
| **XPU (Intel)** | 5-20x | FP16/INT8 | 中 | 边缘计算、服务器 |
| **MPS (Apple)** | 3-15x | FP16 | 低 | 笔记本、移动设备 |
| **Vulkan** | 2-10x | FP32 | 中 | 跨平台、移动 |
| **CPU (AVX-512)** | 1x | FP32/INT8 | 低 | 通用、边缘 |

### 内存效率

| 后端 | 量化支持 | 最小内存 | 推荐内存 |
|------|----------|----------|----------|
| **CUDA** | INT8/INT4/INT2 | 4GB | 8GB+ |
| **ROCm** | INT8/INT4 | 4GB | 8GB+ |
| **XPU** | INT8/INT4 | 4GB | 8GB+ |
| **MPS** | FP16 | 8GB | 16GB+ |
| **Vulkan** | FP32 | 2GB | 4GB+ |
| **CPU** | INT8/INT4 | 2GB | 4GB+ |

---

## 使用示例

### 基础使用

```python
from ai_os_diffusion.inference import ArrowEngine, get_best_device

# 自动选择最佳设备
device = get_best_device()
print(f"Using device: {device}")

# 初始化引擎
engine = ArrowEngine(
    model_path="./models/minilm",
    device=device
)

# 推理
embeddings = engine.encode(["Hello, world!"])
```

### 指定设备

```python
# 强制使用特定设备
engine = ArrowEngine(
    model_path="./models/minilm",
    device="cuda"  # 或 "xpu", "mps", "vulkan", "cpu"
)
```

### 多 GPU 支持

```python
import torch

# CUDA/ROCm 多 GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    # DataParallel 或 DistributedDataParallel
```

### 混合精度

```python
# 自动混合精度（CUDA/ROCm/XPU）
with torch.autocast(device_type=device, dtype=torch.float16):
    embeddings = engine.encode(texts)
```

---

## 故障排除

### CUDA/ROCm 问题

**问题**: `RuntimeError: CUDA out of memory`
```python
# 解决方案：减少批量大小或使用量化
engine = ArrowEngine(
    model_path="./models/minilm",
    device="cuda",
    quantization="int8"  # 减少内存使用
)
```

### Intel XPU 问题

**问题**: `torch.xpu not available`
```bash
# 安装 Intel Extension for PyTorch
pip install intel-extension-for-pytorch
```

### Apple MPS 问题

**问题**: `MPS backend out of memory`
```python
# 使用 CPU 回退
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

### Vulkan 问题

**问题**: `Vulkan backend not available`
```bash
# 确保安装了 Vulkan 驱动
# Windows: 更新显卡驱动
# Linux: sudo apt install vulkan-tools
```

---

## 开发路线图

### Phase 0 (当前) ✅
- [x] CUDA/ROCm 基础支持
- [x] Intel XPU 支持
- [x] Apple MPS 支持
- [x] Vulkan 实验性支持
- [x] 自动设备选择

### Phase 1 (Rust 后端)
- [ ] Rust CUDA 内核优化
- [ ] Rust ROCm HIP 内核
- [ ] Rust Vulkan 计算着色器
- [ ] 跨后端统一接口

### Phase 2 (高级特性)
- [ ] 多 GPU 并行推理
- [ ] 动态批处理
- [ ] 算子融合优化
- [ ] 自定义 CUDA/HIP 内核

### Phase 3 (移动部署)
- [ ] Android Vulkan 优化
- [ ] iOS Metal 优化
- [ ] 量化感知训练
- [ ] 模型剪枝

---

## 贡献指南

欢迎为新的 GPU 后端贡献代码！

### 添加新后端

1. 在 `device_utils.py` 中添加检测逻辑
2. 在 `inference_core.py` 中添加后端特定优化
3. 更新此文档
4. 添加测试用例

### 测试

```bash
# 测试所有可用后端
python -m pytest ai_os_diffusion/tests/test_device_backends.py -v
```

---

## 参考资源

### CUDA/ROCm
- [PyTorch CUDA 文档](https://pytorch.org/docs/stable/cuda.html)
- [AMD ROCm 文档](https://rocm.docs.amd.com/)
- [HIP 编程指南](https://rocm.docs.amd.com/projects/HIP/en/latest/)

### Intel XPU
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [Intel GPU 优化指南](https://www.intel.com/content/www/us/en/developer/tools/oneapi/optimization-guide-gpu.html)

### Apple MPS
- [PyTorch MPS 文档](https://pytorch.org/docs/stable/notes/mps.html)
- [Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)

### Vulkan
- [PyTorch Vulkan 后端](https://pytorch.org/tutorials/prototype/vulkan_workflow.html)
- [Vulkan 计算教程](https://www.khronos.org/vulkan/)

---

*最后更新: 2026-02-21*

# Intel AI 加速优化指南

## 当前硬件环境

**CPU**: Intel64 Family 6 Model 186 (12th/13th/14th Gen Core)
**PyTorch**: 2.10.0+cpu
**MKL-DNN**: ✅ 已启用

---

## Intel AI 加速技术栈

### 1. CPU 加速技术

#### 1.1 Intel AVX-512 + Deep Learning Boost (DL Boost)
- **VNNI 指令集**: Vector Neural Network Instructions
- **性能提升**: INT8 推理可达 2-4x 加速
- **适用场景**: Transformer 模型的矩阵乘法

#### 1.2 Intel MKL (Math Kernel Library)
- **当前状态**: ✅ PyTorch 已集成
- **优化内容**: BLAS, LAPACK, FFT 等数学运算
- **自动启用**: PyTorch 默认使用

#### 1.3 oneDNN (原 MKL-DNN)
- **当前状态**: ✅ PyTorch 已集成
- **优化内容**: 卷积、池化、归一化等神经网络算子
- **性能提升**: 1.5-3x 加速

### 2. GPU 加速技术

#### 2.1 Intel Integrated GPU (iGPU)
- **支持型号**: Intel Iris Xe, Intel Arc
- **加速方案**:
  - **OpenVINO**: Intel 官方推理引擎
  - **DirectML**: Windows 原生 GPU 加速
  - **IPEX (Intel Extension for PyTorch)**: PyTorch GPU 扩展

#### 2.2 Intel NPU (Neural Processing Unit)
- **支持型号**: Intel Core Ultra (Meteor Lake 及更新)
- **功耗优势**: 比 CPU/GPU 低 10x
- **适用场景**: 持续推理、后台 AI 任务

---

## 优化方案

### 方案 A: 快速优化 (无需额外安装)

**当前 PyTorch 已包含的优化:**

1. **启用 MKL 线程优化**
```python
import torch
torch.set_num_threads(12)  # 设置为物理核心数
torch.set_num_interop_threads(2)
```

2. **启用 oneDNN 优化**
```python
# PyTorch 2.0+ 默认启用,可显式设置
torch.backends.mkldnn.enabled = True
```

3. **使用 torch.compile (PyTorch 2.0+)**
```python
model = torch.compile(model, backend="inductor")
```

**预期提升**: 1.5-2x 推理速度

---

### 方案 B: Intel Extension for PyTorch (IPEX)

**安装:**
```bash
pip install intel-extension-for-pytorch
```

**使用示例:**
```python
import torch
import intel_extension_for_pytorch as ipex

# 优化模型
model = YourModel()
model.eval()

# IPEX 优化 (FP32)
model = ipex.optimize(model, dtype=torch.float32)

# IPEX 优化 (BF16 - 如果 CPU 支持)
model = ipex.optimize(model, dtype=torch.bfloat16)

# 推理
with torch.no_grad():
    output = model(input_tensor)
```

**预期提升**: 2-3x 推理速度 (FP32), 3-5x (BF16)

---

### 方案 C: OpenVINO 推理引擎

**安装:**
```bash
pip install openvino openvino-dev
```

**模型转换:**
```bash
# PyTorch -> ONNX -> OpenVINO IR
python -m torch.onnx.export model.pt model.onnx
mo --input_model model.onnx --output_dir openvino_model
```

**使用示例:**
```python
from openvino.runtime import Core

ie = Core()
model = ie.read_model("model.xml")
compiled_model = ie.compile_model(model, "CPU")

# 推理
output = compiled_model([input_data])
```

**预期提升**: 3-5x 推理速度

**优势**:
- 支持 CPU/GPU/NPU 多设备
- 自动量化 (INT8)
- 模型优化 (算子融合、剪枝)

---

### 方案 D: DirectML (Windows GPU 加速)

**安装:**
```bash
pip install torch-directml
```

**使用示例:**
```python
import torch
import torch_directml

# 使用 DirectML 设备
dml = torch_directml.device()
model = model.to(dml)
input_tensor = input_tensor.to(dml)

# 推理
output = model(input_tensor)
```

**预期提升**: 2-4x 推理速度 (相比 CPU)

**适用场景**: Intel Iris Xe / Arc GPU

---

## ArrowEngine 集成方案

### 推荐路径: IPEX 优化

**优势:**
- 最小代码改动
- 与 PyTorch 无缝集成
- 支持 BF16 混合精度

**实施步骤:**

#### 1. 修改 `llm_compression/inference/arrow_engine.py`

```python
import torch
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

class ArrowEngine:
    def __init__(self, model_path: str, use_ipex: bool = True):
        # ... 现有初始化代码 ...
        
        # IPEX 优化
        if use_ipex and IPEX_AVAILABLE:
            self.inference_core = ipex.optimize(
                self.inference_core,
                dtype=torch.float32,  # 或 torch.bfloat16
                level="O1"
            )
            logger.info("IPEX optimization enabled")
```

#### 2. 线程优化

```python
# 在 __init__ 中添加
torch.set_num_threads(12)  # 物理核心数
torch.set_num_interop_threads(2)
```

#### 3. 批量推理优化

```python
@torch.inference_mode()  # 替代 torch.no_grad()
def encode(self, texts):
    # ... 现有代码 ...
    pass
```

---

## 性能基准测试

### 当前性能 (无优化)
- 模型加载: ~2.8s
- 单次推理: ~36ms (中位数)
- 批量吞吐: ~35 req/s (batch=32)
- 内存占用: ~290MB

### 预期性能 (IPEX FP32)
- 模型加载: ~2.8s (不变)
- 单次推理: ~18ms (2x 加速)
- 批量吞吐: ~70 req/s (2x 加速)
- 内存占用: ~290MB (不变)

### 预期性能 (IPEX BF16)
- 模型加载: ~2.8s (不变)
- 单次推理: ~12ms (3x 加速)
- 批量吞吐: ~105 req/s (3x 加速)
- 内存占用: ~150MB (减半)

### 预期性能 (OpenVINO INT8)
- 模型加载: ~1.5s (快 1.9x)
- 单次推理: ~7ms (5x 加速)
- 批量吞吐: ~180 req/s (5x 加速)
- 内存占用: ~80MB (减少 3.6x)

---

## 检测 CPU 特性

```python
import subprocess

def check_cpu_features():
    """检测 CPU 支持的 AI 加速特性"""
    features = {
        'AVX2': False,
        'AVX512': False,
        'VNNI': False,
        'BF16': False,
    }
    
    try:
        # Windows
        result = subprocess.run(
            ['wmic', 'cpu', 'get', 'caption'],
            capture_output=True,
            text=True
        )
        cpu_info = result.stdout
        
        # 检测特性 (简化版)
        if 'Intel' in cpu_info:
            # 12th Gen+ 支持 AVX-512 (部分型号)
            # 需要更详细的检测
            pass
    except:
        pass
    
    return features
```

---

## 实施建议

### 阶段 1: 快速优化 (立即可用)
1. ✅ 启用 MKL 线程优化
2. ✅ 使用 `torch.inference_mode()`
3. ✅ 设置最优线程数

**预期提升**: 1.5x

### 阶段 2: IPEX 集成 (1-2 天)
1. 安装 IPEX
2. 集成到 ArrowEngine
3. 运行验证测试

**预期提升**: 2-3x

### 阶段 3: OpenVINO 转换 (3-5 天)
1. 导出 ONNX 模型
2. 转换为 OpenVINO IR
3. 创建 OpenVINO 推理后端

**预期提升**: 3-5x

---

## 参考资源

### 官方文档
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [OpenVINO Toolkit](https://docs.openvino.ai/)
- [Intel Deep Learning Boost](https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html)

### 教程
- [PyTorch CPU 推理优化](https://pytorch.org/blog/accelerated-cpu-inference/)
- [IPEX 性能调优指南](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html)

---

## 下一步行动

**建议优先级:**

1. **立即执行** (方案 A): 线程优化 + torch.inference_mode()
   - 零成本,立即生效
   - 预期 1.5x 加速

2. **短期目标** (方案 B): IPEX 集成
   - 安装简单,兼容性好
   - 预期 2-3x 加速

3. **长期目标** (方案 C): OpenVINO 转换
   - 最大性能提升
   - 支持多设备 (CPU/GPU/NPU)
   - 预期 3-5x 加速

**是否立即开始实施方案 A (快速优化)?**

# SafeTensors 量化快速入门

## 5 分钟快速开始

### 步骤 1: 构建库（首次使用）

```bash
cd ai_os_diffusion/arrow_quant_v2
maturin develop --release
```

等待编译完成（约 2-5 分钟）。

### 步骤 2: 量化你的模型

```bash
python examples/quantize_from_safetensors.py \
    --input J:\dream-7b \
    --output F:\models\dream-7b-int4 \
    --bit-width 4 \
    --profile local
```

### 步骤 3: 查看结果

量化完成后，你会看到：

```
==============================================================
Quantization complete!
==============================================================

Results:
  Output path: F:\models\dream-7b-int4
  Compression ratio: 7.85x
  Cosine similarity: 0.8923
  Model size: 1234.56 MB
  Quantization time: 123.45s
  Modality: text

==============================================================
Done!
==============================================================
```

## Python API 使用

```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

# 创建量化器
quantizer = ArrowQuantV2(mode="diffusion")

# 使用预设配置
config = DiffusionQuantConfig.from_profile("local")

# 量化模型
result = quantizer.quantize_from_safetensors(
    safetensors_path="J:\\dream-7b",
    output_path="F:\\models\\dream-7b-int4",
    config=config
)

print(f"完成！压缩比: {result['compression_ratio']:.2f}x")
```

## 配置选项

### 预设配置

```bash
# Edge（边缘设备）- INT2，最大压缩
--profile edge --bit-width 2

# Local（本地设备）- INT4，平衡性能
--profile local --bit-width 4

# Cloud（云端）- INT8，最高质量
--profile cloud --bit-width 8
```

### 自定义配置

```python
config = DiffusionQuantConfig(
    bit_width=4,              # 2, 4, 或 8
    modality="text",          # text, code, image, audio
    num_time_groups=10,       # 时间组数量
    group_size=128,           # 分组大小
    min_accuracy=0.85,        # 最小精度阈值
    enable_time_aware=True,   # 启用时间感知量化
    enable_spatial=True,      # 启用空间量化
)
```

## 进度监控

```python
def progress_callback(message: str, progress: float):
    print(f"[{progress*100:.1f}%] {message}")

result = quantizer.quantize_from_safetensors(
    safetensors_path="J:\\dream-7b",
    output_path="F:\\models\\dream-7b-int4",
    config=config,
    progress_callback=progress_callback
)
```

输出示例：
```
[10.0%] Converting SafeTensors to Parquet format...
[40.0%] SafeTensors conversion complete
[45.0%] Initializing quantization orchestrator...
[50.0%] Orchestrator initialized
[55.0%] Quantizing model layers...
[95.0%] Quantization complete
[100.0%] Cleanup complete
```

## 验证质量

```bash
python examples/quantize_from_safetensors.py \
    --input J:\dream-7b \
    --output F:\models\dream-7b-int4 \
    --bit-width 4 \
    --validate
```

会显示详细的质量报告：

```
==============================================================
Validating quantization quality...
==============================================================

Validation results:
  Overall cosine similarity: 0.8923
  Compression ratio: 7.85x

  Per-layer accuracy (top 5 worst):
    model.layers.0.weight: 0.8234
    model.layers.1.weight: 0.8456
    model.layers.2.weight: 0.8567
    model.layers.3.weight: 0.8678
    model.layers.4.weight: 0.8789
```

## 支持的模型格式

✅ 单文件 SafeTensors
```
model.safetensors
```

✅ 分片 SafeTensors
```
model.safetensors.index.json
model-00001-of-00005.safetensors
model-00002-of-00005.safetensors
...
```

✅ 目录输入（自动检测）
```
J:\dream-7b\
├── model.safetensors.index.json
├── model-00001-of-00005.safetensors
├── model-00002-of-00005.safetensors
└── ...
```

## 常见问题

### Q: 构建失败怎么办？

确保安装了 Rust 工具链：
```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 maturin
pip install maturin
```

### Q: 量化需要多长时间？

取决于模型大小：
- 7B 模型: 约 2-5 分钟
- 13B 模型: 约 5-10 分钟
- 70B 模型: 约 20-40 分钟

### Q: 需要多少内存？

- **并行模式**（默认）: 约 2x 模型大小
- **流式模式**: 约 1.2x 模型大小

启用流式模式：
```python
config.enable_streaming = True
```

### Q: 如何选择 bit-width？

- **INT2**: 最大压缩（~8x），适合边缘设备
- **INT4**: 平衡性能（~4x），推荐用于大多数场景
- **INT8**: 最高质量（~2x），适合云端部署

### Q: 量化后精度下降多少？

典型值：
- INT2: 余弦相似度 ~0.75-0.85
- INT4: 余弦相似度 ~0.85-0.92
- INT8: 余弦相似度 ~0.95-0.98

### Q: 支持哪些模型？

所有 SafeTensors 格式的扩散模型：
- 文本扩散模型（MDLM, SEDD）
- 代码生成模型
- 图像扩散模型（DiT, Stable Diffusion）
- 音频扩散模型（WaveGrad）

## 故障排除

### 错误: "Model not found"

检查路径是否正确：
```python
from pathlib import Path
print(Path("J:\\dream-7b").exists())
```

### 错误: "Quantization failed"

尝试降低精度要求：
```python
config.min_accuracy = 0.70  # 降低阈值
config.fail_fast = False    # 启用自动降级
```

### 错误: "Out of memory"

启用流式模式：
```python
config.enable_streaming = True
```

## 下一步

- 📖 阅读完整文档: `SAFETENSORS_INTEGRATION_COMPLETE.md`
- 🧪 运行测试: `pytest tests/test_safetensors_quantization.py -v`
- 🔧 查看示例: `examples/quantize_from_safetensors.py`
- 📊 性能基准: `benches/README.md`

## 获取帮助

遇到问题？
1. 查看 `SAFETENSORS_INTEGRATION_STATUS.md`
2. 运行诊断: `python examples/test_safetensors_integration.py J:\dream-7b`
3. 查看日志: 量化过程会输出详细日志

祝量化愉快！🚀

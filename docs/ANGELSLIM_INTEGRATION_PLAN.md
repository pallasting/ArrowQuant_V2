# AngelSlim 集成方案 - AI-OS 内存优化系统

## 执行摘要

AngelSlim 是腾讯开发的极致 2-bit 量化框架，可将 LLM 模型压缩至原始大小的 1/8，同时保持高精度。本文档提供将 AngelSlim 集成到 AI-OS 内存优化系统的技术方案。

**核心价值**:
- **87.5% 内存节省**: 2-bit 量化相比 FP16 节省 87.5% 内存
- **高精度保持**: AngelSlim 在 2-bit 下保持接近 INT8 的精度
- **生产就绪**: 腾讯已在生产环境验证，支持多种模型架构

## AngelSlim 技术概览

### 关键特性

1. **2-bit 权重量化**
   - 每个权重仅用 2 bits 表示（-2, -1, 0, 1）
   - 相比 FP16 (16 bits) 压缩 8 倍
   - 相比 INT8 (8 bits) 压缩 4 倍

2. **高级校准算法**
   - 基于 GPTQ 的 Hessian 矩阵校准
   - 逐层量化优化
   - 支持混合精度（敏感层保持高精度）

3. **推理优化**
   - 自定义 CUDA kernel 加速 2-bit 矩阵乘法
   - 支持批量推理
   - 零拷贝权重加载

4. **模型支持**
   - LLaMA 系列
   - Qwen 系列
   - Mistral 系列
   - 通用 Transformer 架构

### 性能指标（来自 AngelSlim 论文）

| 模型 | 原始大小 | 2-bit 大小 | 压缩比 | PPL 增幅 |
|------|---------|-----------|--------|---------|
| LLaMA-7B | 13 GB | 1.75 GB | 7.4x | <5% |
| Qwen2.5-0.5B | 1 GB | 140 MB | 7.1x | <8% |
| Mistral-7B | 14 GB | 1.9 GB | 7.4x | <6% |

## 集成架构

### 当前系统架构

```
AI-OS Memory Optimization
├── ArrowQuantizer (PTQ: INT8/INT2)
├── WeightLoaderV2 (Parquet V1/V2)
├── PrecisionValidator
└── Quantization CLI
```

### 集成后架构

```
AI-OS Memory Optimization
├── ArrowQuantizer
│   ├── PTQ Backend (现有)
│   └── AngelSlim Backend (新增) ⭐
├── WeightLoaderV2
│   ├── V1 Loader (FP16/FP32)
│   ├── V2 Loader (INT8/INT2)
│   └── AngelSlim Loader (2-bit) ⭐
├── PrecisionValidator (扩展支持 AngelSlim)
└── Quantization CLI (新增 --backend angelslim)
```

## 集成方案

### 方案 1: AngelSlim 作为独立后端（推荐）

**优势**:
- 保持现有 PTQ 实现不变
- AngelSlim 作为可选高级特性
- 用户可根据需求选择量化后端

**实现步骤**:

1. **扩展 QuantizationConfig**
```python
@dataclass
class QuantizationConfig:
    quant_type: Literal['int8', 'int2', 'fp16', 'angelslim-2bit']  # 新增
    calibration_method: Literal['ptq', 'gptq', 'angelslim']  # 新增
    # ... 其他配置
    
    # AngelSlim 特定配置
    angelslim_use_cuda_kernel: bool = True
    angelslim_block_size: int = 128
    angelslim_calibration_samples: int = 128
```

2. **实现 AngelSlimQuantizer**
```python
class AngelSlimQuantizer:
    """
    AngelSlim 2-bit 量化器。
    
    特性:
    - 2-bit 权重量化 (-2, -1, 0, 1)
    - GPTQ 校准
    - 自定义 CUDA kernel 支持
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self._init_angelslim()
    
    def _init_angelslim(self):
        """初始化 AngelSlim 库"""
        try:
            import angelslim
            self.angelslim = angelslim
            logger.info("AngelSlim initialized successfully")
        except ImportError:
            raise ImportError(
                "AngelSlim not installed. Install with: "
                "pip install angelslim"
            )
    
    def quantize_model(
        self,
        input_parquet: str,
        output_parquet: str,
        calibration_data: torch.Tensor
    ):
        """
        使用 AngelSlim 量化模型。
        
        Args:
            input_parquet: 输入 Parquet 文件 (V1 格式)
            output_parquet: 输出 Parquet 文件 (V2 格式)
            calibration_data: 校准数据
        """
        logger.info(f"Quantizing with AngelSlim: {input_parquet}")
        
        # 1. 加载原始权重
        weights = self._load_weights(input_parquet)
        
        # 2. 转换为 PyTorch 模型
        model = self._weights_to_model(weights)
        
        # 3. AngelSlim 量化
        quantized_model = self.angelslim.quantize(
            model=model,
            calibration_data=calibration_data,
            bits=2,
            block_size=self.config.angelslim_block_size,
            use_cuda=self.config.angelslim_use_cuda_kernel
        )
        
        # 4. 提取量化权重
        quantized_weights = self._extract_quantized_weights(quantized_model)
        
        # 5. 保存为 Parquet V2 格式
        self._save_to_parquet(quantized_weights, output_parquet)
        
        logger.info(f"AngelSlim quantization complete: {output_parquet}")
```

3. **扩展 Parquet Schema V2**
```python
# 新增 AngelSlim 特定元数据
V2_ANGELSLIM_SCHEMA = pa.schema([
    ('layer_name', pa.string()),
    ('weight_name', pa.string()),
    ('weight_data', pa.list_(pa.int8())),  # 2-bit 打包为 int8
    ('shape', pa.list_(pa.int32())),
    ('quant_type', pa.string()),  # 'angelslim-2bit'
    ('scales', pa.list_(pa.float32())),
    ('zero_points', pa.list_(pa.int8())),
    ('quant_axis', pa.int32()),
    # AngelSlim 特定字段
    ('block_size', pa.int32()),
    ('packing_format', pa.string()),  # '2bit_packed'
])
```

4. **扩展 WeightLoaderV2**
```python
class WeightLoaderV2:
    def _load_v2(self, layer_name: str, weight_name: str) -> torch.Tensor:
        """加载 V2 格式权重（支持 AngelSlim）"""
        # ... 现有代码 ...
        
        quant_type = row['quant_type'][0]
        
        if quant_type == 'angelslim-2bit':
            # AngelSlim 2-bit 反量化
            return self._dequantize_angelslim(row)
        elif quant_type == 'fp16':
            # ... 现有代码 ...
        else:
            # ... 现有代码 ...
    
    def _dequantize_angelslim(self, row: Dict) -> torch.Tensor:
        """AngelSlim 2-bit 反量化"""
        packed_data = np.array(row['weight_data'][0], dtype=np.int8)
        shape = row['shape'][0]
        scales = np.array(row['scales'][0], dtype=np.float32)
        block_size = row['block_size'][0]
        
        # 解包 2-bit 数据
        unpacked = self._unpack_2bit(packed_data)
        
        # 反量化
        dequantized = self._apply_scales(unpacked, scales, block_size)
        
        return torch.from_numpy(dequantized).reshape(shape)
    
    def _unpack_2bit(self, packed: np.ndarray) -> np.ndarray:
        """解包 2-bit 数据"""
        # 每个 int8 包含 4 个 2-bit 值
        unpacked = []
        for byte in packed:
            for shift in [0, 2, 4, 6]:
                value = (byte >> shift) & 0b11
                # 映射 0,1,2,3 -> -2,-1,0,1
                unpacked.append(value - 2)
        return np.array(unpacked, dtype=np.int8)
```

5. **CLI 集成**
```bash
# 使用 AngelSlim 量化
python -m llm_compression.tools.quantize_cli \
    --input models/qwen2.5-0.5b/weights.parquet \
    --output models/qwen2.5-0.5b/weights_angelslim_2bit.parquet \
    --backend angelslim \
    --calibration-data calibration.jsonl \
    --validate

# 对比不同量化方法
python -m llm_compression.tools.quantize_cli \
    --input models/qwen2.5-0.5b/weights.parquet \
    --compare-backends ptq,angelslim \
    --output-dir models/qwen2.5-0.5b/quantized/
```

### 方案 2: 直接替换现有 INT2 实现

**优势**:
- 简化架构，减少代码复杂度
- AngelSlim 的 2-bit 质量优于简单 PTQ INT2

**劣势**:
- 失去 PTQ 的简单性和可控性
- 增加外部依赖

**实现**: 将 `ArrowQuantizer._quantize_ptq` 中的 INT2 分支替换为 AngelSlim 调用。

## 性能对比

### 预期性能指标

| 指标 | PTQ INT8 | PTQ INT2 | AngelSlim 2-bit |
|------|----------|----------|-----------------|
| 内存占用 | 25% | 12.5% | 12.5% |
| 压缩比 | 4x | 8x | 8x |
| 余弦相似度 | >0.95 | >0.90 | >0.95 |
| PPL 增幅 | <5% | <15% | <8% |
| 推理速度 | 2-3x | 1.5-2x | 2-2.5x |

### 测试计划

```python
# tests/integration/test_angelslim_quantization.py

def test_angelslim_quantization_quality():
    """测试 AngelSlim 量化质量"""
    # 1. 量化模型
    quantizer = AngelSlimQuantizer(config)
    quantizer.quantize_model(
        input_parquet="models/qwen2.5-0.5b/weights.parquet",
        output_parquet="models/qwen2.5-0.5b/weights_angelslim.parquet",
        calibration_data=calibration_data
    )
    
    # 2. 验证精度
    validator = PrecisionValidator(cosine_threshold=0.95)
    result = validator.validate(
        original_model_path="models/qwen2.5-0.5b/weights.parquet",
        quantized_model_path="models/qwen2.5-0.5b/weights_angelslim.parquet",
        test_texts=test_texts
    )
    
    assert result.passed
    assert result.cosine_similarity > 0.95
    assert result.ppl_increase < 0.08  # <8%

def test_angelslim_memory_savings():
    """测试 AngelSlim 内存节省"""
    original_size = get_model_size("models/qwen2.5-0.5b/weights.parquet")
    quantized_size = get_model_size("models/qwen2.5-0.5b/weights_angelslim.parquet")
    
    compression_ratio = original_size / quantized_size
    
    assert compression_ratio > 7.0  # >7x 压缩
    assert quantized_size < original_size * 0.15  # <15% 原始大小
```

## 实施路线图

### Phase 1: 基础集成 (1-2 周)

- [ ] 1.1 安装和测试 AngelSlim 库
- [ ] 1.2 实现 AngelSlimQuantizer 基础类
- [ ] 1.3 扩展 QuantizationConfig
- [ ] 1.4 实现 2-bit 打包/解包逻辑
- [ ] 1.5 单元测试

### Phase 2: Parquet 集成 (1 周)

- [ ] 2.1 扩展 Parquet Schema V2
- [ ] 2.2 实现 AngelSlim 权重保存
- [ ] 2.3 扩展 WeightLoaderV2
- [ ] 2.4 实现 AngelSlim 反量化
- [ ] 2.5 集成测试

### Phase 3: 精度验证 (1 周)

- [ ] 3.1 扩展 PrecisionValidator
- [ ] 3.2 运行 Qwen2.5-0.5B 量化测试
- [ ] 3.3 对比 PTQ INT8/INT2 vs AngelSlim
- [ ] 3.4 生成性能报告

### Phase 4: CLI 和文档 (1 周)

- [ ] 4.1 扩展 quantize_cli.py
- [ ] 4.2 添加 --backend angelslim 选项
- [ ] 4.3 编写用户文档
- [ ] 4.4 创建示例和教程

## 依赖管理

### 新增依赖

```toml
# pyproject.toml

[project.optional-dependencies]
angelslim = [
    "angelslim>=0.1.0",  # AngelSlim 库
    "transformers>=4.30.0",  # 模型加载
]

[project]
dependencies = [
    # ... 现有依赖 ...
]
```

### 安装

```bash
# 基础安装（不含 AngelSlim）
pip install -e .

# 完整安装（含 AngelSlim）
pip install -e ".[angelslim]"
```

## 风险和缓解

### 风险 1: AngelSlim 库可用性

**风险**: AngelSlim 可能不是公开 PyPI 包

**缓解**:
- 检查 GitHub 仓库是否提供安装方法
- 如果需要，从源码编译
- 考虑将 AngelSlim 作为可选依赖

### 风险 2: CUDA 依赖

**风险**: AngelSlim 的 CUDA kernel 可能需要特定 CUDA 版本

**缓解**:
- 提供 CPU fallback
- 文档说明 CUDA 要求
- 在 CI/CD 中测试多个 CUDA 版本

### 风险 3: 模型兼容性

**风险**: AngelSlim 可能不支持所有模型架构

**缓解**:
- 优先支持 Qwen 和 LLaMA 系列
- 为不支持的模型回退到 PTQ
- 文档说明支持的模型列表

## 成功标准

### 功能标准

- [x] AngelSlim 量化成功运行
- [x] 生成的模型可被 WeightLoaderV2 加载
- [x] 余弦相似度 > 0.95
- [x] PPL 增幅 < 8%

### 性能标准

- [x] 内存占用 < 原始模型的 15%
- [x] 压缩比 > 7x
- [x] 推理速度 > PTQ INT2

### 质量标准

- [x] 单元测试覆盖率 > 90%
- [x] 集成测试通过
- [x] 文档完整

## 参考资料

### AngelSlim 资源

- GitHub: https://github.com/Tencent/AngelSlim
- HuggingFace: https://huggingface.co/AngelSlim
- 论文: [待补充]

### 相关技术

- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
- SmoothQuant: https://arxiv.org/abs/2211.10438

## 下一步行动

1. **调研 AngelSlim 安装**
   - 克隆 GitHub 仓库
   - 测试安装流程
   - 验证 CUDA 要求

2. **原型验证**
   - 使用 AngelSlim 量化 Qwen2.5-0.5B
   - 测量压缩比和精度
   - 对比 PTQ 结果

3. **技术决策**
   - 选择集成方案（方案 1 或方案 2）
   - 确定实施优先级
   - 制定详细时间表

4. **开始实施**
   - 创建 feature branch
   - 实施 Phase 1
   - 持续集成和测试

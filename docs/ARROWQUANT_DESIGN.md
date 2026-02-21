# ArrowQuant-2Bit: 系统设计文档

> **版本**: v1.0  |  **日期**: 2026-02-20  |  **状态**: Draft  
> **前置文档**: `ARROWQUANT_REQUIREMENTS.md`

---

## 1. 设计总纲 (Design Overview)

ArrowQuant 是 AI-OS 的原生权重量化子系统，目标是在 **不引入外部格式**（GGUF/GPTQ-Marlin/AWQ）的前提下，利用 ArrowEngine 既有的 Parquet 存储 + 零拷贝 + 惰性加载架构，实现 **Weight-Only INT2/INT4/INT8 量化**。

### 核心理念

```
 ArrowQuant = Per-Group Quantization + Arrow-Native Storage + Lazy Dequantization
```

所有量化元信息（scale、zero_point、group_size）与权重数据**存储在同一 Parquet 行中**，做到"一行一层、自包含反量化"。

---

## 2. 数学基础 (Mathematical Foundation)

### 2.1 Asymmetric Per-Group Quantization

对一组大小为 `G` 的连续权重 $W_{[i:i+G]}$：

$$
\text{scale} = \frac{\max(W) - \min(W)}{2^b - 1}
$$

$$
\text{zero\_point} = \text{round}\left(-\frac{\min(W)}{\text{scale}}\right)
$$

$$
W_q = \text{clamp}\left(\text{round}\left(\frac{W}{\text{scale}}\right) + \text{zero\_point},\; 0,\; 2^b-1\right)
$$

**反量化**：

$$
\hat{W} = (W_q - \text{zero\_point}) \times \text{scale}
$$

其中 $b = 2$（INT2 取值 {0,1,2,3}），$G = 128$（默认组大小）。

### 2.2 位打包 (Bit Packing)

4 个 INT2 值打包为 1 个 `uint8`：

```
byte = (val_0) | (val_1 << 2) | (val_2 << 4) | (val_3 << 6)
```

→ 原始 FP16 权重：每参数 16 bit  
→ 打包后 INT2 权重：每参数 2 bit  
→ 理论压缩比：**8x**（含 scale/zp 开销后约 **6–7x**）

### 2.3 Mixed-Precision 策略

以下参数层**不做量化**，保留 FP16：

| 层类型 | 原因 |
|--------|------|
| `embeddings.word_embeddings.weight` | 嵌入表对量化极度敏感，误差直接传播到所有后续层 |
| `embeddings.position_embeddings.weight` | 位置编码精度直接影响注意力模式 |
| `*.LayerNorm.weight` / `*.LayerNorm.bias` | 归一化参数体积极小(< 0.1%)，量化收益可忽略 |
| `lm_head.weight` | 输出层误差直接反映到 token logits |
| `model.embed_tokens.weight` | Decoder 嵌入表（同理） |
| `model.norm.weight` | 最终 RMSNorm |

---

## 3. 存储格式设计 (Storage Format)

### 3.1 Parquet Schema V2

```python
WEIGHT_SCHEMA_V2 = pa.schema([
    # ---- 原有字段（V1 兼容） ----
    ("layer_name",  pa.string()),        # "encoder.layer.0.attention.self.query.weight"
    ("shape",       pa.list_(pa.int32())), # [384, 384]
    ("dtype",       pa.string()),        # "torch.float16" 或 "torch.uint8" (packed)
    ("data",        pa.binary()),        # 原始 FP16 bytes 或 packed INT2 bytes
    ("num_params",  pa.int64()),         # 147456
    # ---- 量化扩展字段 ----
    ("quant_type",  pa.string()),        # "none" | "int2_asym_group" | "int4_asym_group" | "int8_sym"
    ("group_size",  pa.int32()),         # 128 (0 表示 per-tensor)
    ("scales",      pa.binary()),        # FP32 scale 数组 (len = num_params / group_size)
    ("zero_points", pa.binary()),        # FP32 zero_point 数组 (同上)
])
```

### 3.2 兼容性保证

| 场景 | 行为 |
|------|------|
| V1 Parquet（无 quant_type 列） | `WeightLoader` 回退至 V1 逻辑，完全兼容 |
| V2 Parquet，`quant_type="none"` | 按 FP16 正常加载 |
| V2 Parquet，`quant_type="int2_asym_group"` | 触发反量化路径 |

检测逻辑：

```python
# WeightLoader._load_table()
if "quant_type" in self._table.column_names:
    self._schema_version = 2
else:
    self._schema_version = 1  # Legacy
```

### 3.3 Metadata 扩展

`metadata.json` 新增字段：

```json
{
  "quantization": {
    "method": "arrowquant_v1",
    "bit_width": 2,
    "group_size": 128,
    "calibration": "minmax",
    "skip_layers": ["embeddings.*", "*.LayerNorm.*", "lm_head.*"],
    "original_dtype": "torch.float16",
    "quantized_layers": 60,
    "total_layers": 66,
    "estimated_compression_ratio": 6.8
  }
}
```

---

## 4. 模块设计 (Module Design)

### 4.1 系统架构图

```
                        ┌─────────────────────────────────────────────┐
                        │              用户 API（不变）                 │
                        │  ArrowEngine.encode() / .generate()         │
                        └───────────────┬─────────────────────────────┘
                                        │
                        ┌───────────────▼─────────────────────────────┐
                        │           InferenceCore.forward()            │
                        │        (FP32 张量，完全不变)                  │
                        └───────────────┬─────────────────────────────┘
                                        │  weights[layer_name]
                        ┌───────────────▼─────────────────────────────┐
                        │       WeightLoader._row_to_tensor()          │
                        │  ┌────────────────────────────────────────┐  │
                        │  │  if quant_type == "none":              │  │
                        │  │    → 原 V1 零拷贝路径                  │  │
                        │  │  elif quant_type == "int2_asym_group": │  │
                        │  │    → _dequantize_int2() 反量化路径      │  │
                        │  └────────────────────────────────────────┘  │
                        └───────────────┬─────────────────────────────┘
                                        │  as_buffer() 零拷贝
                        ┌───────────────▼─────────────────────────────┐
                        │           Parquet / Arrow Table              │
                        │   mmap → data(uint8) + scales(fp32)         │
                        └─────────────────────────────────────────────┘

   ==================== 离线工具链 ====================

   ┌────────────────────────────────────────────────────────┐
   │  scripts/quantize_arrowquant.py                        │
   │  ┌──────────┐   ┌──────────┐   ┌───────────────────┐  │
   │  │ 加载 HF   │ → │ Per-Group │ → │  写入 V2 Parquet  │  │
   │  │ 模型权重   │   │ 量化+打包  │   │  (data+scales)   │  │
   │  └──────────┘   └──────────┘   └───────────────────┘  │
   └────────────────────────────────────────────────────────┘
```

### 4.2 新增模块清单

| 模块 | 路径 | 职责 |
|------|------|------|
| **ArrowQuantizer** | `llm_compression/inference/quantizer.py` | 核心量化逻辑（量化、打包、反量化、校准） |
| **quantize_arrowquant.py** | `scripts/quantize_arrowquant.py` | CLI 工具，调用 ArrowQuantizer 进行离线量化 |
| **WeightLoader V2** | `weight_loader.py`（扩展） | 新增反量化分支和 Schema V2 检测 |
| **ModelConverter V2** | `model_converter.py`（扩展） | 新增 `WEIGHT_SCHEMA_V2` 和 quantized 转换管线 |

### 4.3 ArrowQuantizer 核心类设计

```python
class ArrowQuantizer:
    """
    核心量化器：负责 FP16/FP32 → INT2/INT4/INT8 转换。
    
    支持：
    - Per-Group Asymmetric Quantization (INT2/INT4)
    - Per-Tensor Symmetric Quantization (INT8)
    - GPTQ 校准增强 (Phase 2)
    - Mixed-Precision 自动跳过敏感层
    """
    
    def __init__(
        self,
        bit_width: int = 2,            # 2, 4, 或 8
        group_size: int = 128,          # 量化组大小
        symmetric: bool = False,        # 对称/非对称
        skip_patterns: List[str] = None # 跳过的层名模式
    ):
        ...
    
    def quantize_tensor(
        self, 
        tensor: torch.Tensor,
        layer_name: str
    ) -> QuantizedTensor:
        """
        量化单个张量。
        
        Returns:
            QuantizedTensor(
                packed_data: bytes,     # uint8 打包数据
                scales: np.ndarray,     # FP32 scale 数组
                zero_points: np.ndarray,# FP32 zero_point 数组
                shape: List[int],       # 原始形状
                bit_width: int,
                group_size: int,
                is_skipped: bool        # 是否跳过量化
            )
        """
        ...
    
    def dequantize_tensor(
        self,
        packed_data: bytes,
        scales: np.ndarray,
        zero_points: np.ndarray,
        shape: List[int],
        bit_width: int,
        group_size: int
    ) -> torch.Tensor:
        """反量化为 FP32 张量。"""
        ...
    
    @staticmethod
    def pack_int2(values: np.ndarray) -> np.ndarray:
        """将 INT2 值（0-3）打包为 uint8。每 4 个值 → 1 byte。"""
        ...
    
    @staticmethod
    def unpack_int2(packed: np.ndarray, num_elements: int) -> np.ndarray:
        """从 uint8 解包 INT2 值。"""
        ...
```

### 4.4 WeightLoader 扩展

在 `_row_to_tensor()` 中增加量化分支：

```python
def _row_to_tensor(self, table, row_idx):
    # 检测量化类型
    quant_type = "none"
    if self._schema_version == 2:
        quant_type = table['quant_type'][row_idx].as_py() or "none"
    
    if quant_type == "none":
        # 原有的零拷贝 V1 路径（完全不变）
        return self._row_to_tensor_v1(table, row_idx)
    
    elif quant_type.startswith("int"):
        # 反量化路径
        data_buffer = table['data'][row_idx].as_buffer()       # 零拷贝
        scales_buffer = table['scales'][row_idx].as_buffer()   # 零拷贝
        zp_buffer = table['zero_points'][row_idx].as_buffer()  # 零拷贝
        shape = table['shape'][row_idx].as_py()
        group_size = table['group_size'][row_idx].as_py()
        
        # 反量化
        tensor = self._quantizer.dequantize_tensor(
            packed_data=data_buffer,
            scales=np.frombuffer(scales_buffer, dtype=np.float32),
            zero_points=np.frombuffer(zp_buffer, dtype=np.float32),
            shape=shape,
            bit_width=int(quant_type.replace("int","").split("_")[0]),
            group_size=group_size
        )
        return tensor
```

### 4.5 LazyWeightDict + 反量化的联动

```
                  LazyWeightDict["encoder.layer.3.attention.self.query.weight"]
                           │
                           ▼
                  WeightLoader.get_layer("encoder.layer.3...")
                           │
                           ▼  O(1) 查找 (cached _layer_name_map)
                  _row_to_tensor(table, row_idx=42)
                           │
                       ┌───┴───┐
                quant? │  No   │  Yes (int2_asym_group)
                       ▼       ▼
                 V1 零拷贝    反量化路径
                 (FP16)       (uint8 → unpack → ×scale)
                       │       │
                       ▼       ▼
                   torch.Tensor (FP32)
                           │
                           ▼
                  InferenceCore._load_param()
```

**优势**：惰性加载 + 惰性反量化 = **只反量化当前推理需要的层**。对于 24 层 Qwen-0.5B，不会一次性反量化全部 291 个张量。

---

## 5. 离线量化工具设计 (Offline Quantization Tool)

### 5.1 使用方式

```bash
# 基础 INT2 量化（MinMax PTQ）
python scripts/quantize_arrowquant.py \
    --model models/minilm \
    --output models/minilm-q2 \
    --bit-width 2 \
    --group-size 128

# INT4 量化
python scripts/quantize_arrowquant.py \
    --model models/qwen2.5-0.5b-arrow \
    --output models/qwen2.5-0.5b-q4 \
    --bit-width 4

# GPTQ 校准增强（Phase 2）
python scripts/quantize_arrowquant.py \
    --model models/qwen2.5-0.5b-arrow \
    --output models/qwen2.5-0.5b-q2-gptq \
    --bit-width 2 \
    --calibration gptq \
    --calibration-data data/calibration_qa.jsonl \
    --num-samples 128
```

### 5.2 工具内部流程

```
1. 加载源模型权重（FP16 Parquet 或 HuggingFace）
2. 遍历每一层：
   a. 检查是否在 skip_patterns 内 → 跳过，保留 FP16
   b. 将 FP16 张量展平
   c. 按 group_size 分组
   d. 对每组计算 scale, zero_point (MinMax 或 GPTQ)
   e. 量化：round + clamp → int2/int4 值
   f. 位打包：4 个 int2 → 1 个 uint8
3. 将结果写入 V2 Schema Parquet
4. 复制 tokenizer/ 和更新 metadata.json
5. 运行精度验证（余弦相似度 / PPL）
```

---

## 6. GPTQ 校准设计 (Phase 2)

### 6.1 算法概述

GPTQ 是一种基于二阶信息（Hessian 近似）的逐列量化算法，核心思想：

> 在量化第 $j$ 列时，利用 Hessian 逆矩阵 $H^{-1}$ 将该列的量化误差**最优地分配到尚未量化的列**上，使全局重建误差最小化。

### 6.2 校准流程

```
1. 用 128 条校准数据运行前向传播
2. 收集每层的输入激活 X
3. 计算 Hessian: H = 2 * X^T * X
4. 对每层逐列量化：
   a. 选择当前列 w_j
   b. 量化: q_j = quantize(w_j)
   c. 计算误差: δ = (w_j - q_j) / H_jj
   d. 更新剩余列: W[:, j+1:] -= δ * H[j, j+1:]
5. 打包量化后的权重
```

### 6.3 实施要点

- 每层独立校准（完全按层并行，适配 `LazyWeightDict`）
- Hessian 近似使用 Cholesky 分解
- 校准数据不需要标签，仅需输入文本
- 内存开销：每层约 `hidden_size^2 * 4` bytes（MiniLM: ~0.5MB / Qwen: ~3MB）

---

## 7. 实施路径 (Implementation Roadmap)

### Phase 2a: 基础 PTQ (MinMax) — 预计 2–3 天

| 步骤 | 任务 | 产出 |
|------|------|------|
| S1 | 实现 `ArrowQuantizer` 核心类（pack/unpack/quantize/dequantize） | `quantizer.py` |
| S2 | 扩展 `WeightLoader` 支持 Schema V2 + 反量化路径 | `weight_loader.py` 修改 |
| S3 | 扩展 `ModelConverter.WEIGHT_SCHEMA` 为 V2 | `model_converter.py` 修改 |
| S4 | 编写离线量化 CLI | `scripts/quantize_arrowquant.py` |
| S5 | 量化 MiniLM-L6 → INT2 并验证余弦相似度 | 验证报告 |
| S6 | 量化 Qwen2.5-0.5B → INT2 并验证 PPL | 验证报告 |

### Phase 2b: GPTQ 校准 — 预计 4–5 天

| 步骤 | 任务 | 产出 |
|------|------|------|
| S7 | 实现 GPTQ Hessian 校准核心逻辑 | `quantizer.py` 扩展 |
| S8 | 准备校准数据集（Wikipedia句子 + QA对） | `data/calibration/` |
| S9 | GPTQ 校准 MiniLM → INT2 并验证 | 相似度 ≥ 0.98 |
| S10 | GPTQ 校准 Qwen-0.5B → INT2 并验证 | PPL 增幅 ≤ 5% |

### Phase 2c: 集成与文档 — 预计 1–2 天

| 步骤 | 任务 | 产出 |
|------|------|------|
| S11 | `ArrowEngine` 自动检测量化模型 | 端到端集成测试 |
| S12 | 更新 `OPTIMIZATION_TASKS.md` 和基准测试 | 文档更新 |
| S13 | 合并至主分支 | PR & Merge |

---

## 8. 风险与缓解 (Risks & Mitigations)

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| INT2 PTQ 精度不达标 (< 0.95) | 中 | 高 | 降级为 INT4 → 仍有 4x 压缩；或启用 GPTQ 校准 |
| 反量化延迟超过 5% | 低 | 中 | 使用 NumPy 向量化操作替代循环；考虑缓存反量化结果 |
| Parquet Schema 变更导致回归 | 低 | 高 | V1 路径完全不动；新增字段为 optional |
| GPTQ Hessian 计算 OOM | 中 | 中 | 分块处理（blocksize=128）；逐列量化而非矩阵量化 |

---

## 9. 测试策略 (Testing Strategy)

### 9.1 单元测试

```python
# tests/unit/test_quantizer.py
class TestArrowQuantizer:
    def test_pack_unpack_int2_roundtrip()       # 打包解包一致性
    def test_quantize_dequantize_accuracy()     # 量化-反量化误差 < ε
    def test_skip_sensitive_layers()            # Embedding/LN 不被量化
    def test_group_size_boundary()              # 非 group_size 整除的张量
    
# tests/unit/test_weight_loader_v2.py
class TestWeightLoaderV2:
    def test_v1_parquet_backward_compat()       # V1 格式仍可加载
    def test_v2_quantized_load()                # V2 量化格式加载
    def test_lazy_dequantize()                  # LazyWeightDict + 反量化
```

### 9.2 端到端测试

```python
# tests/e2e/test_arrowquant_pipeline.py
def test_minilm_int2_encode_similarity()        # MiniLM INT2 → encode → cosine ≥ 0.95
def test_qwen_int2_generate_coherent()          # Qwen INT2 → generate → 可读文本
def test_quantize_then_load_roundtrip()         # 量化 → 保存 → 加载 → 验证
```

---

## 10. 与 AngelSlim 的技术对比 (Technical Comparison)

| 维度 | AngelSlim HY-1.8B-2Bit | ArrowQuant-2Bit |
|------|------------------------|-----------------|
| 量化方法 | QAT（训练时量化） | PTQ + GPTQ 校准 |
| 存储格式 | SafeTensors / GGUF | **Parquet (Arrow-Native)** ← 差异化 |
| 反量化方式 | 硬件内核 (SME2/KleidiAI) | PyTorch + NumPy 向量化 |
| 加载机制 | 全量加载 | **惰性反量化** ← 差异化 |
| 零拷贝 | ❌ 需要反序列化 | ✅ `as_buffer()` + `mmap` |
| 精度损失 | ~3.97% (QAT) | ~8-15% (PTQ) / ~4-6% (GPTQ) |
| 可扩展性 | 仅 ARM SME2 | 跨平台 (CPU/CUDA/MPS) |
| Meta 存储 | 独立 config 文件 | **同行存储** (scale + data 同一 Parquet 行) |

**我们的核心差异化**：
1. **存储与推理完全统一** — 量化元数据嵌入 Parquet 行，无需额外 config
2. **惰性反量化** — LazyWeightDict 天然兼容，只反量化当前推理层
3. **零拷贝链路** — 从磁盘到反量化，中间无内存拷贝

---
*最后更新: 2026-02-20*

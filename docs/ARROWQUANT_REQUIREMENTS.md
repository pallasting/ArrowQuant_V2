# ArrowQuant-2Bit: 原生 Arrow 极限量化方案需求文档

> **版本**: v1.0  |  **日期**: 2026-02-20  |  **状态**: Draft  
> **依据**: Tencent AngelSlim HY-1.8B-2Bit 技术报告 + AI-OS ArrowEngine 架构

---

## 1. 背景与动机 (Background & Motivation)

### 1.1 当前痛点

AI-OS 的 ArrowEngine 推理架构已完成 Phase 1 计算加速，但模型体积仍然是端侧部署的核心瓶颈：

| 模型 | 当前格式 | FP16 体积 | 推理时内存 |
|------|----------|-----------|-----------|
| MiniLM-L6 | Parquet (FP16) | 43.5 MB | ~90 MB (FP32 upcast) |
| Qwen2.5-0.5B | Parquet (FP16) | 1.2 GB | ~2.4 GB (FP32 upcast) |

对于端侧（手机、嵌入式设备）场景，2.4GB 的内存占用不可接受。

### 1.2 AngelSlim 技术报告的核心启示

腾讯 HY-1.8B-2Bit 证明了 **2-bit QAT** 可以在仅丢失 3.97% 精度的前提下，将模型体积压缩到 FP16 的 **1/8**。其关键技术点：

- **Per-Group Quantization**：每 128 个权重共享一个 scale，比 per-tensor 精度高
- **QAT（量化感知训练）**：在训练时模拟量化误差，但**需要训练数据和算力**
- **Dual-CoT 策略**：保留模型完整推理能力
- **SME2/KleidiAI 端侧内核**：ARM 专用加速，但仅限特定硬件

### 1.3 我们的差异化机会

ArrowEngine 拥有**其他框架不具备的原生优势**：

1. **Parquet 列式存储** → 可以在同一行中存储 `packed_data`（uint8）和 `scale`（float32），无需引入新格式（GGUF/SafeTensors）
2. **零拷贝内存映射** → 量化后的权重文件更小，`mmap` 占用显著降低
3. **LazyWeightDict** → 逐层按需反量化，峰值内存只占一层的 FP32 + 全模型 INT2
4. **已有 `force_float32` 路径** → 可复用为"反量化 + upcast"的统一入口

基于此，我们提出 **ArrowQuant**：一套原生 Arrow 格式的极限量化方案。

---

## 2. 需求定义 (Requirements)

### 2.1 功能性需求 (Functional Requirements)

#### REQ-AQ-1: 离线量化工具 (Offline Quantizer)

- **描述**：提供 `scripts/quantize_2bit.py` 工具，将 HuggingFace 或已有 Arrow 格式的 FP16/FP32 模型转换为 2-bit 紧凑格式，存储为新的 Parquet 文件。
- **输入**：HuggingFace 模型路径 或 已有 `weights.parquet` 路径
- **输出**：新的 `weights_q2.parquet`（量化权重）+ 更新的 `metadata.json`
- **量化粒度**：
  - **per-group**：每 `group_size`（默认 128）个权重共享一个 `scale` 和 `zero_point`
  - **Mixed-Precision**：Embedding 层和最终 LayerNorm/LM-head 保留 FP16（这些层对量化极度敏感）
- **验收标准**：
  - 对 MiniLM-L6：量化后余弦相似度 ≥ 0.95
  - 对 Qwen2.5-0.5B：PPL 增加不超过 15%
  - 转换时间（MiniLM）< 30 秒

#### REQ-AQ-2: 运行时反量化引擎 (Runtime Dequantizer)

- **描述**：`WeightLoader` 原生支持 `quant_mode` 参数，自动检测量化 Parquet 格式并在加载时逐层反量化为 FP32。
- **反量化公式**：`W_fp32 = packed_int2.dequantize() * scale + zero_point`
- **零拷贝要求**：反量化前的 `uint8` buffer 必须通过 Arrow 的 `as_buffer()` 零拷贝获取，不允许 `.as_py()` 中间转换
- **验收标准**：
  - INT2 模型加载时间 ≤ FP16 模型加载时间的 50%（因文件更小）
  - 反量化后的权重直接进入 `InferenceCore`，不改变任何现有 forward() 逻辑

#### REQ-AQ-3: GPTQ 校准增强 (GPTQ Calibration - Phase 2)

- **描述**：实现基于 Hessian 信息的逐列量化校准（GPTQ 算法），使用少量校准数据（128–512 条）显著降低 PTQ 量化误差。
- **校准数据**：
  - 对 Encoder（MiniLM）：使用 Wikipedia / STS-B 句子
  - 对 Decoder（Qwen）：使用通用 QA 对
- **验收标准**：
  - 校准后 MiniLM 余弦相似度 ≥ 0.98
  - Qwen2.5-0.5B PPL 增加不超过 5%

#### REQ-AQ-4: Parquet Schema 扩展

- **描述**：扩展现有的 `WEIGHT_SCHEMA`，兼容量化和非量化格式。
- **新 Schema**：

```python
WEIGHT_SCHEMA_V2 = pa.schema([
    ("layer_name",  pa.string()),
    ("shape",       pa.list_(pa.int32())),
    ("dtype",       pa.string()),       # "torch.int2_packed" 或 "torch.float16"
    ("data",        pa.binary()),       # 原始数据或 packed uint8
    ("num_params",  pa.int64()),
    # ---- 量化扩展字段 ----
    ("quant_type",  pa.string()),       # "none", "int2_group", "int4_group", "int8"
    ("group_size",  pa.int32()),        # 量化组大小
    ("scales",      pa.binary()),       # FP32 scale 数组
    ("zero_points", pa.binary()),       # FP32 zero_point 数组
])
```

- **向下兼容**：当 `quant_type` 为空或 `"none"` 时，行为与 V1 Schema 完全一致
- **验收标准**：现有所有 MiniLM / Qwen / CLIP 模型的 FP16 权重在新 Schema 下仍可正常加载

### 2.2 非功能性需求

#### NFR-AQ-1: 内存效率

| 模型 | FP16 内存 | INT2 内存 (目标) | 压缩比 |
|------|-----------|-----------------|--------|
| MiniLM-L6 | 90 MB | **~15 MB** | 6x |
| Qwen2.5-0.5B | 2,400 MB | **~350 MB** | 7x |

> INT2 实际占用比 8x 略多，因为 scale/zero_point 元数据也占空间。

#### NFR-AQ-2: 推理延迟

- 反量化开销（per-batch）< 总推理时间的 5%
- 即：如果 FP32 推理延迟为 5ms，反量化不得超过 0.25ms

#### NFR-AQ-3: 向下兼容

- `ArrowEngine(model_path)` 自动检测模型是 FP16 还是 INT2 格式
- 不需要额外的 `--quantized` 标志
- 现有所有 API（`encode()`, `generate()`, `encode_with_lora()`）行为不变

#### NFR-AQ-4: 可扩展性

- Schema 设计必须同时支持 INT2、INT4、INT8 量化方式
- `quant_type` 字段允许未来扩展更多量化格式

---

## 3. 排除范围 (Out of Scope)

- **QAT（量化感知训练）**：需要训练数据和 GPU 资源，不在本期范围内
- **自定义 CUDA/ARM 内核**：AngelSlim 的 SME2 内核依赖特定硬件，我们在 PyTorch 层面实现
- **GGUF 格式兼容**：目标是 Arrow 原生，不做 GGUF 互操作
- **Activation 量化（W2A8 等）**：仅做 Weight-only 量化

---

## 4. 验收标准汇总 (Acceptance Criteria)

| 编号 | 指标 | 目标 |
|------|------|------|
| AC-1 | MiniLM-L6 余弦相似度 (Baseline vs INT2) | ≥ 0.95 |
| AC-2 | Qwen-0.5B PPL 增幅 | ≤ 15% (PTQ) / ≤ 5% (GPTQ) |
| AC-3 | MiniLM INT2 模型文件大小 | ≤ 12 MB |
| AC-4 | Qwen-0.5B INT2 模型文件大小 | ≤ 200 MB |
| AC-5 | 现有 FP16 模型在新 Schema 下加载 ✓ | 100% 通过 |
| AC-6 | 反量化延迟占比 | < 5% |
| AC-7 | `ArrowEngine` API 无变更 | 所有现有调用方式不变 |

---

## 5. 下一步 (Next Steps)

→ 基于本需求文档，撰写 `ARROWQUANT_DESIGN.md` 系统设计文档  
→ 将 T2.1 任务拆解为可执行的子任务，更新 `OPTIMIZATION_TASKS.md`

---
*最后更新: 2026-02-20*

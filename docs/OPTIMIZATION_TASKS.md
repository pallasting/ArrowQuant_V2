# AI-OS: 优化阶段任务跟踪 (Optimization Task Tracking)

## Phase 1: 计算层极限加速 (Compute Latency Optimization) - [x] 60%
*   **T1.1: 基础设施调研与基准测试更新**
    *   [x] 编写 `scripts/bench_baseline_advanced.py`，细化算子级延迟。
*   **T1.2: 集成 CPU 加速后端**
    *   [x] 在 `inference/intel_opt.py` 中支持 `torch.compile`。
    *   [x] 测试 IPEX (Intel Extension for PyTorch) 的加速效果。
    *   [x] 实现 True Zero-Copy 和 Lazy Loading 机制。
*   **T1.3: GPU 支持初步实现**
    *   [x] 实现 `inference/cuda_backend.py`。
    *   [x] 为 `ArrowEngine` 增加 `device` 切换逻辑与 VRAM Fallback。
*   **T1.4: 精度校验 (Precision Validation)**
    *   [ ] 编写 `scripts/validate_precision.py` 测量基准模型与加速模型间的余弦相似度漂移。
    *   [ ] 确保在 `torch.compile` 和 FP16 下相似度 > 0.99。

## Phase 2: ArrowQuant 极限量化 (ArrowQuant Quantization) - [/] 30%

> 设计文档: [`ARROWQUANT_REQUIREMENTS.md`](./ARROWQUANT_REQUIREMENTS.md) | [`ARROWQUANT_DESIGN.md`](./ARROWQUANT_DESIGN.md)

### Phase 2a: 基础 PTQ (MinMax) 与存储架构 - [x]
*   **T2.1: ArrowQuantizer 核心类实现**
    *   [x] 实现 `quantizer.py`：pack/unpack INT2、per-group 量化/反量化 (基础架构已就绪)
    *   [x] 实现 Mixed-Precision 敏感层跳过逻辑
*   **T2.2: Parquet Schema V2 扩展**
    *   [x] 扩展 `ModelConverter.WEIGHT_SCHEMA` 为 V2（新增 quant_type, scales, zero_points 列）
    *   [x] 扩展 `WeightLoader._row_to_tensor()` 支持反量化分支
    *   [x] 保证 V1 Schema 向下兼容
*   **T2.3: 离线量化工具**
    *   [x] 编写 `scripts/quantize_cli.py` CLI 工具
    *   [ ] 验证基础 INT2 PTQ 精度 (目前精度损失较大)

### Phase 2b: 定制化 2-Bit GPTQ (基于 AngelSlim 原理) - [ ]
由于腾讯并未开源2-bit量化工具链，我们需要手搓一个针对扩散模型和AR模型的高精度 2-Bit 量化器：
*   **T2.4: Asymmetric GPTQ 核心算法实现**
    *   [ ] 在 `arrow_quantizer.py` 中实现非对称 2-bit (Asymmetric) GPTQ 算法
    *   [ ] 实现基于 Hessian 矩阵的逆序权重更新机制 (Hessian-based compensation)
    *   [ ] 实现 Block-wise 或 Channel-wise 的细粒度 Scale/Zero-point 计算
*   **T2.5: 量化校准与精度验证**
    *   [ ] 编写校准数据集加载逻辑 (Calibration Data Loader)
    *   [ ] 验证 2-Bit GPTQ MiniLM 余弦相似度 ≥ 0.95 (大幅超越当前 PTQ 方案)
    *   [ ] 为未来的扩散模型 (MDLM) 的 2-Bit 量化做好框架准备

### Phase 2c: 集成与验收
*   **T2.6: ArrowEngine 端到端集成**
    *   [ ] `ArrowEngine` 自动检测量化 Parquet 格式
    *   [ ] LazyWeightDict + 反量化 联动测试
    *   [ ] 全量回归测试（现有 FP16 模型功能不变）
*   **T2.7: 基准测试与文档**
    *   [ ] 内存占用对比（FP16 vs INT2）
    *   [ ] 推理延迟对比（含反量化开销）
    *   [ ] 合并至主分支

## Phase 3: 统一扩散生成架构 (Unified Diffusion Architecture) - [ ] 0%

> 设计文档: [`UNIFIED_DIFFUSION_REQUIREMENTS.md`](./UNIFIED_DIFFUSION_REQUIREMENTS.md) | [`UNIFIED_DIFFUSION_DESIGN.md`](./UNIFIED_DIFFUSION_DESIGN.md)

### Phase 3a: 离散扩散文本 PoC
*   **T3.1: 扩散推理框架实现**
    *   [ ] 实现 `inference/diffusion_core.py`：DiffusionCore + NoiseScheduler + DiscreteSampler
    *   [ ] 实现 `inference/continuous_sampler.py`：ContinuousSampler (Langevin/ODE)
*   **T3.2: 文本投影头 + 模型转换**
    *   [ ] 实现 `TextProjectionHead`（encode: tokens→hidden, decode: hidden→logits）
    *   [ ] 转换开源 MDLM/SEDD 权重 → Parquet V2 + ArrowQuant
*   **T3.3: ArrowEngine.diffuse() 集成**
    *   [ ] 实现 `ArrowEngine.diffuse(modality="text")` API
    *   [ ] Infilling 质量验证 + 延迟基准测试

### Phase 3b: 统一 Score Network + 记忆条件
*   **T3.4: UnifiedScoreNetwork 实现**
    *   [ ] 实现共享 TransformerStack + ModalityEmbedding + SinusoidalTimeEmbedding
    *   [ ] 实现模态投影 Head 注册机制 (ModuleDict)
*   **T3.5: MemoryConditioner 记忆引导**
    *   [ ] 实现 ArrowStorage → 条件向量投射
    *   [ ] Cross-Attention 条件注入到 Score Network
*   **T3.6: 渐进式自进化层 (5-Level Evolution)**
    *   [ ] 实现 `EvolutionRouter`：支持 L0 Score组合、L1 ControlNet、L2 LoRA
    *   [ ] 实现 `SelectiveFinetuner` (L3)：基于不确定性的骨架层级局部解冻
    *   [ ] 集成到 `CognitiveLoop._trigger_evolution()` 替代启发式阈值
*   **T3.7: 能量模型校验融合 (EBM)**
    *   [ ] 实现轻量级约束模型 `EnergyModelValidator`
    *   [ ] 在生成末端注入物理/逻辑约束梯度 ($\nabla E_{constraint}$)
*   **T3.7: 图像扩散集成**
    *   [ ] 实现 `ImageProjectionHead` + PatchEmbed
    *   [ ] 集成轻量 VAE Encoder/Decoder
    *   [ ] 转换 PixArt-α/DiT-S 权重 → Parquet + ArrowQuant INT4
*   **T3.8: 音频扩散集成**
    *   [ ] 实现 `AudioProjectionHead` (WaveGrad 路线)
    *   [ ] 转换 WaveGrad 2 权重 → Parquet
*   **T3.9: Consistency Distillation 加速**
    *   [ ] 蒸馏标准 50 步 → 4 步
    *   [ ] 验证: 4 步质量 ≥ 50 步的 90%
*   **T3.10: 多模态并行生成验证**
    *   [ ] 虚拟具身场景端到端测试（文字+语音+口型同步生成）
    *   [ ] 边缘部署验证（ARM + INT2）
    *   [ ] 合并至主分支

## Phase 4: 记忆生命周期管理 (Memory Lifecycle) - [ ] 0%
*   **T4.1: 异步浓缩 Worker**
    *   [ ] 实现 `evolution/consolidator.py`。
*   **T4.2: 记忆分层策略引擎**
    *   [ ] 实现基于 LRU 和 TF-IDF 的冷热交换逻辑。
*   **T4.3: 自动化任务编排**
    *   [ ] 集成至 `aios_cli.py` 的 `cron` 命令。

## Phase 5: 联邦同步优化 (Federation Optimization) - [ ] 0%
*   **T5.1: 增量权重计算 (Delta-weight)**
    *   [ ] 实现 `federation/delta_generator.py`。
*   **T5.2: Flight RPC 高级传输模式**
    *   [ ] 启用 `DoExchange` 实现双向流式增量同步。

---
最后更新: 2026-02-20
状态: 执行中 (In-progress)


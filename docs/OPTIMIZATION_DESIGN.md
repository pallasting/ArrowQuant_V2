# AI-OS: 优化阶段系统设计 (Optimization Design)

## 1. 架构目标 (Architectural Goals)

本设计旨在将 ArrowEngine 的静态零拷贝检索能力转化为动态、高性能的多模态认知核心。

## 2. 核心模块设计 (Core Module Design)

### 2.1 ArrowEngine 计算加速层 (ArrowEngine Compute Accelerator)
*   **Provider 抽象层扩展**：在 `llm_compression/inference/` 中引入 `AcceleratedProvider`。
*   **后端集成**：
    *   **CPU (Intel/AMD)**：通过 `torch.compile` (Inductor) 或直接集成 `intel_extension_for_pytorch` (IPEX) 利用 AMX/AVX-512 指令集。
    *   **GPU (NVIDIA)**：实现 `CudaProvider`，利用 `torch.cuda.amp` 进行自动混合精度推理。
*   **动态 Device 切换**：支持配置驱动的驱动选择（Auto-detection of CUDA/MPS/CPU）。

### 2.2 量化推理引擎 (Quantized Inference Engine)
*   **量化策略**：采用 **Post-Training Quantization (PTQ)**。
*   **权重格式**：在 Parquet 中存储 Int8 权重，并保持对应的 `scale` 和 `zero_point` 列。
*   **算子级优化**：自定义 Arrow 扩展算子，直接在 Arrow Buffer 上进行有符号整数计算，减少向 Float 的中间转换。

### 2.3 统一多模态关联空间 (Unified Multimodal Embedding Space)
*   **Joint Embedding Layer**：设计一个轻量级的融合 MLP，接收来自 Vision、Audio、Text 的投影向量（Projected Vectors）。
*   **关联图索引**：在 Arrow 表中建立 `association_links` 列，存储跨模态的 ID 映射。
*   **检索逻辑**：从“基于向量的 K-NN”升级为“向量相似度 + 关联图权重”的混合排序算法。

### 2.4 异步记忆浓缩器 (Asynchronous Memory Consolidator)
*   **Worker 模式**：独立于主进程的 `ConsolidationWorker`。
*   **Pipeline**：
    1.  **Scan**：识别高频率访问的“热点”短期记忆。
    2.  **Summarize**：批量发送至 LLM 生成核心摘要。
    3.  **Merge**：更新长期存储，并将原始短期记忆标记为 `archived` 或删除。
*   **触发机制**：基于时间（Nightly）或存储压力（Pressure-based）。

## 3. 接口变更 (Interface Changes)

*   `ArrowEngine.express()`：新增 `multimodal_context` 参数。
*   `LoRAFlightClient.sync_delta()`：新增增量同步接口。

## 4. 安全与边界 (Safety & Boundaries)

*   **VRAM 保护**：在 GPU 启用时，增加 VRAM 占用监控，若超过负载自动回退至 CPU (Fall-back to CPU)。
*   **量化精度保护**：每次量化转换后，自动运行 `accuracy_checker` 确保相似度漂移在允许范围内。

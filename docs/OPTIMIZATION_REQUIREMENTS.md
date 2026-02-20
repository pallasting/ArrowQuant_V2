# AI-OS: 多模态与ArrowEngine内存检索优化需求 (Optimization Requirements)

## 1. 概述 (Overview)

AI-OS 系统目前的架构已实现了零拷贝（Zero-Copy）、多模态感知（Multimodal Perception）以及联邦技能树（Federated Learning）的基础原型设计。然而，作为 P0 级别核心基础设施，当前的纯 CPU Float32/Float16 计算机制无法满足大规模部署或低延迟认知循环的需求。本阶段我们的核心目标是**系统性消除计算与业务层面的性能瓶颈**，实现端到端的计算加速，并重构部分落后的业务逻辑。

## 2. 需求分类 (Requirements Breakdown)

本阶段的优化需求分为两大类：**底层效能优化 (Technical Efficiency)** 和 **业务逻辑与架构演进 (Business Logic Evolution)**。

### 2.1 底层效能优化 (Technical Efficiency Optimization)

*   **REQ-OPT-1: CPU 计算层极限优化 (Phase 1 Quick Wins)**
    *   **描述**：针对 ArrowEngine 的推理（特别是 Vision 和 Audio Encoder），集成 `MKL-DNN/ONEDNN` 等优化的数学运算库。
    *   **验收标准**：在不支持 GPU 的机器上，纯 CPU 推理吞吐率提升至少 2 倍（例如 Whisper-tiny 编码耗时从 300ms 降低到 <150ms）。
*   **REQ-OPT-2: 模型量化 (Model Quantization)**
    *   **描述**：实现或集成 Int8 模型量化支持。
    *   **验收标准**：在精度损失 (Cosine Similarity) < 0.05 的前提下，模型显存/内存占用下降至目前的 1/4（例如 380MB -> 95MB），推理速度提升 2x-3x。
*   **REQ-OPT-3: 异构加速与 GPU 支持 (GPU/CUDA Acceleration)**
    *   **描述**：打破现有的纯 CPU 限制，为 ArrowEngine 和各个 Multimodal Encoder 引入显性的 `device='cuda'` (或 `'mps'`) 支持，并在数据从 GPU VRAM 到 Arrow 内存的拷贝路径上实现极致的 Zero-Copy 传输。
    *   **验收标准**：在配备 GPU 的设备上，推理速度应达到类似规模 HF 模型的 90% 以上，同时保留 Arrow 原先在显存外的内存扩展优势。

### 2.2 业务逻辑与架构演进 (Business Logic Evolution)

*   **REQ-BIZ-1: 真正的跨模态融合 (True Multimodal Fusion)**
    *   **描述**：重构当前视觉、听觉各自分离编码和检索的逻辑。引入一个统一的关联语义空间池，使 "A看到的东西" 能够直接激发 "A听到的东西" 的网络导航。
    *   **验收标准**：CognitiveLoop 支持输入 `{vision: ..., audio: ..., text: ...}` 形成一个复合的联合向量 (Joint Embedding) 进行单一入口检索。
*   **REQ-BIZ-2: 增量联邦技能树同步 (Incremental Federated Sync)**
    *   **描述**：优化已经实现的 Phase 8 Flight RPC 逻辑，使其不仅支持全量 LoRA Card 传输，更支持细粒度的 Delta 权重同步，降低带宽。
    *   **验收标准**：连续两次技能训练后，同步的 Flight 数据量减少 80% 以上。
*   **REQ-BIZ-3: 大模型驱动的记忆浓缩 (LLM-Driven Memory Consolidation)**
    *   **描述**：在 ArrowEngine 上层增加一个异步/定时作业 (Nightly Batch Task)。调用高层级 LLM (如 Ollama/Claude) 读取大量零散的短期记忆，进行聚类、去重、核心抽象抽取，最后重新存入“长期核心原则区”。
    *   **验收标准**：包含长短期自动分层逻辑，支持定制化的淘汰 (Eviction) 或压缩 (Compression) 策略配置。

## 3. 非功能性需求 (Non-Functional Requirements)

*   **向下兼容 (Backward Compatibility)**: 所有的模型格式必须兼容现有已转换的 `models/` 目录结构和 Parquet/Arrow 格式。
*   **测试覆盖率 (Test Coverage)**: 针对所有的优化特性（量化、GPU 支持），必须编写对应的单元测试和至少 2 个端到端集成测试，测试通过率需达到 100%。

## 4. 下一步阶段 (Next Steps)

*   据此需求，产出对应的系统级设计文档 `OPTIMIZATION_DESIGN.md`。
*   基于设计拆分为可执行的 Task，并用 `TASKS_OPTIMIZATION.md` 跟踪。

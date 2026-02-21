# 统一扩散生成架构需求文档 (Unified Diffusion Architecture Requirements)

> **版本**: v1.0  |  **日期**: 2026-02-20  |  **状态**: Draft  
> **前置分析**: `unified_diffusion_analysis.md`  
> **依赖**: Phase 2 ArrowQuant 量化基础设施

---

## 1. 背景与愿景

### 1.1 问题陈述

AI-OS 当前的生成端依赖**自回归 (AR) 模型**（Qwen2.5-0.5B），各模态（文本/代码/图像/语音/虚拟具身）需要独立的模型实例，导致：

- **内存膨胀**：N 个模态 = N 份独立权重（手机端 4 个模型 ≈ 2+ GB）
- **级联延迟**：文本 → 语音 → 口型是串行管线
- **上下文割裂**：语音模型不感知图像模型的状态
- **技能不可组合**：新学的 LoRA 无法与跨模态能力自然叠加
- **不确定性盲区**：AR 模型不知道自己"不知道什么"，无法精确触发自我进化

### 1.2 目标愿景

用**热力学约束的扩散模型 (Thermodynamic Diffusion)** 替代 AR 模型，实现：

```
一个 Score Network + 多个投影 Head = 统一生成所有模态
```

**终极场景**：手机启动后出现虚拟具身形象，面对面语音对话，同时展示图片/数据——所有输出由**一次 Transformer forward** 的不同投影同步产出。

---

## 2. 功能性需求

### REQ-UD-1: 离散扩散文本/代码生成引擎

- **描述**：基于 MDLM/SEDD 算法实现离散扩散文本生成，支持前向 mask 加噪和反向迭代去噪
- **能力**：
  - 无序生成（非左→右，支持 infilling 和代码补全）
  - 双向上下文感知
  - 可控生成长度
- **接口**：`ArrowEngine.diffuse(prompt, modality="text"|"code")`
- **验收标准**：
  - 在同参数规模下，PPL 与 AR 基线差距 < 20%
  - 支持 infilling：给定上下文头尾，中间生成质量可用
  - 4 步 Consistency Distillation 后延迟 < 500ms (MiniLM 规模)

### REQ-UD-2: 潜空间连续扩散图像生成引擎

- **描述**：基于 DiT (Diffusion Transformer) 实现 text-to-image 潜空间扩散
- **组件**：VAE Encoder/Decoder + DiT Score Network + CLIP 条件编码器
- **接口**：`ArrowEngine.diffuse(prompt, modality="image", size=(512,512))`
- **验收标准**：
  - 轻量模型 (DiT-S/PixArt-α, < 600M 参数) 在本地 CPU 上 < 30s 生成一张 512×512 图
  - INT4 量化后模型大小 < 200MB
  - 支持 ArrowStorage 记忆条件注入

### REQ-UD-3: 波形/潜空间音频扩散引擎

- **描述**：支持条件音频生成（TTS、音效），可选 WaveGrad 波形路线或 SoundStorm 离散 token 路线
- **接口**：`ArrowEngine.diffuse(prompt, modality="audio", voice_id="default")`
- **验收标准**：
  - TTS 延迟 < 2s (端到端，含文本理解)
  - 生成音频采样率 ≥ 16kHz
  - 支持 zero-shot voice cloning（给定 5s 参考音频）

### REQ-UD-4: 统一 Score Network (Unified Backbone)

- **描述**：所有模态共享一个 Transformer 骨架，通过模态嵌入 + 时间步嵌入区分，各模态仅有轻量投影 Head 不同
- **架构**：
  - 共享骨架：`SharedTransformer(hidden_dim, num_layers)`
  - 模态嵌入：`ModalityEmbedding(4, hidden_dim)` — text/code/image/audio
  - 时间嵌入：`SinusoidalTimeEmbedding(hidden_dim)` — 去噪步骤 t
  - 投影 Head：`TextHead` / `ImageHead` / `AudioHead`（各 < 10M 参数）
- **验收标准**：
  - 共享骨架参数占比 > 90%
  - 添加新模态仅需训练新 Head（< 10M 参数），骨架冻结
  - 多模态同时生成时，一次 forward 产出所有 Head 的隐表示

### REQ-UD-5: 记忆引导的扩散条件注入

- **描述**：利用 ArrowStorage 的向量检索结果作为扩散生成的条件向量，实现个性化记忆引导生成
- **流程**：
  1. 用户 Query → ArrowStorage.search() → Top-K 记忆向量
  2. 记忆向量经 `MemoryConditioner` 投影为条件嵌入 c
  3. Score Network 在每步去噪中接收 c 作为 cross-attention 条件
- **验收标准**：
  - 生成结果与检索到的记忆内容语义相关（人工评估 > 80% 相关率）
  - 条件注入延迟 < 10ms

### REQ-UD-6: 不确定性感知的自进化触发

- **描述**：利用扩散模型去噪过程中的噪声残差作为不确定性度量，替代当前 LoRA Router 的启发式 confidence 阈值
- **机制**：
  - 去噪收敛速度慢 → 高不确定性 → 触发 `_trigger_evolution()`
  - 噪声残差 $\|x_t - \hat{x}_0\|$ 作为定量不确定性指标
- **验收标准**：
  - 不确定性指标与人工标注的"模型不确定场景"相关性 > 0.7
  - 误触发率（不需要进化时触发）< 5%

### REQ-UD-7: Consistency Distillation 加速

- **描述**：将标准 50 步扩散过程蒸馏为 1–4 步，满足端侧实时性要求
- **方法**：Consistency Models / Progressive Distillation
- **验收标准**：
  - 4 步生成质量 ≥ 50 步生成质量的 90%
  - 文本生成延迟 < 500ms (350M 参数, INT2, CPU)

### REQ-UD-8: ControlNet 行为与结构约束

- **描述**：支持动态挂载 ControlNet 旁路网络，对生成过程施加额外的结构、逻辑或风格约束
- **机制**：在不去修改 Score Network 基础权重的前提下，零初始化旁路注入
- **Agent 专属扩展**：
  - **`CoT-ControlNet`**：强制注入规划与推理的骨架模板（思考→行动→观察），防止死循环。
  - **`ToolSchema-ControlNet`**：注入工具调用的 JSON Schema 约束，确保工具调用参数 100% 符合被调工具的格式。
- **验收标准**：
  - 支持多个 ControlNet 权重叠加组合
  - ControlNet 参数量 < 基座的 10%

### REQ-UD-9: 渐进式自我进化机制 (5-Level)

- **描述**：建立从"零训练组合"到"全量微调"的五层自进化体系
- **机制**：
  - L0: Score 组合 (实时叠加)
  - L1: ControlNet (结构偏好，~10% 参数训练)
  - L2: LoRA 微调 (知识吸收，~1% 参数训练)
  - L3: 选择性骨架微调 (不确定性驱动的层级解冻)
  - L4: 全量微调 (长期 Consolidation)

### REQ-UD-10: 能量约束模型 (EBM) 前置/后置校验融合

- **描述**：支持引入轻量级能量模型对扩散生成的候选空间进行约束，施加绝对物理或逻辑法则
- **机制**：$\nabla\log p_{final} = \nabla\log p_{diffusion} - \eta \nabla E_{constraint}$

---

## 3. 非功能性需求

### NFR-UD-1: 分层部署

| 部署层 | 设备 | 必须支持的模态 | 模型规模 |
|--------|------|---------------|---------|
| 边缘层 | 手机/嵌入式, 2–4GB | 文本 + 音频 | < 100M params, INT2, < 35MB |
| 本地层 | 工作站, 8+GB | 文本 + 音频 + 图像 | < 600M params, INT4, < 200MB |
| 云端层 | GPU 服务器 | 全模态 + 统一 Score Network | < 3B params |

### NFR-UD-2: 存储格式统一

- 所有扩散模型权重必须存储为 **Parquet V2** 格式
- 支持 **ArrowQuant INT2/INT4** 量化
- 零拷贝加载 + 惰性反量化

### NFR-UD-3: 向下兼容

- 现有 `ArrowEngine.encode()` 和 `ArrowEngine.generate()` 接口不变
- 新增 `ArrowEngine.diffuse()` 接口
- 已有 AR 模型（Qwen-0.5B）可与扩散模型共存

### NFR-UD-4: 扩展与自训练兼容

- 扩散模型的 Score Network 支持 ControlNet 和 LoRA 动态挂载
- 现有 LoRA Router 可扩建为 **Evolution Router** 进行 L0-L4 路由
- ⚠️ AR 模型的 LoRA 不可与扩散模型的 LoRA 混用

### NFR-UD-5: 自进化闭环

```
ArrowStorage (Hopfield) → 记忆/条件注入 → 扩散生成 (Score) + 能量校验 (EBM) 
   ↑                                   │
   └────────── 5-Level 增量学习 ◄───────┴─ 不确定性判断
```


---

## 4. 排除范围

- **AR 模型替换**：不在本期删除 AR 模型；扩散模型作为并行路径引入
- **视频生成**：本期不做视频扩散
- **联合训练**：初期各模态独立训练，不做多模态联合训练
- **自定义 CUDA/ARM 内核**：在 PyTorch 层面实现，不编写硬件专用内核

---

## 5. 验收标准汇总

| 编号 | 指标 | 目标 |
|------|------|------|
| AC-1 | 文本 PPL 与 AR 差距 | < 20% |
| AC-2 | 4 步文本生成延迟 (350M, INT2, CPU) | < 500ms |
| AC-3 | 图像生成延迟 (600M, INT4, CPU) | < 30s |
| AC-4 | TTS 端到端延迟 | < 2s |
| AC-5 | 共享骨架参数占比 | > 90% |
| AC-6 | 记忆引导相关率 | > 80% |
| AC-7 | 不确定性触发误触率 | < 5% |
| AC-8 | 边缘层模型大小 (INT2) | < 35 MB |
| AC-9 | 现有 API 兼容性 | 100% 向下兼容 |

---
*最后更新: 2026-02-20*

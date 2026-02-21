# 统一扩散生成架构系统设计 (Unified Diffusion Architecture Design)

> **版本**: v1.0  |  **日期**: 2026-02-20  |  **状态**: Draft  
> **前置**: `UNIFIED_DIFFUSION_REQUIREMENTS.md` | `unified_diffusion_analysis.md`

---

## 1. 设计总纲

### 核心理念

```
统一扩散 = 共享 Score Network + 模态投影 Head + 热力学采样器
```

所有模态的生成被统一为**一个去噪过程**——从噪声/[MASK] 状态出发，通过学到的 score function（对数概率梯度）迭代收敛到数据分布，最终由各模态的投影 Head 解码为具体形式。

### 热力学对应

| 物理概念 | 架构对应 | 代码对应 |
|----------|---------|---------|
| 热力学平衡 | 纯噪声 $\mathcal{N}(0,I)$ / 全 [MASK] | 采样初始状态 |
| 朗之文动力学 | 连续模态去噪步 | `ContinuousSampler.step()` |
| CTMC 跳跃过程 | 离散模态 unmask 步 | `DiscreteSampler.step()` |
| 自由能最小化 | Score Matching Loss | 训练目标 |
| 熵产生率 | 噪声残差 $\|x_t - \hat{x}_0\|$ | 不确定性度量 |

---

## 2. 系统架构

### 2.1 全局架构图

```
┌──────────────────────────────────────────────────────────┐
│                     用户 API 层                           │
│  .encode()  .generate()  .diffuse()  .render_avatar()    │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                  ArrowEngine (路由层)                      │
│  mode="ar"  → InferenceCore (现有 AR 路径)                │
│  mode="diffusion" → DiffusionCore (新增扩散路径)           │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                DiffusionCore (新增)                        │
│  ┌──────────────────────────────────────────────────┐    │
│  │  UnifiedScoreNetwork                              │    │
│  │  ┌──────────┐  ┌───────────┐  ┌───────────────┐  │    │
│  │  │Modality  │  │Sinusoidal │  │  Condition     │  │    │
│  │  │Embedding │  │Time Embed │  │  Encoder       │  │    │
│  │  └────┬─────┘  └─────┬─────┘  └──────┬────────┘  │    │
│  │       └──────────────┼───────────────┘            │    │
│  │                      ▼                            │    │
│  │           SharedTransformer (N layers)             │    │
│  │                      │                            │    │
│  │       ┌─────┬────────┼────────┬──────┐            │    │
│  │       ▼     ▼        ▼        ▼      ▼            │    │
│  │    TextH ImageH  AudioH  CodeH  AvatarH           │    │
│  └──────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────┐    │
│  │  EvolutionRouter (5-Level 自进化引擎)             │    │
│  │   ├── L0: Score Mixer                             │    │
│  │   ├── L1: ControlNet Bank (结构偏好)               │    │
│  │   ├── L2: LoRA Router (领域知识)                  │    │
│  │   └── L3: Selective Finetuner (深度认知修正)       │    │
│  └──────────────────────────────────────────────────┘    │
│                          │                                │
│  ┌──────────┐  ┌────────▼─────────┐  ┌──────────────┐  │
│  │ Discrete │  │ NoiseScheduler   │  │ Continuous   │  │
│  │ Sampler  │  │ (统一调度)        │  │ Sampler      │  │
│  │(text/code│  │ + Energy Models  │  │(image/audio) │  │
│  └──────────┘  └───────────────────┘  └──────────────┘  │
└──────────────────────────┬───────────────────────────────┘
                           │ weights
┌──────────────────────────▼───────────────────────────────┐
│  WeightLoader V2 + ArrowQuant (Parquet V2, 零拷贝)       │
└──────────────────────────────────────────────────────────┘
```

### 2.2 与现有架构的关系

```
ArrowEngine
  ├── mode="ar" (现有，保留)
  │     ├── InferenceCore
  │     ├── LoRA Router
  │     └── WeightLoader V1/V2
  │
  └── mode="diffusion" (新增)
        ├── DiffusionCore (本文档的核心)
        ├── EvolutionRouter (L0-L4 渐进式进化的核心中枢)
        │     ├── ControlNet Bank (L1: 行为控制)
        │     ├── LoRA Router (L2: 知识注入)
        │     └── Selective Finetuner (L3: 部分解冻训练)
        ├── EnergyModelValidator (后处理与约束叠加)
        └── WeightLoader V2 (复用，含 ArrowQuant)
```

---

## 3. 核心模块设计

### 3.1 UnifiedScoreNetwork

```python
class UnifiedScoreNetwork(nn.Module):
    """
    统一 Score Network：所有模态共享 Transformer 骨架。
    
    参数量分布:
      SharedTransformer: ~90% (骨架，负责"理解")
      各 Head: ~10% (投影，负责"表达")
    """
    def __init__(self, config: DiffusionConfig):
        # 共享组件
        self.modality_embed = nn.Embedding(5, config.hidden_dim)  
        self.time_embed = SinusoidalTimeEmbedding(config.hidden_dim)
        self.condition_proj = nn.Linear(config.condition_dim, config.hidden_dim)
        self.shared_transformer = TransformerStack(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_dim=config.intermediate_dim,
        )
        
        # 模态投影 Head (各 < 10M 参数)
        self.heads = nn.ModuleDict({
            "text": TextProjectionHead(config.hidden_dim, config.vocab_size),
            "code": TextProjectionHead(config.hidden_dim, config.vocab_size),
            "image": PatchProjectionHead(config.hidden_dim, patch_size=16, channels=4),
            "audio": WaveformProjectionHead(config.hidden_dim, hop_length=256),
        })
    
    def forward(self, x_t, t, modality, condition=None):
        """
        Args:
            x_t: 当前噪声状态 (已由 Head.encode 投射到隐空间)
            t: 时间步 [0, T]
            modality: "text" | "code" | "image" | "audio"
            condition: 条件向量 (来自记忆检索或 CLIP)
        Returns:
            score 或 ε 预测 (由 Head.decode 投射回模态空间)
        """
        # 1. Input Head: 模态数据 → 隐空间
        h = self.heads[modality].encode(x_t)
        
        # 2. 叠加时间和模态嵌入
        h = h + self.time_embed(t) + self.modality_embed(MODALITY_IDS[modality])
        
        # 3. 条件注入 (Cross-Attention)
        if condition is not None:
            condition = self.condition_proj(condition)
        
        # 5. Output Head: 隐空间 → 模态空间 (score/ε/logits)
        base_score = self.heads[modality].decode(h)
        
        return base_score
```

### 3.1b EvolutionRouter (渐进式进化中枢)

```python
class EvolutionRouter:
    """
    5层渐进式自我进化中枢。
    在基础 score 上叠加 ControlNet, LoRA, 和能量约束。
    """
    def __init__(self, score_network, control_nets, lora_manager, energy_models):
        self.score_net = score_network
        self.control_nets = control_nets  # L1
        self.loras = lora_manager         # L2
        self.energy_models = energy_models # EBM 融合
        
    def get_fused_score(self, x_t, t, modality, condition, active_profiles):
        """
        L0 Score 组合: $\nabla p_{final} = \nabla p_{base} + \alpha \nabla p_{control} - \eta \nabla E$
        """
        # 基础 Score
        score = self.score_net(x_t, t, modality, condition)
        
        # L1: ControlNet 行为约束注入
        for profile in active_profiles.control_nets:
            c_net = self.control_nets[profile.id]
            score += profile.weight * c_net(x_t, t, profile.condition_template)
            
        # EBM: 获取能量模型约束梯度
        for e_model in active_profiles.energy_models:
            energy_grad = torch.autograd.grad(e_model(x_t, t).sum(), x_t)[0]
            score -= e_model.weight * energy_grad
            
        return score

```

### 3.2 DiffusionCore

```python
class DiffusionCore:
    """
    扩散推理核心：管理去噪循环。
    
    相当于 InferenceCore 在扩散范式下的对应物。
    """
    def __init__(self, score_network, scheduler, config):
        self.score_net = score_network
        self.scheduler = scheduler    # NoiseScheduler
        self.config = config
    
    def generate(self, condition, modality, num_steps=4):
        """
        统一生成入口。
        
        1. 初始化噪声状态
        2. 迭代去噪
        3. 投影头解码
        """
        # 初始化
        if modality in ("text", "code"):
            x_t = self._init_masked_sequence(condition)  # 全 [MASK]
            sampler = DiscreteSampler(self.scheduler)
        else:
            x_t = torch.randn(self._get_latent_shape(modality))  # 高斯噪声
            sampler = ContinuousSampler(self.scheduler)
        
        # 去噪循环
        for t in self.scheduler.timesteps(num_steps):
            score = self.score_net(x_t, t, modality, condition)
            x_t = sampler.step(score, t, x_t)
        
        return x_t  # 最终结果
```

### 3.3 NoiseScheduler

```python
class NoiseScheduler:
    """
    统一噪声调度器，支持离散和连续两种模式。
    
    离散模式 (文本/代码):  β(t) = mask 概率
    连续模式 (图像/音频):  σ(t) = 噪声标准差
    """
    def __init__(self, schedule_type="cosine", num_train_steps=1000):
        self.schedule_type = schedule_type
        self.num_train_steps = num_train_steps
    
    def timesteps(self, num_inference_steps):
        """返回推理时的时间步序列 (均匀或非均匀采样)"""
        ...
    
    def add_noise(self, x_0, t, mode="continuous"):
        """前向加噪"""
        if mode == "discrete":
            return self._mask_tokens(x_0, t)
        else:
            return x_0 + self.sigma(t) * torch.randn_like(x_0)
    
    def sigma(self, t):
        """连续噪声标准差 σ(t)"""
        ...
    
    def mask_rate(self, t):
        """离散 mask 概率 β(t)"""
        ...
```

### 3.4 MemoryConditioner

```python
class MemoryConditioner:
    """
    记忆引导条件器：将 ArrowStorage 检索结果转为扩散条件。
    
    这是 AI-OS 的差异化核心——个人记忆驱动生成。
    """
    def __init__(self, arrow_storage, condition_dim):
        self.storage = arrow_storage
        self.projector = nn.Linear(384, condition_dim)  # MiniLM dim → condition dim
    
    def get_condition(self, query, top_k=5):
        """
        检索相关记忆并投射为条件向量。
        
        Returns:
            condition: [K, condition_dim] 条件矩阵
        """
        # 1. 向量检索 Top-K 记忆
        results = self.storage.search(query, limit=top_k)
        
        # 2. 提取记忆嵌入向量
        memory_vectors = torch.stack([r.embedding for r in results])
        
        # 3. 投射到扩散条件空间
        condition = self.projector(memory_vectors)
        
        return condition
```

### 3.5 不确定性度量器

```python
class UncertaintyEstimator:
    """
    基于去噪残差的不确定性度量。
    
    替代 LoRA Router 的启发式 confidence 阈值。
    """
    def estimate(self, x_t, x_0_pred, t):
        """
        计算当前去噪步的不确定性。
        
        高: 模型不确定，应触发自进化
        低: 模型确定，正常输出
        """
        # 去噪残差 (应该在 t→0 时趋向 0)
        residual = (x_t - x_0_pred).norm(dim=-1).mean()
        
        # 归一化到 [0, 1]
        expected_residual = self.scheduler.sigma(t)
        uncertainty = residual / (expected_residual + 1e-8)
        
        return uncertainty.item()
    
    def should_evolve(self, uncertainty, threshold=1.5):
        """不确定性超过阈值时触发自进化"""
        return uncertainty > threshold
```

---

## 4. 数据流设计

### 4.1 文本生成流

```
"写一首关于春天的诗"
    │
    ▼ MemoryConditioner
ArrowStorage.search("春天") → [记忆: 去年春游照片, 读过的春天诗句]
    │
    ▼ condition = projector(memory_vectors)
    │
    ▼ DiscreteSampler 初始化
x_T = [MASK] [MASK] [MASK] [MASK] ... [MASK]  (L=128)
    │
    ▼ 4 步去噪 (Consistency Distillation)
    │  t=4: score_net(x_4, t=4, "text", condition)
    │       → 置信度最高的 20% 位置 unmask
    │  t=3: score_net(x_3, t=3, "text", condition)
    │       → 再 unmask 30%
    │  t=2 → t=1 → ...
    ▼
x_0 = "春风拂面花渐开，记忆中的那片樱..."  (完整文本)
```

### 4.2 多模态并行生成流 (虚拟具身场景)

```
用户语音输入: "帮我回忆一下上次去日本的旅行"
    │
    ▼ 统一条件准备
condition = MemoryConditioner.get_condition("日本旅行")
    │
    ▼ 一次 SharedTransformer forward
hidden_states = shared_transformer(x_t, t, condition)
    │
    ├── TextHead.decode(h) → "上次去日本是在..."     (文字回复)
    ├── AudioHead.decode(h) → [16kHz 波形]           (语音合成)
    ├── AvatarHead.decode(h) → [blendshape 参数]     (口型+表情)
    └── ImageHead.decode(h) → [512×512 图]           (相关图片)
    
    ▲ 所有输出天然同步，无级联延迟
```

---

## 5. 存储设计

### 5.1 模型目录结构

```
models/
  diffusion-base/                   # 统一扩散基座
    metadata.json                   # 含 diffusion 配置
    shared_transformer.parquet      # 共享骨架权重 (ArrowQuant INT2)
    tokenizer/                      # 文本 tokenizer
    heads/
      text_head.parquet             # 文本投影头
      code_head.parquet             # 代码投影头 (可共享 text)
      image_head.parquet            # 图像投影头
      audio_head.parquet            # 音频投影头
    vae/                            # 图像 VAE (可选)
      encoder.parquet
      decoder.parquet
  lora_cards/
    writing_style.parquet           # 基于扩散基座微调的 LoRA
    code_python.parquet
```

### 5.2 metadata.json 扩展

```json
{
  "model_type": "unified_diffusion",
  "diffusion_config": {
    "score_network": {
      "hidden_dim": 768,
      "num_layers": 12,
      "num_heads": 12,
      "intermediate_dim": 3072
    },
    "scheduler": {
      "type": "cosine",
      "num_train_steps": 1000,
      "num_inference_steps": 4
    },
    "supported_modalities": ["text", "code", "image", "audio"],
    "consistency_distilled": true
  },
  "quantization": {
    "method": "arrowquant_v1",
    "bit_width": 2
  }
}
```

---

## 6. ArrowEngine 集成接口

```python
class ArrowEngine:
    # 现有接口 (不变)
    def encode(self, sentences, **kwargs): ...
    def generate(self, prompt, **kwargs): ...
    
    # 新增统一扩散接口
    def diffuse(
        self,
        prompt: str,
        modality: Literal["text", "code", "image", "audio"] = "text",
        num_steps: int = 4,
        guidance_scale: float = 3.0,
        memory_guided: bool = True,
        **kwargs,
    ) -> Union[str, np.ndarray, torch.Tensor]:
        """
        统一扩散生成。
        
        Args:
            prompt: 输入 prompt
            modality: 生成目标模态
            num_steps: 去噪步数 (1-50, 默认 4 for consistency)
            guidance_scale: Classifier-Free Guidance 强度
            memory_guided: 是否使用 ArrowStorage 记忆条件
        """
        # 1. 条件编码
        condition = None
        if memory_guided:
            condition = self.memory_conditioner.get_condition(prompt)
        
        # 2. 扩散生成
        output = self.diffusion_core.generate(
            condition=condition,
            modality=modality,
            num_steps=num_steps,
        )
        
        # 3. 后处理
        if modality in ("text", "code"):
            return self.tokenizer.decode(output)
        elif modality == "image":
            return self.vae_decoder(output)  # latent → pixel
        elif modality == "audio":
            return output.numpy()  # waveform
```

---

## 7. 实施路线

### Phase 3a: 离散扩散文本 PoC — 预计 2 周

| 步骤 | 任务 | 产出 |
|------|------|------|
| S1 | 实现 `DiffusionCore` + `NoiseScheduler` + `DiscreteSampler` | 核心推理框架 |
| S2 | 实现 `TextProjectionHead` | 文本投影 |
| S3 | 转换开源 MDLM 权重 → Parquet V2 + ArrowQuant | 模型文件 |
| S4 | 实现 `ArrowEngine.diffuse(modality="text")` | API 集成 |
| S5 | Infilling 质量验证 + 延迟基准 | 验证报告 |

### Phase 3b: 统一 Score Network + 记忆条件 — 预计 2 周

| 步骤 | 任务 | 产出 |
|------|------|------|
| S6 | 实现 `UnifiedScoreNetwork` (共享 Transformer) | 统一骨架 |
| S7 | 实现 `MemoryConditioner` (ArrowStorage → 条件向量) | 记忆引导 |
| S8 | 实现 `UncertaintyEstimator` (不确定性自进化触发) | 自进化闭环 |
| S9 | 跨模态端到端测试 | 集成验证 |

### Phase 3c: 图像/音频扩散 + 虚拟具身 — 预计 3 周

| 步骤 | 任务 | 产出 |
|------|------|------|
| S10 | 实现 `ImageProjectionHead` + VAE 集成 | 图像生成 |
| S11 | 实现 `AudioProjectionHead` (WaveGrad) | 音频生成 |
| S12 | Consistency Distillation 训练 (4步压缩) | 加速模型 |
| S13 | 多模态并行生成验证（虚拟具身场景） | 场景验证 |
| S14 | 边缘部署验证 (ARM + INT2) | 部署验证 |

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 开源 MDLM 权重质量不足 | 中 | 高 | 降级为 SEDD；或从 BERT 权重初始化自训 |
| Consistency Distillation 后质量下降 | 中 | 中 | 保留多步推理作为高质量回退路径 |
| 统一骨架跨模态干扰 | 中 | 高 | 初期各 Head 独立训练，骨架分阶段解冻 |
| 边缘设备算力不足 | 低 | 中 | INT2 + 极小骨架 (~50M) 子集部署 |

---

## 9. 与 AngelSlim/ArrowQuant 的协同

统一扩散架构与 Phase 2 ArrowQuant 形成完美闭环：

```
ArrowQuant INT2 量化 (Phase 2)
        ↓ 压缩权重
UnifiedScoreNetwork.shared_transformer (Phase 3)
        ↓ ~200MB (INT2)
边缘设备可部署
```

ArrowQuant 为扩散模型提供存储基础设施，扩散模型为 ArrowQuant 提供最大的应用场景。

---
*最后更新: 2026-02-20*

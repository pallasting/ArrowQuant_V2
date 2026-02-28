# Design Document: Unified Diffusion Architecture

## 1. Design Overview

### Architecture Philosophy: Rust Skeleton + Python Brain

This system follows a **dual-layer architecture** inspired by biological systems:

- 🦴 **Rust Skeleton**: Stable, high-performance infrastructure that rarely changes
  - ArrowStorage (vector search with SIMD acceleration)
  - ArrowQuant (INT2/INT4 quantization with zero-copy)
  - VectorSearch (simsimd for similarity)
  - FastTokenizer (parallel tokenization)
  
- 🧠 **Python Brain**: Flexible, evolution-friendly learning layer that adapts constantly
  - DiffusionCore (denoising inference)
  - EvolutionRouter (L0-L4 adaptation)
  - Training scripts (experimentation)
  - User-specific adaptations

**Why This Design?**
- Skeleton provides stable foundation (10-50x performance boost)
- Brain enables rapid experimentation and user-specific evolution
- PyO3 bindings create seamless integration
- Users can evolve their "brain" without touching infrastructure

### Core Concept

```
Unified Diffusion = Shared Score Network + Modality Projection Heads + Thermodynamic Samplers
                    (Python Brain)          (Python Brain)              (Python Brain)
                    
                    Built on Rust Skeleton (ArrowStorage + ArrowQuant + VectorSearch + FastTokenizer)
```

All modality generation is unified as a single denoising process—starting from noise/[MASK] state, iteratively converging to data distribution through learned score function (log-probability gradient), and finally decoded by modality-specific projection heads.

### Thermodynamic Correspondence

| Physical Concept | Architecture Mapping | Code Mapping |
|-----------------|---------------------|--------------|
| Thermodynamic Equilibrium | Pure noise N(0,I) / Full [MASK] | Sampling initial state |
| Langevin Dynamics | Continuous modality denoising | `ContinuousSampler.step()` |
| CTMC Jump Process | Discrete modality unmask | `DiscreteSampler.step()` |
| Free Energy Minimization | Score Matching Loss | Training objective |
| Entropy Production Rate | Noise residual ‖x_t - x̂_0‖ | Uncertainty metric |

## 2. System Architecture

### 2.0 ArrowEngine: The Unified Entry Point

**ArrowEngine 是整个系统的统一入口和路由层**，它是 🧠 Python Brain 的顶层组件，负责：

#### ArrowEngine 的职责

1. **统一 API 入口**
   - 用户通过 ArrowEngine 访问所有功能
   - 提供一致的接口：`.encode()`, `.generate()`, `.diffuse()`
   - 隐藏底层实现细节（AR vs Diffusion）

2. **模式路由**
   - `mode="ar"` → 路由到 InferenceCore（现有 AR 推理路径）
   - `mode="diffusion"` → 路由到 DiffusionCore（新 Diffusion 推理路径）
   - 自动根据请求参数选择合适的模式

3. **组件协调**
   - 协调 Python Brain 各组件（DiffusionCore, EvolutionRouter, etc.）
   - 通过 PyO3 调用 Rust Skeleton 基础设施
   - 管理模型加载、权重管理、配置

4. **向后兼容**
   - 保持现有 AR 模式完全兼容
   - 新增 Diffusion 模式不影响现有功能
   - 平滑迁移路径

#### ArrowEngine 层次结构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户代码 (User Code)                      │
│  engine = ArrowEngine()                                      │
│  result = engine.diffuse("生成一首春天的诗", modality="text") │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  ArrowEngine (🧠 Python - 统一入口/路由层)                   │
│  ┌────────────────────────────────────────────────────┐     │
│  │  公共接口 (Public API)                              │     │
│  │  ├─ .encode(sentences) → embeddings                │     │
│  │  ├─ .generate(prompt, mode="ar") → text            │     │
│  │  └─ .diffuse(prompt, modality, mode="diffusion")   │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  路由逻辑 (Routing Logic)                           │     │
│  │  ├─ if mode == "ar":                               │     │
│  │  │     return self.inference_core.generate(...)    │     │
│  │  └─ if mode == "diffusion":                        │     │
│  │        return self.diffusion_core.generate(...)    │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  组件管理 (Component Management)                    │     │
│  │  ├─ WeightLoader (加载模型权重)                     │     │
│  │  ├─ ConfigManager (管理配置)                        │     │
│  │  └─ ResourceManager (管理资源)                      │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────┬───────────────┬────────────────────┘
                          │               │
                  (AR模式) │               │ (Diffusion模式)
                          ↓               ↓
        ┌──────────────────────┐  ┌──────────────────────────┐
        │ InferenceCore        │  │ DiffusionCore (🧠 Python)│
        │ (现有AR推理)          │  │  ├─ UnifiedScoreNetwork  │
        │  ├─ Transformer      │  │  ├─ EvolutionRouter      │
        │  ├─ LoRA Router      │  │  ├─ MemoryConditioner    │
        │  └─ Sampler          │  │  ├─ NoiseScheduler       │
        │                      │  │  └─ Samplers             │
        └──────────┬───────────┘  └──────────┬───────────────┘
                   │                         │
                   │ 调用Rust基础设施         │ 调用Rust基础设施
                   ↓                         ↓
        ┌────────────────────────────────────────────────────┐
        │  🦴 Rust Skeleton (基础设施层 - PyO3绑定)          │
        │  ├─ ArrowStorage  (向量搜索, 10-50x加速)           │
        │  ├─ ArrowQuant    (量化, 5-10x加速)                │
        │  ├─ VectorSearch  (SIMD相似度计算)                 │
        │  └─ FastTokenizer (并行分词, 10-100x加速)          │
        └────────────────────────────────────────────────────┘
```

#### ArrowEngine 代码示例

```python
class ArrowEngine:
    """
    统一推理引擎 - 系统的唯一入口点
    
    职责：
    1. 提供统一API
    2. 路由到正确的推理模式
    3. 协调Python Brain和Rust Skeleton
    """
    
    def __init__(self, config_path: str):
        # 初始化配置
        self.config = Config.from_yaml(config_path)
        
        # 初始化Rust基础设施 (通过PyO3)
        self.arrow_storage = ArrowStorage.new(self.config.storage_path)
        self.arrow_quant = ArrowQuant.new(bit_width=2)
        self.fast_tokenizer = FastTokenizer.from_pretrained(self.config.tokenizer)
        
        # 初始化Python推理组件
        self.inference_core = InferenceCore(config=self.config)  # AR模式
        self.diffusion_core = DiffusionCore(config=self.config)  # Diffusion模式
        
        # 初始化权重加载器
        self.weight_loader = WeightLoader(
            arrow_quant=self.arrow_quant  # 使用Rust量化
        )
    
    # ========== 公共API ==========
    
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        """编码文本为向量 (使用Rust FastTokenizer)"""
        tokens = self.fast_tokenizer.encode_batch(sentences)
        return self.inference_core.encode(tokens, **kwargs)
    
    def generate(self, prompt: str, mode: str = "ar", **kwargs) -> str:
        """
        生成文本 (AR模式)
        
        Args:
            prompt: 输入提示
            mode: 推理模式 ("ar" 或 "diffusion")
        """
        if mode == "ar":
            return self._generate_ar(prompt, **kwargs)
        elif mode == "diffusion":
            return self.diffuse(prompt, modality="text", **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
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
        统一扩散生成 (Diffusion模式)
        
        Args:
            prompt: 输入提示
            modality: 目标生成模态
            num_steps: 去噪步数
            guidance_scale: 引导强度
            memory_guided: 是否使用记忆引导
        
        Returns:
            生成结果 (文本字符串或numpy数组)
        """
        # 1. 准备条件向量 (如果启用记忆引导)
        condition = None
        if memory_guided:
            # 使用Rust ArrowStorage进行向量搜索
            memory_results = self.arrow_storage.search(
                query=self._encode_query(prompt),
                top_k=5
            )
            condition = self._prepare_condition(memory_results)
        
        # 2. 调用DiffusionCore生成
        result = self.diffusion_core.generate(
            prompt=prompt,
            modality=modality,
            condition=condition,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )
        
        return result
    
    # ========== 内部方法 ==========
    
    def _generate_ar(self, prompt: str, **kwargs) -> str:
        """AR模式生成 (现有实现)"""
        return self.inference_core.generate(prompt, **kwargs)
    
    def _encode_query(self, query: str) -> np.ndarray:
        """编码查询为向量"""
        return self.encode([query])[0]
    
    def _prepare_condition(self, memory_results) -> torch.Tensor:
        """准备条件向量"""
        # 从记忆结果中提取嵌入
        embeddings = [result.embedding for result in memory_results]
        return torch.tensor(embeddings)
```

#### ArrowEngine vs 其他组件的关系

| 组件 | 层级 | 职责 | 实现语言 |
|------|------|------|----------|
| **ArrowEngine** | API层 | 统一入口、路由、协调 | 🧠 Python |
| InferenceCore | 业务逻辑层 | AR推理 | 🧠 Python |
| DiffusionCore | 业务逻辑层 | Diffusion推理 | 🧠 Python |
| UnifiedScoreNetwork | 业务逻辑层 | 统一评分网络 | 🧠 Python (PyTorch) |
| EvolutionRouter | 业务逻辑层 | 进化路由 | 🧠 Python |
| MemoryConditioner | 业务逻辑层 | 记忆条件化 | 🧠 Python |
| ArrowStorage | 基础设施层 | 向量存储/搜索 | 🦴 Rust (PyO3) |
| ArrowQuant | 基础设施层 | 权重量化 | 🦴 Rust (PyO3) |
| FastTokenizer | 基础设施层 | 快速分词 | 🦴 Rust (PyO3) |

**关键理解**：
- ArrowEngine 是**用户唯一接触的接口**
- DiffusionCore、EvolutionRouter 等是 ArrowEngine **内部使用的组件**
- Rust Skeleton 是**底层基础设施**，被 Python Brain 通过 PyO3 调用
- 用户不直接调用 DiffusionCore，而是通过 `ArrowEngine.diffuse()` 间接调用

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     User API Layer (🧠 Python)            │
│  .encode()  .generate()  .diffuse()  .render_avatar()    │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│              ArrowEngine (🧠 Python Router)               │
│  mode="ar"  → InferenceCore (Existing AR path)           │
│  mode="diffusion" → DiffusionCore (New diffusion path)   │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                DiffusionCore (🧠 Python)                  │
│  ┌──────────────────────────────────────────────────┐    │
│  │  UnifiedScoreNetwork (🧠 Python/PyTorch)          │    │
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
│  │  EvolutionRouter (🧠 Python - 5-Level Evolution)  │    │
│  │   ├── L0: Score Mixer                             │    │
│  │   ├── L1: ControlNet Bank                         │    │
│  │   ├── L2: LoRA Router                             │    │
│  │   └── L3: Selective Finetuner                     │    │
│  └──────────────────────────────────────────────────┘    │
│                          │                                │
│  ┌──────────┐  ┌────────▼─────────┐  ┌──────────────┐  │
│  │ Discrete │  │ NoiseScheduler   │  │ Continuous   │  │
│  │ Sampler  │  │ (Unified)        │  │ Sampler      │  │
│  │(text/code│  │ + Energy Models  │  │(image/audio) │  │
│  └──────────┘  └───────────────────┘  └──────────────┘  │
└──────────────────────────┬───────────────────────────────┘
                           │ weights (via PyO3)
┌──────────────────────────▼───────────────────────────────┐
│  🦴 Rust Skeleton Infrastructure (PyO3 Bindings)         │
│  ├── ArrowStorage (vector search, 10-50x speedup)        │
│  ├── ArrowQuant (INT2/INT4, 5-10x speedup)               │
│  ├── VectorSearch (simsimd SIMD acceleration)            │
│  └── FastTokenizer (parallel tokenization, 10-100x)      │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Integration with Existing Architecture

```
ArrowEngine (🧠 Python)
  ├── mode="ar" (Existing, Keep)
  │     ├── InferenceCore
  │     ├── LoRA Router
  │     └── WeightLoader V1/V2
  │
  └── mode="diffusion" (New)
        ├── DiffusionCore (🧠 Python - Core of this design)
        ├── EvolutionRouter (🧠 Python - L0-L4 progressive evolution hub)
        │     ├── ControlNet Bank (L1: Behavior control)
        │     ├── LoRA Router (L2: Knowledge injection)
        │     └── Selective Finetuner (L3: Partial unfreezing)
        ├── EnergyModelValidator (🧠 Python - Post-processing & constraint)
        └── 🦴 Rust Infrastructure (via PyO3)
              ├── ArrowStorage (vector search)
              ├── ArrowQuant (quantization)
              ├── VectorSearch (similarity)
              └── FastTokenizer (tokenization)
```

### 2.3 Rust-Python Integration via PyO3

All Rust components expose Python bindings through PyO3:

```python
# Python code seamlessly calls Rust
from ai_os_rust import ArrowStorage, ArrowQuant, FastTokenizer

# Rust ArrowStorage with 10-50x speedup
storage = ArrowStorage.new("./memory.arrow")
results = storage.search(query_vector, top_k=5)  # SIMD-accelerated

# Rust ArrowQuant with 5-10x speedup
quant = ArrowQuant.new(bit_width=2)
quantized = quant.quantize(weights)  # Zero-copy

# Rust FastTokenizer with 10-100x speedup
tokenizer = FastTokenizer.from_pretrained("bert-base")
tokens = tokenizer.encode_batch(texts)  # Parallel processing
```


## 3. Core Module Design

### 3.1 UnifiedScoreNetwork

**Purpose**: Shared Transformer backbone processing all modalities with modality-specific projection heads.

**Key Components**:
- SharedTransformer: ~90% of parameters, handles "understanding"
- Projection Heads: ~10% of parameters, handles "expression"
- Modality Embeddings: Distinguish between text/code/image/audio
- Time Embeddings: Encode denoising timestep
- Condition Projector: Project memory/CLIP conditions

**Interface**:
```python
class UnifiedScoreNetwork(nn.Module):
    def forward(self, x_t, t, modality, condition=None):
        """
        Args:
            x_t: Current noisy state (projected to hidden space by Head.encode)
            t: Timestep [0, T]
            modality: "text" | "code" | "image" | "audio"
            condition: Condition vector (from memory retrieval or CLIP)
        Returns:
            score or ε prediction (projected back to modality space by Head.decode)
        """
```

### 3.2 DiffusionCore

**Purpose**: Manages the iterative denoising loop for diffusion generation.

**Key Responsibilities**:
- Initialize noise state (masked tokens or Gaussian noise)
- Execute denoising loop with appropriate sampler
- Route to correct projection head for decoding

**Interface**:
```python
class DiffusionCore:
    def generate(self, condition, modality, num_steps=4):
        """
        Unified generation entry point.
        
        1. Initialize noise state
        2. Iterative denoising
        3. Projection head decoding
        """
```

### 3.3 NoiseScheduler

**Purpose**: Unified noise scheduling for both discrete and continuous modalities.

**Key Features**:
- Discrete mode: β(t) = mask probability
- Continuous mode: σ(t) = noise standard deviation
- Supports cosine, linear, and custom schedules

**Interface**:
```python
class NoiseScheduler:
    def timesteps(self, num_inference_steps):
        """Returns inference timestep sequence"""
    
    def add_noise(self, x_0, t, mode="continuous"):
        """Forward noising"""
    
    def sigma(self, t):
        """Continuous noise std σ(t)"""
    
    def mask_rate(self, t):
        """Discrete mask probability β(t)"""
```

### 3.3 MemoryConditioner

**Purpose**: Convert ArrowStorage retrieval results into diffusion conditions.

**Implementation**: 🧠 Python (uses 🦴 Rust ArrowStorage backend)

**Key Features**:
- Query Rust ArrowStorage with user prompt (10-50x faster than Python)
- Extract top-K memory embeddings
- Project to condition dimension

**Interface**:
```python
class MemoryConditioner:
    def __init__(self):
        # Initialize Rust ArrowStorage via PyO3
        self.storage = ArrowStorage.new("./memory.arrow")
    
    def get_condition(self, query, top_k=5):
        """
        Retrieve relevant memories and project to condition vectors.
        
        Uses Rust ArrowStorage for SIMD-accelerated vector search.
        
        Returns:
            condition: [K, condition_dim] condition matrix
        """
```

### 3.5 UncertaintyEstimator

**Purpose**: Measure uncertainty based on denoising residuals to trigger self-evolution.

**Key Features**:
- Compute denoising residual norm
- Normalize by expected noise level
- Trigger evolution when uncertainty exceeds threshold

**Interface**:
```python
class UncertaintyEstimator:
    def estimate(self, x_t, x_0_pred, t):
        """
        Compute uncertainty at current denoising step.
        
        High: Model uncertain, should trigger evolution
        Low: Model confident, normal output
        """
    
    def should_evolve(self, uncertainty, threshold=1.5):
        """Trigger evolution when uncertainty exceeds threshold"""
```

### 3.6 EvolutionRouter

**Purpose**: 5-level progressive self-evolution hub combining ControlNet, LoRA, and energy constraints.

**Key Features**:
- L0: Score composition (real-time mixing)
- L1: ControlNet injection (~10% params)
- L2: LoRA fine-tuning (~1% params)
- L3: Selective backbone fine-tuning
- L4: Full fine-tuning

**Interface**:
```python
class EvolutionRouter:
    def get_fused_score(self, x_t, t, modality, condition, active_profiles):
        """
        L0 Score composition: ∇log p_final = ∇log p_base + α∇log p_control - η∇E
        """
```


## 4. Data Flow Design

### 4.1 Text Generation Flow

```
"Write a poem about spring"
    │
    ▼ MemoryConditioner
ArrowStorage.search("spring") → [Memory: last year spring trip, spring poems read]
    │
    ▼ condition = projector(memory_vectors)
    │
    ▼ DiscreteSampler initialization
x_T = [MASK] [MASK] [MASK] [MASK] ... [MASK]  (L=128)
    │
    ▼ 4-step denoising (Consistency Distillation)
    │  t=4: score_net(x_4, t=4, "text", condition)
    │       → unmask 20% highest confidence positions
    │  t=3: score_net(x_3, t=3, "text", condition)
    │       → unmask another 30%
    │  t=2 → t=1 → ...
    ▼
x_0 = "Spring breeze caresses, flowers bloom, memories of cherry blossoms..."
```

### 4.2 Multimodal Parallel Generation Flow (Virtual Avatar Scenario)

```
User voice input: "Help me recall my last trip to Japan"
    │
    ▼ Unified condition preparation
condition = MemoryConditioner.get_condition("Japan trip")
    │
    ▼ Single SharedTransformer forward
hidden_states = shared_transformer(x_t, t, condition)
    │
    ├── TextHead.decode(h) → "Last time in Japan was..."     (Text reply)
    ├── AudioHead.decode(h) → [16kHz waveform]                (Voice synthesis)
    ├── AvatarHead.decode(h) → [blendshape params]            (Lip sync + expression)
    └── ImageHead.decode(h) → [512×512 image]                 (Related photo)
    
    ▲ All outputs naturally synchronized, no cascading delays
```

## 5. Storage Design

### 5.1 Model Directory Structure

```
models/
  diffusion-base/                   # Unified diffusion base
    metadata.json                   # Contains diffusion config
    shared_transformer.parquet      # Shared backbone weights (🦴 Rust ArrowQuant INT2)
    tokenizer/                      # 🦴 Rust FastTokenizer
    heads/
      text_head.parquet             # Text projection head (🦴 Rust ArrowQuant)
      code_head.parquet             # Code projection head
      image_head.parquet            # Image projection head
      audio_head.parquet            # Audio projection head
    vae/                            # Image VAE (optional)
      encoder.parquet
      decoder.parquet
  lora_cards/                       # 🧠 Python LoRA adaptations
    writing_style.parquet
    code_python.parquet
  controlnets/                      # 🧠 Python ControlNets
    cot_control.parquet             # Chain-of-thought ControlNet
    tool_schema_control.parquet     # Tool schema ControlNet
```

**Note**: All `.parquet` files use Rust ArrowQuant for quantization and zero-copy loading via PyO3 bindings.

### 5.2 metadata.json Schema

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

## 6. ArrowEngine Integration

### 6.1 Extended API

```python
class ArrowEngine:
    # Existing interfaces (unchanged)
    def encode(self, sentences, **kwargs): ...
    def generate(self, prompt, **kwargs): ...
    
    # New unified diffusion interface
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
        Unified diffusion generation.
        
        Args:
            prompt: Input prompt
            modality: Target generation modality
            num_steps: Denoising steps (1-50, default 4 for consistency)
            guidance_scale: Classifier-Free Guidance strength
            memory_guided: Whether to use ArrowStorage memory conditioning
        """
```

## 7. Implementation Roadmap

### Phase 3a: Discrete Diffusion Text PoC (2 weeks)

| Step | Task | Output |
|------|------|--------|
| S1 | Implement DiffusionCore + NoiseScheduler + DiscreteSampler | Core inference framework |
| S2 | Implement TextProjectionHead | Text projection |
| S3 | Convert open-source MDLM weights → Parquet V2 + ArrowQuant | Model files |
| S4 | Implement ArrowEngine.diffuse(modality="text") | API integration |
| S5 | Infilling quality validation + latency benchmark | Validation report |

### Phase 3b: Unified Score Network + Memory Conditioning (2 weeks)

| Step | Task | Output |
|------|------|--------|
| S6 | Implement UnifiedScoreNetwork (Shared Transformer) | Unified backbone |
| S7 | Implement MemoryConditioner (ArrowStorage → condition vectors) | Memory guidance |
| S8 | Implement UncertaintyEstimator (uncertainty-driven evolution trigger) | Evolution loop |
| S9 | Cross-modal end-to-end testing | Integration validation |

### Phase 3c: Image/Audio Diffusion + Virtual Avatar (3 weeks)

| Step | Task | Output |
|------|------|--------|
| S10 | Implement ImageProjectionHead + VAE integration | Image generation |
| S11 | Implement AudioProjectionHead (WaveGrad) | Audio generation |
| S12 | Consistency Distillation training (4-step compression) | Accelerated model |
| S13 | Multimodal parallel generation validation (virtual avatar scenario) | Scenario validation |
| S14 | Edge deployment validation (ARM + INT2) | Deployment validation |

## 8. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Open-source MDLM weights insufficient quality | Medium | High | Fallback to SEDD; or self-train from BERT weights |
| Quality degradation after Consistency Distillation | Medium | Medium | Keep multi-step inference as high-quality fallback |
| Cross-modal interference in unified backbone | Medium | High | Train heads independently initially, unfreeze backbone in stages |
| Edge device compute insufficient | Low | Medium | INT2 + minimal backbone (~50M) subset deployment |

## 9. Synergy with Rust Infrastructure

Unified Diffusion Architecture forms perfect closed loop with Rust Skeleton:

```
🦴 Rust ArrowQuant INT2 quantization
        ↓ Compress weights (5-10x speedup)
🧠 Python UnifiedScoreNetwork.shared_transformer
        ↓ ~200MB (INT2), loaded via PyO3
🦴 Rust ArrowStorage vector search
        ↓ Memory retrieval (10-50x speedup)
🧠 Python MemoryConditioner
        ↓ Condition vectors for generation
🧠 Python DiffusionCore
        ↓ Generates outputs
🦴 Rust ArrowStorage persistence
        ↓ Store adaptations for evolution
```

**Performance Benefits**:
- ArrowQuant provides storage infrastructure with 5-10x speedup
- ArrowStorage provides memory retrieval with 10-50x speedup
- FastTokenizer provides tokenization with 10-100x speedup
- Overall system achieves 50-70% latency reduction
- Edge device deployable (<35MB with INT2)

**Evolution Benefits**:
- Rust skeleton remains stable (infrastructure doesn't change)
- Python brain evolves freely (user-specific adaptations)
- PyO3 bindings provide seamless integration
- Users can experiment without touching Rust code


## 10. Diffusion Model Quantization Strategy

### 10.1 Quantization Challenges for Diffusion Models

Diffusion models present unique quantization challenges compared to traditional LLMs:

| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Temporal Variance** | Activation distributions vary significantly across denoising timesteps | Fixed quantization parameters cause precision loss |
| **Spatial Variance** | Different channels have vastly different weight/activation ranges | Per-tensor quantization is too coarse |
| **Error Accumulation** | Iterative denoising amplifies quantization errors | Generation quality degrades significantly |
| **Discrete Diffusion Specificity** | Dream 7B uses mask-based denoising (discrete) vs Gaussian noise (continuous) | Mask operations are highly sensitive to quantization |

### 10.2 Current ArrowQuant Capabilities

**✅ Existing Features**:
- INT2/INT4 quantization with bit packing
- Symmetric/asymmetric quantization
- Rayon parallel processing
- Zero-copy Parquet loading via PyO3
- Per-tensor scaling

**⚠️ Limitations for Diffusion Models**:
- No per-channel quantization (spatial variance)
- No dynamic activation quantization (temporal variance)
- No time-aware quantization strategies
- Basic quantization algorithm (not SOTA)

### 10.3 State-of-the-Art Diffusion Quantization (2025-2026)

Based on latest research (CVPR 2025, arXiv 2025), we identify three key techniques:

#### 10.3.1 Q-DiT (CVPR 2025) - Diffusion Transformer Quantization

**Source**: https://github.com/Juanerx/Q-DiT

**Core Techniques**:
1. **Dynamic Activation Quantization**: Adapts quantization parameters per timestep
   ```python
   # Different quantization params for different timesteps
   scale_t = self.timestep_scales[t]
   quantized_act = quantize(activation, scale_t, zero_point_t)
   ```

2. **Automatic Granularity Allocation**: Uses evolutionary search to find optimal quantization granularity per layer
   ```python
   # Layer-wise group size optimization
   group_sizes = evolutionary_search(model, calibration_data)
   # e.g., attention layers: group_size=128, FFN layers: group_size=64
   ```

3. **Group-wise Quantization**: Divides channels into groups with independent scales
   ```python
   # Per-group scaling instead of per-tensor
   for group in split_channels(weights, group_size):
       scale, zp = compute_params(group)
       quantized_group = quantize(group, scale, zp)
   ```

**Performance**: W4A8 quantization with FID increase <0.1 (nearly lossless)

#### 10.3.2 DiTAS (arXiv 2024) - Activation Smoothing

**Source**: https://github.com/DZY122/DiTAS

**Core Techniques**:
1. **Activation Smoothing**: Reduces activation variance before quantization
2. **Channel Equalization**: Balances different channels to reduce spatial variance

**Performance**: W4A8 quantization maintaining FP16-equivalent quality

#### 10.3.3 Time-Aware Quantization (arXiv 2025)

**Core Techniques**:
1. **Time-Grouping Quantization (TGQ)**: Groups similar timesteps with shared quantization params
2. **Multi-Region Quantization (MRQ)**: Handles asymmetric activation distributions

**Performance**: 60% computation reduction for DiT models

### 10.4 Quantization Strategy for Dream 7B

Dream 7B is a **discrete diffusion model** with unique characteristics:

| Aspect | Image Diffusion (DiT) | Text Diffusion (Dream 7B) |
|--------|----------------------|---------------------------|
| **Noise Type** | Gaussian (continuous) | Mask (discrete) |
| **Denoising** | Langevin dynamics | CTMC jump process |
| **Activation Distribution** | Continuous | Discrete + Sparse |
| **Quantization Difficulty** | Medium | **Higher** (discreteness) |

**Recommended Approach**:
- ✅ Transformer backbone: Use Q-DiT techniques
- ⚠️ Discrete sampler: Special handling for mask operations
- ✅ Projection heads: Standard quantization

### 10.5 ArrowQuant V2 Enhancement Plan

#### Phase 1: Per-Channel Quantization (2 weeks)

```rust
// Rust implementation
impl ArrowQuant {
    pub fn quantize_per_channel(
        &self,
        weights: &Array2<f32>,
        axis: usize,
    ) -> PyResult<QuantizedTensor> {
        // Compute scale/zero_point per output channel
        let mut scales = Vec::new();
        let mut zero_points = Vec::new();
        
        for channel in weights.axis_iter(Axis(axis)) {
            let (scale, zp) = self.compute_quantization_params(channel)?;
            scales.push(scale);
            zero_points.push(zp);
        }
        
        // Quantize with per-channel params
        // ...
    }
}
```

#### Phase 2: Dynamic Activation Quantization (2 weeks)

```python
# Python strategy layer
class DynamicActivationQuant:
    def __init__(self, num_timesteps=1000):
        # Pre-compute quantization params for each timestep
        self.timestep_params = self._calibrate_timesteps()
    
    def quantize_activation(self, act, timestep):
        # Select params based on timestep
        scale, zp = self.timestep_params[timestep]
        return self.arrow_quant.quantize(act, scale, zp)
    
    def _calibrate_timesteps(self):
        # Calibrate on sample data across all timesteps
        # Group similar timesteps together
        pass
```

#### Phase 3: Q-DiT Integration (3 weeks)

```python
# Integrate Q-DiT evolutionary search
class QDiTQuantizer:
    def __init__(self, model, calibration_data):
        self.model = model
        self.calibration_data = calibration_data
        self.arrow_quant = ArrowQuant.new(bit_width=4)
    
    def find_optimal_granularity(self):
        # Evolutionary search for layer-wise group sizes
        population = self._initialize_population()
        
        for generation in range(num_generations):
            # Evaluate FID for each configuration
            fitness = self._evaluate_population(population)
            # Select and mutate
            population = self._evolve(population, fitness)
        
        return best_configuration
    
    def quantize_model(self, config):
        # Apply optimal configuration
        for layer_name, group_size in config.items():
            layer = self.model.get_layer(layer_name)
            quantized = self.arrow_quant.quantize_per_channel(
                layer.weight, group_size=group_size
            )
            layer.weight = quantized
```

### 10.6 Quantization Usage Modes

#### Mode 1: Offline Quantization (Recommended for Production)

```python
# One-time conversion during deployment preparation
python scripts/convert_diffusion_weights.py \
    --model dream-org/Dream-v0-7B \
    --output models/dream-7b-int2/ \
    --quantizer q-dit \
    --bit-width 2

# Deployment: Load pre-quantized model (no quantizer needed)
engine = ArrowEngine(model_path="models/dream-7b-int2/")
# Zero-copy loading, ~200ms startup
```

**Advantages**:
- ✅ No quantizer in deployment (minimal memory)
- ✅ Fast startup (~200ms)
- ✅ Suitable for edge devices

#### Mode 2: Online Quantization (Recommended for Development)

```python
# Dynamic quantization for LoRA/ControlNet hot-swapping
engine = ArrowEngine(config)
engine.arrow_quant = ArrowQuant.new(bit_width=2)  # Load quantizer

# Dynamically quantize LoRA
lora_weights = load_lora("writing_style.safetensors")
quantized_lora = engine.arrow_quant.quantize(lora_weights)
engine.apply_lora(quantized_lora)
```

**Advantages**:
- ✅ High flexibility
- ✅ Supports hot-swapping
- ⚠️ Slightly higher memory (~50MB for quantizer)

#### Mode 3: Hybrid Mode (Best Practice)

```python
class ArrowEngine:
    def __init__(self, config):
        # Base model: Pre-quantized (offline)
        self.base_model = self.load_quantized_model(
            "models/dream-7b-int2/shared_transformer.parquet"
        )
        
        # Quantizer: Load on-demand
        self.arrow_quant = None
    
    def apply_lora(self, lora_path):
        # Load quantizer only when needed
        if self.arrow_quant is None:
            self.arrow_quant = ArrowQuant.new(bit_width=2)
        
        # Dynamically quantize LoRA
        lora_weights = load_weights(lora_path)
        quantized = self.arrow_quant.quantize(lora_weights)
        self.lora_router.add(quantized)
```

**Resource Comparison**:

| Mode | Memory | Startup | Flexibility | Use Case |
|------|--------|---------|-------------|----------|
| Offline | 1.75GB | ~200ms | Low | Production, Edge |
| Online | 1.8GB | ~300ms | High | Development |
| Hybrid | 1.75GB + on-demand | ~200ms | Medium-High | **Recommended** ✅ |

### 10.7 Performance Expectations

**Current (Basic Quantization)**:
- Compression: 16x (FP32 → INT2)
- Speedup: 5-10x (Rust vs Python)
- Quality: ~5-10% degradation

**After Q-DiT Integration**:
- Compression: 16x (FP32 → INT2)
- Speedup: 10-20x (SOTA algorithms + Rust)
- Quality: <1% degradation (W4A8), <3% (W4A4)

**Deployment Targets**:

| Device Tier | Model Size | Quantization | Modalities | Latency |
|-------------|-----------|--------------|------------|---------|
| Edge (2-4GB RAM) | <35MB | INT2 | Text, Audio | <500ms |
| Local (8+GB RAM) | <200MB | INT4 | Text, Audio, Image | <2s |
| Cloud (32+GB RAM) | <2GB | INT2/INT4 | All | <1s |

## 11. Multimodal Evolution Strategy

### 11.1 Core Concept: Unified Score Network + Modality Heads

Dream 7B is a text diffusion model, but can evolve to support multimodal generation through our architecture:

```
Dream 7B (Text Diffusion)
    ↓ Preserve Transformer backbone (90% params)
SharedTransformer (Unified Score Network)
    ↓ Add modality projection heads (10% params)
    ├─ TextHead (existing, from Dream 7B)
    ├─ ImageHead (new, <10M params) ← Add via LoRA/Fine-tuning
    ├─ AudioHead (new, <10M params) ← Add via LoRA/Fine-tuning
    └─ CodeHead (new, <10M params) ← Add via LoRA/Fine-tuning
```

**Key Advantages**:
- ✅ **Parameter Efficiency**: New modality requires <10M params (<1% of total)
- ✅ **Knowledge Transfer**: Shared backbone enables cross-modal knowledge reuse
- ✅ **Progressive Evolution**: No need to retrain entire model

### 11.2 Five-Level Evolution Strategy (L0-L4)

| Level | Technique | Params | Training Cost | Use Case |
|-------|-----------|--------|---------------|----------|
| **L0** | Score Composition | 0% | None | Real-time model mixing |
| **L1** | ControlNet | ~10% | Low | Structural constraints (CoT, JSON) |
| **L2** | LoRA | ~1% | Medium | Domain knowledge injection |
| **L3** | Selective Fine-tuning | ~20% | High | Partial layer adaptation |
| **L4** | Full Fine-tuning | 100% | Very High | Long-term consolidation |

### 11.3 Multimodal Evolution Examples

#### Example 1: Text → Image Generation (L1 ControlNet)

```python
# Step 1: Preserve Dream 7B Transformer backbone
shared_transformer = dream_7b.transformer  # 7B params

# Step 2: Add ImageProjectionHead (<10M params)
image_head = ImageProjectionHead(
    hidden_dim=4096,
    output_dim=512*512*3,
    params="~8M"
)

# Step 3: Train ControlNet (freeze shared_transformer)
image_controlnet = ControlNet(
    base=shared_transformer,
    head=image_head,
    params="~10M"
)

train(image_controlnet, laion_400m_subset, epochs=10)

# Result:
# - Total params: 7B + 8M + 10M = 7.018B (+0.26%)
# - Training: Only 18M params (<0.3%)
# - Inference: engine.diffuse(prompt, modality="image")
```

#### Example 2: Text → Audio Generation (L2 LoRA)

```python
# Step 1: Preserve Dream 7B Transformer backbone
shared_transformer = dream_7b.transformer  # 7B params

# Step 2: Add AudioProjectionHead (<10M params)
audio_head = AudioProjectionHead(
    hidden_dim=4096,
    output_dim=16000*10,  # 10s audio, 16kHz
    params="~6M"
)

# Step 3: Train LoRA (freeze shared_transformer)
audio_lora = LoRA(
    base=shared_transformer,
    head=audio_head,
    rank=16,
    params="~700K"
)

train(audio_lora, musiccaps_dataset, epochs=5)

# Result:
# - Total params: 7B + 6M + 0.7M = 7.0067B (+0.01%)
# - Training: Only 6.7M params (<0.1%)
# - Inference: engine.diffuse(prompt, modality="audio")
```

#### Example 3: Multimodal Parallel Generation (L0 Score Composition)

```python
# No training required - real-time model mixing
engine = ArrowEngine(config)

# Load multiple modality heads
engine.load_head("text", dream_7b_text_head)
engine.load_head("image", stable_diffusion_head)
engine.load_head("audio", audioldm_head)

# Single forward pass, generate multiple modalities
result = engine.diffuse(
    prompt="春天的樱花",
    modalities=["text", "image", "audio"],
    num_steps=4
)

# Output:
# - result["text"]: "春风拂面，樱花飘落..."
# - result["image"]: [512x512 cherry blossom image]
# - result["audio"]: [16kHz spring music]

# Key: Single SharedTransformer forward pass, all modalities share
```

### 11.4 Evolution Strategy Selection Guide

| Goal | Strategy | Params | Training Time | Quality |
|------|----------|--------|---------------|---------|
| **Quick Prototype** | L0 Score Composition | 0% | 0 | Medium |
| **Add Constraints** | L1 ControlNet | ~10% | 1-2 days | Medium-High |
| **Domain Adaptation** | L2 LoRA | ~1% | Hours | High |
| **Deep Customization** | L3 Selective FT | ~20% | 1-2 weeks | Very High |
| **Complete Retraining** | L4 Full FT | 100% | Weeks | Highest |

**Recommended Progressive Path**:
```
1. L0 Validate feasibility (no training)
   ↓ If insufficient
2. L1 Add ControlNet (low cost)
   ↓ If need stronger adaptation
3. L2 Train LoRA (medium cost)
   ↓ If need deep customization
4. L3 Selective Fine-tuning (high cost)
   ↓ Only if necessary
5. L4 Full Fine-tuning (very high cost)
```

### 11.5 Implementation Architecture

```python
class MultimodalEvolutionEngine:
    def __init__(self, base_model="dream-7b"):
        # Load base model (text diffusion)
        self.shared_transformer = load_model(base_model)
        
        # Initialize evolution router
        self.evolution_router = EvolutionRouter()
        
        # Modality heads registry
        self.heads = {
            "text": dream_7b.text_head  # Existing
        }
    
    def add_modality(self, modality, strategy="L2", **kwargs):
        """
        Add new modality support via evolution strategy.
        
        Args:
            modality: "image", "audio", "video", etc.
            strategy: "L0", "L1", "L2", "L3", "L4"
        """
        if strategy == "L0":
            # Score composition - no training
            external_head = kwargs["external_head"]
            self.heads[modality] = external_head
            
        elif strategy == "L1":
            # ControlNet - train lightweight adapter
            head = self._create_projection_head(modality)
            controlnet = ControlNet(self.shared_transformer, head)
            self._train_controlnet(controlnet, kwargs["dataset"])
            self.heads[modality] = head
            
        elif strategy == "L2":
            # LoRA - train low-rank adaptation
            head = self._create_projection_head(modality)
            lora = LoRA(self.shared_transformer, head, rank=16)
            self._train_lora(lora, kwargs["dataset"])
            self.heads[modality] = head
            
        elif strategy == "L3":
            # Selective fine-tuning
            head = self._create_projection_head(modality)
            self._selective_finetune(
                self.shared_transformer, 
                head, 
                kwargs["dataset"],
                unfreeze_layers=kwargs.get("unfreeze_layers", [10,11,12])
            )
            self.heads[modality] = head
            
        elif strategy == "L4":
            # Full fine-tuning
            head = self._create_projection_head(modality)
            self._full_finetune(
                self.shared_transformer,
                head,
                kwargs["dataset"]
            )
            self.heads[modality] = head
    
    def generate(self, prompt, modality, **kwargs):
        """Unified generation interface."""
        # Route to appropriate head
        head = self.heads[modality]
        
        # Shared transformer forward pass
        hidden_states = self.shared_transformer(
            prompt, 
            modality_embedding=head.modality_id
        )
        
        # Modality-specific decoding
        output = head.decode(hidden_states)
        return output
```

### 11.6 Quantization Integration with Evolution

**Key Insight**: Quantization strategy must adapt to evolution level:

| Evolution Level | Quantization Strategy |
|----------------|----------------------|
| **L0** | Use pre-quantized models (offline) |
| **L1** | Quantize ControlNet weights (online) |
| **L2** | Quantize LoRA weights (online) |
| **L3** | Re-quantize fine-tuned layers (offline) |
| **L4** | Full model re-quantization (offline) |

```python
class QuantizationAwareEvolution:
    def add_modality_with_quantization(self, modality, strategy, bit_width=4):
        # Train new modality head
        head = self.add_modality(modality, strategy)
        
        # Quantize based on strategy
        if strategy in ["L1", "L2"]:
            # Online quantization for adapters
            quantized_head = self.arrow_quant.quantize(
                head.state_dict(),
                bit_width=bit_width
            )
            self.heads[modality] = quantized_head
            
        elif strategy in ["L3", "L4"]:
            # Offline quantization for fine-tuned models
            # Use Q-DiT for optimal quantization
            quantized_model = self.q_dit_quantizer.quantize(
                self.shared_transformer,
                calibration_data=self.calibration_data,
                bit_width=bit_width
            )
            self.shared_transformer = quantized_model
```

### 11.7 Performance Expectations

**Parameter Efficiency**:
- Base model (Dream 7B): 7B params
- Add image modality (L1): +18M params (+0.26%)
- Add audio modality (L2): +6.7M params (+0.01%)
- Add video modality (L3): +50M params (+0.7%)
- **Total multimodal**: 7.075B params (+1.07%)

**Memory Footprint** (with INT2 quantization):
- Base model: 1.75GB
- Image head: 4.5MB
- Audio head: 1.7MB
- Video head: 12.5MB
- **Total**: 1.77GB (+1.1%)

**Inference Latency** (4-step consistency distillation):
- Text generation: <500ms (CPU)
- Image generation: <2s (CPU)
- Audio generation: <1s (CPU)
- Multimodal parallel: <2.5s (CPU, all modalities)


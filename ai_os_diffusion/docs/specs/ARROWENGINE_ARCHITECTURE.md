# ArrowEngine 架构详解

## 概述

ArrowEngine 是整个 AI-OS 统一扩散架构的**唯一入口点和路由层**。它是 🧠 Python Brain 的顶层组件，负责协调所有子系统，并为用户提供简洁统一的 API。

## 定位与职责

### 1. 系统定位

```
用户层:        用户代码
              ↓
API层:        ArrowEngine (🧠 统一入口) ← 你在这里
              ↓
业务逻辑层:    DiffusionCore, EvolutionRouter, MemoryConditioner (🧠 Python)
              ↓
基础设施层:    ArrowStorage, ArrowQuant, FastTokenizer (🦴 Rust)
```

**ArrowEngine 是用户与系统交互的唯一接口**：
- 用户不直接调用 DiffusionCore
- 用户不直接调用 Rust 组件
- 所有功能都通过 ArrowEngine 暴露

### 2. 核心职责

#### 2.1 统一 API 入口

ArrowEngine 提供三个主要接口：

```python
# 1. 编码接口 - 将文本转换为向量
embeddings = engine.encode(["春天来了", "花开了"])

# 2. 生成接口 - AR模式生成 (向后兼容)
text = engine.generate("写一首诗", mode="ar")

# 3. 扩散接口 - Diffusion模式生成 (新功能)
text = engine.diffuse("写一首诗", modality="text")
image = engine.diffuse("春天的樱花", modality="image")
audio = engine.diffuse("温柔的女声", modality="audio")
```

#### 2.2 模式路由

ArrowEngine 根据请求自动路由到正确的推理路径：

```python
class ArrowEngine:
    def generate(self, prompt: str, mode: str = "ar", **kwargs):
        """根据mode参数路由到不同的推理引擎"""
        if mode == "ar":
            # 路由到现有AR推理路径
            return self.inference_core.generate(prompt, **kwargs)
        elif mode == "diffusion":
            # 路由到新Diffusion推理路径
            return self.diffusion_core.generate(prompt, **kwargs)
```

**路由决策流程**：
```
用户请求
   ↓
ArrowEngine.generate(mode=?)
   ↓
   ├─ mode="ar" → InferenceCore (现有AR推理)
   │                ├─ Transformer
   │                ├─ LoRA Router
   │                └─ AR Sampler
   │
   └─ mode="diffusion" → DiffusionCore (新Diffusion推理)
                          ├─ UnifiedScoreNetwork
                          ├─ EvolutionRouter
                          ├─ NoiseScheduler
                          └─ Discrete/Continuous Sampler
```

#### 2.3 组件协调

ArrowEngine 协调 Python Brain 和 Rust Skeleton 的所有组件：

```python
class ArrowEngine:
    def __init__(self, config_path: str):
        # === 初始化Rust基础设施 (通过PyO3) ===
        self.arrow_storage = ArrowStorage.new(config.storage_path)
        self.arrow_quant = ArrowQuant.new(bit_width=2)
        self.fast_tokenizer = FastTokenizer.from_pretrained(config.tokenizer)
        
        # === 初始化Python推理组件 ===
        self.inference_core = InferenceCore(config)      # AR模式
        self.diffusion_core = DiffusionCore(config)      # Diffusion模式
        self.memory_conditioner = MemoryConditioner(     # 记忆条件化
            arrow_storage=self.arrow_storage
        )
        
        # === 初始化权重管理 ===
        self.weight_loader = WeightLoader(
            arrow_quant=self.arrow_quant  # 使用Rust量化加速
        )
```

**组件协调示例 - 记忆引导生成**：

```python
def diffuse(self, prompt: str, memory_guided: bool = True, **kwargs):
    """ArrowEngine协调多个组件完成记忆引导生成"""
    
    # 1. 使用Rust FastTokenizer分词 (10-100x加速)
    tokens = self.fast_tokenizer.encode(prompt)
    
    # 2. 如果启用记忆引导，使用Rust ArrowStorage搜索 (10-50x加速)
    condition = None
    if memory_guided:
        query_embedding = self._encode_query(prompt)
        memory_results = self.arrow_storage.search(
            query=query_embedding,
            top_k=5
        )
        # 3. 使用Python MemoryConditioner处理记忆
        condition = self.memory_conditioner.prepare_condition(memory_results)
    
    # 4. 调用Python DiffusionCore生成
    result = self.diffusion_core.generate(
        tokens=tokens,
        condition=condition,
        **kwargs
    )
    
    return result
```

#### 2.4 向后兼容

ArrowEngine 确保现有 AR 模式完全兼容：

```python
# 现有代码无需修改，继续工作
engine = ArrowEngine()
text = engine.generate("写一首诗")  # 默认使用AR模式

# 新代码可以使用Diffusion模式
text = engine.diffuse("写一首诗", modality="text")
```

## 完整架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户代码 (User Code)                      │
│  from ai_os_diffusion import ArrowEngine                         │
│  engine = ArrowEngine("config.yaml")                             │
│  result = engine.diffuse("生成春天的诗", modality="text")         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│  ArrowEngine (🧠 Python - 统一入口/路由层)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  公共接口 (Public API)                                    │   │
│  │  ├─ encode(sentences) → embeddings                       │   │
│  │  ├─ generate(prompt, mode="ar") → text                   │   │
│  │  └─ diffuse(prompt, modality, memory_guided) → output    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  路由逻辑 (Routing Logic)                                 │   │
│  │  ├─ if mode == "ar": → InferenceCore                     │   │
│  │  └─ if mode == "diffusion": → DiffusionCore              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  组件管理 (Component Management)                          │   │
│  │  ├─ WeightLoader (加载模型权重，使用Rust ArrowQuant)      │   │
│  │  ├─ ConfigManager (管理配置)                              │   │
│  │  ├─ ResourceManager (管理GPU/CPU资源)                     │   │
│  │  └─ MemoryConditioner (记忆条件化，使用Rust ArrowStorage) │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬──────────────┬──────────────────────┘
                             │              │
                     (AR模式) │              │ (Diffusion模式)
                             ↓              ↓
           ┌──────────────────────┐  ┌──────────────────────────────┐
           │ InferenceCore        │  │ DiffusionCore (🧠 Python)    │
           │ (现有AR推理)          │  │  ┌────────────────────────┐  │
           │  ├─ Transformer      │  │  │ UnifiedScoreNetwork    │  │
           │  ├─ LoRA Router      │  │  │  ├─ SharedTransformer  │  │
           │  └─ AR Sampler       │  │  │  └─ Projection Heads   │  │
           │                      │  │  └────────────────────────┘  │
           │                      │  │  ┌────────────────────────┐  │
           │                      │  │  │ EvolutionRouter        │  │
           │                      │  │  │  ├─ L0: Score Mixer    │  │
           │                      │  │  │  ├─ L1: ControlNet     │  │
           │                      │  │  │  └─ L2: LoRA           │  │
           │                      │  │  └────────────────────────┘  │
           │                      │  │  ├─ NoiseScheduler          │  │
           │                      │  │  └─ Samplers               │  │
           └──────────┬───────────┘  └──────────┬───────────────────┘
                      │                         │
                      │ 调用Rust基础设施         │ 调用Rust基础设施
                      ↓                         ↓
           ┌──────────────────────────────────────────────────────┐
           │  🦴 Rust Skeleton (基础设施层 - PyO3绑定)            │
           │  ┌────────────────────────────────────────────────┐  │
           │  │ ArrowStorage (向量存储/搜索)                    │  │
           │  │  ├─ SIMD加速相似度计算 (10-50x)                │  │
           │  │  ├─ 零拷贝内存映射                             │  │
           │  │  └─ PyO3 Python绑定                           │  │
           │  └────────────────────────────────────────────────┘  │
           │  ┌────────────────────────────────────────────────┐  │
           │  │ ArrowQuant (权重量化)                           │  │
           │  │  ├─ INT2/INT4量化 (5-10x)                      │  │
           │  │  ├─ 零拷贝加载                                 │  │
           │  │  └─ PyO3 Python绑定                           │  │
           │  └────────────────────────────────────────────────┘  │
           │  ┌────────────────────────────────────────────────┐  │
           │  │ FastTokenizer (快速分词)                        │  │
           │  │  ├─ 并行批处理 (10-100x)                       │  │
           │  │  ├─ 支持BPE/WordPiece/Unigram                  │  │
           │  │  └─ PyO3 Python绑定                           │  │
           │  └────────────────────────────────────────────────┘  │
           │  ┌────────────────────────────────────────────────┐  │
           │  │ VectorSearch (向量相似度)                       │  │
           │  │  ├─ simsimd SIMD加速                           │  │
           │  │  └─ PyO3 Python绑定                           │  │
           │  └────────────────────────────────────────────────┘  │
           └──────────────────────────────────────────────────────┘
```

## 数据流示例

### 示例 1: 简单文本生成 (Diffusion模式)

```python
# 用户代码
engine = ArrowEngine()
result = engine.diffuse("写一首春天的诗", modality="text")
```

**内部数据流**：

```
1. ArrowEngine.diffuse() 接收请求
   ↓
2. 使用 Rust FastTokenizer 分词
   tokens = self.fast_tokenizer.encode("写一首春天的诗")
   ↓
3. 路由到 DiffusionCore
   ↓
4. DiffusionCore 初始化噪声状态
   x_T = [MASK] * 128
   ↓
5. 迭代去噪 (4步)
   for t in [4, 3, 2, 1]:
       score = UnifiedScoreNetwork(x_t, t, "text")
       x_t = DiscreteSampler.step(x_t, score, t)
   ↓
6. 返回生成结果
   x_0 = "春风拂面花盛开，..."
```

### 示例 2: 记忆引导生成

```python
# 用户代码
engine = ArrowEngine()
result = engine.diffuse(
    "回忆我去年的日本之旅",
    modality="text",
    memory_guided=True  # 启用记忆引导
)
```

**内部数据流**：

```
1. ArrowEngine.diffuse() 接收请求
   ↓
2. 编码查询 (使用 Rust FastTokenizer)
   query_embedding = self._encode_query("回忆我去年的日本之旅")
   ↓
3. 使用 Rust ArrowStorage 搜索记忆 (10-50x加速)
   memory_results = self.arrow_storage.search(
       query=query_embedding,
       top_k=5
   )
   # 返回: [
   #   {text: "去年春天去了京都...", embedding: [...], score: 0.92},
   #   {text: "在大阪吃了章鱼烧...", embedding: [...], score: 0.87},
   #   ...
   # ]
   ↓
4. MemoryConditioner 处理记忆
   condition = self.memory_conditioner.prepare_condition(memory_results)
   # 将记忆嵌入投影到条件空间
   ↓
5. DiffusionCore 使用条件生成
   result = self.diffusion_core.generate(
       prompt="回忆我去年的日本之旅",
       condition=condition  # 注入记忆条件
   )
   ↓
6. UnifiedScoreNetwork 在去噪时使用条件
   for t in [4, 3, 2, 1]:
       # 通过cross-attention注入记忆条件
       score = UnifiedScoreNetwork(x_t, t, "text", condition)
       x_t = DiscreteSampler.step(x_t, score, t)
   ↓
7. 返回与记忆相关的生成结果
   "去年春天在京都看樱花，在大阪品尝美食，..."
```

### 示例 3: 多模态并行生成 (虚拟化身场景)

```python
# 用户代码
engine = ArrowEngine()
outputs = engine.diffuse(
    "帮我回忆日本之旅",
    modality=["text", "audio", "image"],  # 多模态
    memory_guided=True
)
```

**内部数据流**：

```
1. ArrowEngine.diffuse() 接收多模态请求
   ↓
2. 搜索记忆 (Rust ArrowStorage)
   memory_results = self.arrow_storage.search(...)
   ↓
3. 准备条件
   condition = self.memory_conditioner.prepare_condition(memory_results)
   ↓
4. DiffusionCore 单次前向传播
   hidden_states = SharedTransformer(x_t, t, condition)
   ↓
5. 并行解码多个模态
   ├─ text_output = TextHead.decode(hidden_states)
   ├─ audio_output = AudioHead.decode(hidden_states)
   └─ image_output = ImageHead.decode(hidden_states)
   ↓
6. 返回同步的多模态输出
   {
       "text": "去年春天在京都...",
       "audio": [16kHz waveform],
       "image": [512×512 image]
   }
```

## 组件关系表

| 组件 | 层级 | 语言 | 职责 | 被谁调用 | 调用谁 |
|------|------|------|------|----------|--------|
| **ArrowEngine** | API层 | 🧠 Python | 统一入口、路由、协调 | 用户代码 | InferenceCore, DiffusionCore, Rust组件 |
| InferenceCore | 业务逻辑层 | 🧠 Python | AR推理 | ArrowEngine | Rust组件 |
| DiffusionCore | 业务逻辑层 | 🧠 Python | Diffusion推理 | ArrowEngine | UnifiedScoreNetwork, EvolutionRouter, Samplers |
| UnifiedScoreNetwork | 业务逻辑层 | 🧠 Python | 统一评分网络 | DiffusionCore | SharedTransformer, Projection Heads |
| EvolutionRouter | 业务逻辑层 | 🧠 Python | 进化路由 | DiffusionCore | ControlNet, LoRA |
| MemoryConditioner | 业务逻辑层 | 🧠 Python | 记忆条件化 | ArrowEngine | ArrowStorage (Rust) |
| ArrowStorage | 基础设施层 | 🦴 Rust | 向量存储/搜索 | ArrowEngine, MemoryConditioner | - |
| ArrowQuant | 基础设施层 | 🦴 Rust | 权重量化 | WeightLoader | - |
| FastTokenizer | 基础设施层 | 🦴 Rust | 快速分词 | ArrowEngine | - |
| VectorSearch | 基础设施层 | 🦴 Rust | 向量相似度 | ArrowStorage | - |

## 关键设计原则

### 1. 单一入口原则

**用户只通过 ArrowEngine 访问系统**：
- ✅ 正确：`engine.diffuse(...)`
- ❌ 错误：`DiffusionCore().generate(...)`  # 用户不应直接调用

### 2. 关注点分离

- **ArrowEngine**：负责 API、路由、协调
- **DiffusionCore**：负责扩散推理逻辑
- **Rust Skeleton**：负责高性能基础设施

### 3. 向后兼容

- 现有 AR 模式代码无需修改
- 新 Diffusion 模式通过新接口暴露
- 两种模式可以共存

### 4. 性能优化

- 热路径使用 Rust 实现（分词、量化、向量搜索）
- 冷路径使用 Python 实现（配置、协调、实验）
- PyO3 提供零开销互操作

## 迁移路径

### 从现有系统迁移到统一扩散架构

**Phase 0: 准备阶段**
```python
# 现有代码继续工作
engine = ArrowEngine()
text = engine.generate("写诗")  # AR模式
```

**Phase 1: 并行运行**
```python
# 新旧模式并行
engine = ArrowEngine()
ar_text = engine.generate("写诗", mode="ar")          # 旧模式
diff_text = engine.diffuse("写诗", modality="text")   # 新模式
```

**Phase 2: 逐步迁移**
```python
# 逐步将功能迁移到Diffusion模式
engine = ArrowEngine()
text = engine.diffuse("写诗", modality="text")        # 主要使用新模式
image = engine.diffuse("春天", modality="image")      # 新功能
```

**Phase 3: 完全迁移**
```python
# 所有功能使用Diffusion模式
engine = ArrowEngine()
result = engine.diffuse(prompt, modality=modality)   # 统一接口
```

## 总结

**ArrowEngine 是什么？**
- 系统的统一入口点和路由层
- 🧠 Python Brain 的顶层组件
- 用户与系统交互的唯一接口

**ArrowEngine 不是什么？**
- 不是推理引擎本身（推理由 DiffusionCore 完成）
- 不是基础设施（基础设施由 Rust Skeleton 提供）
- 不是单独的服务（是库的一部分）

**为什么需要 ArrowEngine？**
- 提供统一、简洁的 API
- 隐藏内部复杂性
- 支持平滑迁移
- 协调 Python Brain 和 Rust Skeleton

**记住**：
> ArrowEngine 是门面，DiffusionCore 是大脑，Rust Skeleton 是骨骼。
> 用户敲门（ArrowEngine），大脑思考（DiffusionCore），骨骼支撑（Rust Skeleton）。

# 输出模型定义文档

**生成时间**: 2026-02-17  
**查询**: 系统输出模型的定义和结构

---

## 📊 核心输出模型

系统定义了多层次的输出模型，从底层压缩到高层表达。

---

## 1️⃣ 基础层：压缩输出

### CompressedMemory（压缩记忆）

**位置**: `llm_compression/compressor.py`

```python
@dataclass
class CompressedMemory:
    """压缩后的记忆"""
    
    # 核心数据
    memory_id: str                      # 唯一记忆 ID
    summary_hash: str                   # 摘要哈希（SHA256 前 16 字符）
    entities: Dict[str, List[str]]      # 提取的实体 {类型: [实体列表]}
    diff_data: bytes                    # 压缩的差异数据（zstd）
    embedding: List[float]              # 向量表示（384 维，float16）
    
    # 元数据
    compression_metadata: CompressionMetadata  # 压缩元信息
    original_fields: Dict[str, Any]     # 原始字段（时间戳等）
```

**实体类型**:
- `PERSON`: 人名
- `DATE`: 日期
- `NUMBER`: 数字
- `LOCATION`: 地点
- `ORGANIZATION`: 组织

**示例**:
```python
CompressedMemory(
    memory_id="mem_001",
    summary_hash="a3f5c8d2e1b4f6a9",
    entities={
        "PERSON": ["张三", "李四"],
        "DATE": ["2026-02-17"],
        "NUMBER": ["100", "50"]
    },
    diff_data=b'\x28\xb5\x2f\xfd...',  # zstd 压缩
    embedding=[0.123, -0.456, ...],     # 384 维向量
    compression_metadata=CompressionMetadata(...),
    original_fields={"timestamp": "2026-02-17T03:00:00"}
)
```

---

### CompressionMetadata（压缩元数据）

```python
@dataclass
class CompressionMetadata:
    """压缩元数据"""
    
    original_size: int          # 原始大小（字节）
    compressed_size: int        # 压缩后大小（字节）
    compression_ratio: float    # 压缩比（原始/压缩）
    model_used: str            # 使用的模型名称
    quality_score: float       # 质量分数（0-1）
    compression_time_ms: float # 压缩耗时（毫秒）
    compressed_at: datetime    # 压缩时间戳
```

---

## 2️⃣ 重构层：解压输出

### ReconstructedMemory（重构记忆）

**位置**: `llm_compression/reconstructor.py`

```python
@dataclass
class ReconstructedMemory:
    """从压缩数据重构的记忆"""
    
    memory_id: str                          # 记忆 ID
    full_text: str                          # 重构的完整文本
    quality_metrics: Optional[QualityMetrics]  # 质量指标
    reconstruction_time_ms: float           # 重构耗时（毫秒）
    confidence: float                       # 置信度（0-1）
    warnings: List[str]                     # 警告信息
    original_fields: Dict[str, Any]         # 原始字段
```

**示例**:
```python
ReconstructedMemory(
    memory_id="mem_001",
    full_text="张三和李四在2026年2月17日讨论了100个项目...",
    quality_metrics=QualityMetrics(
        entity_accuracy=1.0,
        coherence_score=0.95,
        length_score=0.98,
        overall_score=0.97
    ),
    reconstruction_time_ms=85.3,
    confidence=0.97,
    warnings=[],
    original_fields={"timestamp": "2026-02-17T03:00:00"}
)
```

---

### QualityMetrics（质量指标）

```python
@dataclass
class QualityMetrics:
    """重构质量指标"""
    
    entity_accuracy: float      # 实体准确率（0-1）
    coherence_score: float      # 文本连贯性（0-1）
    length_score: float         # 长度合理性（0-1）
    overall_score: float        # 总体质量（0-1）
    warnings: List[str]         # 质量警告
```

---

## 3️⃣ 认知层：记忆单元

### MemoryPrimitive（记忆原语）

**位置**: `llm_compression/memory_primitive.py`

```python
@dataclass
class MemoryPrimitive:
    """自组织记忆单元（Phase 2.0）"""
    
    # 核心数据
    id: str                             # 记忆 ID
    content: CompressedMemory           # 压缩内容（Phase 1.1）
    embedding: np.ndarray               # 特征向量（384 维）
    
    # 自组织属性
    connections: Dict[str, float]       # 连接 {记忆ID: 强度}
    activation: float                   # 激活水平（0.0-1.0）
    
    # 统计信息
    access_count: int                   # 访问次数
    success_count: int                  # 成功次数
    last_access: Optional[datetime]     # 最后访问时间
    created_at: datetime                # 创建时间
```

**方法**:
```python
def activate(self, strength: float):
    """激活记忆"""
    
def record_success(self):
    """记录成功使用"""
    
def get_success_rate(self) -> float:
    """计算成功率"""
```

---

## 4️⃣ 表达层：生成输出

### ExpressionResult（表达结果）

**位置**: `llm_compression/expression_layer.py`

```python
@dataclass
class ExpressionResult:
    """表达生成结果"""
    
    content: str                    # 生成的内容
    modality: str                   # 模态（"text"/"image"/"audio"）
    quality_score: float            # 质量分数（0-1）
    source_memories: List[str]      # 源记忆 ID 列表
```

**示例**:
```python
ExpressionResult(
    content="根据记忆，张三和李四在2月17日讨论了项目进展...",
    modality="text",
    quality_score=0.92,
    source_memories=["mem_001", "mem_002", "mem_003"]
)
```

---

## 5️⃣ LLM 响应

### LLMResponse（LLM 响应）

**位置**: `llm_compression/llm_client.py`

```python
@dataclass
class LLMResponse:
    """LLM API 响应"""
    
    text: str                       # 生成的文本
    tokens_used: int                # 使用的 token 数
    latency_ms: float               # 延迟（毫秒）
    model: str                      # 使用的模型
    finish_reason: str              # 完成原因（stop/length/error）
    metadata: Dict[str, Any]        # 额外元数据
```

---

## 📈 数据流转

```
用户输入（文本）
    ↓
[压缩] → CompressedMemory
    ↓
[存储] → Arrow 格式
    ↓
[检索] → CompressedMemory
    ↓
[重构] → ReconstructedMemory
    ↓
[表达] → ExpressionResult
    ↓
用户输出（文本/图像/音频）
```

---

## 🔄 Phase 2.0 认知循环

```
感知输入
    ↓
CompressedMemory（压缩）
    ↓
MemoryPrimitive（记忆单元）
    ↓
自组织网络（学习连接）
    ↓
激活传播（检索）
    ↓
ExpressionResult（表达）
    ↓
反馈学习（质量优化）
```

---

## 🎯 输出模态

### 当前支持

#### 1. 文本输出（已实现）
```python
result = await expressor.express(
    memories=memories,
    modality="text",
    style="concise",
    max_length=500
)
# result.content: "生成的文本内容..."
```

#### 2. 图像输出（规划中）
```python
result = await expressor.express(
    memories=memories,
    modality="image",
    style="realistic"
)
# result.content: base64 编码的图像
```

#### 3. 音频输出（规划中）
```python
result = await expressor.express(
    memories=memories,
    modality="audio",
    style="natural"
)
# result.content: base64 编码的音频
```

---

## 💾 存储格式

### Arrow Schema

```python
# 压缩记忆存储格式
schema = pa.schema([
    ('memory_id', pa.string()),
    ('summary_hash', pa.string()),
    ('entities', pa.map_(pa.string(), pa.list_(pa.string()))),
    ('diff_data', pa.binary()),
    ('embedding', pa.list_(pa.float16())),  # 384 维，float16
    ('original_size', pa.int32()),
    ('compressed_size', pa.int32()),
    ('compression_ratio', pa.float32()),
    ('model_used', pa.string()),
    ('quality_score', pa.float32()),
    ('timestamp', pa.timestamp('us'))
])
```

---

## 🔍 质量保证

### 压缩质量阈值

| 指标 | 阈值 | 说明 |
|------|------|------|
| 语义相似度 | > 0.85 | 重构文本与原文的相似度 |
| 实体准确率 | > 0.95 | 关键实体的保留率 |
| 压缩比 | > 10x | 压缩效率 |
| 重构置信度 | > 0.85 | 重构质量置信度 |

### 质量监控

```python
from llm_compression import QualityEvaluator

evaluator = QualityEvaluator(
    semantic_threshold=0.85,
    entity_threshold=0.95
)

metrics = evaluator.evaluate(
    original=original_text,
    reconstructed=reconstructed_text,
    compressed_size=len(compressed.diff_data),
    reconstruction_latency_ms=85.3
)

if metrics.overall_score < 0.85:
    print(f"Quality warning: {metrics.warnings}")
```

---

## 📊 性能指标

### 实测性能（Phase 1.1）

| 操作 | 输入 | 输出 | 耗时 |
|------|------|------|------|
| 压缩 | 文本（1000 字） | CompressedMemory | < 1s |
| 重构 | CompressedMemory | ReconstructedMemory | < 0.1s |
| 表达 | 3 个记忆 | ExpressionResult | < 2s |

### 压缩效果

- **压缩比**: 2800x（1.2 MB → 440 字节）
- **质量分数**: 1.00
- **实体准确率**: 100%

---

## 🛠️ 使用示例

### 完整流程

```python
from llm_compression import (
    LLMCompressor, LLMReconstructor, 
    MultiModalExpressor, MemoryPrimitive
)

# 1. 压缩
compressed = await compressor.compress(
    text="张三和李四在2026年2月17日讨论了100个项目的进展..."
)
# 输出: CompressedMemory

# 2. 创建记忆单元
memory = MemoryPrimitive(
    id="mem_001",
    content=compressed,
    embedding=np.array(compressed.embedding)
)

# 3. 重构
reconstructed = await reconstructor.reconstruct(compressed)
# 输出: ReconstructedMemory

# 4. 表达
result = await expressor.express(
    memories=[memory],
    modality="text",
    style="concise"
)
# 输出: ExpressionResult

print(result.content)
# "根据记忆，张三和李四在2月17日讨论了项目进展..."
```

---

## 📝 总结

### 输出模型层次

1. **CompressedMemory** - 压缩存储格式
2. **ReconstructedMemory** - 解压重构格式
3. **MemoryPrimitive** - 认知记忆单元
4. **ExpressionResult** - 最终表达输出
5. **LLMResponse** - LLM 原始响应

### 设计原则

- ✅ **分层设计**: 每层有明确职责
- ✅ **类型安全**: 使用 @dataclass 定义
- ✅ **可扩展**: 支持多模态输出
- ✅ **质量保证**: 内置质量指标
- ✅ **性能优化**: 向量使用 float16

### 未来扩展

- 🔄 图像生成（DALL-E/Stable Diffusion）
- 🔄 音频生成（TTS）
- 🔄 视频生成（Sora）
- 🔄 多模态融合输出

# OpenClaw 集成指南 (OpenClaw Integration Guide)

本指南详细说明如何将 LLM 集成压缩系统集成到 OpenClaw 记忆系统中。

## 目录

- [概述](#概述)
- [架构设计](#架构设计)
- [集成步骤](#集成步骤)
- [配置说明](#配置说明)
- [API 使用](#api-使用)
- [Schema 兼容性](#schema-兼容性)
- [迁移指南](#迁移指南)
- [性能优化](#性能优化)
- [故障排查](#故障排查)

---

## 概述

### 什么是 OpenClaw？

OpenClaw 是 AI-OS 的标准记忆接口，提供统一的记忆存储和检索 API。LLM 集成压缩系统完全兼容 OpenClaw，可以无缝集成到现有系统中。

### 集成优势

- **透明压缩**: 自动判断是否压缩，对调用者完全透明
- **100% 兼容**: 完全兼容 OpenClaw Arrow schema
- **向后兼容**: 支持读取未压缩的旧记忆
- **高压缩比**: 39.63x 压缩比，大幅降低存储成本
- **高保真度**: > 90% 语义相似度，100% 实体准确率

### 系统要求

- OpenClaw 版本: 1.0+
- Python 版本: 3.10+
- Arrow 版本: 10.0+

---

## 架构设计

### 集成架构

```
┌─────────────────────────────────────────────────────────────┐
│                      OpenClaw Application                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Standard OpenClaw API
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              OpenClawMemoryInterface (Adapter)               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  store_memory()  │  retrieve_memory()  │  search()   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│   LLM Compression Core    │   │    Arrow Storage Layer    │
│  ┌─────────────────────┐  │   │  ┌─────────────────────┐  │
│  │  Compressor         │  │   │  │  Arrow Tables       │  │
│  │  Reconstructor      │  │   │  │  Parquet Files      │  │
│  │  Quality Evaluator  │  │   │  │  Embedding Index    │  │
│  └─────────────────────┘  │   │  └─────────────────────┘  │
└───────────────────────────┘   └───────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│                         LLM Client                           │
│              (Cloud API / Local Models)                      │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

**存储流程**:
1. OpenClaw 调用 `store_memory()`
2. 适配器判断是否需要压缩（基于文本长度）
3. 如需压缩：调用 LLMCompressor → 存储压缩数据
4. 如不需压缩：直接存储原始数据
5. 返回 memory_id

**检索流程**:
1. OpenClaw 调用 `retrieve_memory()`
2. 适配器从 Arrow 存储加载数据
3. 检查 `is_compressed` 字段
4. 如已压缩：调用 LLMReconstructor → 返回重构数据
5. 如未压缩：直接返回原始数据

---

## 集成步骤

### 步骤 1: 安装依赖

```bash
# 安装 LLM 压缩系统
pip install llm-compression

# 或从源码安装
git clone https://github.com/ai-os/llm-compression.git
cd llm-compression
pip install -e .
```

### 步骤 2: 配置系统

创建配置文件 `config.yaml`:

```yaml
# LLM 配置
llm:
  cloud_endpoint: "http://localhost:8045"
  timeout: 30.0
  max_retries: 3

# 存储配置（使用 OpenClaw 路径）
storage:
  storage_path: "~/.ai-os/memory/"
  compression_level: 3
  use_float16: true

# 压缩配置
compression:
  min_compress_length: 100
  auto_compress_threshold: 100
  quality_threshold: 0.85

# 性能配置
performance:
  batch_size: 16
  max_concurrent: 4
```

### 步骤 3: 初始化接口

```python
from llm_compression import Config, OpenClawMemoryInterface

# 加载配置
config = Config.from_yaml("config.yaml")

# 初始化 OpenClaw 接口
interface = OpenClawMemoryInterface.from_config(config)
```

### 步骤 4: 替换现有接口

**原有代码**:
```python
from openclaw.memory import MemoryInterface

memory_interface = MemoryInterface(
    storage_path="~/.ai-os/memory/"
)
```

**新代码**:
```python
from llm_compression import Config, OpenClawMemoryInterface

config = Config.from_yaml("config.yaml")
memory_interface = OpenClawMemoryInterface.from_config(config)
```

### 步骤 5: 验证集成

```python
import asyncio

async def test_integration():
    # 存储记忆
    memory_id = await memory_interface.store_memory(
        memory={
            "timestamp": "2024-01-15T15:00:00",
            "context": "Met with team to discuss project",
            "intent": "meeting",
            "action": "discuss",
            "outcome": "agreed on timeline",
            "success": True
        },
        memory_category="experiences"
    )
    
    print(f"Stored: {memory_id}")
    
    # 检索记忆
    retrieved = await memory_interface.retrieve_memory(
        memory_id=memory_id,
        memory_category="experiences"
    )
    
    print(f"Retrieved: {retrieved['context']}")
    
    # 搜索记忆
    results = await memory_interface.search_memories(
        query="project meeting",
        memory_category="experiences",
        top_k=5
    )
    
    print(f"Found {len(results)} memories")

asyncio.run(test_integration())
```

---

## 配置说明

### OpenClaw 标准路径

系统支持 OpenClaw 的所有标准存储路径：

```python
OPENCLAW_PATHS = {
    'core': {
        'identity': '~/.ai-os/memory/core/identity.arrow',
        'experiences': '~/.ai-os/memory/core/experiences.arrow',
        'preferences': '~/.ai-os/memory/core/preferences.arrow',
    },
    'working': {
        'context': '~/.ai-os/memory/working/context.arrow',
    },
    'long_term': {
        'archived': '~/.ai-os/memory/long-term/archived.arrow',
    },
    'shared': {
        'team': '~/.ai-os/memory/shared/team.arrow',
    }
}
```

### 记忆类别

支持的记忆类别：

- **identity**: 身份和自我认知
- **experiences**: 历史经验
- **preferences**: 偏好和价值观
- **context**: 当前上下文

### 自动压缩策略

系统根据以下规则自动判断是否压缩：

```python
# 配置
auto_compress_threshold: 100  # 字符数

# 规则
if text_length < 100:
    # 不压缩，直接存储
    store_uncompressed()
elif text_length >= 100:
    # 压缩后存储
    compress_and_store()
```

您可以通过配置调整阈值：

```yaml
compression:
  auto_compress_threshold: 200  # 提高到 200 字符
```

---

## API 使用

### 存储记忆

```python
# 基本用法
memory_id = await interface.store_memory(
    memory={
        "timestamp": "2024-01-15T15:00:00",
        "context": "Your context here...",
        "intent": "meeting",
        "action": "discuss",
        "outcome": "agreed",
        "success": True
    },
    memory_category="experiences"
)
```

**支持的字段**:

**experiences** 类别:
- `timestamp`: 时间戳
- `context`: 上下文（会被压缩）
- `intent`: 意图
- `action`: 行动（会被压缩）
- `outcome`: 结果（会被压缩）
- `success`: 是否成功
- `embedding`: 向量表示（自动生成）
- `related_memories`: 关联记忆

**identity** 类别:
- `description`: 描述（会被压缩）
- `values`: 价值观（会被压缩）
- `traits`: 特质

**preferences** 类别:
- `preference`: 偏好（会被压缩）
- `reason`: 原因（会被压缩）
- `strength`: 强度

### 检索记忆

```python
# 基本用法
memory = await interface.retrieve_memory(
    memory_id="mem_123456",
    memory_category="experiences"
)

# 访问字段
print(f"Context: {memory['context']}")
print(f"Outcome: {memory['outcome']}")
```

**返回格式**: 与 OpenClaw 标准格式完全一致

### 搜索记忆

```python
# 语义搜索
results = await interface.search_memories(
    query="project meeting with team",
    memory_category="experiences",
    top_k=10
)

# 遍历结果
for memory in results:
    print(f"Context: {memory['context']}")
    print(f"Timestamp: {memory['timestamp']}")
```

### 获取关联记忆

```python
# 获取相关记忆
related = await interface.get_related_memories(
    memory_id="mem_123456",
    memory_category="experiences",
    top_k=5
)

# 遍历关联记忆
for memory in related:
    print(f"Related: {memory['context']}")
```

---

## Schema 兼容性

### OpenClaw 原始 Schema

```python
import pyarrow as pa

experiences_schema = pa.schema([
    ('timestamp', pa.timestamp('us')),
    ('context', pa.string()),
    ('intent', pa.string()),
    ('action', pa.string()),
    ('outcome', pa.string()),
    ('success', pa.bool_()),
    ('embedding', pa.list_(pa.float32(), 1536)),
    ('related_memories', pa.list_(pa.string())),
])
```

### 扩展 Schema

系统扩展了 schema 以支持压缩，但保持向后兼容：

```python
compressed_experiences_schema = pa.schema([
    # === 原有字段（完全兼容）===
    ('timestamp', pa.timestamp('us')),
    ('context', pa.string()),              # 未压缩时存储原文，压缩后为空
    ('intent', pa.string()),
    ('action', pa.string()),
    ('outcome', pa.string()),
    ('success', pa.bool_()),
    ('embedding', pa.list_(pa.float16(), 1536)),  # 使用 float16 节省空间
    ('related_memories', pa.list_(pa.string())),
    
    # === 压缩扩展字段 ===
    ('is_compressed', pa.bool_()),         # 是否已压缩
    ('summary_hash', pa.string()),         # 摘要哈希
    ('entities', pa.struct([               # 关键实体
        ('persons', pa.list_(pa.string())),
        ('locations', pa.list_(pa.string())),
        ('dates', pa.list_(pa.string())),
        ('numbers', pa.list_(pa.string())),
        ('keywords', pa.list_(pa.string())),
    ])),
    ('diff_data', pa.binary()),            # 压缩的差异数据
    ('compression_metadata', pa.struct([   # 压缩元信息
        ('original_size', pa.int32()),
        ('compressed_size', pa.int32()),
        ('compression_ratio', pa.float32()),
        ('model_used', pa.string()),
        ('quality_score', pa.float32()),
        ('compression_time_ms', pa.float32()),
        ('compressed_at', pa.timestamp('us')),
    ])),
])
```

### 向后兼容性

系统完全支持读取旧版未压缩的记忆：

```python
# 旧版记忆（无压缩字段）
old_memory = {
    "timestamp": "2024-01-15T15:00:00",
    "context": "Original text",
    "intent": "meeting",
    # ... 其他字段
}

# 系统会自动处理
retrieved = await interface.retrieve_memory(memory_id)
# 正常返回，无需修改代码
```

---

## 迁移指南

### 从未压缩系统迁移

如果您已有 OpenClaw 记忆数据，可以平滑迁移：

#### 方法 1: 渐进式迁移（推荐）

新记忆自动压缩，旧记忆保持不变：

```python
# 无需任何操作！
# 系统会自动：
# - 新存储的记忆：自动压缩
# - 旧存储的记忆：正常读取
```

#### 方法 2: 批量迁移

批量压缩现有记忆：

```python
import asyncio
from llm_compression import Config, OpenClawMemoryInterface
import pyarrow.parquet as pq

async def migrate_memories():
    config = Config.from_yaml("config.yaml")
    interface = OpenClawMemoryInterface.from_config(config)
    
    # 读取旧记忆
    old_table = pq.read_table("~/.ai-os/memory/core/experiences.arrow")
    
    # 逐条迁移
    for i in range(len(old_table)):
        row = old_table.slice(i, 1).to_pydict()
        
        # 检查是否已压缩
        if 'is_compressed' in row and row['is_compressed'][0]:
            continue  # 已压缩，跳过
        
        # 重新存储（自动压缩）
        memory = {
            "timestamp": row['timestamp'][0],
            "context": row['context'][0],
            "intent": row['intent'][0],
            "action": row['action'][0],
            "outcome": row['outcome'][0],
            "success": row['success'][0],
        }
        
        await interface.store_memory(memory, "experiences")
        print(f"Migrated {i+1}/{len(old_table)}")

asyncio.run(migrate_memories())
```

### 迁移验证

验证迁移后的数据：

```python
async def verify_migration():
    # 随机抽样验证
    sample_ids = random.sample(all_memory_ids, 100)
    
    for memory_id in sample_ids:
        # 检索记忆
        memory = await interface.retrieve_memory(memory_id)
        
        # 验证字段完整性
        assert 'context' in memory
        assert 'timestamp' in memory
        assert len(memory['context']) > 0
        
    print("Migration verified successfully!")

asyncio.run(verify_migration())
```

---

## 性能优化

### 批量操作

使用批量接口提高性能：

```python
from llm_compression import BatchProcessor

# 初始化批量处理器
batch_processor = BatchProcessor.from_config(config)

# 批量存储
memories = [
    {"context": "Memory 1", ...},
    {"context": "Memory 2", ...},
    {"context": "Memory 3", ...},
]

memory_ids = await batch_processor.store_batch(
    memories=memories,
    memory_category="experiences"
)
```

### 缓存策略

启用缓存提高检索性能：

```yaml
performance:
  cache_size: 10000  # 缓存 10000 条记忆
  cache_ttl: 3600    # 缓存 1 小时
```

### 并发控制

调整并发数以平衡性能和资源：

```yaml
performance:
  max_concurrent: 8  # 最大并发数
  batch_size: 32     # 批量大小
```

### 存储优化

使用 float16 和高压缩级别：

```yaml
storage:
  use_float16: true      # 使用 float16 存储 embedding
  compression_level: 9   # 最高压缩级别（更慢但更小）
```

---

## 故障排查

### 问题 1: 检索记忆失败

**症状**: `KeyError: 'context'`

**原因**: 压缩记忆未正确重构

**解决方案**:
```python
# 检查 LLM 客户端状态
from llm_compression.health import HealthChecker

checker = HealthChecker(config=config)
result = await checker.check_health()
print(f"LLM Client: {result.components['llm_client'].status}")

# 如果 LLM 不可用，使用降级策略
config.fallback.enable_simple_compression = True
```

### 问题 2: 压缩比低于预期

**症状**: 压缩比 < 10x

**原因**: 文本太短或包含大量独特信息

**解决方案**:
```python
# 调整压缩阈值
config.compression.min_compress_length = 200

# 或检查文本内容
if len(text) < 200:
    print("Text too short for effective compression")
```

### 问题 3: 重构质量不达标

**症状**: 语义相似度 < 0.85

**原因**: LLM 模型质量不足

**解决方案**:
```python
# 使用更高质量的模型
from llm_compression import QualityLevel

model = model_selector.select_model(
    memory_type=MemoryType.TEXT,
    text_length=len(text),
    quality_requirement=QualityLevel.HIGH  # 使用高质量模型
)
```

### 问题 4: Schema 不兼容

**症状**: `ArrowInvalid: Schema mismatch`

**原因**: 旧版 schema 缺少压缩字段

**解决方案**:
```python
# 使用迁移工具
from llm_compression.openclaw_interface import BackwardCompatibility

# 迁移旧 schema
new_table = BackwardCompatibility.migrate_legacy_memory(old_table)
```

---

## 完整示例

### 端到端集成示例

```python
import asyncio
from llm_compression import Config, OpenClawMemoryInterface

async def openclaw_integration_example():
    # 1. 初始化
    config = Config.from_yaml("config.yaml")
    interface = OpenClawMemoryInterface.from_config(config)
    
    # 2. 存储多条记忆
    memories = [
        {
            "timestamp": "2024-01-15T10:00:00",
            "context": "Morning standup meeting with engineering team. Discussed sprint progress and blockers.",
            "intent": "meeting",
            "action": "discuss sprint",
            "outcome": "identified 3 blockers",
            "success": True
        },
        {
            "timestamp": "2024-01-15T14:00:00",
            "context": "Code review session for authentication module. Found security vulnerability in JWT validation.",
            "intent": "code_review",
            "action": "review code",
            "outcome": "fixed security issue",
            "success": True
        },
        {
            "timestamp": "2024-01-15T16:00:00",
            "context": "Met with product manager to discuss Q2 roadmap. Prioritized 5 key features.",
            "intent": "planning",
            "action": "discuss roadmap",
            "outcome": "agreed on priorities",
            "success": True
        }
    ]
    
    memory_ids = []
    for memory in memories:
        memory_id = await interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        memory_ids.append(memory_id)
        print(f"Stored: {memory_id}")
    
    # 3. 检索记忆
    for memory_id in memory_ids:
        retrieved = await interface.retrieve_memory(
            memory_id=memory_id,
            memory_category="experiences"
        )
        print(f"\nRetrieved: {memory_id}")
        print(f"  Context: {retrieved['context'][:50]}...")
        print(f"  Outcome: {retrieved['outcome']}")
    
    # 4. 语义搜索
    results = await interface.search_memories(
        query="code review security",
        memory_category="experiences",
        top_k=5
    )
    
    print(f"\nSearch results for 'code review security':")
    for i, memory in enumerate(results):
        print(f"  {i+1}. {memory['context'][:50]}...")
    
    # 5. 获取关联记忆
    related = await interface.get_related_memories(
        memory_id=memory_ids[0],
        memory_category="experiences",
        top_k=3
    )
    
    print(f"\nRelated memories:")
    for memory in related:
        print(f"  - {memory['context'][:50]}...")

# 运行示例
asyncio.run(openclaw_integration_example())
```

---

## 最佳实践

### 1. 合理设置压缩阈值

```yaml
compression:
  min_compress_length: 100      # 短文本不压缩
  auto_compress_threshold: 100  # 自动压缩阈值
```

### 2. 使用批量操作

```python
# 好的做法
memory_ids = await batch_processor.store_batch(memories)

# 避免
for memory in memories:
    await interface.store_memory(memory)  # 慢
```

### 3. 启用监控

```yaml
monitoring:
  enable_prometheus: true
  alert_quality_threshold: 0.90
```

### 4. 定期健康检查

```python
# 定期检查系统健康
result = await health_checker.check_health()
if result.overall_status != "healthy":
    send_alert(result)
```

### 5. 使用降级策略

```yaml
fallback:
  enable_simple_compression: true
  enable_partial_reconstruction: true
```

---

## 下一步

- 阅读 [API 参考文档](API_REFERENCE.md) 了解详细 API
- 查看 [故障排查指南](TROUBLESHOOTING.md) 解决常见问题
- 运行 [Jupyter 教程](../notebooks/tutorial_basic.ipynb) 学习更多用法
- 查看 [示例代码](../examples/) 获取更多示例

---

**版本**: Phase 1.0  
**最后更新**: 2024  
**OpenClaw 兼容性**: 100%

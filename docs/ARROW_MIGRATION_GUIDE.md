# Arrow 零拷贝优化迁移指南

## 概述

本指南帮助您将现有代码迁移到 Arrow 零拷贝优化版本，实现 10-20x 性能提升和 80% 内存节省。

**迁移收益**:
- ✅ 10-64x 性能提升
- ✅ 76-80% 内存节省
- ✅ 支持 100K+ 记忆规模
- ✅ 向后兼容（旧代码继续工作）

**迁移成本**:
- 🔄 代码修改：最小（主要是方法名变化）
- 🔄 学习曲线：低（API 设计相似）
- 🔄 测试工作：中等（需要验证功能）

---

## 快速开始

### 1. 最小迁移（5 分钟）

只需添加 `_arrow` 后缀即可使用优化版本：

```python
# 旧代码
from llm_compression.embedder import LocalEmbedder

embedder = LocalEmbedder()
embeddings = embedder.encode_batch(texts)

# 新代码（零拷贝优化）
from llm_compression.embedder_arrow import LocalEmbedderArrow

embedder_arrow = LocalEmbedderArrow()
embeddings_array = embedder_arrow.batch_encode_arrow(texts)  # 返回 Arrow Array
```

### 2. 完整迁移（30 分钟）

迁移到完整的 Arrow 流水线：

```python
# 旧代码
from llm_compression.cognitive_loop import CognitiveLoop

loop = CognitiveLoop()
# ... 添加记忆 ...
result = await loop.process(query, query_embedding)

# 新代码（端到端零拷贝）
from llm_compression.cognitive_loop_arrow import CognitiveLoopArrow

loop_arrow = CognitiveLoopArrow()
# ... 批量添加记忆（零拷贝）...
loop_arrow.batch_add_memories_arrow(memory_ids, contents)
result = await loop_arrow.process_arrow(query)  # 自动编码查询
```

---

## 模块迁移指南

### ArrowStorage (Task 12.1)

#### 旧代码
```python
from llm_compression.storage import ArrowStorage

storage = ArrowStorage()
storage.save(table, "memories.parquet")
table = storage.load("memories.parquet")

# 逐行处理（慢）
for i in range(len(table)):
    row = table.slice(i, 1)
    embedding = row['embedding'][0].as_py()  # 数据复制！
    # ... 处理 ...
```

#### 新代码（零拷贝）
```python
from llm_compression.arrow_storage_zero_copy import ArrowStorageZeroCopy
from llm_compression.arrow_zero_copy import ArrowBatchView, get_embeddings_buffer

storage = ArrowStorageZeroCopy()

# 内存映射加载（零拷贝）
table = storage.load_table_mmap("memories.parquet")

# 方法 1: 批量处理（零拷贝）
embeddings = get_embeddings_buffer(table, 'embedding')  # 零拷贝提取
# ... 向量化处理 ...

# 方法 2: 迭代处理（零拷贝）
batch_view = ArrowBatchView(table)
for memory_view in batch_view:
    # 延迟物化，只在需要时转换
    embedding = memory_view.get_numpy('embedding', zero_copy=True)
    # ... 处理 ...
```

**性能提升**: 16-64x

---

### LocalEmbedder (Task 12.2)

#### 旧代码
```python
from llm_compression.embedder import LocalEmbedder

embedder = LocalEmbedder()

# 单个编码
embedding = embedder.encode("text")

# 批量编码
embeddings = embedder.encode_batch(texts)  # 返回 NumPy 数组

# 相似度搜索
similarities = embedder.similarity(query_vec, embeddings)
top_indices = np.argsort(similarities)[::-1][:top_k]
```

#### 新代码（Arrow 原生）
```python
from llm_compression.embedder_arrow import LocalEmbedderArrow

embedder_arrow = LocalEmbedderArrow()

# 单个编码（返回 Arrow Array）
embedding_array = embedder_arrow.encode_to_arrow("text")

# 批量编码（零拷贝）
embeddings_array = embedder_arrow.batch_encode_arrow(texts)

# 创建 embedding 表
embedding_table = embedder_arrow.create_embedding_table(
    texts=texts,
    include_text=True
)

# 语义搜索（零拷贝）
result_table = embedder_arrow.semantic_search_arrow(
    query="search query",
    corpus_table=embedding_table,
    top_k=10
)

# 批量搜索（向量化）
results = embedder_arrow.batch_similarity_search(
    queries=["query1", "query2"],
    corpus_table=embedding_table,
    top_k=10
)
```

**性能提升**: 2-10x

---

### NetworkNavigator (Task 12.3)

#### 旧代码
```python
from llm_compression.network_navigator import NetworkNavigator

navigator = NetworkNavigator()

# 检索（逐个处理）
result = navigator.retrieve(
    query_embedding=query_vec,
    memory_network=memory_dict,
    max_results=10
)

# 访问记忆
for memory in result.memories:
    print(memory.content)
```

#### 新代码（向量化检索）
```python
from llm_compression.network_navigator_arrow import NetworkNavigatorArrow

navigator_arrow = NetworkNavigatorArrow()

# 检索（向量化，零拷贝）
result = navigator_arrow.retrieve_arrow(
    query_embedding=query_vec,
    memory_table=memory_table,  # Arrow Table
    max_results=10
)

# 访问记忆（零拷贝）
memories_table = result.table
contents = memories_table['content'].to_pylist()

# 简化版相似度搜索（无激活扩散）
similar_table = navigator_arrow.find_similar_memories_vectorized(
    query_embedding=query_vec,
    memory_table=memory_table,
    top_k=10
)

# 批量检索（并行）
results = navigator_arrow.batch_retrieve_arrow(
    query_embeddings=query_vecs,
    memory_table=memory_table,
    max_results=10
)
```

**性能提升**: 16-20x

---

### CognitiveLoop (Task 12.5)

#### 旧代码
```python
from llm_compression.cognitive_loop import CognitiveLoop
from llm_compression.memory_primitive import MemoryPrimitive

loop = CognitiveLoop()

# 添加记忆（逐个）
for i, text in enumerate(texts):
    memory = MemoryPrimitive(
        id=f"mem{i}",
        content=text,
        embedding=embedder.encode(text)
    )
    loop.add_memory(memory)

# 处理查询
query_embedding = embedder.encode(query)
result = await loop.process(query, query_embedding, max_memories=10)

print(result.output)
```

#### 新代码（端到端零拷贝）
```python
from llm_compression.cognitive_loop_arrow import CognitiveLoopArrow

loop_arrow = CognitiveLoopArrow()

# 批量添加记忆（零拷贝）
loop_arrow.batch_add_memories_arrow(
    memory_ids=[f"mem{i}" for i in range(len(texts))],
    contents=texts
    # embeddings 自动编码
)

# 或从 Arrow Table 加载
loop_arrow.load_memories_from_table(memory_table)

# 处理查询（自动编码）
result = await loop_arrow.process_arrow(
    query=query,
    max_memories=10
)

print(result.output)
print(f"Processing time: {result.processing_time_ms:.1f}ms")

# 批量处理查询
results = await loop_arrow.batch_process_queries(
    queries=["query1", "query2", "query3"],
    max_memories=10
)
```

**性能提升**: 10x 端到端

---

## 数据格式迁移

### 从 Python 对象到 Arrow Table

#### 旧格式（Python 字典列表）
```python
memories = [
    {
        'id': 'mem1',
        'content': 'Python is a programming language',
        'embedding': [0.1, 0.2, ...],
        'timestamp': 1234567890
    },
    # ...
]
```

#### 新格式（Arrow Table）
```python
import pyarrow as pa

# 方法 1: 从字典创建
memory_table = pa.table({
    'memory_id': pa.array(['mem1', 'mem2', ...]),
    'content': pa.array(['text1', 'text2', ...]),
    'embedding': embedder_arrow.batch_encode_arrow(texts),
    'timestamp': pa.array([1234567890, ...])
})

# 方法 2: 使用 create_embedding_table
memory_table = embedder_arrow.create_embedding_table(
    texts=texts,
    include_text=True,
    additional_columns={
        'memory_id': memory_ids,
        'timestamp': timestamps
    }
)

# 保存到 Parquet
pa.parquet.write_table(memory_table, "memories.parquet")

# 加载（内存映射，零拷贝）
from llm_compression.arrow_zero_copy import load_table_mmap
memory_table = load_table_mmap("memories.parquet")
```

---

## 性能优化最佳实践

### 1. 使用批量操作

❌ **不推荐**（逐个处理）:
```python
for text in texts:
    embedding = embedder.encode(text)
    # ... 处理 ...
```

✅ **推荐**（批量处理）:
```python
embeddings_array = embedder_arrow.batch_encode_arrow(texts, batch_size=32)
# ... 向量化处理 ...
```

### 2. 避免 .as_py() 调用

❌ **不推荐**（数据复制）:
```python
for i in range(len(table)):
    content = table['content'][i].as_py()  # 复制！
    embedding = table['embedding'][i].as_py()  # 复制！
```

✅ **推荐**（零拷贝）:
```python
# 批量提取
contents = table['content'].to_pylist()  # 一次性转换
embeddings = get_embeddings_buffer(table, 'embedding')  # 零拷贝

# 或使用 ArrowBatchView
batch_view = ArrowBatchView(table)
for memory_view in batch_view:
    content = memory_view.get_py('content')  # 延迟物化
```

### 3. 使用列裁剪

❌ **不推荐**（加载所有列）:
```python
table = pa.parquet.read_table("memories.parquet")
# 只需要 embedding 列，但加载了所有列
```

✅ **推荐**（只加载需要的列）:
```python
from llm_compression.arrow_zero_copy import prune_columns

table = pa.parquet.read_table(
    "memories.parquet",
    columns=['memory_id', 'embedding']  # 只加载需要的列
)
```

### 4. 使用内存映射

❌ **不推荐**（全部加载到内存）:
```python
table = pa.parquet.read_table("large_memories.parquet")
```

✅ **推荐**（内存映射，按需加载）:
```python
from llm_compression.arrow_zero_copy import load_table_mmap

table = load_table_mmap("large_memories.parquet")  # 支持 10GB+ 文件
```

### 5. 使用向量化操作

❌ **不推荐**（Python 循环）:
```python
similarities = []
for embedding in embeddings:
    sim = np.dot(query_vec, embedding)
    similarities.append(sim)
```

✅ **推荐**（向量化）:
```python
from llm_compression.arrow_zero_copy import compute_similarity_zero_copy

similarities = compute_similarity_zero_copy(embeddings, query_vec)
```

---

## 常见问题

### Q1: 旧代码还能用吗？

**A**: 是的！所有旧 API 保持不变，新的 Arrow 优化是可选的。

```python
# 旧代码继续工作
from llm_compression.embedder import LocalEmbedder
embedder = LocalEmbedder()
embeddings = embedder.encode_batch(texts)

# 新代码提供更好性能
from llm_compression.embedder_arrow import LocalEmbedderArrow
embedder_arrow = LocalEmbedderArrow()
embeddings_array = embedder_arrow.batch_encode_arrow(texts)
```

### Q2: 如何在旧代码和新代码之间转换？

**A**: 使用简单的转换函数：

```python
import pyarrow as pa
import numpy as np

# NumPy → Arrow
embeddings_np = np.array([[0.1, 0.2], [0.3, 0.4]])
embeddings_arrow = embedder_arrow._numpy_to_arrow_list(embeddings_np)

# Arrow → NumPy
embeddings_np = get_embeddings_buffer(table, 'embedding')

# Arrow Table → Pandas DataFrame
df = table.to_pandas()

# Pandas DataFrame → Arrow Table
table = pa.Table.from_pandas(df)
```

### Q3: 什么时候应该迁移？

**A**: 根据场景选择：

| 场景 | 是否迁移 | 原因 |
|------|---------|------|
| 记忆数 < 1K | 可选 | 性能提升不明显 |
| 记忆数 1K-10K | 推荐 | 10-20x 性能提升 |
| 记忆数 > 10K | 强烈推荐 | 必需，否则内存不足 |
| 批量处理 | 推荐 | 显著提升吞吐量 |
| 实时查询 | 推荐 | 降低延迟 |

### Q4: 迁移会破坏现有数据吗？

**A**: 不会。Arrow 和 Parquet 是标准格式，可以与现有工具互操作。

```python
# 旧数据（Parquet）
table_old = pa.parquet.read_table("old_memories.parquet")

# 新代码可以直接使用
loop_arrow.load_memories_from_table(table_old)
```

### Q5: 如何验证迁移正确性？

**A**: 使用对比测试：

```python
# 旧代码结果
embeddings_old = embedder.encode_batch(texts)

# 新代码结果
embeddings_array = embedder_arrow.batch_encode_arrow(texts)
embeddings_new = get_embeddings_buffer(
    pa.table({'embedding': embeddings_array}),
    'embedding'
)

# 验证一致性
np.testing.assert_allclose(embeddings_old, embeddings_new, rtol=1e-5)
```

---

## 迁移检查清单

### 准备阶段
- [ ] 阅读本迁移指南
- [ ] 了解 Arrow 基础概念
- [ ] 备份现有代码和数据

### 迁移阶段
- [ ] 安装依赖：`pip install pyarrow`
- [ ] 更新导入语句
- [ ] 修改方法调用（添加 `_arrow` 后缀）
- [ ] 转换数据格式（Python 对象 → Arrow Table）
- [ ] 更新测试代码

### 验证阶段
- [ ] 运行单元测试
- [ ] 对比新旧结果一致性
- [ ] 性能基准测试
- [ ] 内存使用分析

### 优化阶段
- [ ] 应用最佳实践
- [ ] 使用批量操作
- [ ] 启用内存映射
- [ ] 列裁剪优化

---

## 示例：完整迁移流程

### 步骤 1: 旧代码（基线）

```python
from llm_compression.embedder import LocalEmbedder
from llm_compression.cognitive_loop import CognitiveLoop
from llm_compression.memory_primitive import MemoryPrimitive

# 初始化
embedder = LocalEmbedder()
loop = CognitiveLoop()

# 添加记忆
texts = ["text1", "text2", "text3"]
for i, text in enumerate(texts):
    embedding = embedder.encode(text)
    memory = MemoryPrimitive(
        id=f"mem{i}",
        content=text,
        embedding=embedding
    )
    loop.add_memory(memory)

# 处理查询
query = "search query"
query_embedding = embedder.encode(query)
result = await loop.process(query, query_embedding, max_memories=5)
```

### 步骤 2: 迁移到 Arrow（优化）

```python
from llm_compression.embedder_arrow import LocalEmbedderArrow
from llm_compression.cognitive_loop_arrow import CognitiveLoopArrow

# 初始化
embedder_arrow = LocalEmbedderArrow()
loop_arrow = CognitiveLoopArrow()

# 批量添加记忆（零拷贝）
texts = ["text1", "text2", "text3"]
loop_arrow.batch_add_memories_arrow(
    memory_ids=[f"mem{i}" for i in range(len(texts))],
    contents=texts
)

# 处理查询（自动编码）
query = "search query"
result = await loop_arrow.process_arrow(query, max_memories=5)
```

### 步骤 3: 验证结果

```python
# 对比输出
print(f"Old output: {result_old.output}")
print(f"New output: {result_new.output}")

# 对比性能
print(f"Old time: {time_old:.1f}ms")
print(f"New time: {result_new.processing_time_ms:.1f}ms")
print(f"Speedup: {time_old / result_new.processing_time_ms:.1f}x")
```

---

## 获取帮助

### 文档资源
- `docs/ARROW_ZERO_COPY_OPTIMIZATION.md` - 优化方案详解
- `docs/ARROW_UNIFIED_PIPELINE.md` - 统一流水线架构
- `docs/ARROW_ZERO_COPY_USAGE.md` - 使用指南
- `docs/TASK_12_FINAL_SUMMARY.md` - 完整总结

### 代码示例
- `tests/unit/test_*_arrow.py` - 单元测试示例
- `tests/performance/test_*_benchmark.py` - 性能测试示例

### 社区支持
- GitHub Issues: 报告问题
- GitHub Discussions: 技术讨论

---

**文档版本**: 1.0  
**最后更新**: 2026-02-17  
**适用版本**: Phase 2.0 Task 12

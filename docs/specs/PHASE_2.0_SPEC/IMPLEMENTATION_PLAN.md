# Phase 2.0 实施计划

**版本**: 1.0
**创建日期**: 2026-02-17
**状态**: 已批准，准备实施
**策略**: 策略 C - 语义索引 + Arrow 原文存储

---

## 执行摘要

本文档定义了 Phase 2.0 的完整实施计划，基于 2026-02-17 的设计决策分析和基准测试结果。

**核心策略**: 采用**渐进式演进**方案，先实施策略 C（语义索引），未来按需演进到策略 D（智能分流）。

**关键优势**:
- ✅ 零数据迁移成本
- ✅ 用户零感知延迟
- ✅ 10x 检索速度提升
- ✅ 87.5% API 成本节省
- ✅ 100% 原文保真度

---

## 目录

1. [总体时间线](#总体时间线)
2. [Week 1-2: 基础存储层](#week-1-2-基础存储层)
3. [Week 3-4: 语义索引集成](#week-3-4-语义索引集成)
4. [Week 5-6: 优化与监控](#week-5-6-优化与监控)
5. [验收标准](#验收标准)
6. [风险管理](#风险管理)
7. [后续演进](#后续演进)

---

## 总体时间线

```
Week 1-2: 基础存储层 (P0)
├─ Arrow 压缩存储
├─ 本地向量化
├─ 基础向量检索
└─ 数据结构定义

Week 3-4: 语义索引集成 (P0)
├─ Protocol Adapter 集成
├─ 后台批处理队列
├─ LLM 批量索引
└─ 语义检索逻辑

Week 5-6: 优化与监控 (P1)
├─ 成本监控
├─ 性能优化
├─ 文档完善
└─ 生产部署

✅ Milestone: Phase 2.0 完成
```

---

## Week 1-2: 基础存储层

### 目标

建立核心存储和检索能力，实现零延迟的用户体验。

### 任务清单

#### Task 1.1: Arrow 压缩存储实现 (2 天)

**文件**: `llm_compression/storage.py`

**实现内容**:
```python
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional
import zstandard as zstd

class ArrowStorage:
    """Arrow/Parquet 压缩存储引擎"""

    def __init__(self, storage_path: str = "~/.ai-os/memory/"):
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def compress(self, text: str) -> bytes:
        """
        压缩文本为 Arrow 格式

        Args:
            text: 原始文本

        Returns:
            压缩后的字节数据
        """
        # 创建 Arrow 表
        table = pa.table({
            'text': [text],
            'length': [len(text)]
        })

        # 序列化为 IPC 格式（零拷贝）
        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, table.schema)
        writer.write_table(table)
        writer.close()

        # ZSTD 压缩
        compressed = zstd.compress(sink.getvalue().to_pybytes(), level=3)
        return compressed

    def decompress(self, compressed: bytes) -> str:
        """
        解压缩 Arrow 数据

        Args:
            compressed: 压缩的字节数据

        Returns:
            原始文本
        """
        # ZSTD 解压
        decompressed = zstd.decompress(compressed)

        # 反序列化 Arrow
        reader = pa.ipc.open_stream(decompressed)
        table = reader.read_all()

        return table['text'][0].as_py()

    def save(self, memory_id: str, compressed: bytes) -> None:
        """保存压缩数据到磁盘"""
        file_path = self.storage_path / f"{memory_id}.arrow"
        file_path.write_bytes(compressed)

    def load(self, memory_id: str) -> bytes:
        """从磁盘加载压缩数据"""
        file_path = self.storage_path / f"{memory_id}.arrow"
        return file_path.read_bytes()
```

**测试用例**:
```python
def test_arrow_compression():
    storage = ArrowStorage()

    # 测试压缩
    text = "Test memory content" * 100
    compressed = storage.compress(text)

    # 验证压缩比
    compression_ratio = len(text) / len(compressed)
    assert compression_ratio > 2.0  # 至少 2x

    # 测试解压
    decompressed = storage.decompress(compressed)
    assert decompressed == text  # 100% 保真

def test_storage_persistence():
    storage = ArrowStorage()

    # 保存
    memory_id = "test_001"
    compressed = storage.compress("Test content")
    storage.save(memory_id, compressed)

    # 加载
    loaded = storage.load(memory_id)
    assert loaded == compressed
```

**验收标准**:
- ✅ 压缩比 > 2.5x
- ✅ 压缩/解压延迟 < 1ms
- ✅ 100% 数据保真
- ✅ 测试覆盖率 > 90%

---

#### Task 1.2: 本地向量化实现 (1 天)

**文件**: `llm_compression/embedder.py`

**实现内容**:
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class LocalEmbedder:
    """本地向量化引擎（零 API 成本）"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化本地嵌入模型

        Args:
            model_name: 模型名称（默认 384 维）
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = 384

    def encode(self, text: str) -> np.ndarray:
        """
        文本向量化

        Args:
            text: 输入文本

        Returns:
            384 维向量
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量向量化（更高效）"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```

**测试用例**:
```python
def test_local_embedding():
    embedder = LocalEmbedder()

    # 测试单个文本
    text = "This is a test"
    embedding = embedder.encode(text)

    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32

def test_semantic_similarity():
    embedder = LocalEmbedder()

    # 语义相似的文本
    text1 = "The cat sits on the mat"
    text2 = "A cat is sitting on a mat"
    text3 = "Python is a programming language"

    vec1 = embedder.encode(text1)
    vec2 = embedder.encode(text2)
    vec3 = embedder.encode(text3)

    # 相似文本的相似度应该高
    sim_12 = embedder.similarity(vec1, vec2)
    sim_13 = embedder.similarity(vec1, vec3)

    assert sim_12 > 0.8  # 高相似度
    assert sim_13 < 0.5  # 低相似度
```

**验收标准**:
- ✅ 向量化延迟 < 10ms
- ✅ 语义相似度准确率 > 85%
- ✅ 批量处理支持

---

#### Task 1.3: StoredMemory 数据结构 (1 天)

**文件**: `llm_compression/stored_memory.py`

**实现内容**:
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import uuid

@dataclass
class SemanticIndex:
    """语义索引（后台填充）"""
    summary: str
    entities: list[str]
    topics: list[str]
    indexed_at: datetime
    model_used: str
    quality_score: float = 0.0

@dataclass
class StoredMemory:
    """存储的记忆单元"""

    # 核心标识
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    # 原文（Arrow 压缩，100% 保真）
    original_compressed: bytes = b""

    # 语义索引（可选，后台填充）
    semantic_index: Optional[SemanticIndex] = None

    # 本地向量（用于向量检索）
    embedding: Optional[np.ndarray] = None

    # 元数据（灵活扩展）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'original_compressed': self.original_compressed.hex(),
            'semantic_index': self.semantic_index.__dict__ if self.semantic_index else None,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'StoredMemory':
        """从字典反序列化"""
        return cls(
            id=data['id'],
            created_at=datetime.fromisoformat(data['created_at']),
            original_compressed=bytes.fromhex(data['original_compressed']),
            semantic_index=SemanticIndex(**data['semantic_index']) if data['semantic_index'] else None,
            embedding=np.array(data['embedding']) if data['embedding'] else None,
            metadata=data['metadata']
        )
```

**验收标准**:
- ✅ 数据结构完整定义
- ✅ 序列化/反序列化正常
- ✅ 向后兼容性设计

---

#### Task 1.4: 基础向量检索 (2 天)

**文件**: `llm_compression/vector_search.py`

**实现内容**:
```python
import numpy as np
from typing import List, Tuple
from .stored_memory import StoredMemory

class VectorSearch:
    """向量检索引擎（降级路径）"""

    def __init__(self):
        self.memories: List[StoredMemory] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_memory(self, memory: StoredMemory) -> None:
        """添加记忆到索引"""
        self.memories.append(memory)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """重建向量索引"""
        if not self.memories:
            return

        embeddings = [m.embedding for m in self.memories if m.embedding is not None]
        if embeddings:
            self.embeddings = np.vstack(embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[StoredMemory, float]]:
        """
        向量相似度检索

        Args:
            query_embedding: 查询向量
            top_k: 返回前 K 个结果

        Returns:
            (记忆, 相似度分数) 列表
        """
        if self.embeddings is None or len(self.memories) == 0:
            return []

        # 计算余弦相似度
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # 排序并返回 top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (self.memories[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results
```

**测试用例**:
```python
def test_vector_search():
    search = VectorSearch()
    embedder = LocalEmbedder()

    # 添加记忆
    texts = [
        "Python is a programming language",
        "Java is also a programming language",
        "The cat sits on the mat"
    ]

    for text in texts:
        memory = StoredMemory(
            original_compressed=b"...",
            embedding=embedder.encode(text)
        )
        search.add_memory(memory)

    # 搜索
    query = "programming languages"
    query_embedding = embedder.encode(query)
    results = search.search(query_embedding, top_k=2)

    # 验证结果
    assert len(results) == 2
    assert results[0][1] > 0.5  # 高相似度
```

**验收标准**:
- ✅ 检索延迟 < 50ms (1000 条记忆)
- ✅ Top-K 准确率 > 85%
- ✅ 支持增量添加

---

### Week 1-2 里程碑

**完成标准**:
- ✅ 所有 4 个任务完成
- ✅ 单元测试覆盖率 > 90%
- ✅ 集成测试通过
- ✅ 性能基准达标

**可交付成果**:
- 可用的存储和检索系统
- 完整的测试套件
- 性能基准报告

---

## Week 3-4: 语义索引集成

### 目标

集成 LLM 语义索引，实现 10x 检索速度提升。

### 任务清单

#### Task 2.1: 后台批处理队列 (2 天)

**文件**: `llm_compression/background_queue.py`

**实现内容**:
```python
from queue import Queue
from threading import Thread, Event
from typing import Callable, List
import time
from datetime import datetime

class BackgroundQueue:
    """后台批处理队列"""

    def __init__(
        self,
        batch_size: int = 1000,
        interval_seconds: int = 3600  # 1 小时
    ):
        self.batch_size = batch_size
        self.interval_seconds = interval_seconds
        self.queue = Queue()
        self.stop_event = Event()
        self.worker_thread = None

    def add(self, memory_id: str, text: str) -> None:
        """添加到队列"""
        self.queue.put({
            'memory_id': memory_id,
            'text': text,
            'added_at': datetime.now()
        })

    def get_batch(self, size: int = None) -> List[dict]:
        """获取一批待处理项"""
        size = size or self.batch_size
        batch = []

        while len(batch) < size and not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                batch.append(item)
            except:
                break

        return batch

    def start_worker(self, process_func: Callable) -> None:
        """启动后台工作线程"""
        def worker():
            while not self.stop_event.is_set():
                # 等待间隔
                time.sleep(self.interval_seconds)

                # 处理批次
                batch = self.get_batch()
                if batch:
                    try:
                        process_func(batch)
                    except Exception as e:
                        print(f"Batch processing error: {e}")

        self.worker_thread = Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        """停止工作线程"""
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
```

**验收标准**:
- ✅ 批处理队列正常工作
- ✅ 线程安全
- ✅ 优雅停止

---

#### Task 2.2: LLM 批量索引 (3 天)

**文件**: `llm_compression/semantic_indexer.py`

**实现内容**:
```python
from typing import List
from .protocol_adapter import ProtocolAdapter
from .stored_memory import SemanticIndex
from datetime import datetime

class SemanticIndexer:
    """LLM 语义索引器"""

    def __init__(self, api_key: str, base_url: str = "http://localhost:8045"):
        self.adapter = ProtocolAdapter(base_url=base_url, api_key=api_key)
        self.model = "claude-opus-4"  # 最高质量

    def extract_index(self, text: str) -> SemanticIndex:
        """
        提取单个文本的语义索引

        Args:
            text: 原始文本

        Returns:
            语义索引
        """
        # 提取摘要
        summary_prompt = f"Summarize this text in 1-2 concise sentences: {text}"
        summary = self.adapter.complete(summary_prompt, model=self.model, max_tokens=100)

        # 提取实体
        entities_prompt = f"Extract key entities (names, dates, numbers, locations) as a JSON list: {text}"
        entities_str = self.adapter.complete(entities_prompt, model=self.model, max_tokens=100)

        # 解析实体
        import json
        try:
            entities = json.loads(entities_str)
            if not isinstance(entities, list):
                entities = [entities_str[:50]]
        except:
            entities = []

        # 提取主题
        topics_prompt = f"Extract 2-3 main topics/themes as a JSON list: {text}"
        topics_str = self.adapter.complete(topics_prompt, model=self.model, max_tokens=50)

        try:
            topics = json.loads(topics_str)
            if not isinstance(topics, list):
                topics = []
        except:
            topics = []

        return SemanticIndex(
            summary=summary,
            entities=entities,
            topics=topics,
            indexed_at=datetime.now(),
            model_used=self.model,
            quality_score=0.9  # 默认高质量
        )

    def batch_extract_indices(self, texts: List[str]) -> List[SemanticIndex]:
        """
        批量提取语义索引（成本优化）

        Args:
            texts: 文本列表

        Returns:
            语义索引列表
        """
        indices = []

        for text in texts:
            try:
                index = self.extract_index(text)
                indices.append(index)
            except Exception as e:
                print(f"Indexing error: {e}")
                # 创建降级索引
                indices.append(SemanticIndex(
                    summary=text[:100],
                    entities=[],
                    topics=[],
                    indexed_at=datetime.now(),
                    model_used="fallback",
                    quality_score=0.5
                ))

        return indices
```

**验收标准**:
- ✅ 批量索引正常工作
- ✅ 错误处理和降级
- ✅ API 成本 < $0.5/天 (1000 条)

---

#### Task 2.3: 语义索引数据库 (2 天)

**文件**: `llm_compression/semantic_index_db.py`

**实现内容**:
```python
import sqlite3
from typing import List, Optional
from .stored_memory import SemanticIndex

class SemanticIndexDB:
    """语义索引数据库（快速检索）"""

    def __init__(self, db_path: str = "~/.ai-os/memory/semantic_index.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_indices (
                memory_id TEXT PRIMARY KEY,
                summary TEXT,
                entities TEXT,  -- JSON array
                topics TEXT,    -- JSON array
                indexed_at TEXT,
                model_used TEXT,
                quality_score REAL
            )
        """)

        # 创建全文搜索索引
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts
            USING fts5(memory_id, summary, entities, topics)
        """)

        conn.commit()
        conn.close()

    def index(self, memory_id: str, semantic_index: SemanticIndex) -> None:
        """添加语义索引"""
        import json

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 插入主表
        cursor.execute("""
            INSERT OR REPLACE INTO semantic_indices
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            semantic_index.summary,
            json.dumps(semantic_index.entities),
            json.dumps(semantic_index.topics),
            semantic_index.indexed_at.isoformat(),
            semantic_index.model_used,
            semantic_index.quality_score
        ))

        # 插入全文搜索表
        cursor.execute("""
            INSERT OR REPLACE INTO semantic_fts
            VALUES (?, ?, ?, ?)
        """, (
            memory_id,
            semantic_index.summary,
            ' '.join(semantic_index.entities),
            ' '.join(semantic_index.topics)
        ))

        conn.commit()
        conn.close()

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[tuple]:
        """
        语义检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            (memory_id, summary, score) 列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 全文搜索
        cursor.execute("""
            SELECT memory_id, summary, rank
            FROM semantic_fts
            WHERE semantic_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, top_k))

        results = cursor.fetchall()
        conn.close()

        return results
```

**验收标准**:
- ✅ 检索延迟 < 10ms
- ✅ 全文搜索准确
- ✅ 支持增量更新

---

#### Task 2.4: 语义检索逻辑 (2 天)

**文件**: `llm_compression/memory_search.py`

**实现内容**:
```python
from typing import List
from .stored_memory import StoredMemory
from .semantic_index_db import SemanticIndexDB
from .vector_search import VectorSearch
from .storage import ArrowStorage
from .embedder import LocalEmbedder

class MemorySearch:
    """统一的记忆检索接口"""

    def __init__(self):
        self.semantic_db = SemanticIndexDB()
        self.vector_search = VectorSearch()
        self.storage = ArrowStorage()
        self.embedder = LocalEmbedder()

    def search(self, query: str, top_k: int = 10) -> List[StoredMemory]:
        """
        智能检索（语义优先，向量降级）

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            记忆列表
        """
        # 尝试语义索引检索（快速路径）
        try:
            semantic_results = self.semantic_db.search(query, top_k=top_k * 2)

            if semantic_results:
                # 加载完整记忆
                memories = []
                for memory_id, summary, score in semantic_results[:top_k]:
                    compressed = self.storage.load(memory_id)
                    text = self.storage.decompress(compressed)

                    memory = StoredMemory(
                        id=memory_id,
                        original_compressed=compressed
                    )
                    memories.append(memory)

                return memories

        except Exception as e:
            print(f"Semantic search failed, falling back to vector search: {e}")

        # 降级到向量检索
        query_embedding = self.embedder.encode(query)
        vector_results = self.vector_search.search(query_embedding, top_k=top_k)

        return [memory for memory, score in vector_results]
```

**验收标准**:
- ✅ 语义检索 < 10ms
- ✅ 向量降级正常
- ✅ 结果质量高

---

### Week 3-4 里程碑

**完成标准**:
- ✅ 语义索引系统完整
- ✅ 检索速度 10x 提升
- ✅ API 成本 < $0.5/天
- ✅ 集成测试通过

---

## Week 5-6: 优化与监控

### 目标

生产就绪，建立监控和运维能力。

### 任务清单

#### Task 3.1: 成本监控 (2 天)
#### Task 3.2: 性能优化 (2 天)
#### Task 3.3: 文档完善 (1 天)
#### Task 3.4: 生产部署 (1 天)

---

## 验收标准

### 功能验收

- ✅ 存储延迟 < 15ms
- ✅ 检索延迟 < 10ms (语义) / < 50ms (向量)
- ✅ 压缩比 > 2.5x (Arrow)
- ✅ 原文保真度 100%
- ✅ 索引覆盖率 > 95%

### 成本验收

- ✅ 日均 API 成本 < $1
- ✅ 存储成本增长 < 20%

### 质量验收

- ✅ 测试覆盖率 > 90%
- ✅ 检索准确率 > 85%
- ✅ 系统可用性 > 99.9%

---

## 风险管理

### 已识别风险

| 风险 | 概率 | 影响 | 缓解措施 |
|-----|------|------|---------|
| API 成本超预算 | 中 | 高 | 批量处理 + 成本监控 |
| 索引质量不佳 | 低 | 中 | Prompt 优化 + 质量评估 |
| 性能不达标 | 低 | 高 | 性能测试 + 优化 |

---

## 后续演进

### Phase 2.5: 智能分流（触发条件）

- 日均 API 成本 > $5
- 短文本占比 > 50%
- 重复语义 > 20%

### Phase 3.0: 多模态压缩

- 视频/图像场景描述
- 100-1000x 压缩比

---

**文档版本**: 1.0
**最后更新**: 2026-02-17
**负责人**: AI-OS 团队
**审核状态**: 已批准

"""
存储记忆数据结构

定义 Phase 2.0 的核心数据结构：StoredMemory 和 SemanticIndex。
设计原则：灵活扩展、向后兼容、零迁移成本。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np
import uuid
import json


@dataclass
class Entity:
    """结构化实体"""
    name: str
    type: str  # PERSON, ORG, DATE, NUMBER, TECH, LOCATION, etc.
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'type': self.type,
            'confidence': self.confidence
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Entity':
        return cls(
            name=data['name'],
            type=data['type'],
            confidence=data.get('confidence', 1.0)
        )


@dataclass
class Relation:
    """语义关系"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'confidence': self.confidence
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Relation':
        return cls(
            subject=data['subject'],
            predicate=data['predicate'],
            object=data['object'],
            confidence=data.get('confidence', 1.0)
        )


@dataclass
class SemanticIndex:
    """
    语义索引（后台填充）

    轻量级的语义信息，用于快速检索和理解。
    大小约为原文的 10-20%。
    """

    # 核心摘 bytes）
    summary: str

    # 结构化实体
    entities: List[Entity] = field(default_factory=list)

    # 主题标签
    topics: List[str] = field(default_factory=list)

    # 语义关系（可选）
    relations: List[Relation] = field(default_factory=list)

    # 索引元数据
    indexed_at: datetime = field(default_factory=datetime.now)
    model_used: str = "unknown"
    quality_score: float = 0.0

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'summary': self.summary,
            'entities': [e.to_dict() for e in self.entities],
            'topics': self.topics,
   'relations': [r.to_dict() for r in self.relations],
            'indexed_at': self.indexed_at.isoformat(),
            'model_used': self.model_used,
            'quality_score': self.quality_score
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SemanticIndex':
        """从字典反序列化"""
        return cls(
            summary=data['summary'],
            entities=[Entity.from_dict(e) for e in data.get('entities', [])],
            topics=data.get('topics', []),
            relations=[Relation.from_dict(r) for r in data.get('relations', [])],
            indexed_at=datetime.fromisoformat(data['indexed_at']),
            model_used=data.get('model_used', 'unknown'),
            quality_score=data.get('quality_score', 0.0)
        )

    def to_json(self) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'SemanticIndex':
        """从 JSON 字符串反序列化"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class StoredMemory:
    """
    存储的记忆单元（Phase 2.0 核心数据结构）

    设计原则：
    1. 原文完整保存（Arrow 压缩，100% 保真）
    2. 语义索引可选（后台填充，渐进式增强）
    3. 灵活的 metadata（支持未来演进，零迁移成本）
    """

    # 核心标识
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    # 原文（Arrow 压缩，100% 保真）
    original_compressed: bytes = b""

    # 语义索引（可选，后台填充）
    semantic_index: Optional[SemanticIndex] = None

    # 本地向量（用于向量检索）
    embedding: Optional[np.ndarray] = None

    # 元数据（灵活扩展，支持策略演进）
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Phase 2: Vector Space Compression Data
    sparse_vector: Optional[bytes] = None   # Serialized int8 array
    sparse_indices: Optional[bytes] = None  # Serialized uint16 array
    sparse_meta: Optional[Dict[str, Any]] = None # scale, norm
    key_tokens: List[str] = field(default_factory=list) # Attention-extracted tokens

    def has_semantic_index(self) -> bool:
        """检查是否有语义索引"""
        return self.semantic_index is not None

    def get_compression_strategy(self) -> str:
        """
        获取压缩策略

        Returns:
            策略名称（semantic_index/arrow_only/reference/etc.）
        """
        return self.metadata.get('compression_strategy', 'semantic_index')

    def set_compression_strategy(self, strategy: str) -> None:
        """设置压缩策略（用于策略 D）"""
        self.metadata['compression_strategy'] = strategy

    def to_dict(self) -> dict:
        """
        序列化为字典

        Returns:
            可 JSON 序列化的字典
        """
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'original_compressed': self.original_compressed.hex(),
            'semantic_index': self.semantic_index.to_dict() if self.semantic_index else None,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'metadata': self.metadata,
            'sparse_vector': self.sparse_vector.hex() if self.sparse_vector else None,
            'sparse_indices': self.sparse_indices.hex() if self.sparse_indices else None,
            'sparse_meta': self.sparse_meta,
            'key_tokens': self.key_tokens
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'StoredMemory':
        """
        从字典反序列化

        Args:
            data: 字典数据

        Returns:
            StoredMemory 实例
        """
        return cls(
            id=data['id'],
            created_at=datetime.fromisoformat(data['created_at']),
            original_compressed=bytes.fromhex(data['original_compressed']),
            semantic_index=SemanticIndex.from_dict(data['semantic_index']) if data.get('semantic_index') else None,
            embedding=np.array(data['embedding'], dtype=np.float32) if data.get('embedding') else None,
            metadata=data.get('metadata', {}),
            sparse_vector=bytes.fromhex(data['sparse_vector']) if data.get('sparse_vector') else None,
            sparse_indices=bytes.fromhex(data['sparse_indices']) if data.get('sparse_indices') else None,
            sparse_meta=data.get('sparse_meta'),
            key_tokens=data.get('key_tokens', [])
        )

    def to_json(self) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'StoredMemory':
        """从 JSON 字符串反序列化"""
        return cls.from_dict(json.loads(json_str))

    def get_size_bytes(self) -> int:
        """
        计算存储大小

        Returns:
            总字节数
        """
        size = len(self.original_compressed)

        if self.embedding is not None:
            size += self.embedding.nbytes

        if self.semantic_index:
            # 估算索引大小
            size += len(self.semantic_index.to_json().encode('utf-8'))

        return size

    def __repr__(self) -> str:
        has_index = "✅" if self.has_semantic_index() else "⏳"
        has_embedding = "✅" if self.embedding is not None else "❌"
        return (
            f"StoredMemory(id={self.id[:8]}..., "
            f"size={self.get_size_bytes()}B, "
            f"index={has_index}, "
            f"embedding={has_embedding})"
        )


# 便捷函数
def create_memory(
    text: str,
    storage,
    embedder,
    metadata: Optional[Dict[str, Any]] = None
) -> StoredMemory:
    """
    创建记忆的便捷函数

    Args:
        text: 原始文本
        storage: ArrowStorage 实例
        embedder: EmbeddingProvider 实例
        metadata: 可选的元数据

    Returns:
        StoredMemory 实例
    """
    memory = StoredMemory(
        original_compressed=storage.compress(text),
        embedding=embedder.encode(text),
        metadata=metadata or {}
    )

    return memory

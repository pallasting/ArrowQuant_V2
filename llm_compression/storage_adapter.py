"""
Storage Adapter - 数据结构适配器

在 CompressedMemory 和 StoredMemory 之间转换,实现存储层兼容性。
"""

from typing import Union, Optional
from datetime import datetime
import numpy as np

from llm_compression.compressor import CompressedMemory, CompressionMetadata
from llm_compression.stored_memory import StoredMemory, SemanticIndex, Entity
from llm_compression.logger import logger


class StorageAdapter:
    """
    存储适配器
    
    提供 CompressedMemory 和 StoredMemory 之间的双向转换。
    """
    
    @staticmethod
    def stored_to_compressed(stored: StoredMemory) -> CompressedMemory:
        """
        StoredMemory -> CompressedMemory
        
        Args:
            stored: StoredMemory 实例
            
        Returns:
            CompressedMemory 实例
        """
        # 从 semantic_index 提取实体 (如果有)
        entities = {}
        if stored.semantic_index:
            # 将 Entity 对象转换为字符串列表
            entities = {
                'persons': [e.name for e in stored.semantic_index.entities if e.type == 'PERSON'],
                'locations': [e.name for e in stored.semantic_index.entities if e.type == 'LOCATION'],
                'dates': [e.name for e in stored.semantic_index.entities if e.type == 'DATE'],
                'numbers': [e.name for e in stored.semantic_index.entities if e.type == 'NUMBER'],
                'keywords': stored.semantic_index.topics,
            }
        
        # 转换 embedding
        embedding_list = []
        if stored.embedding is not None:
            if isinstance(stored.embedding, np.ndarray):
                embedding_list = stored.embedding.tolist()
            else:
                embedding_list = list(stored.embedding)
        
        # 创建 compression_metadata
        original_size = len(stored.original_compressed)
        metadata = CompressionMetadata(
            original_size=original_size,
            compressed_size=original_size,
            compression_ratio=1.0,  # Arrow 压缩比率未知
            model_used="arrow",
            quality_score=1.0,  # 无损压缩
            compression_time_ms=0.0,
            compressed_at=stored.created_at
        )
        
        # 创建 CompressedMemory
        compressed = CompressedMemory(
            memory_id=stored.id,
            summary_hash="",  # StoredMemory 没有 summary_hash
            entities=entities,
            diff_data=stored.original_compressed,
            embedding=embedding_list,
            compression_metadata=metadata,
            original_fields=stored.metadata.copy(),
            sparse_vector=stored.sparse_vector,
            sparse_indices=stored.sparse_indices,
            sparse_meta=stored.sparse_meta,
            key_tokens=stored.key_tokens.copy() if stored.key_tokens else [],
            token_scores=[]  # StoredMemory 没有 token_scores
        )
        
        logger.debug(f"Converted StoredMemory -> CompressedMemory: {stored.id}")
        return compressed
    
    @staticmethod
    def compressed_to_stored(compressed: CompressedMemory) -> StoredMemory:
        """
        CompressedMemory -> StoredMemory
        
        Args:
            compressed: CompressedMemory 实例
            
        Returns:
            StoredMemory 实例
        """
        # 构建 semantic_index (如果有实体)
        semantic_index = None
        if compressed.entities or compressed.summary_hash:
            # 从 entities 字典构建 Entity 对象列表
            entity_list = []
            for entity_type, names in compressed.entities.items():
                type_map = {
                    'persons': 'PERSON',
                    'locations': 'LOCATION',
                    'dates': 'DATE',
                    'numbers': 'NUMBER',
                }
                mapped_type = type_map.get(entity_type, entity_type.upper())
                
                for name in names:
                    entity_list.append(Entity(
                        name=name,
                        type=mapped_type,
                        confidence=1.0
                    ))
            
            # 创建 SemanticIndex
            semantic_index = SemanticIndex(
                summary="",  # 需要从 summary_hash 恢复 (如果有缓存)
                entities=entity_list,
                topics=compressed.entities.get('keywords', []),
                relations=[],
                indexed_at=compressed.compression_metadata.compressed_at,
                model_used=compressed.compression_metadata.model_used,
                quality_score=compressed.compression_metadata.quality_score
            )
        
        # 转换 embedding
        embedding_array = None
        if compressed.embedding:
            embedding_array = np.array(compressed.embedding, dtype=np.float32)
        
        # 创建 StoredMemory
        stored = StoredMemory(
            id=compressed.memory_id,
            created_at=compressed.compression_metadata.compressed_at,
            original_compressed=compressed.diff_data,
            semantic_index=semantic_index,
            embedding=embedding_array,
            metadata=compressed.original_fields.copy(),
            sparse_vector=compressed.sparse_vector,
            sparse_indices=compressed.sparse_indices,
            sparse_meta=compressed.sparse_meta,
            key_tokens=compressed.key_tokens.copy() if compressed.key_tokens else []
        )
        
        logger.debug(f"Converted CompressedMemory -> StoredMemory: {compressed.memory_id}")
        return stored
    
    @staticmethod
    def normalize_memory(memory: Union[CompressedMemory, StoredMemory]) -> CompressedMemory:
        """
        统一转换为 CompressedMemory (用于存储)
        
        Args:
            memory: CompressedMemory 或 StoredMemory
            
        Returns:
            CompressedMemory 实例
        """
        if isinstance(memory, StoredMemory):
            return StorageAdapter.stored_to_compressed(memory)
        elif isinstance(memory, CompressedMemory):
            return memory
        else:
            raise TypeError(f"Unsupported memory type: {type(memory)}")

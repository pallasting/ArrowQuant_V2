"""
Unit tests for StorageAdapter

Tests bidirectional conversion between CompressedMemory and StoredMemory.
"""

import pytest
import numpy as np
from datetime import datetime

from llm_compression.storage_adapter import StorageAdapter
from llm_compression.compressor import CompressedMemory, CompressionMetadata
from llm_compression.stored_memory import StoredMemory, SemanticIndex, Entity


class TestStorageAdapter:
    """Test StorageAdapter conversion methods"""
    
    def test_stored_to_compressed_basic(self):
        """Test basic StoredMemory -> CompressedMemory conversion"""
        # Create StoredMemory
        stored = StoredMemory(
            id="test_001",
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            original_compressed=b"test data",
            embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            metadata={"source": "test"},
            key_tokens=["test", "data"]
        )
        
        # Convert
        compressed = StorageAdapter.stored_to_compressed(stored)
        
        # Verify
        assert compressed.memory_id == "test_001"
        assert compressed.diff_data == b"test data"
        # Use approximate comparison for float32 precision
        assert len(compressed.embedding) == 3
        assert abs(compressed.embedding[0] - 0.1) < 0.001
        assert abs(compressed.embedding[1] - 0.2) < 0.001
        assert abs(compressed.embedding[2] - 0.3) < 0.001
        assert compressed.original_fields == {"source": "test"}
        assert compressed.key_tokens == ["test", "data"]
        assert compressed.compression_metadata.model_used == "arrow"
        assert compressed.compression_metadata.compression_ratio == 1.0
    
    def test_stored_to_compressed_with_semantic_index(self):
        """Test conversion with semantic index"""
        # Create semantic index
        semantic_index = SemanticIndex(
            summary="Test summary",
            entities=[
                Entity(name="Alice", type="PERSON", confidence=0.9),
                Entity(name="New York", type="LOCATION", confidence=0.85),
                Entity(name="2024-01-15", type="DATE", confidence=0.95),
                Entity(name="42", type="NUMBER", confidence=0.8)
            ],
            topics=["AI", "ML"],
            indexed_at=datetime(2024, 1, 15, 10, 30, 0),
            model_used="test_model",
            quality_score=0.9
        )
        
        # Create StoredMemory
        stored = StoredMemory(
            id="test_002",
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            original_compressed=b"test data with entities",
            semantic_index=semantic_index,
            embedding=np.array([0.1, 0.2], dtype=np.float32),
            metadata={}
        )
        
        # Convert
        compressed = StorageAdapter.stored_to_compressed(stored)
        
        # Verify entities extraction
        assert "persons" in compressed.entities
        assert "Alice" in compressed.entities["persons"]
        assert "locations" in compressed.entities
        assert "New York" in compressed.entities["locations"]
        assert "dates" in compressed.entities
        assert "2024-01-15" in compressed.entities["dates"]
        assert "numbers" in compressed.entities
        assert "42" in compressed.entities["numbers"]
        assert "keywords" in compressed.entities
        assert compressed.entities["keywords"] == ["AI", "ML"]
    
    def test_compressed_to_stored_basic(self):
        """Test basic CompressedMemory -> StoredMemory conversion"""
        # Create CompressedMemory
        compressed = CompressedMemory(
            memory_id="test_003",
            summary_hash="abc123",
            entities={
                "persons": ["Bob"],
                "keywords": ["test"]
            },
            diff_data=b"compressed data",
            embedding=[0.5, 0.6, 0.7],
            compression_metadata=CompressionMetadata(
                original_size=100,
                compressed_size=50,
                compression_ratio=2.0,
                model_used="gpt-4",
                quality_score=0.95,
                compression_time_ms=100.0,
                compressed_at=datetime(2024, 1, 15, 10, 30, 0)
            ),
            original_fields={"source": "test"},
            key_tokens=["compressed", "data"]
        )
        
        # Convert
        stored = StorageAdapter.compressed_to_stored(compressed)
        
        # Verify
        assert stored.id == "test_003"
        assert stored.original_compressed == b"compressed data"
        assert np.array_equal(stored.embedding, np.array([0.5, 0.6, 0.7], dtype=np.float32))
        assert stored.metadata == {"source": "test"}
        assert stored.key_tokens == ["compressed", "data"]
        assert stored.created_at == datetime(2024, 1, 15, 10, 30, 0)
    
    def test_compressed_to_stored_with_entities(self):
        """Test conversion with entities to semantic index"""
        # Create CompressedMemory with entities
        compressed = CompressedMemory(
            memory_id="test_004",
            summary_hash="def456",
            entities={
                "persons": ["Alice", "Bob"],
                "locations": ["Paris"],
                "dates": ["2024-01-15"],
                "numbers": ["100"],
                "keywords": ["AI", "ML"]
            },
            diff_data=b"data",
            embedding=[0.1, 0.2],
            compression_metadata=CompressionMetadata(
                original_size=100,
                compressed_size=50,
                compression_ratio=2.0,
                model_used="gpt-4",
                quality_score=0.9,
                compression_time_ms=50.0,
                compressed_at=datetime(2024, 1, 15, 10, 30, 0)
            )
        )
        
        # Convert
        stored = StorageAdapter.compressed_to_stored(compressed)
        
        # Verify semantic index
        assert stored.semantic_index is not None
        # 2 persons + 1 location + 1 date + 1 number + 2 keywords = 7 entities
        assert len(stored.semantic_index.entities) == 7
        
        # Check entity types
        entity_names = [e.name for e in stored.semantic_index.entities]
        assert "Alice" in entity_names
        assert "Bob" in entity_names
        assert "Paris" in entity_names
        assert "2024-01-15" in entity_names
        assert "100" in entity_names
        assert "AI" in entity_names
        assert "ML" in entity_names
        
        # Check topics (keywords are also added as topics)
        assert stored.semantic_index.topics == ["AI", "ML"]
    
    def test_normalize_memory_stored(self):
        """Test normalize_memory with StoredMemory"""
        stored = StoredMemory(
            id="test_005",
            created_at=datetime.now(),
            original_compressed=b"test",
            embedding=np.array([0.1], dtype=np.float32)
        )
        
        result = StorageAdapter.normalize_memory(stored)
        
        assert isinstance(result, CompressedMemory)
        assert result.memory_id == "test_005"
    
    def test_normalize_memory_compressed(self):
        """Test normalize_memory with CompressedMemory"""
        compressed = CompressedMemory(
            memory_id="test_006",
            summary_hash="",
            entities={},
            diff_data=b"test",
            embedding=[0.1],
            compression_metadata=CompressionMetadata(
                original_size=10,
                compressed_size=10,
                compression_ratio=1.0,
                model_used="test",
                quality_score=1.0,
                compression_time_ms=0.0,
                compressed_at=datetime.now()
            )
        )
        
        result = StorageAdapter.normalize_memory(compressed)
        
        assert isinstance(result, CompressedMemory)
        assert result is compressed  # Should return same object
    
    def test_normalize_memory_invalid_type(self):
        """Test normalize_memory with invalid type"""
        with pytest.raises(TypeError):
            StorageAdapter.normalize_memory("invalid")
    
    def test_roundtrip_conversion(self):
        """Test StoredMemory -> CompressedMemory -> StoredMemory roundtrip"""
        # Create original StoredMemory
        original = StoredMemory(
            id="test_007",
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            original_compressed=b"roundtrip test data",
            embedding=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            metadata={"source": "roundtrip", "version": 1},
            key_tokens=["roundtrip", "test"]
        )
        
        # Convert to CompressedMemory
        compressed = StorageAdapter.stored_to_compressed(original)
        
        # Convert back to StoredMemory
        restored = StorageAdapter.compressed_to_stored(compressed)
        
        # Verify key fields match
        assert restored.id == original.id
        assert restored.original_compressed == original.original_compressed
        assert np.array_equal(restored.embedding, original.embedding)
        assert restored.metadata == original.metadata
        assert restored.key_tokens == original.key_tokens
        assert restored.created_at == original.created_at
    
    def test_vector_compression_fields(self):
        """Test conversion preserves vector compression fields"""
        # Create StoredMemory with vector compression data
        stored = StoredMemory(
            id="test_008",
            created_at=datetime.now(),
            original_compressed=b"test",
            embedding=np.array([0.1], dtype=np.float32),
            sparse_vector=b"\x01\x02\x03",
            sparse_indices=b"\x04\x05\x06",
            sparse_meta={
                "scale_factor": 1.5,
                "full_dim": 384,
                "original_norm": 2.0,
                "is_4bit": True,
                "packed_length": 96
            },
            key_tokens=["test"]
        )
        
        # Convert
        compressed = StorageAdapter.stored_to_compressed(stored)
        
        # Verify vector compression fields
        assert compressed.sparse_vector == b"\x01\x02\x03"
        assert compressed.sparse_indices == b"\x04\x05\x06"
        assert compressed.sparse_meta == stored.sparse_meta
        assert compressed.key_tokens == ["test"]
    
    def test_empty_embedding(self):
        """Test conversion with None embedding"""
        stored = StoredMemory(
            id="test_009",
            created_at=datetime.now(),
            original_compressed=b"test",
            embedding=None
        )
        
        compressed = StorageAdapter.stored_to_compressed(stored)
        
        assert compressed.embedding == []
    
    def test_empty_entities(self):
        """Test conversion with no semantic index"""
        stored = StoredMemory(
            id="test_010",
            created_at=datetime.now(),
            original_compressed=b"test",
            semantic_index=None
        )
        
        compressed = StorageAdapter.stored_to_compressed(stored)
        
        assert compressed.entities == {}

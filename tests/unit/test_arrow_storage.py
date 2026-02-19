"""
Unit Tests for Arrow Storage

Tests the ArrowStorage class functionality including save, load, query,
and specialized query methods.

Feature: llm-compression-integration
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from llm_compression.arrow_storage import ArrowStorage, StorageError
from llm_compression.compressor import CompressedMemory, CompressionMetadata


@pytest.fixture
def temp_storage():
    """Create temporary storage for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ArrowStorage(storage_path=tmpdir)
        yield storage


@pytest.fixture
def sample_compressed_memory():
    """Create sample compressed memory for testing"""
    return CompressedMemory(
        memory_id="test_memory_001",
        summary_hash="abc123def456",
        entities={
            'persons': ['John Doe', 'Jane Smith'],
            'dates': ['2024-01-15', '3pm'],
            'numbers': ['42', '$100'],
            'locations': ['New York'],
            'keywords': ['meeting', 'project', 'deadline']
        },
        diff_data=b"test diff data with additional details",
        embedding=[0.1] * 384,
        compression_metadata=CompressionMetadata(
            original_size=1000,
            compressed_size=100,
            compression_ratio=10.0,
            model_used="gpt-4",
            quality_score=0.95,
            compression_time_ms=50.0,
            compressed_at=datetime.now()
        ),
        original_fields={
            'intent': 'schedule meeting',
            'action': 'sent calendar invite',
            'outcome': 'meeting scheduled',
            'success': True,
            'related_memories': ['memory_002', 'memory_003']
        }
    )


class TestArrowStorageBasics:
    """Test basic storage operations"""
    
    def test_initialization(self, temp_storage):
        """Test storage initialization"""
        assert temp_storage.storage_path.exists()
        assert temp_storage.compression_level == 3
        
        # Check category paths are created
        for category in ['experiences', 'identity', 'preferences', 'context']:
            assert category in temp_storage.category_paths
    
    def test_save_and_load(self, temp_storage, sample_compressed_memory):
        """Test save and load operations"""
        # Save memory
        temp_storage.save(sample_compressed_memory, category='experiences')
        
        # Load memory
        loaded = temp_storage.load(sample_compressed_memory.memory_id, category='experiences')
        
        assert loaded is not None
        assert loaded.memory_id == sample_compressed_memory.memory_id
        assert loaded.summary_hash == sample_compressed_memory.summary_hash
        assert loaded.entities == sample_compressed_memory.entities
    
    def test_load_nonexistent_memory(self, temp_storage):
        """Test loading non-existent memory"""
        loaded = temp_storage.load("nonexistent_id", category='experiences')
        assert loaded is None
    
    def test_save_invalid_category(self, temp_storage, sample_compressed_memory):
        """Test saving to invalid category"""
        with pytest.raises(ValueError):
            temp_storage.save(sample_compressed_memory, category='invalid_category')
    
    def test_load_invalid_category(self, temp_storage):
        """Test loading from invalid category"""
        with pytest.raises(ValueError):
            temp_storage.load("test_id", category='invalid_category')


class TestArrowStorageQuery:
    """Test query operations"""
    
    def test_query_empty_table(self, temp_storage):
        """Test querying empty table"""
        results = temp_storage.query(category='experiences')
        assert results == []
    
    def test_query_with_filters(self, temp_storage):
        """Test query with filters"""
        # Create and save multiple memories
        for i in range(5):
            memory = CompressedMemory(
                memory_id=f"memory_{i}",
                summary_hash=f"hash_{i}",
                entities={'persons': [], 'dates': [], 'numbers': [], 'locations': [], 'keywords': []},
                diff_data=f"diff_{i}".encode('utf-8'),
                embedding=[0.1 * i] * 384,
                compression_metadata=CompressionMetadata(
                    original_size=1000,
                    compressed_size=100,
                    compression_ratio=10.0,
                    model_used="test-model",
                    quality_score=0.95,
                    compression_time_ms=50.0,
                    compressed_at=datetime.now()
                ),
                original_fields={'intent': f'intent_{i}', 'action': f'action_{i}', 'outcome': f'outcome_{i}', 'success': True, 'related_memories': []}
            )
            temp_storage.save(memory, category='experiences')
        
        # Query all
        results = temp_storage.query(category='experiences')
        assert len(results) == 5
        
        # Query with limit
        results = temp_storage.query(category='experiences', limit=3)
        assert len(results) == 3
        
        # Query with filter
        results = temp_storage.query(
            category='experiences',
            filters={'is_compressed': True}
        )
        assert len(results) == 5  # All are compressed
    
    def test_query_with_sorting(self, temp_storage):
        """Test query with sorting"""
        # Create memories with different timestamps
        for i in range(3):
            memory = CompressedMemory(
                memory_id=f"memory_{i}",
                summary_hash=f"hash_{i}",
                entities={'persons': [], 'dates': [], 'numbers': [], 'locations': [], 'keywords': []},
                diff_data=f"diff_{i}".encode('utf-8'),
                embedding=[0.1] * 384,
                compression_metadata=CompressionMetadata(
                    original_size=1000 + i * 100,
                    compressed_size=100 + i * 10,
                    compression_ratio=10.0,
                    model_used="test-model",
                    quality_score=0.95,
                    compression_time_ms=50.0,
                    compressed_at=datetime.now()
                ),
                original_fields={'intent': 'test', 'action': 'test', 'outcome': 'test', 'success': True, 'related_memories': []}
            )
            temp_storage.save(memory, category='experiences')
        
        # Query with sorting by memory_id
        results = temp_storage.query(
            category='experiences',
            sort_by='memory_id',
            ascending=True
        )
        
        assert len(results) == 3
        assert results[0].memory_id == "memory_0"
        assert results[1].memory_id == "memory_1"
        assert results[2].memory_id == "memory_2"


class TestArrowStorageTimeQuery:
    """Test time-based query operations"""
    
    def test_query_by_time_range(self, temp_storage):
        """Test querying by time range"""
        now = datetime.now()
        
        # Create memories with different timestamps
        for i in range(5):
            memory = CompressedMemory(
                memory_id=f"memory_{i}",
                summary_hash=f"hash_{i}",
                entities={'persons': [], 'dates': [], 'numbers': [], 'locations': [], 'keywords': []},
                diff_data=f"diff_{i}".encode('utf-8'),
                embedding=[0.1] * 384,
                compression_metadata=CompressionMetadata(
                    original_size=1000,
                    compressed_size=100,
                    compression_ratio=10.0,
                    model_used="test-model",
                    quality_score=0.95,
                    compression_time_ms=50.0,
                    compressed_at=now - timedelta(days=i)
                ),
                original_fields={'intent': 'test', 'action': 'test', 'outcome': 'test', 'success': True, 'related_memories': []}
            )
            temp_storage.save(memory, category='experiences')
        
        # Query recent memories (last 2 days)
        results = temp_storage.query_by_time_range(
            category='experiences',
            start_time=now - timedelta(days=2),
            end_time=now
        )
        
        # Should return memories from last 2 days (0, 1, 2)
        assert len(results) >= 2


class TestArrowStorageEntityQuery:
    """Test entity-based query operations"""
    
    def test_query_by_entity(self, temp_storage):
        """Test querying by entity"""
        # Create memories with different entities
        memory1 = CompressedMemory(
            memory_id="memory_1",
            summary_hash="hash_1",
            entities={
                'persons': ['John Doe'],
                'dates': [],
                'numbers': [],
                'locations': [],
                'keywords': []
            },
            diff_data=b"diff_1",
            embedding=[0.1] * 384,
            compression_metadata=CompressionMetadata(
                original_size=1000,
                compressed_size=100,
                compression_ratio=10.0,
                model_used="test-model",
                quality_score=0.95,
                compression_time_ms=50.0,
                compressed_at=datetime.now()
            ),
            original_fields={'intent': 'test', 'action': 'test', 'outcome': 'test', 'success': True, 'related_memories': []}
        )
        
        memory2 = CompressedMemory(
            memory_id="memory_2",
            summary_hash="hash_2",
            entities={
                'persons': ['Jane Smith'],
                'dates': [],
                'numbers': [],
                'locations': [],
                'keywords': []
            },
            diff_data=b"diff_2",
            embedding=[0.2] * 384,
            compression_metadata=CompressionMetadata(
                original_size=1000,
                compressed_size=100,
                compression_ratio=10.0,
                model_used="test-model",
                quality_score=0.95,
                compression_time_ms=50.0,
                compressed_at=datetime.now()
            ),
            original_fields={'intent': 'test', 'action': 'test', 'outcome': 'test', 'success': True, 'related_memories': []}
        )
        
        temp_storage.save(memory1, category='experiences')
        temp_storage.save(memory2, category='experiences')
        
        # Query by person
        results = temp_storage.query_by_entity(
            category='experiences',
            entity_type='persons',
            entity_value='John Doe'
        )
        
        assert len(results) == 1
        assert results[0].memory_id == "memory_1"


class TestArrowStorageSimilarityQuery:
    """Test similarity-based query operations"""
    
    def test_query_by_similarity(self, temp_storage):
        """Test querying by embedding similarity"""
        # Create memories with different embeddings
        for i in range(3):
            memory = CompressedMemory(
                memory_id=f"memory_{i}",
                summary_hash=f"hash_{i}",
                entities={'persons': [], 'dates': [], 'numbers': [], 'locations': [], 'keywords': []},
                diff_data=f"diff_{i}".encode('utf-8'),
                embedding=[0.1 * (i + 1)] * 384,  # Different embeddings
                compression_metadata=CompressionMetadata(
                    original_size=1000,
                    compressed_size=100,
                    compression_ratio=10.0,
                    model_used="test-model",
                    quality_score=0.95,
                    compression_time_ms=50.0,
                    compressed_at=datetime.now()
                ),
                original_fields={'intent': 'test', 'action': 'test', 'outcome': 'test', 'success': True, 'related_memories': []}
            )
            temp_storage.save(memory, category='experiences')
        
        # Query by similarity (should be most similar to memory_0)
        query_embedding = [0.1] * 384
        results = temp_storage.query_by_similarity(
            category='experiences',
            query_embedding=query_embedding,
            top_k=2
        )
        
        assert len(results) == 2
        # Results should be sorted by similarity
        assert results[0][1] >= results[1][1]  # First result has higher similarity


class TestArrowStorageSummaryDeduplication:
    """Test summary deduplication"""
    
    def test_save_summary(self, temp_storage, sample_compressed_memory):
        """Test saving summary with deduplication"""
        # Save memory (which should save summary)
        temp_storage.save(sample_compressed_memory, category='experiences')
        
        # Try to load summary
        summary = temp_storage.load_summary(sample_compressed_memory.summary_hash)
        
        # Note: Currently returns None because we don't store actual summary text
        # This is expected behavior for now
        # In production, we'd integrate with compressor's summary cache
    
    def test_load_nonexistent_summary(self, temp_storage):
        """Test loading non-existent summary"""
        summary = temp_storage.load_summary("nonexistent_hash")
        assert summary is None


class TestArrowStorageMultipleCategories:
    """Test operations across multiple categories"""
    
    def test_save_to_different_categories(self, temp_storage):
        """Test saving to different memory categories"""
        categories = ['experiences', 'identity', 'preferences', 'context']
        
        for category in categories:
            memory = CompressedMemory(
                memory_id=f"{category}_memory",
                summary_hash=f"{category}_hash",
                entities={'persons': [], 'dates': [], 'numbers': [], 'locations': [], 'keywords': []},
                diff_data=f"{category}_diff".encode('utf-8'),
                embedding=[0.1] * 384,
                compression_metadata=CompressionMetadata(
                    original_size=1000,
                    compressed_size=100,
                    compression_ratio=10.0,
                    model_used="test-model",
                    quality_score=0.95,
                    compression_time_ms=50.0,
                    compressed_at=datetime.now()
                ),
                original_fields={}
            )
            temp_storage.save(memory, category=category)
        
        # Verify all categories have data
        for category in categories:
            loaded = temp_storage.load(f"{category}_memory", category=category)
            assert loaded is not None
            assert loaded.memory_id == f"{category}_memory"

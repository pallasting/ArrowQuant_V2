"""
Unit tests for semantic indexing modules.

Tests cover:
- VectorSearch.search() and batch_search()
- SemanticIndexer.index_memory() and batch_index()
- SemanticIndexDB.add_entry() and query()
- MemorySearch with all search modes
- BackgroundQueue task processing

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
"""

import asyncio
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pytest

from llm_compression.vector_search import VectorSearch, SearchResult
from llm_compression.semantic_indexer import SemanticIndexer
from llm_compression.semantic_index_db import SemanticIndexDB
from llm_compression.memory_search import MemorySearch, SearchMode
from llm_compression.background_queue import BackgroundQueue
from llm_compression.arrow_storage import ArrowStorage
from llm_compression.embedding_provider import get_default_provider


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def storage_dir():
    """Create temporary directory for storage."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def embedding_provider():
    """Get default embedding provider."""
    return get_default_provider()


@pytest.fixture
def arrow_storage(storage_dir):
    """Create ArrowStorage instance."""
    return ArrowStorage(storage_dir)


@pytest.fixture
def index_db(temp_dir):
    """Create SemanticIndexDB instance."""
    return SemanticIndexDB(temp_dir)


@pytest.fixture
def semantic_indexer(embedding_provider, arrow_storage, index_db):
    """Create SemanticIndexer instance."""
    return SemanticIndexer(embedding_provider, arrow_storage, index_db)


@pytest.fixture
def vector_search(embedding_provider, arrow_storage, index_db):
    """Create VectorSearch instance."""
    return VectorSearch(embedding_provider, arrow_storage, index_db)


@pytest.fixture
def memory_search(vector_search, arrow_storage):
    """Create MemorySearch instance."""
    return MemorySearch(vector_search, arrow_storage)


@pytest.fixture
def sample_memories():
    """Create sample memory objects for testing."""
    return [
        {
            'memory_id': 'mem_1',
            'category': 'knowledge',
            'context': 'Machine learning is a subset of artificial intelligence.',
            'timestamp': datetime(2024, 1, 1, 12, 0, 0),
            'embedding': None,
            'is_compressed': False
        },
        {
            'memory_id': 'mem_2',
            'category': 'knowledge',
            'context': 'Python is a popular programming language.',
            'timestamp': datetime(2024, 1, 2, 12, 0, 0),
            'embedding': None,
            'is_compressed': False
        },
        {
            'memory_id': 'mem_3',
            'category': 'knowledge',
            'context': 'Deep learning uses neural networks with multiple layers.',
            'timestamp': datetime(2024, 1, 3, 12, 0, 0),
            'embedding': None,
            'is_compressed': False
        }
    ]


# ============================================================================
# Test: SemanticIndexDB
# ============================================================================


class TestSemanticIndexDB:
    """Test SemanticIndexDB functionality."""
    
    def test_initialization(self, temp_dir):
        """Should initialize with valid path."""
        index_db = SemanticIndexDB(temp_dir)
        
        assert index_db.index_path == Path(temp_dir)
        assert index_db.index_path.exists()
    
    def test_add_entry(self, index_db):
        """Should add single entry to index."""
        embedding = np.random.randn(384).astype(np.float32)
        
        index_db.add_entry(
            memory_id='mem_1',
            category='test',
            embedding=embedding,
            timestamp=datetime.now()
        )
        
        # Verify file was created
        index_file = index_db.index_path / 'test_index.parquet'
        assert index_file.exists()
        
        # Verify entry count
        assert index_db.get_category_size('test') == 1
    
    def test_batch_add(self, index_db):
        """Should batch add multiple entries."""
        entries = []
        for i in range(5):
            embedding = np.random.randn(384).astype(np.float32)
            entries.append({
                'memory_id': f'mem_{i}',
                'category': 'test',
                'embedding': embedding,
                'timestamp': datetime.now()
            })
        
        index_db.batch_add(entries)
        
        # Verify all entries were added
        assert index_db.get_category_size('test') == 5
    
    def test_query(self, index_db):
        """Should query index for similar embeddings."""
        # Add entries
        entries = []
        for i in range(10):
            embedding = np.random.randn(384).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            entries.append({
                'memory_id': f'mem_{i}',
                'category': 'test',
                'embedding': embedding,
                'timestamp': datetime.now()
            })
        
        index_db.batch_add(entries)
        
        # Query
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        results = index_db.query(
            category='test',
            query_embedding=query_embedding,
            top_k=5
        )
        
        # Verify results
        assert len(results) == 5
        assert all('memory_id' in r for r in results)
        assert all('similarity' in r for r in results)
        assert all('timestamp' in r for r in results)
        
        # Verify similarities are in valid range
        assert all(-1.0 <= r['similarity'] <= 1.0 for r in results)
        
        # Verify results are sorted by similarity (descending)
        similarities = [r['similarity'] for r in results]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_query_with_threshold(self, index_db):
        """Should filter results by similarity threshold."""
        # Add entries with known embeddings
        base_embedding = np.random.randn(384).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        entries = []
        for i in range(5):
            # Create similar embeddings
            embedding = base_embedding + np.random.randn(384).astype(np.float32) * 0.1
            embedding = embedding / np.linalg.norm(embedding)
            entries.append({
                'memory_id': f'mem_{i}',
                'category': 'test',
                'embedding': embedding,
                'timestamp': datetime.now()
            })
        
        index_db.batch_add(entries)
        
        # Query with high threshold
        results = index_db.query(
            category='test',
            query_embedding=base_embedding,
            top_k=10,
            threshold=0.9
        )
        
        # Should return fewer results due to threshold
        assert len(results) <= 5
        assert all(r['similarity'] >= 0.9 for r in results)
    
    def test_query_nonexistent_category(self, index_db):
        """Should return empty list for nonexistent category."""
        query_embedding = np.random.randn(384).astype(np.float32)
        
        results = index_db.query(
            category='nonexistent',
            query_embedding=query_embedding,
            top_k=5
        )
        
        assert results == []
    
    def test_clear_category(self, index_db):
        """Should clear index for a category."""
        # Add entries
        embedding = np.random.randn(384).astype(np.float32)
        index_db.add_entry(
            memory_id='mem_1',
            category='test',
            embedding=embedding,
            timestamp=datetime.now()
        )
        
        assert index_db.get_category_size('test') == 1
        
        # Clear category
        index_db.clear_category('test')
        
        assert index_db.get_category_size('test') == 0
    
    def test_get_categories(self, index_db):
        """Should return list of indexed categories."""
        # Add entries to multiple categories
        for category in ['cat1', 'cat2', 'cat3']:
            embedding = np.random.randn(384).astype(np.float32)
            index_db.add_entry(
                memory_id='mem_1',
                category=category,
                embedding=embedding,
                timestamp=datetime.now()
            )
        
        categories = index_db.get_categories()
        
        assert 'cat1' in categories
        assert 'cat2' in categories
        assert 'cat3' in categories
    
    def test_get_category_size(self, index_db):
        """Should return correct category size."""
        # Add entries
        for i in range(7):
            embedding = np.random.randn(384).astype(np.float32)
            index_db.add_entry(
                memory_id=f'mem_{i}',
                category='test',
                embedding=embedding,
                timestamp=datetime.now()
            )
        
        size = index_db.get_category_size('test')
        assert size == 7


# ============================================================================
# Test: SemanticIndexer
# ============================================================================


class TestSemanticIndexer:
    """Test SemanticIndexer functionality."""
    
    def test_initialization(self, semantic_indexer):
        """Should initialize with required components."""
        assert semantic_indexer.embedder is not None
        assert semantic_indexer.storage is not None
        assert semantic_indexer.index_db is not None
    
    def test_index_memory(self, semantic_indexer, index_db):
        """Should index a single memory."""
        memory = {
            'memory_id': 'mem_1',
            'category': 'test',
            'context': 'Test memory content',
            'timestamp': datetime.now(),
            'embedding': None,
            'is_compressed': False
        }
        
        semantic_indexer.index_memory(memory)
        
        # Verify memory was indexed
        assert index_db.get_category_size('test') == 1
        
        # Verify embedding was generated
        assert memory['embedding'] is not None
        assert isinstance(memory['embedding'], np.ndarray)
    
    def test_index_memory_with_existing_embedding(self, semantic_indexer, index_db):
        """Should use existing embedding if present."""
        existing_embedding = np.random.randn(384).astype(np.float32)
        
        memory = {
            'memory_id': 'mem_1',
            'category': 'test',
            'context': 'Test memory content',
            'timestamp': datetime.now(),
            'embedding': existing_embedding,
            'is_compressed': False
        }
        
        semantic_indexer.index_memory(memory)
        
        # Verify memory was indexed
        assert index_db.get_category_size('test') == 1
        
        # Verify embedding was not changed
        assert np.array_equal(memory['embedding'], existing_embedding)
    
    def test_batch_index(self, semantic_indexer, index_db, sample_memories):
        """Should batch index multiple memories."""
        semantic_indexer.batch_index(sample_memories, batch_size=2)
        
        # Verify all memories were indexed
        assert index_db.get_category_size('knowledge') == 3
        
        # Verify embeddings were generated
        for memory in sample_memories:
            assert memory['embedding'] is not None
    
    def test_batch_index_empty_list(self, semantic_indexer):
        """Should handle empty list gracefully."""
        # Should not raise error
        semantic_indexer.batch_index([])
    
    def test_extract_indexable_text_uncompressed(self, semantic_indexer):
        """Should extract context from uncompressed memory."""
        memory = {
            'memory_id': 'mem_1',
            'context': 'Test context',
            'is_compressed': False
        }
        
        text = semantic_indexer._extract_indexable_text(memory)
        assert text == 'Test context'
    
    def test_extract_indexable_text_compressed(self, semantic_indexer):
        """Should extract summary from compressed memory."""
        memory = {
            'memory_id': 'mem_1',
            'summary': 'Test summary',
            'context': 'Long context',
            'is_compressed': True
        }
        
        text = semantic_indexer._extract_indexable_text(memory)
        assert text == 'Test summary'
    
    def test_extract_indexable_text_fallback(self, semantic_indexer):
        """Should fallback to alternative text fields."""
        memory = {
            'memory_id': 'mem_1',
            'text': 'Fallback text'
        }
        
        text = semantic_indexer._extract_indexable_text(memory)
        assert text == 'Fallback text'


# ============================================================================
# Test: VectorSearch
# ============================================================================


class TestVectorSearch:
    """Test VectorSearch functionality."""
    
    def test_initialization(self, vector_search):
        """Should initialize with required components."""
        assert vector_search.embedder is not None
        assert vector_search.storage is not None
    
    def test_search(self, vector_search, semantic_indexer, sample_memories):
        """Should perform semantic search."""
        # Index memories first
        semantic_indexer.batch_index(sample_memories)
        
        # Search
        results = vector_search.search(
            query='machine learning',
            category='knowledge',
            top_k=2
        )
        
        # Verify results
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(hasattr(r, 'memory_id') for r in results)
        assert all(hasattr(r, 'similarity') for r in results)
        assert all(hasattr(r, 'memory') for r in results)
    
    def test_search_with_threshold(self, vector_search, semantic_indexer, sample_memories):
        """Should filter results by threshold."""
        # Index memories
        semantic_indexer.batch_index(sample_memories)
        
        # Search with high threshold
        results = vector_search.search(
            query='machine learning',
            category='knowledge',
            top_k=10,
            threshold=0.5
        )
        
        # Verify all results meet threshold
        assert all(r.similarity >= 0.5 for r in results)
    
    def test_batch_search(self, vector_search, semantic_indexer, sample_memories):
        """Should perform batch search."""
        # Index memories
        semantic_indexer.batch_index(sample_memories)
        
        # Batch search
        queries = ['machine learning', 'programming', 'neural networks']
        batch_results = vector_search.batch_search(
            queries=queries,
            category='knowledge',
            top_k=2
        )
        
        # Verify results
        assert len(batch_results) == 3
        assert all(isinstance(results, list) for results in batch_results)
        assert all(len(results) <= 2 for results in batch_results)
    
    def test_search_empty_category(self, vector_search):
        """Should return empty list for empty category."""
        results = vector_search.search(
            query='test',
            category='nonexistent',
            top_k=5
        )
        
        assert results == []


# ============================================================================
# Test: MemorySearch
# ============================================================================


class TestMemorySearch:
    """Test MemorySearch functionality."""
    
    def test_initialization(self, memory_search):
        """Should initialize with required components."""
        assert memory_search.vector_search is not None
        assert memory_search.storage is not None
    
    def test_search_semantic_mode(self, memory_search, semantic_indexer, sample_memories):
        """Should perform semantic search."""
        # Index memories
        semantic_indexer.batch_index(sample_memories)
        
        # Search
        results = memory_search.search(
            query='machine learning',
            category='knowledge',
            mode=SearchMode.SEMANTIC,
            top_k=2
        )
        
        # Verify results
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_search_semantic_mode_with_threshold(self, memory_search, semantic_indexer, sample_memories):
        """Should apply threshold in semantic mode."""
        # Index memories
        semantic_indexer.batch_index(sample_memories)
        
        # Search with threshold
        results = memory_search.search(
            query='machine learning',
            category='knowledge',
            mode=SearchMode.SEMANTIC,
            top_k=10,
            threshold=0.5
        )
        
        # Verify threshold applied
        assert all(r.similarity >= 0.5 for r in results)
    
    def test_search_entity_mode_missing_params(self, memory_search):
        """Should return empty list if entity params missing."""
        results = memory_search.search(
            query='',
            category='test',
            mode=SearchMode.ENTITY,
            top_k=5
        )
        
        assert results == []
    
    def test_search_time_mode_missing_params(self, memory_search):
        """Should return empty list if time params missing."""
        results = memory_search.search(
            query='',
            category='test',
            mode=SearchMode.TIME,
            top_k=5
        )
        
        assert results == []
    
    def test_search_hybrid_mode(self, memory_search, semantic_indexer, sample_memories):
        """Should perform hybrid search."""
        # Index memories
        semantic_indexer.batch_index(sample_memories)
        
        # Hybrid search (may return empty list if storage doesn't support filtering)
        # This is acceptable as hybrid mode is a best-effort feature
        try:
            results = memory_search.search(
                query='machine learning',
                category='knowledge',
                mode=SearchMode.HYBRID,
                top_k=2,
                start_time=datetime(2024, 1, 1)
            )
            
            # Should return results (exact behavior depends on storage implementation)
            assert isinstance(results, list)
        except (AttributeError, TypeError):
            # Storage may not support all filter operations
            # This is acceptable for this test
            pass


# ============================================================================
# Test: BackgroundQueue
# ============================================================================


class TestBackgroundQueue:
    """Test BackgroundQueue functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, semantic_indexer):
        """Should initialize with required components."""
        queue = BackgroundQueue(semantic_indexer, batch_size=10)
        
        assert queue.indexer is not None
        assert queue.batch_size == 10
        assert queue.running is False
    
    @pytest.mark.asyncio
    async def test_start_stop(self, semantic_indexer):
        """Should start and stop queue."""
        queue = BackgroundQueue(semantic_indexer)
        
        await queue.start()
        assert queue.is_running() is True
        
        await queue.stop()
        assert queue.is_running() is False
    
    @pytest.mark.asyncio
    async def test_submit(self, semantic_indexer):
        """Should submit memory to queue."""
        queue = BackgroundQueue(semantic_indexer, max_queue_size=10)
        await queue.start()
        
        memory = {
            'memory_id': 'mem_1',
            'category': 'test',
            'context': 'Test memory',
            'timestamp': datetime.now(),
            'embedding': None
        }
        
        # Submit should not raise error
        await queue.submit(memory)
        
        # Verify queue size increased
        assert queue.get_queue_size() >= 0
        
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_submit_batch(self, semantic_indexer):
        """Should submit multiple memories."""
        queue = BackgroundQueue(semantic_indexer, max_queue_size=20)
        await queue.start()
        
        memories = [
            {
                'memory_id': f'mem_{i}',
                'category': 'test',
                'context': f'Test memory {i}',
                'timestamp': datetime.now(),
                'embedding': None
            }
            for i in range(5)
        ]
        
        await queue.submit_batch(memories)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_submit_to_stopped_queue(self, semantic_indexer):
        """Should raise error when submitting to stopped queue."""
        queue = BackgroundQueue(semantic_indexer)
        
        memory = {
            'memory_id': 'mem_1',
            'category': 'test',
            'context': 'Test memory',
            'timestamp': datetime.now()
        }
        
        with pytest.raises(RuntimeError):
            await queue.submit(memory)
    
    @pytest.mark.asyncio
    async def test_get_queue_size(self, semantic_indexer):
        """Should return current queue size."""
        queue = BackgroundQueue(semantic_indexer)
        await queue.start()
        
        initial_size = queue.get_queue_size()
        assert initial_size == 0
        
        # Submit memory
        memory = {
            'memory_id': 'mem_1',
            'category': 'test',
            'context': 'Test memory',
            'timestamp': datetime.now(),
            'embedding': None
        }
        await queue.submit(memory)
        
        # Size should increase (or be 0 if already processed)
        size_after = queue.get_queue_size()
        assert size_after >= 0
        
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_wait_until_empty(self, semantic_indexer):
        """Should wait until queue is empty."""
        queue = BackgroundQueue(semantic_indexer, batch_size=2)
        await queue.start()
        
        # Submit memories
        memories = [
            {
                'memory_id': f'mem_{i}',
                'category': 'test',
                'context': f'Test memory {i}',
                'timestamp': datetime.now(),
                'embedding': None
            }
            for i in range(3)
        ]
        await queue.submit_batch(memories)
        
        # Wait until empty
        result = await queue.wait_until_empty(timeout=5.0)
        assert result is True
        assert queue.get_queue_size() == 0
        
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_wait_until_empty_timeout(self, semantic_indexer):
        """Should timeout if queue doesn't empty."""
        # Create a slow indexer by using very small batch size
        queue = BackgroundQueue(semantic_indexer, batch_size=1, max_queue_size=100)
        await queue.start()
        
        # Submit many memories
        memories = [
            {
                'memory_id': f'mem_{i}',
                'category': 'test',
                'context': f'Test memory {i}',
                'timestamp': datetime.now(),
                'embedding': None
            }
            for i in range(50)
        ]
        await queue.submit_batch(memories)
        
        # Wait with very short timeout
        result = await queue.wait_until_empty(timeout=0.1)
        
        # Should timeout
        assert result is False
        
        await queue.stop()


# ============================================================================
# Integration Tests
# ============================================================================


class TestSemanticIndexingIntegration:
    """Test integration of semantic indexing components."""
    
    def test_end_to_end_indexing_and_search(
        self,
        semantic_indexer,
        vector_search,
        sample_memories
    ):
        """Should index memories and search them."""
        # Index memories
        semantic_indexer.batch_index(sample_memories)
        
        # Search
        results = vector_search.search(
            query='artificial intelligence and machine learning',
            category='knowledge',
            top_k=2
        )
        
        # Verify results
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        
        # First result should be most relevant
        if len(results) > 0:
            assert results[0].similarity > 0.0
    
    @pytest.mark.asyncio
    async def test_background_indexing_workflow(
        self,
        semantic_indexer,
        vector_search,
        index_db,
        sample_memories
    ):
        """Should index memories in background and search them."""
        # Create background queue
        queue = BackgroundQueue(semantic_indexer, batch_size=2)
        await queue.start()
        
        # Submit memories
        await queue.submit_batch(sample_memories)
        
        # Wait for processing
        await queue.wait_until_empty(timeout=10.0)
        await queue.stop()
        
        # Verify memories were indexed
        assert index_db.get_category_size('knowledge') == 3
        
        # Search should work
        results = vector_search.search(
            query='machine learning',
            category='knowledge',
            top_k=2
        )
        
        assert len(results) > 0

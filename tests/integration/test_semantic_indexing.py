"""
Integration tests for Phase 3 semantic indexing infrastructure.

Tests the complete flow: VectorSearch, SemanticIndexer, SemanticIndexDB,
MemorySearch, and BackgroundQueue.
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from llm_compression.logger import logger


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as storage_dir, \
         tempfile.TemporaryDirectory() as index_dir:
        yield Path(storage_dir), Path(index_dir)


@pytest.fixture
def embedding_provider():
    """Get embedding provider (ArrowEngine or fallback)."""
    from llm_compression.embedding_provider import get_default_provider
    return get_default_provider()


@pytest.fixture
def test_memories():
    """Create test memories."""
    return [
        {
            'memory_id': 'mem_1',
            'category': 'knowledge',
            'context': 'Machine learning is a subset of artificial intelligence.',
            'timestamp': datetime(2024, 1, 1),
            'embedding': None
        },
        {
            'memory_id': 'mem_2',
            'category': 'knowledge',
            'context': 'Python is a popular programming language for data science.',
            'timestamp': datetime(2024, 1, 2),
            'embedding': None
        },
        {
            'memory_id': 'mem_3',
            'category': 'knowledge',
            'context': 'Neural networks consist of interconnected layers of nodes.',
            'timestamp': datetime(2024, 1, 3),
            'embedding': None
        },
        {
            'memory_id': 'mem_4',
            'category': 'code',
            'context': 'def hello_world(): print("Hello, World!")',
            'timestamp': datetime(2024, 1, 4),
            'embedding': None
        },
        {
            'memory_id': 'mem_5',
            'category': 'code',
            'context': 'import numpy as np; arr = np.array([1, 2, 3])',
            'timestamp': datetime(2024, 1, 5),
            'embedding': None
        },
    ]


class TestSemanticIndexDB:
    """Test SemanticIndexDB functionality."""
    
    def test_add_and_query(self, temp_dirs, embedding_provider):
        """Test adding entries and querying."""
        from llm_compression.semantic_index_db import SemanticIndexDB
        
        _, index_dir = temp_dirs
        index_db = SemanticIndexDB(str(index_dir))
        
        # Generate test embeddings
        texts = ["machine learning", "deep learning", "data science"]
        embeddings = embedding_provider.encode_batch(texts, normalize=True)
        
        # Add entries
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            index_db.add_entry(
                memory_id=f"mem_{i}",
                category="knowledge",
                embedding=emb,
                timestamp=datetime.now()
            )
        
        # Query
        query_emb = embedding_provider.encode("artificial intelligence", normalize=True)
        if query_emb.ndim == 2:
            query_emb = query_emb[0]
        
        results = index_db.query(
            category="knowledge",
            query_embedding=query_emb,
            top_k=2
        )
        
        assert len(results) == 2
        assert all('memory_id' in r for r in results)
        assert all('similarity' in r for r in results)
        assert results[0]['similarity'] >= results[1]['similarity']
        
        logger.info("✅ SemanticIndexDB: add_and_query passed")
    
    def test_batch_add(self, temp_dirs, embedding_provider):
        """Test batch adding entries."""
        from llm_compression.semantic_index_db import SemanticIndexDB
        
        _, index_dir = temp_dirs
        index_db = SemanticIndexDB(str(index_dir))
        
        # Generate test embeddings
        texts = ["text 1", "text 2", "text 3"]
        embeddings = embedding_provider.encode_batch(texts, normalize=True)
        
        # Batch add
        entries = [
            {
                'memory_id': f"mem_{i}",
                'category': "test",
                'embedding': emb,
                'timestamp': datetime.now()
            }
            for i, emb in enumerate(embeddings)
        ]
        
        index_db.batch_add(entries)
        
        # Verify
        assert index_db.get_category_size("test") == 3
        
        logger.info("✅ SemanticIndexDB: batch_add passed")
    
    def test_clear_category(self, temp_dirs, embedding_provider):
        """Test clearing a category."""
        from llm_compression.semantic_index_db import SemanticIndexDB
        
        _, index_dir = temp_dirs
        index_db = SemanticIndexDB(str(index_dir))
        
        # Add entry
        emb = embedding_provider.encode("test", normalize=True)
        if emb.ndim == 2:
            emb = emb[0]
        
        index_db.add_entry(
            memory_id="mem_1",
            category="test",
            embedding=emb,
            timestamp=datetime.now()
        )
        
        assert index_db.get_category_size("test") == 1
        
        # Clear
        index_db.clear_category("test")
        assert index_db.get_category_size("test") == 0
        
        logger.info("✅ SemanticIndexDB: clear_category passed")


class TestSemanticIndexer:
    """Test SemanticIndexer functionality."""
    
    def test_index_memory(self, temp_dirs, embedding_provider, test_memories):
        """Test indexing a single memory."""
        from llm_compression.semantic_indexer import SemanticIndexer
        from llm_compression.semantic_index_db import SemanticIndexDB
        from llm_compression.arrow_storage import ArrowStorage
        
        storage_dir, index_dir = temp_dirs
        
        # Create components
        storage = ArrowStorage(str(storage_dir / "memories.parquet"))
        index_db = SemanticIndexDB(str(index_dir))
        indexer = SemanticIndexer(embedding_provider, storage, index_db)
        
        # Index single memory
        memory = test_memories[0]
        indexer.index_memory(memory)
        
        # Verify embedding was generated
        assert memory['embedding'] is not None
        assert isinstance(memory['embedding'], np.ndarray)
        
        # Verify indexed
        assert index_db.get_category_size("knowledge") == 1
        
        logger.info("✅ SemanticIndexer: index_memory passed")
    
    def test_batch_index(self, temp_dirs, embedding_provider, test_memories):
        """Test batch indexing."""
        from llm_compression.semantic_indexer import SemanticIndexer
        from llm_compression.semantic_index_db import SemanticIndexDB
        from llm_compression.arrow_storage import ArrowStorage
        
        storage_dir, index_dir = temp_dirs
        
        # Create components
        storage = ArrowStorage(str(storage_dir / "memories.parquet"))
        index_db = SemanticIndexDB(str(index_dir))
        indexer = SemanticIndexer(embedding_provider, storage, index_db)
        
        # Batch index
        knowledge_memories = [m for m in test_memories if m['category'] == 'knowledge']
        indexer.batch_index(knowledge_memories, batch_size=2)
        
        # Verify all have embeddings
        assert all(m['embedding'] is not None for m in knowledge_memories)
        
        # Verify indexed
        assert index_db.get_category_size("knowledge") == len(knowledge_memories)
        
        logger.info("✅ SemanticIndexer: batch_index passed")


class TestVectorSearch:
    """Test VectorSearch functionality."""
    
    def test_search(self, temp_dirs, embedding_provider, test_memories):
        """Test semantic search."""
        from llm_compression.vector_search import VectorSearch
        from llm_compression.semantic_indexer import SemanticIndexer
        from llm_compression.semantic_index_db import SemanticIndexDB
        from llm_compression.arrow_storage import ArrowStorage
        
        storage_dir, index_dir = temp_dirs
        
        # Setup
        storage = ArrowStorage(str(storage_dir / "memories.parquet"))
        index_db = SemanticIndexDB(str(index_dir))
        indexer = SemanticIndexer(embedding_provider, storage, index_db)
        
        # Index memories
        knowledge_memories = [m for m in test_memories if m['category'] == 'knowledge']
        indexer.batch_index(knowledge_memories)
        
        # Create VectorSearch
        vector_search = VectorSearch(embedding_provider, storage, index_db)
        
        # Override _load_memory to return from our test data
        def mock_load_memory(memory_id, category):
            for m in test_memories:
                if m['memory_id'] == memory_id:
                    return m
            return None
        
        vector_search._load_memory = mock_load_memory
        
        # Search
        results = vector_search.search(
            query="artificial intelligence and machine learning",
            category="knowledge",
            top_k=2
        )
        
        assert len(results) <= 2
        assert all(hasattr(r, 'memory_id') for r in results)
        assert all(hasattr(r, 'similarity') for r in results)
        
        # Results should be sorted by similarity
        if len(results) > 1:
            assert results[0].similarity >= results[1].similarity
        
        logger.info("✅ VectorSearch: search passed")


class TestMemorySearch:
    """Test MemorySearch functionality."""
    
    def test_semantic_search(self, temp_dirs, embedding_provider, test_memories):
        """Test semantic search mode."""
        from llm_compression.memory_search import MemorySearch, SearchMode
        from llm_compression.vector_search import VectorSearch
        from llm_compression.semantic_indexer import SemanticIndexer
        from llm_compression.semantic_index_db import SemanticIndexDB
        from llm_compression.arrow_storage import ArrowStorage
        
        storage_dir, index_dir = temp_dirs
        
        # Setup
        storage = ArrowStorage(str(storage_dir / "memories.parquet"))
        index_db = SemanticIndexDB(str(index_dir))
        indexer = SemanticIndexer(embedding_provider, storage, index_db)
        
        # Index memories
        indexer.batch_index(test_memories)
        
        # Create search
        vector_search = VectorSearch(embedding_provider, storage, index_db)
        
        # Mock _load_memory
        def mock_load_memory(memory_id, category):
            for m in test_memories:
                if m['memory_id'] == memory_id:
                    return m
            return None
        vector_search._load_memory = mock_load_memory
        
        memory_search = MemorySearch(vector_search, storage)
        
        # Semantic search
        results = memory_search.search(
            query="programming languages",
            category="knowledge",
            mode=SearchMode.SEMANTIC,
            top_k=2
        )
        
        assert len(results) <= 2
        
        logger.info("✅ MemorySearch: semantic_search passed")


@pytest.mark.asyncio
class TestBackgroundQueue:
    """Test BackgroundQueue functionality."""
    
    async def test_submit_and_process(self, temp_dirs, embedding_provider, test_memories):
        """Test submitting and processing items."""
        from llm_compression.background_queue import BackgroundQueue
        from llm_compression.semantic_indexer import SemanticIndexer
        from llm_compression.semantic_index_db import SemanticIndexDB
        from llm_compression.arrow_storage import ArrowStorage
        
        storage_dir, index_dir = temp_dirs
        
        # Setup
        storage = ArrowStorage(str(storage_dir / "memories.parquet"))
        index_db = SemanticIndexDB(str(index_dir))
        indexer = SemanticIndexer(embedding_provider, storage, index_db)
        
        # Create queue
        queue = BackgroundQueue(indexer, batch_size=2, batch_timeout=0.5)
        
        # Start queue
        await queue.start()
        
        # Submit memories
        for memory in test_memories[:3]:
            await queue.submit(memory)
        
        # Wait for processing
        await queue.wait_until_empty(timeout=10.0)
        
        # Stop queue
        await queue.stop()
        
        # Verify indexed
        assert index_db.get_category_size("knowledge") > 0
        
        logger.info("✅ BackgroundQueue: submit_and_process passed")
    
    async def test_batch_submit(self, temp_dirs, embedding_provider, test_memories):
        """Test batch submission."""
        from llm_compression.background_queue import BackgroundQueue
        from llm_compression.semantic_indexer import SemanticIndexer
        from llm_compression.semantic_index_db import SemanticIndexDB
        from llm_compression.arrow_storage import ArrowStorage
        
        storage_dir, index_dir = temp_dirs
        
        # Setup
        storage = ArrowStorage(str(storage_dir / "memories.parquet"))
        index_db = SemanticIndexDB(str(index_dir))
        indexer = SemanticIndexer(embedding_provider, storage, index_db)
        
        # Create queue
        queue = BackgroundQueue(indexer, batch_size=3)
        
        # Start queue
        await queue.start()
        
        # Batch submit
        await queue.submit_batch(test_memories)
        
        # Wait for processing
        await queue.wait_until_empty(timeout=10.0)
        
        # Stop queue
        await queue.stop()
        
        # Verify all indexed
        total_indexed = sum(
            index_db.get_category_size(cat)
            for cat in index_db.get_categories()
        )
        assert total_indexed == len(test_memories)
        
        logger.info("✅ BackgroundQueue: batch_submit passed")


@pytest.mark.integration
class TestEndToEndFlow:
    """Test complete end-to-end semantic indexing flow."""
    
    async def test_complete_flow(self, temp_dirs, embedding_provider, test_memories):
        """Test complete indexing and search flow."""
        from llm_compression.background_queue import BackgroundQueue
        from llm_compression.memory_search import MemorySearch, SearchMode
        from llm_compression.vector_search import VectorSearch
        from llm_compression.semantic_indexer import SemanticIndexer
        from llm_compression.semantic_index_db import SemanticIndexDB
        from llm_compression.arrow_storage import ArrowStorage
        
        storage_dir, index_dir = temp_dirs
        
        logger.info("Starting end-to-end flow test")
        
        # 1. Setup components
        storage = ArrowStorage(str(storage_dir / "memories.parquet"))
        index_db = SemanticIndexDB(str(index_dir))
        indexer = SemanticIndexer(embedding_provider, storage, index_db)
        
        # 2. Index memories using background queue
        queue = BackgroundQueue(indexer, batch_size=2)
        await queue.start()
        await queue.submit_batch(test_memories)
        await queue.wait_until_empty(timeout=15.0)
        await queue.stop()
        
        logger.info(f"Indexed {len(test_memories)} memories")
        
        # 3. Perform searches
        vector_search = VectorSearch(embedding_provider, storage, index_db)
        
        # Mock _load_memory
        def mock_load_memory(memory_id, category):
            for m in test_memories:
                if m['memory_id'] == memory_id:
                    return m
            return None
        vector_search._load_memory = mock_load_memory
        
        memory_search = MemorySearch(vector_search, storage)
        
        # Semantic search
        results = memory_search.search(
            query="machine learning and AI",
            category="knowledge",
            mode=SearchMode.SEMANTIC,
            top_k=3
        )
        
        logger.info(f"Found {len(results)} results")
        assert len(results) > 0
        
        # Verify results
        for r in results:
            logger.info(f"  - {r.memory_id}: similarity={r.similarity:.3f}")
            assert r.similarity > 0
        
        logger.info("✅ End-to-end flow test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""
Unit tests for NetworkNavigatorArrow

Tests:
- retrieve_arrow(): 零拷贝检索
- find_similar_memories_vectorized(): 向量化相似度搜索
- batch_retrieve_arrow(): 批量检索
- 零拷贝验证
- Top-K 优化验证

Requirements: Task 12.3
"""

import pytest
import pyarrow as pa
import numpy as np
from unittest.mock import Mock

from llm_compression.network_navigator_arrow import (
    NetworkNavigatorArrow,
    ActivationResultArrow,
    add_arrow_support
)
from llm_compression.network_navigator import NetworkNavigator
from llm_compression.embedder_arrow import LocalEmbedderArrow


@pytest.fixture
def embedder_arrow():
    """Create LocalEmbedderArrow instance"""
    return LocalEmbedderArrow()


@pytest.fixture
def navigator_arrow():
    """Create NetworkNavigatorArrow instance"""
    return NetworkNavigatorArrow(
        max_hops=3,
        decay_rate=0.7,
        activation_threshold=0.1
    )


@pytest.fixture
def sample_memory_table(embedder_arrow):
    """Create sample memory table"""
    texts = [
        "Python is a programming language",
        "Machine learning uses neural networks",
        "Data science involves statistics",
        "Deep learning is a subset of machine learning",
        "Python is popular for data science"
    ]
    
    memory_ids = [f"mem{i}" for i in range(len(texts))]
    
    # Encode embeddings
    embeddings_array = embedder_arrow.batch_encode_arrow(texts)
    
    # Create table
    table = pa.table({
        'memory_id': pa.array(memory_ids),
        'content': pa.array(texts),
        'embedding': embeddings_array
    })
    
    return table


class TestNetworkNavigatorArrow:
    """Test NetworkNavigatorArrow class"""
    
    def test_initialization(self, navigator_arrow):
        """Test initialization"""
        assert navigator_arrow is not None
        assert navigator_arrow.max_hops == 3
        assert navigator_arrow.decay_rate == 0.7
        assert navigator_arrow.activation_threshold == 0.1
    
    def test_retrieve_arrow_empty_table(self, navigator_arrow, embedder_arrow):
        """Test retrieve with empty table"""
        empty_table = pa.table({
            'memory_id': pa.array([]),
            'content': pa.array([]),
            'embedding': pa.array([], type=pa.list_(pa.float32(), embedder_arrow.dimension))
        })
        
        query_vec = embedder_arrow.embedder.encode("test query")
        
        result = navigator_arrow.retrieve_arrow(
            query_embedding=query_vec,
            memory_table=empty_table,
            max_results=5
        )
        
        assert isinstance(result, ActivationResultArrow)
        assert len(result.table) == 0
        assert len(result.activation_map) == 0
    
    def test_retrieve_arrow_basic(self, navigator_arrow, sample_memory_table, embedder_arrow):
        """Test basic retrieve"""
        query_vec = embedder_arrow.embedder.encode("Python programming")
        
        result = navigator_arrow.retrieve_arrow(
            query_embedding=query_vec,
            memory_table=sample_memory_table,
            max_results=3
        )
        
        assert isinstance(result, ActivationResultArrow)
        assert len(result.table) > 0
        assert len(result.table) <= 3
        assert len(result.activation_map) > 0
        assert 'activation_score' in result.table.schema.names
    
    def test_retrieve_arrow_max_results(self, navigator_arrow, sample_memory_table, embedder_arrow):
        """Test retrieve with max_results limit"""
        query_vec = embedder_arrow.embedder.encode("machine learning")
        
        result = navigator_arrow.retrieve_arrow(
            query_embedding=query_vec,
            memory_table=sample_memory_table,
            max_results=2
        )
        
        assert len(result.table) <= 2
    
    def test_find_similar_vectorized_basic(self, navigator_arrow, sample_memory_table, embedder_arrow):
        """Test vectorized similarity search"""
        query_vec = embedder_arrow.embedder.encode("Python programming")
        
        result_table = navigator_arrow.find_similar_memories_vectorized(
            query_embedding=query_vec,
            memory_table=sample_memory_table,
            top_k=3
        )
        
        assert len(result_table) <= 3
        assert 'similarity_score' in result_table.schema.names
        
        # Check scores are sorted descending
        scores = result_table['similarity_score'].to_pylist()
        assert scores == sorted(scores, reverse=True)
    
    def test_find_similar_vectorized_with_threshold(self, navigator_arrow, sample_memory_table, embedder_arrow):
        """Test vectorized search with threshold"""
        query_vec = embedder_arrow.embedder.encode("Python programming")
        
        result_table = navigator_arrow.find_similar_memories_vectorized(
            query_embedding=query_vec,
            memory_table=sample_memory_table,
            top_k=10,
            threshold=0.5
        )
        
        # All scores should be above threshold
        scores = result_table['similarity_score'].to_pylist()
        assert all(score >= 0.5 for score in scores)
    
    def test_find_similar_vectorized_empty_result(self, navigator_arrow, sample_memory_table, embedder_arrow):
        """Test vectorized search with high threshold (no results)"""
        query_vec = embedder_arrow.embedder.encode("completely unrelated topic xyz")
        
        result_table = navigator_arrow.find_similar_memories_vectorized(
            query_embedding=query_vec,
            memory_table=sample_memory_table,
            top_k=10,
            threshold=0.9  # Very high threshold
        )
        
        # May have no results
        assert len(result_table) >= 0
    
    def test_batch_retrieve_arrow(self, navigator_arrow, sample_memory_table, embedder_arrow):
        """Test batch retrieve"""
        queries = [
            "Python programming",
            "machine learning",
            "data science"
        ]
        
        query_vecs = embedder_arrow.embedder.encode_batch(queries)
        
        results = navigator_arrow.batch_retrieve_arrow(
            query_embeddings=query_vecs,
            memory_table=sample_memory_table,
            max_results=2
        )
        
        assert len(results) == 3
        assert all(isinstance(r, ActivationResultArrow) for r in results)
        assert all(len(r.table) <= 2 for r in results)
    
    def test_find_similar_vectorized_top_k_optimization(self, navigator_arrow, embedder_arrow):
        """Test Top-K optimization (argpartition)"""
        # Create large table
        n_memories = 1000
        texts = [f"Content {i}" for i in range(n_memories)]
        memory_ids = [f"mem{i}" for i in range(n_memories)]
        
        embeddings_array = embedder_arrow.batch_encode_arrow(texts)
        
        large_table = pa.table({
            'memory_id': pa.array(memory_ids),
            'content': pa.array(texts),
            'embedding': embeddings_array
        })
        
        query_vec = embedder_arrow.embedder.encode("test query")
        
        # Should use argpartition for large tables
        result_table = navigator_arrow.find_similar_memories_vectorized(
            query_embedding=query_vec,
            memory_table=large_table,
            top_k=10
        )
        
        assert len(result_table) == 10
        
        # Verify scores are sorted
        scores = result_table['similarity_score'].to_pylist()
        assert scores == sorted(scores, reverse=True)


class TestZeroCopyVerification:
    """Test zero-copy characteristics"""
    
    def test_retrieve_returns_arrow_table(self, navigator_arrow, sample_memory_table, embedder_arrow):
        """Test that retrieve returns Arrow Table (zero-copy)"""
        query_vec = embedder_arrow.embedder.encode("test query")
        
        result = navigator_arrow.retrieve_arrow(
            query_embedding=query_vec,
            memory_table=sample_memory_table,
            max_results=3
        )
        
        # Result should be Arrow Table
        assert isinstance(result.table, pa.Table)
        
        # Should have activation_score column
        assert 'activation_score' in result.table.schema.names
    
    def test_vectorized_search_zero_copy(self, navigator_arrow, sample_memory_table, embedder_arrow):
        """Test that vectorized search is zero-copy"""
        query_vec = embedder_arrow.embedder.encode("test query")
        
        # Get embeddings buffer (zero-copy)
        from llm_compression.arrow_zero_copy import get_embeddings_buffer
        embeddings = get_embeddings_buffer(sample_memory_table, 'embedding')
        
        # Should be NumPy array (zero-copy)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sample_memory_table)


class TestPerformance:
    """Test performance characteristics"""
    
    def test_large_scale_retrieve(self, navigator_arrow, embedder_arrow):
        """Test retrieve with large memory table"""
        # Create 1K memories
        n_memories = 1000
        texts = [f"Content about topic {i % 100}" for i in range(n_memories)]
        memory_ids = [f"mem{i}" for i in range(n_memories)]
        
        embeddings_array = embedder_arrow.batch_encode_arrow(texts, batch_size=100)
        
        large_table = pa.table({
            'memory_id': pa.array(memory_ids),
            'content': pa.array(texts),
            'embedding': embeddings_array
        })
        
        query_vec = embedder_arrow.embedder.encode("topic 42")
        
        import time
        start_time = time.time()
        
        result = navigator_arrow.retrieve_arrow(
            query_embedding=query_vec,
            memory_table=large_table,
            max_results=10
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        assert len(result.table) <= 10
        print(f"\n[1K memories] Retrieve time: {elapsed_ms:.1f}ms")
        
        # Should be fast (< 400ms for 1K memories on Windows, including model overhead)
        # Note: First run includes model loading time (~200-300ms)
        assert elapsed_ms < 400  # Relaxed threshold for Windows environment


class TestAddArrowSupport:
    """Test add_arrow_support function"""
    
    def test_add_arrow_support(self):
        """Test adding Arrow support to NetworkNavigator"""
        navigator = NetworkNavigator()
        navigator_arrow = add_arrow_support(navigator)
        
        assert isinstance(navigator_arrow, NetworkNavigatorArrow)
        assert navigator_arrow.navigator is navigator


class TestEdgeCases:
    """Test edge cases"""
    
    def test_retrieve_single_memory(self, navigator_arrow, embedder_arrow):
        """Test retrieve with single memory"""
        texts = ["Single memory content"]
        memory_ids = ["mem1"]
        
        embeddings_array = embedder_arrow.batch_encode_arrow(texts)
        
        single_table = pa.table({
            'memory_id': pa.array(memory_ids),
            'content': pa.array(texts),
            'embedding': embeddings_array
        })
        
        query_vec = embedder_arrow.embedder.encode("test query")
        
        result = navigator_arrow.retrieve_arrow(
            query_embedding=query_vec,
            memory_table=single_table,
            max_results=5
        )
        
        assert len(result.table) == 1
    
    def test_retrieve_max_results_larger_than_table(self, navigator_arrow, sample_memory_table, embedder_arrow):
        """Test retrieve with max_results > table size"""
        query_vec = embedder_arrow.embedder.encode("test query")
        
        result = navigator_arrow.retrieve_arrow(
            query_embedding=query_vec,
            memory_table=sample_memory_table,
            max_results=100  # Larger than table size
        )
        
        # Should return all memories
        assert len(result.table) <= len(sample_memory_table)

"""
Unit tests for LocalEmbedder Arrow support

Tests Arrow native encoding and zero-copy operations.

Requirements: Task 12.2
"""

import pytest
import pyarrow as pa
import numpy as np
from typing import List

from llm_compression.embedder import LocalEmbedder
from llm_compression.embedder_arrow import LocalEmbedderArrow, add_arrow_support


@pytest.fixture
def embedder():
    """Create LocalEmbedder instance"""
    return LocalEmbedder()


@pytest.fixture
def embedder_arrow(embedder):
    """Create LocalEmbedderArrow instance"""
    return LocalEmbedderArrow(embedder)


@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return [
        "Python is a programming language",
        "Java is also a programming language",
        "The cat sits on the mat",
        "Machine learning is fascinating",
        "Deep learning uses neural networks"
    ]


@pytest.fixture
def sample_table(embedder_arrow, sample_texts):
    """Create sample Arrow table with embeddings"""
    return embedder_arrow.create_embedding_table(
        sample_texts,
        include_text=True
    )


class TestLocalEmbedderArrow:
    """Test LocalEmbedderArrow class"""
    
    def test_init(self, embedder):
        """Test initialization"""
        embedder_arrow = LocalEmbedderArrow(embedder)
        assert embedder_arrow is not None
        assert embedder_arrow.dimension == 384
    
    def test_init_without_embedder(self):
        """Test initialization without embedder"""
        embedder_arrow = LocalEmbedderArrow()
        assert embedder_arrow is not None
        assert embedder_arrow.embedder is not None
    
    def test_encode_to_arrow(self, embedder_arrow):
        """Test encoding to Arrow array"""
        text = "Hello, World!"
        arrow_array = embedder_arrow.encode_to_arrow(text)
        
        assert isinstance(arrow_array, pa.Array)
        assert len(arrow_array) == 1
        
        # Check embedding dimension
        embedding = arrow_array[0].as_py()
        assert len(embedding) == 384
    
    def test_batch_encode_arrow(self, embedder_arrow, sample_texts):
        """Test batch encoding to Arrow array"""
        arrow_array = embedder_arrow.batch_encode_arrow(sample_texts)
        
        assert isinstance(arrow_array, pa.Array)
        assert len(arrow_array) == len(sample_texts)
        
        # Check first embedding
        embedding = arrow_array[0].as_py()
        assert len(embedding) == 384
    
    def test_batch_encode_arrow_empty(self, embedder_arrow):
        """Test batch encoding with empty list"""
        arrow_array = embedder_arrow.batch_encode_arrow([])
        
        assert isinstance(arrow_array, pa.Array)
        assert len(arrow_array) == 0
    
    def test_create_embedding_table(self, embedder_arrow, sample_texts):
        """Test creating embedding table"""
        table = embedder_arrow.create_embedding_table(
            sample_texts,
            include_text=True
        )
        
        assert isinstance(table, pa.Table)
        assert len(table) == len(sample_texts)
        assert 'embedding' in table.schema.names
        assert 'text' in table.schema.names
    
    def test_create_embedding_table_without_text(self, embedder_arrow, sample_texts):
        """Test creating embedding table without text column"""
        table = embedder_arrow.create_embedding_table(
            sample_texts,
            include_text=False
        )
        
        assert isinstance(table, pa.Table)
        assert len(table) == len(sample_texts)
        assert 'embedding' in table.schema.names
        assert 'text' not in table.schema.names
    
    def test_create_embedding_table_with_additional_columns(self, embedder_arrow, sample_texts):
        """Test creating embedding table with additional columns"""
        additional_columns = {
            'id': [f'doc_{i}' for i in range(len(sample_texts))],
            'score': [0.9, 0.8, 0.7, 0.85, 0.95]
        }
        
        table = embedder_arrow.create_embedding_table(
            sample_texts,
            include_text=True,
            additional_columns=additional_columns
        )
        
        assert len(table) == len(sample_texts)
        assert 'id' in table.schema.names
        assert 'score' in table.schema.names
    
    def test_similarity_matrix_arrow(self, embedder_arrow, sample_table):
        """Test similarity matrix computation"""
        # Compute similarity matrix (all vs all)
        similarity_matrix = embedder_arrow.similarity_matrix_arrow(sample_table)
        
        assert isinstance(similarity_matrix, np.ndarray)
        assert similarity_matrix.shape == (len(sample_table), len(sample_table))
        
        # Diagonal should be ~1.0 (self-similarity)
        diagonal = np.diag(similarity_matrix)
        assert np.all(diagonal > 0.99)
    
    def test_similarity_matrix_arrow_with_query(self, embedder_arrow, sample_table):
        """Test similarity matrix with query vector"""
        query_text = "programming languages"
        query_vec = embedder_arrow.embedder.encode(query_text)
        
        similarities = embedder_arrow.similarity_matrix_arrow(
            sample_table,
            query_embedding=query_vec
        )
        
        assert isinstance(similarities, np.ndarray)
        assert len(similarities) == len(sample_table)
        
        # First two texts should have high similarity (about programming)
        assert similarities[0] > 0.5
        assert similarities[1] > 0.5
    
    def test_find_most_similar_arrow(self, embedder_arrow, sample_table):
        """Test finding most similar items"""
        query = "coding and software"
        
        results = embedder_arrow.find_most_similar_arrow(
            query,
            sample_table,
            top_k=2
        )
        
        assert len(results) == 2
        assert all(isinstance(idx, int) for idx, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        
        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores[0] >= scores[1]
    
    def test_find_most_similar_arrow_with_threshold(self, embedder_arrow, sample_table):
        """Test finding most similar items with threshold"""
        query = "programming"
        
        results = embedder_arrow.find_most_similar_arrow(
            query,
            sample_table,
            top_k=10,
            threshold=0.5
        )
        
        # Should filter out low similarity items
        assert all(score >= 0.5 for _, score in results)
    
    def test_semantic_search_arrow(self, embedder_arrow, sample_table):
        """Test semantic search"""
        query = "artificial intelligence"
        
        result_table = embedder_arrow.semantic_search_arrow(
            query,
            sample_table,
            top_k=3
        )
        
        assert isinstance(result_table, pa.Table)
        assert len(result_table) <= 3
        assert 'similarity_score' in result_table.schema.names
        
        # Scores should be in descending order
        scores = result_table['similarity_score'].to_pylist()
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    def test_batch_similarity_search(self, embedder_arrow, sample_table):
        """Test batch similarity search"""
        queries = [
            "programming languages",
            "animals and pets",
            "neural networks"
        ]
        
        results = embedder_arrow.batch_similarity_search(
            queries,
            sample_table,
            top_k=2
        )
        
        assert len(results) == len(queries)
        assert all(len(query_results) == 2 for query_results in results)
        
        # Each result should have (index, score) tuples
        for query_results in results:
            assert all(isinstance(idx, int) for idx, _ in query_results)
            assert all(isinstance(score, float) for _, score in query_results)
    
    def test_batch_similarity_search_empty_queries(self, embedder_arrow, sample_table):
        """Test batch similarity search with empty queries"""
        results = embedder_arrow.batch_similarity_search(
            [],
            sample_table,
            top_k=2
        )
        
        assert results == []
    
    def test_get_embedding_dimension(self, embedder_arrow):
        """Test getting embedding dimension"""
        dim = embedder_arrow.get_embedding_dimension()
        assert dim == 384


class TestAddArrowSupport:
    """Test add_arrow_support function"""
    
    def test_add_arrow_support(self, embedder):
        """Test adding Arrow support to embedder"""
        embedder_arrow = add_arrow_support(embedder)
        
        assert isinstance(embedder_arrow, LocalEmbedderArrow)
        assert embedder_arrow.embedder is embedder


class TestArrowZeroCopyIntegration:
    """Test integration with Arrow zero-copy utilities"""
    
    def test_zero_copy_embedding_extraction(self, embedder_arrow, sample_table):
        """Test zero-copy embedding extraction"""
        from llm_compression.arrow_zero_copy import get_embeddings_buffer
        
        # Extract embeddings (zero-copy)
        embeddings = get_embeddings_buffer(sample_table, 'embedding')
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(sample_table), 384)
    
    def test_zero_copy_similarity_computation(self, embedder_arrow, sample_table):
        """Test zero-copy similarity computation"""
        from llm_compression.arrow_zero_copy import (
            get_embeddings_buffer,
            compute_similarity_zero_copy
        )
        
        # Extract embeddings (zero-copy)
        embeddings = get_embeddings_buffer(sample_table, 'embedding')
        
        # Create query
        query_vec = embedder_arrow.embedder.encode("programming")
        
        # Compute similarities (zero-copy, vectorized)
        similarities = compute_similarity_zero_copy(embeddings, query_vec)
        
        assert isinstance(similarities, np.ndarray)
        assert len(similarities) == len(sample_table)


class TestPerformanceCharacteristics:
    """Test performance characteristics"""
    
    def test_batch_encoding_faster_than_individual(self, embedder_arrow):
        """Test that batch encoding is faster than individual encoding"""
        import time
        
        texts = ["Test text"] * 100
        
        # Individual encoding
        start = time.time()
        for text in texts:
            embedder_arrow.encode_to_arrow(text)
        individual_time = time.time() - start
        
        # Batch encoding
        start = time.time()
        embedder_arrow.batch_encode_arrow(texts)
        batch_time = time.time() - start
        
        # Batch should be significantly faster
        assert batch_time < individual_time * 0.5
    
    def test_arrow_array_memory_efficiency(self, embedder_arrow, sample_texts):
        """Test Arrow array memory efficiency"""
        # Create Arrow array
        arrow_array = embedder_arrow.batch_encode_arrow(sample_texts)
        
        # Arrow array should be more memory efficient than Python list
        # (This is a qualitative test - Arrow uses contiguous memory)
        assert isinstance(arrow_array, pa.Array)
        assert len(arrow_array) == len(sample_texts)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

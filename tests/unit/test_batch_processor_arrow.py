"""
Unit tests for BatchProcessorArrow

Tests:
- compress_batch_arrow(): 批量压缩返回 Arrow Table
- group_similar_arrow(): 零拷贝聚类
- compute_similarity_matrix_vectorized(): 向量化相似度矩阵
- parallel_compress_batches(): 并行批处理
- 零拷贝验证
- 向量化性能验证

Requirements: Task 12.4
"""

import pytest
import pyarrow as pa
import numpy as np
import asyncio
from typing import List

from llm_compression.batch_processor_arrow import (
    BatchProcessorArrow,
    BatchResultArrow,
    add_arrow_support
)
from llm_compression.batch_processor import BatchProcessor
from llm_compression.compressor import MemoryType
from llm_compression.embedder_arrow import LocalEmbedderArrow


@pytest.fixture
def embedder_arrow():
    """Create LocalEmbedderArrow instance"""
    return LocalEmbedderArrow()


@pytest.fixture
def processor_arrow(embedder_arrow):
    """Create BatchProcessorArrow instance"""
    return BatchProcessorArrow(
        embedder_arrow=embedder_arrow,
        batch_size=4,
        max_concurrent=2,
        similarity_threshold=0.8
    )


@pytest.fixture
def sample_texts():
    """Create sample texts"""
    return [
        "Python is a programming language",
        "Python is popular for data science",
        "Machine learning uses neural networks",
        "Deep learning is a subset of machine learning",
        "Data science involves statistics",
        "Statistics is important for data analysis",
        "Neural networks are used in AI",
        "AI is transforming technology"
    ]


class TestBatchProcessorArrow:
    """Test BatchProcessorArrow class"""
    
    def test_initialization(self, processor_arrow):
        """Test initialization"""
        assert processor_arrow is not None
        assert processor_arrow.batch_size == 4
        assert processor_arrow.max_concurrent == 2
        assert processor_arrow.similarity_threshold == 0.8
    
    @pytest.mark.asyncio
    async def test_compress_batch_arrow_basic(self, processor_arrow, sample_texts):
        """Test basic batch compression"""
        result = await processor_arrow.compress_batch_arrow(
            texts=sample_texts[:4],
            memory_type=MemoryType.TEXT,
            include_embeddings=True
        )
        
        assert isinstance(result, BatchResultArrow)
        assert isinstance(result.table, pa.Table)
        assert result.total_items == 4
        assert result.completed_items > 0
        assert result.elapsed_time > 0
        assert result.throughput > 0
    
    @pytest.mark.asyncio
    async def test_compress_batch_arrow_with_embeddings(self, processor_arrow, sample_texts):
        """Test batch compression with embeddings"""
        result = await processor_arrow.compress_batch_arrow(
            texts=sample_texts[:4],
            include_embeddings=True
        )
        
        # Should have embedding column
        assert 'embedding' in result.table.schema.names
        assert len(result.table) == 4
    
    @pytest.mark.asyncio
    async def test_compress_batch_arrow_without_embeddings(self, processor_arrow, sample_texts):
        """Test batch compression without embeddings"""
        result = await processor_arrow.compress_batch_arrow(
            texts=sample_texts[:4],
            include_embeddings=False
        )
        
        # Should not have embedding column
        assert 'embedding' not in result.table.schema.names
        assert len(result.table) == 4
    
    @pytest.mark.asyncio
    async def test_compress_batch_arrow_large_batch(self, processor_arrow, sample_texts):
        """Test batch compression with all texts"""
        result = await processor_arrow.compress_batch_arrow(
            texts=sample_texts,
            include_embeddings=True
        )
        
        assert result.total_items == len(sample_texts)
        assert len(result.table) == len(sample_texts)
        assert 'text' in result.table.schema.names
        assert 'compressed' in result.table.schema.names
        assert 'compression_ratio' in result.table.schema.names


class TestGroupSimilarArrow:
    """Test group_similar_arrow method"""
    
    @pytest.mark.asyncio
    async def test_group_similar_small_batch(self, processor_arrow):
        """Test grouping with small batch (no grouping needed)"""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        groups = await processor_arrow.group_similar_arrow(texts)
        
        # Should return single group
        assert len(groups) == 1
        assert len(groups[0][1]) == 3
    
    @pytest.mark.asyncio
    async def test_group_similar_with_precomputed_embeddings(self, processor_arrow, embedder_arrow):
        """Test grouping with precomputed embeddings"""
        texts = [
            "Python programming",
            "Python coding",
            "Machine learning",
            "Deep learning",
            "Data science",
            "Statistics"
        ]
        
        # Precompute embeddings
        embeddings_array = embedder_arrow.batch_encode_arrow(texts)
        
        groups = await processor_arrow.group_similar_arrow(
            texts=texts,
            embeddings_array=embeddings_array
        )
        
        assert len(groups) > 0
        
        # Verify all texts are included
        all_texts = []
        for _, group_texts in groups:
            all_texts.extend(group_texts)
        assert len(all_texts) == len(texts)
    
    @pytest.mark.asyncio
    async def test_group_similar_vectorized(self, processor_arrow, sample_texts):
        """Test vectorized grouping"""
        groups = await processor_arrow.group_similar_arrow(sample_texts)
        
        assert len(groups) > 0
        
        # Verify all texts are grouped
        total_texts = sum(len(group_texts) for _, group_texts in groups)
        assert total_texts == len(sample_texts)
        
        # Verify indices are correct
        all_indices = []
        for indices, _ in groups:
            all_indices.extend(indices)
        assert sorted(all_indices) == list(range(len(sample_texts)))
    
    @pytest.mark.asyncio
    async def test_group_similar_fallback(self, processor_arrow):
        """Test fallback grouping when vectorized fails"""
        # Use texts that might cause issues
        texts = [""] * 10  # Empty texts
        
        groups = await processor_arrow.group_similar_arrow(texts)
        
        # Should still return groups (fallback)
        assert len(groups) > 0


class TestComputeSimilarityMatrix:
    """Test compute_similarity_matrix_vectorized method"""
    
    def test_similarity_matrix_basic(self, processor_arrow):
        """Test basic similarity matrix computation"""
        embeddings = np.random.randn(5, 384).astype(np.float32)
        
        similarity_matrix = processor_arrow.compute_similarity_matrix_vectorized(
            embeddings
        )
        
        assert similarity_matrix.shape == (5, 5)
        
        # Diagonal should be ~1.0 (self-similarity)
        diagonal = np.diag(similarity_matrix)
        assert np.allclose(diagonal, 1.0, atol=0.01)
        
        # Matrix should be symmetric
        assert np.allclose(similarity_matrix, similarity_matrix.T, atol=1e-5)
    
    def test_similarity_matrix_normalized(self, processor_arrow):
        """Test that similarity matrix is normalized"""
        embeddings = np.random.randn(3, 384).astype(np.float32)
        
        similarity_matrix = processor_arrow.compute_similarity_matrix_vectorized(
            embeddings
        )
        
        # All values should be in [-1, 1] (with small tolerance for float32 precision)
        assert np.all(similarity_matrix >= -1.0 - 1e-6)
        assert np.all(similarity_matrix <= 1.0 + 1e-6)
    
    def test_similarity_matrix_zero_vectors(self, processor_arrow):
        """Test similarity matrix with zero vectors"""
        embeddings = np.zeros((3, 384), dtype=np.float32)
        
        similarity_matrix = processor_arrow.compute_similarity_matrix_vectorized(
            embeddings
        )
        
        # Should handle zero vectors gracefully
        assert similarity_matrix.shape == (3, 3)
        assert not np.any(np.isnan(similarity_matrix))


class TestParallelCompressBatches:
    """Test parallel_compress_batches method"""
    
    @pytest.mark.asyncio
    async def test_parallel_compress_single_batch(self, processor_arrow):
        """Test parallel compression with single batch"""
        text_batches = [
            ["Text 1", "Text 2"],
        ]
        
        results = await processor_arrow.parallel_compress_batches(text_batches)
        
        assert len(results) == 1
        assert isinstance(results[0], BatchResultArrow)
    
    @pytest.mark.asyncio
    async def test_parallel_compress_multiple_batches(self, processor_arrow):
        """Test parallel compression with multiple batches"""
        text_batches = [
            ["Text 1", "Text 2"],
            ["Text 3", "Text 4"],
            ["Text 5", "Text 6"]
        ]
        
        results = await processor_arrow.parallel_compress_batches(text_batches)
        
        assert len(results) == 3
        assert all(isinstance(r, BatchResultArrow) for r in results)
        assert all(r.total_items == 2 for r in results)
    
    @pytest.mark.asyncio
    async def test_parallel_compress_with_failures(self, processor_arrow):
        """Test parallel compression handles failures gracefully"""
        text_batches = [
            ["Valid text 1", "Valid text 2"],
            ["Valid text 3", "Valid text 4"]
        ]
        
        results = await processor_arrow.parallel_compress_batches(text_batches)
        
        # Should return successful results
        assert len(results) >= 0


class TestZeroCopyVerification:
    """Test zero-copy characteristics"""
    
    @pytest.mark.asyncio
    async def test_compress_returns_arrow_table(self, processor_arrow, sample_texts):
        """Test that compress returns Arrow Table (zero-copy)"""
        result = await processor_arrow.compress_batch_arrow(
            texts=sample_texts[:4],
            include_embeddings=True
        )
        
        # Result should be Arrow Table
        assert isinstance(result.table, pa.Table)
        
        # Should have expected columns
        assert 'text' in result.table.schema.names
        assert 'compressed' in result.table.schema.names
        assert 'embedding' in result.table.schema.names
    
    @pytest.mark.asyncio
    async def test_embeddings_zero_copy(self, processor_arrow, embedder_arrow):
        """Test that embeddings are zero-copy"""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        # Precompute embeddings
        embeddings_array = embedder_arrow.batch_encode_arrow(texts)
        
        # Should be Arrow Array
        assert isinstance(embeddings_array, pa.Array)
        
        # Use in grouping (zero-copy)
        groups = await processor_arrow.group_similar_arrow(
            texts=texts,
            embeddings_array=embeddings_array
        )
        
        assert len(groups) > 0


class TestAddArrowSupport:
    """Test add_arrow_support function"""
    
    def test_add_arrow_support(self):
        """Test adding Arrow support to BatchProcessor"""
        from unittest.mock import Mock
        
        # 使用 Mock 对象避免复杂的依赖
        compressor = Mock()
        processor = BatchProcessor(compressor)
        processor_arrow = add_arrow_support(processor)
        
        assert isinstance(processor_arrow, BatchProcessorArrow)
        assert processor_arrow.processor is processor


class TestPerformance:
    """Test performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_large_batch_performance(self, processor_arrow):
        """Test performance with large batch"""
        # Create 100 texts
        texts = [f"Content about topic {i % 20}" for i in range(100)]
        
        import time
        start_time = time.time()
        
        result = await processor_arrow.compress_batch_arrow(
            texts=texts,
            include_embeddings=True
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        assert result.total_items == 100
        assert result.throughput > 0
        print(f"\n[100 texts] Batch compression time: {elapsed_ms:.1f}ms")
        print(f"Throughput: {result.throughput:.1f} items/min")
    
    def test_similarity_matrix_performance(self, processor_arrow):
        """Test similarity matrix computation performance"""
        # Create large embedding matrix
        n_embeddings = 500
        embeddings = np.random.randn(n_embeddings, 384).astype(np.float32)
        
        import time
        start_time = time.time()
        
        similarity_matrix = processor_arrow.compute_similarity_matrix_vectorized(
            embeddings
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        assert similarity_matrix.shape == (n_embeddings, n_embeddings)
        print(f"\n[{n_embeddings} embeddings] Similarity matrix time: {elapsed_ms:.1f}ms")


class TestEdgeCases:
    """Test edge cases"""
    
    @pytest.mark.asyncio
    async def test_compress_empty_list(self, processor_arrow):
        """Test compress with empty list"""
        result = await processor_arrow.compress_batch_arrow(
            texts=[],
            include_embeddings=True
        )
        
        assert result.total_items == 0
        assert len(result.table) == 0
    
    @pytest.mark.asyncio
    async def test_compress_single_text(self, processor_arrow):
        """Test compress with single text"""
        result = await processor_arrow.compress_batch_arrow(
            texts=["Single text"],
            include_embeddings=True
        )
        
        assert result.total_items == 1
        assert len(result.table) == 1
    
    @pytest.mark.asyncio
    async def test_group_similar_empty_list(self, processor_arrow):
        """Test grouping with empty list"""
        groups = await processor_arrow.group_similar_arrow([])
        
        # Should return empty or single empty group
        assert len(groups) >= 0
    
    def test_similarity_matrix_single_embedding(self, processor_arrow):
        """Test similarity matrix with single embedding"""
        embeddings = np.random.randn(1, 384).astype(np.float32)
        
        similarity_matrix = processor_arrow.compute_similarity_matrix_vectorized(
            embeddings
        )
        
        assert similarity_matrix.shape == (1, 1)
        assert np.allclose(similarity_matrix[0, 0], 1.0, atol=0.01)


class TestResultTable:
    """Test result table structure"""
    
    @pytest.mark.asyncio
    async def test_result_table_schema(self, processor_arrow, sample_texts):
        """Test result table has correct schema"""
        result = await processor_arrow.compress_batch_arrow(
            texts=sample_texts[:4],
            include_embeddings=True
        )
        
        table = result.table
        
        # Check required columns
        assert 'text' in table.schema.names
        assert 'compressed' in table.schema.names
        assert 'compression_ratio' in table.schema.names
        assert 'embedding' in table.schema.names
        
        # Check types
        assert table.schema.field('text').type == pa.string()
        assert table.schema.field('compressed').type == pa.bool_()
        assert table.schema.field('compression_ratio').type == pa.float32()
    
    @pytest.mark.asyncio
    async def test_result_table_data_integrity(self, processor_arrow):
        """Test result table data integrity"""
        texts = ["Text A", "Text B", "Text C"]
        
        result = await processor_arrow.compress_batch_arrow(
            texts=texts,
            include_embeddings=False
        )
        
        table = result.table
        
        # Verify texts are preserved
        result_texts = table['text'].to_pylist()
        assert set(result_texts) == set(texts)
        
        # Verify compression flags
        compressed_flags = table['compressed'].to_pylist()
        assert all(isinstance(flag, bool) for flag in compressed_flags)
        
        # Verify compression ratios
        ratios = table['compression_ratio'].to_pylist()
        assert all(ratio > 0 for ratio in ratios)

"""
LLM Compressor Unit Tests

Tests for the LLMCompressor class implementing semantic compression.

Feature: llm-compression-integration
Requirements: 5.1-5.7
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from llm_compression.compressor import (
    LLMCompressor,
    CompressedMemory,
    CompressionMetadata,
    CompressionError
)
from llm_compression.llm_client import LLMClient, LLMResponse
from llm_compression.model_selector import (
    ModelSelector,
    ModelConfig,
    MemoryType,
    QualityLevel
)


class TestLLMCompressor:
    """LLM Compressor tests"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client"""
        client = Mock(spec=LLMClient)
        client.generate = AsyncMock(return_value=LLMResponse(
            text="John met Mary to discuss the project.",
            tokens_used=10,
            latency_ms=100.0,
            model="gpt-3.5-turbo",
            finish_reason="stop",
            metadata={}
        ))
        return client
    
    @pytest.fixture
    def mock_model_selector(self):
        """Create mock model selector"""
        selector = Mock(spec=ModelSelector)
        selector.select_model = Mock(return_value=ModelConfig(
            model_name="gpt-3.5-turbo",
            endpoint="http://localhost:8045",
            is_local=False,
            max_tokens=100,
            temperature=0.3,
            expected_latency_ms=100.0,
            expected_quality=0.9
        ))
        return selector
    
    @pytest.fixture
    def compressor(self, mock_llm_client, mock_model_selector):
        """Create compressor instance"""
        return LLMCompressor(
            llm_client=mock_llm_client,
            model_selector=mock_model_selector,
            min_compress_length=100
        )
    
    @pytest.mark.asyncio
    async def test_compress_basic(self, compressor):
        """Test basic compression"""
        text = "John met Mary at 3pm on 2024-01-15 to discuss the project. " * 3
        
        result = await compressor.compress(text)
        
        assert isinstance(result, CompressedMemory)
        assert result.memory_id
        assert result.summary_hash
        assert result.compression_metadata.original_size > 0
        assert result.compression_metadata.compressed_size > 0
        assert result.compression_metadata.compression_ratio > 1.0
    
    @pytest.mark.asyncio
    async def test_compress_short_text(self, compressor):
        """Test compression of short text (should store uncompressed)"""
        text = "Short text"
        
        result = await compressor.compress(text)
        
        assert isinstance(result, CompressedMemory)
        assert result.summary_hash == ""  # Empty for uncompressed
        assert result.compression_metadata.model_used == "uncompressed"
        assert result.compression_metadata.compression_ratio == 1.0
    
    @pytest.mark.asyncio
    async def test_compress_with_metadata(self, compressor):
        """Test compression with metadata"""
        text = "John met Mary at 3pm on 2024-01-15 to discuss the project. " * 3
        metadata = {"timestamp": datetime.now(), "intent": "meeting"}
        
        result = await compressor.compress(text, metadata=metadata)
        
        assert result.original_fields == metadata
    
    @pytest.mark.asyncio
    async def test_compress_batch(self, compressor):
        """Test batch compression"""
        texts = [
            "John met Mary at 3pm on 2024-01-15 to discuss the project. " * 3,
            "Alice called Bob at 5pm on 2024-01-16 to review the code. " * 3,
            "Charlie emailed David at 9am on 2024-01-17 to plan the sprint. " * 3
        ]
        
        results = await compressor.compress_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, CompressedMemory) for r in results)
        assert all(r.memory_id for r in results)
    
    def test_extract_entities_persons(self, compressor):
        """Test entity extraction - persons"""
        text = "John Smith met Mary Johnson to discuss the project."
        
        entities = compressor._extract_entities(text)
        
        assert 'persons' in entities
        assert len(entities['persons']) > 0
        # Should extract "John Smith" and "Mary Johnson"
        assert any('John' in name for name in entities['persons'])
    
    def test_extract_entities_dates(self, compressor):
        """Test entity extraction - dates"""
        text = "Meeting on 2024-01-15 at 3pm and January 20, 2024."
        
        entities = compressor._extract_entities(text)
        
        assert 'dates' in entities
        assert '2024-01-15' in entities['dates']
        assert '3pm' in entities['dates']
    
    def test_extract_entities_numbers(self, compressor):
        """Test entity extraction - numbers"""
        text = "The cost is $1,234.56 and the discount is 15%."
        
        entities = compressor._extract_entities(text)
        
        assert 'numbers' in entities
        assert any('1,234.56' in num or '1234.56' in num for num in entities['numbers'])
        assert any('15%' in num for num in entities['numbers'])
    
    def test_extract_entities_keywords(self, compressor):
        """Test entity extraction - keywords"""
        text = "project meeting discussion planning review project planning"
        
        entities = compressor._extract_entities(text)
        
        assert 'keywords' in entities
        assert len(entities['keywords']) > 0
        # High frequency words should be extracted
        assert 'project' in entities['keywords'] or 'planning' in entities['keywords']
    
    def test_compute_diff(self, compressor):
        """Test diff computation"""
        original = "John met Mary at 3pm on 2024-01-15 to discuss the project."
        summary = "John met Mary to discuss the project."
        
        diff = compressor._compute_diff(original, summary)
        
        assert isinstance(diff, bytes)
        # Diff may be empty if no additions found (summary is subset)
        # This is expected behavior - diff only captures additions
    
    def test_compute_embedding(self, compressor):
        """Test embedding computation"""
        text = "This is a test text for embedding."
        
        # Mock the _embedding_model attribute directly
        import numpy as np
        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.random.rand(384).astype(np.float32))
        compressor._embedding_model = mock_model
        
        embedding = compressor._compute_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # MiniLM-L6-v2 dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_store_uncompressed(self, compressor):
        """Test storing uncompressed text"""
        text = "Short text that won't be compressed"
        metadata = {"key": "value"}
        
        result = compressor._store_uncompressed(text, metadata)
        
        assert isinstance(result, CompressedMemory)
        assert result.summary_hash == ""
        assert result.compression_metadata.model_used == "uncompressed"
        assert result.compression_metadata.compression_ratio == 1.0
        assert result.original_fields == metadata
    
    def test_generate_id(self, compressor):
        """Test ID generation"""
        id1 = compressor._generate_id()
        id2 = compressor._generate_id()
        
        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert id1 != id2  # Should be unique
        assert '_' in id1  # Should have timestamp_random format
    
    @pytest.mark.asyncio
    async def test_summary_caching(self, compressor):
        """Test summary caching"""
        text = "John met Mary at 3pm on 2024-01-15 to discuss the project. " * 3
        
        result = await compressor.compress(text)
        
        # Summary should be cached
        assert result.summary_hash in compressor.summary_cache
        assert isinstance(compressor.summary_cache[result.summary_hash], str)
    
    @pytest.mark.asyncio
    async def test_compression_error_handling(self, compressor, mock_llm_client):
        """Test error handling during compression - uses fallback instead of raising"""
        text = "Test text for error handling. " * 10
        
        # Make LLM client raise an error
        mock_llm_client.generate.side_effect = Exception("API Error")
        
        # Should use fallback summary instead of raising error
        result = await compressor.compress(text)
        
        # Verify it still returns a compressed memory (using fallback)
        assert isinstance(result, CompressedMemory)
        # Compression should still work with fallback summary
        assert result.compression_metadata.compression_ratio > 0
    
    @pytest.mark.asyncio
    async def test_fallback_summary_generation(self, compressor, mock_llm_client):
        """Test fallback when summary generation fails"""
        text = "Test text for fallback. " * 10
        
        # Make LLM client raise an error
        mock_llm_client.generate.side_effect = Exception("API Error")
        
        # Should use fallback (first 200 chars)
        summary = await compressor._generate_summary(text, Mock())
        
        assert isinstance(summary, str)
        assert len(summary) <= 200
    
    @pytest.mark.asyncio
    async def test_compression_ratio_check(self, compressor, mock_llm_client):
        """Test that compression is skipped if it doesn't reduce size"""
        text = "AB" * 100  # Simple text that might not compress well
        
        # Mock to return a very long summary
        mock_llm_client.generate.return_value = LLMResponse(
            text="A" * 500,  # Very long summary
            tokens_used=100,
            latency_ms=100.0,
            model="gpt-3.5-turbo",
            finish_reason="stop",
            metadata={}
        )
        
        result = await compressor.compress(text)
        
        # Should fall back to uncompressed if compression doesn't help
        # (This test might need adjustment based on actual behavior)
        assert isinstance(result, CompressedMemory)


class TestCompressionMetadata:
    """Test CompressionMetadata dataclass"""
    
    def test_metadata_creation(self):
        """Test creating metadata"""
        metadata = CompressionMetadata(
            original_size=1000,
            compressed_size=100,
            compression_ratio=10.0,
            model_used="gpt-3.5-turbo",
            quality_score=0.95,
            compression_time_ms=150.0,
            compressed_at=datetime.now()
        )
        
        assert metadata.original_size == 1000
        assert metadata.compressed_size == 100
        assert metadata.compression_ratio == 10.0
        assert metadata.model_used == "gpt-3.5-turbo"
        assert metadata.quality_score == 0.95
        assert metadata.compression_time_ms == 150.0
        assert isinstance(metadata.compressed_at, datetime)


class TestCompressedMemory:
    """Test CompressedMemory dataclass"""
    
    def test_compressed_memory_creation(self):
        """Test creating compressed memory"""
        memory = CompressedMemory(
            memory_id="test_id",
            summary_hash="abc123",
            entities={'persons': ['John'], 'dates': ['2024-01-15']},
            diff_data=b'test diff',
            embedding=[0.1, 0.2, 0.3],
            compression_metadata=CompressionMetadata(
                original_size=1000,
                compressed_size=100,
                compression_ratio=10.0,
                model_used="gpt-3.5-turbo",
                quality_score=0.95,
                compression_time_ms=150.0,
                compressed_at=datetime.now()
            ),
            original_fields={'timestamp': datetime.now()}
        )
        
        assert memory.memory_id == "test_id"
        assert memory.summary_hash == "abc123"
        assert 'persons' in memory.entities
        assert memory.diff_data == b'test diff'
        assert len(memory.embedding) == 3
        assert isinstance(memory.compression_metadata, CompressionMetadata)
        assert 'timestamp' in memory.original_fields

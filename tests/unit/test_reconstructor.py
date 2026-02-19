"""
LLM Reconstructor Unit Tests

Tests for the LLMReconstructor class implementing memory reconstruction.

Feature: llm-compression-integration
Requirements: 6.1-6.7
"""

import pytest
import asyncio
try:
    import zstandard as zstd
except ImportError:
    import zstd
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from llm_compression.reconstructor import (
    LLMReconstructor,
    ReconstructedMemory,
    QualityMetrics,
    ReconstructionError
)
from llm_compression.compressor import CompressedMemory, CompressionMetadata
from llm_compression.llm_client import LLMClient, LLMResponse


class TestLLMReconstructor:
    """LLM Reconstructor tests"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client"""
        client = Mock(spec=LLMClient)
        client.generate = AsyncMock(return_value=LLMResponse(
            text="John met Mary at 3pm on 2024-01-15 to discuss the project.",
            tokens_used=15,
            latency_ms=100.0,
            model="gpt-3.5-turbo",
            finish_reason="stop",
            metadata={}
        ))
        return client
    
    @pytest.fixture
    def reconstructor(self, mock_llm_client):
        """Create reconstructor instance"""
        return LLMReconstructor(
            llm_client=mock_llm_client,
            quality_threshold=0.85
        )
    
    @pytest.fixture
    def sample_compressed_memory(self):
        """Create sample compressed memory"""
        entities = {
            'persons': ['John Smith', 'Mary Johnson'],
            'dates': ['2024-01-15', '3pm'],
            'numbers': ['100'],
            'locations': [],
            'keywords': ['meeting', 'project']
        }
        
        diff_text = "at 3pm\non 2024-01-15"
        diff_data = zstd.compress(diff_text.encode('utf-8'), level=3)
        
        return CompressedMemory(
            memory_id="test_123",
            summary_hash="abc123def456",
            entities=entities,
            diff_data=diff_data,
            embedding=[0.1] * 384,
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
    
    @pytest.mark.asyncio
    async def test_reconstruct_basic(self, reconstructor, sample_compressed_memory):
        """Test basic reconstruction"""
        result = await reconstructor.reconstruct(sample_compressed_memory)
        
        assert isinstance(result, ReconstructedMemory)
        assert result.memory_id == "test_123"
        assert result.full_text
        assert result.reconstruction_time_ms > 0
        assert 0.0 <= result.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_reconstruct_with_quality_verification(
        self,
        reconstructor,
        sample_compressed_memory
    ):
        """Test reconstruction with quality verification"""
        result = await reconstructor.reconstruct(
            sample_compressed_memory,
            verify_quality=True
        )
        
        assert result.quality_metrics is not None
        assert isinstance(result.quality_metrics, QualityMetrics)
        assert 0.0 <= result.quality_metrics.overall_score <= 1.0
        assert 0.0 <= result.quality_metrics.entity_accuracy <= 1.0
        assert 0.0 <= result.quality_metrics.coherence_score <= 1.0
        assert 0.0 <= result.quality_metrics.length_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_reconstruct_without_quality_verification(
        self,
        reconstructor,
        sample_compressed_memory
    ):
        """Test reconstruction without quality verification"""
        result = await reconstructor.reconstruct(
            sample_compressed_memory,
            verify_quality=False
        )
        
        assert result.quality_metrics is None
        assert result.confidence == 1.0  # Default when no verification
    
    @pytest.mark.asyncio
    async def test_reconstruct_batch(self, reconstructor, sample_compressed_memory):
        """Test batch reconstruction"""
        compressed_list = [sample_compressed_memory] * 3
        
        results = await reconstructor.reconstruct_batch(compressed_list)
        
        assert len(results) == 3
        assert all(isinstance(r, ReconstructedMemory) for r in results)
        assert all(r.memory_id for r in results)
    
    def test_lookup_summary_cache_hit(self, reconstructor):
        """Test summary lookup - cache hit"""
        summary_hash = "test_hash"
        summary_text = "This is a cached summary"
        
        # Add to cache
        reconstructor.summary_cache[summary_hash] = summary_text
        
        result = reconstructor._lookup_summary(summary_hash)
        
        assert result == summary_text
    
    def test_lookup_summary_cache_miss(self, reconstructor):
        """Test summary lookup - cache miss"""
        summary_hash = "nonexistent_hash"
        
        result = reconstructor._lookup_summary(summary_hash)
        
        assert result == ""  # Returns empty string when not found
    
    def test_cache_summary(self, reconstructor):
        """Test summary caching"""
        summary_hash = "new_hash"
        summary_text = "New summary to cache"
        
        reconstructor._cache_summary(summary_hash, summary_text)
        
        assert summary_hash in reconstructor.summary_cache
        assert reconstructor.summary_cache[summary_hash] == summary_text
    
    def test_cache_summary_lru_eviction(self, reconstructor):
        """Test LRU eviction when cache is full"""
        reconstructor.max_cache_size = 3
        
        # Fill cache
        reconstructor._cache_summary("hash1", "summary1")
        reconstructor._cache_summary("hash2", "summary2")
        reconstructor._cache_summary("hash3", "summary3")
        
        assert len(reconstructor.summary_cache) == 3
        
        # Add one more - should evict oldest
        reconstructor._cache_summary("hash4", "summary4")
        
        assert len(reconstructor.summary_cache) == 3
        assert "hash1" not in reconstructor.summary_cache  # Oldest evicted
        assert "hash4" in reconstructor.summary_cache
    
    @pytest.mark.asyncio
    async def test_expand_summary(self, reconstructor):
        """Test summary expansion"""
        summary = "John met Mary to discuss the project."
        entities = {
            'persons': ['John Smith', 'Mary Johnson'],
            'dates': ['2024-01-15'],
            'numbers': [],
            'locations': [],
            'keywords': ['meeting']
        }
        
        result = await reconstructor._expand_summary(summary, entities)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_expand_summary_error_handling(self, reconstructor, mock_llm_client):
        """Test summary expansion error handling"""
        summary = "Test summary"
        entities = {}
        
        # Make LLM client raise error
        mock_llm_client.generate.side_effect = Exception("API Error")
        
        result = await reconstructor._expand_summary(summary, entities)
        
        # Should return original summary on error
        assert result == summary
    
    def test_format_entities(self, reconstructor):
        """Test entity formatting"""
        entities = {
            'persons': ['John Smith', 'Mary Johnson'],
            'dates': ['2024-01-15'],
            'numbers': ['100', '200'],
            'locations': [],
            'keywords': ['meeting', 'project']
        }
        
        result = reconstructor._format_entities(entities)
        
        assert isinstance(result, str)
        assert 'persons:' in result
        assert 'John Smith' in result
        assert 'dates:' in result
    
    def test_format_entities_empty(self, reconstructor):
        """Test entity formatting with empty entities"""
        entities = {
            'persons': [],
            'dates': [],
            'numbers': [],
            'locations': [],
            'keywords': []
        }
        
        result = reconstructor._format_entities(entities)
        
        assert result == "none"
    
    def test_apply_diff(self, reconstructor):
        """Test diff application"""
        reconstructed = "John met Mary to discuss the project."
        diff_text = "at 3pm\non 2024-01-15"
        diff_data = zstd.compress(diff_text.encode('utf-8'), level=3)
        
        result = reconstructor._apply_diff(reconstructed, diff_data)
        
        assert isinstance(result, str)
        assert len(result) > len(reconstructed)
        assert "3pm" in result
        assert "2024-01-15" in result
    
    def test_apply_diff_empty(self, reconstructor):
        """Test diff application with empty diff"""
        reconstructed = "Test text"
        diff_data = b''
        
        result = reconstructor._apply_diff(reconstructed, diff_data)
        
        assert result == reconstructed
    
    def test_apply_diff_error_handling(self, reconstructor):
        """Test diff application error handling"""
        reconstructed = "Test text"
        diff_data = b'invalid_compressed_data'
        
        result = reconstructor._apply_diff(reconstructed, diff_data)
        
        # Should return original on error
        assert result == reconstructed
    
    def test_verify_reconstruction_quality(self, reconstructor):
        """Test quality verification"""
        reconstructed = "John Smith met Mary Johnson on 2024-01-15 at 3pm to discuss the project."
        expected_entities = {
            'persons': ['John Smith', 'Mary Johnson'],
            'dates': ['2024-01-15', '3pm'],
            'numbers': [],
            'locations': [],
            'keywords': ['meeting']
        }
        
        result = reconstructor._verify_reconstruction_quality(
            reconstructed,
            expected_entities
        )
        
        assert isinstance(result, QualityMetrics)
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.entity_accuracy <= 1.0
        assert 0.0 <= result.coherence_score <= 1.0
        assert 0.0 <= result.length_score <= 1.0
    
    def test_check_entity_completeness_full(self, reconstructor):
        """Test entity completeness check - all entities present"""
        text = "John Smith met Mary Johnson on 2024-01-15"
        expected_entities = {
            'persons': ['John Smith', 'Mary Johnson'],
            'dates': ['2024-01-15'],
            'numbers': [],
            'locations': [],
            'keywords': []
        }
        warnings = []
        
        accuracy = reconstructor._check_entity_completeness(
            text,
            expected_entities,
            warnings
        )
        
        assert accuracy == 1.0
        assert len(warnings) == 0
    
    def test_check_entity_completeness_partial(self, reconstructor):
        """Test entity completeness check - some entities missing"""
        text = "John Smith met someone on some date"
        expected_entities = {
            'persons': ['John Smith', 'Mary Johnson'],
            'dates': ['2024-01-15'],
            'numbers': [],
            'locations': [],
            'keywords': []
        }
        warnings = []
        
        accuracy = reconstructor._check_entity_completeness(
            text,
            expected_entities,
            warnings
        )
        
        assert 0.0 < accuracy < 1.0
        assert len(warnings) > 0
    
    def test_check_coherence_good(self, reconstructor):
        """Test coherence check - good text"""
        text = "This is a well-formed sentence with proper punctuation."
        warnings = []
        
        score = reconstructor._check_coherence(text, warnings)
        
        assert score == 1.0
        assert len(warnings) == 0
    
    def test_check_coherence_no_ending_punctuation(self, reconstructor):
        """Test coherence check - no ending punctuation"""
        text = "This sentence has no ending punctuation"
        warnings = []
        
        score = reconstructor._check_coherence(text, warnings)
        
        assert score < 1.0
        assert any("punctuation" in w.lower() for w in warnings)
    
    def test_check_coherence_high_repetition(self, reconstructor):
        """Test coherence check - high word repetition"""
        text = "test test test test test."
        warnings = []
        
        score = reconstructor._check_coherence(text, warnings)
        
        assert score < 1.0
        assert any("repetition" in w.lower() for w in warnings)
    
    def test_check_length_reasonableness_good(self, reconstructor):
        """Test length reasonableness - good length"""
        text = "This is a reasonable length text with several words and more content."
        expected_entities = {
            'persons': ['John'],
            'dates': ['2024-01-15'],
            'numbers': [],
            'locations': [],
            'keywords': []
        }
        warnings = []
        
        score = reconstructor._check_length_reasonableness(
            text,
            expected_entities,
            warnings
        )
        
        assert score >= 0.9  # Allow for slight variations
        assert len(warnings) == 0
    
    def test_check_length_reasonableness_too_short(self, reconstructor):
        """Test length reasonableness - too short"""
        text = "Short"
        expected_entities = {
            'persons': ['John', 'Mary', 'Bob'],
            'dates': ['2024-01-15'],
            'numbers': ['100'],
            'locations': [],
            'keywords': []
        }
        warnings = []
        
        score = reconstructor._check_length_reasonableness(
            text,
            expected_entities,
            warnings
        )
        
        assert score < 1.0
        assert any("too short" in w.lower() for w in warnings)
    
    @pytest.mark.asyncio
    async def test_reconstruct_from_diff_only(self, reconstructor, sample_compressed_memory):
        """Test reconstruction from diff only (fallback)"""
        result = await reconstructor._reconstruct_from_diff_only(sample_compressed_memory)
        
        assert isinstance(result, ReconstructedMemory)
        assert result.confidence == 0.5  # Lower confidence for diff-only
        assert any("diff only" in w.lower() for w in result.warnings)
    
    @pytest.mark.asyncio
    async def test_reconstruct_low_quality_warning(
        self,
        reconstructor,
        sample_compressed_memory,
        mock_llm_client
    ):
        """Test that low quality reconstruction generates warning"""
        # Make LLM return very short text
        mock_llm_client.generate.return_value = LLMResponse(
            text="Short",
            tokens_used=1,
            latency_ms=100.0,
            model="gpt-3.5-turbo",
            finish_reason="stop",
            metadata={}
        )
        
        result = await reconstructor.reconstruct(
            sample_compressed_memory,
            verify_quality=True
        )
        
        # Should have warnings due to low quality
        assert len(result.warnings) > 0
        assert result.confidence < reconstructor.quality_threshold
    
    @pytest.mark.asyncio
    async def test_reconstruct_batch_with_errors(
        self,
        reconstructor,
        sample_compressed_memory,
        mock_llm_client
    ):
        """Test batch reconstruction with some failures"""
        # Make second call fail
        call_count = [0]
        
        async def mock_generate_with_error(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("API Error")
            return LLMResponse(
                text="Reconstructed text",
                tokens_used=10,
                latency_ms=100.0,
                model="gpt-3.5-turbo",
                finish_reason="stop",
                metadata={}
            )
        
        mock_llm_client.generate = AsyncMock(side_effect=mock_generate_with_error)
        
        compressed_list = [sample_compressed_memory] * 3
        results = await reconstructor.reconstruct_batch(compressed_list)
        
        assert len(results) == 3
        # Second one should have error - but it still reconstructs from diff
        # So it won't have confidence 0.0, but will have warnings
        assert len(results[1].warnings) > 0


class TestQualityMetrics:
    """Test QualityMetrics dataclass"""
    
    def test_quality_metrics_creation(self):
        """Test creating quality metrics"""
        metrics = QualityMetrics(
            entity_accuracy=0.95,
            coherence_score=0.90,
            length_score=0.85,
            overall_score=0.90,
            warnings=["Test warning"]
        )
        
        assert metrics.entity_accuracy == 0.95
        assert metrics.coherence_score == 0.90
        assert metrics.length_score == 0.85
        assert metrics.overall_score == 0.90
        assert len(metrics.warnings) == 1


class TestReconstructedMemory:
    """Test ReconstructedMemory dataclass"""
    
    def test_reconstructed_memory_creation(self):
        """Test creating reconstructed memory"""
        memory = ReconstructedMemory(
            memory_id="test_id",
            full_text="Reconstructed text",
            quality_metrics=QualityMetrics(
                entity_accuracy=0.95,
                coherence_score=0.90,
                length_score=0.85,
                overall_score=0.90,
                warnings=[]
            ),
            reconstruction_time_ms=150.0,
            confidence=0.90,
            warnings=["Test warning"],
            original_fields={'timestamp': datetime.now()}
        )
        
        assert memory.memory_id == "test_id"
        assert memory.full_text == "Reconstructed text"
        assert isinstance(memory.quality_metrics, QualityMetrics)
        assert memory.reconstruction_time_ms == 150.0
        assert memory.confidence == 0.90
        assert len(memory.warnings) == 1
        assert 'timestamp' in memory.original_fields

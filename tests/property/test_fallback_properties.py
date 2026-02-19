"""
Property-Based Tests for Fallback Strategy

Tests Property 10: Model Fallback Strategy

Feature: llm-compression-integration
Requirements: 13.1, 13.2, 13.3
Property 10: Model Fallback Strategy
"""

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from unittest.mock import Mock, AsyncMock

from llm_compression.fallback import FallbackStrategy
from llm_compression.compressor import LLMCompressor, MemoryType, CompressedMemory, CompressionMetadata
from llm_compression.model_selector import ModelSelector, ModelConfig, QualityLevel
from llm_compression.llm_client import LLMClient
from llm_compression.errors import LLMAPIError, LLMTimeoutError, GPUResourceError
from datetime import datetime


# Test strategies
text_strategy = st.text(min_size=100, max_size=1000, alphabet=st.characters(blacklist_categories=('Cs',)))
memory_type_strategy = st.sampled_from([MemoryType.TEXT, MemoryType.CODE, MemoryType.LONG_TEXT])


def create_mock_llm_client():
    """Create mock LLM client"""
    client = Mock(spec=LLMClient)
    client.generate = AsyncMock()
    return client


def create_mock_model_selector():
    """Create mock model selector"""
    return Mock(spec=ModelSelector)


def create_mock_compressor():
    """Create mock compressor"""
    compressor = Mock(spec=LLMCompressor)
    compressor.llm_client = create_mock_llm_client()
    compressor.model_selector = create_mock_model_selector()
    compressor.min_compress_length = 100
    compressor._compute_embedding = Mock(return_value=[0.1] * 384)
    compressor._generate_id = Mock(return_value="test_id_123")
    return compressor


def create_fallback_strategy():
    """Create fallback strategy instance"""
    compressor = create_mock_compressor()
    selector = create_mock_model_selector()
    return FallbackStrategy(
        compressor=compressor,
        model_selector=selector,
        enable_local_fallback=True,
        enable_simple_compression=True
    ), compressor, selector


class TestFallbackStrategyProperties:
    """
    Property-based tests for fallback strategy
    
    Feature: llm-compression-integration, Property 10: Model Fallback Strategy
    """
    
    @pytest.mark.asyncio
    @settings(
        max_examples=30,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        text=text_strategy,
        memory_type=memory_type_strategy
    )
    async def test_property_10_simple_compression_fallback(
        self,
        text,
        memory_type
    ):
        """
        Feature: llm-compression-integration, Property 10: Model Fallback Strategy
        
        Test: *For any* compression request, if both Cloud API and Local Model fail,
        should use simple compression (Phase 0)
        
        Validates: Requirements 13.2
        """
        fallback_strategy, mock_compressor, mock_model_selector = create_fallback_strategy()
        
        # Setup: Both Cloud API and Local Model fail
        mock_model_selector.select_model.side_effect = [
            ModelConfig(
                model_name="cloud-api",
                endpoint="http://localhost:8045",
                is_local=False,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=1000,
                expected_quality=0.95
            ),
            ModelConfig(
                model_name="local-model",
                endpoint="http://localhost:8046",
                is_local=True,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=500,
                expected_quality=0.85
            )
        ]
        
        mock_compressor.compress.side_effect = [
            LLMAPIError("Cloud API unavailable"),
            LLMTimeoutError("Local model timeout")
        ]
        
        # Execute
        result = await fallback_strategy.compress_with_fallback(
            text=text,
            memory_type=memory_type
        )
        
        # Verify: Should use simple compression
        assert result is not None
        assert result.compression_metadata.model_used == "zstd-level9"
        assert result.summary_hash == "simple_compression"
        
        # Verify: Both higher levels were tried
        stats = fallback_strategy.get_fallback_stats()
        assert stats['cloud_api_failure'] >= 1
        assert stats['local_model_failure'] >= 1
        assert stats['simple_compression_success'] >= 1
    
    @pytest.mark.asyncio
    @settings(
        max_examples=30,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        text=text_strategy,
        memory_type=memory_type_strategy
    )
    async def test_property_10_direct_storage_fallback(
        self,
        text,
        memory_type
    ):
        """
        Feature: llm-compression-integration, Property 10: Model Fallback Strategy
        
        Test: *For any* compression request, if all compression methods fail,
        should store uncompressed
        
        Validates: Requirements 13.3
        """
        mock_compressor = create_mock_compressor()
        mock_model_selector = create_mock_model_selector()
        
        # Setup: All compression methods fail
        fallback_strategy = FallbackStrategy(
            compressor=mock_compressor,
            model_selector=mock_model_selector,
            enable_local_fallback=True,
            enable_simple_compression=False  # Disable simple compression to force direct storage
        )
        
        mock_model_selector.select_model.side_effect = [
            ModelConfig(
                model_name="cloud-api",
                endpoint="http://localhost:8045",
                is_local=False,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=1000,
                expected_quality=0.95
            ),
            ModelConfig(
                model_name="local-model",
                endpoint="http://localhost:8046",
                is_local=True,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=500,
                expected_quality=0.85
            )
        ]
        
        mock_compressor.compress.side_effect = [
            LLMAPIError("Cloud API unavailable"),
            GPUResourceError("GPU out of memory")
        ]
        
        # Execute
        result = await fallback_strategy.compress_with_fallback(
            text=text,
            memory_type=memory_type
        )
        
        # Verify: Should store uncompressed
        assert result is not None
        assert result.compression_metadata.model_used == "uncompressed"
        assert result.compression_metadata.compression_ratio == 1.0
        assert result.summary_hash == ""
        
        # Verify: All levels were tried
        stats = fallback_strategy.get_fallback_stats()
        assert stats['direct_storage'] >= 1
    
    @pytest.mark.asyncio
    @settings(
        max_examples=20,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        text=text_strategy,
        memory_type=memory_type_strategy
    )
    async def test_property_10_always_returns_result(
        self,
        text,
        memory_type
    ):
        """
        Feature: llm-compression-integration, Property 10: Model Fallback Strategy
        
        Test: *For any* compression request, fallback strategy should ALWAYS
        return a result (never raise exception)
        
        Validates: Requirements 13.1, 13.2, 13.3
        """
        fallback_strategy, mock_compressor, mock_model_selector = create_fallback_strategy()
        
        # Setup: Random failures
        mock_model_selector.select_model.side_effect = [
            ModelConfig(
                model_name="cloud-api",
                endpoint="http://localhost:8045",
                is_local=False,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=1000,
                expected_quality=0.95
            ),
            ModelConfig(
                model_name="local-model",
                endpoint="http://localhost:8046",
                is_local=True,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=500,
                expected_quality=0.85
            )
        ]
        
        mock_compressor.compress.side_effect = [
            LLMAPIError("Random failure"),
            LLMTimeoutError("Random timeout")
        ]
        
        # Execute - should never raise exception
        result = await fallback_strategy.compress_with_fallback(
            text=text,
            memory_type=memory_type
        )
        
        # Verify: Always returns a result
        assert result is not None
        assert hasattr(result, 'compression_metadata')
        assert hasattr(result, 'diff_data')
        assert len(result.diff_data) > 0


class TestSimpleCompression:
    """Test simple compression (Phase 0) functionality"""
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=3000)
    @given(text=text_strategy)
    async def test_simple_compression_lossless(
        self,
        text
    ):
        """
        Test: Simple compression should be lossless
        
        Validates: Requirements 13.2
        """
        import zstandard as zstd
        
        fallback_strategy, _, _ = create_fallback_strategy()
        
        # Execute simple compression
        result = fallback_strategy._simple_compress(text, {})
        
        # Verify: Can decompress to original
        decompressed = zstd.decompress(result.diff_data).decode('utf-8')
        assert decompressed == text
        
        # Verify: Metadata is correct
        assert result.compression_metadata.model_used == "zstd-level9"
        assert result.compression_metadata.quality_score == 1.0
        assert result.summary_hash == "simple_compression"


class TestDirectStorage:
    """Test direct storage (uncompressed) functionality"""
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=3000)
    @given(text=text_strategy)
    async def test_direct_storage_preserves_text(
        self,
        text
    ):
        """
        Test: Direct storage should preserve original text exactly
        
        Validates: Requirements 13.3
        """
        fallback_strategy, _, _ = create_fallback_strategy()
        
        # Execute direct storage
        result = fallback_strategy._store_uncompressed(text, {})
        
        # Verify: Original text is preserved
        stored_text = result.diff_data.decode('utf-8')
        assert stored_text == text
        
        # Verify: Metadata indicates no compression
        assert result.compression_metadata.model_used == "uncompressed"
        assert result.compression_metadata.compression_ratio == 1.0
        assert result.summary_hash == ""

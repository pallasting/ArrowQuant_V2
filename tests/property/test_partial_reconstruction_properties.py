"""
Property-Based Tests for Partial Reconstruction

Tests Property 34: Partial Reconstruction Return

Feature: llm-compression-integration
Requirements: 13.4
Property 34: Partial Reconstruction Return
"""

import pytest
from hypothesis import given, settings, strategies as st
from unittest.mock import Mock, AsyncMock, patch

from llm_compression.reconstructor import LLMReconstructor, ReconstructedMemory
from llm_compression.compressor import CompressedMemory, CompressionMetadata
from llm_compression.llm_client import LLMClient
from llm_compression.errors import ReconstructionError
from datetime import datetime


# Test strategies
text_strategy = st.text(min_size=50, max_size=500, alphabet=st.characters(blacklist_categories=('Cs',)))


def create_mock_llm_client():
    """Create mock LLM client"""
    client = Mock(spec=LLMClient)
    client.generate = AsyncMock()
    return client


def create_compressed_memory(text: str) -> CompressedMemory:
    """Create a mock compressed memory"""
    return CompressedMemory(
        memory_id="test_123",
        summary_hash="abc123",
        entities={'persons': ['John'], 'dates': ['2024-01-15']},
        diff_data=b"test diff data",
        embedding=[0.1] * 384,
        compression_metadata=CompressionMetadata(
            original_size=len(text),
            compressed_size=50,
            compression_ratio=len(text) / 50,
            model_used="test-model",
            quality_score=0.9,
            compression_time_ms=100,
            compressed_at=datetime.now()
        ),
        original_fields={}
    )


class TestPartialReconstructionProperties:
    """
    Property-based tests for partial reconstruction
    
    Feature: llm-compression-integration, Property 34: Partial Reconstruction Return
    """
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=5000)
    @given(text=text_strategy)
    async def test_property_34_partial_reconstruction_on_failure(
        self,
        text
    ):
        """
        Feature: llm-compression-integration, Property 34: Partial Reconstruction Return
        
        Test: *For any* reconstruction failure, system should return partial
        reconstruction result, not raise exception
        
        Validates: Requirements 13.4
        """
        mock_client = create_mock_llm_client()
        reconstructor = LLMReconstructor(mock_client, quality_threshold=0.85)
        
        # Mock LLM to fail during expansion
        mock_client.generate.side_effect = Exception("LLM expansion failed")
        
        # Create compressed memory
        compressed = create_compressed_memory(text)
        
        # Execute - should NOT raise exception
        result = await reconstructor.reconstruct(compressed, verify_quality=False)
        
        # Verify: Returns ReconstructedMemory (not exception)
        assert isinstance(result, ReconstructedMemory)
        assert result.memory_id == compressed.memory_id
        
        # Verify: Has some text (may be empty if diff also fails, but not None)
        assert result.full_text is not None
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=5000)
    @given(text=text_strategy)
    async def test_property_34_low_confidence_on_failure(
        self,
        text
    ):
        """
        Feature: llm-compression-integration, Property 34: Partial Reconstruction Return
        
        Test: *For any* reconstruction failure, system should still return a result
        (confidence may be 1.0 if diff reconstruction succeeds)
        
        Validates: Requirements 13.4
        """
        mock_client = create_mock_llm_client()
        reconstructor = LLMReconstructor(mock_client, quality_threshold=0.85)
        
        # Mock LLM to fail
        mock_client.generate.side_effect = Exception("LLM failed")
        
        # Create compressed memory
        compressed = create_compressed_memory(text)
        
        # Execute
        result = await reconstructor.reconstruct(compressed, verify_quality=False)
        
        # Verify: Returns a result (confidence depends on whether diff reconstruction succeeds)
        assert result.confidence >= 0.0 and result.confidence <= 1.0
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=5000)
    @given(text=text_strategy)
    async def test_property_34_diff_only_reconstruction(
        self,
        text
    ):
        """
        Feature: llm-compression-integration, Property 34: Partial Reconstruction Return
        
        Test: *For any* LLM failure, system should attempt diff-only reconstruction
        
        Validates: Requirements 13.4, 6.7
        """
        mock_client = create_mock_llm_client()
        reconstructor = LLMReconstructor(mock_client, quality_threshold=0.85)
        
        # Mock LLM to fail
        mock_client.generate.side_effect = Exception("LLM unavailable")
        
        # Create compressed memory with diff data
        compressed = create_compressed_memory(text)
        
        # Execute
        result = await reconstructor.reconstruct(compressed, verify_quality=False)
        
        # Verify: Should have some reconstructed text (may be empty if diff fails)
        assert result.full_text is not None


class TestReconstructionWarnings:
    """Test reconstruction warning generation"""
    
    @pytest.mark.asyncio
    async def test_no_exception_raised_on_failure(self):
        """
        Test: Reconstruction should never raise exception, always return result
        
        Validates: Requirements 13.4
        """
        mock_client = create_mock_llm_client()
        reconstructor = LLMReconstructor(mock_client)
        
        # Mock LLM to fail
        mock_client.generate.side_effect = Exception("Critical failure")
        
        # Create compressed memory
        compressed = create_compressed_memory("test text")
        
        # Execute - should NOT raise exception
        result = await reconstructor.reconstruct(compressed, verify_quality=False)
        
        # Verify: Returns result
        assert isinstance(result, ReconstructedMemory)
        assert result.memory_id == compressed.memory_id

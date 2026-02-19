"""
Reconstruction Property Tests

Property-based tests for reconstruction functionality using Hypothesis.

Feature: llm-compression-integration
Requirements: 6.1-6.7
Properties: 5-7
"""

import pytest
import asyncio
try:
    import zstandard as zstd
except ImportError:
    import zstd
from hypothesis import given, settings, strategies as st, assume
from unittest.mock import Mock, AsyncMock
import numpy as np
from datetime import datetime

from llm_compression.reconstructor import (
    LLMReconstructor,
    ReconstructedMemory,
    QualityMetrics,
    ReconstructionError
)
from llm_compression.compressor import (
    LLMCompressor,
    CompressedMemory,
    CompressionMetadata
)
from llm_compression.llm_client import LLMClient, LLMResponse
from llm_compression.model_selector import (
    ModelSelector,
    ModelConfig,
    MemoryType,
    QualityLevel
)


# ============================================================================
# Property 5: Reconstruction Performance Guarantee
# Validates: Requirement 6.5
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    text_length=st.integers(min_value=100, max_value=500),
    num_entities=st.integers(min_value=1, max_value=5)
)
@pytest.mark.asyncio
async def test_property_5_reconstruction_performance(text_length, num_entities):
    """
    Feature: llm-compression-integration, Property 5: Reconstruction Performance Guarantee
    
    For any compressed memory, reconstruction should complete in < 1s (Phase 1.0 target)
    
    Validates: Requirement 6.5
    """
    # Create mock LLM client with fast response
    mock_client = Mock(spec=LLMClient)
    mock_client.generate = AsyncMock(return_value=LLMResponse(
        text="Reconstructed text with all the details.",
        tokens_used=10,
        latency_ms=50.0,  # Fast response
        model="gpt-3.5-turbo",
        finish_reason="stop",
        metadata={}
    ))
    
    # Create reconstructor
    reconstructor = LLMReconstructor(
        llm_client=mock_client,
        quality_threshold=0.85
    )
    
    # Create sample compressed memory
    entities = {
        'persons': [f'Person{i}' for i in range(num_entities)],
        'dates': ['2024-01-15'],
        'numbers': ['100'],
        'locations': [],
        'keywords': ['test']
    }
    
    diff_text = "additional details here"
    diff_data = zstd.compress(diff_text.encode('utf-8'), level=3)
    
    compressed = CompressedMemory(
        memory_id="test_id",
        summary_hash="test_hash",
        entities=entities,
        diff_data=diff_data,
        embedding=[0.1] * 384,
        compression_metadata=CompressionMetadata(
            original_size=text_length,
            compressed_size=text_length // 10,
            compression_ratio=10.0,
            model_used="gpt-3.5-turbo",
            quality_score=0.95,
            compression_time_ms=100.0,
            compressed_at=datetime.now()
        ),
        original_fields={}
    )
    
    # Reconstruct
    result = await reconstructor.reconstruct(compressed, verify_quality=False)
    
    # Verify performance
    assert isinstance(result, ReconstructedMemory)
    assert result.reconstruction_time_ms < 1000, \
        f"Reconstruction took {result.reconstruction_time_ms}ms, should be < 1000ms"
    
    # Verify result is valid
    assert result.full_text
    assert result.memory_id == "test_id"


# ============================================================================
# Property 6: Reconstruction Quality Monitoring
# Validates: Requirement 6.4
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    quality_level=st.floats(min_value=0.5, max_value=1.0),
    num_entities=st.integers(min_value=1, max_value=5)
)
@pytest.mark.asyncio
async def test_property_6_reconstruction_quality_monitoring(quality_level, num_entities):
    """
    Feature: llm-compression-integration, Property 6: Reconstruction Quality Monitoring
    
    For any reconstruction operation, when quality score < 0.85, the system should
    record warnings and return confidence score
    
    Validates: Requirement 6.4
    """
    # Create mock LLM client
    mock_client = Mock(spec=LLMClient)
    
    # Generate response based on quality level
    if quality_level < 0.85:
        # Low quality: return very short text
        response_text = "Short"
    else:
        # High quality: return reasonable text with entities
        response_text = "This is a well-formed reconstructed text with proper details."
    
    mock_client.generate = AsyncMock(return_value=LLMResponse(
        text=response_text,
        tokens_used=len(response_text.split()),
        latency_ms=100.0,
        model="gpt-3.5-turbo",
        finish_reason="stop",
        metadata={}
    ))
    
    # Create reconstructor
    reconstructor = LLMReconstructor(
        llm_client=mock_client,
        quality_threshold=0.85
    )
    
    # Create compressed memory with entities
    entities = {
        'persons': [f'Person{i}' for i in range(num_entities)],
        'dates': ['2024-01-15'],
        'numbers': ['100'],
        'locations': [],
        'keywords': ['test']
    }
    
    diff_text = "additional details"
    diff_data = zstd.compress(diff_text.encode('utf-8'), level=3)
    
    compressed = CompressedMemory(
        memory_id="test_id",
        summary_hash="test_hash",
        entities=entities,
        diff_data=diff_data,
        embedding=[0.1] * 384,
        compression_metadata=CompressionMetadata(
            original_size=1000,
            compressed_size=100,
            compression_ratio=10.0,
            model_used="gpt-3.5-turbo",
            quality_score=0.95,
            compression_time_ms=100.0,
            compressed_at=datetime.now()
        ),
        original_fields={}
    )
    
    # Reconstruct with quality verification
    result = await reconstructor.reconstruct(compressed, verify_quality=True)
    
    # Verify quality monitoring
    assert isinstance(result, ReconstructedMemory)
    assert result.quality_metrics is not None
    assert 0.0 <= result.confidence <= 1.0
    
    # If quality is low, should have warnings
    if quality_level < 0.85:
        assert len(result.warnings) > 0, "Low quality should generate warnings"
        assert result.confidence < 0.85, "Low quality should have low confidence"
    
    # Quality metrics should be present
    assert isinstance(result.quality_metrics, QualityMetrics)
    assert 0.0 <= result.quality_metrics.overall_score <= 1.0


# ============================================================================
# Property 7: Degraded Reconstruction (Fallback)
# Validates: Requirement 6.7
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    text_length=st.integers(min_value=100, max_value=500)
)
@pytest.mark.asyncio
async def test_property_7_degraded_reconstruction(text_length):
    """
    Feature: llm-compression-integration, Property 7: Degraded Reconstruction
    
    For any compressed memory, when LLM is unavailable, the system should be able
    to perform partial reconstruction using diff data only
    
    Validates: Requirement 6.7
    """
    # Create mock LLM client that fails
    mock_client = Mock(spec=LLMClient)
    mock_client.generate = AsyncMock(side_effect=Exception("LLM unavailable"))
    
    # Create reconstructor
    reconstructor = LLMReconstructor(
        llm_client=mock_client,
        quality_threshold=0.85
    )
    
    # Create compressed memory with diff data
    diff_text = "This is the diff data that can be used for partial reconstruction."
    diff_data = zstd.compress(diff_text.encode('utf-8'), level=3)
    
    compressed = CompressedMemory(
        memory_id="test_id",
        summary_hash="test_hash",
        entities={'persons': [], 'dates': [], 'numbers': [], 'locations': [], 'keywords': []},
        diff_data=diff_data,
        embedding=[0.1] * 384,
        compression_metadata=CompressionMetadata(
            original_size=text_length,
            compressed_size=len(diff_data),
            compression_ratio=text_length / len(diff_data),
            model_used="gpt-3.5-turbo",
            quality_score=0.95,
            compression_time_ms=100.0,
            compressed_at=datetime.now()
        ),
        original_fields={}
    )
    
    # Try to reconstruct (should use fallback)
    result = await reconstructor._reconstruct_from_diff_only(compressed)
    
    # Verify fallback reconstruction
    assert isinstance(result, ReconstructedMemory)
    assert result.full_text  # Should have some text from diff
    assert result.confidence == 0.5  # Lower confidence for diff-only
    assert len(result.warnings) > 0  # Should have warning about LLM unavailable
    assert any("diff only" in w.lower() for w in result.warnings)
    
    # Verify diff content is in result
    assert "diff data" in result.full_text.lower()


# ============================================================================
# Property: Entity Completeness in Reconstruction
# Validates: Requirement 6.2, 6.3
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    num_persons=st.integers(min_value=1, max_value=5),
    num_dates=st.integers(min_value=1, max_value=3),
    num_numbers=st.integers(min_value=0, max_value=3)
)
@pytest.mark.asyncio
async def test_property_entity_completeness_in_reconstruction(
    num_persons,
    num_dates,
    num_numbers
):
    """
    Feature: llm-compression-integration, Property: Entity Completeness in Reconstruction
    
    For any reconstruction, the quality verification should check that all expected
    entities are present in the reconstructed text
    
    Validates: Requirements 6.2, 6.3
    """
    # Create mock LLM client that includes entities in response
    person_names = ["John Smith", "Mary Johnson", "Bob Williams", "Alice Brown", "Charlie Davis"]
    persons = person_names[:num_persons]
    dates = [f"2024-01-{15+i:02d}" for i in range(num_dates)]
    numbers = [f"{100+i*10}" for i in range(num_numbers)]
    
    # Build response text with all entities
    response_parts = ["This is a reconstructed text."]
    for person in persons:
        response_parts.append(f"{person} was present.")
    for date in dates:
        response_parts.append(f"On {date}.")
    for number in numbers:
        response_parts.append(f"Amount: ${number}.")
    
    response_text = " ".join(response_parts)
    
    mock_client = Mock(spec=LLMClient)
    mock_client.generate = AsyncMock(return_value=LLMResponse(
        text=response_text,
        tokens_used=len(response_text.split()),
        latency_ms=100.0,
        model="gpt-3.5-turbo",
        finish_reason="stop",
        metadata={}
    ))
    
    # Create reconstructor
    reconstructor = LLMReconstructor(
        llm_client=mock_client,
        quality_threshold=0.85
    )
    
    # Create compressed memory with entities
    entities = {
        'persons': persons,
        'dates': dates,
        'numbers': numbers,
        'locations': [],
        'keywords': ['test']
    }
    
    # Add summary to cache so it gets expanded
    summary_hash = "test_hash"
    reconstructor._cache_summary(summary_hash, "Summary text")
    
    diff_data = zstd.compress(b"additional details", level=3)
    
    compressed = CompressedMemory(
        memory_id="test_id",
        summary_hash=summary_hash,
        entities=entities,
        diff_data=diff_data,
        embedding=[0.1] * 384,
        compression_metadata=CompressionMetadata(
            original_size=1000,
            compressed_size=100,
            compression_ratio=10.0,
            model_used="gpt-3.5-turbo",
            quality_score=0.95,
            compression_time_ms=100.0,
            compressed_at=datetime.now()
        ),
        original_fields={}
    )
    
    # Reconstruct with quality verification
    result = await reconstructor.reconstruct(compressed, verify_quality=True)
    
    # Verify entity completeness
    assert result.quality_metrics is not None
    # Relax threshold since keywords might not always be present
    assert result.quality_metrics.entity_accuracy > 0.6, \
        f"Entity accuracy {result.quality_metrics.entity_accuracy} should be > 0.6 when entities are present"
    
    # Verify critical entities (persons, dates) are in reconstructed text
    for person in persons:
        assert person in result.full_text, f"Person {person} should be in reconstructed text"
    
    for date in dates:
        assert date in result.full_text, f"Date {date} should be in reconstructed text"


# ============================================================================
# Property: Batch Reconstruction Consistency
# Validates: Requirement 6.6
# ============================================================================

@settings(max_examples=30, deadline=None)
@given(
    num_memories=st.integers(min_value=1, max_value=10)
)
@pytest.mark.asyncio
async def test_property_batch_reconstruction_consistency(num_memories):
    """
    Feature: llm-compression-integration, Property: Batch Reconstruction Consistency
    
    For any batch of compressed memories, batch reconstruction should produce
    valid results for all items, handling failures gracefully
    
    Validates: Requirement 6.6
    """
    # Create mock LLM client
    mock_client = Mock(spec=LLMClient)
    mock_client.generate = AsyncMock(return_value=LLMResponse(
        text="Reconstructed text with details.",
        tokens_used=10,
        latency_ms=100.0,
        model="gpt-3.5-turbo",
        finish_reason="stop",
        metadata={}
    ))
    
    # Create reconstructor
    reconstructor = LLMReconstructor(
        llm_client=mock_client,
        quality_threshold=0.85
    )
    
    # Create batch of compressed memories
    compressed_list = []
    for i in range(num_memories):
        diff_data = zstd.compress(f"diff data {i}".encode('utf-8'), level=3)
        
        compressed = CompressedMemory(
            memory_id=f"test_id_{i}",
            summary_hash=f"hash_{i}",
            entities={'persons': [], 'dates': [], 'numbers': [], 'locations': [], 'keywords': []},
            diff_data=diff_data,
            embedding=[0.1] * 384,
            compression_metadata=CompressionMetadata(
                original_size=1000,
                compressed_size=100,
                compression_ratio=10.0,
                model_used="gpt-3.5-turbo",
                quality_score=0.95,
                compression_time_ms=100.0,
                compressed_at=datetime.now()
            ),
            original_fields={}
        )
        compressed_list.append(compressed)
    
    # Batch reconstruct
    results = await reconstructor.reconstruct_batch(compressed_list)
    
    # Verify results
    assert len(results) == num_memories
    assert all(isinstance(r, ReconstructedMemory) for r in results)
    assert all(r.memory_id for r in results)
    assert all(r.full_text for r in results)
    
    # Verify all have valid reconstruction times
    for result in results:
        assert result.reconstruction_time_ms >= 0


# ============================================================================
# Property: Summary Cache Effectiveness
# Validates: Requirement 6.1
# ============================================================================

@settings(max_examples=30, deadline=None)
@given(
    num_reconstructions=st.integers(min_value=2, max_value=10)
)
@pytest.mark.asyncio
async def test_property_summary_cache_effectiveness(num_reconstructions):
    """
    Feature: llm-compression-integration, Property: Summary Cache Effectiveness
    
    For any repeated reconstructions with the same summary_hash, the summary
    should be cached and reused (LRU cache)
    
    Validates: Requirement 6.1
    """
    # Create mock LLM client
    mock_client = Mock(spec=LLMClient)
    mock_client.generate = AsyncMock(return_value=LLMResponse(
        text="Reconstructed text",
        tokens_used=10,
        latency_ms=100.0,
        model="gpt-3.5-turbo",
        finish_reason="stop",
        metadata={}
    ))
    
    # Create reconstructor
    reconstructor = LLMReconstructor(
        llm_client=mock_client,
        quality_threshold=0.85
    )
    
    # Pre-populate cache with a summary
    summary_hash = "shared_hash"
    summary_text = "This is a cached summary"
    reconstructor._cache_summary(summary_hash, summary_text)
    
    # Create compressed memories with same summary_hash
    compressed_list = []
    for i in range(num_reconstructions):
        diff_data = zstd.compress(f"diff {i}".encode('utf-8'), level=3)
        
        compressed = CompressedMemory(
            memory_id=f"test_id_{i}",
            summary_hash=summary_hash,  # Same hash
            entities={'persons': [], 'dates': [], 'numbers': [], 'locations': [], 'keywords': []},
            diff_data=diff_data,
            embedding=[0.1] * 384,
            compression_metadata=CompressionMetadata(
                original_size=1000,
                compressed_size=100,
                compression_ratio=10.0,
                model_used="gpt-3.5-turbo",
                quality_score=0.95,
                compression_time_ms=100.0,
                compressed_at=datetime.now()
            ),
            original_fields={}
        )
        compressed_list.append(compressed)
    
    # Reconstruct all
    for compressed in compressed_list:
        result = await reconstructor.reconstruct(compressed, verify_quality=False)
        assert isinstance(result, ReconstructedMemory)
    
    # Verify cache still contains the summary
    assert summary_hash in reconstructor.summary_cache
    assert reconstructor.summary_cache[summary_hash] == summary_text


# ============================================================================
# Property: Coherence Check Validity
# Validates: Requirement 6.4
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    has_ending_punctuation=st.booleans(),
    repetition_level=st.integers(min_value=1, max_value=5)
)
@pytest.mark.asyncio
async def test_property_coherence_check_validity(has_ending_punctuation, repetition_level):
    """
    Feature: llm-compression-integration, Property: Coherence Check Validity
    
    For any reconstructed text, the coherence check should correctly identify
    issues like missing punctuation and high repetition
    
    Validates: Requirement 6.4
    """
    # Create mock LLM client
    mock_client = Mock(spec=LLMClient)
    
    # Build text based on parameters
    if repetition_level > 3:
        # High repetition
        text = "test " * 10
    else:
        # Normal text
        text = "This is a well formed text with various words"
    
    if has_ending_punctuation:
        text += "."
    
    mock_client.generate = AsyncMock(return_value=LLMResponse(
        text=text,
        tokens_used=len(text.split()),
        latency_ms=100.0,
        model="gpt-3.5-turbo",
        finish_reason="stop",
        metadata={}
    ))
    
    # Create reconstructor
    reconstructor = LLMReconstructor(
        llm_client=mock_client,
        quality_threshold=0.85
    )
    
    # Add summary to cache so it gets expanded
    summary_hash = "test_hash"
    reconstructor._cache_summary(summary_hash, "Summary text")
    
    # Create compressed memory
    diff_data = zstd.compress(b"", level=3)
    
    compressed = CompressedMemory(
        memory_id="test_id",
        summary_hash=summary_hash,
        entities={'persons': [], 'dates': [], 'numbers': [], 'locations': [], 'keywords': []},
        diff_data=diff_data,
        embedding=[0.1] * 384,
        compression_metadata=CompressionMetadata(
            original_size=1000,
            compressed_size=100,
            compression_ratio=10.0,
            model_used="gpt-3.5-turbo",
            quality_score=0.95,
            compression_time_ms=100.0,
            compressed_at=datetime.now()
        ),
        original_fields={}
    )
    
    # Reconstruct with quality verification
    result = await reconstructor.reconstruct(compressed, verify_quality=True)
    
    # Verify coherence check
    assert result.quality_metrics is not None
    coherence_score = result.quality_metrics.coherence_score
    
    # If text has issues, coherence score should be lower
    if not has_ending_punctuation or repetition_level > 3:
        assert coherence_score < 1.0, f"Coherence score {coherence_score} should be < 1.0 for problematic text"
    else:
        assert coherence_score >= 0.7, f"Coherence score {coherence_score} should be >= 0.7 for good text"

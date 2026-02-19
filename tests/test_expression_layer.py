"""
Unit tests for MultiModalExpressor (Task 35)
"""

import pytest
import numpy as np
from llm_compression.expression_layer import MultiModalExpressor, ExpressionResult
from llm_compression.memory_primitive import MemoryPrimitive
from llm_compression.compressor import CompressedMemory, CompressionMetadata
from llm_compression.reconstructor import LLMReconstructor, ReconstructedMemory
from llm_compression.llm_client import LLMClient, LLMResponse
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = MagicMock(spec=LLMClient)
    client.generate = AsyncMock(return_value=LLMResponse(
        text="This is a generated response based on the memories.",
        model="test_model",
        tokens_used=50,
        latency_ms=100.0,
        finish_reason="stop"
    ))
    return client


@pytest.fixture
def mock_reconstructor():
    """Create mock reconstructor."""
    reconstructor = MagicMock(spec=LLMReconstructor)
    reconstructor.reconstruct = AsyncMock(return_value=ReconstructedMemory(
        memory_id="test_mem",
        full_text="Reconstructed memory content about quantum computing.",
        quality_metrics=None,
        reconstruction_time_ms=100.0,
        confidence=0.95,
        warnings=[],
        original_fields={}
    ))
    return reconstructor


@pytest.fixture
def expressor(mock_llm_client, mock_reconstructor):
    """Create MultiModalExpressor instance."""
    return MultiModalExpressor(
        llm_client=mock_llm_client,
        reconstructor=mock_reconstructor
    )


@pytest.fixture
def sample_memory():
    """Create a sample memory."""
    metadata = CompressionMetadata(
        original_size=100,
        compressed_size=10,
        compression_ratio=10.0,
        model_used="test",
        quality_score=0.9,
        compression_time_ms=100.0,
        compressed_at=datetime.now()
    )
    
    compressed = CompressedMemory(
        memory_id="mem_001",
        summary_hash="hash_001",
        entities={},
        diff_data=b"test_data",
        embedding=[0.1] * 384,
        compression_metadata=metadata
    )
    
    return MemoryPrimitive(
        id="mem_001",
        content=compressed,
        embedding=np.random.rand(384)
    )


class TestMultiModalExpressorCreation:
    """Test MultiModalExpressor initialization."""
    
    def test_create_expressor(self, expressor):
        """Test basic creation."""
        assert expressor.llm_client is not None
        assert expressor.reconstructor is not None


class TestTextCombination:
    """Test text combination logic."""
    
    def test_combine_empty_texts(self, expressor):
        """Test combining empty list."""
        result = expressor._combine_texts([])
        assert result == ""
    
    def test_combine_single_text(self, expressor):
        """Test combining single text."""
        result = expressor._combine_texts(["Hello world"])
        assert result == "Hello world"
    
    def test_combine_multiple_texts(self, expressor):
        """Test combining multiple texts."""
        texts = ["First memory", "Second memory", "Third memory"]
        result = expressor._combine_texts(texts)
        
        assert "[Memory 1]: First memory" in result
        assert "[Memory 2]: Second memory" in result
        assert "[Memory 3]: Third memory" in result


class TestQualityEstimation:
    """Test quality estimation."""
    
    def test_empty_generated_text(self, expressor):
        """Test quality of empty text."""
        quality = expressor._estimate_quality("", "context")
        assert quality == 0.0
    
    def test_very_short_text(self, expressor):
        """Test quality of very short text."""
        quality = expressor._estimate_quality("Hi", "context")
        assert 0.0 < quality < 0.5
    
    def test_good_length_text(self, expressor):
        """Test quality of good length text."""
        generated = "This is a good response with reasonable length."
        context = "This is context about response and length."
        quality = expressor._estimate_quality(generated, context)
        assert quality > 0.5
    
    def test_keyword_overlap(self, expressor):
        """Test quality with keyword overlap."""
        generated = "quantum computing uses qubits for computation"
        context = "quantum computing is revolutionary technology using qubits"
        quality = expressor._estimate_quality(generated, context)
        assert quality > 0.5  # Good overlap


@pytest.mark.asyncio
class TestTextGeneration:
    """Test text generation."""
    
    async def test_generate_text_default(self, expressor):
        """Test basic text generation."""
        result = await expressor._generate_text(
            context="Test context",
            style=None,
            max_length=100
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        expressor.llm_client.generate.assert_called_once()
    
    async def test_generate_text_concise_style(self, expressor):
        """Test text generation with concise style."""
        result = await expressor._generate_text(
            context="Test context",
            style="concise",
            max_length=100
        )
        
        assert isinstance(result, str)
        # Check that prompt includes style instruction
        call_args = expressor.llm_client.generate.call_args
        assert "concise" in call_args.kwargs["prompt"].lower()
    
    async def test_generate_text_detailed_style(self, expressor):
        """Test text generation with detailed style."""
        result = await expressor._generate_text(
            context="Test context",
            style="detailed",
            max_length=100
        )
        
        assert isinstance(result, str)
        call_args = expressor.llm_client.generate.call_args
        assert "detailed" in call_args.kwargs["prompt"].lower()


@pytest.mark.asyncio
class TestExpression:
    """Test full expression pipeline."""
    
    async def test_express_single_memory(self, expressor, sample_memory):
        """Test expressing single memory."""
        result = await expressor.express([sample_memory], modality="text")
        
        assert isinstance(result, ExpressionResult)
        assert result.modality == "text"
        assert len(result.content) > 0
        assert result.quality_score >= 0.0
        assert result.quality_score <= 1.0
        assert sample_memory.id in result.source_memories
    
    async def test_express_multiple_memories(self, expressor, sample_memory):
        """Test expressing multiple memories."""
        # Create second memory
        metadata = CompressionMetadata(
            original_size=100,
            compressed_size=10,
            compression_ratio=10.0,
            model_used="test",
            quality_score=0.9,
            compression_time_ms=100.0,
            compressed_at=datetime.now()
        )
        
        compressed = CompressedMemory(
            memory_id="mem_002",
            summary_hash="hash_002",
            entities={},
            diff_data=b"test_data_2",
            embedding=[0.2] * 384,
            compression_metadata=metadata
        )
        
        memory2 = MemoryPrimitive(
            id="mem_002",
            content=compressed,
            embedding=np.random.rand(384)
        )
        
        result = await expressor.express([sample_memory, memory2], modality="text")
        
        assert isinstance(result, ExpressionResult)
        assert len(result.source_memories) == 2
        assert sample_memory.id in result.source_memories
        assert memory2.id in result.source_memories
    
    async def test_express_with_style(self, expressor, sample_memory):
        """Test expressing with style."""
        result = await expressor.express(
            [sample_memory],
            modality="text",
            style="concise"
        )
        
        assert isinstance(result, ExpressionResult)
        assert result.content is not None
    
    async def test_express_unsupported_modality(self, expressor, sample_memory):
        """Test expressing with unsupported modality."""
        with pytest.raises(ValueError, match="Unsupported modality"):
            await expressor.express([sample_memory], modality="video")
    
    async def test_express_image_not_implemented(self, expressor, sample_memory):
        """Test image generation raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Image generation"):
            await expressor.express([sample_memory], modality="image")
    
    async def test_express_audio_not_implemented(self, expressor, sample_memory):
        """Test audio generation raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Audio generation"):
            await expressor.express([sample_memory], modality="audio")


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests."""
    
    async def test_realistic_expression_flow(self, expressor, sample_memory):
        """Test realistic expression flow."""
        # Activate memory (simulate usage)
        sample_memory.activate(0.8)
        
        # Express
        result = await expressor.express(
            [sample_memory],
            modality="text",
            style="detailed",
            max_length=200
        )
        
        # Verify result
        assert result.content is not None
        assert result.quality_score > 0.0
        assert len(result.source_memories) == 1
        
        # Verify reconstructor was called
        expressor.reconstructor.reconstruct.assert_called_once()
        
        # Verify LLM was called
        expressor.llm_client.generate.assert_called_once()

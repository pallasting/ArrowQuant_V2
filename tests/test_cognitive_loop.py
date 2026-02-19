"""
Unit tests for CognitiveLoop (Task 42)
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from llm_compression.cognitive_loop import CognitiveLoop, CognitiveResult
from llm_compression.memory_primitive import MemoryPrimitive
from llm_compression.compressor import CompressedMemory, CompressionMetadata
from llm_compression.expression_layer import ExpressionResult
from llm_compression.internal_feedback import QualityScore, Correction, CorrectionType


@pytest.fixture
def cognitive_loop():
    """Create CognitiveLoop instance."""
    # Create mock components
    expressor = MagicMock()
    feedback = MagicMock()
    
    return CognitiveLoop(
        expressor=expressor,
        feedback=feedback,
        quality_threshold=0.85,
        max_corrections=2,
        learning_rate=0.1
    )


@pytest.fixture
def test_memories():
    """Create test memory network."""
    metadata = CompressionMetadata(
        original_size=100,
        compressed_size=10,
        compression_ratio=10.0,
        model_used="test",
        quality_score=0.9,
        compression_time_ms=100.0,
        compressed_at=datetime.now()
    )
    
    memories = {}
    
    # Memory A
    compressed_a = CompressedMemory(
        memory_id="mem_a",
        summary_hash="hash_a",
        entities={},
        diff_data=b"Python is a programming language",
        embedding=[0.1] * 384,
        compression_metadata=metadata
    )
    mem_a = MemoryPrimitive(
        id="mem_a",
        content=compressed_a,
        embedding=np.array([1.0, 0.0, 0.0])
    )
    memories["mem_a"] = mem_a
    
    # Memory B
    compressed_b = CompressedMemory(
        memory_id="mem_b",
        summary_hash="hash_b",
        entities={},
        diff_data=b"Python is used for AI and web development",
        embedding=[0.1] * 384,
        compression_metadata=metadata
    )
    mem_b = MemoryPrimitive(
        id="mem_b",
        content=compressed_b,
        embedding=np.array([0.9, 0.1, 0.0])
    )
    memories["mem_b"] = mem_b
    
    # Memory C
    compressed_c = CompressedMemory(
        memory_id="mem_c",
        summary_hash="hash_c",
        entities={},
        diff_data=b"Machine learning is a subset of AI",
        embedding=[0.1] * 384,
        compression_metadata=metadata
    )
    mem_c = MemoryPrimitive(
        id="mem_c",
        content=compressed_c,
        embedding=np.array([0.8, 0.2, 0.0])
    )
    memories["mem_c"] = mem_c
    
    # Add connections
    mem_a.add_connection("mem_b", 0.8)
    mem_b.add_connection("mem_a", 0.8)
    mem_b.add_connection("mem_c", 0.6)
    mem_c.add_connection("mem_b", 0.6)
    
    return memories


class TestCognitiveLoopCreation:
    """Test CognitiveLoop initialization."""
    
    def test_create_loop(self, cognitive_loop):
        """Test basic creation."""
        assert cognitive_loop.quality_threshold == 0.85
        assert cognitive_loop.max_corrections == 2
        assert cognitive_loop.learning_rate == 0.1
        assert len(cognitive_loop.memory_network) == 0


class TestMemoryManagement:
    """Test memory management."""
    
    def test_add_memory(self, cognitive_loop, test_memories):
        """Test adding memory."""
        mem = test_memories["mem_a"]
        cognitive_loop.add_memory(mem)
        
        assert len(cognitive_loop.memory_network) == 1
        assert "mem_a" in cognitive_loop.memory_network
    
    def test_get_memory(self, cognitive_loop, test_memories):
        """Test getting memory."""
        mem = test_memories["mem_a"]
        cognitive_loop.add_memory(mem)
        
        retrieved = cognitive_loop.get_memory("mem_a")
        assert retrieved is not None
        assert retrieved.id == "mem_a"
    
    def test_get_nonexistent_memory(self, cognitive_loop):
        """Test getting nonexistent memory."""
        retrieved = cognitive_loop.get_memory("nonexistent")
        assert retrieved is None


class TestNetworkStats:
    """Test network statistics."""
    
    def test_empty_network_stats(self, cognitive_loop):
        """Test stats for empty network."""
        stats = cognitive_loop.get_network_stats()
        
        assert stats["total_memories"] == 0
        assert stats["total_connections"] == 0
        assert stats["avg_connections"] == 0.0
        assert stats["avg_success_rate"] == 0.0
    
    def test_network_stats(self, cognitive_loop, test_memories):
        """Test stats for populated network."""
        for mem in test_memories.values():
            cognitive_loop.add_memory(mem)
        
        stats = cognitive_loop.get_network_stats()
        
        assert stats["total_memories"] == 3
        assert stats["total_connections"] > 0
        assert stats["avg_connections"] > 0.0


class TestLearning:
    """Test learning mechanisms."""
    
    def test_learn_from_interaction(self, cognitive_loop, test_memories):
        """Test learning from interaction."""
        mem_a = test_memories["mem_a"]
        mem_b = test_memories["mem_b"]
        
        quality = QualityScore(
            overall=0.9,
            consistency=0.9,
            completeness=0.9,
            accuracy=0.9,
            coherence=0.9
        )
        
        learning_occurred = cognitive_loop._learn_from_interaction(
            [mem_a, mem_b],
            quality
        )
        
        assert learning_occurred is True
        # Success should be recorded
        assert mem_a.access_count > 0
        assert mem_b.access_count > 0
    
    def test_learn_records_success(self, cognitive_loop, test_memories):
        """Test success recording."""
        mem_a = test_memories["mem_a"]
        
        # High quality
        quality_high = QualityScore(
            overall=0.95,
            consistency=0.95,
            completeness=0.95,
            accuracy=0.95,
            coherence=0.95
        )
        
        cognitive_loop._learn_from_interaction([mem_a], quality_high)
        
        assert mem_a.success_count > 0
    
    def test_learn_records_failure(self, cognitive_loop, test_memories):
        """Test failure recording."""
        mem_a = test_memories["mem_a"]
        
        # Low quality
        quality_low = QualityScore(
            overall=0.5,
            consistency=0.5,
            completeness=0.5,
            accuracy=0.5,
            coherence=0.5
        )
        
        cognitive_loop._learn_from_interaction([mem_a], quality_low)
        
        assert mem_a.access_count > 0
        assert mem_a.success_count == 0


@pytest.mark.asyncio
class TestCognitiveProcess:
    """Test full cognitive process."""
    
    async def test_process_basic(self, cognitive_loop, test_memories):
        """Test basic cognitive process."""
        # Add memories
        for mem in test_memories.values():
            cognitive_loop.add_memory(mem)
        
        # Mock components
        with patch.object(cognitive_loop.expressor, 'express_text', new_callable=AsyncMock) as mock_express, \
             patch.object(cognitive_loop.feedback, 'evaluate', new_callable=AsyncMock) as mock_evaluate:
            
            mock_express.return_value = ExpressionResult(
                content="Python is a programming language used for AI.",
                quality_score=0.9,
                modality="text", source_memories=[]
            )
            
            mock_evaluate.return_value = QualityScore(
                overall=0.9,
                consistency=0.9,
                completeness=0.9,
                accuracy=0.9,
                coherence=0.9
            )
            
            query = "What is Python?"
            query_embedding = np.array([1.0, 0.0, 0.0])
            
            result = await cognitive_loop.process(query, query_embedding)
            
            assert isinstance(result, CognitiveResult)
            assert result.output is not None
            assert result.quality.overall >= 0.85
            assert len(result.memories_used) > 0
            assert result.learning_occurred is True
    
    async def test_process_with_correction(self, cognitive_loop, test_memories):
        """Test process with correction loop."""
        for mem in test_memories.values():
            cognitive_loop.add_memory(mem)
        
        with patch.object(cognitive_loop.expressor, 'express_text', new_callable=AsyncMock) as mock_express, \
             patch.object(cognitive_loop.feedback, 'evaluate', new_callable=AsyncMock) as mock_evaluate, \
             patch.object(cognitive_loop.feedback, 'suggest_correction') as mock_suggest:
            
            # First attempt: low quality
            mock_express.side_effect = [
                ExpressionResult(content="Low quality output", quality_score=0.5, modality="text", source_memories=[]),
                ExpressionResult(content="Improved output", quality_score=0.9, modality="text", source_memories=[])
            ]
            
            # First eval: low, second eval: high
            mock_evaluate.side_effect = [
                QualityScore(overall=0.5, consistency=0.5, completeness=0.5, accuracy=0.5, coherence=0.5),
                QualityScore(overall=0.9, consistency=0.9, completeness=0.9, accuracy=0.9, coherence=0.9)
            ]
            
            mock_suggest.return_value = Correction(
                type=CorrectionType.SUPPLEMENT,
                action="retrieve_more_memories",
                reason="Low completeness"
            )
            
            query = "What is Python?"
            query_embedding = np.array([1.0, 0.0, 0.0])
            
            result = await cognitive_loop.process(query, query_embedding)
            
            assert result.corrections_applied > 0
            assert result.quality.overall >= 0.85
    
    async def test_process_max_corrections(self, cognitive_loop, test_memories):
        """Test max corrections limit."""
        for mem in test_memories.values():
            cognitive_loop.add_memory(mem)
        
        with patch.object(cognitive_loop.expressor, 'express_text', new_callable=AsyncMock) as mock_express, \
             patch.object(cognitive_loop.feedback, 'evaluate', new_callable=AsyncMock) as mock_evaluate, \
             patch.object(cognitive_loop.feedback, 'suggest_correction') as mock_suggest:
            
            # Always low quality
            mock_express.return_value = ExpressionResult(
                content="Low quality", quality_score=0.5, modality="text", source_memories=[]
            )
            
            mock_evaluate.return_value = QualityScore(
                overall=0.5, consistency=0.5, completeness=0.5, accuracy=0.5, coherence=0.5
            )
            
            mock_suggest.return_value = Correction(
                type=CorrectionType.RECTIFY,
                action="requery",
                reason="Low accuracy"
            )
            
            query = "What is Python?"
            query_embedding = np.array([1.0, 0.0, 0.0])
            
            result = await cognitive_loop.process(query, query_embedding)
            
            # Should stop at max_corrections
            assert result.corrections_applied == cognitive_loop.max_corrections
    
    async def test_process_empty_network(self, cognitive_loop):
        """Test process with empty network."""
        with patch.object(cognitive_loop.expressor, 'express_text', new_callable=AsyncMock) as mock_express, \
             patch.object(cognitive_loop.feedback, 'evaluate', new_callable=AsyncMock) as mock_evaluate:
            
            mock_express.return_value = ExpressionResult(
                content="No relevant memories found.",
                quality_score=0.0,
                modality="text", source_memories=[]
            )
            
            mock_evaluate.return_value = QualityScore(
                overall=0.0, consistency=0.0, completeness=0.0, accuracy=0.0, coherence=0.0
            )
            
            query = "What is Python?"
            query_embedding = np.array([1.0, 0.0, 0.0])
            
            result = await cognitive_loop.process(query, query_embedding)
            
            assert result.output == "No relevant memories found."
            assert len(result.memories_used) == 0


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_cycle(self, cognitive_loop, test_memories):
        """Test full cognitive cycle."""
        # Build network
        for mem in test_memories.values():
            cognitive_loop.add_memory(mem)
        
        initial_stats = cognitive_loop.get_network_stats()
        
        # Mock components
        with patch.object(cognitive_loop.expressor, 'express_text', new_callable=AsyncMock) as mock_express, \
             patch.object(cognitive_loop.feedback, 'evaluate', new_callable=AsyncMock) as mock_evaluate:
            
            mock_express.return_value = ExpressionResult(
                content="Python is great for AI", quality_score=0.9, modality="text", source_memories=[]
            )
            
            mock_evaluate.return_value = QualityScore(
                overall=0.9, consistency=0.9, completeness=0.9, accuracy=0.9, coherence=0.9
            )
            
            # Process query
            result = await cognitive_loop.process(
                "What is Python?",
                np.array([1.0, 0.0, 0.0])
            )
            
            # Verify learning occurred
            assert result.learning_occurred is True
            
            # Verify network evolved
            final_stats = cognitive_loop.get_network_stats()
            assert final_stats["total_memories"] == initial_stats["total_memories"]

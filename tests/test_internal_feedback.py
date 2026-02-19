"""
Unit tests for InternalFeedbackSystem (Task 37)
"""

import pytest
import numpy as np
from llm_compression.internal_feedback import (
    InternalFeedbackSystem,
    QualityScore,
    Correction,
    CorrectionType
)
from llm_compression.memory_primitive import MemoryPrimitive
from llm_compression.compressor import CompressedMemory, CompressionMetadata
from datetime import datetime


@pytest.fixture
def feedback_system():
    """Create InternalFeedbackSystem instance."""
    return InternalFeedbackSystem(
        quality_threshold=0.7,
        completeness_threshold=0.7,
        accuracy_threshold=0.7,
        coherence_threshold=0.7
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


class TestInternalFeedbackCreation:
    """Test InternalFeedbackSystem initialization."""
    
    def test_create_feedback_system(self, feedback_system):
        """Test basic creation."""
        assert feedback_system.quality_threshold == 0.7
        assert feedback_system.completeness_threshold == 0.7
        assert feedback_system.accuracy_threshold == 0.7
        assert feedback_system.coherence_threshold == 0.7
        assert feedback_system.evaluator is not None


class TestCompletenessCheck:
    """Test completeness checking."""
    
    def test_completeness_empty_output(self, feedback_system):
        """Test completeness with empty output."""
        score = feedback_system._check_completeness("", "expected content")
        assert score == 0.0
    
    def test_completeness_empty_expected(self, feedback_system):
        """Test completeness with empty expected."""
        score = feedback_system._check_completeness("output", "")
        assert score == 1.0
    
    def test_completeness_full_overlap(self, feedback_system):
        """Test completeness with full overlap."""
        expected = "quantum computing uses qubits"
        output = "quantum computing uses qubits"
        score = feedback_system._check_completeness(output, expected)
        assert score == 1.0
    
    def test_completeness_partial_overlap(self, feedback_system):
        """Test completeness with partial overlap."""
        expected = "quantum computing uses qubits for computation"
        output = "quantum computing is revolutionary"
        score = feedback_system._check_completeness(output, expected)
        assert 0.0 < score < 1.0


class TestCoherenceCheck:
    """Test coherence checking."""
    
    def test_coherence_empty_output(self, feedback_system):
        """Test coherence with empty output."""
        score = feedback_system._check_coherence("")
        assert score == 0.0
    
    def test_coherence_very_short(self, feedback_system):
        """Test coherence with very short output."""
        score = feedback_system._check_coherence("Hi")
        assert score < 0.5
    
    def test_coherence_reasonable(self, feedback_system):
        """Test coherence with reasonable output."""
        output = "This is a reasonable output with multiple words and good structure."
        score = feedback_system._check_coherence(output)
        assert score > 0.7


class TestCorrectionGeneration:
    """Test correction generation."""
    
    def test_no_correction_needed(self, feedback_system):
        """Test no correction when quality is good."""
        quality_score = QualityScore(
            overall=0.9,
            consistency=0.9,
            completeness=0.9,
            accuracy=0.9,
            coherence=0.9
        )
        
        correction = feedback_system.generate_correction(quality_score)
        assert correction is None
    
    def test_correction_for_incompleteness(self, feedback_system):
        """Test correction for incomplete output."""
        quality_score = QualityScore(
            overall=0.6,
            consistency=0.8,
            completeness=0.5,  # Low
            accuracy=0.8,
            coherence=0.8
        )
        
        correction = feedback_system.generate_correction(quality_score)
        assert correction is not None
        assert correction.type == CorrectionType.SUPPLEMENT
        assert "incomplete" in correction.reason
    
    def test_correction_for_inaccuracy(self, feedback_system):
        """Test correction for inaccurate output."""
        quality_score = QualityScore(
            overall=0.6,
            consistency=0.8,
            completeness=0.8,
            accuracy=0.5,  # Low
            coherence=0.8
        )
        
        correction = feedback_system.generate_correction(quality_score)
        assert correction is not None
        assert correction.type == CorrectionType.RECTIFY
        assert "inaccurate" in correction.reason
    
    def test_correction_for_incoherence(self, feedback_system):
        """Test correction for incoherent output."""
        quality_score = QualityScore(
            overall=0.6,
            consistency=0.8,
            completeness=0.8,
            accuracy=0.8,
            coherence=0.5  # Low
        )
        
        correction = feedback_system.generate_correction(quality_score)
        assert correction is not None
        assert correction.type == CorrectionType.RESTRUCTURE
        assert "incoherent" in correction.reason
    
    def test_correction_confidence(self, feedback_system):
        """Test correction confidence calculation."""
        quality_score = QualityScore(
            overall=0.6,
            consistency=0.8,
            completeness=0.3,  # Very low
            accuracy=0.8,
            coherence=0.8
        )
        
        correction = feedback_system.generate_correction(quality_score)
        assert correction is not None
        assert correction.confidence > 0.5  # High confidence due to low completeness


class TestShouldCorrect:
    """Test should_correct decision."""
    
    def test_should_correct_low_quality(self, feedback_system):
        """Test should correct with low quality."""
        quality_score = QualityScore(
            overall=0.5,
            consistency=0.5,
            completeness=0.5,
            accuracy=0.5,
            coherence=0.5
        )
        
        assert feedback_system.should_correct(quality_score) is True
    
    def test_should_not_correct_high_quality(self, feedback_system):
        """Test should not correct with high quality."""
        quality_score = QualityScore(
            overall=0.9,
            consistency=0.9,
            completeness=0.9,
            accuracy=0.9,
            coherence=0.9
        )
        
        assert feedback_system.should_correct(quality_score) is False
    
    def test_should_correct_threshold(self, feedback_system):
        """Test should correct at threshold."""
        quality_score = QualityScore(
            overall=0.7,  # Exactly at threshold
            consistency=0.7,
            completeness=0.7,
            accuracy=0.7,
            coherence=0.7
        )
        
        assert feedback_system.should_correct(quality_score) is False


class TestEvaluateOutput:
    """Test output evaluation."""
    
    def test_evaluate_output_basic(self, feedback_system, sample_memory):
        """Test basic output evaluation."""
        output = "This is a test output about quantum computing."
        query = "Tell me about quantum computing"
        
        quality_score = feedback_system.evaluate_output(
            output=output,
            original_query=query,
            used_memories=[sample_memory]
        )
        
        assert isinstance(quality_score, QualityScore)
        assert 0.0 <= quality_score.overall <= 1.0
        assert 0.0 <= quality_score.consistency <= 1.0
        assert 0.0 <= quality_score.completeness <= 1.0
        assert 0.0 <= quality_score.accuracy <= 1.0
        assert 0.0 <= quality_score.coherence <= 1.0
    
    def test_evaluate_empty_output(self, feedback_system, sample_memory):
        """Test evaluation of empty output."""
        quality_score = feedback_system.evaluate_output(
            output="",
            original_query="test query",
            used_memories=[sample_memory]
        )
        
        assert quality_score.completeness == 0.0
        assert quality_score.coherence == 0.0


class TestIntegration:
    """Integration tests."""
    
    def test_full_feedback_loop(self, feedback_system, sample_memory):
        """Test full feedback loop."""
        # Generate output
        output = "Short"
        query = "Tell me about quantum computing"
        
        # Evaluate
        quality_score = feedback_system.evaluate_output(
            output=output,
            original_query=query,
            used_memories=[sample_memory]
        )
        
        # Check if correction needed
        needs_correction = feedback_system.should_correct(quality_score)
        
        # Generate correction if needed
        if needs_correction:
            correction = feedback_system.generate_correction(quality_score)
            assert correction is not None
            assert correction.type in [
                CorrectionType.SUPPLEMENT,
                CorrectionType.RECTIFY,
                CorrectionType.RESTRUCTURE
            ]
    
    def test_configurable_thresholds(self):
        """Test configurable quality thresholds."""
        # Strict system
        strict_system = InternalFeedbackSystem(quality_threshold=0.9)
        
        quality_score = QualityScore(
            overall=0.8,
            consistency=0.8,
            completeness=0.8,
            accuracy=0.8,
            coherence=0.8
        )
        
        assert strict_system.should_correct(quality_score) is True
        
        # Lenient system
        lenient_system = InternalFeedbackSystem(quality_threshold=0.5)
        assert lenient_system.should_correct(quality_score) is False

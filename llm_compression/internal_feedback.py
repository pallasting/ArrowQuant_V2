"""
Internal Feedback System - Self-correction through quality evaluation

Implements internal feedback loop:
1. Evaluate output quality
2. Detect issues
3. Generate corrections
4. Apply self-correction
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from .memory_primitive import MemoryPrimitive
from .quality_evaluator import QualityEvaluator, QualityMetrics


class CorrectionType(Enum):
    """Type of correction needed."""
    SUPPLEMENT = "supplement"      # Add more information
    RECTIFY = "rectify"           # Fix inaccuracies
    RESTRUCTURE = "restructure"   # Improve coherence


@dataclass
class QualityScore:
    """Quality score breakdown."""
    overall: float          # Overall quality (0-1)
    consistency: float      # Semantic consistency
    completeness: float     # Information completeness
    accuracy: float         # Factual accuracy
    coherence: float        # Logical coherence


@dataclass
class Correction:
    """Correction suggestion."""
    type: CorrectionType
    reason: str
    action: str
    confidence: float = 1.0


class InternalFeedbackSystem:
    """
    Internal feedback system for self-correction.
    
    Evaluates output quality and generates corrections when needed.
    Implements closed-loop self-improvement.
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.7,
        completeness_threshold: float = 0.7,
        accuracy_threshold: float = 0.7,
        coherence_threshold: float = 0.7
    ):
        """
        Initialize internal feedback system.
        
        Args:
            quality_threshold: Minimum overall quality (0-1)
            completeness_threshold: Minimum completeness (0-1)
            accuracy_threshold: Minimum accuracy (0-1)
            coherence_threshold: Minimum coherence (0-1)
        """
        self.quality_threshold = quality_threshold
        self.completeness_threshold = completeness_threshold
        self.accuracy_threshold = accuracy_threshold
        self.coherence_threshold = coherence_threshold
        
        self.evaluator = QualityEvaluator()
    
    def evaluate_output(
        self,
        output: str,
        original_query: str,
        used_memories: List[MemoryPrimitive]
    ) -> QualityScore:
        """
        Evaluate output quality.
        
        Args:
            output: Generated output
            original_query: Original query/prompt
            used_memories: Memories used to generate output
            
        Returns:
            QualityScore with detailed breakdown
        """
        # Reconstruct expected content from memories
        expected = self._reconstruct_expected(used_memories)
        
        # Evaluate using Phase 1.1 quality evaluator
        quality_metrics = self.evaluator.evaluate(
            original=expected,
            reconstructed=output,
            compressed_size=sum(
                m.content.compression_metadata.compressed_size 
                for m in used_memories
            ),
            reconstruction_latency_ms=0.0
        )
        
        # Calculate detailed scores
        completeness = self._check_completeness(output, expected)
        coherence = self._check_coherence(output)
        
        return QualityScore(
            overall=quality_metrics.overall_score,
            consistency=quality_metrics.semantic_similarity,
            completeness=completeness,
            accuracy=quality_metrics.entity_accuracy,
            coherence=coherence
        )
    
    # Alias for compatibility
    async def evaluate(self, output: str, original_query: str, used_memories: List) -> QualityScore:
        """Async wrapper for evaluate_output (compatibility)"""
        # Convert CompressedMemory to MemoryPrimitive if needed
        from llm_compression.memory_primitive import MemoryPrimitive
        from llm_compression.compressor import CompressedMemory
        import numpy as np
        
        mock_memories = []
        for mem in used_memories:
            if isinstance(mem, MemoryPrimitive):
                mock_memories.append(mem)
            elif isinstance(mem, CompressedMemory):
                # Wrap CompressedMemory in MemoryPrimitive
                mock_memories.append(MemoryPrimitive(
                    id=mem.memory_id,
                    content=mem,
                    embedding=np.array(mem.embedding)
                ))
            elif isinstance(mem, str):
                # String - create mock CompressedMemory
                from llm_compression.compressor import CompressionMetadata
                compressed = CompressedMemory(
                    memory_id="mock",
                    summary_hash="mock",
                    entities={},
                    diff_data=mem[:100].encode('utf-8'),
                    embedding=[0.0] * 384,
                    compression_metadata=CompressionMetadata(
                        original_size=len(mem),
                        compressed_size=len(mem) // 2,
                        compression_ratio=2.0,
                        model_used="mock",
                        quality_score=1.0,
                        compression_time_ms=0.0,
                        compressed_at=__import__('datetime').datetime.now()
                    )
                )
                mock_memories.append(MemoryPrimitive(
                    id="mock",
                    content=compressed,
                    embedding=np.array([0.0] * 384)
                ))
        
        return self.evaluate_output(output, original_query, mock_memories)
    
    def generate_correction(
        self,
        quality_score: QualityScore
    ) -> Optional[Correction]:
        """
        Generate correction if quality is below threshold.
        
        Args:
            quality_score: Quality score from evaluation
            
        Returns:
            Correction suggestion or None if quality is acceptable
        """
        if quality_score.overall >= self.quality_threshold:
            return None  # Quality is acceptable
        
        # Identify primary issue
        if quality_score.completeness < self.completeness_threshold:
            return Correction(
                type=CorrectionType.SUPPLEMENT,
                reason="incomplete_output",
                action="retrieve_more_memories",
                confidence=1.0 - quality_score.completeness
            )
        
        if quality_score.accuracy < self.accuracy_threshold:
            return Correction(
                type=CorrectionType.RECTIFY,
                reason="inaccurate_output",
                action="requery_with_constraints",
                confidence=1.0 - quality_score.accuracy
            )
        
        if quality_score.coherence < self.coherence_threshold:
            return Correction(
                type=CorrectionType.RESTRUCTURE,
                reason="incoherent_output",
                action="regenerate_with_structure",
                confidence=1.0 - quality_score.coherence
            )
        
        # General quality issue
        return Correction(
            type=CorrectionType.RESTRUCTURE,
            reason="low_overall_quality",
            action="regenerate",
            confidence=1.0 - quality_score.overall
        )
    
    # Alias for compatibility
    def suggest_correction(self, quality_score: QualityScore) -> Optional[Correction]:
        """Alias for generate_correction"""
        return self.generate_correction(quality_score)
    
    def should_correct(self, quality_score: QualityScore) -> bool:
        """
        Check if correction is needed.
        
        Args:
            quality_score: Quality score from evaluation
            
        Returns:
            True if correction is needed
        """
        return quality_score.overall < self.quality_threshold
    
    def _reconstruct_expected(self, memories: List[MemoryPrimitive]) -> str:
        """
        Reconstruct expected content from memories.
        
        Args:
            memories: List of memories
            
        Returns:
            Combined expected content
        """
        if not memories:
            return ""
        
        # Simple combination - in real system would use reconstructor
        # For now, use memory IDs as proxy
        return " ".join(f"memory_{m.id}" for m in memories)
    
    def _check_completeness(self, output: str, expected: str) -> float:
        """
        Check completeness of output.
        
        Args:
            output: Generated output
            expected: Expected content
            
        Returns:
            Completeness score (0-1)
        """
        if not expected:
            return 1.0
        
        if not output:
            return 0.0
        
        # Simple word overlap
        expected_words = set(expected.lower().split())
        output_words = set(output.lower().split())
        
        if not expected_words:
            return 1.0
        
        overlap = len(expected_words & output_words)
        return min(1.0, overlap / len(expected_words))
    
    def _check_coherence(self, output: str) -> float:
        """
        Check coherence of output.
        
        Args:
            output: Generated output
            
        Returns:
            Coherence score (0-1)
        """
        if not output:
            return 0.0
        
        # Simple heuristics
        sentences = output.split('.')
        
        # Length check
        if len(output) < 10:
            return 0.3
        
        # Sentence count check
        if len(sentences) < 1:
            return 0.5
        
        # Word count check
        words = output.split()
        if len(words) < 5:
            return 0.4
        
        # Reasonable length and structure
        return 0.85

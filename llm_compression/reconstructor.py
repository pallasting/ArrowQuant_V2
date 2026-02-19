"""
LLM Memory Reconstructor

Reconstructs original memories from compressed data using LLM-based expansion
and diff application.

Feature: llm-compression-integration
Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

import asyncio
import logging
try:
    import zstandard as zstd
except ImportError:
    import zstd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from collections import OrderedDict

from llm_compression.llm_client import LLMClient, LLMResponse
from llm_compression.compressor import CompressedMemory
from llm_compression.errors import ReconstructionError


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for reconstruction"""
    entity_accuracy: float          # Entity completeness (0-1)
    coherence_score: float          # Text coherence (0-1)
    length_score: float             # Length reasonableness (0-1)
    overall_score: float            # Overall quality (0-1)
    warnings: List[str]             # Quality warnings


@dataclass
class ReconstructedMemory:
    """Reconstructed memory from compressed data"""
    memory_id: str                  # Memory ID
    full_text: str                  # Reconstructed full text
    quality_metrics: Optional[QualityMetrics]  # Quality metrics
    reconstruction_time_ms: float   # Reconstruction time
    confidence: float               # Confidence score (0-1)
    warnings: List[str]             # Warnings
    original_fields: Dict[str, Any] # Original fields


class LLMReconstructor:
    """
    LLM-based memory reconstructor
    
    Reconstructs original memories from compressed data using:
    1. Summary lookup (3-level cache)
    2. Summary expansion (LLM)
    3. Diff application (intelligent insertion)
    4. Quality verification (entity completeness, coherence, length)
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        quality_threshold: float = 0.85,
        max_cache_size: int = 10000,
        summary_cache: Optional[Dict[str, str]] = None
    ):
        """
        Initialize reconstructor
        
        Args:
            llm_client: LLM client for summary expansion
            quality_threshold: Quality threshold for warnings (default: 0.85)
            max_cache_size: Maximum summary cache size (default: 10000)
            summary_cache: Optional shared summary cache from compressor
        """
        self.llm_client = llm_client
        self.quality_threshold = quality_threshold
        self.max_cache_size = max_cache_size
        
        # Summary cache (LRU) - can be shared with compressor
        if summary_cache is not None:
            self.summary_cache = summary_cache
        else:
            self.summary_cache: OrderedDict[str, str] = OrderedDict()
        
        logger.info(
            f"LLMReconstructor initialized: quality_threshold={quality_threshold}, "
            f"max_cache_size={max_cache_size}"
        )
    
    async def reconstruct(
        self,
        compressed: CompressedMemory,
        verify_quality: bool = True
    ) -> ReconstructedMemory:
        """
        Reconstruct memory from compressed data
        
        Algorithm:
        1. Lookup summary (cache → Arrow table → empty)
        2. Expand summary to full text (LLM)
        3. Apply diff to add missing details
        4. Verify reconstruction quality
        5. Return reconstructed memory
        
        Args:
            compressed: Compressed memory
            verify_quality: Whether to verify quality (default: True)
            
        Returns:
            ReconstructedMemory: Reconstructed memory
            
        Raises:
            ReconstructionError: If reconstruction fails
        
        Requirements: 6.1, 6.2, 6.3, 6.4
        """
        import time
        start_time = time.time()
        
        try:
            # Step 1: Lookup summary
            summary = self._lookup_summary(compressed.summary_hash)
            
            # Step 2: Expand summary (if available)
            if summary:
                reconstructed_text = await self._expand_summary(
                    summary,
                    compressed.entities
                )
            else:
                # No summary, start with empty text
                reconstructed_text = ""
            
            # Step 3: Apply diff
            final_text = self._apply_diff(reconstructed_text, compressed.diff_data)
            
            # Step 4: Verify quality (if requested)
            quality_metrics = None
            if verify_quality:
                quality_metrics = self._verify_reconstruction_quality(
                    final_text,
                    compressed.entities
                )
            
            # Calculate reconstruction time
            reconstruction_time_ms = (time.time() - start_time) * 1000
            
            # Step 5: Build reconstructed memory
            confidence = quality_metrics.overall_score if quality_metrics else 1.0
            warnings = []
            
            if quality_metrics and quality_metrics.overall_score < self.quality_threshold:
                warnings.append(
                    f"Low reconstruction quality: {quality_metrics.overall_score:.2f} "
                    f"(threshold: {self.quality_threshold})"
                )
                warnings.extend(quality_metrics.warnings)
            
            reconstructed = ReconstructedMemory(
                memory_id=compressed.memory_id,
                full_text=final_text,
                quality_metrics=quality_metrics,
                reconstruction_time_ms=reconstruction_time_ms,
                confidence=confidence,
                warnings=warnings,
                original_fields=compressed.original_fields
            )
            
            logger.info(
                f"Reconstruction complete: {compressed.memory_id} "
                f"({len(final_text)} chars) in {reconstruction_time_ms:.2f}ms, "
                f"confidence={confidence:.2f}"
            )
            
            return reconstructed
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            raise ReconstructionError(f"Failed to reconstruct memory: {e}")
    
    async def reconstruct_batch(
        self,
        compressed_list: List[CompressedMemory],
        verify_quality: bool = True
    ) -> List[ReconstructedMemory]:
        """
        Reconstruct multiple memories in parallel
        
        Args:
            compressed_list: List of compressed memories
            verify_quality: Whether to verify quality
            
        Returns:
            List[ReconstructedMemory]: List of reconstructed memories
            
        Requirements: 6.6
        """
        tasks = [
            self.reconstruct(compressed, verify_quality)
            for compressed in compressed_list
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        reconstructed_list = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch reconstruction failed for item {i}: {result}")
                # Create fallback reconstructed memory
                compressed = compressed_list[i]
                reconstructed_list.append(ReconstructedMemory(
                    memory_id=compressed.memory_id,
                    full_text="",
                    quality_metrics=None,
                    reconstruction_time_ms=0.0,
                    confidence=0.0,
                    warnings=[f"Reconstruction failed: {result}"],
                    original_fields=compressed.original_fields
                ))
            else:
                reconstructed_list.append(result)
        
        logger.info(f"Batch reconstruction complete: {len(reconstructed_list)} memories")
        return reconstructed_list
    
    def _lookup_summary(self, summary_hash: str) -> str:
        """
        Lookup summary using 3-level strategy
        
        Strategy:
        1. Memory cache (LRU)
        2. Arrow table lookup (TODO: implement in storage layer)
        3. Return empty string (use diff-only reconstruction)
        
        Args:
            summary_hash: Summary hash
            
        Returns:
            str: Summary text or empty string
            
        Requirements: 6.1
        """
        # Level 1: Memory cache
        if summary_hash in self.summary_cache:
            # Move to end (LRU) - only if OrderedDict
            if hasattr(self.summary_cache, 'move_to_end'):
                self.summary_cache.move_to_end(summary_hash)
            logger.debug(f"Summary cache hit: {summary_hash}")
            return self.summary_cache[summary_hash]
        
        # Level 2: Arrow table lookup (TODO: implement when storage layer is ready)
        # For now, return empty string
        
        # Level 3: Not found
        logger.debug(f"Summary not found: {summary_hash}")
        return ""
    
    def _cache_summary(self, summary_hash: str, summary: str):
        """
        Cache summary with LRU eviction
        
        Args:
            summary_hash: Summary hash
            summary: Summary text
        """
        # Add to cache
        self.summary_cache[summary_hash] = summary
        self.summary_cache.move_to_end(summary_hash)
        
        # Evict oldest if cache is full
        if len(self.summary_cache) > self.max_cache_size:
            oldest_key = next(iter(self.summary_cache))
            del self.summary_cache[oldest_key]
            logger.debug(f"Evicted summary from cache: {oldest_key}")
    
    async def _expand_summary(
        self,
        summary: str,
        entities: Dict[str, List[str]]
    ) -> str:
        """
        Expand summary into complete text using LLM
        
        Args:
            summary: Summary text
            entities: Extracted entities
            
        Returns:
            str: Expanded text
            
        Requirements: 6.1
        """
        # Format entities for prompt
        entities_str = self._format_entities(entities)
        
        # Build expansion prompt
        prompt = f"""Expand the following summary into a complete, natural text.
Incorporate these key entities: {entities_str}

Summary: {summary}

Expanded text:"""
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            expanded_text = response.text.strip()
            logger.debug(f"Summary expanded: {len(summary)} -> {len(expanded_text)} chars")
            return expanded_text
            
        except Exception as e:
            logger.warning(f"Summary expansion failed: {e}, using summary as-is")
            return summary
    
    def _format_entities(self, entities: Dict[str, List[str]]) -> str:
        """
        Format entities for prompt
        
        Args:
            entities: Entity dictionary
            
        Returns:
            str: Formatted entities string
        """
        parts = []
        for entity_type, entity_list in entities.items():
            if entity_list:
                parts.append(f"{entity_type}: {', '.join(entity_list[:5])}")
        
        return "; ".join(parts) if parts else "none"
    
    def _apply_diff(self, reconstructed: str, diff_data: bytes) -> str:
        """
        Apply diff data to reconstructed text
        
        Algorithm:
        1. Decompress diff data (zstd)
        2. Parse additions
        3. Intelligently insert into reconstructed text
        
        Args:
            reconstructed: Reconstructed text from summary
            diff_data: Compressed diff data
            
        Returns:
            str: Final text with diff applied
            
        Requirements: 6.1
        """
        if not diff_data:
            return reconstructed
        
        try:
            # Try to decompress diff
            try:
                diff_text = zstd.decompress(diff_data).decode('utf-8')
            except Exception:
                # If decompression fails, assume it's uncompressed (for short texts)
                diff_text = diff_data.decode('utf-8')
            
            if not diff_text.strip():
                return reconstructed
            
            # Parse additions
            additions = [line.strip() for line in diff_text.split('\n') if line.strip()]
            
            if not additions:
                return reconstructed
            
            # Simple strategy: append additions to reconstructed text
            # TODO: Implement intelligent insertion (fuzzy matching, position detection)
            if reconstructed:
                final_text = reconstructed + " " + " ".join(additions)
            else:
                final_text = " ".join(additions)
            
            logger.debug(f"Diff applied: {len(additions)} additions")
            return final_text
            
        except Exception as e:
            logger.warning(f"Diff application failed: {e}, using reconstructed text as-is")
            return reconstructed
    
    def _verify_reconstruction_quality(
        self,
        reconstructed: str,
        expected_entities: Dict[str, List[str]]
    ) -> QualityMetrics:
        """
        Verify reconstruction quality without original text
        
        Checks:
        1. Entity completeness (all expected entities present)
        2. Text coherence (grammar, punctuation)
        3. Length reasonableness (based on entity count)
        
        Args:
            reconstructed: Reconstructed text
            expected_entities: Expected entities
            
        Returns:
            QualityMetrics: Quality metrics
            
        Requirements: 6.4
        """
        warnings = []
        
        # 1. Entity completeness
        entity_accuracy = self._check_entity_completeness(
            reconstructed,
            expected_entities,
            warnings
        )
        
        # 2. Text coherence
        coherence_score = self._check_coherence(reconstructed, warnings)
        
        # 3. Length reasonableness
        length_score = self._check_length_reasonableness(
            reconstructed,
            expected_entities,
            warnings
        )
        
        # Calculate overall score (weighted average)
        overall_score = (
            entity_accuracy * 0.5 +
            coherence_score * 0.3 +
            length_score * 0.2
        )
        
        return QualityMetrics(
            entity_accuracy=entity_accuracy,
            coherence_score=coherence_score,
            length_score=length_score,
            overall_score=overall_score,
            warnings=warnings
        )
    
    def _check_entity_completeness(
        self,
        text: str,
        expected_entities: Dict[str, List[str]],
        warnings: List[str]
    ) -> float:
        """
        Check if all expected entities are present
        
        Args:
            text: Reconstructed text
            expected_entities: Expected entities
            warnings: Warning list to append to
            
        Returns:
            float: Entity accuracy (0-1)
        """
        if not expected_entities:
            return 1.0
        
        total_entities = 0
        found_entities = 0
        
        for entity_type, entity_list in expected_entities.items():
            for entity in entity_list:
                total_entities += 1
                if entity.lower() in text.lower():
                    found_entities += 1
                else:
                    warnings.append(f"Missing {entity_type}: {entity}")
        
        accuracy = found_entities / total_entities if total_entities > 0 else 1.0
        
        if accuracy < 0.9:
            warnings.append(f"Low entity completeness: {accuracy:.1%}")
        
        return accuracy
    
    def _check_coherence(self, text: str, warnings: List[str]) -> float:
        """
        Check text coherence
        
        Simple heuristics:
        - Sentence completeness (ends with punctuation)
        - No excessive repetition
        - Reasonable punctuation
        
        Args:
            text: Reconstructed text
            warnings: Warning list to append to
            
        Returns:
            float: Coherence score (0-1)
        """
        if not text.strip():
            warnings.append("Empty text")
            return 0.0
        
        score = 1.0
        
        # Check sentence completeness
        if not text.strip().endswith(('.', '!', '?')):
            score -= 0.2
            warnings.append("Incomplete sentence (no ending punctuation)")
        
        # Check for excessive repetition
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 0.3
                warnings.append(f"High word repetition (unique ratio: {unique_ratio:.1%})")
        
        return max(0.0, score)
    
    def _check_length_reasonableness(
        self,
        text: str,
        expected_entities: Dict[str, List[str]],
        warnings: List[str]
    ) -> float:
        """
        Check if text length is reasonable based on entity count
        
        Args:
            text: Reconstructed text
            expected_entities: Expected entities
            warnings: Warning list to append to
            
        Returns:
            float: Length score (0-1)
        """
        entity_count = sum(len(v) for v in expected_entities.values())
        
        if entity_count == 0:
            # No entities, any length is reasonable
            return 1.0
        
        # Estimate reasonable length range
        expected_min_length = entity_count * 5   # 5 words per entity minimum
        expected_max_length = entity_count * 50  # 50 words per entity maximum
        
        actual_length = len(text.split())
        
        if expected_min_length <= actual_length <= expected_max_length:
            return 1.0
        elif actual_length < expected_min_length:
            score = actual_length / expected_min_length
            warnings.append(
                f"Text too short: {actual_length} words "
                f"(expected: {expected_min_length}-{expected_max_length})"
            )
            return score
        else:
            score = expected_max_length / actual_length
            warnings.append(
                f"Text too long: {actual_length} words "
                f"(expected: {expected_min_length}-{expected_max_length})"
            )
            return score
    
    async def _reconstruct_from_diff_only(
        self,
        compressed: CompressedMemory
    ) -> ReconstructedMemory:
        """
        Reconstruct from diff only (fallback when LLM unavailable)
        
        Args:
            compressed: Compressed memory
            
        Returns:
            ReconstructedMemory: Partial reconstruction
            
        Requirements: 6.7
        """
        import time
        start_time = time.time()
        
        try:
            # Apply diff without summary expansion
            final_text = self._apply_diff("", compressed.diff_data)
            
            reconstruction_time_ms = (time.time() - start_time) * 1000
            
            return ReconstructedMemory(
                memory_id=compressed.memory_id,
                full_text=final_text,
                quality_metrics=None,
                reconstruction_time_ms=reconstruction_time_ms,
                confidence=0.5,  # Lower confidence for diff-only
                warnings=["Reconstructed from diff only (LLM unavailable)"],
                original_fields=compressed.original_fields
            )
            
        except Exception as e:
            logger.error(f"Diff-only reconstruction failed: {e}")
            raise ReconstructionError(f"Failed to reconstruct from diff: {e}")

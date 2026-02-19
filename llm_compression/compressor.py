"""
LLM Compressor Module

Implements semantic compression using LLM to achieve 10-50x compression ratio.
Uses summary generation, entity extraction, and diff computation.

Requirements: 5.1-5.7
"""

import asyncio
import hashlib
import time
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter

try:
    import zstandard as zstd
except ImportError:
    import zstd  # fallback

from llm_compression.logger import logger
from llm_compression.llm_client import LLMClient, LLMResponse
from llm_compression.model_selector import ModelSelector, MemoryType, QualityLevel
from llm_compression.errors import CompressionError, LLMAPIError, LLMTimeoutError


@dataclass
class CompressionMetadata:
    """Compression metadata
    
    Attributes:
        original_size: Original size in bytes
        compressed_size: Compressed size in bytes
        compression_ratio: Compression ratio (original / compressed)
        model_used: Model name used for compression
        quality_score: Quality score (0-1)
        compression_time_ms: Compression time in milliseconds
        compressed_at: Compression timestamp
    """
    original_size: int
    compressed_size: int
    compression_ratio: float
    model_used: str
    quality_score: float
    compression_time_ms: float
    compressed_at: datetime


@dataclass
class CompressedMemory:
    """Compressed memory
    
    Attributes:
        memory_id: Unique memory ID
        summary_hash: Summary hash (SHA256, first 16 chars)
        entities: Extracted entities by type
        diff_data: Compressed diff data (zstd)
        embedding: Vector representation (float16)
        compression_metadata: Compression metadata
        original_fields: Original fields (timestamp, intent, etc.)
    """
    memory_id: str
    summary_hash: str
    entities: Dict[str, List[str]]
    diff_data: bytes
    embedding: List[float]
    compression_metadata: CompressionMetadata
    original_fields: Dict[str, Any] = field(default_factory=dict)
    
    # Phase 2: Vector Space Compression Fields
    sparse_vector: Optional[bytes] = None
    sparse_indices: Optional[bytes] = None
    sparse_meta: Optional[Dict[str, Any]] = None
    key_tokens: List[str] = field(default_factory=list)
    token_scores: List[float] = field(default_factory=list)
    relations: List[Tuple[str, str, float]] = field(default_factory=list)


# Remove old CompressionError definition - now imported from errors module


class LLMCompressor:
    """LLM semantic compressor
    
    Implements semantic compression algorithm:
    1. Generate semantic summary using LLM
    2. Extract key entities (persons, dates, numbers, locations)
    3. Compute diff between original and summary
    4. Store summary_hash + entities + diff
    
    Requirements: 5.1-5.7
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        model_selector: ModelSelector,
        min_compress_length: int = 100,
        max_tokens: int = 100,
        temperature: float = 0.3,
        prewarm_embedding: bool = True
    ):
        """Initialize compressor
        
        Args:
            llm_client: LLM client for summary generation
            model_selector: Model selector for choosing optimal model
            min_compress_length: Minimum text length to compress (chars)
            max_tokens: Maximum tokens for summary generation
            temperature: Sampling temperature for LLM
            prewarm_embedding: Pre-load embedding model on initialization
        
        Requirements: 5.1
        """
        self.llm_client = llm_client
        self.model_selector = model_selector
        self.min_compress_length = min_compress_length
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Summary cache (summary_hash -> summary text)
        self.summary_cache: Dict[str, str] = {}
        self._max_cache_size = 10000
        
        # Lazy load embedding model
        self._embedding_model = None
        
        # Pre-warm embedding model if requested
        if prewarm_embedding:
            _ = self.embedding_model  # Trigger lazy loading
        
        logger.info(
            f"LLMCompressor initialized: "
            f"min_length={min_compress_length}, "
            f"max_tokens={max_tokens}, "
            f"temperature={temperature}, "
            f"prewarm_embedding={prewarm_embedding}"
        )
    
    @property
    def embedding_model(self):
        """Lazy load embedding provider"""
        if self._embedding_model is None:
            try:
                from llm_compression.embedding_provider import get_default_provider
                logger.info("Loading embedding provider for compressor")
                self._embedding_model = get_default_provider()
                logger.info(f"Embedding provider loaded: {self._embedding_model}")
            except Exception as e:
                logger.error(f"Failed to load embedding provider: {e}")
                raise
        return self._embedding_model
    
    async def compress(
        self,
        text: str,
        memory_type: MemoryType = MemoryType.TEXT,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CompressedMemory:
        """Compress memory
        
        Algorithm:
        1. Check text length (< min_length -> store uncompressed)
        2. Select optimal model
        3. Generate semantic summary using LLM
        4. Extract key entities
        5. Compute diff between original and summary
        6. Compress diff with zstd
        7. Calculate summary hash
        8. Build CompressedMemory object
        
        Args:
            text: Original text to compress
            memory_type: Memory type (TEXT/CODE/MULTIMODAL/LONG_TEXT)
            metadata: Additional metadata
        
        Returns:
            CompressedMemory: Compressed memory object
        
        Raises:
            CompressionError: If compression fails
        
        Requirements: 5.1, 5.2
        """
        start_time = time.time()
        
        try:
            # Step 1: Check text length
            if len(text) < self.min_compress_length:
                logger.debug(
                    f"Text too short ({len(text)} < {self.min_compress_length}), "
                    "storing uncompressed"
                )
                return self._store_uncompressed(text, metadata)
            
            # Step 2: Select model
            model_config = self.model_selector.select_model(
                memory_type=memory_type,
                text_length=len(text),
                quality_requirement=QualityLevel.STANDARD
            )
            
            logger.debug(
                f"Compressing text (length={len(text)}) "
                f"with model={model_config.model_name}"
            )
            
            # Step 3: Generate summary
            summary = await self._generate_summary(text, model_config)
            
            # Step 4: Extract entities
            entities = self._extract_entities(text)
            
            # Step 5: Compute diff
            diff_data = self._compute_diff(text, summary)
            
            # Step 6: Compress diff with zstd
            compressed_diff = zstd.compress(diff_data, level=3)
            
            # Step 7: Calculate summary hash
            summary_hash = hashlib.sha256(summary.encode()).hexdigest()[:16]
            
            # Cache summary for reconstruction
            self.summary_cache[summary_hash] = summary
            if len(self.summary_cache) > self._max_cache_size:
                # Remove oldest entry (simple FIFO)
                self.summary_cache.pop(next(iter(self.summary_cache)))
            
            # Step 8: Compute embedding
            embedding = self._compute_embedding(text)
            
            # Calculate sizes and ratio
            original_size = len(text.encode('utf-8'))
            compressed_size = len(compressed_diff)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0.0
            
            compression_time_ms = (time.time() - start_time) * 1000
            
            # Build compressed memory
            compressed = CompressedMemory(
                memory_id=self._generate_id(),
                summary_hash=summary_hash,
                entities=entities,
                diff_data=compressed_diff,
                embedding=embedding,
                compression_metadata=CompressionMetadata(
                    original_size=original_size,
                    compressed_size=compressed_size,
                    compression_ratio=compression_ratio,
                    model_used=model_config.model_name,
                    quality_score=0.0,  # Will be set by quality evaluator
                    compression_time_ms=compression_time_ms,
                    compressed_at=datetime.now()
                ),
                original_fields=metadata or {}
            )
            
            # Check if compression actually reduced size
            if compressed_size >= original_size:
                logger.warning(
                    f"Compression failed to reduce size: "
                    f"{original_size} -> {compressed_size}, "
                    "falling back to uncompressed"
                )
                return self._store_uncompressed(text, metadata)
            
            logger.info(
                f"Compression complete: "
                f"{original_size} -> {compressed_size} bytes "
                f"({compression_ratio:.2f}x ratio) "
                f"in {compression_time_ms:.2f}ms"
            )
            
            return compressed
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise CompressionError(f"Failed to compress text: {e}") from e
    
    async def compress_batch(
        self,
        texts: List[str],
        memory_type: MemoryType = MemoryType.TEXT
    ) -> List[CompressedMemory]:
        """Compress batch of texts
        
        Uses async concurrency for parallel compression.
        
        Args:
            texts: List of texts to compress
            memory_type: Memory type
        
        Returns:
            List[CompressedMemory]: List of compressed memories
        
        Requirements: 9.1
        """
        logger.info(f"Compressing batch of {len(texts)} texts")
        
        tasks = [
            self.compress(text, memory_type)
            for text in texts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        compressed_list = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch compression failed for item {i}: {result}")
                # Store uncompressed as fallback
                compressed_list.append(
                    self._store_uncompressed(texts[i], {})
                )
            else:
                compressed_list.append(result)
        
        logger.info(f"Batch compression complete: {len(compressed_list)} results")
        return compressed_list
    
    async def _generate_summary(self, text: str, model_config) -> str:
        """Generate semantic summary using LLM
        
        Prompt: "Summarize the following text in 50-100 tokens,
                 preserving key facts and entities: {text}"
        
        Args:
            text: Original text
            model_config: Model configuration
        
        Returns:
            str: Generated summary
        
        Requirements: 5.1
        """
        prompt = f"""Summarize the following text in 50-100 tokens, preserving key facts and entities:

{text}

Summary:"""
        
        try:
            response: LLMResponse = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            summary = response.text.strip()
            
            logger.debug(
                f"Generated summary: {len(summary)} chars "
                f"(original: {len(text)} chars)"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Fallback: use first N characters as summary
            fallback_summary = text[:200]
            logger.warning(f"Using fallback summary: first 200 chars")
            return fallback_summary
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract key entities from text
        
        Extracts:
        - persons: Capitalized names (2-4 words)
        - dates: ISO dates, natural language dates, times
        - numbers: Integers, decimals, percentages, currency
        - locations: Common location patterns
        - keywords: High-frequency meaningful words (top 5)
        
        Args:
            text: Input text
        
        Returns:
            Dict mapping entity types to lists of entities
        
        Requirements: 5.1, 5.5
        """
        entities = {
            'persons': [],
            'locations': [],
            'dates': [],
            'numbers': [],
            'keywords': []
        }
        
        try:
            # Extract dates
            # ISO format: 2024-01-15
            iso_dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
            entities['dates'].extend(iso_dates)
            
            # Natural language dates
            natural_dates = re.findall(
                r'(?:January|February|March|April|May|June|July|August|'
                r'September|October|November|December|'
                r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
                r'\s+\d{1,2},?\s+\d{4}',
                text
            )
            entities['dates'].extend(natural_dates)
            
            # Time patterns: 3pm, 15:30, 3:30pm
            times = re.findall(r'\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)', text)
            entities['dates'].extend(times)
            
            # Extract numbers (integers, decimals, percentages, currency)
            numbers = re.findall(r'[$€£¥]?\d+(?:,\d{3})*(?:\.\d+)?%?', text)
            entities['numbers'].extend(numbers)
            
            # Extract person names (capitalized words, 2-4 words)
            potential_names = re.findall(
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b',
                text
            )
            entities['persons'].extend(potential_names)
            
            # Extract keywords (meaningful words, 4+ characters)
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            if words:
                word_freq = Counter(words)
                # Get top 5 most frequent words
                top_keywords = [word for word, _ in word_freq.most_common(5)]
                entities['keywords'] = top_keywords
            
            # Remove duplicates while preserving order
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))
            
            logger.debug(
                f"Extracted entities: "
                f"persons={len(entities['persons'])}, "
                f"dates={len(entities['dates'])}, "
                f"numbers={len(entities['numbers'])}"
            )
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
        
        return entities
    
    def _compute_diff(self, original: str, summary: str) -> bytes:
        """Compute diff between original and summary
        
        Algorithm:
        1. Split original and summary into words
        2. Use difflib.unified_diff to compute differences
        3. Keep only additions (+ lines)
        4. Return as bytes
        
        The diff captures information present in original but not in summary.
        
        Args:
            original: Original text
            summary: Summary text
        
        Returns:
            bytes: Diff data (uncompressed)
        
        Requirements: 5.1
        """
        import difflib
        
        try:
            # Split into words for better granularity
            original_words = original.split()
            summary_words = summary.split()
            
            # Compute unified diff
            diff = difflib.unified_diff(
                summary_words,
                original_words,
                lineterm=''
            )
            
            # Keep only additions (lines starting with '+')
            additions = []
            for line in diff:
                if line.startswith('+ '):
                    additions.append(line[2:])  # Remove '+ ' prefix
            
            # Join additions
            diff_text = '\n'.join(additions)
            
            logger.debug(
                f"Computed diff: {len(diff_text)} chars "
                f"({len(additions)} additions)"
            )
            
            return diff_text.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Diff computation failed: {e}")
            # Fallback: return empty diff
            return b''
    
    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding vector for text
        
        Uses EmbeddingProvider to generate embedding.
        Converts to float16 to save space (50% reduction).
        
        Args:
            text: Input text
        
        Returns:
            List[float]: Embedding vector (float16)
        
        Requirements: 8.3
        """
        try:
            import numpy as np
            
            # Generate embedding using provider (returns float32 ndarray)
            embedding = self.embedding_model.encode(text, normalize=True)
            
            # Convert to float16 for space savings
            embedding_float16 = embedding.astype(np.float16)
            
            # Convert to list
            embedding_list = embedding_float16.tolist()
            
            logger.debug(f"Computed embedding: {len(embedding_list)} dimensions")
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * 384  # MiniLM-L6-v2 has 384 dimensions
    
    async def get_embedding(self, text: str) -> List[float]:
        """Public async method to get embedding for text
        
        Args:
            text: Input text
            
        Returns:
            List[float]: Embedding vector
        """
        return self._compute_embedding(text)
    
    def _store_uncompressed(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> CompressedMemory:
        """Store text without compression
        
        Used for:
        - Text shorter than min_compress_length
        - Compression that doesn't reduce size
        - Compression failures
        
        Args:
            text: Original text
            metadata: Additional metadata
        
        Returns:
            CompressedMemory: Uncompressed memory (marked as such)
        
        Requirements: 5.2, 5.7
        """
        original_size = len(text.encode('utf-8'))
        
        # Store original text as "diff"
        diff_data = text.encode('utf-8')
        
        # Compute embedding
        embedding = self._compute_embedding(text)
        
        compressed = CompressedMemory(
            memory_id=self._generate_id(),
            summary_hash="",  # Empty for uncompressed
            entities={},  # No entities extracted
            diff_data=diff_data,
            embedding=embedding,
            compression_metadata=CompressionMetadata(
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,  # No compression
                model_used="uncompressed",
                quality_score=1.0,  # Perfect quality (lossless)
                compression_time_ms=0.0,
                compressed_at=datetime.now()
            ),
            original_fields=metadata or {}
        )
        
        logger.debug(f"Stored uncompressed: {original_size} bytes")
        
        return compressed
    
    def _generate_id(self) -> str:
        """Generate unique memory ID
        
        Returns:
            str: Unique ID (timestamp + random)
        """
        import uuid
        timestamp = int(time.time() * 1000)
        random_part = uuid.uuid4().hex[:8]
        return f"{timestamp}_{random_part}"

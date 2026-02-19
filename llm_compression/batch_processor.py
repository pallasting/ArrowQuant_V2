"""
Batch Processor Module

Implements efficient batch processing for memory compression with:
- Asynchronous concurrent processing
- Similar text grouping for cache efficiency
- Configurable batch size and concurrency limits
- Checkpoint/resume support for fault tolerance

Feature: llm-compression-integration
Requirements: 9.1, 9.3, 9.4, 9.6, 9.7
"""

import asyncio
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

from llm_compression.logger import logger
from llm_compression.compressor import LLMCompressor, CompressedMemory, MemoryType
from llm_compression.errors import CompressionError


@dataclass
class BatchProgress:
    """Batch processing progress for checkpoint/resume"""
    total_items: int
    completed_items: int
    failed_items: int
    start_time: float
    last_checkpoint_time: float
    completed_indices: List[int]
    failed_indices: List[int]


class BatchProcessor:
    """
    Batch processor for efficient memory compression
    
    Features:
    - Asynchronous concurrent processing (asyncio)
    - Similar text grouping (improves cache hit rate)
    - Configurable batch size and concurrency
    - Checkpoint/resume support (fault tolerance)
    - Throughput: > 100/min (local model), > 50/min (cloud API)
    
    Requirements: 9.1, 9.3, 9.4, 9.6, 9.7
    """
    
    def __init__(
        self,
        compressor: LLMCompressor,
        batch_size: int = 16,
        max_concurrent: int = 4,
        similarity_threshold: float = 0.8,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize batch processor
        
        Args:
            compressor: LLM compressor instance
            batch_size: Maximum batch size (1-32)
            max_concurrent: Maximum concurrent tasks
            similarity_threshold: Threshold for grouping similar texts (0-1)
            checkpoint_dir: Directory for checkpoint files (optional)
        
        Requirements: 9.1, 9.3
        """
        self.compressor = compressor
        self.batch_size = max(1, min(32, batch_size))
        self.max_concurrent = max_concurrent
        self.similarity_threshold = similarity_threshold
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Progress tracking
        self.current_progress: Optional[BatchProgress] = None
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"BatchProcessor initialized: batch_size={self.batch_size}, "
            f"max_concurrent={max_concurrent}, "
            f"similarity_threshold={similarity_threshold}"
        )
    
    async def compress_batch(
        self,
        texts: List[str],
        memory_type: MemoryType = MemoryType.TEXT,
        checkpoint_id: Optional[str] = None,
        resume: bool = False
    ) -> List[CompressedMemory]:
        """
        Compress batch of texts with optimizations
        
        Optimizations:
        1. Group similar texts (improves cache hit rate)
        2. Concurrent processing (asyncio)
        3. Checkpoint/resume support (fault tolerance)
        
        Args:
            texts: List of texts to compress
            memory_type: Memory type
            checkpoint_id: Checkpoint ID for resume (optional)
            resume: Whether to resume from checkpoint
        
        Returns:
            List[CompressedMemory]: Compressed memories
        
        Requirements: 9.1, 9.3, 9.4, 9.6
        """
        start_time = time.time()
        
        logger.info(
            f"Starting batch compression: {len(texts)} texts, "
            f"type={memory_type.value}, resume={resume}"
        )
        
        # Initialize or load progress
        if resume and checkpoint_id:
            progress = self._load_checkpoint(checkpoint_id)
            if progress:
                logger.info(
                    f"Resuming from checkpoint: "
                    f"{progress.completed_items}/{progress.total_items} completed"
                )
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_id}, starting fresh")
                progress = self._init_progress(len(texts), start_time)
        else:
            progress = self._init_progress(len(texts), start_time)
        
        self.current_progress = progress
        
        # Step 1: Group similar texts
        grouped = self._group_similar_texts(texts)
        logger.info(f"Grouped {len(texts)} texts into {len(grouped)} groups")
        
        # Step 2: Process each group concurrently
        results: List[Optional[CompressedMemory]] = [None] * len(texts)
        
        tasks = []
        for group_indices, group_texts in grouped:
            # Skip already completed items
            if resume:
                group_indices = [
                    idx for idx in group_indices
                    if idx not in progress.completed_indices
                ]
                if not group_indices:
                    continue
                group_texts = [texts[idx] for idx in group_indices]
            
            task = self._compress_group(
                group_indices,
                group_texts,
                memory_type,
                results,
                checkpoint_id
            )
            tasks.append(task)
        
        # Wait for all groups to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Fill in any None values with failed compressions
        for i, result in enumerate(results):
            if result is None:
                logger.error(f"Compression failed for item {i}, using uncompressed")
                results[i] = self.compressor._store_uncompressed(texts[i], {})
        
        # Calculate throughput
        elapsed_time = time.time() - start_time
        throughput = len(texts) / (elapsed_time / 60)  # items per minute
        
        logger.info(
            f"Batch compression complete: {len(texts)} texts in {elapsed_time:.2f}s "
            f"({throughput:.1f} items/min), "
            f"completed={progress.completed_items}, failed={progress.failed_items}"
        )
        
        # Clean up checkpoint if successful
        if checkpoint_id and progress.failed_items == 0:
            self._delete_checkpoint(checkpoint_id)
        
        return results
    
    async def _compress_group(
        self,
        indices: List[int],
        texts: List[str],
        memory_type: MemoryType,
        results: List[Optional[CompressedMemory]],
        checkpoint_id: Optional[str]
    ):
        """
        Compress a group of similar texts
        
        Args:
            indices: Original indices in the batch
            texts: Texts to compress
            memory_type: Memory type
            results: Results list to populate
            checkpoint_id: Checkpoint ID for progress tracking
        
        Requirements: 9.3, 9.4
        """
        async with self.semaphore:
            for idx, text in zip(indices, texts):
                try:
                    # Compress single text
                    compressed = await self.compressor.compress(text, memory_type)
                    results[idx] = compressed
                    
                    # Update progress
                    if self.current_progress:
                        self.current_progress.completed_items += 1
                        self.current_progress.completed_indices.append(idx)
                        
                        # Save checkpoint periodically
                        if checkpoint_id and self._should_checkpoint():
                            self._save_checkpoint(checkpoint_id, self.current_progress)
                    
                except Exception as e:
                    logger.error(f"Compression failed for item {idx}: {e}")
                    
                    # Update progress
                    if self.current_progress:
                        self.current_progress.failed_items += 1
                        self.current_progress.failed_indices.append(idx)
                    
                    # Store uncompressed as fallback
                    results[idx] = self.compressor._store_uncompressed(text, {})
    
    def _group_similar_texts(
        self,
        texts: List[str]
    ) -> List[tuple[List[int], List[str]]]:
        """
        Group similar texts for efficient batch processing
        
        Algorithm:
        1. Compute embeddings for all texts
        2. Use simple clustering based on cosine similarity
        3. Group texts with similarity > threshold
        4. Limit group size to batch_size
        
        Args:
            texts: List of texts to group
        
        Returns:
            List of (indices, texts) tuples for each group
        
        Requirements: 9.4
        """
        if len(texts) <= self.batch_size:
            # Small batch, no need to group
            return [(list(range(len(texts))), texts)]
        
        try:
            # Compute embeddings
            embeddings = [
                self.compressor._compute_embedding(text)
                for text in texts
            ]
            
            # Simple clustering
            groups = []
            used = set()
            
            for i, text in enumerate(texts):
                if i in used:
                    continue
                
                group_indices = [i]
                group_texts = [text]
                used.add(i)
                
                # Find similar texts
                for j in range(i + 1, len(texts)):
                    if j in used:
                        continue
                    
                    similarity = self._cosine_similarity(
                        embeddings[i],
                        embeddings[j]
                    )
                    
                    if similarity > self.similarity_threshold:
                        group_indices.append(j)
                        group_texts.append(texts[j])
                        used.add(j)
                    
                    if len(group_indices) >= self.batch_size:
                        break
                
                groups.append((group_indices, group_texts))
            
            logger.debug(f"Grouped {len(texts)} texts into {len(groups)} groups")
            return groups
            
        except Exception as e:
            logger.warning(f"Text grouping failed: {e}, using sequential processing")
            # Fallback: create groups of batch_size
            groups = []
            for i in range(0, len(texts), self.batch_size):
                end = min(i + self.batch_size, len(texts))
                indices = list(range(i, end))
                group_texts = texts[i:end]
                groups.append((indices, group_texts))
            return groups
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            float: Cosine similarity (0-1)
        """
        try:
            import numpy as np
            
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Cosine similarity computation failed: {e}")
            return 0.0
    
    def _init_progress(self, total_items: int, start_time: float) -> BatchProgress:
        """Initialize batch progress"""
        return BatchProgress(
            total_items=total_items,
            completed_items=0,
            failed_items=0,
            start_time=start_time,
            last_checkpoint_time=start_time,
            completed_indices=[],
            failed_indices=[]
        )
    
    def _should_checkpoint(self) -> bool:
        """Check if checkpoint should be saved"""
        if not self.current_progress:
            return False
        
        # Checkpoint every 10 items or every 30 seconds
        items_since_checkpoint = (
            len(self.current_progress.completed_indices) % 10 == 0
        )
        time_since_checkpoint = (
            time.time() - self.current_progress.last_checkpoint_time > 30
        )
        
        return items_since_checkpoint or time_since_checkpoint
    
    def _save_checkpoint(self, checkpoint_id: str, progress: BatchProgress):
        """
        Save checkpoint to disk
        
        Args:
            checkpoint_id: Checkpoint ID
            progress: Progress to save
        
        Requirements: 9.6
        """
        if not self.checkpoint_dir:
            return
        
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            
            with open(checkpoint_file, 'w') as f:
                json.dump(asdict(progress), f, indent=2)
            
            progress.last_checkpoint_time = time.time()
            
            logger.debug(f"Checkpoint saved: {checkpoint_id}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, checkpoint_id: str) -> Optional[BatchProgress]:
        """
        Load checkpoint from disk
        
        Args:
            checkpoint_id: Checkpoint ID
        
        Returns:
            BatchProgress or None if not found
        
        Requirements: 9.6
        """
        if not self.checkpoint_dir:
            return None
        
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            
            if not checkpoint_file.exists():
                return None
            
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            progress = BatchProgress(**data)
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return progress
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _delete_checkpoint(self, checkpoint_id: str):
        """Delete checkpoint file"""
        if not self.checkpoint_dir:
            return
        
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.debug(f"Checkpoint deleted: {checkpoint_id}")
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")


class CompressionCache:
    """
    LRU cache for compression results
    
    Caches compressed memories by text hash to avoid redundant compression.
    
    Requirements: 9.3
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 3600
    ):
        """
        Initialize compression cache
        
        Args:
            max_size: Maximum cache size
            ttl_seconds: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple[CompressedMemory, float]] = {}
        self.lock = asyncio.Lock()
        
        logger.info(
            f"CompressionCache initialized: max_size={max_size}, ttl={ttl_seconds}s"
        )
    
    async def get(self, text_hash: str) -> Optional[CompressedMemory]:
        """
        Get compressed memory from cache
        
        Args:
            text_hash: Hash of the text
        
        Returns:
            CompressedMemory or None if not found/expired
        """
        async with self.lock:
            if text_hash not in self.cache:
                return None
            
            compressed, timestamp = self.cache[text_hash]
            
            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[text_hash]
                return None
            
            logger.debug(f"Cache hit: {text_hash}")
            return compressed
    
    async def set(self, text_hash: str, compressed: CompressedMemory):
        """
        Store compressed memory in cache
        
        Args:
            text_hash: Hash of the text
            compressed: Compressed memory
        """
        async with self.lock:
            # Evict oldest if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[text_hash] = (compressed, time.time())
            logger.debug(f"Cache set: {text_hash}")
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        # Find oldest entry
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
        del self.cache[oldest_key]
        logger.debug(f"Cache evicted: {oldest_key}")
    
    async def clear(self):
        """Clear all cache entries"""
        async with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")

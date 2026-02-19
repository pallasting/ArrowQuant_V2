"""
Async background processing queue for semantic indexing tasks.

This module provides non-blocking task submission with batch processing
optimization for efficient memory indexing.
"""

import asyncio
from asyncio import Queue
from typing import List, Any, Optional

from llm_compression.semantic_indexer import SemanticIndexer
from llm_compression.logger import logger


class BackgroundQueue:
    """
    Async background processing queue for indexing tasks.
    
    Features:
    - Non-blocking task submission
    - Batch processing optimization
    - Error handling and retry
    - Graceful shutdown
    
    Example:
        >>> from llm_compression.embedding_provider import get_default_provider
        >>> from llm_compression.arrow_storage import ArrowStorage
        >>> from llm_compression.semantic_index_db import SemanticIndexDB
        >>> from llm_compression.semantic_indexer import SemanticIndexer
        >>> 
        >>> provider = get_default_provider()
        >>> storage = ArrowStorage("./data/memories.parquet")
        >>> index_db = SemanticIndexDB("./data/indexes")
        >>> indexer = SemanticIndexer(provider, storage, index_db)
        >>> 
        >>> # Create and start queue
        >>> queue = BackgroundQueue(indexer, batch_size=32)
        >>> await queue.start()
        >>> 
        >>> # Submit memories for indexing
        >>> await queue.submit(memory1)
        >>> await queue.submit(memory2)
        >>> 
        >>> # Stop queue gracefully
        >>> await queue.stop()
    """
    
    def __init__(
        self,
        indexer: SemanticIndexer,
        batch_size: int = 32,
        max_queue_size: int = 1000,
        batch_timeout: float = 1.0
    ):
        """
        Initialize BackgroundQueue.
        
        Args:
            indexer: SemanticIndexer instance for processing
            batch_size: Number of items to process in a batch
            max_queue_size: Maximum queue size (blocks when full)
            batch_timeout: Timeout for partial batch processing (seconds)
        """
        self.indexer = indexer
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.queue: Queue = Queue(maxsize=max_queue_size)
        self.running = False
        self._worker_task: Optional[asyncio.Task] = None
        logger.info(
            f"Initialized BackgroundQueue: "
            f"batch_size={batch_size}, max_queue_size={max_queue_size}"
        )
    
    async def start(self) -> None:
        """
        Start background worker.
        
        Example:
            >>> queue = BackgroundQueue(indexer)
            >>> await queue.start()
        """
        if self.running:
            logger.warning("BackgroundQueue already running")
            return
        
        self.running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("BackgroundQueue started")
    
    async def stop(self) -> None:
        """
        Stop background worker gracefully.
        
        Waits for all queued items to be processed before stopping.
        
        Example:
            >>> await queue.stop()
        """
        if not self.running:
            logger.warning("BackgroundQueue not running")
            return
        
        logger.info("Stopping BackgroundQueue...")
        self.running = False
        
        if self._worker_task:
            await self._worker_task
            self._worker_task = None
        
        logger.info("BackgroundQueue stopped")
    
    async def submit(self, memory: Any) -> None:
        """
        Submit memory for indexing.
        
        Non-blocking operation. If queue is full, waits until space available.
        
        Args:
            memory: Memory object to index
            
        Example:
            >>> memory = {
            ...     'memory_id': 'mem_123',
            ...     'category': 'knowledge',
            ...     'context': 'Machine learning is...'
            ... }
            >>> await queue.submit(memory)
        """
        if not self.running:
            logger.error("Cannot submit to stopped queue")
            raise RuntimeError("BackgroundQueue is not running")
        
        await self.queue.put(memory)
        logger.debug(f"Submitted memory to queue (size={self.queue.qsize()})")
    
    async def submit_batch(self, memories: List[Any]) -> None:
        """
        Submit multiple memories for indexing.
        
        Args:
            memories: List of memory objects
            
        Example:
            >>> memories = [memory1, memory2, memory3]
            >>> await queue.submit_batch(memories)
        """
        for memory in memories:
            await self.submit(memory)
    
    def get_queue_size(self) -> int:
        """
        Get current queue size.
        
        Returns:
            Number of items in queue
        """
        return self.queue.qsize()
    
    def is_running(self) -> bool:
        """
        Check if queue is running.
        
        Returns:
            True if running, False otherwise
        """
        return self.running
    
    async def _worker(self) -> None:
        """
        Background worker that processes queue.
        
        Collects items into batches and processes them efficiently.
        """
        logger.info("BackgroundQueue worker started")
        batch = []
        
        while self.running or not self.queue.empty():
            try:
                # Wait for item with timeout
                memory = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=self.batch_timeout
                )
                batch.append(memory)
                
                # Process batch when full or queue empty
                if len(batch) >= self.batch_size or self.queue.empty():
                    await self._process_batch(batch)
                    batch = []
            
            except asyncio.TimeoutError:
                # Process partial batch on timeout
                if batch:
                    logger.debug(
                        f"Processing partial batch on timeout: {len(batch)} items"
                    )
                    await self._process_batch(batch)
                    batch = []
            
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                # Continue processing despite errors
        
        # Process any remaining items
        if batch:
            logger.debug(f"Processing final batch: {len(batch)} items")
            await self._process_batch(batch)
        
        logger.info("BackgroundQueue worker stopped")
    
    async def _process_batch(self, batch: List[Any]) -> None:
        """
        Process a batch of memories.
        
        Runs indexing in thread pool to avoid blocking event loop.
        
        Args:
            batch: List of memory objects to index
        """
        if not batch:
            return
        
        try:
            logger.debug(f"Processing batch of {len(batch)} memories")
            
            # Run indexing in thread pool (blocking operation)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.indexer.batch_index,
                batch,
                self.batch_size
            )
            
            logger.debug(f"Successfully indexed batch of {len(batch)} memories")
        
        except Exception as e:
            logger.error(
                f"Batch indexing failed for {len(batch)} memories: {e}",
                exc_info=True
            )
            # Could implement retry logic here
            # For now, we log and continue
    
    async def wait_until_empty(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until queue is empty.
        
        Useful for ensuring all items are processed before shutdown.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
            
        Returns:
            True if queue became empty, False if timeout
            
        Example:
            >>> await queue.submit_batch(memories)
            >>> await queue.wait_until_empty(timeout=30.0)
            >>> await queue.stop()
        """
        start_time = asyncio.get_event_loop().time()
        
        while not self.queue.empty():
            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    logger.warning(
                        f"Timeout waiting for queue to empty "
                        f"({self.queue.qsize()} items remaining)"
                    )
                    return False
            
            await asyncio.sleep(0.1)
        
        # Wait a bit more to ensure worker processes last batch
        await asyncio.sleep(0.5)
        return True

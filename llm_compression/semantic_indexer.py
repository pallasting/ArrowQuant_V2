"""
Semantic indexer for building and maintaining memory indexes.

This module provides incremental and batch indexing capabilities
for the AI-OS memory system.
"""

from typing import List, Any, Optional

from llm_compression.embedding_provider import EmbeddingProvider
from llm_compression.arrow_storage import ArrowStorage
from llm_compression.logger import logger


class SemanticIndexer:
    """
    Builds and maintains semantic indexes for memories.
    
    Features:
    - Incremental index updates
    - Batch indexing for efficiency
    - Index validation
    - Background processing support
    
    Example:
        >>> from llm_compression.embedding_provider import get_default_provider
        >>> from llm_compression.arrow_storage import ArrowStorage
        >>> from llm_compression.semantic_index_db import SemanticIndexDB
        >>> 
        >>> provider = get_default_provider()
        >>> storage = ArrowStorage("./data/memories.parquet")
        >>> index_db = SemanticIndexDB("./data/index.parquet")
        >>> indexer = SemanticIndexer(provider, storage, index_db)
        >>> 
        >>> # Index a single memory
        >>> indexer.index_memory(memory)
        >>> 
        >>> # Batch index multiple memories
        >>> indexer.batch_index(memories, batch_size=32)
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        storage: ArrowStorage,
        index_db: 'SemanticIndexDB'
    ):
        """
        Initialize SemanticIndexer.
        
        Args:
            embedding_provider: Provider for generating embeddings
            storage: Arrow storage backend for memories
            index_db: Database for storing semantic indexes
        """
        self.embedder = embedding_provider
        self.storage = storage
        self.index_db = index_db
        logger.info("Initialized SemanticIndexer")
    
    def index_memory(self, memory: Any) -> None:
        """
        Index a single memory.
        
        Generates embedding if not present and stores in index database.
        
        Args:
            memory: Memory object to index (CompressedMemory or dict)
            
        Example:
            >>> memory = {
            ...     'memory_id': 'mem_123',
            ...     'category': 'knowledge',
            ...     'context': 'Machine learning is...',
            ...     'timestamp': '2024-01-01T00:00:00'
            ... }
            >>> indexer.index_memory(memory)
        """
        # Extract memory attributes
        memory_id = self._get_attr(memory, 'memory_id')
        category = self._get_attr(memory, 'category')
        timestamp = self._get_attr(memory, 'timestamp')
        embedding = self._get_attr(memory, 'embedding')
        
        logger.debug(f"Indexing memory: {memory_id}")
        
        # Generate embedding if not present
        if embedding is None:
            text = self._extract_indexable_text(memory)
            embedding = self.embedder.encode(text, normalize=True)
            
            # Ensure embedding is 1D
            if embedding.ndim == 2:
                embedding = embedding[0]
            
            # Update memory with embedding
            self._set_attr(memory, 'embedding', embedding)
        
        # Store in index
        self.index_db.add_entry(
            memory_id=memory_id,
            category=category,
            embedding=embedding,
            timestamp=timestamp
        )
        
        logger.debug(f"Indexed memory: {memory_id}")
    
    def batch_index(
        self,
        memories: List[Any],
        batch_size: int = 32
    ) -> None:
        """
        Batch index multiple memories.
        
        More efficient than calling index_memory() multiple times as it
        batches the embedding generation.
        
        Args:
            memories: List of memory objects to index
            batch_size: Batch size for embedding generation
            
        Example:
            >>> memories = [
            ...     {'memory_id': 'mem_1', 'context': 'Text 1', ...},
            ...     {'memory_id': 'mem_2', 'context': 'Text 2', ...},
            ...     {'memory_id': 'mem_3', 'context': 'Text 3', ...}
            ... ]
            >>> indexer.batch_index(memories, batch_size=32)
        """
        if not memories:
            logger.warning("No memories to index")
            return
        
        logger.info(f"Batch indexing {len(memories)} memories")
        
        # Separate memories with/without embeddings
        needs_embedding = []
        has_embedding = []
        
        for memory in memories:
            embedding = self._get_attr(memory, 'embedding')
            if embedding is None:
                needs_embedding.append(memory)
            else:
                has_embedding.append(memory)
        
        # Generate embeddings in batches
        if needs_embedding:
            logger.debug(
                f"Generating embeddings for {len(needs_embedding)} memories"
            )
            texts = [
                self._extract_indexable_text(m)
                for m in needs_embedding
            ]
            embeddings = self.embedder.encode_batch(
                texts,
                batch_size=batch_size,
                normalize=True
            )
            
            # Update memories with embeddings
            for memory, embedding in zip(needs_embedding, embeddings):
                self._set_attr(memory, 'embedding', embedding)
        
        # Add all to index
        all_memories = needs_embedding + has_embedding
        self.index_db.batch_add(all_memories)
        
        logger.info(f"Batch indexed {len(all_memories)} memories")
    
    def rebuild_index(self, category: str) -> None:
        """
        Rebuild index for a category from scratch.
        
        Loads all memories from storage, clears existing index,
        and re-indexes everything.
        
        Args:
            category: Category to rebuild
            
        Example:
            >>> indexer.rebuild_index('knowledge')
        """
        logger.info(f"Rebuilding index for category: {category}")
        
        try:
            # Load all memories from storage
            memories = self.storage.load_all(category)
            logger.info(f"Loaded {len(memories)} memories from storage")
            
            # Clear existing index
            self.index_db.clear_category(category)
            logger.info(f"Cleared existing index for {category}")
            
            # Batch index all memories
            self.batch_index(memories)
            
            logger.info(
                f"Rebuilt index for {category}: {len(memories)} memories"
            )
            
        except Exception as e:
            logger.error(f"Failed to rebuild index for {category}: {e}")
            raise
    
    def _extract_indexable_text(self, memory: Any) -> str:
        """
        Extract text for indexing from memory.
        
        Args:
            memory: Memory object
            
        Returns:
            Text to index
        """
        # Check if memory is compressed
        is_compressed = self._get_attr(memory, 'is_compressed', False)
        
        if is_compressed:
            # Use summary for compressed memories
            text = self._get_attr(memory, 'summary', '')
        else:
            # Use full context for uncompressed
            text = self._get_attr(memory, 'context', '')
        
        # Fallback to any available text field
        if not text:
            text = (
                self._get_attr(memory, 'text', '') or
                self._get_attr(memory, 'content', '') or
                self._get_attr(memory, 'data', '')
            )
        
        if not text:
            logger.warning(
                f"No indexable text found for memory: "
                f"{self._get_attr(memory, 'memory_id', 'unknown')}"
            )
            text = ""
        
        return text
    
    def _get_attr(
        self,
        obj: Any,
        attr: str,
        default: Any = None
    ) -> Any:
        """
        Get attribute from object (supports both dict and object).
        
        Args:
            obj: Object or dict
            attr: Attribute name
            default: Default value if not found
            
        Returns:
            Attribute value or default
        """
        if isinstance(obj, dict):
            return obj.get(attr, default)
        else:
            return getattr(obj, attr, default)
    
    def _set_attr(
        self,
        obj: Any,
        attr: str,
        value: Any
    ) -> None:
        """
        Set attribute on object (supports both dict and object).
        
        Args:
            obj: Object or dict
            attr: Attribute name
            value: Value to set
        """
        if isinstance(obj, dict):
            obj[attr] = value
        else:
            setattr(obj, attr, value)

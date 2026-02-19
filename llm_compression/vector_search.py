"""
Semantic vector search engine for AI-OS memory system.

This module provides fast similarity search using ArrowEngine embeddings
with Arrow-native data flow for zero-copy operations.
"""

from dataclasses import dataclass
from typing import List, Optional, Any

import numpy as np

from llm_compression.embedding_provider import EmbeddingProvider
from llm_compression.arrow_storage import ArrowStorage
from llm_compression.logger import logger


# Import SemanticIndexDB for similarity search
try:
    from llm_compression.semantic_index_db import SemanticIndexDB
except ImportError:
    SemanticIndexDB = None


@dataclass
class SearchResult:
    """
    Search result containing memory and similarity score.
    
    Attributes:
        memory_id: Unique identifier of the memory
        similarity: Cosine similarity score (0-1)
        memory: The actual memory object
    """
    
    memory_id: str
    similarity: float
    memory: Any  # CompressedMemory or dict


class VectorSearch:
    """
    Semantic search engine using ArrowEngine embeddings.
    
    Features:
    - Fast similarity search using vectorized NumPy operations
    - Arrow-native data flow (zero-copy)
    - Configurable similarity thresholds
    - Top-k retrieval
    
    Example:
        >>> from llm_compression.embedding_provider import get_default_provider
        >>> from llm_compression.arrow_storage import ArrowStorage
        >>> 
        >>> provider = get_default_provider()
        >>> storage = ArrowStorage("./data/memories.parquet")
        >>> search = VectorSearch(provider, storage)
        >>> 
        >>> results = search.search(
        ...     query="machine learning concepts",
        ...     category="knowledge",
        ...     top_k=5
        ... )
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        storage: ArrowStorage,
        index_db: Optional['SemanticIndexDB'] = None
    ):
        """
        Initialize VectorSearch.
        
        Args:
            embedding_provider: Provider for generating embeddings
            storage: Arrow storage backend for memories
            index_db: Optional semantic index database for fast similarity search
        """
        self.embedder = embedding_provider
        self.storage = storage
        self.index_db = index_db
        logger.info(
            f"Initialized VectorSearch with "
            f"embedding_dim={embedding_provider.get_embedding_dimension()}"
        )
    
    def search(
        self,
        query: str,
        category: str,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Semantic search for memories.
        
        Args:
            query: Search query text
            category: Memory category to search
            top_k: Number of results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of search results, sorted by similarity (descending)
            
        Example:
            >>> results = search.search(
            ...     query="Python programming",
            ...     category="code",
            ...     top_k=5,
            ...     threshold=0.7
            ... )
            >>> for r in results:
            ...     print(f"{r.memory_id}: {r.similarity:.3f}")
        """
        logger.debug(
            f"Searching: query='{query[:50]}...', "
            f"category={category}, top_k={top_k}, threshold={threshold}"
        )
        
        # Encode query
        query_embedding = self.embedder.encode(query, normalize=True)
        
        # Ensure query_embedding is 1D
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]
        
        # Query index_db if available, otherwise fall back to storage
        if self.index_db is not None:
            results = self.index_db.query(
                category=category,
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=threshold
            )
        else:
            results = self.storage.query_by_similarity(
                category=category,
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=threshold
            )
        
        # Convert to SearchResult objects
        search_results = []
        for r in results:
            if isinstance(r, tuple):
                # ArrowStorage returns (CompressedMemory, similarity)
                memory_obj = r[0]
                similarity = r[1]
                mid = memory_obj.memory_id
            else:
                # index_db or other might return dict
                mid = r['memory_id']
                similarity = float(r['similarity'])
                memory_obj = self._load_memory(mid, category)
                
            search_results.append(SearchResult(
                memory_id=mid,
                similarity=similarity,
                memory=memory_obj
            ))
        
        logger.debug(f"Found {len(search_results)} results")
        return search_results
    
    def batch_search(
        self,
        queries: List[str],
        category: str,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[List[SearchResult]]:
        """
        Batch search for multiple queries.
        
        More efficient than calling search() multiple times as it
        batches the embedding generation.
        
        Args:
            queries: List of search query texts
            category: Memory category to search
            top_k: Number of results per query
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of result lists, one per query
            
        Example:
            >>> queries = [
            ...     "machine learning",
            ...     "deep learning",
            ...     "neural networks"
            ... ]
            >>> batch_results = search.batch_search(
            ...     queries=queries,
            ...     category="knowledge",
            ...     top_k=3
            ... )
            >>> for i, results in enumerate(batch_results):
            ...     print(f"Query {i}: {len(results)} results")
        """
        logger.debug(
            f"Batch searching: {len(queries)} queries, "
            f"category={category}, top_k={top_k}"
        )
        
        # Batch encode queries
        query_embeddings = self.embedder.encode_batch(
            queries,
            normalize=True
        )
        
        # Search for each query
        all_results = []
        for i, query_emb in enumerate(query_embeddings):
            # Query index_db if available, otherwise fall back to storage
            if self.index_db is not None:
                results = self.index_db.query(
                    category=category,
                    query_embedding=query_emb,
                    top_k=top_k,
                    threshold=threshold
                )
            else:
                results = self.storage.query_by_similarity(
                    category=category,
                    query_embedding=query_emb,
                    top_k=top_k,
                    threshold=threshold
                )
            
            search_results = []
            for r in results:
                if isinstance(r, tuple):
                    memory_obj = r[0]
                    similarity = r[1]
                    mid = memory_obj.memory_id
                else:
                    mid = r['memory_id']
                    similarity = float(r['similarity'])
                    memory_obj = self._load_memory(mid, category)
                    
                search_results.append(SearchResult(
                    memory_id=mid,
                    similarity=similarity,
                    memory=memory_obj
                ))
            all_results.append(search_results)
        
        logger.debug(
            f"Batch search complete: "
            f"{sum(len(r) for r in all_results)} total results"
        )
        return all_results
    
    def _load_memory(
        self,
        memory_id: str,
        category: str
    ) -> Optional[Any]:
        """
        Load memory object from storage.
        
        Args:
            memory_id: Memory identifier
            category: Memory category
            
        Returns:
            Memory object or None if not found
        """
        try:
            # Query storage for the specific memory
            memories = self.storage.query(
                category=category,
                filters={'memory_id': memory_id},
                limit=1
            )
            
            if memories:
                return memories[0]
            else:
                logger.warning(f"Memory not found: {memory_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load memory {memory_id}: {e}")
            return None

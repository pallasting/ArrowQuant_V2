"""
EmbeddingTool for AI-OS Integration

Provides a high-level tool interface for text embedding generation
using the ArrowEngine service. Designed for AI-OS memory compression
and LLM workflows.

Features:
- Automatic batching for large text collections
- Semantic caching for repeated queries
- Configuration management
- Error handling and retries
- Integration with AI-OS memory system
"""

import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np

from llm_compression.client import ArrowEngineClient, ArrowEngineClientError


@dataclass
class EmbeddingConfig:
    """Configuration for EmbeddingTool"""
    endpoint: str = "http://localhost:8000"
    timeout: float = 30.0
    max_retries: int = 3
    batch_size: int = 32
    normalize: bool = False
    enable_cache: bool = True
    cache_size: int = 1000


@dataclass
class EmbeddingToolResult:
    """Result from EmbeddingTool operations"""
    embeddings: np.ndarray
    texts: List[str]
    dimension: int
    cache_hits: int = 0
    cache_misses: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingTool:
    """
    AI-OS tool for text embedding generation
    
    Wraps the ArrowEngine HTTP client and provides:
    - Automatic batching for large text collections
    - LRU caching for frequently used texts
    - Configuration management
    - Error handling with graceful degradation
    
    Usage:
        from llm_compression.tools import EmbeddingTool
        
        tool = EmbeddingTool(endpoint="http://localhost:8000")
        result = tool.embed(["Hello, world!", "AI is amazing!"])
        print(f"Embeddings shape: {result.embeddings.shape}")
        print(f"Cache hits: {result.cache_hits}")
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None
    ):
        """
        Initialize EmbeddingTool
        
        Args:
            endpoint: ArrowEngine service endpoint (overrides config)
            config: EmbeddingConfig object (uses defaults if None)
        """
        if config is None:
            config = EmbeddingConfig()
        
        if endpoint is not None:
            config.endpoint = endpoint
        
        self.config = config
        self.client = ArrowEngineClient(
            base_url=config.endpoint,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
    
    def embed(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None
    ) -> EmbeddingToolResult:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of texts to embed
            batch_size: Override default batch size
            normalize: Override default normalization
            
        Returns:
            EmbeddingToolResult with embeddings and statistics
            
        Raises:
            ArrowEngineClientError: If embedding generation fails
        """
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        batch_size = batch_size or self.config.batch_size
        normalize = normalize if normalize is not None else self.config.normalize
        
        all_embeddings = []
        cache_hits = 0
        cache_misses = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if self.config.enable_cache:
                batch_embeddings, hits, misses = self._embed_with_cache(
                    batch, normalize
                )
                cache_hits += hits
                cache_misses += misses
            else:
                batch_embeddings = self._embed_batch(batch, normalize)
            
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        return EmbeddingToolResult(
            embeddings=embeddings,
            texts=texts,
            dimension=embeddings.shape[1],
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            metadata={
                "batch_size": batch_size,
                "normalize": normalize,
                "total_batches": len(all_embeddings)
            }
        )
    
    def similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (-1 to 1)
        """
        result = self.client.similarity(text1=text1, text2=text2)
        if result.similarity is None:
            raise ValueError("Expected similarity score, got None")
        return result.similarity
    
    def similarity_matrix(
        self,
        texts1: List[str],
        texts2: List[str]
    ) -> np.ndarray:
        """
        Compute similarity matrix between two text lists
        
        Args:
            texts1: First list of texts
            texts2: Second list of texts
            
        Returns:
            Similarity matrix (len(texts1) x len(texts2))
        """
        emb1 = self.embed(texts1).embeddings
        emb2 = self.embed(texts2).embeddings
        
        emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
        emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
        
        return np.dot(emb1_norm, emb2_norm.T)
    
    def health_check(self) -> bool:
        """
        Check if the service is healthy
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            result = self.client.health()
            return result.status == "healthy" and result.model_loaded
        except Exception:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model metadata
        """
        info = self.client.info()
        return {
            "model_name": info.model_name,
            "embedding_dimension": info.embedding_dimension,
            "max_seq_length": info.max_seq_length,
            "version": info.version,
            "device": info.device,
            "endpoint": self.config.endpoint
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        self._cache.clear()
        self._cache_order.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache size and capacity
        """
        return {
            "size": len(self._cache),
            "capacity": self.config.cache_size
        }
    
    def _embed_with_cache(
        self,
        texts: List[str],
        normalize: bool
    ) -> tuple[np.ndarray, int, int]:
        """Embed texts with caching"""
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        hits = 0
        misses = 0
        
        for idx, text in enumerate(texts):
            cache_key = self._get_cache_key(text, normalize)
            
            if cache_key in self._cache:
                cached_embeddings.append((idx, self._cache[cache_key]))
                hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(idx)
                misses += 1
        
        if uncached_texts:
            new_embeddings = self._embed_batch(uncached_texts, normalize)
            
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text, normalize)
                self._add_to_cache(cache_key, embedding)
        else:
            new_embeddings = np.array([])
        
        result = np.zeros((len(texts), cached_embeddings[0][1].shape[0] if cached_embeddings else new_embeddings.shape[1]))
        
        for idx, embedding in cached_embeddings:
            result[idx] = embedding
        
        for i, idx in enumerate(uncached_indices):
            result[idx] = new_embeddings[i]
        
        return result, hits, misses
    
    def _embed_batch(
        self,
        texts: List[str],
        normalize: bool
    ) -> np.ndarray:
        """Embed a batch of texts without caching"""
        result = self.client.embed(texts, normalize=normalize)
        return np.array(result.embeddings)
    
    def _get_cache_key(self, text: str, normalize: bool) -> str:
        """Generate cache key for text"""
        content = f"{text}|{normalize}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, embedding: np.ndarray) -> None:
        """Add embedding to cache with LRU eviction"""
        if key in self._cache:
            self._cache_order.remove(key)
        
        self._cache[key] = embedding
        self._cache_order.append(key)
        
        while len(self._cache) > self.config.cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]
    
    def close(self) -> None:
        """Close the client connection"""
        self.client.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

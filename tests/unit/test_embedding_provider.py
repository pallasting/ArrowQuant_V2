"""
Unit tests for EmbeddingProvider interface

Tests Property 15: Provider API Compatibility
Validates: Requirements 4.4

Feature: arrowengine-core-implementation
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import pytest

from llm_compression.embedding_provider import (
    EmbeddingProvider,
    ArrowEngineProvider,
    SentenceTransformerProvider,
    LocalEmbedderProvider,
    get_default_provider,
    reset_provider,
)


# ============================================================================
# Test Configuration
# ============================================================================

MODEL_PATH = "./models/minilm"

# Skip ArrowEngine tests if model not available
ARROW_MODEL_AVAILABLE = (
    Path(MODEL_PATH).exists() 
    and Path(MODEL_PATH, "metadata.json").exists()
)

# Check if sentence-transformers is available
try:
    import sentence_transformers
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# ============================================================================
# Property 15: Provider API Compatibility
# Validates: Requirements 4.4
# ============================================================================

class TestEmbeddingProviderInterface:
    """
    Test that all EmbeddingProvider implementations have compatible APIs.
    
    Property 15: Provider API Compatibility
    For any EmbeddingProvider implementation (ArrowEngine or SentenceTransformer),
    all required methods should be callable with the same signatures and return
    compatible types.
    """
    
    @pytest.fixture(params=[
        pytest.param(
            "arrow",
            marks=pytest.mark.skipif(
                not ARROW_MODEL_AVAILABLE,
                reason="Arrow model not available"
            )
        ),
        pytest.param(
            "sentence_transformers",
            marks=pytest.mark.skipif(
                not SENTENCE_TRANSFORMERS_AVAILABLE,
                reason="sentence-transformers not available"
            )
        ),
        "local_embedder",
    ])
    def provider(self, request) -> EmbeddingProvider:
        """Fixture that provides different EmbeddingProvider implementations."""
        provider_type = request.param
        
        if provider_type == "arrow":
            return ArrowEngineProvider(MODEL_PATH, device='cpu')
        elif provider_type == "sentence_transformers":
            return SentenceTransformerProvider("all-MiniLM-L6-v2", device='cpu')
        elif provider_type == "local_embedder":
            return LocalEmbedderProvider()
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    def test_has_dimension_property(self, provider: EmbeddingProvider):
        """Test that provider has dimension property."""
        dim = provider.dimension
        assert isinstance(dim, int)
        assert dim > 0
        assert dim == 384  # Expected for all-MiniLM-L6-v2
    
    def test_has_get_embedding_dimension_method(self, provider: EmbeddingProvider):
        """Test that provider has get_embedding_dimension method (legacy compatibility)."""
        dim = provider.get_embedding_dimension()
        assert isinstance(dim, int)
        assert dim > 0
        assert dim == provider.dimension
    
    def test_encode_single_text(self, provider: EmbeddingProvider):
        """Test encode method with single text."""
        text = "Hello, world!"
        embedding = provider.encode(text, normalize=True)
        
        # Check return type
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        
        # Check shape
        assert embedding.shape == (provider.dimension,)
        
        # Check normalization (L2 norm should be ~1.0)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5
    
    def test_encode_empty_text(self, provider: EmbeddingProvider):
        """Test encode method with empty text."""
        embedding = provider.encode("", normalize=True)
        
        # Should return zero vector
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (provider.dimension,)
        assert np.allclose(embedding, 0.0)
    
    def test_encode_without_normalization(self, provider: EmbeddingProvider):
        """Test encode method without normalization."""
        text = "Hello, world!"
        embedding = provider.encode(text, normalize=False)
        
        # Check return type
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert embedding.shape == (provider.dimension,)
        
        # Norm should NOT be 1.0 (unless by chance)
        norm = np.linalg.norm(embedding)
        # Just check it's a reasonable value (not zero, not too large)
        assert 0.1 < norm < 100.0
    
    def test_encode_batch(self, provider: EmbeddingProvider):
        """Test encode_batch method."""
        texts = ["Hello", "World", "Test sentence"]
        embeddings = provider.encode_batch(texts, batch_size=2, normalize=True)
        
        # Check return type
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32
        
        # Check shape
        assert embeddings.shape == (len(texts), provider.dimension)
        
        # Check normalization for each embedding
        for i in range(len(texts)):
            norm = np.linalg.norm(embeddings[i])
            assert abs(norm - 1.0) < 1e-5
    
    def test_encode_batch_empty_list(self, provider: EmbeddingProvider):
        """Test encode_batch with empty list."""
        embeddings = provider.encode_batch([], normalize=True)
        
        # Should return empty array with correct shape
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, provider.dimension)
    
    def test_encode_batch_single_item(self, provider: EmbeddingProvider):
        """Test encode_batch with single item."""
        texts = ["Single text"]
        embeddings = provider.encode_batch(texts, normalize=True)
        
        assert embeddings.shape == (1, provider.dimension)
    
    def test_similarity_cosine(self, provider: EmbeddingProvider):
        """Test similarity method with cosine similarity."""
        vec1 = np.random.randn(provider.dimension).astype(np.float32)
        vec2 = np.random.randn(provider.dimension).astype(np.float32)
        
        # Normalize vectors
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        similarity = provider.similarity(vec1, vec2, method="cosine")
        
        # Check return type
        assert isinstance(similarity, float)
        
        # Cosine similarity should be in [-1, 1]
        assert -1.0 <= similarity <= 1.0
    
    def test_similarity_dot(self, provider: EmbeddingProvider):
        """Test similarity method with dot product."""
        vec1 = np.random.randn(provider.dimension).astype(np.float32)
        vec2 = np.random.randn(provider.dimension).astype(np.float32)
        
        similarity = provider.similarity(vec1, vec2, method="dot")
        
        # Check return type
        assert isinstance(similarity, float)
        
        # Verify it matches manual dot product
        expected = float(np.dot(vec1, vec2))
        assert abs(similarity - expected) < 1e-5
    
    def test_similarity_euclidean(self, provider: EmbeddingProvider):
        """Test similarity method with euclidean distance."""
        vec1 = np.random.randn(provider.dimension).astype(np.float32)
        vec2 = np.random.randn(provider.dimension).astype(np.float32)
        
        similarity = provider.similarity(vec1, vec2, method="euclidean")
        
        # Check return type
        assert isinstance(similarity, float)
        
        # Euclidean similarity should be in (0, 1]
        assert 0.0 < similarity <= 1.0
    
    def test_similarity_matrix_pairwise(self, provider: EmbeddingProvider):
        """Test similarity_matrix for pairwise similarities."""
        n = 5
        vectors = np.random.randn(n, provider.dimension).astype(np.float32)
        
        # Normalize vectors
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        sim_matrix = provider.similarity_matrix(vectors)
        
        # Check shape
        assert sim_matrix.shape == (n, n)
        
        # Diagonal should be ~1.0 (self-similarity)
        for i in range(n):
            assert abs(sim_matrix[i, i] - 1.0) < 1e-5
        
        # Matrix should be symmetric
        assert np.allclose(sim_matrix, sim_matrix.T, atol=1e-5)
    
    def test_similarity_matrix_query(self, provider: EmbeddingProvider):
        """Test similarity_matrix with query vector."""
        n = 5
        vectors = np.random.randn(n, provider.dimension).astype(np.float32)
        query = np.random.randn(provider.dimension).astype(np.float32)
        
        # Normalize
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        query = query / np.linalg.norm(query)
        
        similarities = provider.similarity_matrix(vectors, query)
        
        # Check shape
        assert similarities.shape == (n,)
        
        # Verify against manual computation
        expected = np.dot(vectors, query)
        assert np.allclose(similarities, expected, atol=1e-5)


class TestGetDefaultProvider:
    """Test the get_default_provider factory function."""
    
    def setup_method(self):
        """Reset provider singleton before each test."""
        reset_provider()
    
    def teardown_method(self):
        """Reset provider singleton after each test."""
        reset_provider()
    
    @pytest.mark.skipif(
        not ARROW_MODEL_AVAILABLE,
        reason="Arrow model not available"
    )
    def test_returns_arrow_provider_when_available(self):
        """Test that ArrowEngineProvider is returned when model is available."""
        provider = get_default_provider(model_path=MODEL_PATH)
        
        assert isinstance(provider, ArrowEngineProvider)
    
    def test_returns_local_embedder_when_arrow_unavailable(self):
        """Test fallback to LocalEmbedderProvider when Arrow model unavailable."""
        # Use non-existent path
        provider = get_default_provider(model_path="/nonexistent/path")
        
        assert isinstance(provider, LocalEmbedderProvider)
    
    @pytest.mark.skipif(
        not ARROW_MODEL_AVAILABLE,
        reason="Arrow model not available"
    )
    def test_force_arrow_succeeds_when_available(self):
        """Test force_arrow=True succeeds when model is available."""
        provider = get_default_provider(model_path=MODEL_PATH, force_arrow=True)
        
        assert isinstance(provider, ArrowEngineProvider)
    
    def test_force_arrow_raises_when_unavailable(self):
        """Test force_arrow=True raises exception when model unavailable."""
        with pytest.raises(RuntimeError, match="Arrow model not found"):
            get_default_provider(model_path="/nonexistent/path", force_arrow=True)
    
    def test_force_st_returns_local_embedder(self):
        """Test force_st=True returns LocalEmbedderProvider."""
        provider = get_default_provider(force_st=True)
        
        assert isinstance(provider, LocalEmbedderProvider)
    
    def test_singleton_behavior(self):
        """Test that get_default_provider returns singleton."""
        provider1 = get_default_provider()
        provider2 = get_default_provider()
        
        # Should be the same instance
        assert provider1 is provider2
    
    def test_reset_provider_clears_singleton(self):
        """Test that reset_provider clears the singleton."""
        provider1 = get_default_provider()
        reset_provider()
        provider2 = get_default_provider()
        
        # Should be different instances
        assert provider1 is not provider2


class TestProviderConsistency:
    """
    Test that different providers produce consistent results.
    
    This validates that swapping providers doesn't break downstream code.
    """
    
    @pytest.mark.skipif(
        not (ARROW_MODEL_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE),
        reason="Both Arrow and SentenceTransformers needed"
    )
    def test_arrow_vs_sentence_transformers_consistency(self):
        """Test that ArrowEngine and SentenceTransformers produce similar embeddings."""
        arrow_provider = ArrowEngineProvider(MODEL_PATH, device='cpu')
        st_provider = SentenceTransformerProvider("all-MiniLM-L6-v2", device='cpu')
        
        text = "This is a test sentence for consistency checking."
        
        arrow_emb = arrow_provider.encode(text, normalize=True)
        st_emb = st_provider.encode(text, normalize=True)
        
        # Compute similarity
        similarity = np.dot(arrow_emb, st_emb)
        
        # Should be very similar (≥ 0.99)
        assert similarity >= 0.99, (
            f"Embeddings differ too much: similarity={similarity:.4f}"
        )
    
    def test_encode_vs_encode_batch_consistency(self):
        """Test that encode and encode_batch produce same results."""
        provider = get_default_provider()
        
        texts = ["Hello", "World", "Test"]
        
        # Encode individually
        individual_embs = np.array([
            provider.encode(text, normalize=True) for text in texts
        ])
        
        # Encode as batch
        batch_embs = provider.encode_batch(texts, normalize=True)
        
        # Should be identical
        assert np.allclose(individual_embs, batch_embs, atol=1e-5)


class TestArrowEngineProvider:
    """Specific tests for ArrowEngineProvider."""
    
    @pytest.mark.skipif(
        not ARROW_MODEL_AVAILABLE,
        reason="Arrow model not available"
    )
    def test_initialization(self):
        """Test ArrowEngineProvider initialization."""
        provider = ArrowEngineProvider(MODEL_PATH, device='cpu')
        
        assert provider.dimension == 384
        assert provider._normalize is True
    
    @pytest.mark.skipif(
        not ARROW_MODEL_AVAILABLE,
        reason="Arrow model not available"
    )
    def test_repr(self):
        """Test string representation."""
        provider = ArrowEngineProvider(MODEL_PATH, device='cpu')
        repr_str = repr(provider)
        
        assert "ArrowEngineProvider" in repr_str
        assert "dim=384" in repr_str
        assert "device=" in repr_str


class TestSentenceTransformerProvider:
    """Specific tests for SentenceTransformerProvider."""
    
    @pytest.mark.skipif(
        not SENTENCE_TRANSFORMERS_AVAILABLE,
        reason="sentence-transformers not available"
    )
    def test_initialization_with_deprecation_warning(self):
        """Test that initialization emits deprecation warning."""
        with pytest.warns(DeprecationWarning, match="fallback"):
            provider = SentenceTransformerProvider("all-MiniLM-L6-v2", device='cpu')
        
        assert provider.dimension == 384
    
    @pytest.mark.skipif(
        not SENTENCE_TRANSFORMERS_AVAILABLE,
        reason="sentence-transformers not available"
    )
    def test_repr(self):
        """Test string representation."""
        with pytest.warns(DeprecationWarning):
            provider = SentenceTransformerProvider("all-MiniLM-L6-v2", device='cpu')
        
        repr_str = repr(provider)
        
        assert "SentenceTransformerProvider" in repr_str
        assert "model=all-MiniLM-L6-v2" in repr_str
        assert "dim=384" in repr_str


class TestLocalEmbedderProvider:
    """Specific tests for LocalEmbedderProvider."""
    
    def test_initialization_with_default_embedder(self):
        """Test initialization with default LocalEmbedder."""
        provider = LocalEmbedderProvider()
        
        assert provider.dimension > 0
    
    def test_repr(self):
        """Test string representation."""
        provider = LocalEmbedderProvider()
        repr_str = repr(provider)
        
        assert "LocalEmbedderProvider" in repr_str
        assert "embedder=" in repr_str

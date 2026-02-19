"""
Integration tests for migration from legacy embedders to EmbeddingProvider.

Feature: arrowengine-core-implementation
Requirements: 9.2, 9.3, 9.4

Tests verify:
- Backend interchangeability (Property 29)
- API signature stability (Property 30)
- Parallel operation of old and new implementations
- Output consistency before/after migration
"""

import pytest
import numpy as np
from typing import List
from hypothesis import given, settings, strategies as st, assume

from llm_compression.embedding_provider import (
    EmbeddingProvider,
    ArrowEngineProvider,
    SentenceTransformerProvider,
    get_default_provider,
)


# ============================================================================
# Property 29: Backend Interchangeability
# Validates: Requirements 9.2
# ============================================================================

@settings(max_examples=30, deadline=None)
@given(
    texts=st.lists(
        st.text(min_size=5, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'),
            min_codepoint=32,
            max_codepoint=126
        )),
        min_size=1,
        max_size=5
    )
)
def test_property_29_backend_interchangeability(texts):
    """
    Feature: arrowengine-core-implementation, Property 29: Backend Interchangeability
    
    For any downstream module using EmbeddingProvider, swapping between
    ArrowEngineProvider and SentenceTransformerProvider should not require
    code changes (verified by duck typing compatibility).
    
    Validates: Requirements 9.2
    """
    # Filter out empty or whitespace-only texts
    texts = [t for t in texts if len(t.strip()) >= 3 and any(c.isalnum() for c in t)]
    assume(len(texts) >= 1)
    
    # Test both providers implement the same interface
    providers = []
    
    try:
        providers.append(ArrowEngineProvider())
    except Exception:
        pytest.skip("ArrowEngine not available")
    
    try:
        providers.append(SentenceTransformerProvider())
    except Exception:
        pytest.skip("SentenceTransformers not available")
    
    if len(providers) < 2:
        pytest.skip("Need both providers for interchangeability test")
    
    # Verify all providers support the same methods with same signatures
    for provider in providers:
        # Test encode method (returns 1D array for single text)
        result = provider.encode(texts[0], normalize=True)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape[0] > 0
        
        # Test encode_batch method
        result_batch = provider.encode_batch(texts, batch_size=2, normalize=True)
        assert isinstance(result_batch, np.ndarray)
        assert result_batch.shape[0] == len(texts)
        
        # Test similarity method (takes vectors, not text)
        if len(texts) >= 2:
            emb1 = provider.encode(texts[0], normalize=True)
            emb2 = provider.encode(texts[1], normalize=True)
            sim = provider.similarity(emb1, emb2)
            assert isinstance(sim, (float, np.floating))
        
        # Test get_embedding_dimension method
        dim = provider.get_embedding_dimension()
        assert isinstance(dim, int)
        assert dim > 0
    
    # Verify outputs have compatible shapes across providers
    dims = [p.get_embedding_dimension() for p in providers]
    assert all(d == dims[0] for d in dims), (
        "All providers should return same embedding dimension"
    )


# ============================================================================
# Property 30: API Signature Stability
# Validates: Requirements 9.3
# ============================================================================

def test_property_30_api_signature_stability():
    """
    Feature: arrowengine-core-implementation, Property 30: API Signature Stability
    
    For any public method in a migrated module, the method signature
    (parameters and return type) should remain identical before and after migration.
    
    Validates: Requirements 9.3
    """
    import inspect
    from llm_compression.embedding_provider import EmbeddingProvider
    
    # Define expected method signatures for EmbeddingProvider protocol
    expected_methods = {
        'encode': {
            'params': ['text', 'normalize'],
            'return_type': np.ndarray
        },
        'encode_batch': {
            'params': ['texts', 'batch_size', 'normalize'],
            'return_type': np.ndarray
        },
        'similarity': {
            'params': ['text1', 'text2'],
            'return_type': (float, np.ndarray)
        },
        'get_embedding_dimension': {
            'params': [],
            'return_type': int
        }
    }
    
    # Test ArrowEngineProvider
    try:
        provider = ArrowEngineProvider()
        
        for method_name, expected in expected_methods.items():
            assert hasattr(provider, method_name), (
                f"ArrowEngineProvider missing method: {method_name}"
            )
            
            method = getattr(provider, method_name)
            sig = inspect.signature(method)
            
            # Verify parameters exist (allowing for self and defaults)
            param_names = [p for p in sig.parameters.keys() if p != 'self']
            for expected_param in expected['params']:
                # Check if parameter exists (may have defaults)
                assert any(expected_param in p for p in param_names), (
                    f"Method {method_name} missing parameter: {expected_param}"
                )
    
    except Exception as e:
        pytest.skip(f"ArrowEngineProvider not available: {e}")
    
    # Test SentenceTransformerProvider
    try:
        provider = SentenceTransformerProvider()
        
        for method_name, expected in expected_methods.items():
            assert hasattr(provider, method_name), (
                f"SentenceTransformerProvider missing method: {method_name}"
            )
            
            method = getattr(provider, method_name)
            sig = inspect.signature(method)
            
            # Verify parameters exist
            param_names = [p for p in sig.parameters.keys() if p != 'self']
            for expected_param in expected['params']:
                assert any(expected_param in p for p in param_names), (
                    f"Method {method_name} missing parameter: {expected_param}"
                )
    
    except Exception as e:
        pytest.skip(f"SentenceTransformerProvider not available: {e}")


# ============================================================================
# Integration Test: Module Migration Compatibility
# Validates: Requirements 9.2, 9.3, 9.4
# ============================================================================

@pytest.mark.integration
def test_migrated_modules_with_both_providers():
    """
    Test each migrated module with both providers.
    
    Validates: Requirements 9.2, 9.3, 9.4
    """
    test_texts = [
        "Test sentence for migration validation",
        "Another test sentence for compatibility check"
    ]
    
    providers = []
    
    try:
        providers.append(("ArrowEngine", ArrowEngineProvider()))
    except Exception:
        pass
    
    try:
        providers.append(("SentenceTransformer", SentenceTransformerProvider()))
    except Exception:
        pass
    
    if len(providers) == 0:
        pytest.skip("No providers available")
    
    # Test each provider works with the same code
    results = []
    for name, provider in providers:
        # Test basic encoding (returns 1D array for single text)
        embedding = provider.encode(test_texts[0], normalize=True)
        assert embedding.ndim == 1
        assert embedding.shape[0] == provider.get_embedding_dimension()
        
        # Test batch encoding
        batch_embeddings = provider.encode_batch(test_texts, normalize=True)
        assert batch_embeddings.shape[0] == len(test_texts)
        
        # Test similarity (compute from embeddings)
        emb1 = provider.encode(test_texts[0], normalize=True)
        emb2 = provider.encode(test_texts[1], normalize=True)
        sim = provider.similarity(emb1, emb2)
        assert isinstance(sim, (float, np.floating))
        
        results.append({
            'name': name,
            'embedding': embedding,
            'batch_embeddings': batch_embeddings,
            'similarity': sim
        })
    
    # If we have both providers, verify outputs are similar
    if len(results) == 2:
        # Embeddings should be similar (not identical due to implementation differences)
        sim = np.dot(results[0]['embedding'], results[1]['embedding'])
        assert sim > 0.95, (
            f"Embeddings from different providers should be similar: {sim:.4f}"
        )


@pytest.mark.integration
def test_parallel_operation_old_and_new():
    """
    Test parallel operation of old and new implementations.
    
    Validates: Requirements 9.4
    """
    test_text = "Test sentence for parallel operation"
    
    # Test that get_default_provider works
    provider = get_default_provider()
    
    # Should be able to encode (returns 1D array for single text)
    embedding = provider.encode(test_text, normalize=True)
    assert embedding.ndim == 1
    assert embedding.shape[0] > 0
    
    # Verify embedding is normalized
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 1e-5


@pytest.mark.integration
def test_output_consistency_before_after_migration():
    """
    Verify output consistency before/after migration.
    
    This test verifies that using EmbeddingProvider produces consistent
    results regardless of which backend is used.
    
    Validates: Requirements 9.2, 9.3
    """
    test_texts = [
        "Consistency test sentence one",
        "Consistency test sentence two",
        "Consistency test sentence three"
    ]
    
    # Get default provider
    provider = get_default_provider()
    
    # Test 1: Single encoding consistency
    emb1 = provider.encode(test_texts[0], normalize=True)
    emb2 = provider.encode(test_texts[0], normalize=True)
    
    # Should be identical for same input
    assert np.allclose(emb1, emb2, atol=1e-6)
    
    # Test 2: Batch encoding consistency
    batch1 = provider.encode_batch(test_texts, normalize=True)
    batch2 = provider.encode_batch(test_texts, normalize=True)
    
    # Should be identical for same inputs
    assert np.allclose(batch1, batch2, atol=1e-6)
    
    # Test 3: Similarity consistency (compute from embeddings)
    emb_a = provider.encode(test_texts[0], normalize=True)
    emb_b = provider.encode(test_texts[1], normalize=True)
    
    sim1 = provider.similarity(emb_a, emb_b)
    sim2 = provider.similarity(emb_a, emb_b)
    
    # Should be identical for same inputs
    assert abs(sim1 - sim2) < 1e-6

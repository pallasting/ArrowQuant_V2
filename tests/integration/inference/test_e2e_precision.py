"""
End-to-end precision validation for ArrowEngine vs sentence-transformers.

This test validates that ArrowEngine produces embeddings with high similarity
to sentence-transformers, ensuring no accuracy regression when migrating.

Success Criteria:
- Per-text cosine similarity ≥ 0.99
- Average similarity ≥ 0.995
- No similarity < 0.95 (hard failure)
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from llm_compression.logger import logger


# Test texts covering diverse domains, lengths, and complexity
TEST_TEXTS = [
    # Short, simple
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world!",
    
    # Technical
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language for data science.",
    "Neural networks consist of interconnected layers of nodes.",
    
    # Conversational
    "How are you doing today?",
    "I really enjoyed that movie last night.",
    "Can you help me with this problem?",
    
    # Formal/Academic
    "Climate change poses significant challenges to global ecosystems.",
    "The research demonstrates a correlation between variables.",
    "Economic indicators suggest a potential market downturn.",
    
    # Geographic/Factual
    "The Eiffel Tower is located in Paris, France.",
    "Mount Everest is the highest mountain in the world.",
    "The Amazon River flows through South America.",
    
    # Long, complex
    "Artificial intelligence and machine learning have revolutionized "
    "numerous industries by enabling computers to learn from data and "
    "make intelligent decisions without explicit programming.",
    
    "The development of quantum computing represents a paradigm shift "
    "in computational capabilities, potentially solving problems that "
    "are intractable for classical computers.",
    
    # Edge cases
    "A",  # Very short
    "The " * 50,  # Repetitive
    "🎉 Emoji test 🚀 with symbols!",  # Special characters
    "",  # Empty (will be handled)
]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity in range [-1, 1]
    """
    # Handle zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


@pytest.fixture(scope="module")
def arrow_engine():
    """Load ArrowEngine (requires converted model)."""
    from llm_compression.inference.arrow_engine import ArrowEngine
    
    model_path = os.environ.get("ARROW_MODEL_PATH", "./models/minilm")
    
    if not Path(model_path).exists():
        pytest.skip(
            f"ArrowEngine model not found at {model_path}. "
            "Please convert the model first:\n"
            "python -m llm_compression.tools.cli convert "
            "--model sentence-transformers/all-MiniLM-L6-v2 "
            f"--output {model_path} --float16 --validate"
        )
    
    try:
        engine = ArrowEngine(model_path)
        logger.info(f"Loaded ArrowEngine from {model_path}")
        return engine
    except Exception as e:
        pytest.skip(f"Failed to load ArrowEngine: {e}")


@pytest.fixture(scope="module")
def sentence_transformer():
    """Load sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded sentence-transformers model")
        return model
    except ImportError:
        pytest.skip("sentence-transformers not installed")
    except Exception as e:
        pytest.skip(f"Failed to load sentence-transformers: {e}")


class TestE2EPrecision:
    """End-to-end precision validation tests."""
    
    def test_arrowengine_vs_sentence_transformers(
        self,
        arrow_engine,
        sentence_transformer
    ):
        """
        Validate ArrowEngine embeddings match sentence-transformers.
        
        This is the core validation test ensuring no accuracy regression.
        """
        # Filter out empty texts
        test_texts = [t for t in TEST_TEXTS if t.strip()]
        
        logger.info(f"Testing {len(test_texts)} texts")
        
        # Encode with ArrowEngine
        logger.info("Encoding with ArrowEngine...")
        arrow_embeddings = arrow_engine.encode(test_texts, normalize=True)
        
        # Encode with sentence-transformers
        logger.info("Encoding with sentence-transformers...")
        st_embeddings = sentence_transformer.encode(
            test_texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Validate shapes match
        assert arrow_embeddings.shape == st_embeddings.shape, (
            f"Shape mismatch: ArrowEngine {arrow_embeddings.shape} "
            f"vs sentence-transformers {st_embeddings.shape}"
        )
        
        # Compute pairwise similarities
        similarities = []
        failures = []
        warnings_list = []
        
        for i, text in enumerate(test_texts):
            sim = cosine_similarity(arrow_embeddings[i], st_embeddings[i])
            similarities.append(sim)
            
            # Log similarity
            logger.debug(f"Text {i}: similarity={sim:.6f}")
            
            # Check thresholds
            if sim < 0.95:
                failures.append((i, text[:50], sim))
            elif sim < 0.99:
                warnings_list.append((i, text[:50], sim))
        
        # Report warnings
        if warnings_list:
            logger.warning(
                f"{len(warnings_list)} texts with similarity < 0.99:"
            )
            for i, text, sim in warnings_list[:5]:  # Show first 5
                logger.warning(f"  Text {i} ({text}...): {sim:.6f}")
        
        # Report failures
        if failures:
            error_msg = f"{len(failures)} texts with similarity < 0.95:\n"
            for i, text, sim in failures:
                error_msg += f"  Text {i} ({text}...): {sim:.6f}\n"
            pytest.fail(error_msg)
        
        # Check average similarity
        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        
        logger.info(
            f"Similarity stats: avg={avg_sim:.6f}, "
            f"min={min_sim:.6f}, max={max_sim:.6f}"
        )
        
        # Assert average similarity threshold
        assert avg_sim >= 0.995, (
            f"Average similarity {avg_sim:.6f} < 0.995. "
            f"This indicates systematic accuracy issues."
        )
        
        logger.info("✅ Precision validation passed!")
    
    def test_single_vs_batch_consistency(self, arrow_engine):
        """
        Validate that single and batch encoding produce identical results.
        
        This ensures batch processing doesn't introduce errors.
        """
        test_text = "Machine learning is a subset of artificial intelligence."
        
        # Single encoding
        single_emb = arrow_engine.encode(test_text, normalize=True)
        
        # Batch encoding
        batch_emb = arrow_engine.encode([test_text], normalize=True)
        
        # Should be identical
        similarity = cosine_similarity(single_emb[0], batch_emb[0])
        
        assert similarity > 0.9999, (
            f"Single vs batch similarity {similarity:.6f} < 0.9999. "
            f"Batch processing introduces inconsistency."
        )
    
    def test_batch_size_consistency(self, arrow_engine):
        """
        Validate that different batch sizes produce identical results.
        
        This ensures batch size doesn't affect output quality.
        """
        test_texts = TEST_TEXTS[:10]  # Use first 10 texts
        test_texts = [t for t in test_texts if t.strip()]
        
        # Encode with different batch sizes
        emb_batch_1 = arrow_engine.encode(
            test_texts,
            batch_size=1,
            normalize=True
        )
        emb_batch_4 = arrow_engine.encode(
            test_texts,
            batch_size=4,
            normalize=True
        )
        emb_batch_32 = arrow_engine.encode(
            test_texts,
            batch_size=32,
            normalize=True
        )
        
        # Compare batch_1 vs batch_4
        for i in range(len(test_texts)):
            sim = cosine_similarity(emb_batch_1[i], emb_batch_4[i])
            assert sim > 0.9999, (
                f"Text {i}: batch_1 vs batch_4 similarity {sim:.6f} < 0.9999"
            )
        
        # Compare batch_1 vs batch_32
        for i in range(len(test_texts)):
            sim = cosine_similarity(emb_batch_1[i], emb_batch_32[i])
            assert sim > 0.9999, (
                f"Text {i}: batch_1 vs batch_32 similarity {sim:.6f} < 0.9999"
            )
    
    def test_normalization_produces_unit_vectors(self, arrow_engine):
        """
        Validate that normalized embeddings are unit vectors.
        
        This ensures L2 normalization works correctly.
        """
        test_texts = ["Hello, world!", "Machine learning"]
        
        embeddings = arrow_engine.encode(test_texts, normalize=True)
        
        for i, emb in enumerate(embeddings):
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 1e-5, (
                f"Text {i}: norm={norm:.6f}, expected 1.0"
            )
    
    def test_embedding_dimension(self, arrow_engine, sentence_transformer):
        """
        Validate embedding dimensions match.
        """
        arrow_dim = arrow_engine.get_embedding_dimension()
        st_dim = sentence_transformer.get_sentence_embedding_dimension()
        
        assert arrow_dim == st_dim, (
            f"Dimension mismatch: ArrowEngine {arrow_dim} "
            f"vs sentence-transformers {st_dim}"
        )
    
    def test_deterministic_output(self, arrow_engine):
        """
        Validate that encoding the same text twice produces identical results.
        
        This ensures inference is deterministic.
        """
        test_text = "Deterministic test"
        
        emb1 = arrow_engine.encode(test_text, normalize=True)
        emb2 = arrow_engine.encode(test_text, normalize=True)
        
        # Should be exactly identical (not just similar)
        assert np.allclose(emb1, emb2, atol=1e-7), (
            "Encoding is not deterministic"
        )


@pytest.mark.integration
class TestE2EEdgeCases:
    """Edge case tests for robustness."""
    
    def test_empty_string_handling(self, arrow_engine, sentence_transformer):
        """Test handling of empty strings."""
        # ArrowEngine should handle empty strings gracefully
        try:
            arrow_emb = arrow_engine.encode("", normalize=True)
            assert arrow_emb.shape[0] == 1
            assert arrow_emb.shape[1] == arrow_engine.get_embedding_dimension()
        except Exception as e:
            pytest.fail(f"ArrowEngine failed on empty string: {e}")
    
    def test_very_long_text(self, arrow_engine):
        """Test handling of text exceeding max sequence length."""
        # Create text longer than max_seq_length (512 tokens)
        long_text = "word " * 1000
        
        try:
            emb = arrow_engine.encode(long_text, normalize=True)
            assert emb.shape[0] == 1
            assert emb.shape[1] == arrow_engine.get_embedding_dimension()
        except Exception as e:
            pytest.fail(f"ArrowEngine failed on long text: {e}")
    
    def test_special_characters(self, arrow_engine, sentence_transformer):
        """Test handling of special characters and emojis."""
        special_texts = [
            "🎉 Emoji test 🚀",
            "Special chars: @#$%^&*()",
            "Unicode: 你好世界",
            "Mixed: Hello 世界 🌍",
        ]
        
        for text in special_texts:
            try:
                arrow_emb = arrow_engine.encode(text, normalize=True)
                st_emb = sentence_transformer.encode(
                    text,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                
                sim = cosine_similarity(arrow_emb[0], st_emb)
                assert sim >= 0.95, (
                    f"Special char text '{text[:30]}...' "
                    f"similarity {sim:.6f} < 0.95"
                )
            except Exception as e:
                pytest.fail(f"Failed on special text '{text}': {e}")


if __name__ == "__main__":
    # Allow running directly for quick testing
    pytest.main([__file__, "-v", "-s"])

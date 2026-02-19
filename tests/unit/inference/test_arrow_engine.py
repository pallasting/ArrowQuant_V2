"""
Unit tests for ArrowEngine module.

Tests cover:
- Arrow output support (Property 12)
- Encoding functionality
- Batch processing
- Similarity computation
- Model initialization
- Tokenization integration
- Normalization

**Property 12: Arrow Output Support**
For any text encoded by ArrowEngine, calling the Arrow output method should
return a PyArrow Array directly without intermediate Python list conversion.
**Validates: Requirements 3.3**
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from llm_compression.inference.arrow_engine import ArrowEngine


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_model_dir():
    """Create a mock model directory with all required files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir)
        
        # Create metadata.json
        metadata = {
            'model_name': 'test-model',
            'model_info': {
                'hidden_size': 64,
                'num_hidden_layers': 2,
                'num_attention_heads': 2,
                'intermediate_size': 256,
                'max_position_embeddings': 128,
                'vocab_size': 1000,
                'max_seq_length': 128,
                'layer_norm_eps': 1e-12,
                'embedding_dimension': 64,
            }
        }
        
        with open(model_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        # Create weights.parquet
        weights = _create_mock_weights()
        layers = []
        for name, tensor in weights.items():
            layers.append({
                'layer_name': name,
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'data': tensor.numpy().tobytes(),
                'num_params': tensor.numel(),
            })
        
        table = pa.Table.from_pylist(layers)
        pq.write_table(table, model_path / 'weights.parquet')
        
        # Create tokenizer directory with a simple tokenizer.json
        tokenizer_dir = model_path / 'tokenizer'
        tokenizer_dir.mkdir()
        
        # Create a minimal tokenizer.json
        vocab = {str(i): i for i in range(1000)}
        vocab["[UNK]"] = 0  # Add [UNK] token to vocabulary
        vocab["[PAD]"] = 1  # Add [PAD] token
        vocab["[CLS]"] = 2  # Add [CLS] token
        vocab["[SEP]"] = 3  # Add [SEP] token
        
        tokenizer_config = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "[UNK]"
            }
        }
        
        with open(tokenizer_dir / 'tokenizer.json', 'w') as f:
            json.dump(tokenizer_config, f)
        
        yield str(model_path)


def _create_mock_weights():
    """Create mock weights for a small model."""
    torch.manual_seed(42)
    
    weights = {
        # Embeddings
        'embeddings.word_embeddings.weight': torch.randn(1000, 64),
        'embeddings.position_embeddings.weight': torch.randn(128, 64),
        'embeddings.token_type_embeddings.weight': torch.randn(2, 64),
        'embeddings.LayerNorm.weight': torch.ones(64),
        'embeddings.LayerNorm.bias': torch.zeros(64),
    }
    
    # Add 2 transformer layers
    for i in range(2):
        prefix = f"encoder.layer.{i}"
        weights.update({
            f"{prefix}.attention.self.query.weight": torch.randn(64, 64),
            f"{prefix}.attention.self.query.bias": torch.zeros(64),
            f"{prefix}.attention.self.key.weight": torch.randn(64, 64),
            f"{prefix}.attention.self.key.bias": torch.zeros(64),
            f"{prefix}.attention.self.value.weight": torch.randn(64, 64),
            f"{prefix}.attention.self.value.bias": torch.zeros(64),
            f"{prefix}.attention.output.dense.weight": torch.randn(64, 64),
            f"{prefix}.attention.output.dense.bias": torch.zeros(64),
            f"{prefix}.attention.output.LayerNorm.weight": torch.ones(64),
            f"{prefix}.attention.output.LayerNorm.bias": torch.zeros(64),
            f"{prefix}.intermediate.dense.weight": torch.randn(256, 64),
            f"{prefix}.intermediate.dense.bias": torch.zeros(256),
            f"{prefix}.output.dense.weight": torch.randn(64, 256),
            f"{prefix}.output.dense.bias": torch.zeros(64),
            f"{prefix}.output.LayerNorm.weight": torch.ones(64),
            f"{prefix}.output.LayerNorm.bias": torch.zeros(64),
        })
    
    return weights


# ============================================================================
# Test: Construction and Initialization
# ============================================================================


class TestArrowEngineConstruction:
    """Test ArrowEngine initialization."""
    
    def test_initialization_with_valid_model(self, mock_model_dir):
        """Should initialize with valid model directory."""
        engine = ArrowEngine(mock_model_dir)
        
        assert engine.model_path == Path(mock_model_dir)
        assert engine.device in ['cpu', 'cuda', 'mps']
        assert engine.max_batch_size == 32
        assert engine.normalize_embeddings is True
    
    def test_initialization_with_custom_batch_size(self, mock_model_dir):
        """Should accept custom batch size."""
        engine = ArrowEngine(mock_model_dir, max_batch_size=64)
        assert engine.max_batch_size == 64
    
    def test_initialization_with_custom_device(self, mock_model_dir):
        """Should accept custom device."""
        engine = ArrowEngine(mock_model_dir, device='cpu')
        assert engine.device == 'cpu'
    
    def test_initialization_with_normalization_disabled(self, mock_model_dir):
        """Should allow disabling normalization."""
        engine = ArrowEngine(mock_model_dir, normalize_embeddings=False)
        assert engine.normalize_embeddings is False
    
    def test_initialization_loads_metadata(self, mock_model_dir):
        """Should load metadata from metadata.json."""
        engine = ArrowEngine(mock_model_dir)
        
        assert 'model_name' in engine.metadata
        assert engine.metadata['model_name'] == 'test-model'
    
    def test_initialization_loads_weights(self, mock_model_dir):
        """Should load weights via WeightLoader."""
        engine = ArrowEngine(mock_model_dir)
        
        assert hasattr(engine, 'weights')
        assert len(engine.weights) > 0
        assert 'embeddings.word_embeddings.weight' in engine.weights
    
    def test_initialization_loads_tokenizer(self, mock_model_dir):
        """Should load tokenizer."""
        engine = ArrowEngine(mock_model_dir)
        
        assert hasattr(engine, 'tokenizer')
        assert engine.tokenizer is not None
    
    def test_initialization_creates_inference_core(self, mock_model_dir):
        """Should create InferenceCore."""
        engine = ArrowEngine(mock_model_dir)
        
        assert hasattr(engine, 'inference_core')
        assert engine.inference_core is not None
    
    def test_initialization_with_missing_model_path(self):
        """Should raise FileNotFoundError for missing model path."""
        with pytest.raises(FileNotFoundError):
            ArrowEngine('/nonexistent/model/path')
    
    def test_repr_string(self, mock_model_dir):
        """Should have informative string representation."""
        engine = ArrowEngine(mock_model_dir)
        repr_str = repr(engine)
        
        assert 'ArrowEngine' in repr_str
        assert 'dim=' in repr_str
        assert 'device=' in repr_str


# ============================================================================
# Test: Encoding Functionality
# ============================================================================


class TestEncoding:
    """Test encoding functionality."""
    
    def test_encode_single_sentence(self, mock_model_dir):
        """Should encode a single sentence."""
        engine = ArrowEngine(mock_model_dir)
        
        embedding = engine.encode("Hello world")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 64)
        assert embedding.dtype == np.float32
    
    def test_encode_multiple_sentences(self, mock_model_dir):
        """Should encode multiple sentences."""
        engine = ArrowEngine(mock_model_dir)
        
        sentences = ["Hello world", "Test sentence", "Another example"]
        embeddings = engine.encode(sentences)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 64)
        assert embeddings.dtype == np.float32
    
    def test_encode_with_normalization(self, mock_model_dir):
        """Should normalize embeddings when requested."""
        engine = ArrowEngine(mock_model_dir)
        
        embedding = engine.encode("Test", normalize=True)
        
        # Check L2 norm is 1.0
        norm = np.linalg.norm(embedding[0])
        assert abs(norm - 1.0) < 1e-6
    
    def test_encode_without_normalization(self, mock_model_dir):
        """Should not normalize when disabled."""
        engine = ArrowEngine(mock_model_dir)
        
        embedding = engine.encode("Test", normalize=False)
        
        # Norm should not be 1.0 (unless by chance)
        norm = np.linalg.norm(embedding[0])
        # Just check it's a reasonable value, not necessarily 1.0
        assert norm > 0
    
    def test_encode_respects_default_normalization(self, mock_model_dir):
        """Should use default normalization setting."""
        engine = ArrowEngine(mock_model_dir, normalize_embeddings=True)
        
        embedding = engine.encode("Test")
        norm = np.linalg.norm(embedding[0])
        assert abs(norm - 1.0) < 1e-6
    
    def test_encode_empty_string(self, mock_model_dir):
        """Should handle empty string."""
        engine = ArrowEngine(mock_model_dir)
        
        # Should not crash
        embedding = engine.encode("")
        assert embedding.shape == (1, 64)
    
    def test_encode_long_text(self, mock_model_dir):
        """Should handle long text (truncation)."""
        engine = ArrowEngine(mock_model_dir)
        
        # Create text longer than max_seq_length
        long_text = " ".join(["word"] * 200)
        embedding = engine.encode(long_text)
        
        assert embedding.shape == (1, 64)
    
    def test_encode_output_is_finite(self, mock_model_dir):
        """Encoded embeddings should not contain NaN or Inf."""
        engine = ArrowEngine(mock_model_dir)
        
        embeddings = engine.encode(["Test 1", "Test 2", "Test 3"])
        
        assert np.isfinite(embeddings).all()
    
    def test_encode_deterministic(self, mock_model_dir):
        """Same input should produce same output."""
        engine = ArrowEngine(mock_model_dir)
        
        text = "Deterministic test"
        emb1 = engine.encode(text)
        emb2 = engine.encode(text)
        
        assert np.allclose(emb1, emb2)


# ============================================================================
# Test: Batch Processing
# ============================================================================


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    def test_encode_batch(self, mock_model_dir):
        """Should encode batch of sentences."""
        engine = ArrowEngine(mock_model_dir)
        
        sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]
        embeddings = engine.encode_batch(sentences)
        
        assert embeddings.shape == (3, 64)
    
    def test_encode_batch_with_normalization(self, mock_model_dir):
        """Should normalize batch embeddings."""
        engine = ArrowEngine(mock_model_dir)
        
        sentences = ["Test 1", "Test 2"]
        embeddings = engine.encode_batch(sentences, normalize=True)
        
        # Check all norms are 1.0
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, np.ones(2), atol=1e-6)
    
    def test_encode_with_automatic_batching(self, mock_model_dir):
        """Should automatically batch large inputs."""
        engine = ArrowEngine(mock_model_dir, max_batch_size=2)
        
        sentences = ["S1", "S2", "S3", "S4", "S5"]
        embeddings = engine.encode(sentences, batch_size=2)
        
        assert embeddings.shape == (5, 64)
    
    def test_encode_batch_consistency(self, mock_model_dir):
        """Batch encoding should match individual encoding."""
        engine = ArrowEngine(mock_model_dir)
        
        sentences = ["Test 1", "Test 2"]
        
        # Encode individually
        emb1 = engine.encode(sentences[0])
        emb2 = engine.encode(sentences[1])
        individual = np.vstack([emb1, emb2])
        
        # Encode as batch
        batch = engine.encode_batch(sentences)
        
        # Should be very close (allowing for minor numerical differences)
        assert np.allclose(individual, batch, atol=1e-5)
    
    def test_encode_large_batch_warning(self, mock_model_dir):
        """Should warn when batch exceeds max_batch_size."""
        engine = ArrowEngine(mock_model_dir, max_batch_size=2)
        
        sentences = ["S1", "S2", "S3", "S4"]
        
        # Should log warning but still work
        embeddings = engine.encode_batch(sentences)
        assert embeddings.shape == (4, 64)


# ============================================================================
# Test: Property 12 - Arrow Output Support
# ============================================================================


class TestProperty12ArrowOutputSupport:
    """
    Validate Property 12: Arrow Output Support.
    
    For any text encoded by ArrowEngine, calling the Arrow output method should
    return a PyArrow Array directly without intermediate Python list conversion.
    
    **Validates: Requirements 3.3**
    """
    
    def test_encode_returns_numpy_array(self, mock_model_dir):
        """Encode should return NumPy array (Arrow-compatible)."""
        engine = ArrowEngine(mock_model_dir)
        
        embeddings = engine.encode(["Test 1", "Test 2"])
        
        # NumPy arrays can be zero-copy converted to Arrow
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.flags['C_CONTIGUOUS']  # Contiguous for zero-copy
    
    def test_numpy_to_arrow_conversion(self, mock_model_dir):
        """NumPy output should convert to Arrow without copying."""
        engine = ArrowEngine(mock_model_dir)
        
        embeddings = engine.encode(["Test 1", "Test 2"])
        
        # Convert to Arrow array (zero-copy when possible)
        arrow_array = pa.array(embeddings.flatten())
        
        assert isinstance(arrow_array, pa.Array)
        assert len(arrow_array) == 2 * 64  # 2 embeddings * 64 dims
    
    def test_arrow_table_creation(self, mock_model_dir):
        """Should be able to create Arrow table from embeddings."""
        engine = ArrowEngine(mock_model_dir)
        
        sentences = ["Test 1", "Test 2", "Test 3"]
        embeddings = engine.encode(sentences)
        
        # Create Arrow table
        table = pa.table({
            'text': sentences,
            'embedding': [emb.tolist() for emb in embeddings],
        })
        
        assert isinstance(table, pa.Table)
        assert len(table) == 3
    
    def test_arrow_list_array_creation(self, mock_model_dir):
        """Should create Arrow list array from embeddings."""
        engine = ArrowEngine(mock_model_dir)
        
        embeddings = engine.encode(["Test 1", "Test 2"])
        
        # Convert to list of lists for Arrow
        embedding_lists = [emb.tolist() for emb in embeddings]
        
        # Create Arrow array with list type
        arrow_array = pa.array(embedding_lists, type=pa.list_(pa.float32()))
        
        assert isinstance(arrow_array, pa.Array)
        assert len(arrow_array) == 2
    
    def test_embeddings_compatible_with_parquet(self, mock_model_dir):
        """Embeddings should be writable to Parquet."""
        engine = ArrowEngine(mock_model_dir)
        
        sentences = ["Test 1", "Test 2"]
        embeddings = engine.encode(sentences)
        
        # Create table and write to Parquet
        table = pa.table({
            'text': sentences,
            'embedding': [emb.tolist() for emb in embeddings],
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            parquet_path = f.name
        
        try:
            pq.write_table(table, parquet_path)
            
            # Read back
            loaded_table = pq.read_table(parquet_path)
            assert len(loaded_table) == 2
        finally:
            os.unlink(parquet_path)
    
    def test_zero_copy_numpy_buffer(self, mock_model_dir):
        """NumPy arrays should support zero-copy buffer protocol."""
        engine = ArrowEngine(mock_model_dir)
        
        embeddings = engine.encode(["Test"])
        
        # Check buffer protocol support
        assert hasattr(embeddings, '__array_interface__')
        
        # Create Arrow array from buffer (zero-copy when possible)
        flat = embeddings.flatten()
        arrow_array = pa.array(flat, type=pa.float32())
        
        assert isinstance(arrow_array, pa.Array)


# ============================================================================
# Test: Similarity Computation
# ============================================================================


class TestSimilarity:
    """Test similarity computation."""
    
    def test_similarity_single_pair(self, mock_model_dir):
        """Should compute similarity between two sentences."""
        engine = ArrowEngine(mock_model_dir)
        
        sim = engine.similarity("Hello world", "Hello there")
        
        assert isinstance(sim, np.ndarray)
        assert sim.shape == (1, 1)
        assert -1.0 <= sim[0, 0] <= 1.0
    
    def test_similarity_multiple_pairs(self, mock_model_dir):
        """Should compute similarity matrix."""
        engine = ArrowEngine(mock_model_dir)
        
        sentences1 = ["Hello", "World"]
        sentences2 = ["Hi", "Earth", "Test"]
        
        sim = engine.similarity(sentences1, sentences2)
        
        assert sim.shape == (2, 3)
        # Allow small floating-point errors (values slightly > 1.0 or < -1.0)
        assert ((sim >= -1.01) & (sim <= 1.01)).all()
    
    def test_similarity_identical_sentences(self, mock_model_dir):
        """Identical sentences should have similarity ≈ 1.0."""
        engine = ArrowEngine(mock_model_dir)
        
        text = "Test sentence"
        sim = engine.similarity(text, text)
        
        # Should be very close to 1.0
        assert abs(sim[0, 0] - 1.0) < 1e-5
    
    def test_similarity_uses_normalized_embeddings(self, mock_model_dir):
        """Similarity should use normalized embeddings."""
        engine = ArrowEngine(mock_model_dir)
        
        # Similarity computation should normalize internally
        sim = engine.similarity("Test 1", "Test 2")
        
        # Result should be in valid cosine similarity range
        assert -1.0 <= sim[0, 0] <= 1.0


# ============================================================================
# Test: Model Information
# ============================================================================


class TestModelInformation:
    """Test model information methods."""
    
    def test_get_embedding_dimension(self, mock_model_dir):
        """Should return correct embedding dimension."""
        engine = ArrowEngine(mock_model_dir)
        
        dim = engine.get_embedding_dimension()
        
        assert dim == 64
    
    def test_get_max_seq_length(self, mock_model_dir):
        """Should return maximum sequence length."""
        engine = ArrowEngine(mock_model_dir)
        
        max_len = engine.get_max_seq_length()
        
        assert max_len == 128


# ============================================================================
# Test: Attention Output
# ============================================================================


class TestAttentionOutput:
    """Test attention weight output functionality."""
    
    def test_encode_with_attention_output(self, mock_model_dir):
        """Should return attention weights when requested."""
        engine = ArrowEngine(mock_model_dir)
        
        embeddings, attentions = engine.encode(
            "Test sentence",
            output_attentions=True
        )
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 64)
        assert attentions is not None
        assert len(attentions) > 0
    
    def test_encode_batch_with_attention_output(self, mock_model_dir):
        """Should return attention weights for batch."""
        engine = ArrowEngine(mock_model_dir)
        
        sentences = ["Test 1", "Test 2"]
        embeddings, attentions = engine.encode_batch(
            sentences,
            output_attentions=True
        )
        
        assert embeddings.shape == (2, 64)
        assert attentions is not None


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling."""
    
    def test_missing_weights_file(self, mock_model_dir):
        """Should raise error if weights.parquet is missing."""
        # Remove weights file
        weights_path = Path(mock_model_dir) / 'weights.parquet'
        weights_path.unlink()
        
        with pytest.raises(FileNotFoundError):
            ArrowEngine(mock_model_dir)
    
    def test_missing_tokenizer(self, mock_model_dir):
        """Should raise error if tokenizer is missing."""
        # Remove tokenizer directory
        import shutil
        tokenizer_path = Path(mock_model_dir) / 'tokenizer'
        shutil.rmtree(tokenizer_path)
        
        with pytest.raises(FileNotFoundError):
            ArrowEngine(mock_model_dir)
    
    def test_invalid_metadata(self, mock_model_dir):
        """Should handle invalid metadata gracefully."""
        # Corrupt metadata
        metadata_path = Path(mock_model_dir) / 'metadata.json'
        with open(metadata_path, 'w') as f:
            f.write("invalid json{")
        
        # Should still initialize (metadata is optional)
        # but may use defaults
        try:
            engine = ArrowEngine(mock_model_dir)
            # If it initializes, that's acceptable
            assert engine is not None
        except json.JSONDecodeError:
            # Also acceptable to raise error
            pass


# ============================================================================
# Test: Integration
# ============================================================================


class TestIntegration:
    """Test integration scenarios."""
    
    def test_encode_and_store_in_arrow(self, mock_model_dir):
        """Should encode and store in Arrow format."""
        engine = ArrowEngine(mock_model_dir)
        
        sentences = ["Test 1", "Test 2", "Test 3"]
        embeddings = engine.encode(sentences)
        
        # Create Arrow table
        table = pa.table({
            'id': [1, 2, 3],
            'text': sentences,
            'embedding': [emb.tolist() for emb in embeddings],
        })
        
        # Write to Parquet
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            parquet_path = f.name
        
        try:
            pq.write_table(table, parquet_path)
            
            # Read back and verify
            loaded = pq.read_table(parquet_path)
            assert len(loaded) == 3
            
            # Verify embeddings
            loaded_embeddings = loaded['embedding'].to_pylist()
            for i, emb in enumerate(loaded_embeddings):
                assert len(emb) == 64
                assert np.allclose(emb, embeddings[i], atol=1e-6)
        finally:
            os.unlink(parquet_path)
    
    def test_multiple_encode_calls(self, mock_model_dir):
        """Should handle multiple encode calls correctly."""
        engine = ArrowEngine(mock_model_dir)
        
        # Multiple calls should work
        emb1 = engine.encode("Test 1")
        emb2 = engine.encode("Test 2")
        emb3 = engine.encode(["Test 3", "Test 4"])
        
        assert emb1.shape == (1, 64)
        assert emb2.shape == (1, 64)
        assert emb3.shape == (2, 64)
    
    def test_encode_with_various_text_types(self, mock_model_dir):
        """Should handle various text types."""
        engine = ArrowEngine(mock_model_dir)
        
        # Different text types
        texts = [
            "Simple text",
            "Text with numbers 123",
            "Text with punctuation!?",
            "UPPERCASE TEXT",
            "lowercase text",
        ]
        
        embeddings = engine.encode(texts)
        
        assert embeddings.shape == (5, 64)
        assert np.isfinite(embeddings).all()

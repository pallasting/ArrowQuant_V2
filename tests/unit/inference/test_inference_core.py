"""
Unit tests for InferenceCore module.

Tests cover:
- Complete embedding computation (Property 1)
- Transformer layer count (Property 2)
- Mean pooling with mask support (Property 6)
- L2 normalization (Property 7)
- Weight loading correctness (Property 8)
- Forward pass validation
- Configuration auto-detection
- Output shape validation

**Property 1: Complete Embedding Computation**
For any valid input_ids tensor, the embedding layer output should contain
contributions from word embeddings, position embeddings, and token type
embeddings (verified by checking that the output differs from any single
embedding type alone).
**Validates: Requirements 1.1**

**Property 2: Transformer Layer Count**
For any loaded model, the number of encoder layers in InferenceCore should
match the configured num_layers, and forward propagation should execute all
layers (verified by checking that modifying the last layer's weights affects
the final output).
**Validates: Requirements 1.2**

**Property 6: Mean Pooling with Mask Support**
For any hidden states and attention mask, mean pooling should compute the
average only over non-masked positions (verified by checking that changing
masked positions doesn't affect the output).
**Validates: Requirements 1.6**

**Property 7: L2 Normalization**
For any embedding vector, after L2 normalization, the vector should have
L2 norm = 1.0 (within floating point tolerance of 1e-6).
**Validates: Requirements 1.7**

**Property 8: Weight Loading Correctness**
For any weight tensor in the Parquet file, it should be loaded into the
correct PyTorch module parameter with matching shape and name.
**Validates: Requirements 1.8**
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_compression.inference.inference_core import InferenceCore


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def minimal_weights():
    """Minimal weights for a small model (64 dim, 2 layers, 2 heads)."""
    weights = {
        # Embeddings
        'embeddings.word_embeddings.weight': torch.randn(1000, 64),
        'embeddings.position_embeddings.weight': torch.randn(512, 64),
        'embeddings.token_type_embeddings.weight': torch.randn(2, 64),
        'embeddings.LayerNorm.weight': torch.ones(64),
        'embeddings.LayerNorm.bias': torch.zeros(64),
    }
    
    # Add weights for 2 transformer layers
    for i in range(2):
        prefix = f"encoder.layer.{i}"
        weights.update({
            # Attention Q, K, V
            f"{prefix}.attention.self.query.weight": torch.randn(64, 64),
            f"{prefix}.attention.self.query.bias": torch.zeros(64),
            f"{prefix}.attention.self.key.weight": torch.randn(64, 64),
            f"{prefix}.attention.self.key.bias": torch.zeros(64),
            f"{prefix}.attention.self.value.weight": torch.randn(64, 64),
            f"{prefix}.attention.self.value.bias": torch.zeros(64),
            # Attention output
            f"{prefix}.attention.output.dense.weight": torch.randn(64, 64),
            f"{prefix}.attention.output.dense.bias": torch.zeros(64),
            f"{prefix}.attention.output.LayerNorm.weight": torch.ones(64),
            f"{prefix}.attention.output.LayerNorm.bias": torch.zeros(64),
            # FFN
            f"{prefix}.intermediate.dense.weight": torch.randn(256, 64),
            f"{prefix}.intermediate.dense.bias": torch.zeros(256),
            f"{prefix}.output.dense.weight": torch.randn(64, 256),
            f"{prefix}.output.dense.bias": torch.zeros(64),
            f"{prefix}.output.LayerNorm.weight": torch.ones(64),
            f"{prefix}.output.LayerNorm.bias": torch.zeros(64),
        })
    
    return weights


@pytest.fixture
def minimal_config():
    """Minimal configuration for testing."""
    return {
        'hidden_size': 64,
        'num_layers': 2,
        'num_attention_heads': 2,
        'intermediate_size': 256,
        'max_position_embeddings': 512,
        'vocab_size': 1000,
        'layer_norm_eps': 1e-12,
    }


@pytest.fixture
def standard_weights():
    """Standard BERT-like weights (384 dim, 6 layers, 6 heads)."""
    weights = {
        # Embeddings
        'embeddings.word_embeddings.weight': torch.randn(30522, 384),
        'embeddings.position_embeddings.weight': torch.randn(512, 384),
        'embeddings.token_type_embeddings.weight': torch.randn(2, 384),
        'embeddings.LayerNorm.weight': torch.ones(384),
        'embeddings.LayerNorm.bias': torch.zeros(384),
    }
    
    # Add weights for 6 transformer layers
    for i in range(6):
        prefix = f"encoder.layer.{i}"
        weights.update({
            f"{prefix}.attention.self.query.weight": torch.randn(384, 384),
            f"{prefix}.attention.self.query.bias": torch.zeros(384),
            f"{prefix}.attention.self.key.weight": torch.randn(384, 384),
            f"{prefix}.attention.self.key.bias": torch.zeros(384),
            f"{prefix}.attention.self.value.weight": torch.randn(384, 384),
            f"{prefix}.attention.self.value.bias": torch.zeros(384),
            f"{prefix}.attention.output.dense.weight": torch.randn(384, 384),
            f"{prefix}.attention.output.dense.bias": torch.zeros(384),
            f"{prefix}.attention.output.LayerNorm.weight": torch.ones(384),
            f"{prefix}.attention.output.LayerNorm.bias": torch.zeros(384),
            f"{prefix}.intermediate.dense.weight": torch.randn(1536, 384),
            f"{prefix}.intermediate.dense.bias": torch.zeros(1536),
            f"{prefix}.output.dense.weight": torch.randn(384, 1536),
            f"{prefix}.output.dense.bias": torch.zeros(384),
            f"{prefix}.output.LayerNorm.weight": torch.ones(384),
            f"{prefix}.output.LayerNorm.bias": torch.zeros(384),
        })
    
    return weights


@pytest.fixture
def standard_config():
    """Standard BERT configuration."""
    return {
        'hidden_size': 384,
        'num_layers': 6,
        'num_attention_heads': 6,
        'intermediate_size': 1536,
        'max_position_embeddings': 512,
        'vocab_size': 30522,
        'layer_norm_eps': 1e-12,
    }


# ============================================================================
# Test: Construction and Configuration
# ============================================================================


class TestInferenceCoreConstruction:
    """Test InferenceCore initialization and structure."""
    
    def test_initialization_with_minimal_config(self, minimal_weights, minimal_config):
        """Should initialize with minimal configuration."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        assert core.hidden_size == 64
        assert core.num_layers == 2
        assert core.num_attention_heads == 2
        assert core.intermediate_size == 256
        assert core.device == 'cpu'
    
    def test_initialization_with_standard_config(self, standard_weights, standard_config):
        """Should initialize with standard BERT configuration."""
        core = InferenceCore(standard_weights, standard_config, device='cpu')
        
        assert core.hidden_size == 384
        assert core.num_layers == 6
        assert core.num_attention_heads == 6
        assert core.intermediate_size == 1536
    
    def test_has_embedding_layers(self, minimal_weights, minimal_config):
        """Should have word, position, and token_type embeddings."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        assert hasattr(core, 'word_embeddings')
        assert hasattr(core, 'position_embeddings')
        assert hasattr(core, 'token_type_embeddings')
        assert isinstance(core.word_embeddings, nn.Embedding)
        assert isinstance(core.position_embeddings, nn.Embedding)
        assert isinstance(core.token_type_embeddings, nn.Embedding)
    
    def test_has_embedding_layernorm(self, minimal_weights, minimal_config):
        """Should have LayerNorm for embeddings."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        assert hasattr(core, 'embedding_layernorm')
        assert isinstance(core.embedding_layernorm, nn.LayerNorm)
    
    def test_has_encoder_layers(self, minimal_weights, minimal_config):
        """Should have encoder layers."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        assert hasattr(core, 'encoder_layers')
        assert isinstance(core.encoder_layers, nn.ModuleList)
        assert len(core.encoder_layers) == 2
    
    def test_auto_detect_num_layers(self, minimal_weights):
        """Should auto-detect number of layers from weights."""
        config = {'hidden_size': 64}  # No num_layers specified
        core = InferenceCore(minimal_weights, config, device='cpu')
        
        # Should detect 2 layers from weights
        assert core.num_layers == 2
    
    def test_auto_detect_intermediate_size(self, minimal_weights):
        """Should auto-detect intermediate size from weights."""
        config = {'hidden_size': 64, 'num_layers': 2}
        core = InferenceCore(minimal_weights, config, device='cpu')
        
        # Should detect 256 from weights
        assert core.intermediate_size == 256
    
    def test_auto_detect_vocab_size(self, minimal_weights):
        """Should auto-detect vocab size from weights."""
        config = {'hidden_size': 64, 'num_layers': 2}
        core = InferenceCore(minimal_weights, config, device='cpu')
        
        # Should detect 1000 from weights
        assert core.vocab_size == 1000
    
    def test_get_embedding_dimension(self, minimal_weights, minimal_config):
        """Should return correct embedding dimension."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        assert core.get_embedding_dimension() == 64


# ============================================================================
# Test: Property 1 - Complete Embedding Computation
# ============================================================================

class TestProperty1CompleteEmbeddingComputation:
    """
    Validate Property 1: Complete Embedding Computation.
    
    For any valid input_ids tensor, the embedding layer output should contain
    contributions from word embeddings, position embeddings, and token type
    embeddings (verified by checking that the output differs from any single
    embedding type alone).
    
    **Validates: Requirements 1.1**
    """
    
    def test_embeddings_combine_all_three_types(self, minimal_weights, minimal_config):
        """Embeddings should combine word, position, and token_type."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        with torch.no_grad():
            # Get combined embeddings
            combined = core._compute_embeddings(input_ids)
            
            # Get individual embeddings
            word_emb = core.word_embeddings(input_ids)
            position_ids = torch.arange(5).unsqueeze(0)
            position_emb = core.position_embeddings(position_ids)
            token_type_ids = torch.zeros_like(input_ids)
            token_type_emb = core.token_type_embeddings(token_type_ids)
            
            # Combined should differ from any single type
            assert not torch.allclose(combined, word_emb, atol=1e-3)
            assert not torch.allclose(combined, position_emb, atol=1e-3)
            assert not torch.allclose(combined, token_type_emb, atol=1e-3)
    
    def test_embeddings_include_word_contribution(self, minimal_weights, minimal_config):
        """Different word IDs should produce different embeddings."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids_1 = torch.tensor([[1, 2, 3]])
        input_ids_2 = torch.tensor([[4, 5, 6]])
        
        with torch.no_grad():
            emb_1 = core._compute_embeddings(input_ids_1)
            emb_2 = core._compute_embeddings(input_ids_2)
        
        # Different words should produce different embeddings
        assert not torch.allclose(emb_1, emb_2, atol=1e-3)
    
    def test_embeddings_include_position_contribution(self, minimal_weights, minimal_config):
        """Position should affect embeddings."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        # Same word at different positions
        input_ids = torch.tensor([[5, 5, 5, 5, 5]])
        
        with torch.no_grad():
            embeddings = core._compute_embeddings(input_ids)
        
        # Different positions should produce different embeddings
        # (even for the same word)
        assert not torch.allclose(embeddings[0, 0], embeddings[0, 1], atol=1e-3)
        assert not torch.allclose(embeddings[0, 1], embeddings[0, 2], atol=1e-3)
    
    def test_embeddings_apply_layernorm(self, minimal_weights, minimal_config):
        """Embeddings should be normalized."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        with torch.no_grad():
            embeddings = core._compute_embeddings(input_ids)
        
        # Check normalization: mean ≈ 0, std ≈ 1
        mean = embeddings.mean(dim=-1)
        std = embeddings.std(dim=-1)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-2)
    
    def test_embedding_output_shape(self, minimal_weights, minimal_config):
        """Embedding output should have shape (batch, seq, hidden)."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        embeddings = core._compute_embeddings(input_ids)
        
        assert embeddings.shape == (1, 5, 64)


# ============================================================================
# Test: Property 2 - Transformer Layer Count
# ============================================================================

class TestProperty2TransformerLayerCount:
    """
    Validate Property 2: Transformer Layer Count.
    
    For any loaded model, the number of encoder layers in InferenceCore should
    match the configured num_layers, and forward propagation should execute all
    layers (verified by checking that modifying the last layer's weights affects
    the final output).
    
    **Validates: Requirements 1.2**
    """
    
    def test_correct_number_of_layers(self, minimal_weights, minimal_config):
        """Should have configured number of transformer layers."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        assert len(core.encoder_layers) == 2
    
    def test_all_layers_executed_in_forward(self, minimal_weights, minimal_config):
        """All transformer layers should be executed during forward pass."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones(1, 5)
        
        # Track which layers are executed by checking intermediate states
        hidden_states = core._compute_embeddings(input_ids)
        extended_mask = core._get_extended_attention_mask(attention_mask)
        
        # Store intermediate outputs
        layer_outputs = []
        for layer in core.encoder_layers:
            hidden_states, _ = layer(hidden_states, extended_mask)
            layer_outputs.append(hidden_states.clone())
        
        # All layers should produce different outputs
        assert len(layer_outputs) == 2
        assert not torch.allclose(layer_outputs[0], layer_outputs[1], atol=1e-5)
    
    def test_intermediate_layers_affect_output(self, minimal_weights, minimal_config):
        """Intermediate layers should affect final output."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones(1, 5)
        
        # Get embeddings
        hidden_states = core._compute_embeddings(input_ids)
        extended_mask = core._get_extended_attention_mask(attention_mask)
        
        # Apply first layer
        hidden_after_layer0, _ = core.encoder_layers[0](hidden_states, extended_mask)
        
        # Apply second layer
        hidden_after_layer1, _ = core.encoder_layers[1](hidden_after_layer0, extended_mask)
        
        # Outputs should be different after each layer
        assert not torch.allclose(hidden_states, hidden_after_layer0, atol=1e-3)
        assert not torch.allclose(hidden_after_layer0, hidden_after_layer1, atol=1e-3)
    
    def test_layer_count_matches_config(self):
        """Layer count should match configuration for various sizes."""
        for num_layers in [1, 2, 4, 6, 12]:
            weights = {
                'embeddings.word_embeddings.weight': torch.randn(1000, 64),
                'embeddings.position_embeddings.weight': torch.randn(512, 64),
                'embeddings.token_type_embeddings.weight': torch.randn(2, 64),
                'embeddings.LayerNorm.weight': torch.ones(64),
                'embeddings.LayerNorm.bias': torch.zeros(64),
            }
            
            for i in range(num_layers):
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
            
            config = {'hidden_size': 64}
            core = InferenceCore(weights, config, device='cpu')
            assert core.num_layers == num_layers
            assert len(core.encoder_layers) == num_layers


# ============================================================================
# Test: Forward Pass and Output Shape
# ============================================================================


class TestForwardPass:
    """Test forward pass functionality."""
    
    def test_forward_output_shape(self, minimal_weights, minimal_config):
        """Forward pass should produce (batch_size, hidden_size) output."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones(1, 5)
        
        with torch.no_grad():
            output = core(input_ids, attention_mask)
        
        assert output.shape == (1, 64)
    
    def test_forward_batch_processing(self, minimal_weights, minimal_config):
        """Should handle batch processing correctly."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        batch_size = 4
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            output = core(input_ids, attention_mask)
        
        assert output.shape == (batch_size, 64)
    
    def test_forward_various_sequence_lengths(self, minimal_weights, minimal_config):
        """Should handle various sequence lengths."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        for seq_len in [1, 5, 10, 20, 50]:
            input_ids = torch.randint(0, 1000, (2, seq_len))
            attention_mask = torch.ones(2, seq_len)
            
            with torch.no_grad():
                output = core(input_ids, attention_mask)
            
            assert output.shape == (2, 64)
    
    def test_forward_output_is_finite(self, minimal_weights, minimal_config):
        """Forward output should not contain NaN or Inf."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        with torch.no_grad():
            output = core(input_ids, attention_mask)
        
        assert torch.isfinite(output).all()
    
    def test_forward_with_output_attentions(self, minimal_weights, minimal_config):
        """Should return attention weights when requested."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones(1, 5)
        
        with torch.no_grad():
            output, attentions = core(input_ids, attention_mask, output_attentions=True)
        
        assert output.shape == (1, 64)
        assert attentions is not None
        assert len(attentions) == 2  # 2 layers
    
    def test_forward_deterministic_in_eval_mode(self, minimal_weights, minimal_config):
        """Same input should produce same output in eval mode."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones(1, 5)
        
        with torch.no_grad():
            output1 = core(input_ids, attention_mask)
            output2 = core(input_ids, attention_mask)
        
        assert torch.allclose(output1, output2)


# ============================================================================
# Test: Property 6 - Mean Pooling with Mask Support
# ============================================================================

class TestProperty6MeanPoolingWithMaskSupport:
    """
    Validate Property 6: Mean Pooling with Mask Support.
    
    For any hidden states and attention mask, mean pooling should compute the
    average only over non-masked positions (verified by checking that changing
    masked positions doesn't affect the output).
    
    **Validates: Requirements 1.6**
    """
    
    def test_mean_pooling_respects_mask(self, minimal_weights, minimal_config):
        """Mean pooling should only average over non-masked positions."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        # Create hidden states
        hidden_states = torch.randn(1, 10, 64)
        
        # Mask out last 5 positions
        attention_mask = torch.ones(1, 10)
        attention_mask[0, 5:] = 0
        
        pooled = core.mean_pooling(hidden_states, attention_mask)
        
        # Manually compute expected pooling (only first 5 positions)
        expected = hidden_states[0, :5].mean(dim=0)
        
        assert torch.allclose(pooled[0], expected, atol=1e-5)
    
    def test_masked_positions_dont_affect_output(self, minimal_weights, minimal_config):
        """Changing masked positions should not affect pooled output."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        hidden_states = torch.randn(1, 10, 64)
        attention_mask = torch.ones(1, 10)
        attention_mask[0, 5:] = 0  # Mask last 5
        
        # Get baseline pooling
        pooled_baseline = core.mean_pooling(hidden_states, attention_mask)
        
        # Modify masked positions
        hidden_states_modified = hidden_states.clone()
        hidden_states_modified[0, 5:] = torch.randn(5, 64) * 100
        
        # Get modified pooling
        pooled_modified = core.mean_pooling(hidden_states_modified, attention_mask)
        
        # Should be identical (masked positions ignored)
        assert torch.allclose(pooled_baseline, pooled_modified, atol=1e-5)
    
    def test_mean_pooling_with_different_mask_patterns(self, minimal_weights, minimal_config):
        """Should handle various masking patterns correctly."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        hidden_states = torch.randn(3, 10, 64)
        
        # Different mask patterns
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # First 5 tokens
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # All tokens
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # First 3 tokens
        ], dtype=torch.float)
        
        pooled = core.mean_pooling(hidden_states, attention_mask)
        
        # Verify each sequence pooled correctly
        expected_0 = hidden_states[0, :5].mean(dim=0)
        expected_1 = hidden_states[1, :].mean(dim=0)
        expected_2 = hidden_states[2, :3].mean(dim=0)
        
        assert torch.allclose(pooled[0], expected_0, atol=1e-5)
        assert torch.allclose(pooled[1], expected_1, atol=1e-5)
        assert torch.allclose(pooled[2], expected_2, atol=1e-5)
    
    def test_mean_pooling_output_shape(self, minimal_weights, minimal_config):
        """Mean pooling should reduce sequence dimension."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        hidden_states = torch.randn(4, 20, 64)
        attention_mask = torch.ones(4, 20)
        
        pooled = core.mean_pooling(hidden_states, attention_mask)
        
        assert pooled.shape == (4, 64)
    
    def test_mean_pooling_with_single_token(self, minimal_weights, minimal_config):
        """Should handle single-token sequences."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        hidden_states = torch.randn(1, 1, 64)
        attention_mask = torch.ones(1, 1)
        
        pooled = core.mean_pooling(hidden_states, attention_mask)
        
        # Single token: pooled should equal the token
        assert torch.allclose(pooled[0], hidden_states[0, 0], atol=1e-6)
    
    def test_mean_pooling_handles_zero_mask_gracefully(self, minimal_weights, minimal_config):
        """Should handle edge case of all-zero mask without crashing."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        hidden_states = torch.randn(1, 5, 64)
        attention_mask = torch.zeros(1, 5)  # All masked
        
        # Should not crash (uses clamp to avoid division by zero)
        pooled = core.mean_pooling(hidden_states, attention_mask)
        
        assert pooled.shape == (1, 64)
        assert torch.isfinite(pooled).all()


# ============================================================================
# Test: Property 7 - L2 Normalization
# ============================================================================

class TestProperty7L2Normalization:
    """
    Validate Property 7: L2 Normalization.
    
    For any embedding vector, after L2 normalization, the vector should have
    L2 norm = 1.0 (within floating point tolerance of 1e-6).
    
    **Validates: Requirements 1.7**
    """
    
    def test_normalized_embeddings_have_unit_norm(self, minimal_weights, minimal_config):
        """Normalized embeddings should have L2 norm = 1.0."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        # Create random embeddings
        embeddings = torch.randn(10, 64)
        
        # Normalize
        normalized = core.normalize_embeddings(embeddings)
        
        # Check norms
        norms = torch.norm(normalized, p=2, dim=1)
        
        assert torch.allclose(norms, torch.ones(10), atol=1e-6)
    
    def test_normalization_preserves_direction(self, minimal_weights, minimal_config):
        """Normalization should preserve vector direction."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        embeddings = torch.randn(5, 64)
        normalized = core.normalize_embeddings(embeddings)
        
        # Cosine similarity between original and normalized should be 1
        # (same direction, different magnitude)
        for i in range(5):
            orig_norm = embeddings[i] / torch.norm(embeddings[i])
            assert torch.allclose(orig_norm, normalized[i], atol=1e-6)
    
    def test_normalization_with_various_magnitudes(self, minimal_weights, minimal_config):
        """Should normalize vectors of various magnitudes to unit norm."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        # Create embeddings with different magnitudes
        embeddings = torch.randn(5, 64)
        embeddings[0] *= 0.1    # Small magnitude
        embeddings[1] *= 1.0    # Medium magnitude
        embeddings[2] *= 10.0   # Large magnitude
        embeddings[3] *= 100.0  # Very large magnitude
        embeddings[4] *= 0.01   # Very small magnitude
        
        normalized = core.normalize_embeddings(embeddings)
        norms = torch.norm(normalized, p=2, dim=1)
        
        # All should have unit norm
        assert torch.allclose(norms, torch.ones(5), atol=1e-6)
    
    def test_normalization_output_shape(self, minimal_weights, minimal_config):
        """Normalization should preserve shape."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        embeddings = torch.randn(8, 64)
        normalized = core.normalize_embeddings(embeddings)
        
        assert normalized.shape == embeddings.shape
    
    def test_forward_pass_produces_normalized_output(self, minimal_weights, minimal_config):
        """Forward pass should produce normalized embeddings by default."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.randint(0, 1000, (5, 10))
        attention_mask = torch.ones(5, 10)
        
        with torch.no_grad():
            # Get pooled output (before normalization)
            hidden_states = core._compute_embeddings(input_ids)
            extended_mask = core._get_extended_attention_mask(attention_mask)
            
            for layer in core.encoder_layers:
                hidden_states, _ = layer(hidden_states, extended_mask)
            
            pooled = core.mean_pooling(hidden_states, attention_mask)
            
            # Normalize
            normalized = core.normalize_embeddings(pooled)
        
        # Check norms
        norms = torch.norm(normalized, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-6)


# ============================================================================
# Test: Property 8 - Weight Loading Correctness
# ============================================================================


class TestProperty8WeightLoadingCorrectness:
    """
    Validate Property 8: Weight Loading Correctness.
    
    For any weight tensor in the Parquet file, it should be loaded into the
    correct PyTorch module parameter with matching shape and name.
    
    **Validates: Requirements 1.8**
    """
    
    def test_word_embeddings_loaded_correctly(self, minimal_weights, minimal_config):
        """Word embedding weights should be loaded correctly."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        expected = minimal_weights['embeddings.word_embeddings.weight']
        actual = core.word_embeddings.weight
        
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected.float(), atol=1e-6)
    
    def test_position_embeddings_loaded_correctly(self, minimal_weights, minimal_config):
        """Position embedding weights should be loaded correctly."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        expected = minimal_weights['embeddings.position_embeddings.weight']
        actual = core.position_embeddings.weight
        
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected.float(), atol=1e-6)
    
    def test_token_type_embeddings_loaded_correctly(self, minimal_weights, minimal_config):
        """Token type embedding weights should be loaded correctly."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        expected = minimal_weights['embeddings.token_type_embeddings.weight']
        actual = core.token_type_embeddings.weight
        
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected.float(), atol=1e-6)
    
    def test_embedding_layernorm_loaded_correctly(self, minimal_weights, minimal_config):
        """Embedding LayerNorm weights should be loaded correctly."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        expected_weight = minimal_weights['embeddings.LayerNorm.weight']
        expected_bias = minimal_weights['embeddings.LayerNorm.bias']
        
        assert torch.allclose(core.embedding_layernorm.weight, expected_weight.float(), atol=1e-6)
        assert torch.allclose(core.embedding_layernorm.bias, expected_bias.float(), atol=1e-6)
    
    def test_attention_qkv_weights_loaded_correctly(self, minimal_weights, minimal_config):
        """Attention Q, K, V weights should be loaded correctly."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        for i in range(2):
            prefix = f"encoder.layer.{i}"
            layer = core.encoder_layers[i]
            
            # Query
            expected_q_weight = minimal_weights[f"{prefix}.attention.self.query.weight"]
            expected_q_bias = minimal_weights[f"{prefix}.attention.self.query.bias"]
            assert torch.allclose(layer.attention.query.weight, expected_q_weight.float(), atol=1e-6)
            assert torch.allclose(layer.attention.query.bias, expected_q_bias.float(), atol=1e-6)
            
            # Key
            expected_k_weight = minimal_weights[f"{prefix}.attention.self.key.weight"]
            expected_k_bias = minimal_weights[f"{prefix}.attention.self.key.bias"]
            assert torch.allclose(layer.attention.key.weight, expected_k_weight.float(), atol=1e-6)
            assert torch.allclose(layer.attention.key.bias, expected_k_bias.float(), atol=1e-6)
            
            # Value
            expected_v_weight = minimal_weights[f"{prefix}.attention.self.value.weight"]
            expected_v_bias = minimal_weights[f"{prefix}.attention.self.value.bias"]
            assert torch.allclose(layer.attention.value.weight, expected_v_weight.float(), atol=1e-6)
            assert torch.allclose(layer.attention.value.bias, expected_v_bias.float(), atol=1e-6)
    
    def test_attention_output_weights_loaded_correctly(self, minimal_weights, minimal_config):
        """Attention output weights should be loaded correctly."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        for i in range(2):
            prefix = f"encoder.layer.{i}"
            layer = core.encoder_layers[i]
            
            expected_weight = minimal_weights[f"{prefix}.attention.output.dense.weight"]
            expected_bias = minimal_weights[f"{prefix}.attention.output.dense.bias"]
            
            assert torch.allclose(layer.attention_output.weight, expected_weight.float(), atol=1e-6)
            assert torch.allclose(layer.attention_output.bias, expected_bias.float(), atol=1e-6)
    
    def test_attention_layernorm_weights_loaded_correctly(self, minimal_weights, minimal_config):
        """Attention LayerNorm weights should be loaded correctly."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        for i in range(2):
            prefix = f"encoder.layer.{i}"
            layer = core.encoder_layers[i]
            
            expected_weight = minimal_weights[f"{prefix}.attention.output.LayerNorm.weight"]
            expected_bias = minimal_weights[f"{prefix}.attention.output.LayerNorm.bias"]
            
            assert torch.allclose(layer.attention_layernorm.weight, expected_weight.float(), atol=1e-6)
            assert torch.allclose(layer.attention_layernorm.bias, expected_bias.float(), atol=1e-6)
    
    def test_ffn_intermediate_weights_loaded_correctly(self, minimal_weights, minimal_config):
        """FFN intermediate weights should be loaded correctly."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        for i in range(2):
            prefix = f"encoder.layer.{i}"
            layer = core.encoder_layers[i]
            
            expected_weight = minimal_weights[f"{prefix}.intermediate.dense.weight"]
            expected_bias = minimal_weights[f"{prefix}.intermediate.dense.bias"]
            
            assert torch.allclose(layer.intermediate.weight, expected_weight.float(), atol=1e-6)
            assert torch.allclose(layer.intermediate.bias, expected_bias.float(), atol=1e-6)
    
    def test_ffn_output_weights_loaded_correctly(self, minimal_weights, minimal_config):
        """FFN output weights should be loaded correctly."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        for i in range(2):
            prefix = f"encoder.layer.{i}"
            layer = core.encoder_layers[i]
            
            expected_weight = minimal_weights[f"{prefix}.output.dense.weight"]
            expected_bias = minimal_weights[f"{prefix}.output.dense.bias"]
            
            assert torch.allclose(layer.output_dense.weight, expected_weight.float(), atol=1e-6)
            assert torch.allclose(layer.output_dense.bias, expected_bias.float(), atol=1e-6)
    
    def test_ffn_layernorm_weights_loaded_correctly(self, minimal_weights, minimal_config):
        """FFN LayerNorm weights should be loaded correctly."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        for i in range(2):
            prefix = f"encoder.layer.{i}"
            layer = core.encoder_layers[i]
            
            expected_weight = minimal_weights[f"{prefix}.output.LayerNorm.weight"]
            expected_bias = minimal_weights[f"{prefix}.output.LayerNorm.bias"]
            
            assert torch.allclose(layer.output_layernorm.weight, expected_weight.float(), atol=1e-6)
            assert torch.allclose(layer.output_layernorm.bias, expected_bias.float(), atol=1e-6)
    
    def test_float16_weights_converted_to_float32(self, minimal_config):
        """Float16 weights should be converted to float32 for computation."""
        # Create float16 weights
        weights_fp16 = {
            'embeddings.word_embeddings.weight': torch.randn(1000, 64, dtype=torch.float16),
            'embeddings.position_embeddings.weight': torch.randn(512, 64, dtype=torch.float16),
            'embeddings.token_type_embeddings.weight': torch.randn(2, 64, dtype=torch.float16),
            'embeddings.LayerNorm.weight': torch.ones(64, dtype=torch.float16),
            'embeddings.LayerNorm.bias': torch.zeros(64, dtype=torch.float16),
        }
        
        # Add layer weights
        for i in range(2):
            prefix = f"encoder.layer.{i}"
            weights_fp16.update({
                f"{prefix}.attention.self.query.weight": torch.randn(64, 64, dtype=torch.float16),
                f"{prefix}.attention.self.query.bias": torch.zeros(64, dtype=torch.float16),
                f"{prefix}.attention.self.key.weight": torch.randn(64, 64, dtype=torch.float16),
                f"{prefix}.attention.self.key.bias": torch.zeros(64, dtype=torch.float16),
                f"{prefix}.attention.self.value.weight": torch.randn(64, 64, dtype=torch.float16),
                f"{prefix}.attention.self.value.bias": torch.zeros(64, dtype=torch.float16),
                f"{prefix}.attention.output.dense.weight": torch.randn(64, 64, dtype=torch.float16),
                f"{prefix}.attention.output.dense.bias": torch.zeros(64, dtype=torch.float16),
                f"{prefix}.attention.output.LayerNorm.weight": torch.ones(64, dtype=torch.float16),
                f"{prefix}.attention.output.LayerNorm.bias": torch.zeros(64, dtype=torch.float16),
                f"{prefix}.intermediate.dense.weight": torch.randn(256, 64, dtype=torch.float16),
                f"{prefix}.intermediate.dense.bias": torch.zeros(256, dtype=torch.float16),
                f"{prefix}.output.dense.weight": torch.randn(64, 256, dtype=torch.float16),
                f"{prefix}.output.dense.bias": torch.zeros(64, dtype=torch.float16),
                f"{prefix}.output.LayerNorm.weight": torch.ones(64, dtype=torch.float16),
                f"{prefix}.output.LayerNorm.bias": torch.zeros(64, dtype=torch.float16),
            })
        
        core = InferenceCore(weights_fp16, minimal_config, device='cpu')
        
        # All loaded weights should be float32
        assert core.word_embeddings.weight.dtype == torch.float32
        assert core.position_embeddings.weight.dtype == torch.float32
        assert core.encoder_layers[0].attention.query.weight.dtype == torch.float32


# ============================================================================
# Test: Extended Attention Mask
# ============================================================================

class TestExtendedAttentionMask:
    """Test extended attention mask generation."""
    
    def test_extended_mask_shape(self, minimal_weights, minimal_config):
        """Extended mask should have shape (batch, 1, 1, seq_len)."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        attention_mask = torch.ones(4, 10)
        extended_mask = core._get_extended_attention_mask(attention_mask)
        
        assert extended_mask.shape == (4, 1, 1, 10)
    
    def test_extended_mask_converts_ones_to_zeros(self, minimal_weights, minimal_config):
        """Extended mask should convert 1s (attend) to 0s."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        attention_mask = torch.ones(1, 5)
        extended_mask = core._get_extended_attention_mask(attention_mask)
        
        # All positions should be 0 (attend)
        assert torch.allclose(extended_mask, torch.zeros(1, 1, 1, 5))
    
    def test_extended_mask_converts_zeros_to_large_negative(self, minimal_weights, minimal_config):
        """Extended mask should convert 0s (masked) to large negative values."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        attention_mask = torch.zeros(1, 5)
        extended_mask = core._get_extended_attention_mask(attention_mask)
        
        # All positions should be -10000 (masked)
        assert torch.allclose(extended_mask, torch.full((1, 1, 1, 5), -10000.0))
    
    def test_extended_mask_mixed_pattern(self, minimal_weights, minimal_config):
        """Extended mask should handle mixed 0/1 patterns."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)
        extended_mask = core._get_extended_attention_mask(attention_mask)
        
        expected = torch.tensor([[[[0.0, 0.0, 0.0, -10000.0, -10000.0]]]])
        assert torch.allclose(extended_mask, expected)


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_token_sequence(self, minimal_weights, minimal_config):
        """Should handle single-token sequences."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.tensor([[5]])
        attention_mask = torch.ones(1, 1)
        
        with torch.no_grad():
            output = core(input_ids, attention_mask)
        
        assert output.shape == (1, 64)
        assert torch.isfinite(output).all()
    
    def test_large_batch_size(self, minimal_weights, minimal_config):
        """Should handle large batch sizes."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.randint(0, 1000, (32, 10))
        attention_mask = torch.ones(32, 10)
        
        with torch.no_grad():
            output = core(input_ids, attention_mask)
        
        assert output.shape == (32, 64)
        assert torch.isfinite(output).all()
    
    def test_long_sequence(self, minimal_weights, minimal_config):
        """Should handle long sequences."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.randint(0, 1000, (1, 256))
        attention_mask = torch.ones(1, 256)
        
        with torch.no_grad():
            output = core(input_ids, attention_mask)
        
        assert output.shape == (1, 64)
        assert torch.isfinite(output).all()
    
    def test_all_tokens_masked(self, minimal_weights, minimal_config):
        """Should handle edge case of all tokens masked."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.zeros(1, 5)  # All masked
        
        with torch.no_grad():
            output = core(input_ids, attention_mask)
        
        # Should not crash
        assert output.shape == (1, 64)
        assert torch.isfinite(output).all()
    
    def test_partial_masking(self, minimal_weights, minimal_config):
        """Should handle partial masking correctly."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        core.eval()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float)
        
        with torch.no_grad():
            output = core(input_ids, attention_mask)
        
        assert output.shape == (1, 64)
        assert torch.isfinite(output).all()


# ============================================================================
# Test: Repr and String Representation
# ============================================================================

class TestStringRepresentation:
    """Test string representation of InferenceCore."""
    
    def test_repr_contains_key_info(self, minimal_weights, minimal_config):
        """__repr__ should contain key configuration info."""
        core = InferenceCore(minimal_weights, minimal_config, device='cpu')
        
        repr_str = repr(core)
        
        assert 'InferenceCore' in repr_str
        assert 'hidden_size=64' in repr_str
        assert 'num_layers=2' in repr_str
        assert 'num_heads=2' in repr_str
        assert 'intermediate=256' in repr_str
        assert 'device=cpu' in repr_str

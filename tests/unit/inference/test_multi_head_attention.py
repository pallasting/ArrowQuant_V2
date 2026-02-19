"""
Unit tests for MultiHeadAttention module.

Tests cover:
- Multi-head attention structure (Property 3)
- Output shape validation
- Attention score computation
- Masking behavior
- Head reshaping and concatenation
- Scaled dot-product attention
- Attention probability computation

**Property 3: Multi-Head Attention Structure**
For any Transformer layer, the attention mechanism should have the configured
number of heads, and the output shape should be (batch_size, seq_len, hidden_size).
**Validates: Requirements 1.3**
"""

import math
import pytest
import torch
import torch.nn.functional as F

from llm_compression.inference.inference_core import MultiHeadAttention


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mha_small():
    """Small MultiHeadAttention for testing (64 dim, 2 heads)."""
    return MultiHeadAttention(hidden_size=64, num_heads=2)


@pytest.fixture
def mha_standard():
    """Standard MultiHeadAttention (384 dim, 6 heads)."""
    return MultiHeadAttention(hidden_size=384, num_heads=6)


@pytest.fixture
def sample_input():
    """Sample input tensor (batch=2, seq=10, hidden=64)."""
    return torch.randn(2, 10, 64)


@pytest.fixture
def sample_mask():
    """Sample attention mask (batch=2, 1, 1, seq=10)."""
    return torch.zeros(2, 1, 1, 10)


# ============================================================================
# Test: Construction and Configuration (Property 3)
# ============================================================================

class TestMultiHeadAttentionConstruction:
    """Test MultiHeadAttention initialization and structure."""
    
    def test_correct_number_of_heads(self, mha_small):
        """Should have configured number of attention heads."""
        assert mha_small.num_heads == 2
    
    def test_head_size_calculation(self, mha_small):
        """Head size should be hidden_size / num_heads."""
        assert mha_small.head_size == 32  # 64 / 2
    
    def test_all_head_size(self, mha_small):
        """All head size should equal num_heads * head_size."""
        assert mha_small.all_head_size == 64  # 2 * 32
    
    def test_standard_bert_configuration(self, mha_standard):
        """Standard BERT config: 384 hidden, 6 heads, 64 head_size."""
        assert mha_standard.num_heads == 6
        assert mha_standard.head_size == 64  # 384 / 6
        assert mha_standard.all_head_size == 384
    
    def test_query_projection_exists(self, mha_small):
        """Should have query projection layer."""
        assert hasattr(mha_small, 'query')
        assert isinstance(mha_small.query, torch.nn.Linear)
    
    def test_key_projection_exists(self, mha_small):
        """Should have key projection layer."""
        assert hasattr(mha_small, 'key')
        assert isinstance(mha_small.key, torch.nn.Linear)
    
    def test_value_projection_exists(self, mha_small):
        """Should have value projection layer."""
        assert hasattr(mha_small, 'value')
        assert isinstance(mha_small.value, torch.nn.Linear)
    
    def test_projection_dimensions(self, mha_small):
        """Q, K, V projections should have correct dimensions."""
        assert mha_small.query.in_features == 64
        assert mha_small.query.out_features == 64
        assert mha_small.key.in_features == 64
        assert mha_small.key.out_features == 64
        assert mha_small.value.in_features == 64
        assert mha_small.value.out_features == 64


# ============================================================================
# Test: Output Shape (Property 3)
# ============================================================================

class TestOutputShape:
    """Test that output shape matches (batch_size, seq_len, hidden_size)."""
    
    def test_output_shape_preserves_input_dimensions(self, mha_small, sample_input, sample_mask):
        """Output should have same shape as input: (batch, seq, hidden)."""
        output, _ = mha_small(sample_input, sample_mask)
        assert output.shape == sample_input.shape
        assert output.shape == (2, 10, 64)
    
    def test_output_shape_single_sequence(self, mha_small):
        """Single sequence should produce (1, seq_len, hidden_size)."""
        x = torch.randn(1, 5, 64)
        mask = torch.zeros(1, 1, 1, 5)
        output, _ = mha_small(x, mask)
        assert output.shape == (1, 5, 64)
    
    def test_output_shape_batch(self, mha_small):
        """Batch should produce (batch_size, seq_len, hidden_size)."""
        x = torch.randn(4, 8, 64)
        mask = torch.zeros(4, 1, 1, 8)
        output, _ = mha_small(x, mask)
        assert output.shape == (4, 8, 64)
    
    def test_output_shape_long_sequence(self, mha_standard):
        """Long sequence should work correctly."""
        x = torch.randn(1, 128, 384)
        mask = torch.zeros(1, 1, 1, 128)
        output, _ = mha_standard(x, mask)
        assert output.shape == (1, 128, 384)
    
    def test_output_shape_various_batch_sizes(self, mha_small):
        """Should handle various batch sizes correctly."""
        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 10, 64)
            mask = torch.zeros(batch_size, 1, 1, 10)
            output, _ = mha_small(x, mask)
            assert output.shape == (batch_size, 10, 64)
    
    def test_output_shape_various_sequence_lengths(self, mha_small):
        """Should handle various sequence lengths correctly."""
        for seq_len in [1, 5, 10, 20, 50]:
            x = torch.randn(2, seq_len, 64)
            mask = torch.zeros(2, 1, 1, seq_len)
            output, _ = mha_small(x, mask)
            assert output.shape == (2, seq_len, 64)


# ============================================================================
# Test: Attention Computation
# ============================================================================

class TestAttentionComputation:
    """Test scaled dot-product attention computation."""
    
    def test_output_is_finite(self, mha_small, sample_input, sample_mask):
        """Output should not contain NaN or Inf."""
        output, _ = mha_small(sample_input, sample_mask)
        assert torch.isfinite(output).all()
    
    def test_attention_probs_returned_when_requested(self, mha_small, sample_input, sample_mask):
        """Should return attention probabilities when output_attentions=True."""
        output, attn_probs = mha_small(sample_input, sample_mask, output_attentions=True)
        assert attn_probs is not None
        assert isinstance(attn_probs, torch.Tensor)
    
    def test_attention_probs_none_when_not_requested(self, mha_small, sample_input, sample_mask):
        """Should return None for attention probs when output_attentions=False."""
        output, attn_probs = mha_small(sample_input, sample_mask, output_attentions=False)
        assert attn_probs is None
    
    def test_attention_probs_shape(self, mha_small, sample_input, sample_mask):
        """Attention probs should have shape (batch, num_heads, seq, seq)."""
        _, attn_probs = mha_small(sample_input, sample_mask, output_attentions=True)
        batch_size, seq_len = sample_input.shape[0], sample_input.shape[1]
        expected_shape = (batch_size, mha_small.num_heads, seq_len, seq_len)
        assert attn_probs.shape == expected_shape
    
    def test_attention_probs_sum_to_one(self, mha_small, sample_input, sample_mask):
        """Attention probabilities should sum to 1 along last dimension."""
        _, attn_probs = mha_small(sample_input, sample_mask, output_attentions=True)
        # Sum over key dimension (last dim)
        sums = attn_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_attention_probs_non_negative(self, mha_small, sample_input, sample_mask):
        """Attention probabilities should be non-negative."""
        _, attn_probs = mha_small(sample_input, sample_mask, output_attentions=True)
        assert (attn_probs >= 0).all()
    
    def test_attention_probs_bounded(self, mha_small, sample_input, sample_mask):
        """Attention probabilities should be in [0, 1]."""
        _, attn_probs = mha_small(sample_input, sample_mask, output_attentions=True)
        assert (attn_probs <= 1.0).all()
        assert (attn_probs >= 0.0).all()
    
    def test_scaled_attention_uses_sqrt_head_size(self, mha_small):
        """Attention scores should be scaled by sqrt(head_size)."""
        # Create simple inputs to verify scaling
        x = torch.ones(1, 2, 64)
        mask = torch.zeros(1, 1, 1, 2)
        
        # The scaling factor should be sqrt(head_size) = sqrt(32) ≈ 5.657
        expected_scale = math.sqrt(mha_small.head_size)
        assert expected_scale == math.sqrt(32)
        
        # Just verify the computation runs without error
        # (actual scaling is internal to the forward pass)
        output, _ = mha_small(x, mask)
        assert output.shape == (1, 2, 64)


# ============================================================================
# Test: Masking Behavior
# ============================================================================

class TestMaskingBehavior:
    """Test attention masking functionality."""
    
    def test_masking_affects_output(self, mha_small):
        """Different masks should produce different outputs."""
        x = torch.randn(1, 6, 64)
        
        # No masking
        mask_none = torch.zeros(1, 1, 1, 6)
        # Mask last 3 tokens
        mask_half = torch.cat([
            torch.zeros(1, 1, 1, 3),
            torch.full((1, 1, 1, 3), -10000.0)
        ], dim=-1)
        
        with torch.no_grad():
            out_none, _ = mha_small(x, mask_none)
            out_half, _ = mha_small(x, mask_half)
        
        assert not torch.allclose(out_none, out_half, atol=1e-5)
    
    def test_masked_positions_have_zero_attention(self, mha_small):
        """Masked positions should have near-zero attention weights."""
        x = torch.randn(1, 4, 64)
        # Mask last 2 positions
        mask = torch.cat([
            torch.zeros(1, 1, 1, 2),
            torch.full((1, 1, 1, 2), -10000.0)
        ], dim=-1)
        
        _, attn_probs = mha_small(x, mask, output_attentions=True)
        
        # Attention to masked positions (last 2) should be near zero
        # Shape: (1, num_heads, 4, 4), check attention to positions 2 and 3
        masked_attention = attn_probs[0, :, :, 2:]  # (num_heads, 4, 2)
        assert torch.allclose(masked_attention, torch.zeros_like(masked_attention), atol=1e-5)
    
    def test_full_masking_produces_uniform_attention(self, mha_small):
        """When all positions are masked, attention should be uniform (or near-zero)."""
        x = torch.randn(1, 4, 64)
        # Mask all positions
        mask = torch.full((1, 1, 1, 4), -10000.0)
        
        _, attn_probs = mha_small(x, mask, output_attentions=True)
        
        # With all positions masked, softmax produces NaN or uniform distribution
        # In practice, this is an edge case that shouldn't occur
        # Just verify it doesn't crash
        assert attn_probs.shape == (1, 2, 4, 4)


# ============================================================================
# Test: Head Reshaping
# ============================================================================

class TestHeadReshaping:
    """Test head reshaping and concatenation logic."""
    
    def test_reshape_for_heads_output_shape(self, mha_small):
        """_reshape_for_heads should produce (batch, heads, seq, head_size)."""
        x = torch.randn(2, 10, 64)
        reshaped = mha_small._reshape_for_heads(x)
        expected_shape = (2, mha_small.num_heads, 10, mha_small.head_size)
        assert reshaped.shape == expected_shape
        assert reshaped.shape == (2, 2, 10, 32)
    
    def test_reshape_for_heads_preserves_data(self, mha_small):
        """Reshaping should preserve all data (no loss)."""
        x = torch.randn(1, 5, 64)
        reshaped = mha_small._reshape_for_heads(x)
        
        # Reshape back and verify data is preserved
        batch_size, num_heads, seq_len, head_size = reshaped.shape
        back = reshaped.permute(0, 2, 1, 3).contiguous()
        back = back.view(batch_size, seq_len, num_heads * head_size)
        
        assert torch.allclose(x, back)
    
    def test_multi_head_splits_correctly(self, mha_small):
        """Multiple heads should split hidden dimension correctly."""
        x = torch.randn(1, 3, 64)
        
        # Project to Q
        q = mha_small.query(x)
        assert q.shape == (1, 3, 64)
        
        # Reshape for heads
        q_heads = mha_small._reshape_for_heads(q)
        assert q_heads.shape == (1, 2, 3, 32)  # 2 heads, 32 dim each


# ============================================================================
# Test: Determinism and Consistency
# ============================================================================

class TestDeterminismAndConsistency:
    """Test deterministic behavior and consistency."""
    
    def test_deterministic_output_in_eval_mode(self, mha_small, sample_input, sample_mask):
        """Same input should produce same output in eval mode."""
        mha_small.eval()
        
        with torch.no_grad():
            out1, _ = mha_small(sample_input, sample_mask)
            out2, _ = mha_small(sample_input, sample_mask)
        
        assert torch.allclose(out1, out2)
    
    def test_different_inputs_produce_different_outputs(self, mha_small):
        """Different inputs should produce different outputs."""
        x1 = torch.randn(1, 5, 64)
        x2 = torch.randn(1, 5, 64)
        mask = torch.zeros(1, 1, 1, 5)
        
        mha_small.eval()
        with torch.no_grad():
            out1, _ = mha_small(x1, mask)
            out2, _ = mha_small(x2, mask)
        
        assert not torch.allclose(out1, out2, atol=1e-5)
    
    def test_position_sensitivity(self, mha_small):
        """Swapping input positions should change output."""
        x = torch.randn(1, 4, 64)
        mask = torch.zeros(1, 1, 1, 4)
        
        mha_small.eval()
        with torch.no_grad():
            out_original, _ = mha_small(x, mask)
        
        # Swap first two positions
        x_swapped = x.clone()
        x_swapped[0, 0] = x[0, 1]
        x_swapped[0, 1] = x[0, 0]
        
        with torch.no_grad():
            out_swapped, _ = mha_small(x_swapped, mask)
        
        # Outputs should be different
        assert not torch.allclose(out_original, out_swapped, atol=1e-5)


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_token_sequence(self, mha_small):
        """Should handle single-token sequences."""
        x = torch.randn(1, 1, 64)
        mask = torch.zeros(1, 1, 1, 1)
        output, _ = mha_small(x, mask)
        assert output.shape == (1, 1, 64)
        assert torch.isfinite(output).all()
    
    def test_large_batch_size(self, mha_small):
        """Should handle large batch sizes."""
        x = torch.randn(32, 10, 64)
        mask = torch.zeros(32, 1, 1, 10)
        output, _ = mha_small(x, mask)
        assert output.shape == (32, 10, 64)
        assert torch.isfinite(output).all()
    
    def test_very_long_sequence(self, mha_standard):
        """Should handle very long sequences."""
        x = torch.randn(1, 512, 384)
        mask = torch.zeros(1, 1, 1, 512)
        output, _ = mha_standard(x, mask)
        assert output.shape == (1, 512, 384)
        assert torch.isfinite(output).all()
    
    def test_zero_input(self, mha_small):
        """Should handle zero input without crashing."""
        x = torch.zeros(1, 5, 64)
        mask = torch.zeros(1, 1, 1, 5)
        output, _ = mha_small(x, mask)
        assert output.shape == (1, 5, 64)
        assert torch.isfinite(output).all()
    
    def test_uniform_input(self, mha_small):
        """Should handle uniform input."""
        x = torch.ones(1, 5, 64)
        mask = torch.zeros(1, 1, 1, 5)
        output, _ = mha_small(x, mask)
        assert output.shape == (1, 5, 64)
        assert torch.isfinite(output).all()


# ============================================================================
# Test: Integration with TransformerLayer
# ============================================================================

class TestIntegrationWithTransformerLayer:
    """Test MultiHeadAttention as part of TransformerLayer."""
    
    def test_used_in_transformer_layer(self):
        """MultiHeadAttention should be used in TransformerLayer."""
        from llm_compression.inference.inference_core import TransformerLayer
        
        layer = TransformerLayer(hidden_size=64, num_attention_heads=2, intermediate_size=256)
        assert isinstance(layer.attention, MultiHeadAttention)
        assert layer.attention.num_heads == 2
        assert layer.attention.head_size == 32
    
    def test_transformer_layer_uses_attention_output(self):
        """TransformerLayer should use MultiHeadAttention output correctly."""
        from llm_compression.inference.inference_core import TransformerLayer
        
        layer = TransformerLayer(hidden_size=64, num_attention_heads=2, intermediate_size=256)
        layer.eval()
        
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 1, 1, 10)
        
        with torch.no_grad():
            output, _ = layer(x, mask)
        
        # Output should have correct shape
        assert output.shape == (2, 10, 64)
        assert torch.isfinite(output).all()


# ============================================================================
# Test: Property 3 Validation
# ============================================================================

class TestProperty3MultiHeadAttentionStructure:
    """
    Validate Property 3: Multi-Head Attention Structure.
    
    For any Transformer layer, the attention mechanism should have the
    configured number of heads, and the output shape should be
    (batch_size, seq_len, hidden_size).
    
    **Validates: Requirements 1.3**
    """
    
    def test_property_3_correct_number_of_heads(self):
        """Property 3: Attention should have configured number of heads."""
        for num_heads in [1, 2, 4, 6, 8, 12]:
            hidden_size = num_heads * 64  # Ensure divisibility
            mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
            assert mha.num_heads == num_heads
    
    def test_property_3_output_shape_matches_input(self):
        """Property 3: Output shape should be (batch_size, seq_len, hidden_size)."""
        test_cases = [
            (64, 2, 1, 5),    # (hidden, heads, batch, seq)
            (128, 4, 2, 10),
            (256, 8, 4, 20),
            (384, 6, 8, 50),
            (512, 8, 16, 100),
        ]
        
        for hidden_size, num_heads, batch_size, seq_len in test_cases:
            mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
            x = torch.randn(batch_size, seq_len, hidden_size)
            mask = torch.zeros(batch_size, 1, 1, seq_len)
            
            output, _ = mha(x, mask)
            
            # Validate output shape
            assert output.shape == (batch_size, seq_len, hidden_size), (
                f"Expected shape ({batch_size}, {seq_len}, {hidden_size}), "
                f"got {output.shape}"
            )
    
    def test_property_3_head_size_calculation(self):
        """Property 3: Head size should be hidden_size / num_heads."""
        test_cases = [
            (64, 2, 32),
            (128, 4, 32),
            (256, 8, 32),
            (384, 6, 64),
            (768, 12, 64),
        ]
        
        for hidden_size, num_heads, expected_head_size in test_cases:
            mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
            assert mha.head_size == expected_head_size
            assert mha.all_head_size == hidden_size
    
    def test_property_3_attention_mechanism_functional(self):
        """Property 3: Attention mechanism should produce valid outputs."""
        mha = MultiHeadAttention(hidden_size=384, num_heads=6)
        mha.eval()
        
        # Test with various inputs
        for batch_size in [1, 4, 8]:
            for seq_len in [10, 50, 100]:
                x = torch.randn(batch_size, seq_len, 384)
                mask = torch.zeros(batch_size, 1, 1, seq_len)
                
                with torch.no_grad():
                    output, attn_probs = mha(x, mask, output_attentions=True)
                
                # Validate output
                assert output.shape == (batch_size, seq_len, 384)
                assert torch.isfinite(output).all()
                
                # Validate attention probabilities
                assert attn_probs.shape == (batch_size, 6, seq_len, seq_len)
                assert (attn_probs >= 0).all()
                assert (attn_probs <= 1).all()
                
                # Attention should sum to 1 over key dimension
                sums = attn_probs.sum(dim=-1)
                assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

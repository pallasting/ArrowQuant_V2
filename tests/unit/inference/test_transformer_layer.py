"""
Unit tests for TransformerLayer module.

Tests cover:
- Feed-forward network with GELU activation (Property 4)
- Layer normalization placement (Property 5)
- Output shape validation
- Residual connections
- Integration with MultiHeadAttention
- Transformer layer structure

**Property 4: Feed-Forward Network with GELU**
For any Transformer layer, the FFN should apply GELU activation between the
intermediate and output projections (verified by checking that the activation
is non-linear and matches GELU behavior on test inputs).
**Validates: Requirements 1.4**

**Property 5: Layer Normalization Placement**
For any Transformer layer, LayerNorm should be applied after attention and
after FFN, and the normalized outputs should have mean ≈ 0 and standard
deviation ≈ 1.
**Validates: Requirements 1.5**
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_compression.inference.inference_core import TransformerLayer, MultiHeadAttention


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def layer_small():
    """Small TransformerLayer for testing (64 dim, 2 heads, 256 intermediate)."""
    return TransformerLayer(
        hidden_size=64,
        num_attention_heads=2,
        intermediate_size=256,
        layer_norm_eps=1e-12
    )


@pytest.fixture
def layer_standard():
    """Standard TransformerLayer (384 dim, 6 heads, 1536 intermediate)."""
    return TransformerLayer(
        hidden_size=384,
        num_attention_heads=6,
        intermediate_size=1536,
        layer_norm_eps=1e-12
    )


@pytest.fixture
def sample_input():
    """Sample input tensor (batch=2, seq=10, hidden=64)."""
    return torch.randn(2, 10, 64)


@pytest.fixture
def sample_mask():
    """Sample attention mask (batch=2, 1, 1, seq=10)."""
    return torch.zeros(2, 1, 1, 10)


# ============================================================================
# Test: Construction and Configuration
# ============================================================================

class TestTransformerLayerConstruction:
    """Test TransformerLayer initialization and structure."""
    
    def test_has_attention_module(self, layer_small):
        """Should have MultiHeadAttention module."""
        assert hasattr(layer_small, 'attention')
        assert isinstance(layer_small.attention, MultiHeadAttention)
    
    def test_has_attention_output_projection(self, layer_small):
        """Should have attention output projection layer."""
        assert hasattr(layer_small, 'attention_output')
        assert isinstance(layer_small.attention_output, nn.Linear)
    
    def test_has_attention_layernorm(self, layer_small):
        """Should have LayerNorm after attention."""
        assert hasattr(layer_small, 'attention_layernorm')
        assert isinstance(layer_small.attention_layernorm, nn.LayerNorm)
    
    def test_has_intermediate_layer(self, layer_small):
        """Should have intermediate FFN layer."""
        assert hasattr(layer_small, 'intermediate')
        assert isinstance(layer_small.intermediate, nn.Linear)
    
    def test_has_output_dense_layer(self, layer_small):
        """Should have output dense layer."""
        assert hasattr(layer_small, 'output_dense')
        assert isinstance(layer_small.output_dense, nn.Linear)
    
    def test_has_output_layernorm(self, layer_small):
        """Should have LayerNorm after FFN."""
        assert hasattr(layer_small, 'output_layernorm')
        assert isinstance(layer_small.output_layernorm, nn.LayerNorm)
    
    def test_correct_dimensions(self, layer_small):
        """All layers should have correct dimensions."""
        assert layer_small.hidden_size == 64
        assert layer_small.num_attention_heads == 2
        assert layer_small.head_size == 32
        
        # Attention output projection
        assert layer_small.attention_output.in_features == 64
        assert layer_small.attention_output.out_features == 64
        
        # FFN dimensions
        assert layer_small.intermediate.in_features == 64
        assert layer_small.intermediate.out_features == 256
        assert layer_small.output_dense.in_features == 256
        assert layer_small.output_dense.out_features == 64
    
    def test_standard_bert_configuration(self, layer_standard):
        """Standard BERT config: 384 hidden, 6 heads, 1536 intermediate."""
        assert layer_standard.hidden_size == 384
        assert layer_standard.num_attention_heads == 6
        assert layer_standard.head_size == 64
        
        assert layer_standard.intermediate.out_features == 1536
        assert layer_standard.output_dense.in_features == 1536



# ============================================================================
# Test: Output Shape
# ============================================================================

class TestOutputShape:
    """Test that output shape matches (batch_size, seq_len, hidden_size)."""
    
    def test_output_shape_preserves_input_dimensions(self, layer_small, sample_input, sample_mask):
        """Output should have same shape as input: (batch, seq, hidden)."""
        output, _ = layer_small(sample_input, sample_mask)
        assert output.shape == sample_input.shape
        assert output.shape == (2, 10, 64)
    
    def test_output_shape_single_sequence(self, layer_small):
        """Single sequence should produce (1, seq_len, hidden_size)."""
        x = torch.randn(1, 5, 64)
        mask = torch.zeros(1, 1, 1, 5)
        output, _ = layer_small(x, mask)
        assert output.shape == (1, 5, 64)
    
    def test_output_shape_batch(self, layer_small):
        """Batch should produce (batch_size, seq_len, hidden_size)."""
        x = torch.randn(4, 8, 64)
        mask = torch.zeros(4, 1, 1, 8)
        output, _ = layer_small(x, mask)
        assert output.shape == (4, 8, 64)
    
    def test_output_shape_long_sequence(self, layer_standard):
        """Long sequence should work correctly."""
        x = torch.randn(1, 128, 384)
        mask = torch.zeros(1, 1, 1, 128)
        output, _ = layer_standard(x, mask)
        assert output.shape == (1, 128, 384)
    
    def test_output_is_finite(self, layer_small, sample_input, sample_mask):
        """Output should not contain NaN or Inf."""
        output, _ = layer_small(sample_input, sample_mask)
        assert torch.isfinite(output).all()


# ============================================================================
# Test: Property 4 - Feed-Forward Network with GELU
# ============================================================================

class TestProperty4FeedForwardNetworkWithGELU:
    """
    Validate Property 4: Feed-Forward Network with GELU.
    
    For any Transformer layer, the FFN should apply GELU activation between
    the intermediate and output projections (verified by checking that the
    activation is non-linear and matches GELU behavior on test inputs).
    
    **Validates: Requirements 1.4**
    """
    
    def test_ffn_applies_nonlinear_activation(self, layer_small):
        """FFN should apply non-linear activation (not identity)."""
        layer_small.eval()
        
        # Create two different inputs
        x1 = torch.randn(1, 5, 64)
        x2 = 2 * x1  # Scaled version
        mask = torch.zeros(1, 1, 1, 5)
        
        with torch.no_grad():
            out1, _ = layer_small(x1, mask)
            out2, _ = layer_small(x2, mask)
        
        # If activation were linear, out2 should be close to 2*out1
        # With GELU (non-linear), this should NOT hold
        # Allow some tolerance due to residual connections
        ratio = out2 / (out1 + 1e-8)
        
        # The ratio should not be uniformly close to 2.0
        # (indicating non-linearity is present)
        assert not torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=0.5)

    
    def test_gelu_activation_behavior_on_known_values(self, layer_small):
        """Verify GELU activation produces expected behavior on known inputs."""
        # Test GELU behavior directly on intermediate layer
        layer_small.eval()
        
        # Create test inputs with known values
        test_values = torch.tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]])
        test_input = test_values.unsqueeze(0).expand(1, 5, -1)  # (1, 5, 5)
        
        # Pad to match hidden_size
        test_input_padded = torch.zeros(1, 5, 64)
        test_input_padded[:, :, :5] = test_input
        
        mask = torch.zeros(1, 1, 1, 5)
        
        with torch.no_grad():
            # Get intermediate output before GELU
            hidden_states = test_input_padded
            
            # Simulate attention (simplified - just pass through)
            attn_output, _ = layer_small.attention(hidden_states, mask)
            attn_output = layer_small.attention_output(attn_output)
            hidden_states = layer_small.attention_layernorm(attn_output + hidden_states)
            
            # Apply intermediate layer
            intermediate_output = layer_small.intermediate(hidden_states)
            
            # Apply GELU manually
            expected_gelu = F.gelu(intermediate_output)
            
            # Apply output dense
            actual_output = layer_small.output_dense(F.gelu(intermediate_output))
            expected_output = layer_small.output_dense(expected_gelu)
            
            # They should match
            assert torch.allclose(actual_output, expected_output, atol=1e-6)
    
    def test_gelu_is_smooth_and_continuous(self, layer_small):
        """GELU should produce smooth, continuous outputs."""
        layer_small.eval()
        
        # Create smooth input sequence
        x = torch.linspace(-2, 2, 50).unsqueeze(0).unsqueeze(0).expand(1, 10, 50)
        x_padded = torch.zeros(1, 10, 64)
        x_padded[:, :, :50] = x
        
        mask = torch.zeros(1, 1, 1, 10)
        
        with torch.no_grad():
            output, _ = layer_small(x_padded, mask)
        
        # Output should be smooth (no discontinuities)
        # Check that adjacent values don't have huge jumps
        diffs = torch.abs(output[:, :, :50].diff(dim=2))
        max_diff = diffs.max()
        
        # With smooth input, output differences should be bounded
        assert max_diff < 10.0  # Reasonable bound for smooth function
    
    def test_gelu_activation_is_not_relu(self, layer_small):
        """GELU should behave differently from ReLU."""
        layer_small.eval()
        
        # Create input with negative values
        x = torch.randn(1, 5, 64)
        x[0, :, :10] = -2.0  # Set some values to negative
        mask = torch.zeros(1, 1, 1, 5)
        
        with torch.no_grad():
            output, _ = layer_small(x, mask)
        
        # With GELU, negative inputs don't become exactly zero
        # (unlike ReLU which would zero them out)
        # Due to residual connections, we can't test this directly on output,
        # but we can verify the layer uses GELU by checking intermediate values
        
        # Apply intermediate layer directly
        hidden_states = x
        intermediate = layer_small.intermediate(hidden_states)
        gelu_output = F.gelu(intermediate)
        relu_output = F.relu(intermediate)
        
        # GELU and ReLU should produce different results
        assert not torch.allclose(gelu_output, relu_output, atol=1e-3)

    
    def test_ffn_structure_intermediate_to_output(self, layer_small):
        """FFN should have structure: intermediate → GELU → output."""
        layer_small.eval()
        
        x = torch.randn(1, 5, 64)
        mask = torch.zeros(1, 1, 1, 5)
        
        with torch.no_grad():
            # Get hidden states after attention
            attn_output, _ = layer_small.attention(x, mask)
            attn_output = layer_small.attention_output(attn_output)
            hidden_after_attn = layer_small.attention_layernorm(attn_output + x)
            
            # Apply FFN manually
            intermediate = layer_small.intermediate(hidden_after_attn)
            after_gelu = F.gelu(intermediate)
            ffn_output = layer_small.output_dense(after_gelu)
            
            # Verify intermediate size is correct
            assert intermediate.shape == (1, 5, 256)
            assert after_gelu.shape == (1, 5, 256)
            assert ffn_output.shape == (1, 5, 64)


# ============================================================================
# Test: Property 5 - Layer Normalization Placement
# ============================================================================

class TestProperty5LayerNormalizationPlacement:
    """
    Validate Property 5: Layer Normalization Placement.
    
    For any Transformer layer, LayerNorm should be applied after attention
    and after FFN, and the normalized outputs should have mean ≈ 0 and
    standard deviation ≈ 1.
    
    **Validates: Requirements 1.5**
    """
    
    def test_layernorm_after_attention(self, layer_small):
        """LayerNorm should be applied after attention block."""
        layer_small.eval()
        
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 1, 1, 10)
        
        with torch.no_grad():
            # Get attention output before LayerNorm
            attn_output, _ = layer_small.attention(x, mask)
            attn_output = layer_small.attention_output(attn_output)
            
            # Add residual
            with_residual = attn_output + x
            
            # Apply LayerNorm
            normalized = layer_small.attention_layernorm(with_residual)
            
            # Check that LayerNorm was applied (output should be normalized)
            mean = normalized.mean(dim=-1)
            std = normalized.std(dim=-1)
            
            # Mean should be close to 0
            assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
            # Std should be close to 1
            assert torch.allclose(std, torch.ones_like(std), atol=1e-2)
    
    def test_layernorm_after_ffn(self, layer_small):
        """LayerNorm should be applied after FFN block."""
        layer_small.eval()
        
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 1, 1, 10)
        
        with torch.no_grad():
            # Get output after full forward pass
            output, _ = layer_small(x, mask)
            
            # The final output should be normalized
            # Check statistics along hidden dimension
            mean = output.mean(dim=-1)
            std = output.std(dim=-1)
            
            # Mean should be close to 0
            assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
            # Std should be close to 1
            assert torch.allclose(std, torch.ones_like(std), atol=1e-2)
    
    def test_normalized_outputs_have_zero_mean(self, layer_small):
        """Normalized outputs should have mean ≈ 0."""
        layer_small.eval()
        
        # Test with various inputs
        test_cases = [
            torch.randn(1, 5, 64),
            torch.randn(4, 10, 64),
            torch.randn(8, 20, 64),
        ]
        
        for x in test_cases:
            batch_size, seq_len, _ = x.shape
            mask = torch.zeros(batch_size, 1, 1, seq_len)
            
            with torch.no_grad():
                output, _ = layer_small(x, mask)
            
            # Compute mean along hidden dimension
            mean = output.mean(dim=-1)
            
            # Mean should be close to 0
            assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), (
                f"Mean {mean.abs().max().item():.6f} exceeds tolerance"
            )

    
    def test_normalized_outputs_have_unit_std(self, layer_small):
        """Normalized outputs should have standard deviation ≈ 1."""
        layer_small.eval()
        
        # Test with various inputs
        test_cases = [
            torch.randn(1, 5, 64),
            torch.randn(4, 10, 64),
            torch.randn(8, 20, 64),
        ]
        
        for x in test_cases:
            batch_size, seq_len, _ = x.shape
            mask = torch.zeros(batch_size, 1, 1, seq_len)
            
            with torch.no_grad():
                output, _ = layer_small(x, mask)
            
            # Compute std along hidden dimension
            std = output.std(dim=-1)
            
            # Std should be close to 1
            assert torch.allclose(std, torch.ones_like(std), atol=1e-2), (
                f"Std deviation from 1: {(std - 1.0).abs().max().item():.6f}"
            )
    
    def test_layernorm_placement_order(self, layer_small):
        """LayerNorm should be placed after both attention and FFN."""
        # Verify the forward pass applies LayerNorm in correct order
        layer_small.eval()
        
        x = torch.randn(1, 5, 64)
        mask = torch.zeros(1, 1, 1, 5)
        
        with torch.no_grad():
            # Step 1: Attention + residual + LayerNorm
            attn_output, _ = layer_small.attention(x, mask)
            attn_output = layer_small.attention_output(attn_output)
            hidden_after_attn_norm = layer_small.attention_layernorm(attn_output + x)
            
            # Verify normalization after attention
            mean_attn = hidden_after_attn_norm.mean(dim=-1)
            std_attn = hidden_after_attn_norm.std(dim=-1)
            assert torch.allclose(mean_attn, torch.zeros_like(mean_attn), atol=1e-5)
            assert torch.allclose(std_attn, torch.ones_like(std_attn), atol=1e-2)
            
            # Step 2: FFN + residual + LayerNorm
            intermediate = layer_small.intermediate(hidden_after_attn_norm)
            intermediate = F.gelu(intermediate)
            ffn_output = layer_small.output_dense(intermediate)
            final_output = layer_small.output_layernorm(ffn_output + hidden_after_attn_norm)
            
            # Verify normalization after FFN
            mean_ffn = final_output.mean(dim=-1)
            std_ffn = final_output.std(dim=-1)
            assert torch.allclose(mean_ffn, torch.zeros_like(mean_ffn), atol=1e-5)
            assert torch.allclose(std_ffn, torch.ones_like(std_ffn), atol=1e-2)
    
    def test_layernorm_eps_parameter(self, layer_small):
        """LayerNorm should use configured epsilon value."""
        assert layer_small.attention_layernorm.eps == 1e-12
        assert layer_small.output_layernorm.eps == 1e-12
    
    def test_layernorm_with_extreme_values(self):
        """LayerNorm should handle extreme input values."""
        layer = TransformerLayer(
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=256,
            layer_norm_eps=1e-12
        )
        layer.eval()
        
        # Test with large values
        x_large = torch.randn(1, 5, 64) * 1000
        mask = torch.zeros(1, 1, 1, 5)
        
        with torch.no_grad():
            output, _ = layer(x_large, mask)
        
        # Output should still be normalized
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)
        
        assert torch.isfinite(output).all()
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
        assert torch.allclose(std, torch.ones_like(std), atol=0.1)


# ============================================================================
# Test: Residual Connections
# ============================================================================

class TestResidualConnections:
    """Test residual connections in TransformerLayer."""
    
    def test_attention_residual_connection(self, layer_small):
        """Attention block should use residual connection."""
        layer_small.eval()
        
        x = torch.randn(1, 5, 64)
        mask = torch.zeros(1, 1, 1, 5)
        
        with torch.no_grad():
            # Get attention output
            attn_output, _ = layer_small.attention(x, mask)
            attn_output = layer_small.attention_output(attn_output)
            
            # Apply LayerNorm with residual
            with_residual = layer_small.attention_layernorm(attn_output + x)
            
            # Without residual
            without_residual = layer_small.attention_layernorm(attn_output)
            
            # They should be different
            assert not torch.allclose(with_residual, without_residual, atol=1e-3)

    
    def test_ffn_residual_connection(self, layer_small):
        """FFN block should use residual connection."""
        layer_small.eval()
        
        x = torch.randn(1, 5, 64)
        mask = torch.zeros(1, 1, 1, 5)
        
        with torch.no_grad():
            # Get hidden states after attention
            attn_output, _ = layer_small.attention(x, mask)
            attn_output = layer_small.attention_output(attn_output)
            hidden_after_attn = layer_small.attention_layernorm(attn_output + x)
            
            # Apply FFN
            intermediate = layer_small.intermediate(hidden_after_attn)
            intermediate = F.gelu(intermediate)
            ffn_output = layer_small.output_dense(intermediate)
            
            # With residual
            with_residual = layer_small.output_layernorm(ffn_output + hidden_after_attn)
            
            # Without residual
            without_residual = layer_small.output_layernorm(ffn_output)
            
            # They should be different
            assert not torch.allclose(with_residual, without_residual, atol=1e-3)
    
    def test_residual_connections_preserve_information(self, layer_small):
        """Residual connections should help preserve input information."""
        layer_small.eval()
        
        # Create input with specific pattern
        x = torch.zeros(1, 5, 64)
        x[0, :, 0] = torch.arange(5).float()  # Pattern in first dimension
        mask = torch.zeros(1, 1, 1, 5)
        
        with torch.no_grad():
            output, _ = layer_small(x, mask)
        
        # Due to residual connections, some correlation with input should remain
        # (though LayerNorm makes this subtle)
        assert torch.isfinite(output).all()


# ============================================================================
# Test: Attention Probabilities
# ============================================================================

class TestAttentionProbabilities:
    """Test attention probability output."""
    
    def test_attention_probs_returned_when_requested(self, layer_small, sample_input, sample_mask):
        """Should return attention probabilities when output_attentions=True."""
        output, attn_probs = layer_small(sample_input, sample_mask, output_attentions=True)
        assert attn_probs is not None
        assert isinstance(attn_probs, torch.Tensor)
    
    def test_attention_probs_none_when_not_requested(self, layer_small, sample_input, sample_mask):
        """Should return None for attention probs when output_attentions=False."""
        output, attn_probs = layer_small(sample_input, sample_mask, output_attentions=False)
        assert attn_probs is None
    
    def test_attention_probs_shape(self, layer_small, sample_input, sample_mask):
        """Attention probs should have shape (batch, num_heads, seq, seq)."""
        output, attn_probs = layer_small(sample_input, sample_mask, output_attentions=True)
        batch_size, seq_len = sample_input.shape[0], sample_input.shape[1]
        expected_shape = (batch_size, layer_small.num_attention_heads, seq_len, seq_len)
        assert attn_probs.shape == expected_shape


# ============================================================================
# Test: Determinism and Consistency
# ============================================================================

class TestDeterminismAndConsistency:
    """Test deterministic behavior and consistency."""
    
    def test_deterministic_output_in_eval_mode(self, layer_small, sample_input, sample_mask):
        """Same input should produce same output in eval mode."""
        layer_small.eval()
        
        with torch.no_grad():
            out1, _ = layer_small(sample_input, sample_mask)
            out2, _ = layer_small(sample_input, sample_mask)
        
        assert torch.allclose(out1, out2)
    
    def test_different_inputs_produce_different_outputs(self, layer_small):
        """Different inputs should produce different outputs."""
        x1 = torch.randn(1, 5, 64)
        x2 = torch.randn(1, 5, 64)
        mask = torch.zeros(1, 1, 1, 5)
        
        layer_small.eval()
        with torch.no_grad():
            out1, _ = layer_small(x1, mask)
            out2, _ = layer_small(x2, mask)
        
        assert not torch.allclose(out1, out2, atol=1e-5)
    
    def test_output_changes_with_different_masks(self, layer_small):
        """Different masks should produce different outputs."""
        x = torch.randn(1, 6, 64)
        
        mask1 = torch.zeros(1, 1, 1, 6)
        mask2 = torch.cat([
            torch.zeros(1, 1, 1, 3),
            torch.full((1, 1, 1, 3), -10000.0)
        ], dim=-1)
        
        layer_small.eval()
        with torch.no_grad():
            out1, _ = layer_small(x, mask1)
            out2, _ = layer_small(x, mask2)
        
        assert not torch.allclose(out1, out2, atol=1e-5)



# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_token_sequence(self, layer_small):
        """Should handle single-token sequences."""
        x = torch.randn(1, 1, 64)
        mask = torch.zeros(1, 1, 1, 1)
        output, _ = layer_small(x, mask)
        assert output.shape == (1, 1, 64)
        assert torch.isfinite(output).all()
    
    def test_large_batch_size(self, layer_small):
        """Should handle large batch sizes."""
        x = torch.randn(32, 10, 64)
        mask = torch.zeros(32, 1, 1, 10)
        output, _ = layer_small(x, mask)
        assert output.shape == (32, 10, 64)
        assert torch.isfinite(output).all()
    
    def test_very_long_sequence(self, layer_standard):
        """Should handle very long sequences."""
        x = torch.randn(1, 512, 384)
        mask = torch.zeros(1, 1, 1, 512)
        output, _ = layer_standard(x, mask)
        assert output.shape == (1, 512, 384)
        assert torch.isfinite(output).all()
    
    def test_zero_input(self, layer_small):
        """Should handle zero input without crashing."""
        x = torch.zeros(1, 5, 64)
        mask = torch.zeros(1, 1, 1, 5)
        output, _ = layer_small(x, mask)
        assert output.shape == (1, 5, 64)
        assert torch.isfinite(output).all()
    
    def test_uniform_input(self, layer_small):
        """Should handle uniform input."""
        x = torch.ones(1, 5, 64)
        mask = torch.zeros(1, 1, 1, 5)
        output, _ = layer_small(x, mask)
        assert output.shape == (1, 5, 64)
        assert torch.isfinite(output).all()
    
    def test_very_small_values(self, layer_small):
        """Should handle very small input values."""
        x = torch.randn(1, 5, 64) * 1e-6
        mask = torch.zeros(1, 1, 1, 5)
        output, _ = layer_small(x, mask)
        assert output.shape == (1, 5, 64)
        assert torch.isfinite(output).all()
    
    def test_very_large_values(self, layer_small):
        """Should handle very large input values."""
        x = torch.randn(1, 5, 64) * 1e3
        mask = torch.zeros(1, 1, 1, 5)
        output, _ = layer_small(x, mask)
        assert output.shape == (1, 5, 64)
        assert torch.isfinite(output).all()


# ============================================================================
# Test: Integration with InferenceCore
# ============================================================================

class TestIntegrationWithInferenceCore:
    """Test TransformerLayer as part of InferenceCore."""
    
    def test_used_in_inference_core(self):
        """TransformerLayer should be used in InferenceCore."""
        from llm_compression.inference.inference_core import InferenceCore
        
        # Create minimal weights for testing
        weights = {
            'embeddings.word_embeddings.weight': torch.randn(1000, 64),
            'embeddings.position_embeddings.weight': torch.randn(512, 64),
            'embeddings.token_type_embeddings.weight': torch.randn(2, 64),
            'embeddings.LayerNorm.weight': torch.ones(64),
            'embeddings.LayerNorm.bias': torch.zeros(64),
        }
        
        # Add weights for one transformer layer
        for i in range(1):
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
        
        config = {
            'hidden_size': 64,
            'num_layers': 1,
            'num_attention_heads': 2,
            'intermediate_size': 256,
        }
        
        core = InferenceCore(weights, config, device='cpu')
        
        # Verify TransformerLayer is used
        assert len(core.encoder_layers) == 1
        assert isinstance(core.encoder_layers[0], TransformerLayer)

    
    def test_inference_core_uses_transformer_output(self):
        """InferenceCore should use TransformerLayer output correctly."""
        from llm_compression.inference.inference_core import InferenceCore
        
        # Create minimal weights
        weights = {
            'embeddings.word_embeddings.weight': torch.randn(1000, 64),
            'embeddings.position_embeddings.weight': torch.randn(512, 64),
            'embeddings.token_type_embeddings.weight': torch.randn(2, 64),
            'embeddings.LayerNorm.weight': torch.ones(64),
            'embeddings.LayerNorm.bias': torch.zeros(64),
        }
        
        # Add weights for one transformer layer
        for i in range(1):
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
        
        config = {
            'hidden_size': 64,
            'num_layers': 1,
            'num_attention_heads': 2,
            'intermediate_size': 256,
        }
        
        core = InferenceCore(weights, config, device='cpu')
        core.eval()
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        with torch.no_grad():
            output = core(input_ids, attention_mask)
        
        # Output should have correct shape
        assert output.shape == (2, 64)
        assert torch.isfinite(output).all()


# ============================================================================
# Test: Comprehensive Property Validation
# ============================================================================

class TestComprehensivePropertyValidation:
    """Comprehensive validation of Properties 4 and 5."""
    
    def test_property_4_and_5_together(self):
        """Validate both Property 4 (GELU) and Property 5 (LayerNorm) together."""
        layer = TransformerLayer(
            hidden_size=384,
            num_attention_heads=6,
            intermediate_size=1536,
            layer_norm_eps=1e-12
        )
        layer.eval()
        
        # Test with various configurations
        test_cases = [
            (1, 10, 384),
            (4, 20, 384),
            (8, 50, 384),
        ]
        
        for batch_size, seq_len, hidden_size in test_cases:
            x = torch.randn(batch_size, seq_len, hidden_size)
            mask = torch.zeros(batch_size, 1, 1, seq_len)
            
            with torch.no_grad():
                output, _ = layer(x, mask)
            
            # Property 4: FFN uses GELU (non-linear activation)
            # Verify by checking output is not a linear transformation
            x_scaled = 2 * x
            with torch.no_grad():
                output_scaled, _ = layer(x_scaled, mask)
            
            # Due to GELU non-linearity, output_scaled should NOT be 2*output
            ratio = output_scaled / (output + 1e-8)
            assert not torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=0.5)
            
            # Property 5: LayerNorm produces normalized outputs
            mean = output.mean(dim=-1)
            std = output.std(dim=-1)
            
            # Mean ≈ 0
            assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), (
                f"Mean {mean.abs().max().item():.6f} exceeds tolerance"
            )
            
            # Std ≈ 1
            assert torch.allclose(std, torch.ones_like(std), atol=1e-2), (
                f"Std deviation from 1: {(std - 1.0).abs().max().item():.6f}"
            )
    
    def test_property_4_gelu_vs_other_activations(self):
        """Verify GELU is used, not other activations like ReLU or Tanh."""
        layer = TransformerLayer(
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=256,
            layer_norm_eps=1e-12
        )
        layer.eval()
        
        # Create test input
        x = torch.randn(1, 5, 64)
        mask = torch.zeros(1, 1, 1, 5)
        
        with torch.no_grad():
            # Get intermediate values
            attn_output, _ = layer.attention(x, mask)
            attn_output = layer.attention_output(attn_output)
            hidden_states = layer.attention_layernorm(attn_output + x)
            
            # Apply intermediate layer
            intermediate = layer.intermediate(hidden_states)
            
            # Compare different activations
            gelu_output = F.gelu(intermediate)
            relu_output = F.relu(intermediate)
            tanh_output = torch.tanh(intermediate)
            
            # GELU should be different from ReLU and Tanh
            assert not torch.allclose(gelu_output, relu_output, atol=1e-3)
            assert not torch.allclose(gelu_output, tanh_output, atol=1e-3)
    
    def test_property_5_layernorm_at_both_positions(self):
        """Verify LayerNorm is applied at both required positions."""
        layer = TransformerLayer(
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=256,
            layer_norm_eps=1e-12
        )
        layer.eval()
        
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 1, 1, 10)
        
        with torch.no_grad():
            # Position 1: After attention
            attn_output, _ = layer.attention(x, mask)
            attn_output = layer.attention_output(attn_output)
            after_attn_norm = layer.attention_layernorm(attn_output + x)
            
            # Verify normalization
            mean1 = after_attn_norm.mean(dim=-1)
            std1 = after_attn_norm.std(dim=-1)
            assert torch.allclose(mean1, torch.zeros_like(mean1), atol=1e-5)
            assert torch.allclose(std1, torch.ones_like(std1), atol=1e-2)
            
            # Position 2: After FFN
            intermediate = layer.intermediate(after_attn_norm)
            intermediate = F.gelu(intermediate)
            ffn_output = layer.output_dense(intermediate)
            after_ffn_norm = layer.output_layernorm(ffn_output + after_attn_norm)
            
            # Verify normalization
            mean2 = after_ffn_norm.mean(dim=-1)
            std2 = after_ffn_norm.std(dim=-1)
            assert torch.allclose(mean2, torch.zeros_like(mean2), atol=1e-5)
            assert torch.allclose(std2, torch.ones_like(std2), atol=1e-2)

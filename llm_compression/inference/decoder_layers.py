import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Normalization used in Llama/Qwen."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, dim]
        norm = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_normed

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, rope_scaling: Optional[dict] = None) -> torch.Tensor:
    """Precompute the frequency tensor for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    if rope_scaling is not None and rope_scaling.get("factor", 1.0) != 1.0:
        t = t / rope_scaling["factor"]
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape frequency tensor to match the shape of the input tensor x."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to queries and keys using a more compatible rotate_half style."""
    # xq: [batch, seq_len, num_heads, head_dim]
    # freqs_cis: [seq_len, head_dim] (pre-expanded from polar to cos/sin)
    
    # In precompute_freqs_cis, we used torch.polar. Let's simplify and use cos/sin directly.
    cos = freqs_cis.real.view(1, freqs_cis.shape[0], 1, -1)
    sin = freqs_cis.imag.view(1, freqs_cis.shape[0], 1, -1)
    
    def rotate_half(x):
        """Rotates half the hidden dims of the input using split (Llama/Qwen style)."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # For split-style, cos and sin also need to be concatenated, not interleaved
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)

    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class KVCache:
    """Simple KV cache container for autoregressive generation."""
    def __init__(self, max_batch_size: int, max_seq_len: int, num_kv_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype):
        # Allow preallocating largest possible batch size and sequence length
        self.k_cache = torch.zeros((max_batch_size, max_seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)
        self.v_cache = torch.zeros((max_batch_size, max_seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)
        self.seq_len = 0

    def update(self, k_val: torch.Tensor, v_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return concatenated context keys and values."""
        bsz, seqlen, num_heads, head_dim = k_val.shape
        self.k_cache[:bsz, self.seq_len : self.seq_len + seqlen] = k_val
        self.v_cache[:bsz, self.seq_len : self.seq_len + seqlen] = v_val
        self.seq_len += seqlen
        return (
            self.k_cache[:bsz, :self.seq_len],
            self.v_cache[:bsz, :self.seq_len],
        )
    
    def reset(self):
        self.seq_len = 0

class Attention(nn.Module):
    """Grouped-Query Attention (GQA) used by Qwen/Llama."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if kv_cache is not None:
            xk, xv = kv_cache.update(xk, xv)

        # GQA: repeat k/v heads if num_kv_heads < num_heads
        if self.num_key_value_groups > 1:
            xk = torch.repeat_interleave(xk, self.num_key_value_groups, dim=2)
            xv = torch.repeat_interleave(xv, self.num_key_value_groups, dim=2)

        # Transpose to [bsz, num_heads, seqlen, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # mask elements should be 0 or -inf

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # [bsz, num_heads, seqlen, head_dim]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.o_proj(output)

class MLP(nn.Module):
    """SwiGLU Activation MLP used by Qwen/Llama."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    """A standard causal transformer decoder layer."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, intermediate_size: int, rms_norm_eps: float):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x), freqs_cis, kv_cache, mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

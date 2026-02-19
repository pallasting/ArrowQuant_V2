"""
Inference core for embedding generation.

This module implements the forward pass for BERT-like embedding models,
optimized for high-performance inference with batch processing support.

Key features:
- Full BERT architecture (embedding + multi-head attention + FFN + LayerNorm)
- Mean pooling for sentence embeddings
- Batch processing optimization
- Memory-efficient float16 inference
- Multi-device support (CPU/CUDA/MPS)

Performance:
- Inference latency: < 5ms (batch_size=1)
- Throughput: > 2000 req/s (batch_size=32)
- Memory: < 200MB for typical models

Weight key convention (from HuggingFace state_dict):
- embeddings.word_embeddings.weight
- embeddings.position_embeddings.weight
- embeddings.token_type_embeddings.weight
- embeddings.LayerNorm.weight / .bias
- encoder.layer.{i}.attention.self.query.weight / .bias
- encoder.layer.{i}.attention.self.key.weight / .bias
- encoder.layer.{i}.attention.self.value.weight / .bias
- encoder.layer.{i}.attention.output.dense.weight / .bias
- encoder.layer.{i}.attention.output.LayerNorm.weight / .bias
- encoder.layer.{i}.intermediate.dense.weight / .bias
- encoder.layer.{i}.output.dense.weight / .bias
- encoder.layer.{i}.output.LayerNorm.weight / .bias
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_compression.logger import logger
from llm_compression.inference.decoder_layers import DecoderLayer, RMSNorm, precompute_freqs_cis, KVCache


class InferenceCore(nn.Module):
    """
    Core inference engine for embedding generation.
    
    Implements a full BERT-like forward pass with:
    - Word, position, and token type embeddings
    - N transformer layers (multi-head self-attention + FFN + LayerNorm)
    - Mean pooling with attention mask
    
    Performance Target:
    - Inference latency: < 5ms (single sequence)
    - Throughput: > 2000 sequences/s (batch_size=32)
    - Memory usage: < 200MB
    
    Example:
        >>> config = {'hidden_size': 384, 'num_layers': 6, 'num_attention_heads': 6}
        >>> core = InferenceCore(weights, config, device='cpu')
        >>> embeddings = core(input_ids, attention_mask)
        >>> print(embeddings.shape)
        torch.Size([batch_size, 384])
    """
    
    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        config: Dict[str, any],
        device: str = "cpu",
    ):
        """
        Initialize InferenceCore.
        
        Args:
            weights: Model weights dictionary (from WeightLoader).
                Keys should follow HuggingFace BERT naming:
                  embeddings.word_embeddings.weight
                  encoder.layer.{i}.attention.self.query.weight
                  etc.
            config: Model configuration dict with:
                - hidden_size: Hidden dimension (default: 384)
                - num_layers: Number of transformer layers (default: 6)
                - num_attention_heads: Number of attention heads (default: 6)
                - intermediate_size: FFN intermediate size (default: 4*hidden_size)
                - max_position_embeddings: Maximum sequence length (default: 512)
                - vocab_size: Vocabulary size (default: 30522)
                - layer_norm_eps: LayerNorm epsilon (default: 1e-12)
                - hidden_dropout_prob: Dropout probability (default: 0.0 for inference)
            device: Device for inference ("cpu", "cuda", "mps")
        """
        super().__init__()
        
        self.config = config
        self.device = device
        
        # Extract config values with sensible defaults
        self.hidden_size = config.get('hidden_size', 384)
        self.num_layers = config.get('num_layers', 6)
        self.intermediate_size = config.get('intermediate_size', self.hidden_size * 4)
        self.max_position_embeddings = config.get('max_position_embeddings', 512)
        self.vocab_size = config.get('vocab_size', 30522)
        self.layer_norm_eps = config.get('layer_norm_eps', 1e-12)
        
        # num_attention_heads: use config if provided, otherwise derive from hidden_size
        # Standard BERT uses head_size=64, so num_heads = hidden_size / 64
        if 'num_attention_heads' in config:
            self.num_attention_heads = config['num_attention_heads']
        else:
            self.num_attention_heads = max(1, self.hidden_size // 64)
        
        # Ensure heads divides hidden_size evenly
        while self.hidden_size % self.num_attention_heads != 0 and self.num_attention_heads > 1:
            self.num_attention_heads -= 1
        
        # Derived config
        self.head_size = self.hidden_size // self.num_attention_heads
        
        # Check if architecture is Decoder-only (causal LLM)
        self.is_decoder = (
            config.get('architecture', '').endswith('CausalLM') or 
            'rope_theta' in config or 
            'num_key_value_heads' in config
        )
        
        # Auto-detect num_layers from weights if not explicitly provided or if
        # the provided value seems wrong (e.g., total tensor count instead of layers)
        detected_layers = self._detect_num_layers(weights)
        if detected_layers > 0 and detected_layers != self.num_layers:
            logger.info(
                f"Auto-detected {detected_layers} layer blocks from weights "
                f"(config specified {self.num_layers}), using detected value"
            )
            self.num_layers = detected_layers
        
        # Auto-detect intermediate_size from weights
        detected_intermediate = self._detect_intermediate_size(weights)
        if detected_intermediate > 0:
            self.intermediate_size = detected_intermediate
        
        # Build the model architecture and load weights
        self._build_and_load(weights)
        
        self.to(device)
        self.eval()
        
        # LoRA Support
        try:
            from .lora_manager import LoRAManager
            self.lora_manager = LoRAManager(self)
        except ImportError:
            self.lora_manager = None

        logger.info(
            f"Initialized InferenceCore on {device}: "
            f"hidden={self.hidden_size}, layers={self.num_layers}, "
            f"heads={self.num_attention_heads}, intermediate={self.intermediate_size}"
        )

    def load_lora(self, path: str):
        """Load and apply a LoRA adapter."""
        if self.lora_manager:
            card = self.lora_manager.load_card(path)
            self.lora_manager.apply_card(card)
        else:
            logger.warning("LoRA feature disabled.")

    def remove_lora(self, name: str):
        """Unload a LoRA adapter."""
        if self.lora_manager:
            self.lora_manager.remove_card(name)
    
    def _detect_num_layers(self, weights: Dict[str, torch.Tensor]) -> int:
        """Detect the number of transformer layers from weight key names."""
        layer_indices = set()
        for key in weights:
            if key.startswith("encoder.layer."):
                parts = key.split(".")
                if len(parts) > 2:
                    try:
                        layer_indices.add(int(parts[2]))
                    except ValueError:
                        pass
            elif key.startswith("model.layers."):
                parts = key.split(".")
                if len(parts) > 2:
                    try:
                        layer_indices.add(int(parts[2]))
                    except ValueError:
                        pass
        return len(layer_indices) if layer_indices else 0
    
    def _detect_intermediate_size(self, weights: Dict[str, torch.Tensor]) -> int:
        """Detect intermediate FFN size from weights."""
        key = "encoder.layer.0.intermediate.dense.weight"
        if key in weights:
            return weights[key].shape[0]
        key_decoder = "model.layers.0.mlp.gate_proj.weight"
        if key_decoder in weights:
            return weights[key_decoder].shape[0]
        return 0
    
    def _build_and_load(self, weights: Dict[str, torch.Tensor]):
        """
        Build model architecture and load weights.
        
        Instead of register_buffer with flattened names, we build proper
        nn.Module sub-components and load weights directly into them.
        """
        if self.is_decoder:
            self._build_decoder_and_load(weights)
            return

        # === 0. Auto-detect dimensions from weights before building layers ===
        if 'embeddings.word_embeddings.weight' in weights:
            self.vocab_size = weights['embeddings.word_embeddings.weight'].shape[0]
        if 'embeddings.position_embeddings.weight' in weights:
            self.max_position_embeddings = weights['embeddings.position_embeddings.weight'].shape[0]
        
        # === 1. Embedding Layer ===
        self.word_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)  # BERT uses 2 token types
        self.embedding_layernorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        
        # Load embedding weights
        self._load_param(self.word_embeddings, 'weight',
                         weights.get('embeddings.word_embeddings.weight'))
        self._load_param(self.position_embeddings, 'weight',
                         weights.get('embeddings.position_embeddings.weight'))
        self._load_param(self.token_type_embeddings, 'weight',
                         weights.get('embeddings.token_type_embeddings.weight'))
        self._load_param(self.embedding_layernorm, 'weight',
                         weights.get('embeddings.LayerNorm.weight'))
        self._load_param(self.embedding_layernorm, 'bias',
                         weights.get('embeddings.LayerNorm.bias'))
        
        # === 2. Transformer Encoder Layers ===
        self.encoder_layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = TransformerLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                layer_norm_eps=self.layer_norm_eps,
            )
            
            prefix = f"encoder.layer.{i}"
            
            # Attention weights: Q, K, V
            self._load_param(layer.attention.query, 'weight',
                             weights.get(f"{prefix}.attention.self.query.weight"))
            self._load_param(layer.attention.query, 'bias',
                             weights.get(f"{prefix}.attention.self.query.bias"))
            self._load_param(layer.attention.key, 'weight',
                             weights.get(f"{prefix}.attention.self.key.weight"))
            self._load_param(layer.attention.key, 'bias',
                             weights.get(f"{prefix}.attention.self.key.bias"))
            self._load_param(layer.attention.value, 'weight',
                             weights.get(f"{prefix}.attention.self.value.weight"))
            self._load_param(layer.attention.value, 'bias',
                             weights.get(f"{prefix}.attention.self.value.bias"))
            
            # Attention output dense + LayerNorm
            self._load_param(layer.attention_output, 'weight',
                             weights.get(f"{prefix}.attention.output.dense.weight"))
            self._load_param(layer.attention_output, 'bias',
                             weights.get(f"{prefix}.attention.output.dense.bias"))
            self._load_param(layer.attention_layernorm, 'weight',
                             weights.get(f"{prefix}.attention.output.LayerNorm.weight"))
            self._load_param(layer.attention_layernorm, 'bias',
                             weights.get(f"{prefix}.attention.output.LayerNorm.bias"))
            
            # FFN: intermediate dense
            self._load_param(layer.intermediate, 'weight',
                             weights.get(f"{prefix}.intermediate.dense.weight"))
            self._load_param(layer.intermediate, 'bias',
                             weights.get(f"{prefix}.intermediate.dense.bias"))
            
            # FFN: output dense + LayerNorm
            self._load_param(layer.output_dense, 'weight',
                             weights.get(f"{prefix}.output.dense.weight"))
            self._load_param(layer.output_dense, 'bias',
                             weights.get(f"{prefix}.output.dense.bias"))
            self._load_param(layer.output_layernorm, 'weight',
                             weights.get(f"{prefix}.output.LayerNorm.weight"))
            self._load_param(layer.output_layernorm, 'bias',
                             weights.get(f"{prefix}.output.LayerNorm.bias"))
            
            self.encoder_layers.append(layer)
        
        weight_count = sum(1 for v in weights.values() if v is not None)
        logger.debug(f"Loaded weights from {weight_count} tensors into model")

    def _build_decoder_and_load(self, weights: Dict[str, torch.Tensor]):
        """Build causal LLM architecture and load weights."""
        if 'model.embed_tokens.weight' in weights:
            self.vocab_size = weights['model.embed_tokens.weight'].shape[0]
            
        self.num_key_value_heads = self.config.get('num_key_value_heads', self.num_attention_heads)
        self.rms_norm_eps = self.config.get('rms_norm_eps', 1e-6)
        self.rope_theta = self.config.get('rope_theta', 10000.0)

        # 1. Embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self._load_param(self.embed_tokens, 'weight', weights.get('model.embed_tokens.weight'))
        
        # 2. Decoder Layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = DecoderLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                intermediate_size=self.intermediate_size,
                rms_norm_eps=self.rms_norm_eps
            )
            prefix = f"model.layers.{i}"
            
            # Attention
            self._load_param(layer.self_attn.q_proj, 'weight', weights.get(f"{prefix}.self_attn.q_proj.weight"))
            self._load_param(layer.self_attn.k_proj, 'weight', weights.get(f"{prefix}.self_attn.k_proj.weight"))
            self._load_param(layer.self_attn.v_proj, 'weight', weights.get(f"{prefix}.self_attn.v_proj.weight"))
            self._load_param(layer.self_attn.o_proj, 'weight', weights.get(f"{prefix}.self_attn.o_proj.weight"))
            
            # Bias (for models like Qwen)
            self._load_param(layer.self_attn.q_proj, 'bias', weights.get(f"{prefix}.self_attn.q_proj.bias"))
            self._load_param(layer.self_attn.k_proj, 'bias', weights.get(f"{prefix}.self_attn.k_proj.bias"))
            self._load_param(layer.self_attn.v_proj, 'bias', weights.get(f"{prefix}.self_attn.v_proj.bias"))
            self._load_param(layer.self_attn.o_proj, 'bias', weights.get(f"{prefix}.self_attn.o_proj.bias"))

            # MLP
            self._load_param(layer.mlp.gate_proj, 'weight', weights.get(f"{prefix}.mlp.gate_proj.weight"))
            self._load_param(layer.mlp.up_proj, 'weight', weights.get(f"{prefix}.mlp.up_proj.weight"))
            self._load_param(layer.mlp.down_proj, 'weight', weights.get(f"{prefix}.mlp.down_proj.weight"))
            
            # LayerNorms
            self._load_param(layer.input_layernorm, 'weight', weights.get(f"{prefix}.input_layernorm.weight"))
            self._load_param(layer.post_attention_layernorm, 'weight', weights.get(f"{prefix}.post_attention_layernorm.weight"))
            
            self.layers.append(layer)
            
        # 3. Output Norm
        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self._load_param(self.norm, 'weight', weights.get('model.norm.weight'))
        
        # 4. LM Head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self._load_param(self.lm_head, 'weight', weights.get('lm_head.weight'))
        
        # Initialize RoPE
        self.freqs_cis = precompute_freqs_cis(
            self.head_size, 
            self.max_position_embeddings * 2, 
            theta=self.rope_theta
        ).to(self.device)
        
        logger.debug(f"Loaded weights for Causal LM decoder into model")
    
    @staticmethod
    def _load_param(module: nn.Module, param_name: str, tensor: Optional[torch.Tensor]):
        """Load a weight tensor into a module parameter, handling dtype conversion."""
        if tensor is None:
            return
        
        param = getattr(module, param_name, None)
        if param is None:
            return
        
        # Convert float16 weights to float32 for computation
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        
        with torch.no_grad():
            param.copy_(tensor)
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        kv_cache: Optional[List[KVCache]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """
        Forward pass to generate sentence embeddings (BERT models).
        
        If model is a Decoder, this will route to forward_decoder.
        """
        if self.is_decoder:
            return self.forward_decoder(input_ids, kv_cache=kv_cache)
            
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # 1. Embedding layer
        hidden_states = self._compute_embeddings(input_ids)
        
        # 2. Extended attention mask: (batch, 1, 1, seq_len) for broadcasting
        #    in multi-head attention. Convert 0/1 mask to large negative values.
        extended_mask = self._get_extended_attention_mask(attention_mask)
        
        # 3. Transformer encoder layers
        all_attentions = () if output_attentions else None
        
        for layer in self.encoder_layers:
            hidden_states, attention_probs = layer(
                hidden_states, 
                extended_mask,
                output_attentions=output_attentions
            )
            if output_attentions:
                all_attentions = all_attentions + (attention_probs,)
        
        # 4. Mean pooling
        pooled_output = self.mean_pooling(hidden_states, attention_mask)
        
        if output_attentions:
            return pooled_output, all_attentions
        return pooled_output

    @torch.no_grad()
    def forward_decoder(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[List[KVCache]] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """Forward pass for decoder generation."""
        input_ids = input_ids.to(self.device)
        bsz, seqlen = input_ids.shape
        
        h = self.embed_tokens(input_ids)
        
        start_pos = 0
        if kv_cache is not None:
            start_pos = kv_cache[0].seq_len
            
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=self.device)
            mask = torch.triu(mask, diagonal=1).float()
            # If KV cache is populated, mask needs to accommodate past tokens
            if start_pos > 0:
                past_mask = torch.zeros((1, 1, seqlen, start_pos), device=self.device)
                mask = torch.cat([past_mask, mask], dim=-1)
            
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            h = layer(h, freqs_cis, layer_cache, mask)
            
        h = self.norm(h)
        # We generally only care about the last token's logits for generation
        logits = self.lm_head(h[:, -1, :]) 
        return logits

    def _compute_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute combined embeddings (word + position + token_type) + LayerNorm.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            
        Returns:
            Embeddings, shape (batch_size, seq_len, hidden_size)
        """
        seq_len = input_ids.shape[1]
        
        # Position IDs: 0, 1, 2, ..., seq_len-1
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Token type IDs: all zeros (single sentence)
        token_type_ids = torch.zeros_like(input_ids)
        
        # Combine embeddings
        word_emb = self.word_embeddings(input_ids)
        position_emb = self.position_embeddings(position_ids)
        token_type_emb = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_emb + position_emb + token_type_emb
        embeddings = self.embedding_layernorm(embeddings)
        
        return embeddings
    
    def _get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert attention mask (0/1) to extended mask with large negative values
        for masked positions, suitable for addition to attention scores.
        
        Args:
            attention_mask: (batch_size, seq_len), 1 for real tokens, 0 for padding
            
        Returns:
            Extended mask: (batch_size, 1, 1, seq_len), 0.0 for real tokens,
            -10000.0 for padding
        """
        # Shape: (batch_size, 1, 1, seq_len) for broadcasting with attention scores
        extended_mask = attention_mask[:, None, None, :].float()
        # Convert from 1/0 to 0/-10000 (additive mask)
        extended_mask = (1.0 - extended_mask) * -10000.0
        return extended_mask
    
    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean pooling operation with attention mask.
        
        Computes the mean of token embeddings, weighted by attention mask,
        to produce a single sentence embedding.
        
        Args:
            token_embeddings: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            Sentence embeddings, shape (batch_size, hidden_size)
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def normalize_embeddings(
        self,
        embeddings: torch.Tensor,
        p: float = 2.0,
    ) -> torch.Tensor:
        """
        L2-normalize embeddings.
        
        Args:
            embeddings: Embeddings to normalize, shape (batch_size, hidden_size)
            p: Norm order (default: 2 for L2 norm)
            
        Returns:
            Normalized embeddings, shape (batch_size, hidden_size)
        """
        return F.normalize(embeddings, p=p, dim=1)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.hidden_size
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"InferenceCore("
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, "
            f"num_heads={self.num_attention_heads}, "
            f"intermediate={self.intermediate_size}, "
            f"device={self.device})"
        )


class TransformerLayer(nn.Module):
    """
    Single BERT Transformer layer.
    
    Architecture:
        Input → MultiHeadAttention → Add & LayerNorm → FFN → Add & LayerNorm → Output
    
    Where:
        MultiHeadAttention = Q/K/V projection → scaled dot-product attention → output projection
        FFN = Linear(hidden→intermediate) → GELU → Linear(intermediate→hidden)
    """
    
    def __init__(
        self,
        hidden_size: int = 384,
        num_attention_heads: int = 6,
        intermediate_size: int = 1536,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        
        # Multi-head self-attention components
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Feed-forward network (FFN)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer layer.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, 1, 1, seq_len) extended mask
            output_attentions: Whether to return attention probabilities
            
        Returns:
            Tuple of (output_hidden_states, attention_probs)
        """
        # === Multi-Head Self-Attention ===
        attention_output, attention_probs = self.attention(
            hidden_states, 
            attention_mask, 
            output_attentions=output_attentions
        )
        attention_output = self.attention_output(attention_output)
        # Residual connection + LayerNorm
        hidden_states = self.attention_layernorm(attention_output + hidden_states)
        
        # === Feed-Forward Network ===
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = F.gelu(intermediate_output)
        layer_output = self.output_dense(intermediate_output)
        # Residual connection + LayerNorm
        hidden_states = self.output_layernorm(layer_output + hidden_states)
        
        return hidden_states, attention_probs


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Implements scaled dot-product attention with multiple heads:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
    
    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, seq, hidden) → (batch, heads, seq, head_size)."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, 1, 1, seq_len) extended attention mask
            output_attentions: Whether to return attention probabilities
            
        Returns:
            Tuple of (context_output, attention_probs)
            attention_probs is None if output_attentions=False
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self._reshape_for_heads(self.query(hidden_states))  # (B, H, S, D)
        k = self._reshape_for_heads(self.key(hidden_states))    # (B, H, S, D)
        v = self._reshape_for_heads(self.value(hidden_states))  # (B, H, S, D)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_size)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / scale  # (B, H, S, S)
        
        # Apply attention mask (additive: 0 for attend, -10000 for masked)
        attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Weighted sum of values
        context = torch.matmul(attention_probs, v)  # (B, H, S, D)
        
        # Reshape back: (B, H, S, D) → (B, S, H*D)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.all_head_size)
        
        outputs = (context, attention_probs) if output_attentions else (context, None)
        return outputs

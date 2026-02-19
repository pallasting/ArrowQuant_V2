"""
Audio Core - Native Whisper Encoder Implementation.

This module implements a lightweight Whisper encoder architecture
optimized for ArrowEngine, reusing the highly optimized TransformerLayer
from inference_core.py.

It serves as the native engine for audio embeddings.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_compression.logger import logger


class WhisperEncoderLayer(nn.Module):
    """
    Whisper Encoder Layer with Pre-LayerNorm architecture.
    
    Key differences from BERT:
    - Pre-LN: LayerNorm before attention and FFN (not after)
    - K projection has no bias
    - Uses GELU activation
    """
    
    def __init__(
        self,
        hidden_size: int = 384,
        num_attention_heads: int = 6,
        intermediate_size: int = 1536,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        
        # Pre-attention LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Multi-head self-attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # No bias!
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Pre-FFN LayerNorm
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Feed-forward network
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Pre-LN architecture.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            
        Returns:
            output_hidden_states: (batch_size, seq_len, hidden_size)
        """
        residual = hidden_states
        
        # === Pre-LN + Multi-Head Self-Attention ===
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Q, K, V projections
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = self.head_size ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection
        hidden_states = residual + attn_output
        
        # === Pre-LN + Feed-Forward Network ===
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        return hidden_states


class AudioInferenceCore(nn.Module):
    """
    Core inference engine for audio embedding generation (Whisper encoder).
    
    Structure:
    - Conv1d layers for audio embedding
    - Position Embeddings
    - Transformer Encoder (reused from InferenceCore)
    - LayerNorm
    - Mean pooling over time dimension
    """
    
    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        config: Dict[str, any],
        device: str = "cpu"
    ):
        super().__init__()
        self.config = config
        self.device = device
        
        # Config defaults (based on Whisper base)
        self.n_mels = config.get("n_mels", 80)
        self.hidden_size = config.get("hidden_size", 512)
        self.num_layers = config.get("num_layers", 6)
        self.num_heads = config.get("num_attention_heads", 8)
        self.intermediate_size = config.get("intermediate_size", 2048)
        self.layer_norm_eps = config.get("layer_norm_eps", 1e-5)
        self.max_positions = config.get("max_positions", 1500)
        
        # 1. Audio Embedding (Conv1d layers)
        # First conv: (n_mels, T) -> (hidden_size, T)
        self.conv1 = nn.Conv1d(
            in_channels=self.n_mels,
            out_channels=self.hidden_size,
            kernel_size=3,
            padding=1
        )
        
        # Second conv: (hidden_size, T) -> (hidden_size, T/2)
        self.conv2 = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # 2. Position Embeddings
        self.position_embedding = nn.Embedding(self.max_positions, self.hidden_size)
        
        # 3. Transformer Encoder
        self.encoder_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.encoder_layers.append(WhisperEncoderLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                layer_norm_eps=self.layer_norm_eps
            ))
        
        # 4. Layer Normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        
        # Load weights
        self._load_weights(weights)
        
        self.to(device)
        self.eval()
    
    def _load_weights(self, weights: Dict[str, torch.Tensor]):
        """Load weights from dictionary mapping."""
        # Conv layers
        self._load_param(self.conv1, 'weight', weights.get('encoder.conv1.weight'))
        self._load_param(self.conv1, 'bias', weights.get('encoder.conv1.bias'))
        self._load_param(self.conv2, 'weight', weights.get('encoder.conv2.weight'))
        self._load_param(self.conv2, 'bias', weights.get('encoder.conv2.bias'))
        
        # Position embeddings
        self._load_param(self.position_embedding, 'weight', weights.get('encoder.position_embedding'))
        
        # Transformer layers
        for i in range(self.num_layers):
            layer: WhisperEncoderLayer = self.encoder_layers[i]
            prefix = f"encoder.layers.{i}"
            
            # LayerNorms
            self._load_param(layer.self_attn_layer_norm, 'weight', weights.get(f"{prefix}.self_attn_layer_norm.weight"))
            self._load_param(layer.self_attn_layer_norm, 'bias', weights.get(f"{prefix}.self_attn_layer_norm.bias"))
            self._load_param(layer.final_layer_norm, 'weight', weights.get(f"{prefix}.final_layer_norm.weight"))
            self._load_param(layer.final_layer_norm, 'bias', weights.get(f"{prefix}.final_layer_norm.bias"))
            
            # Attention projections
            self._load_param(layer.q_proj, 'weight', weights.get(f"{prefix}.self_attn.q_proj.weight"))
            self._load_param(layer.q_proj, 'bias', weights.get(f"{prefix}.self_attn.q_proj.bias"))
            self._load_param(layer.k_proj, 'weight', weights.get(f"{prefix}.self_attn.k_proj.weight"))
            # k_proj has no bias
            self._load_param(layer.v_proj, 'weight', weights.get(f"{prefix}.self_attn.v_proj.weight"))
            self._load_param(layer.v_proj, 'bias', weights.get(f"{prefix}.self_attn.v_proj.bias"))
            self._load_param(layer.out_proj, 'weight', weights.get(f"{prefix}.self_attn.out_proj.weight"))
            self._load_param(layer.out_proj, 'bias', weights.get(f"{prefix}.self_attn.out_proj.bias"))
            
            # FFN
            self._load_param(layer.fc1, 'weight', weights.get(f"{prefix}.fc1.weight"))
            self._load_param(layer.fc1, 'bias', weights.get(f"{prefix}.fc1.bias"))
            self._load_param(layer.fc2, 'weight', weights.get(f"{prefix}.fc2.weight"))
            self._load_param(layer.fc2, 'bias', weights.get(f"{prefix}.fc2.bias"))
        
        # Layer norm
        self._load_param(self.layer_norm, 'weight', weights.get('encoder.layer_norm.weight'))
        self._load_param(self.layer_norm, 'bias', weights.get('encoder.layer_norm.bias'))
    
    def _load_param(self, module, param_name, tensor):
        """Load parameter from tensor."""
        if tensor is None:
            return
        
        if isinstance(module, nn.Parameter):
            with torch.no_grad():
                module.copy_(tensor)
        else:
            param = getattr(module, param_name, None)
            if param is not None:
                with torch.no_grad():
                    param.copy_(tensor)
    
    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: (Batch, n_mels, n_frames) mel-spectrogram tensor
            
        Returns:
            (Batch, hidden_size) audio embedding
        """
        B = mel_spectrogram.shape[0]
        
        # 1. Conv layers with GELU activation
        # (B, n_mels, T) -> (B, hidden_size, T)
        x = F.gelu(self.conv1(mel_spectrogram))
        # (B, hidden_size, T) -> (B, hidden_size, T/2)
        x = F.gelu(self.conv2(x))
        
        # 2. Transpose to (B, T/2, hidden_size)
        x = x.transpose(1, 2)
        
        # 3. Add position embeddings
        seq_len = x.shape[1]
        
        # Clip sequence length to max_positions to avoid index out of range
        if seq_len > self.max_positions:
            logger.warning(
                f"Sequence length {seq_len} exceeds max_positions {self.max_positions}. "
                f"Truncating to {self.max_positions}."
            )
            x = x[:, :self.max_positions, :]
            seq_len = self.max_positions
        
        positions = torch.arange(seq_len, device=self.device)
        x = x + self.position_embedding(positions)
        
        # 4. Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x, attention_mask=None)
        
        # 5. Layer normalization
        x = self.layer_norm(x)
        
        # 6. Mean pooling over time dimension
        # (B, T/2, hidden_size) -> (B, hidden_size)
        pooled_output = x.mean(dim=1)
        
        return pooled_output

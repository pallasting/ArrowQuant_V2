
"""
Vision Core - Native Vision Transformer Implementation.

This module implements a lightweight Vision Transformer (ViT) architecture
optimized for ArrowEngine, reusing the highly optimized TransformerLayer
from inference_core.py.

It serves as the native engine for vision embeddings, replacing heavy
external dependencies like 'transformers'.
"""

import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_compression.logger import logger
from llm_compression.inference.inference_core import TransformerLayer

class VisionInferenceCore(nn.Module):
    """
    Core inference engine for vision embedding generation (ViT).
    
    Structure:
    - PatchEmbeddings (Conv2d projection)
    - Class Token + Position Embeddings
    - Transformer Encoder (reused from InferenceCore)
    - LayerNorm + Pooler
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
        
        # Config defaults (based on CLIP-ViT-B/32)
        self.image_size = config.get("image_size", 224)
        self.patch_size = config.get("patch_size", 32)
        self.hidden_size = config.get("hidden_size", 768)
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_attention_heads", 12)
        self.intermediate_size = config.get("intermediate_size", 3072)
        self.layer_norm_eps = config.get("layer_norm_eps", 1e-5)
        self.projection_dim = config.get("projection_dim", 512)
        
        # 1. Patch Embeddings
        # In ViT, patch embedding is implemented as a Conv2d layer
        # Input: (B, 3, H, W) -> Output: (B, Hidden, GridH, GridW)
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False
        )
        
        # 2. Class Token & Position Embeddings
        num_patches = (self.image_size // self.patch_size) ** 2
        self.class_embedding = nn.Parameter(torch.randn(self.hidden_size))
        self.position_embedding = nn.Embedding(num_patches + 1, self.hidden_size)
        self.pre_layernorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        
        # 3. Transformer Encoder
        self.encoder_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.encoder_layers.append(TransformerLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                layer_norm_eps=self.layer_norm_eps
            ))
            
        # 4. Post-processing
        self.post_layernorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        if self.projection_dim != self.hidden_size:
            self.visual_projection = nn.Linear(self.hidden_size, self.projection_dim, bias=False)
        else:
            self.visual_projection = None
            
        # Load weights
        self._load_weights(weights)
        
        self.to(device)
        self.eval()
        
    def _load_weights(self, weights: Dict[str, torch.Tensor]):
        """Load weights from dictionary mapping."""
        # Note: Key mapping logic will need to be robust to handle different source formats 
        # (e.g., CLIP vs standard ViT). Here we assume a mapped format.
        
        # Patch Embeddings
        self._load_param(self.patch_embedding, 'weight', weights.get('vision_model.embeddings.patch_embedding.weight'))
        
        # Embeddings
        self._load_param(self.class_embedding, 'data', weights.get('vision_model.embeddings.class_embedding'))
        self._load_param(self.position_embedding, 'weight', weights.get('vision_model.embeddings.position_embedding.weight'))
        self._load_param(self.pre_layernorm, 'weight', weights.get('vision_model.pre_layrnorm.weight'))
        self._load_param(self.pre_layernorm, 'bias', weights.get('vision_model.pre_layrnorm.bias'))
        
        # Layers
        for i in range(self.num_layers):
            layer: TransformerLayer = self.encoder_layers[i]
            prefix = f"vision_model.encoder.layers.{i}"
            
            # Attn
            self._load_param(layer.attention.query, 'weight', weights.get(f"{prefix}.self_attn.q_proj.weight"))
            self._load_param(layer.attention.query, 'bias', weights.get(f"{prefix}.self_attn.q_proj.bias"))
            self._load_param(layer.attention.key, 'weight', weights.get(f"{prefix}.self_attn.k_proj.weight"))
            self._load_param(layer.attention.key, 'bias', weights.get(f"{prefix}.self_attn.k_proj.bias"))
            self._load_param(layer.attention.value, 'weight', weights.get(f"{prefix}.self_attn.v_proj.weight"))
            self._load_param(layer.attention.value, 'bias', weights.get(f"{prefix}.self_attn.v_proj.bias"))
            self._load_param(layer.attention_output, 'weight', weights.get(f"{prefix}.self_attn.out_proj.weight"))
            self._load_param(layer.attention_output, 'bias', weights.get(f"{prefix}.self_attn.out_proj.bias"))
            
            self._load_param(layer.attention_layernorm, 'weight', weights.get(f"{prefix}.layer_norm1.weight"))
            self._load_param(layer.attention_layernorm, 'bias', weights.get(f"{prefix}.layer_norm1.bias"))
            
            # MLP
            self._load_param(layer.intermediate, 'weight', weights.get(f"{prefix}.mlp.fc1.weight"))
            self._load_param(layer.intermediate, 'bias', weights.get(f"{prefix}.mlp.fc1.bias"))
            self._load_param(layer.output_dense, 'weight', weights.get(f"{prefix}.mlp.fc2.weight"))
            self._load_param(layer.output_dense, 'bias', weights.get(f"{prefix}.mlp.fc2.bias"))
            
            self._load_param(layer.output_layernorm, 'weight', weights.get(f"{prefix}.layer_norm2.weight"))
            self._load_param(layer.output_layernorm, 'bias', weights.get(f"{prefix}.layer_norm2.bias"))
            
        # Post Processing
        self._load_param(self.post_layernorm, 'weight', weights.get('vision_model.post_layernorm.weight'))
        self._load_param(self.post_layernorm, 'bias', weights.get('vision_model.post_layernorm.bias'))
        
        if self.visual_projection is not None:
             self._load_param(self.visual_projection, 'weight', weights.get('visual_projection.weight'))

    def _load_param(self, module, param_name, tensor):
        if tensor is None: return
        # Handle nn.Parameter vs simple attribute
        if isinstance(module, nn.Parameter):
            with torch.no_grad():
                module.copy_(tensor)
        else:
            param = getattr(module, param_name, None)
            if param is not None:
                with torch.no_grad():
                    param.copy_(tensor)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (Batch, 3, H, W) normalized image tensor.
        Returns:
            (Batch, projection_dim) image embedding.
        """
        B = pixel_values.shape[0]
        
        # 1. Patch Embedding
        # (B, 3, H, W) -> (B, Hidden, GridH, GridW) -> (B, Hidden, Seq) -> (B, Seq, Hidden)
        embeddings = self.patch_embedding(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        
        # 2. Add Class Token
        # class_embedding: (Hidden) -> (B, 1, Hidden)
        class_tokens = self.class_embedding.expand(B, 1, -1)
        embeddings = torch.cat((class_tokens, embeddings), dim=1)
        
        # 3. Add Position Embeddings
        # (Seq + 1, Hidden)
        embeddings = embeddings + self.position_embedding.weight[:embeddings.size(1)]
        embeddings = self.pre_layernorm(embeddings)
        
        # 4. Transformer Encoder
        # Extended mask is used in BERT but usually not in ViT unless masking patches
        # Here we assume full attention
        extended_mask = torch.zeros(B, 1, 1, embeddings.size(1), device=self.device)
        
        for layer in self.encoder_layers:
            embeddings, _ = layer(embeddings, extended_mask)
            
        # 5. Pooling & Projection
        # Take class token (index 0)
        pooled_output = embeddings[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        
        if self.visual_projection is not None:
            pooled_output = self.visual_projection(pooled_output)
            
        return pooled_output

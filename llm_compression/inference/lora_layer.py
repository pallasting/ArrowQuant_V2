
"""
AI-OS Arrow-Native LoRA Injection Layer.

Implements a transparent LoRA wrapper for nn.Linear layers.
Supports hot-swapping and dynamic enable/disable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer with LoRA adapters.
    The original weights are frozen and kept in place.
    The LoRA path is computed in parallel and added to the output.
    """
    
    def __init__(
        self, 
        original_layer: nn.Linear, 
        rank: int = 8, 
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA weights: B @ A
        # A: (in_features, rank) -> initialized Kaiming Uniform / Normal
        # B: (rank, out_features) -> initialized Zero
        
        in_dim = original_layer.in_features
        out_dim = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        # Status
        self.enabled = True
        
        # Initialize
        self.reset_parameters()
        
        # Freeze original
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
            
    def reset_parameters(self):
        """Initialize A with Gaussian, B with Zeros."""
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def load_weights(self, weight_A: np.ndarray, weight_B: np.ndarray):
        """Load weights from numpy arrays (e.g. from Arrow)."""
        # Ensure shape match
        # NumPy shape is usually [out, in] for linear weights in PyTorch convention?
        # Standard LoRA usually stores A as [rank, in_dim] and B as [out_dim, rank]
        
        with torch.no_grad():
            self.lora_A.copy_(torch.from_numpy(weight_A).float())
            self.lora_B.copy_(torch.from_numpy(weight_B).float())
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        previous_dtype = x.dtype
        result = self.original_layer(x)
        
        if not self.enabled:
            return result
            
        # LoRA forward: x @ A.T @ B.T * scaling
        # x shape: (Batch, Seq, In)
        # A shape: (Rank, In)
        # B shape: (Out, Rank)
        
        # Dropout
        x_d = self.dropout(x)
        
        # Compute Low Rank Path
        # (B, S, I) @ (I, R) -> (B, S, R)
        lora_out = F.linear(x_d, self.lora_A)
        # (B, S, R) @ (R, O) -> (B, S, O)
        lora_out = F.linear(lora_out, self.lora_B)
        
        # Add scaled result
        result = result + lora_out * self.scaling
        
        return result.to(previous_dtype)

    def merge(self):
        """Permanently merge LoRA weights into original (for export/efficiency)."""
        if self.original_layer.weight.data.shape == (self.lora_B @ self.lora_A).shape:
             # Wait, usually weight is [Out, In]. 
             # lora_B is [Out, Rank], lora_A is [Rank, In].
             # Matmul B @ A -> [Out, In]. Correct.
             delta = (self.lora_B @ self.lora_A) * self.scaling
             self.original_layer.weight.data += delta
             self.enabled = False 
             
    def unmerge(self):
        """Subtract LoRA weights if merged."""
        if not self.enabled: # Assuming merged means disabled
             delta = (self.lora_B @ self.lora_A) * self.scaling
             self.original_layer.weight.data -= delta
             self.enabled = True

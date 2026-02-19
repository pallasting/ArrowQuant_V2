
"""
AI-OS Arrow-Native LoRA Manager.

Manages the lifecycle of LoRA adapters:
- Loading from Arrow files
- Dynamic injection into InferenceCore
- Hot-swapping and Merging logic.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from pathlib import Path
import logging

from .inference_core import InferenceCore
from .lora_layer import LoRALinear
from .lora_format import LoRACard, LoRAFormat

logger = logging.getLogger(__name__)

class LoRAManager:
    def __init__(self, model: InferenceCore):
        self.model = model
        self.active_cards: Dict[str, LoRACard] = {}
        self.injected_layers: Dict[str, LoRALinear] = {} # Map layer name to LoRA module
        
    def load_card(self, path: str) -> LoRACard:
        """Load LoRA card from disk."""
        try:
            card = LoRAFormat.load(path)
            logger.info(f"Loaded LoRA Card: {card.name} (Rank={card.rank})")
            return card
        except Exception as e:
            logger.error(f"Failed to load LoRA card from {path}: {e}")
            raise

    def apply_card(self, card: LoRACard):
        """Inject LoRA card into the model."""
        if card.name in self.active_cards:
            logger.warning(f"LoRA {card.name} already applied.")
            return

        # 1. Iterate through targets
        # Targets are usually regex or module suffixes like "attention.q_proj"
        # We need to find the corresponding modules in InferenceCore
        
        injected_count = 0
        
        for name, module in self.model.named_modules():
            # Check if this module is a target
            # e.g. "transformer.layers.0.attention.q_proj" matches suffix "q_proj"
            
            matched = False
            for target in card.target_modules:
                if name.endswith(target):
                    matched = True
                    break
            
            if matched and isinstance(module, nn.Linear):
                # Valid injection point
                
                # Check if already wrapped
                if isinstance(module, LoRALinear):
                    # Already has a LoRA? Support multi-LoRA later.
                    # For now, replace weights or error?
                    logger.warning(f"Layer {name} already has LoRA. Overwriting not supported yet.")
                    continue
                    
                # Create wrapper
                # Find parent module to replace child
                if '.' in name:
                    parent_name, child_name = name.rsplit('.', 1)
                    parent = self.model.get_submodule(parent_name)
                else:
                    # Top-level module
                    parent = self.model
                    child_name = name
                
                # Verify we have weights for this specific layer instance
                # The LoRA Card weights map keys to arrays.
                # Key format in card usually matches full module name.
                # If card weights use "base_model.model.layers.0..." we need to align.
                
                # Let's assume Card keys are relative to model root or match exact names
                # Try exact match first
                
                weight_key = name
                if weight_key not in card.weights_A:
                    # Try fuzzy matching or strip prefix
                    # e.g. "layers.0.attention.q_proj" vs "transformer.layers.0..."
                    # Check suffix match in card keys
                    found_key = None
                    for k in card.weights_A.keys():
                        if k.endswith(name) or name.endswith(k):
                            found_key = k
                            break
                    weight_key = found_key
                
                if not weight_key:
                    # No weights for this layer, skip
                    continue
                
                # Inject
                lora_layer = LoRALinear(
                    module, 
                    rank=card.rank, 
                    alpha=card.alpha
                )
                
                # Load weights
                lora_layer.load_weights(
                    card.weights_A[weight_key],
                    card.weights_B[weight_key]
                )
                
                # Replace in parent
                setattr(parent, child_name, lora_layer)
                
                # Track
                self.injected_layers[name] = lora_layer
                injected_count += 1
                
        self.active_cards[card.name] = card
        logger.info(f"Applied LoRA {card.name}: Injected into {injected_count} layers.")
        
    def remove_card(self, card_name: str):
        """Remove LoRA and restore original layers."""
        if card_name not in self.active_cards:
            return
            
        # Iterate tracked layers
        # This implementation assumes 1 LoRA at a time for simplicity in Phase 1
        # For multi-LoRA, we'd need a LoRA-Stack layer.
        
        # Restore original modules
        for name, lora_layer in list(self.injected_layers.items()):
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = self.model.get_submodule(parent_name)
            else:
                parent = self.model
                child_name = name
            
            # Unwrap
            original = lora_layer.original_layer
            setattr(parent, child_name, original)
            
            del self.injected_layers[name]
            
        del self.active_cards[card_name]
        logger.info(f"Removed LoRA {card_name}.")

    def list_cards(self) -> List[str]:
        return list(self.active_cards.keys())


"""
AI-OS LoRA Extractor.

Extracts LoRA adapters from model weights using SVD decomposition.
Given a weight matrix W (or a delta ΔW between fine-tuned and base),
decomposes it into low-rank matrices A and B such that:

    W ≈ A @ B  (rank-r approximation)

This is the mathematical inverse of LoRA training:
- LoRA trains A and B to approximate a desired weight change.
- LoRAExtractor decomposes an existing weight into A and B.

Core insight from the user:
"通过需求问题探测权重激活区域，直接加载相关技能权重，验证有效之后直接提取作为LoRA卡片"
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

from llm_compression.inference.lora_format import LoRACard, LoRAFormat
from llm_compression.evolution.weight_probe import ActivationHeatMap

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of extracting a LoRA from a weight matrix."""
    layer_name: str
    rank: int
    weights_A: np.ndarray  # (rank, d_in)
    weights_B: np.ndarray  # (d_out, rank)
    explained_variance: float  # How much of the original signal is retained
    original_norm: float  # Frobenius norm of original matrix


class LoRAExtractor:
    """
    Extracts LoRA adapters from model weights via SVD decomposition.
    
    Two modes of operation:
    
    1. **Direct Extraction**: Decompose a weight matrix W directly.
       Useful for extracting skills from a pre-trained model.
       
    2. **Delta Extraction**: Given base weights W_base and fine-tuned W_ft,
       compute ΔW = W_ft - W_base, then decompose ΔW.
       This captures exactly what the fine-tuning learned.
    
    Usage:
        extractor = LoRAExtractor(rank=8)
        
        # From heat map hot zones
        card = extractor.extract_from_heat_map(
            weights=model_weights,
            heat_map=probe_result,
            name="math_expert_v1"
        )
        
        # Save as .lora.arrow
        LoRAFormat.save(card, "math_expert_v1.lora.arrow")
    """
    
    def __init__(
        self, 
        rank: int = 8, 
        alpha: float = 16.0,
        min_explained_variance: float = 0.5
    ):
        """
        Args:
            rank: LoRA rank (r). Higher = more capacity, larger file.
            alpha: LoRA scaling factor.
            min_explained_variance: Minimum variance ratio to include a layer.
        """
        self.rank = rank
        self.alpha = alpha
        self.min_explained_variance = min_explained_variance
    
    def extract_single(
        self, 
        weight: torch.Tensor,
        layer_name: str
    ) -> ExtractionResult:
        """
        Extract LoRA from a single weight matrix using SVD.
        
        For matrix W of shape (d_out, d_in):
            W = U @ diag(S) @ Vt
            
        LoRA approximation with rank r:
            A = Vt[:r, :]           # (r, d_in) - "down projection"
            B = U[:, :r] * S[:r]    # (d_out, r) - "up projection"
            W ≈ B @ A
            
        This matches LoRA convention where:
            h = W @ x + (B @ A) @ x * (alpha / rank)
        
        Args:
            weight: Weight tensor of shape (d_out, d_in).
            layer_name: Name of the layer.
            
        Returns:
            ExtractionResult with A, B matrices and quality metrics.
        """
        if not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight)
        
        weight = weight.float()
        original_norm = torch.norm(weight).item()
        
        # SVD decomposition
        # W = U @ diag(S) @ Vt
        try:
            U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
        except Exception as e:
            logger.warning(f"SVD failed for {layer_name}: {e}")
            raise
        
        r = min(self.rank, len(S))
        
        # Explained variance ratio
        total_variance = (S ** 2).sum().item()
        retained_variance = (S[:r] ** 2).sum().item()
        explained = retained_variance / total_variance if total_variance > 0 else 0.0
        
        # Extract low-rank components
        # A: (r, d_in) - captures input-side patterns
        A = Vt[:r, :].cpu().numpy()
        
        # B: (d_out, r) - captures output-side patterns, scaled by singular values
        B = (U[:, :r] * S[:r].unsqueeze(0)).cpu().numpy()
        
        return ExtractionResult(
            layer_name=layer_name,
            rank=r,
            weights_A=A,
            weights_B=B,
            explained_variance=explained,
            original_norm=original_norm,
        )
    
    def extract_from_heat_map(
        self,
        weights: Dict[str, torch.Tensor],
        heat_map: ActivationHeatMap,
        name: str,
        target_modules: Optional[List[str]] = None,
        description: str = ""
    ) -> LoRACard:
        """
        Extract a LoRA card from the hot zones identified by WeightMapProbe.
        
        Only extracts from layers that:
        1. Are in the heat map's hot zones.
        2. Have weight matrices (2D tensors).
        3. Meet the minimum explained variance threshold.
        
        Args:
            weights: Full model weights dict.
            heat_map: Result from WeightMapProbe.analyze().
            name: Name for the resulting LoRA card.
            target_modules: Optional filter for module types.
            description: Human-readable description.
            
        Returns:
            LoRACard ready to save as .lora.arrow.
        """
        hot_layer_names = set(layer_name for layer_name, _ in heat_map.hot_zones)
        
        extracted_A = {}
        extracted_B = {}
        extraction_stats = []
        
        for layer_name in hot_layer_names:
            # Find corresponding weight key(s).
            # Hot zone names come from named_modules() (e.g. "encoder_layers.2.intermediate")
            # Weight keys come from state_dict() (e.g. "encoder_layers.2.intermediate.weight")
            # We try multiple matching strategies:
            matched_keys = self._find_weight_keys(layer_name, weights)
            
            if not matched_keys:
                logger.debug(f"Weight not found for hot zone: {layer_name}")
                continue
            
            for weight_key in matched_keys:
                tensor = weights[weight_key]
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.as_tensor(tensor)
                
                # Only process 2D weight matrices    
                if tensor.ndim != 2:
                    continue
                    
                # Optional: filter by target module type
                if target_modules:
                    if not any(tm in weight_key for tm in target_modules):
                        continue
                
                # Extract
                try:
                    result = self.extract_single(tensor, weight_key)
                except Exception:
                    continue
                    
                # Quality gate
                if result.explained_variance < self.min_explained_variance:
                    logger.debug(
                        f"Skipping {weight_key}: explained_variance={result.explained_variance:.3f} "
                        f"< threshold={self.min_explained_variance}"
                    )
                    continue
                
                extracted_A[weight_key] = result.weights_A
                extracted_B[weight_key] = result.weights_B
                extraction_stats.append(result)
        
        if not extracted_A:
            logger.warning(f"No layers met extraction criteria for '{name}'.")
            # Return minimal card
            return LoRACard(
                name=name,
                rank=self.rank,
                alpha=self.alpha,
                target_modules=list(target_modules or []),
                weights_A={},
                weights_B={},
                metadata={"description": description, "status": "empty"}
            )
        
        # Compute quality summary
        avg_variance = np.mean([r.explained_variance for r in extraction_stats])
        
        # Infer target modules from extracted layers
        inferred_targets = self._infer_target_modules(list(extracted_A.keys()))
        
        logger.info(
            f"Extracted LoRA '{name}': {len(extracted_A)} layers, "
            f"rank={self.rank}, avg_variance={avg_variance:.3f}"
        )
        
        return LoRACard(
            name=name,
            rank=self.rank,
            alpha=self.alpha,
            target_modules=target_modules or inferred_targets,
            weights_A=extracted_A,
            weights_B=extracted_B,
            metadata={
                "description": description,
                "extraction_method": "svd_heat_map",
                "num_layers": len(extracted_A),
                "avg_explained_variance": float(avg_variance),
                "hot_zone_concentration": float(heat_map.concentration_ratio),
            }
        )

    def extract_delta(
        self,
        base_weights: Dict[str, torch.Tensor],
        finetuned_weights: Dict[str, torch.Tensor],
        name: str,
        target_modules: Optional[List[str]] = None,
        description: str = ""
    ) -> LoRACard:
        """
        Extract LoRA by computing delta between base and fine-tuned weights.
        
        ΔW = W_finetuned - W_base
        Then decompose ΔW via SVD.
        
        This captures EXACTLY what fine-tuning learned.
        
        Args:
            base_weights: Original model weights.
            finetuned_weights: Fine-tuned model weights.
            name: Name for the LoRA card.
            target_modules: Optional module filter.
            description: Human-readable description.
            
        Returns:
            LoRACard representing the fine-tuning delta.
        """
        extracted_A = {}
        extracted_B = {}
        stats = []
        
        # Find common 2D weight keys
        common_keys = set(base_weights.keys()) & set(finetuned_weights.keys())
        
        for key in common_keys:
            base_t = base_weights[key]
            ft_t = finetuned_weights[key]
            
            if not isinstance(base_t, torch.Tensor):
                base_t = torch.as_tensor(base_t)
            if not isinstance(ft_t, torch.Tensor):
                ft_t = torch.as_tensor(ft_t)
            
            if base_t.ndim != 2 or ft_t.ndim != 2:
                continue
                
            if base_t.shape != ft_t.shape:
                continue
                
            if target_modules:
                if not any(tm in key for tm in target_modules):
                    continue
            
            # Compute delta
            delta = ft_t.float() - base_t.float()
            delta_norm = torch.norm(delta).item()
            
            # Skip if delta is negligible
            if delta_norm < 1e-6:
                continue
            
            # SVD on delta
            try:
                result = self.extract_single(delta, key)
            except Exception:
                continue
                
            if result.explained_variance < self.min_explained_variance:
                continue
                
            extracted_A[key] = result.weights_A
            extracted_B[key] = result.weights_B
            stats.append(result)
        
        avg_variance = np.mean([r.explained_variance for r in stats]) if stats else 0.0
        inferred_targets = self._infer_target_modules(list(extracted_A.keys()))
        
        logger.info(
            f"Extracted delta LoRA '{name}': {len(extracted_A)} layers, "
            f"rank={self.rank}, avg_variance={avg_variance:.3f}"
        )
        
        return LoRACard(
            name=name,
            rank=self.rank,
            alpha=self.alpha,
            target_modules=target_modules or inferred_targets,
            weights_A=extracted_A,
            weights_B=extracted_B,
            metadata={
                "description": description,
                "extraction_method": "svd_delta",
                "num_layers": len(extracted_A),
                "avg_explained_variance": float(avg_variance),
            }
        )
    
    @staticmethod
    def _find_weight_keys(
        module_name: str, 
        weights: Dict[str, torch.Tensor]
    ) -> List[str]:
        """
        Find weight keys matching a module name.
        
        Tries multiple strategies:
        1. Exact match (module_name in weights)
        2. With .weight suffix
        3. Substring match (module_name is prefix of weight key)
        """
        matched = []
        
        # Strategy 1: Exact match
        if module_name in weights:
            matched.append(module_name)
            return matched
            
        # Strategy 2: With .weight suffix
        weight_key = f"{module_name}.weight"
        if weight_key in weights:
            matched.append(weight_key)
            return matched
        
        # Strategy 3: All weight keys that start with this module name
        prefix = module_name + "."
        for key in weights:
            if key.startswith(prefix) and key.endswith(".weight"):
                matched.append(key)
        
        if matched:
            return matched
            
        # Strategy 4: Substring containment (last resort)
        # e.g., hot zone "encoder_layers.2.intermediate" should match
        # "encoder_layers.2.intermediate.weight" in weights
        for key in weights:
            if module_name in key:
                matched.append(key)
        
        return matched
    
    @staticmethod
    def _infer_target_modules(layer_names: List[str]) -> List[str]:
        """Infer target module types from layer names."""
        targets = set()
        for name in layer_names:
            parts = name.split(".")
            # Look for common module type names
            for part in parts:
                if part in ("query", "key", "value", "dense", "intermediate", "output"):
                    targets.add(part)
        return list(targets) if targets else ["linear"]


"""
AI-OS Weight Map Probe.

Analyzes weight activation patterns during inference to identify
which layers and attention heads are most relevant to a given task.

This enables "Surgical LoRA Extraction" - extracting only the
relevant portions of a large model as compact LoRA adapters.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActivationHeatMap:
    """Records activation intensity per layer."""
    layer_activations: Dict[str, float] = field(default_factory=dict)
    # Which layers had the highest activation magnitude
    hot_zones: List[Tuple[str, float]] = field(default_factory=list)
    # Total forward pass energy
    total_energy: float = 0.0
    # How much of total energy is in the hot zones
    concentration_ratio: float = 0.0


class WeightMapProbe:
    """
    Probes a model's weight activations to identify task-relevant regions.
    
    When a user query activates certain layers more than others,
    those layers contain the "knowledge" most relevant to that task.
    By identifying these "hot zones", we can surgically extract 
    LoRA adapters from just those regions.
    
    This is the core of the user's insight:
    "通过需求问题探测权重激活区域，直接加载相关技能权重"
    
    Usage:
        probe = WeightMapProbe()
        heat_map = probe.analyze(inference_core, test_inputs)
        hot_zones = heat_map.hot_zones  # [(layer_name, activation_strength), ...]
    """
    
    def __init__(self, top_k: int = 10, threshold_percentile: float = 80.0):
        """
        Args:
            top_k: Number of top activated layers to return.
            threshold_percentile: Percentile above which a layer is "hot".
        """
        self.top_k = top_k
        self.threshold_percentile = threshold_percentile
        self._hooks = []
        self._activations: Dict[str, torch.Tensor] = {}
    
    def analyze(
        self, 
        inference_core, 
        test_inputs: List[str],
        tokenizer=None
    ) -> ActivationHeatMap:
        """
        Run forward pass with hooks to capture layer activations.
        
        Args:
            inference_core: The InferenceCore (or nn.Module) to probe.
            test_inputs: List of representative input strings for the task.
            tokenizer: FastTokenizer instance for encoding inputs.
            
        Returns:
            ActivationHeatMap with layer-level activation analysis.
        """
        self._activations.clear()
        self._hooks.clear()
        
        # 1. Register forward hooks on all linear layers
        self._register_hooks(inference_core)
        
        # 2. Run forward pass
        try:
            with torch.no_grad():
                if tokenizer and test_inputs:
                    for text in test_inputs:
                        encoded = tokenizer.encode([text])
                        input_ids = torch.as_tensor(encoded['input_ids']).long()
                        attention_mask = torch.as_tensor(encoded['attention_mask']).long()
                        
                        device = next(inference_core.parameters()).device
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        
                        inference_core(input_ids, attention_mask)
        finally:
            # 3. Remove hooks
            self._remove_hooks()
        
        # 4. Analyze activations
        heat_map = self._compute_heat_map()
        
        logger.info(
            f"Probed {len(self._activations)} layers. "
            f"Top zone: {heat_map.hot_zones[0][0] if heat_map.hot_zones else 'N/A'} "
            f"(concentration: {heat_map.concentration_ratio:.1%})"
        )
        
        return heat_map
    
    def analyze_from_weights(
        self,
        weights: Dict[str, torch.Tensor],
        reference_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> ActivationHeatMap:
        """
        Analyze weights directly without running inference.
        
        Computes weight magnitude (Frobenius norm) per layer.
        If reference_weights are provided, computes the DELTA
        (useful for comparing fine-tuned vs base model).
        
        This is useful when we have Arrow/Parquet weights but
        no inference capability for the full model.
        
        Args:
            weights: Current model weights dict.
            reference_weights: Optional base model weights for delta analysis.
            
        Returns:
            ActivationHeatMap based on weight magnitude analysis.
        """
        layer_norms = {}
        
        for name, tensor in weights.items():
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
                
            if reference_weights and name in reference_weights:
                ref = reference_weights[name]
                if not isinstance(ref, torch.Tensor):
                    ref = torch.as_tensor(ref)
                # Delta analysis: where did the weights change most?
                delta = tensor - ref
                norm = torch.norm(delta.float()).item()
            else:
                norm = torch.norm(tensor.float()).item()
                
            layer_norms[name] = norm
        
        return self._build_heat_map_from_norms(layer_norms)
    
    def _register_hooks(self, module):
        """Register forward hooks on all named modules."""
        for name, child in module.named_modules():
            if isinstance(child, torch.nn.Linear):
                hook = child.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)
    
    def _make_hook(self, name):
        """Create a hook function that captures output activation."""
        def hook_fn(module, input, output):
            # Accumulate activation norms
            if name not in self._activations:
                self._activations[name] = 0.0
            # Use L2 norm of output as activation strength
            self._activations[name] += torch.norm(output.float()).item()
        return hook_fn
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def _compute_heat_map(self) -> ActivationHeatMap:
        """Compute heat map from captured activations."""
        return self._build_heat_map_from_norms(self._activations)
    
    def _build_heat_map_from_norms(self, norms: Dict[str, float]) -> ActivationHeatMap:
        """Build ActivationHeatMap from a dict of layer -> norm mappings."""
        if not norms:
            return ActivationHeatMap()
            
        # Sort by activation strength
        sorted_layers = sorted(norms.items(), key=lambda x: x[1], reverse=True)
        
        # Total energy
        total = sum(v for v in norms.values())
        
        # Top-K hot zones
        hot_zones = sorted_layers[:self.top_k]
        
        # Concentration: how much energy is in the hot zones
        hot_energy = sum(v for _, v in hot_zones)
        concentration = hot_energy / total if total > 0 else 0.0
        
        return ActivationHeatMap(
            layer_activations=dict(sorted_layers),
            hot_zones=hot_zones,
            total_energy=total,
            concentration_ratio=concentration,
        )

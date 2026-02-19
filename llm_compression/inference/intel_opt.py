
"""
Intel AI Acceleration optimization utilities.
"""

import torch
from .device_utils import is_intel_ipex_available

def optimize_for_intel(model, dtype=torch.float32):
    """
    Apply Intel IPEX optimizations to a model.
    """
    if not is_intel_ipex_available():
        return model
    
    try:
        import intel_extension_for_pytorch as ipex
        # For CPU, IPEX provides jit and layer optimizations
        if next(model.parameters()).device.type == "cpu":
            # ipex.optimize provides significant boost for BERT on Intel CPUs
            model = ipex.optimize(model, dtype=dtype)
            return model
        # For GPU (XPU)
        elif next(model.parameters()).device.type == "xpu":
            model = ipex.optimize(model, dtype=dtype)
            return model
    except Exception as e:
        from llm_compression.logger import logger
        logger.warning(f"Failed to apply Intel IPEX optimizations: {e}")
        
    return model

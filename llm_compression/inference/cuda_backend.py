
"""
CUDA/GPU Acceleration backend for AI-OS.
Provides utilities for GPU memory management and optimized inference.
"""

import torch
import logging

logger = logging.getLogger(__name__)

def optimize_for_cuda(model, use_amp=True):
    """
    Prepare model for CUDA execution.
    
    Args:
        model: PyTorch model
        use_amp: Enable Automatic Mixed Precision (FP16)
        
    Returns:
        Optimized model
    """
    if not torch.cuda.is_available():
        return model
    
    # Move to GPU
    model = model.to("cuda")
    
    # We don't wrap with AMP here, it's usually handled in the forward pass 
    # using torch.cuda.amp.autocast context manager.
    # However, we can ensure weights are in float16 if requested.
    if use_amp:
        model = model.half()
        
    return model

def get_vram_usage():
    """Get current VRAM usage in MB."""
    if not torch.cuda.is_available():
        return 0.0
    
    # Result in MB
    return torch.cuda.memory_allocated(0) / (1024 * 1024)

def check_vram_limit(threshold_mb=None):
    """
    Check if VRAM usage is within limits.
    
    Returns:
        True if within limits, False if exceeded.
    """
    if not torch.cuda.is_available():
        return True
        
    if threshold_mb is None:
        # Default to 90% of total memory
        total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        threshold_mb = total * 0.9
        
    current = get_vram_usage()
    if current > threshold_mb:
        logger.warning(f"VRAM usage high: {current:.2f}MB / {threshold_mb:.2f}MB")
        return False
        
    return True

def clear_cuda_cache():
    """Clear PyTorch CUDA cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared.")

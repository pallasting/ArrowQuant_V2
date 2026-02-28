"""
CUDA/GPU Acceleration backend for AI-OS Diffusion.
Provides utilities for GPU memory management and optimized inference.
"""

import torch

from ..utils.logger import logger


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
    
    # Convert to half precision if requested
    # Note: AMP is usually handled in forward pass with torch.cuda.amp.autocast
    if use_amp:
        model = model.half()
        
    return model


def get_vram_usage():
    """
    Get current VRAM usage in MB.
    
    Returns:
        float: VRAM usage in MB
    """
    if not torch.cuda.is_available():
        return 0.0
    
    return torch.cuda.memory_allocated(0) / (1024 * 1024)


def check_vram_limit(threshold_mb=None):
    """
    Check if VRAM usage is within limits.
    
    Args:
        threshold_mb: Threshold in MB (default: 90% of total memory)
        
    Returns:
        bool: True if within limits, False if exceeded
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
    """Clear PyTorch CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared.")

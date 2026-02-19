
"""
Device utilities for hardware-agnostic acceleration.

Supports:
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm/CUDA)
- Intel GPUs (XPU)
- Apple Silicon (MPS)
- CPU Fallback
"""

import torch
import logging

logger = logging.getLogger(__name__)

def get_best_device() -> str:
    """
    Detect the best available hardware accelerator.
    Returns: "cuda" (NVIDIA/ROCm), "xpu", "mps", "vulkan", or "cpu".
    """
    # 1. NVIDIA / AMD (NVIDIA CUDA or AMD ROCm via HIP)
    if torch.cuda.is_available():
        # Check if it's actually ROCm (AMD)
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
             logger.info("AMD GPU (ROCm/HIP) detected via CUDA interface.")
        return "cuda"
    
    # 2. Intel GPU (Intel Extension for PyTorch / XPU)
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            return "xpu"
    except (ImportError, AttributeError):
        pass

    # 3. Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    
    # 4. Vulkan (Cross-platform GPU)
    try:
        if hasattr(torch, "vulkan") and torch.vulkan.is_available():
            return "vulkan"
    except:
        pass
    
    # 5. CPU (Check for Intel AMX/AVX-512)
    return "cpu"

def is_intel_ipex_available() -> bool:
    """Check if Intel Extension for PyTorch is installed."""
    try:
        import intel_extension_for_pytorch
        return True
    except ImportError:
        return False

def move_to_device(obj, device: str):
    """
    Recursively move tensors in a nested structure to a device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj

def get_device_info(device: str) -> dict:
    """
    Get human-readable info about the device.
    """
    info = {"device_type": device}
    info["ipex_available"] = is_intel_ipex_available()
    
    try:
        if device == "cuda":
            info["name"] = torch.cuda.get_device_name(0)
            info["capability"] = torch.cuda.get_device_capability(0)
            info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
            # Identify if it's AMD ROCm
            info["is_amd"] = hasattr(torch.version, "hip") and torch.version.hip is not None
        elif device == "xpu":
            # Intel GPU info
            info["name"] = torch.xpu.get_device_name(0)
            props = torch.xpu.get_device_properties(0)
            info["memory_total"] = props.total_memory
        elif device == "mps":
            info["name"] = "Apple Silicon (MPS)"
        elif device == "vulkan":
            info["name"] = "Vulkan Optimizer"
        else:
            info["name"] = "CPU"
            # Check for Intel AMX (Advanced Matrix Extensions)
            # This requires PyTorch 2.0+ and Intel CPU
            try:
                import torch.cpu
                info["has_amx"] = torch.cpu.is_amx_available() if hasattr(torch.cpu, "is_amx_available") else False
            except:
                info["has_amx"] = "Unknown"
    except Exception as e:
        info["error"] = str(e)
        if "name" not in info: info["name"] = "Unknown"
        
    return info

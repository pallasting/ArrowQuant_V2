"""
Device utilities for hardware-agnostic acceleration.

Supports:
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm via HIP)
- Intel GPUs (XPU via IPEX)
- Apple Silicon (MPS)
- Vulkan (Cross-platform)
- CPU Fallback (AVX-512/AMX)

Architecture: 🦴 Rust Skeleton (future) + 🧠 Python Brain (current)
"""

import torch
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


def get_best_device() -> str:
    """
    Detect the best available hardware accelerator.
    
    Priority order:
    1. CUDA (NVIDIA) / ROCm (AMD)
    2. XPU (Intel)
    3. MPS (Apple Silicon)
    4. Vulkan (Cross-platform)
    5. CPU (with AVX-512/AMX if available)
    
    Returns:
        str: "cuda", "xpu", "mps", "vulkan", or "cpu"
    """
    # 1. NVIDIA CUDA or AMD ROCm (via HIP compatibility layer)
    if torch.cuda.is_available():
        # Detect if it's AMD ROCm
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            logger.info(f"AMD GPU detected (ROCm {torch.version.hip} via HIP)")
        else:
            logger.info("NVIDIA GPU detected (CUDA)")
        return "cuda"
    
    # 2. Intel GPU (Intel Extension for PyTorch / XPU)
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            logger.info("Intel GPU detected (XPU)")
            return "xpu"
    except (ImportError, AttributeError):
        pass

    # 3. Apple Silicon (Metal Performance Shaders)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple Silicon detected (MPS)")
        return "mps"
    
    # 4. Vulkan (Cross-platform GPU compute)
    if is_vulkan_available():
        logger.info("Vulkan backend detected")
        return "vulkan"
    
    # 5. CPU fallback (check for advanced features)
    cpu_features = get_cpu_features()
    logger.info(f"Using CPU with features: {cpu_features}")
    return "cpu"


def is_vulkan_available() -> bool:
    """
    Check if Vulkan backend is available.
    
    Returns:
        bool: True if Vulkan is available
    """
    try:
        # PyTorch Vulkan backend (experimental)
        if hasattr(torch, "vulkan") and torch.vulkan.is_available():
            return True
    except Exception as e:
        logger.debug(f"Vulkan check failed: {e}")
    
    return False


def is_rocm_platform() -> bool:
    """
    Check if running on AMD ROCm platform.
    
    Returns:
        bool: True if ROCm is detected
    """
    if not torch.cuda.is_available():
        return False
    
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def get_rocm_version() -> Optional[str]:
    """
    Get ROCm version if available.
    
    Returns:
        str or None: ROCm version string
    """
    if is_rocm_platform():
        return torch.version.hip
    return None


def get_cpu_features() -> List[str]:
    """
    Detect CPU advanced features.
    
    Returns:
        List[str]: List of available CPU features
    """
    features = []
    
    try:
        # Check for Intel AMX (Advanced Matrix Extensions)
        if hasattr(torch.cpu, "is_amx_available") and torch.cpu.is_amx_available():
            features.append("AMX")
    except:
        pass
    
    try:
        # Check for AVX-512
        import platform
        if "avx512" in platform.processor().lower():
            features.append("AVX-512")
    except:
        pass
    
    if not features:
        features.append("baseline")
    
    return features


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
    
    Args:
        obj: Object to move (tensor, dict, list, tuple)
        device: Target device
        
    Returns:
        Object with all tensors moved to device
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


def get_device_info(device: str) -> Dict:
    """
    Get detailed information about the device.
    
    Args:
        device: Device string ("cuda", "xpu", "mps", "vulkan", "cpu")
        
    Returns:
        dict: Device information including name, memory, capabilities
    """
    info = {
        "device_type": device,
        "ipex_available": is_intel_ipex_available(),
        "rocm_available": is_rocm_platform(),
    }
    
    try:
        if device == "cuda":
            info["name"] = torch.cuda.get_device_name(0)
            info["capability"] = torch.cuda.get_device_capability(0)
            info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
            info["memory_total_gb"] = info["memory_total"] / (1024**3)
            
            # Identify backend (NVIDIA CUDA vs AMD ROCm)
            if is_rocm_platform():
                info["backend"] = "ROCm (AMD)"
                info["rocm_version"] = get_rocm_version()
                info["hip_version"] = torch.version.hip
            else:
                info["backend"] = "CUDA (NVIDIA)"
                info["cuda_version"] = torch.version.cuda
            
            # Multi-GPU info
            info["device_count"] = torch.cuda.device_count()
            
        elif device == "xpu":
            # Intel GPU info
            info["backend"] = "XPU (Intel)"
            info["name"] = torch.xpu.get_device_name(0)
            props = torch.xpu.get_device_properties(0)
            info["memory_total"] = props.total_memory
            info["memory_total_gb"] = info["memory_total"] / (1024**3)
            info["device_count"] = torch.xpu.device_count()
            
        elif device == "mps":
            info["backend"] = "MPS (Apple)"
            info["name"] = "Apple Silicon (Metal Performance Shaders)"
            # MPS doesn't expose memory info directly
            info["unified_memory"] = True
            
        elif device == "vulkan":
            info["backend"] = "Vulkan (Cross-platform)"
            info["name"] = "Vulkan Compute Backend"
            info["experimental"] = True
            
        else:  # CPU
            info["backend"] = "CPU"
            info["name"] = "CPU"
            info["features"] = get_cpu_features()
            
            # Check for Intel AMX
            try:
                if hasattr(torch.cpu, "is_amx_available"):
                    info["has_amx"] = torch.cpu.is_amx_available()
            except:
                info["has_amx"] = False
                
    except Exception as e:
        info["error"] = str(e)
        if "name" not in info:
            info["name"] = "Unknown"
        
    return info


def print_device_info(device: Optional[str] = None) -> None:
    """
    Print detailed device information.
    
    Args:
        device: Device to query (None = auto-detect)
    """
    if device is None:
        device = get_best_device()
    
    info = get_device_info(device)
    
    print(f"\n{'='*60}")
    print(f"Device Information")
    print(f"{'='*60}")
    print(f"Device Type: {info['device_type']}")
    print(f"Backend: {info.get('backend', 'Unknown')}")
    print(f"Name: {info.get('name', 'Unknown')}")
    
    if "memory_total_gb" in info:
        print(f"Memory: {info['memory_total_gb']:.2f} GB")
    
    if "device_count" in info and info["device_count"] > 1:
        print(f"Device Count: {info['device_count']}")
    
    if "rocm_version" in info:
        print(f"ROCm Version: {info['rocm_version']}")
    
    if "cuda_version" in info:
        print(f"CUDA Version: {info['cuda_version']}")
    
    if "features" in info:
        print(f"CPU Features: {', '.join(info['features'])}")
    
    if info.get("experimental"):
        print("⚠️  Experimental backend - limited operator support")
    
    print(f"{'='*60}\n")

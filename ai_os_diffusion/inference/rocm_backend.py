"""
AMD ROCm Backend Optimizations

Provides ROCm-specific optimizations for AMD GPUs.
ROCm uses HIP (Heterogeneous-compute Interface for Portability) 
which provides CUDA compatibility layer.

Supported GPUs:
- AMD Radeon RX 6000/7000 series
- AMD Instinct MI100/MI200/MI300 series
- AMD Radeon Pro series

Architecture: 🦴 Rust Skeleton (future HIP kernels) + 🧠 Python Brain (current)
"""

import torch
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def is_rocm_available() -> bool:
    """
    Check if ROCm is available.
    
    Returns:
        bool: True if ROCm is detected
    """
    if not torch.cuda.is_available():
        return False
    
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def get_rocm_info() -> Dict[str, Any]:
    """
    Get ROCm platform information.
    
    Returns:
        dict: ROCm information including version, device, capabilities
    """
    if not is_rocm_available():
        return {"available": False}
    
    info = {
        "available": True,
        "hip_version": torch.version.hip,
        "device_count": torch.cuda.device_count(),
        "devices": []
    }
    
    for i in range(torch.cuda.device_count()):
        device_info = {
            "id": i,
            "name": torch.cuda.get_device_name(i),
            "capability": torch.cuda.get_device_capability(i),
            "total_memory": torch.cuda.get_device_properties(i).total_memory,
            "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
        }
        info["devices"].append(device_info)
    
    return info


def optimize_for_rocm(model: torch.nn.Module, device: str = "cuda") -> torch.nn.Module:
    """
    Apply ROCm-specific optimizations to model.
    
    Args:
        model: PyTorch model
        device: Target device (default: "cuda")
    
    Returns:
        torch.nn.Module: Optimized model
    """
    if not is_rocm_available():
        logger.warning("ROCm not available, skipping optimizations")
        return model
    
    logger.info("Applying ROCm optimizations...")
    
    # Move model to device
    model = model.to(device)
    
    # Enable TF32 for matrix multiplications (if supported)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 for ROCm")
    except:
        pass
    
    # Enable cuDNN benchmarking for optimal kernel selection
    torch.backends.cudnn.benchmark = True
    logger.info("Enabled cuDNN benchmarking")
    
    # Set memory allocator settings
    try:
        # Reduce memory fragmentation
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")
    except:
        pass
    
    return model


def enable_rocm_fusion() -> None:
    """
    Enable operator fusion for ROCm.
    
    Operator fusion can improve performance by reducing kernel launches.
    """
    if not is_rocm_available():
        return
    
    try:
        # Enable JIT fusion
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        logger.info("Enabled JIT fusion for ROCm")
    except Exception as e:
        logger.debug(f"Could not enable JIT fusion: {e}")


def get_optimal_batch_size(
    model_size_mb: float,
    sequence_length: int = 512,
    device_id: int = 0
) -> int:
    """
    Calculate optimal batch size for ROCm device.
    
    Args:
        model_size_mb: Model size in MB
        sequence_length: Input sequence length
        device_id: GPU device ID
    
    Returns:
        int: Recommended batch size
    """
    if not is_rocm_available():
        return 1
    
    # Get available memory
    props = torch.cuda.get_device_properties(device_id)
    total_memory_gb = props.total_memory / (1024**3)
    
    # Reserve 20% for overhead
    available_memory_gb = total_memory_gb * 0.8
    
    # Estimate memory per sample (rough approximation)
    # Model weights + activations + gradients
    memory_per_sample_mb = (model_size_mb * 3) + (sequence_length * 0.01)
    memory_per_sample_gb = memory_per_sample_mb / 1024
    
    # Calculate batch size
    batch_size = int(available_memory_gb / memory_per_sample_gb)
    
    # Clamp to reasonable range
    batch_size = max(1, min(batch_size, 128))
    
    logger.info(
        f"Recommended batch size for ROCm: {batch_size} "
        f"(Memory: {total_memory_gb:.1f}GB, Model: {model_size_mb:.1f}MB)"
    )
    
    return batch_size


def print_rocm_info() -> None:
    """Print detailed ROCm information."""
    info = get_rocm_info()
    
    if not info["available"]:
        print("ROCm is not available")
        return
    
    print(f"\n{'='*60}")
    print(f"AMD ROCm Information")
    print(f"{'='*60}")
    print(f"HIP Version: {info['hip_version']}")
    print(f"Device Count: {info['device_count']}")
    print()
    
    for device in info["devices"]:
        print(f"Device {device['id']}: {device['name']}")
        print(f"  Compute Capability: {device['capability']}")
        print(f"  Total Memory: {device['total_memory_gb']:.2f} GB")
        print()
    
    print(f"{'='*60}\n")


class ROCmOptimizer:
    """
    ROCm-specific optimizer for inference.
    
    Provides utilities for optimizing models on AMD GPUs.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize ROCm optimizer.
        
        Args:
            device_id: GPU device ID
        """
        self.device_id = device_id
        self.device = f"cuda:{device_id}"
        
        if not is_rocm_available():
            logger.warning("ROCm not available")
            return
        
        self.info = get_rocm_info()
        logger.info(f"Initialized ROCm optimizer for device {device_id}")
    
    def optimize_model(
        self,
        model: torch.nn.Module,
        enable_fusion: bool = True
    ) -> torch.nn.Module:
        """
        Optimize model for ROCm.
        
        Args:
            model: PyTorch model
            enable_fusion: Enable operator fusion
        
        Returns:
            torch.nn.Module: Optimized model
        """
        model = optimize_for_rocm(model, self.device)
        
        if enable_fusion:
            enable_rocm_fusion()
        
        return model
    
    def get_recommended_batch_size(
        self,
        model_size_mb: float,
        sequence_length: int = 512
    ) -> int:
        """
        Get recommended batch size.
        
        Args:
            model_size_mb: Model size in MB
            sequence_length: Sequence length
        
        Returns:
            int: Recommended batch size
        """
        return get_optimal_batch_size(
            model_size_mb,
            sequence_length,
            self.device_id
        )
    
    def warmup(self, model: torch.nn.Module, input_shape: tuple) -> None:
        """
        Warmup model with dummy input.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
        """
        logger.info("Warming up model on ROCm...")
        
        dummy_input = torch.randn(input_shape, device=self.device)
        
        # Run a few warmup iterations
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        
        torch.cuda.synchronize()
        logger.info("Warmup complete")

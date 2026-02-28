"""
Inference module for AI-OS Diffusion.

This module provides high-performance inference capabilities:
- ArrowEngine: Unified inference engine (AR + Diffusion modes)
- WeightLoader: Zero-copy weight loading from Parquet
- FastTokenizer: Rust-based high-performance tokenization
- InferenceCore: BERT/Decoder forward pass implementation
- Device utilities: Multi-backend GPU support (CUDA/ROCm/XPU/MPS/Vulkan)
- Backend optimizations: Intel IPEX, AMD ROCm, NVIDIA CUDA
"""

from .arrow_engine import ArrowEngine
from .weight_loader import WeightLoader, LazyWeightDict
from .fast_tokenizer import FastTokenizer
from .inference_core import InferenceCore
from .device_utils import (
    get_best_device,
    get_device_info,
    move_to_device,
    is_intel_ipex_available,
    is_rocm_platform,
    is_vulkan_available,
    print_device_info,
)

__all__ = [
    'ArrowEngine',
    'WeightLoader',
    'LazyWeightDict',
    'FastTokenizer',
    'InferenceCore',
    'get_best_device',
    'get_device_info',
    'move_to_device',
    'is_intel_ipex_available',
    'is_rocm_platform',
    'is_vulkan_available',
    'print_device_info',
]

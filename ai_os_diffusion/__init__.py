"""
AI-OS Diffusion: Unified Diffusion Architecture for Multimodal AI

A lightweight, evolution-friendly AI system built on the Rust Skeleton + Python Brain philosophy.

Architecture:
- 🦴 Rust Skeleton: High-performance infrastructure (ArrowStorage, ArrowQuant, VectorSearch, FastTokenizer)
- 🧠 Python Brain: Flexible learning layer (DiffusionCore, EvolutionRouter, training logic)

Example:
    >>> from ai_os_diffusion import ArrowEngine
    >>> 
    >>> engine = ArrowEngine("models/unified-diffusion")
    >>> result = engine.diffuse(
    ...     prompt="A beautiful sunset",
    ...     modality="text",
    ...     num_steps=4
    ... )
"""

__version__ = "0.1.0"
__author__ = "AI-OS Team"

# Core imports
from .inference import ArrowEngine
from .config import Config
from .utils.logger import logger

__all__ = [
    'ArrowEngine',
    'Config',
    'logger',
    '__version__',
]

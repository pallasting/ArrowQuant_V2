"""
Evolution module for AI-OS Diffusion.

Provides self-evolution capabilities through LoRA training and skill management.
"""

from ai_os_diffusion.evolution.lora_trainer import (
    LoRATrainer,
    LoRAConfig,
    LoRACard,
    LoRALinear,
)

__all__ = [
    "LoRATrainer",
    "LoRAConfig",
    "LoRACard",
    "LoRALinear",
]

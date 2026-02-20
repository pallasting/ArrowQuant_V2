"""
Emotion system for the Expression & Presentation Layer.

This module provides emotion mapping capabilities for both voice (TTS)
and text (NLG) generation, enabling emotionally-aware expression across
multiple modalities.
"""

from llm_compression.expression.emotion.text_style_mapper import (
    TextStyleMapper,
    TextStyleParameters
)

__all__ = [
    "TextStyleMapper",
    "TextStyleParameters",
]

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
from llm_compression.expression.emotion.consistency_validator import (
    EmotionConsistencyValidator,
    ConsistencyWarning
)

__all__ = [
    "TextStyleMapper",
    "TextStyleParameters",
    "EmotionConsistencyValidator",
    "ConsistencyWarning",
]

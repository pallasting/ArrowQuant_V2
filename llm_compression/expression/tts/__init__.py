"""
Text-to-Speech module for the Expression & Presentation Layer.

This module provides TTS capabilities with multiple backend support,
emotion control, caching, and streaming.
"""

from llm_compression.expression.tts.tts_engine import (
    TTSEngine,
    TTSCache,
    EmotionMapper,
    TTSError
)

__all__ = [
    "TTSEngine",
    "TTSCache",
    "EmotionMapper",
    "TTSError"
]

"""
Expression & Presentation Layer for AI-OS.

This module provides sophisticated output capabilities including:
- Text-to-Speech (TTS) synthesis
- Natural Language Generation (NLG)
- Emotional expression across modalities
- Multi-modal response coordination
- Visual content generation (optional)
- Multi-language support

Public API:
    - ResponsePlanner: Multi-modal response strategy planning
    - NLGEngine: Text generation with multiple backends
    - TTSEngine: Speech synthesis with multiple backends
    - EmotionMapper: Emotion-to-parameter mapping
    - Synchronizer: Multi-modal output coordination
    - LanguageDetector: Multi-language support
"""

from llm_compression.expression.expression_types import (
    OutputModality,
    ExpressionStyle,
    TTSBackend,
    NLGBackend,
    ExpressionContext,
    ResponsePlan,
    VoiceConfig,
    NLGConfig,
    TTSConfig,
)

from llm_compression.expression.tts import (
    TTSEngine,
    TTSCache,
    EmotionMapper,
)

from llm_compression.expression.synchronizer import (
    Synchronizer,
    SynchronizerConfig,
    StreamState,
    StreamChunk,
    OutputStream,
)

__all__ = [
    # Enums
    "OutputModality",
    "ExpressionStyle",
    "TTSBackend",
    "NLGBackend",
    "StreamState",
    # Data structures
    "ExpressionContext",
    "ResponsePlan",
    "VoiceConfig",
    "NLGConfig",
    "TTSConfig",
    "SynchronizerConfig",
    "StreamChunk",
    "OutputStream",
    # TTS Components
    "TTSEngine",
    "TTSCache",
    "EmotionMapper",
    # Synchronization
    "Synchronizer",
]

__version__ = "0.1.0"

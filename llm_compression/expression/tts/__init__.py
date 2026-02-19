"""
Text-to-Speech (TTS) module.

Provides speech synthesis capabilities with multiple backends:
- Piper: Fast local TTS
- Coqui TTS: High-quality local TTS
- Azure TTS: Cloud-based TTS with many voices
- OpenAI TTS: Cloud-based natural TTS

Public API:
    - TTSEngine: Main TTS engine with backend abstraction
    - TTSCache: Caching for generated speech
    - EmotionMapper: Maps emotions to voice parameters
"""

__all__ = []

__version__ = "0.1.0"

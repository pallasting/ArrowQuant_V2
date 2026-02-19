"""
Test fixtures for Expression Layer tests

Provides sample data and mock objects for testing expression components.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from llm_compression.expression.expression_types import (
    ExpressionContext,
    ResponsePlan,
    VoiceConfig,
    NLGConfig,
    TTSConfig,
    OutputModality,
    ExpressionStyle,
    TTSBackend,
    NLGBackend,
)


# Sample text fixtures
SAMPLE_TEXTS = {
    "short": "Hello, how are you?",
    "medium": "I'm here to help you with your questions. What would you like to know today?",
    "long": """
    The Expression Layer is a sophisticated system that enables natural communication
    through multiple modalities including text, speech, and visual outputs. It integrates
    with the existing ArrowEngine infrastructure and supports emotional expression,
    context-aware adaptation, and real-time streaming capabilities.
    """.strip(),
    "technical": "The system uses Arrow-native data structures for zero-copy operations.",
    "casual": "Hey! What's up? How can I help you today?",
    "formal": "Good afternoon. I am at your service. How may I assist you?",
    "empathetic": "I understand this must be difficult. I'm here to support you.",
    "error": "I apologize, but I encountered an error processing your request.",
}


# Sample conversation histories
SAMPLE_CONVERSATIONS = {
    "empty": [],
    "single_turn": [
        {"role": "user", "content": "Hello"},
    ],
    "multi_turn": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
        {"role": "user", "content": "What's the weather like?"},
    ],
    "technical": [
        {"role": "user", "content": "How does Arrow storage work?"},
        {"role": "assistant", "content": "Arrow provides zero-copy data structures..."},
    ],
}


# Sample expression contexts
def create_sample_context(
    user_id: str = "test_user",
    emotion: str = "neutral",
    formality: float = 0.5,
    language: str = "en",
    **kwargs
) -> ExpressionContext:
    """Create a sample expression context for testing."""
    return ExpressionContext(
        user_id=user_id,
        conversation_history=SAMPLE_CONVERSATIONS["empty"],
        current_emotion=emotion,
        formality_level=formality,
        language=language,
        **kwargs
    )


# Sample response plans
def create_sample_plan(
    modalities: List[OutputModality] = None,
    style: ExpressionStyle = ExpressionStyle.CASUAL,
    emotion: str = "neutral",
    **kwargs
) -> ResponsePlan:
    """Create a sample response plan for testing."""
    if modalities is None:
        modalities = [OutputModality.TEXT]
    
    return ResponsePlan(
        modalities=modalities,
        primary_modality=modalities[0],
        style=style,
        emotion=emotion,
        emotion_intensity=0.5,
        use_template=False,
        template_name=None,
        streaming=False,
        estimated_duration_ms=1000,
        **kwargs
    )


# Sample voice configurations
SAMPLE_VOICES = {
    "en_default": VoiceConfig(
        voice_id="en_US-lessac-medium",
        language="en",
        speed=1.0,
        pitch=1.0,
        volume=1.0,
    ),
    "en_fast": VoiceConfig(
        voice_id="en_US-lessac-medium",
        language="en",
        speed=1.3,
        pitch=1.0,
        volume=1.0,
    ),
    "zh_default": VoiceConfig(
        voice_id="zh-CN-XiaoxiaoNeural",
        language="zh",
        speed=1.0,
        pitch=1.0,
        volume=1.0,
    ),
    "emotional_joy": VoiceConfig(
        voice_id="en_US-lessac-medium",
        language="en",
        speed=1.1,
        pitch=1.1,
        volume=1.0,
        emotion="joy",
        emotion_intensity=0.7,
    ),
}


# Sample audio data
def generate_sample_audio(
    duration_seconds: float = 1.0,
    sample_rate: int = 22050
) -> np.ndarray:
    """Generate sample audio data for testing."""
    num_samples = int(duration_seconds * sample_rate)
    # Generate simple sine wave
    t = np.linspace(0, duration_seconds, num_samples)
    frequency = 440.0  # A4 note
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


# Sample NLG configurations
SAMPLE_NLG_CONFIGS = {
    "openai_default": NLGConfig(
        backend=NLGBackend.OPENAI,
        model="gpt-4",
        temperature=0.7,
        max_tokens=500,
        streaming=True,
    ),
    "template": NLGConfig(
        backend=NLGBackend.TEMPLATE,
        model="template",
        temperature=0.0,
        max_tokens=500,
        streaming=False,
    ),
}


# Sample TTS configurations
SAMPLE_TTS_CONFIGS = {
    "piper_default": TTSConfig(
        backend=TTSBackend.PIPER,
        voice=SAMPLE_VOICES["en_default"],
        sample_rate=22050,
        streaming=True,
        cache_enabled=True,
    ),
    "piper_no_cache": TTSConfig(
        backend=TTSBackend.PIPER,
        voice=SAMPLE_VOICES["en_default"],
        sample_rate=22050,
        streaming=False,
        cache_enabled=False,
    ),
}


# Emotion test cases
EMOTION_TEST_CASES = [
    {"emotion": "joy", "expected_speed": 1.1, "expected_pitch": 1.1},
    {"emotion": "sadness", "expected_speed": 0.9, "expected_pitch": 0.9},
    {"emotion": "anger", "expected_speed": 1.2, "expected_pitch": 1.15},
    {"emotion": "fear", "expected_speed": 1.15, "expected_pitch": 1.2},
    {"emotion": "neutral", "expected_speed": 1.0, "expected_pitch": 1.0},
    {"emotion": "empathetic", "expected_speed": 0.95, "expected_pitch": 0.98},
    {"emotion": "friendly", "expected_speed": 1.05, "expected_pitch": 1.05},
]


# Language test cases
LANGUAGE_TEST_CASES = [
    {
        "text": "Hello, how are you?",
        "expected_language": "en",
        "description": "English text"
    },
    {
        "text": "你好，你好吗？",
        "expected_language": "zh",
        "description": "Chinese text"
    },
    {
        "text": "こんにちは、元気ですか？",
        "expected_language": "ja",
        "description": "Japanese text"
    },
    {
        "text": "¿Hola, cómo estás?",
        "expected_language": "es",
        "description": "Spanish text"
    },
]


# Style test cases
STYLE_TEST_CASES = [
    {
        "intent": "greet",
        "formality": 0.2,
        "expected_style": ExpressionStyle.CASUAL,
    },
    {
        "intent": "formal_request",
        "formality": 0.8,
        "expected_style": ExpressionStyle.FORMAL,
    },
    {
        "intent": "technical_query",
        "formality": 0.5,
        "expected_style": ExpressionStyle.TECHNICAL,
    },
    {
        "intent": "error",
        "formality": 0.5,
        "expected_style": ExpressionStyle.EMPATHETIC,
    },
]

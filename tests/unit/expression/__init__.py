"""
Expression Layer test package

Provides test fixtures, mock backends, and utilities for testing the Expression Layer.
"""

from tests.unit.expression.fixtures import (
    SAMPLE_TEXTS,
    SAMPLE_CONVERSATIONS,
    SAMPLE_VOICES,
    SAMPLE_NLG_CONFIGS,
    SAMPLE_TTS_CONFIGS,
    EMOTION_TEST_CASES,
    LANGUAGE_TEST_CASES,
    STYLE_TEST_CASES,
    create_sample_context,
    create_sample_plan,
    generate_sample_audio,
)

from tests.unit.expression.mock_backends import (
    MockNLGBackend,
    MockTTSBackend,
    MockTemplateEngine,
    MockTTSCache,
    MockLanguageDetector,
    MockEmotionMapper,
    create_mock_nlg_backend,
    create_mock_tts_backend,
    create_mock_template_engine,
    create_mock_tts_cache,
    create_mock_language_detector,
    create_mock_emotion_mapper,
)

__all__ = [
    # Fixtures
    "SAMPLE_TEXTS",
    "SAMPLE_CONVERSATIONS",
    "SAMPLE_VOICES",
    "SAMPLE_NLG_CONFIGS",
    "SAMPLE_TTS_CONFIGS",
    "EMOTION_TEST_CASES",
    "LANGUAGE_TEST_CASES",
    "STYLE_TEST_CASES",
    "create_sample_context",
    "create_sample_plan",
    "generate_sample_audio",
    # Mock backends
    "MockNLGBackend",
    "MockTTSBackend",
    "MockTemplateEngine",
    "MockTTSCache",
    "MockLanguageDetector",
    "MockEmotionMapper",
    "create_mock_nlg_backend",
    "create_mock_tts_backend",
    "create_mock_template_engine",
    "create_mock_tts_cache",
    "create_mock_language_detector",
    "create_mock_emotion_mapper",
]

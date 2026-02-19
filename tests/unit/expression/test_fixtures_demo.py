"""
Demo tests showing how to use fixtures and mock backends

These tests demonstrate the test infrastructure and serve as examples
for writing new tests.
"""

import pytest
import numpy as np

from tests.unit.expression import (
    # Fixtures
    SAMPLE_TEXTS,
    SAMPLE_CONVERSATIONS,
    SAMPLE_VOICES,
    EMOTION_TEST_CASES,
    LANGUAGE_TEST_CASES,
    create_sample_context,
    create_sample_plan,
    generate_sample_audio,
    # Mock backends
    create_mock_nlg_backend,
    create_mock_tts_backend,
    create_mock_template_engine,
    create_mock_tts_cache,
    create_mock_language_detector,
    create_mock_emotion_mapper,
)

from llm_compression.expression.expression_types import (
    OutputModality,
    ExpressionStyle,
    NLGBackend,
    NLGConfig,
)


class TestFixturesDemo:
    """Demonstrate using test fixtures."""
    
    def test_sample_texts(self):
        """Test using sample text fixtures."""
        # Access predefined text samples
        assert len(SAMPLE_TEXTS["short"]) < len(SAMPLE_TEXTS["medium"])
        assert len(SAMPLE_TEXTS["medium"]) < len(SAMPLE_TEXTS["long"])
        
        # All samples should be non-empty strings
        for key, text in SAMPLE_TEXTS.items():
            assert isinstance(text, str)
            assert len(text) > 0
    
    def test_sample_conversations(self):
        """Test using conversation fixtures."""
        # Empty conversation
        assert SAMPLE_CONVERSATIONS["empty"] == []
        
        # Single turn conversation
        single = SAMPLE_CONVERSATIONS["single_turn"]
        assert len(single) == 1
        assert single[0]["role"] == "user"
        
        # Multi-turn conversation
        multi = SAMPLE_CONVERSATIONS["multi_turn"]
        assert len(multi) == 3
        assert multi[0]["role"] == "user"
        assert multi[1]["role"] == "assistant"
    
    def test_create_sample_context(self):
        """Test creating sample contexts."""
        # Create basic context
        ctx = create_sample_context()
        assert ctx.user_id == "test_user"
        assert ctx.current_emotion == "neutral"
        assert ctx.formality_level == 0.5
        
        # Create custom context
        ctx = create_sample_context(
            user_id="custom_user",
            emotion="joy",
            formality=0.8,
            language="zh"
        )
        assert ctx.user_id == "custom_user"
        assert ctx.current_emotion == "joy"
        assert ctx.formality_level == 0.8
        assert ctx.language == "zh"
    
    def test_create_sample_plan(self):
        """Test creating sample response plans."""
        # Create basic plan
        plan = create_sample_plan()
        assert OutputModality.TEXT in plan.modalities
        assert plan.style == ExpressionStyle.CASUAL
        
        # Create custom plan
        plan = create_sample_plan(
            modalities=[OutputModality.TEXT, OutputModality.SPEECH],
            style=ExpressionStyle.FORMAL,
            emotion="empathetic"
        )
        assert len(plan.modalities) == 2
        assert plan.style == ExpressionStyle.FORMAL
        assert plan.emotion == "empathetic"
    
    def test_sample_voices(self):
        """Test using voice configuration fixtures."""
        # English default voice
        voice = SAMPLE_VOICES["en_default"]
        assert voice.language == "en"
        assert voice.speed == 1.0
        
        # Fast voice
        voice = SAMPLE_VOICES["en_fast"]
        assert voice.speed == 1.3
        
        # Emotional voice
        voice = SAMPLE_VOICES["emotional_joy"]
        assert voice.emotion == "joy"
        assert voice.emotion_intensity == 0.7
    
    def test_generate_sample_audio(self):
        """Test generating sample audio."""
        # Generate 1 second of audio
        audio = generate_sample_audio(duration_seconds=1.0, sample_rate=22050)
        assert audio.shape == (22050,)
        assert audio.dtype == np.float32
        
        # Generate 2 seconds at different sample rate
        audio = generate_sample_audio(duration_seconds=2.0, sample_rate=16000)
        assert audio.shape == (32000,)
    
    @pytest.mark.parametrize("case", EMOTION_TEST_CASES)
    def test_emotion_test_cases(self, case):
        """Test using emotion test cases."""
        # Each case should have required fields
        assert "emotion" in case
        assert "expected_speed" in case
        assert "expected_pitch" in case
        
        # Values should be reasonable
        assert 0.5 <= case["expected_speed"] <= 2.0
        assert 0.5 <= case["expected_pitch"] <= 2.0
    
    @pytest.mark.parametrize("case", LANGUAGE_TEST_CASES)
    def test_language_test_cases(self, case):
        """Test using language test cases."""
        # Each case should have required fields
        assert "text" in case
        assert "expected_language" in case
        assert "description" in case
        
        # Language codes should be valid
        assert case["expected_language"] in ["en", "zh", "ja", "es"]


class TestMockBackendsDemo:
    """Demonstrate using mock backends."""
    
    def test_mock_nlg_backend(self):
        """Test using mock NLG backend."""
        # Create mock backend
        config = NLGConfig(backend=NLGBackend.TEMPLATE)
        backend = create_mock_nlg_backend(config)
        
        # Generate streaming response
        chunks = list(backend.generate_streaming("Hello", "You are helpful"))
        assert len(chunks) > 0
        assert backend.call_count == 1
        
        # Generate complete response
        response = backend.generate_complete("Hello", "You are helpful")
        assert isinstance(response, str)
        assert len(response) > 0
        assert backend.call_count == 2
        
        # Check last call
        assert backend.last_prompt == "Hello"
        assert backend.last_system_prompt == "You are helpful"
        
        # Test failure
        backend.should_fail = True
        with pytest.raises(Exception, match="Mock NLG failure"):
            backend.generate_complete("Hello", "System")
        
        # Reset
        backend.reset()
        assert backend.call_count == 0
    
    def test_mock_tts_backend(self):
        """Test using mock TTS backend."""
        # Create mock backend
        backend = create_mock_tts_backend(sample_rate=22050)
        
        # Synthesize speech
        voice = SAMPLE_VOICES["en_default"]
        audio = backend.synthesize("Hello", voice)
        assert audio.shape == (22050,)
        assert backend.call_count == 1
        
        # Synthesize streaming
        chunks = list(backend.synthesize_streaming("Hello. How are you?", voice))
        assert len(chunks) == 2  # Two sentences
        assert backend.call_count == 2
        
        # Check last call
        assert backend.last_text == "Hello. How are you?"
        assert backend.last_voice_config == voice
        
        # Test failure
        backend.should_fail = True
        with pytest.raises(Exception, match="Mock TTS failure"):
            backend.synthesize("Hello", voice)
        
        # Reset
        backend.reset()
        assert backend.call_count == 0
    
    def test_mock_template_engine(self):
        """Test using mock template engine."""
        # Create mock engine
        engine = create_mock_template_engine()
        
        # Generate from template
        response = engine.generate("Hello", ExpressionStyle.CASUAL)
        assert isinstance(response, str)
        assert "casual" in response.lower()
        assert engine.call_count == 1
        
        # Get specific template
        template = engine.get_template("greeting_template")
        assert template == "Hello! How can I help you today?"
        
        # Get non-existent template
        template = engine.get_template("nonexistent")
        assert template is None
        
        # Reset
        engine.reset()
        assert engine.call_count == 0
    
    def test_mock_tts_cache(self):
        """Test using mock TTS cache."""
        # Create mock cache
        cache = create_mock_tts_cache(max_size_mb=100)
        
        # Cache miss
        voice = SAMPLE_VOICES["en_default"]
        cached = cache.get("Hello", voice)
        assert cached is None
        assert cache.miss_count == 1
        
        # Put in cache
        audio = generate_sample_audio(duration_seconds=1.0)
        cache.put("Hello", voice, audio)
        
        # Cache hit
        cached = cache.get("Hello", voice)
        assert cached is not None
        np.testing.assert_array_equal(cached, audio)
        assert cache.hit_count == 1
        
        # Check hit rate
        assert cache.hit_rate == 0.5  # 1 hit, 1 miss
        
        # Clear cache
        cache.clear()
        assert len(cache.cache) == 0
        assert cache.current_size_mb == 0
        
        # Reset stats
        cache.reset_stats()
        assert cache.hit_count == 0
        assert cache.miss_count == 0
    
    def test_mock_language_detector(self):
        """Test using mock language detector."""
        # Create mock detector
        detector = create_mock_language_detector()
        
        # Detect English
        lang = detector.detect_language("Hello, how are you?")
        assert lang == "en"
        assert detector.call_count == 1
        
        # Detect Chinese
        lang = detector.detect_language("你好")
        assert lang == "zh"
        
        # Detect Japanese
        lang = detector.detect_language("こんにちは")
        assert lang == "ja"
        
        # Detect Spanish
        lang = detector.detect_language("¿Hola?")
        assert lang == "es"
        
        # Override detection
        detector.override_language = "fr"
        lang = detector.detect_language("Hello")
        assert lang == "fr"
        
        # Reset
        detector.reset()
        assert detector.call_count == 0
        assert detector.override_language is None
    
    def test_mock_emotion_mapper(self):
        """Test using mock emotion mapper."""
        # Create mock mapper
        mapper = create_mock_emotion_mapper()
        
        # Apply joy emotion
        voice = SAMPLE_VOICES["en_default"]
        voice.emotion = "joy"
        voice.emotion_intensity = 1.0
        
        adjusted = mapper.apply_emotion(voice)
        assert adjusted.speed == 1.1  # Joy increases speed
        assert adjusted.pitch == 1.1  # Joy increases pitch
        assert mapper.call_count == 1
        
        # Apply sadness emotion
        voice.emotion = "sadness"
        adjusted = mapper.apply_emotion(voice)
        assert adjusted.speed == 0.9  # Sadness decreases speed
        assert adjusted.pitch == 0.9  # Sadness decreases pitch
        
        # Apply with partial intensity
        voice.emotion = "joy"
        voice.emotion_intensity = 0.5
        adjusted = mapper.apply_emotion(voice)
        assert adjusted.speed == 1.05  # Half intensity
        assert adjusted.pitch == 1.05
        
        # Reset
        mapper.reset()
        assert mapper.call_count == 0


class TestIntegrationDemo:
    """Demonstrate integration testing with fixtures and mocks."""
    
    def test_nlg_with_context(self):
        """Test NLG generation with context."""
        # Create context and backend
        context = create_sample_context(
            emotion="joy",
            formality=0.3,
            language="en"
        )
        backend = create_mock_nlg_backend()
        
        # Generate response
        prompt = SAMPLE_TEXTS["short"]
        response = backend.generate_complete(prompt, "You are helpful")
        
        # Verify
        assert response is not None
        assert backend.call_count == 1
        assert prompt[:50] in response
    
    def test_tts_with_emotion(self):
        """Test TTS synthesis with emotion."""
        # Create emotional voice
        voice = SAMPLE_VOICES["emotional_joy"]
        backend = create_mock_tts_backend()
        mapper = create_mock_emotion_mapper()
        
        # Apply emotion
        adjusted_voice = mapper.apply_emotion(voice)
        
        # Synthesize
        audio = backend.synthesize(SAMPLE_TEXTS["short"], adjusted_voice)
        
        # Verify
        assert audio is not None
        assert backend.call_count == 1
        assert mapper.call_count == 1
    
    def test_caching_workflow(self):
        """Test TTS caching workflow."""
        # Create components
        backend = create_mock_tts_backend()
        cache = create_mock_tts_cache()
        voice = SAMPLE_VOICES["en_default"]
        text = SAMPLE_TEXTS["short"]
        
        # First synthesis (cache miss)
        cached = cache.get(text, voice)
        assert cached is None
        
        audio = backend.synthesize(text, voice)
        cache.put(text, voice, audio)
        
        # Second synthesis (cache hit)
        cached = cache.get(text, voice)
        assert cached is not None
        np.testing.assert_array_equal(cached, audio)
        
        # Verify cache stats
        assert cache.hit_count == 1
        assert cache.miss_count == 1
        assert cache.hit_rate == 0.5

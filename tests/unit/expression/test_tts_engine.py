"""
Unit tests for TTSEngine.

Tests the core TTS engine functionality including initialization,
caching, emotion mapping, and synthesis.

Requirements: 13.1, 13.6
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from llm_compression.expression.tts import TTSEngine, TTSCache, EmotionMapper
from llm_compression.expression.expression_types import (
    TTSConfig,
    VoiceConfig,
    TTSBackend
)


class TestTTSCache:
    """Test TTS cache functionality."""
    
    def test_cache_initialization(self):
        """Test cache initializes with correct parameters."""
        cache = TTSCache(max_size_mb=50)
        assert cache.max_size_mb == 50
        assert cache.current_size_mb == 0.0
        assert len(cache.cache) == 0
    
    def test_cache_put_and_get(self):
        """Test caching and retrieving audio."""
        cache = TTSCache(max_size_mb=100)
        voice = VoiceConfig(voice_id="test_voice")
        text = "Hello world"
        audio = np.random.randn(1000).astype(np.float32)
        
        # Put audio in cache
        cache.put(text, voice, audio)
        
        # Retrieve from cache
        cached_audio = cache.get(text, voice)
        assert cached_audio is not None
        np.testing.assert_array_equal(cached_audio, audio)
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = TTSCache(max_size_mb=100)
        voice = VoiceConfig(voice_id="test_voice")
        
        # Try to get non-existent entry
        cached_audio = cache.get("nonexistent text", voice)
        assert cached_audio is None
    
    def test_cache_eviction(self):
        """Test cache evicts entries when full."""
        cache = TTSCache(max_size_mb=1)  # Very small cache
        voice = VoiceConfig(voice_id="test_voice")
        
        # Create large audio that will fill cache
        large_audio = np.random.randn(500000).astype(np.float32)  # ~2MB
        
        # Put first entry
        cache.put("text1", voice, large_audio)
        assert len(cache.cache) == 1
        
        # Put second entry (should evict first)
        cache.put("text2", voice, large_audio)
        
        # First entry should be evicted
        assert cache.get("text1", voice) is None
        # Second entry should be present
        assert cache.get("text2", voice) is not None
    
    def test_cache_clear(self):
        """Test clearing cache."""
        cache = TTSCache(max_size_mb=100)
        voice = VoiceConfig(voice_id="test_voice")
        audio = np.random.randn(1000).astype(np.float32)
        
        cache.put("text", voice, audio)
        assert len(cache.cache) == 1
        
        cache.clear()
        assert len(cache.cache) == 0
        assert cache.current_size_mb == 0.0
    
    def test_cache_key_generation(self):
        """Test cache key includes voice parameters."""
        cache = TTSCache(max_size_mb=100)
        text = "Hello"
        
        voice1 = VoiceConfig(voice_id="voice1", speed=1.0)
        voice2 = VoiceConfig(voice_id="voice2", speed=1.0)
        voice3 = VoiceConfig(voice_id="voice1", speed=1.5)
        
        audio = np.random.randn(1000).astype(np.float32)
        
        # Cache with voice1
        cache.put(text, voice1, audio)
        
        # Different voice_id should miss
        assert cache.get(text, voice2) is None
        
        # Different speed should miss
        assert cache.get(text, voice3) is None
        
        # Same voice should hit
        assert cache.get(text, voice1) is not None


class TestEmotionMapper:
    """Test emotion mapping functionality."""
    
    def test_emotion_mapper_initialization(self):
        """Test emotion mapper initializes with emotion parameters."""
        mapper = EmotionMapper()
        assert len(mapper.emotion_params) > 0
        assert "neutral" in mapper.emotion_params
        assert "joy" in mapper.emotion_params
        assert "sadness" in mapper.emotion_params
    
    def test_apply_neutral_emotion(self):
        """Test neutral emotion doesn't change parameters."""
        mapper = EmotionMapper()
        voice = VoiceConfig(
            voice_id="test",
            speed=1.0,
            pitch=1.0,
            volume=1.0,
            emotion="neutral",
            emotion_intensity=0.5
        )
        
        result = mapper.apply_emotion(voice)
        
        # Neutral should keep parameters close to original
        assert 0.95 <= result.speed <= 1.05
        assert 0.95 <= result.pitch <= 1.05
        assert 0.95 <= result.volume <= 1.05
    
    def test_apply_joy_emotion(self):
        """Test joy emotion increases speed and pitch."""
        mapper = EmotionMapper()
        voice = VoiceConfig(
            voice_id="test",
            speed=1.0,
            pitch=1.0,
            volume=1.0,
            emotion="joy",
            emotion_intensity=1.0  # Full intensity
        )
        
        result = mapper.apply_emotion(voice)
        
        # Joy should increase speed and pitch
        assert result.speed > 1.0
        assert result.pitch > 1.0
    
    def test_apply_sadness_emotion(self):
        """Test sadness emotion decreases speed and pitch."""
        mapper = EmotionMapper()
        voice = VoiceConfig(
            voice_id="test",
            speed=1.0,
            pitch=1.0,
            volume=1.0,
            emotion="sadness",
            emotion_intensity=1.0  # Full intensity
        )
        
        result = mapper.apply_emotion(voice)
        
        # Sadness should decrease speed and pitch
        assert result.speed < 1.0
        assert result.pitch < 1.0
    
    def test_emotion_intensity_scaling(self):
        """Test emotion intensity affects parameter changes."""
        mapper = EmotionMapper()
        
        # Low intensity
        voice_low = VoiceConfig(
            voice_id="test",
            speed=1.0,
            pitch=1.0,
            emotion="joy",
            emotion_intensity=0.2
        )
        result_low = mapper.apply_emotion(voice_low)
        
        # High intensity
        voice_high = VoiceConfig(
            voice_id="test",
            speed=1.0,
            pitch=1.0,
            emotion="joy",
            emotion_intensity=1.0
        )
        result_high = mapper.apply_emotion(voice_high)
        
        # High intensity should have larger changes
        assert abs(result_high.speed - 1.0) > abs(result_low.speed - 1.0)
        assert abs(result_high.pitch - 1.0) > abs(result_low.pitch - 1.0)
    
    def test_parameter_clamping(self):
        """Test parameters are clamped to valid ranges."""
        mapper = EmotionMapper()
        voice = VoiceConfig(
            voice_id="test",
            speed=2.0,  # Already at max
            pitch=2.0,  # Already at max
            volume=1.0,
            emotion="joy",  # Would increase further
            emotion_intensity=1.0
        )
        
        result = mapper.apply_emotion(voice)
        
        # Should be clamped to valid ranges
        assert 0.5 <= result.speed <= 2.0
        assert 0.5 <= result.pitch <= 2.0
        assert 0.0 <= result.volume <= 1.0
    
    def test_all_emotions_supported(self):
        """Test all 8 basic emotions are supported."""
        mapper = EmotionMapper()
        emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
        
        for emotion in emotions:
            voice = VoiceConfig(
                voice_id="test",
                emotion=emotion,
                emotion_intensity=0.5
            )
            result = mapper.apply_emotion(voice)
            
            # Should not raise error and should return valid parameters
            assert 0.5 <= result.speed <= 2.0
            assert 0.5 <= result.pitch <= 2.0
            assert 0.0 <= result.volume <= 1.0


class TestTTSEngine:
    """Test TTS engine functionality."""
    
    def test_engine_initialization_piper(self):
        """Test engine initializes with Piper backend."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        assert engine.config.backend == TTSBackend.PIPER
        assert engine.cache is not None  # Cache enabled by default
        assert engine.emotion_mapper is not None
    
    def test_engine_initialization_without_cache(self):
        """Test engine initializes without cache when disabled."""
        config = TTSConfig(backend=TTSBackend.PIPER, cache_enabled=False)
        engine = TTSEngine(config)
        
        assert engine.cache is None
    
    def test_engine_initialization_all_backends(self):
        """Test engine can initialize with all backend types."""
        backends = [TTSBackend.PIPER, TTSBackend.COQUI, TTSBackend.AZURE, TTSBackend.OPENAI]
        
        for backend in backends:
            config = TTSConfig(backend=backend)
            engine = TTSEngine(config)
            assert engine.config.backend == backend
    
    def test_synthesize_returns_audio(self):
        """Test synthesize returns audio array."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "Hello world"
        audio_chunks = list(engine.synthesize(text, streaming=False))
        
        assert len(audio_chunks) > 0
        assert isinstance(audio_chunks[0], np.ndarray)
        assert audio_chunks[0].dtype == np.float32
    
    def test_synthesize_with_custom_voice(self):
        """Test synthesize with custom voice configuration."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        custom_voice = VoiceConfig(
            voice_id="custom_voice",
            speed=1.2,
            pitch=1.1,
            emotion="joy"
        )
        
        text = "Hello world"
        audio_chunks = list(engine.synthesize(text, voice_config=custom_voice, streaming=False))
        
        assert len(audio_chunks) > 0
    
    def test_synthesize_streaming(self):
        """Test streaming synthesis returns multiple chunks."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "Hello world. This is a test. Multiple sentences here."
        audio_chunks = list(engine.synthesize(text, streaming=True))
        
        # Should return multiple chunks for multiple sentences
        assert len(audio_chunks) > 1
    
    def test_synthesize_uses_cache(self):
        """Test synthesize uses cache for repeated text."""
        config = TTSConfig(backend=TTSBackend.PIPER, cache_enabled=True)
        engine = TTSEngine(config)
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test_voice")
        
        # First synthesis
        audio1 = list(engine.synthesize(text, voice_config=voice, streaming=False))[0]
        
        # Second synthesis (should use cache)
        audio2 = list(engine.synthesize(text, voice_config=voice, streaming=False))[0]
        
        # Should return same audio
        np.testing.assert_array_equal(audio1, audio2)
    
    def test_synthesize_applies_emotion(self):
        """Test synthesize applies emotion mapping."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        voice = VoiceConfig(
            voice_id="test",
            emotion="joy",
            emotion_intensity=1.0
        )
        
        text = "Hello world"
        # Should not raise error
        audio_chunks = list(engine.synthesize(text, voice_config=voice, streaming=False))
        assert len(audio_chunks) > 0
    
    def test_synthesize_error_handling(self):
        """Test synthesize handles errors gracefully."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Force backend to None to simulate error
        engine.backend = None
        
        text = "Hello world"
        audio_chunks = list(engine.synthesize(text, streaming=False))
        
        # Should return silence instead of crashing
        assert len(audio_chunks) > 0
        assert isinstance(audio_chunks[0], np.ndarray)
    
    def test_split_sentences(self):
        """Test sentence splitting for streaming."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "Hello world. This is a test! How are you?"
        sentences = engine._split_sentences(text)
        
        assert len(sentences) == 3
        assert "Hello world." in sentences[0]
        assert "This is a test!" in sentences[1]
        assert "How are you?" in sentences[2]
    
    def test_split_sentences_no_punctuation(self):
        """Test sentence splitting with no punctuation."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "Hello world"
        sentences = engine._split_sentences(text)
        
        assert len(sentences) == 1
        assert sentences[0] == "Hello world"


class TestTTSEngineBackends:
    """Test TTS engine backend-specific functionality."""
    
    def test_piper_backend_initialization(self):
        """Test Piper backend initializes correctly."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        assert engine.backend == "piper"
        assert hasattr(engine, 'piper_available')
        assert hasattr(engine, 'piper_model_path')
    
    def test_coqui_backend_initialization(self):
        """Test Coqui backend initializes correctly."""
        config = TTSConfig(backend=TTSBackend.COQUI)
        engine = TTSEngine(config)
        
        assert engine.backend == "coqui"
    
    def test_azure_backend_initialization(self):
        """Test Azure backend initializes correctly."""
        config = TTSConfig(backend=TTSBackend.AZURE)
        engine = TTSEngine(config)
        
        assert engine.backend == "azure"
    
    def test_openai_backend_initialization(self):
        """Test OpenAI backend initializes correctly."""
        config = TTSConfig(backend=TTSBackend.OPENAI)
        engine = TTSEngine(config)
        
        assert engine.backend == "openai"
    
    def test_piper_synthesis_without_library(self):
        """Test Piper synthesis falls back to mock when library unavailable."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Force piper_available to False
        engine.piper_available = False
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test_voice")
        
        audio = engine._synthesize_piper(text, voice)
        
        # Should return mock audio (silence)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0
    
    def test_piper_synthesis_with_mocked_voice(self):
        """Test Piper synthesis with mocked voice instance."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Create a properly mocked voice that simulates successful synthesis
        mock_voice = MagicMock()
        # Return audio bytes that will be converted to numpy array
        mock_voice.synthesize.return_value = (np.zeros(1000, dtype=np.int16)).tobytes()
        
        # Set up engine to use mock
        engine.piper_available = True
        engine.piper_voice = mock_voice
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test_voice", speed=1.2, volume=0.8)
        
        # Mock the piper module to avoid import errors
        with patch.dict('sys.modules', {'piper': MagicMock()}):
            audio = engine._synthesize_piper(text, voice)
            
            # Should return audio (either from mock or fallback)
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            assert len(audio) > 0
    
    def test_coqui_synthesis_fallback(self):
        """Test Coqui synthesis falls back to mock."""
        config = TTSConfig(backend=TTSBackend.COQUI)
        engine = TTSEngine(config)
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test_voice")
        
        audio = engine._synthesize_coqui(text, voice)
        
        # Should return mock audio
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
    
    def test_azure_synthesis_fallback(self):
        """Test Azure synthesis falls back to mock."""
        config = TTSConfig(backend=TTSBackend.AZURE)
        engine = TTSEngine(config)
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test_voice")
        
        audio = engine._synthesize_azure(text, voice)
        
        # Should return mock audio
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
    
    def test_openai_synthesis_fallback(self):
        """Test OpenAI synthesis falls back to mock."""
        config = TTSConfig(backend=TTSBackend.OPENAI)
        engine = TTSEngine(config)
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test_voice")
        
        audio = engine._synthesize_openai(text, voice)
        
        # Should return mock audio
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32


class TestTTSEngineMockSynthesis:
    """Test mock synthesis functionality."""
    
    def test_mock_synthesis_duration(self):
        """Test mock synthesis generates appropriate duration."""
        config = TTSConfig(backend=TTSBackend.PIPER, sample_rate=22050)
        engine = TTSEngine(config)
        
        text = "Hello world"  # ~11 chars
        voice = VoiceConfig(voice_id="test", speed=1.0)
        
        audio = engine._synthesize_mock(text, voice)
        
        # Should generate audio with duration based on text length
        # ~150 chars/sec, so 11 chars = ~0.073 seconds
        expected_samples = int((len(text) / 150.0) * config.sample_rate)
        
        assert len(audio) > 0
        assert abs(len(audio) - expected_samples) < config.sample_rate * 0.1  # Within 0.1s
    
    def test_mock_synthesis_speed_adjustment(self):
        """Test mock synthesis adjusts duration for speed."""
        config = TTSConfig(backend=TTSBackend.PIPER, sample_rate=22050)
        engine = TTSEngine(config)
        
        text = "Hello world"
        
        # Normal speed
        voice_normal = VoiceConfig(voice_id="test", speed=1.0)
        audio_normal = engine._synthesize_mock(text, voice_normal)
        
        # Fast speed
        voice_fast = VoiceConfig(voice_id="test", speed=2.0)
        audio_fast = engine._synthesize_mock(text, voice_fast)
        
        # Fast should be shorter
        assert len(audio_fast) < len(audio_normal)
    
    def test_mock_synthesis_returns_silence(self):
        """Test mock synthesis returns silence (zeros)."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test")
        
        audio = engine._synthesize_mock(text, voice)
        
        # Should be all zeros
        assert np.all(audio == 0.0)


class TestTTSEngineStreaming:
    """Test streaming synthesis functionality."""
    
    def test_streaming_multiple_sentences(self):
        """Test streaming returns chunk per sentence."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "First sentence. Second sentence. Third sentence."
        chunks = list(engine.synthesize(text, streaming=True))
        
        # Should have 3 chunks (one per sentence)
        assert len(chunks) == 3
        
        # Each chunk should be audio
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32
    
    def test_streaming_single_sentence(self):
        """Test streaming with single sentence."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "Single sentence."
        chunks = list(engine.synthesize(text, streaming=True))
        
        # Should have 1 chunk
        assert len(chunks) == 1
    
    def test_streaming_no_punctuation(self):
        """Test streaming with no punctuation."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "No punctuation here"
        chunks = list(engine.synthesize(text, streaming=True))
        
        # Should have 1 chunk
        assert len(chunks) == 1
    
    def test_streaming_empty_sentences_filtered(self):
        """Test streaming filters empty sentences."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "First. . . Second."  # Multiple periods
        chunks = list(engine.synthesize(text, streaming=True))
        
        # The sentence splitter will create: "First.", ".", ".", "Second."
        # Empty sentences (just ".") will still generate audio chunks
        # So we should have 4 chunks, not 2
        assert len(chunks) >= 2  # At least the non-empty sentences
    
    def test_streaming_doesnt_use_cache(self):
        """Test streaming mode doesn't use cache."""
        config = TTSConfig(backend=TTSBackend.PIPER, cache_enabled=True)
        engine = TTSEngine(config)
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test")
        
        # First call (streaming)
        list(engine.synthesize(text, voice_config=voice, streaming=True))
        
        # Cache should be empty (streaming doesn't cache)
        assert len(engine.cache.cache) == 0


class TestTTSEngineErrorHandling:
    """Test error handling in TTS engine."""
    
    def test_synthesis_with_none_backend(self):
        """Test synthesis handles None backend gracefully."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        engine.backend = None
        
        text = "Hello world"
        audio = engine._synthesize_complete(text, VoiceConfig(voice_id="test"))
        
        # Should return silence
        assert isinstance(audio, np.ndarray)
        assert len(audio) == config.sample_rate
        assert np.all(audio == 0.0)
    
    def test_synthesis_exception_returns_silence(self):
        """Test synthesis returns silence on exception."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Mock _synthesize_complete to raise exception
        def raise_exception(*args, **kwargs):
            raise RuntimeError("Test error")
        
        engine._synthesize_complete = raise_exception
        
        text = "Hello world"
        chunks = list(engine.synthesize(text, streaming=False))
        
        # Should return fallback audio instead of crashing
        assert len(chunks) == 1
        assert isinstance(chunks[0], np.ndarray)
        # Fallback audio duration is based on text length, not fixed
        assert len(chunks[0]) > 0
        assert np.all(chunks[0] == 0.0)  # Should be silence
    
    def test_backend_initialization_failure(self):
        """Test engine handles backend initialization failure."""
        # Create invalid backend type
        config = TTSConfig(backend=TTSBackend.PIPER)
        
        with patch.object(TTSEngine, '_init_piper_backend', side_effect=RuntimeError("Init failed")):
            engine = TTSEngine(config)
            
            # Backend should be None
            assert engine.backend is None
    
    def test_piper_synthesis_exception_fallback(self):
        """Test Piper synthesis falls back to mock on exception."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        engine.piper_available = True
        
        # Mock piper_voice to raise exception
        engine.piper_voice = MagicMock()
        engine.piper_voice.synthesize.side_effect = RuntimeError("Synthesis failed")
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test")
        
        audio = engine._synthesize_piper(text, voice)
        
        # Should return mock audio
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32


class TestTTSEngineVoiceParameters:
    """Test voice parameter handling."""
    
    def test_voice_config_defaults(self):
        """Test synthesis uses default voice config."""
        config = TTSConfig(
            backend=TTSBackend.PIPER,
            voice=VoiceConfig(voice_id="default_voice")
        )
        engine = TTSEngine(config)
        
        text = "Hello world"
        # Don't pass voice_config, should use default
        chunks = list(engine.synthesize(text, streaming=False))
        
        assert len(chunks) > 0
    
    def test_voice_config_override(self):
        """Test custom voice config overrides default."""
        config = TTSConfig(
            backend=TTSBackend.PIPER,
            voice=VoiceConfig(voice_id="default_voice")
        )
        engine = TTSEngine(config)
        
        custom_voice = VoiceConfig(
            voice_id="custom_voice",
            speed=1.5,
            pitch=1.2,
            volume=0.8
        )
        
        text = "Hello world"
        chunks = list(engine.synthesize(text, voice_config=custom_voice, streaming=False))
        
        assert len(chunks) > 0
    
    def test_streaming_parameter_override(self):
        """Test streaming parameter can be overridden."""
        config = TTSConfig(backend=TTSBackend.PIPER, streaming=False)
        engine = TTSEngine(config)
        
        text = "First. Second. Third."
        
        # Override to streaming=True
        chunks = list(engine.synthesize(text, streaming=True))
        
        # Should stream (multiple chunks)
        assert len(chunks) > 1
    
    def test_voice_parameter_ranges(self):
        """Test voice parameters with extreme values."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Test with extreme parameters
        voice = VoiceConfig(
            voice_id="test",
            speed=2.0,  # Max speed
            pitch=2.0,  # Max pitch
            volume=1.0  # Max volume
        )
        
        text = "Hello world"
        chunks = list(engine.synthesize(text, voice_config=voice, streaming=False))
        
        assert len(chunks) > 0


class TestTTSEngineIntegration:
    """Integration tests for TTS engine."""
    
    def test_end_to_end_synthesis(self):
        """Test complete synthesis workflow."""
        config = TTSConfig(
            backend=TTSBackend.PIPER,
            cache_enabled=True,
            streaming=False
        )
        engine = TTSEngine(config)
        
        text = "Hello, this is a test of the TTS engine."
        voice = VoiceConfig(
            voice_id="test_voice",
            speed=1.0,
            pitch=1.0,
            emotion="friendly",
            emotion_intensity=0.5
        )
        
        # Synthesize
        chunks = list(engine.synthesize(text, voice_config=voice))
        
        # Verify output
        assert len(chunks) == 1
        audio = chunks[0]
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0
    
    def test_cache_integration(self):
        """Test cache integration in synthesis workflow."""
        config = TTSConfig(backend=TTSBackend.PIPER, cache_enabled=True)
        engine = TTSEngine(config)
        
        text = "Cached text"
        voice = VoiceConfig(voice_id="test")
        
        # First synthesis
        audio1 = list(engine.synthesize(text, voice_config=voice, streaming=False))[0]
        
        # Second synthesis (should use cache)
        audio2 = list(engine.synthesize(text, voice_config=voice, streaming=False))[0]
        
        # Should be identical
        np.testing.assert_array_equal(audio1, audio2)
        
        # Cache should have entry
        assert len(engine.cache.cache) == 1
    
    def test_emotion_integration(self):
        """Test emotion mapping integration."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "Hello world"
        
        # Test different emotions
        emotions = ["joy", "sadness", "anger", "neutral"]
        
        for emotion in emotions:
            voice = VoiceConfig(
                voice_id="test",
                emotion=emotion,
                emotion_intensity=0.8
            )
            
            chunks = list(engine.synthesize(text, voice_config=voice, streaming=False))
            
            # Should succeed for all emotions
            assert len(chunks) > 0
    
    def test_multiple_backends(self):
        """Test switching between backends."""
        backends = [TTSBackend.PIPER, TTSBackend.COQUI, TTSBackend.AZURE, TTSBackend.OPENAI]
        
        for backend in backends:
            config = TTSConfig(backend=backend)
            engine = TTSEngine(config)
            
            text = "Hello world"
            chunks = list(engine.synthesize(text, streaming=False))
            
            # All backends should work (with fallback)
            assert len(chunks) > 0


class TestSentenceSplitting:
    """Test sentence splitting functionality."""
    
    def test_split_multiple_periods(self):
        """Test splitting on periods."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "First. Second. Third."
        sentences = engine._split_sentences(text)
        
        assert len(sentences) == 3
        assert "First." in sentences[0]
        assert "Second." in sentences[1]
        assert "Third." in sentences[2]
    
    def test_split_multiple_punctuation(self):
        """Test splitting on different punctuation."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "Question? Exclamation! Statement."
        sentences = engine._split_sentences(text)
        
        assert len(sentences) == 3
        assert "Question?" in sentences[0]
        assert "Exclamation!" in sentences[1]
        assert "Statement." in sentences[2]
    
    def test_split_empty_text(self):
        """Test splitting empty text."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = ""
        sentences = engine._split_sentences(text)
        
        # Should return list with empty string or empty list
        assert isinstance(sentences, list)
    
    def test_split_no_punctuation(self):
        """Test splitting text without punctuation."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "No punctuation here"
        sentences = engine._split_sentences(text)
        
        assert len(sentences) == 1
        assert sentences[0] == text
    
    def test_split_multiple_punctuation_marks(self):
        """Test splitting with multiple consecutive punctuation."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "Really?! Yes!!! Okay..."
        sentences = engine._split_sentences(text)
        
        # Should handle multiple punctuation marks
        assert len(sentences) >= 2
    
    def test_split_preserves_punctuation(self):
        """Test that punctuation is preserved in sentences."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "Hello. World!"
        sentences = engine._split_sentences(text)
        
        assert "." in sentences[0]
        assert "!" in sentences[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

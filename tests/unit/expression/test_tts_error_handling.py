"""
Unit tests for TTSEngine error handling.

Tests comprehensive error handling including fallback to text-only,
logging with context, user-friendly messages, backend-specific errors,
and timeout scenarios.

Requirements: 11.1, 11.5, 11.6, 13.1, 13.6
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from llm_compression.expression.tts import TTSEngine, TTSError
from llm_compression.expression.expression_types import (
    TTSConfig,
    VoiceConfig,
    TTSBackend
)


class TestTTSError:
    """Test TTSError dataclass."""
    
    def test_tts_error_creation(self):
        """Test creating TTSError with all fields."""
        error = TTSError(
            error_type="synthesis",
            message="User-friendly message",
            details="Technical details",
            fallback_used=True,
            original_exception=ValueError("test")
        )
        
        assert error.error_type == "synthesis"
        assert error.message == "User-friendly message"
        assert error.details == "Technical details"
        assert error.fallback_used is True
        assert isinstance(error.original_exception, ValueError)
    
    def test_tts_error_without_exception(self):
        """Test creating TTSError without original exception."""
        error = TTSError(
            error_type="backend_init",
            message="Backend failed",
            details="Init error",
            fallback_used=True
        )
        
        assert error.original_exception is None


class TestBackendInitializationErrors:
    """Test error handling during backend initialization."""
    
    def test_backend_init_failure_sets_fallback_mode(self):
        """Test backend init failure enables fallback mode."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        
        with patch.object(TTSEngine, '_init_piper_backend', side_effect=RuntimeError("Init failed")):
            engine = TTSEngine(config)
            
            assert engine.fallback_mode is True
            assert engine.backend is None
            assert engine.last_error is not None
            assert engine.last_error.error_type == "backend_init"
    
    def test_backend_init_failure_logs_error(self, caplog):
        """Test backend init failure logs with context."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        
        with patch.object(TTSEngine, '_init_piper_backend', side_effect=RuntimeError("Init failed")):
            engine = TTSEngine(config)
            
            # Check error was logged
            assert any("TTS Error" in record.message for record in caplog.records)
            assert any("backend_init" in record.message for record in caplog.records)
    
    def test_backend_init_success_clears_fallback_mode(self):
        """Test successful backend init clears fallback mode."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Should not be in fallback mode if init succeeds
        assert engine.fallback_mode is False
    
    def test_unsupported_backend_raises_error(self):
        """Test unsupported backend type triggers error handling."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        
        # Mock to raise ValueError for unsupported backend
        with patch.object(TTSEngine, '_init_piper_backend', side_effect=ValueError("Unsupported")):
            engine = TTSEngine(config)
            
            assert engine.fallback_mode is True
            assert engine.last_error is not None


class TestSynthesisErrorHandling:
    """Test error handling during synthesis."""
    
    def test_synthesis_in_fallback_mode_returns_silence(self):
        """Test synthesis in fallback mode returns silence immediately."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        engine.fallback_mode = True
        
        text = "Hello world"
        chunks = list(engine.synthesize(text, streaming=False))
        
        assert len(chunks) == 1
        assert isinstance(chunks[0], np.ndarray)
        assert np.all(chunks[0] == 0.0)  # Should be silence
    
    def test_synthesis_error_returns_fallback_audio(self):
        """Test synthesis error returns fallback audio."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Mock _synthesize_complete to raise error
        with patch.object(engine, '_synthesize_complete', side_effect=RuntimeError("Synthesis failed")):
            text = "Hello world"
            chunks = list(engine.synthesize(text, streaming=False))
            
            assert len(chunks) == 1
            assert isinstance(chunks[0], np.ndarray)
            # Should return fallback audio (silence)
            assert np.all(chunks[0] == 0.0)
    
    def test_synthesis_error_logs_with_context(self, caplog):
        """Test synthesis error logs with context."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        with patch.object(engine, '_synthesize_complete', side_effect=RuntimeError("Synthesis failed")):
            text = "Hello world"
            list(engine.synthesize(text, streaming=False))
            
            # Check error was logged with context
            assert any("TTS Error" in record.message for record in caplog.records)
            assert any("synthesis" in record.message for record in caplog.records)
    
    def test_synthesis_error_sets_last_error(self):
        """Test synthesis error sets last_error."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        with patch.object(engine, '_synthesize_complete', side_effect=RuntimeError("Synthesis failed")):
            text = "Hello world"
            list(engine.synthesize(text, streaming=False))
            
            assert engine.last_error is not None
            assert engine.last_error.error_type == "synthesis"
            assert engine.last_error.fallback_used is True
    
    def test_synthesis_error_increments_error_count(self):
        """Test synthesis errors increment error count."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        initial_count = engine.error_count
        
        with patch.object(engine, '_synthesize_complete', side_effect=RuntimeError("Synthesis failed")):
            text = "Hello world"
            list(engine.synthesize(text, streaming=False))
            
            assert engine.error_count == initial_count + 1


class TestStreamingErrorHandling:
    """Test error handling in streaming mode."""
    
    def test_streaming_sentence_error_continues_with_fallback(self):
        """Test streaming continues with fallback when sentence fails."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Mock to fail on second sentence
        call_count = [0]
        def mock_synthesize_sentence(sentence, voice):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Sentence 2 failed")
            return np.zeros(1000, dtype=np.float32)
        
        with patch.object(engine, '_synthesize_sentence', side_effect=mock_synthesize_sentence):
            text = "First sentence. Second sentence. Third sentence."
            chunks = list(engine.synthesize(text, streaming=True))
            
            # Should still get 3 chunks (with fallback for second)
            assert len(chunks) == 3
            assert all(isinstance(chunk, np.ndarray) for chunk in chunks)
    
    def test_streaming_error_logs_sentence_number(self, caplog):
        """Test streaming error logs which sentence failed."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        with patch.object(engine, '_synthesize_sentence', side_effect=RuntimeError("Failed")):
            text = "First. Second. Third."
            list(engine.synthesize(text, streaming=True))
            
            # Should log sentence numbers
            assert any("sentence" in record.message.lower() for record in caplog.records)
    
    def test_streaming_all_sentences_fail_returns_all_fallbacks(self):
        """Test streaming returns fallback for all failed sentences."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        with patch.object(engine, '_synthesize_sentence', side_effect=RuntimeError("Failed")):
            text = "First. Second. Third."
            chunks = list(engine.synthesize(text, streaming=True))
            
            # Should get 3 fallback chunks
            assert len(chunks) == 3
            assert all(np.all(chunk == 0.0) for chunk in chunks)


class TestTimeoutHandling:
    """Test timeout handling in synthesis."""
    
    @pytest.mark.skipif(not hasattr(__import__('signal'), 'SIGALRM'), reason="SIGALRM not available on Windows")
    def test_synthesis_timeout_returns_fallback(self):
        """Test synthesis timeout returns fallback audio."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Mock to simulate long-running synthesis
        def slow_synthesis(text, voice):
            import time
            time.sleep(2)
            return np.zeros(1000, dtype=np.float32)
        
        with patch.object(engine, '_synthesize_complete', side_effect=slow_synthesis):
            text = "Hello world"
            chunks = list(engine.synthesize(text, streaming=False, timeout_seconds=0.1))
            
            # Should return fallback due to timeout
            assert len(chunks) == 1
            assert isinstance(chunks[0], np.ndarray)
    
    def test_timeout_error_type_is_timeout(self):
        """Test timeout errors have correct error type."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Directly create a timeout scenario
        text = "Hello world"
        voice = VoiceConfig(voice_id="test")
        
        # Mock _synthesize_complete to raise TimeoutError
        with patch.object(engine, '_synthesize_complete', side_effect=TimeoutError("Timeout")):
            # Call the safe wrapper which should handle the timeout
            audio = engine._synthesize_complete_safe(text, voice, timeout_seconds=1.0)
            
            # Should return fallback audio
            assert isinstance(audio, np.ndarray)
            assert engine.last_error is not None
            assert engine.last_error.error_type == "timeout"
    
    @pytest.mark.skipif(not hasattr(__import__('signal'), 'SIGALRM'), reason="SIGALRM not available on Windows")
    def test_streaming_timeout_continues_with_fallback(self):
        """Test streaming continues after timeout on one sentence."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Mock to timeout on second sentence
        call_count = [0]
        def mock_synthesize_sentence(sentence, voice):
            call_count[0] += 1
            if call_count[0] == 2:
                import time
                time.sleep(2)
            return np.zeros(1000, dtype=np.float32)
        
        with patch.object(engine, '_synthesize_sentence', side_effect=mock_synthesize_sentence):
            text = "First. Second. Third."
            chunks = list(engine.synthesize(text, streaming=True, timeout_seconds=0.1))
            
            # Should get 3 chunks (with fallback for second)
            assert len(chunks) == 3


class TestCacheErrorHandling:
    """Test error handling with cache operations."""
    
    def test_cache_retrieval_error_continues_synthesis(self):
        """Test cache retrieval error doesn't prevent synthesis."""
        config = TTSConfig(backend=TTSBackend.PIPER, cache_enabled=True)
        engine = TTSEngine(config)
        
        # Mock cache.get to raise error
        with patch.object(engine.cache, 'get', side_effect=RuntimeError("Cache error")):
            text = "Hello world"
            chunks = list(engine.synthesize(text, streaming=False))
            
            # Should still synthesize successfully
            assert len(chunks) == 1
            assert isinstance(chunks[0], np.ndarray)
    
    def test_cache_put_error_doesnt_fail_synthesis(self, caplog):
        """Test cache put error doesn't fail synthesis."""
        config = TTSConfig(backend=TTSBackend.PIPER, cache_enabled=True)
        engine = TTSEngine(config)
        
        # Mock cache.put to raise error
        with patch.object(engine.cache, 'put', side_effect=RuntimeError("Cache error")):
            text = "Hello world"
            chunks = list(engine.synthesize(text, streaming=False))
            
            # Should still return audio
            assert len(chunks) == 1
            assert isinstance(chunks[0], np.ndarray)
            
            # Should log warning
            assert any("cache" in record.message.lower() for record in caplog.records)


class TestEmotionMapperErrorHandling:
    """Test error handling with emotion mapping."""
    
    def test_emotion_mapping_error_uses_original_voice(self, caplog):
        """Test emotion mapping error uses original voice config."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Mock emotion_mapper.apply_emotion to raise error
        with patch.object(engine.emotion_mapper, 'apply_emotion', side_effect=RuntimeError("Emotion error")):
            text = "Hello world"
            voice = VoiceConfig(voice_id="test", emotion="joy")
            chunks = list(engine.synthesize(text, voice_config=voice, streaming=False))
            
            # Should still synthesize
            assert len(chunks) == 1
            
            # Should log warning
            assert any("emotion" in record.message.lower() for record in caplog.records)


class TestBackendSpecificErrors:
    """Test backend-specific error handling."""
    
    def test_piper_import_error_returns_fallback(self):
        """Test Piper import error returns fallback audio."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        engine.piper_available = True
        engine.piper_voice = None
        
        # Mock the import inside _synthesize_piper to fail
        text = "Hello world"
        voice = VoiceConfig(voice_id="test")
        
        # Patch the import at the point where it's used
        with patch('builtins.__import__', side_effect=ImportError("No piper")):
            audio = engine._synthesize_piper(text, voice)
            
            # Should return fallback
            assert isinstance(audio, np.ndarray)
            assert np.all(audio == 0.0)
            assert engine.last_error is not None
            assert engine.last_error.error_type == "backend_init"
    
    def test_piper_voice_load_error_returns_fallback(self):
        """Test Piper voice loading error returns fallback."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        engine.piper_available = True
        engine.piper_voice = None
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test")
        
        # Mock Path.exists to return True so it tries to load
        # Then mock PiperVoice.load to fail
        with patch('pathlib.Path.exists', return_value=True):
            with patch('piper.PiperVoice.load', side_effect=RuntimeError("Load failed")):
                audio = engine._synthesize_piper(text, voice)
                
                # Should return fallback
                assert isinstance(audio, np.ndarray)
                assert engine.last_error is not None
                assert engine.last_error.error_type == "backend_init"
    
    def test_piper_synthesis_error_returns_fallback(self):
        """Test Piper synthesis error returns fallback."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        engine.piper_available = True
        
        # Mock piper_voice to raise error during synthesis
        engine.piper_voice = MagicMock()
        engine.piper_voice.synthesize.side_effect = RuntimeError("Synthesis failed")
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test")
        audio = engine._synthesize_piper(text, voice)
        
        # Should return fallback
        assert isinstance(audio, np.ndarray)
        assert engine.last_error is not None
        assert engine.last_error.error_type == "synthesis"
    
    def test_coqui_backend_not_implemented(self):
        """Test Coqui backend returns fallback with error."""
        config = TTSConfig(backend=TTSBackend.COQUI)
        engine = TTSEngine(config)
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test")
        audio = engine._synthesize_coqui(text, voice)
        
        # Should return fallback
        assert isinstance(audio, np.ndarray)
        assert engine.last_error is not None
        assert "not implemented" in engine.last_error.message.lower()
    
    def test_azure_backend_not_implemented(self):
        """Test Azure backend returns fallback with error."""
        config = TTSConfig(backend=TTSBackend.AZURE)
        engine = TTSEngine(config)
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test")
        audio = engine._synthesize_azure(text, voice)
        
        # Should return fallback
        assert isinstance(audio, np.ndarray)
        assert engine.last_error is not None
        assert "not implemented" in engine.last_error.message.lower()
    
    def test_openai_backend_not_implemented(self):
        """Test OpenAI backend returns fallback with error."""
        config = TTSConfig(backend=TTSBackend.OPENAI)
        engine = TTSEngine(config)
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test")
        audio = engine._synthesize_openai(text, voice)
        
        # Should return fallback
        assert isinstance(audio, np.ndarray)
        assert engine.last_error is not None
        assert "not implemented" in engine.last_error.message.lower()


class TestErrorReporting:
    """Test error reporting and user-friendly messages."""
    
    def test_get_last_error_returns_none_initially(self):
        """Test get_last_error returns None when no errors."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Clear any init errors
        engine.last_error = None
        
        assert engine.get_last_error() is None
    
    def test_get_last_error_returns_last_error(self):
        """Test get_last_error returns the last error."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        with patch.object(engine, '_synthesize_complete', side_effect=RuntimeError("Test error")):
            text = "Hello world"
            list(engine.synthesize(text, streaming=False))
            
            error = engine.get_last_error()
            assert error is not None
            assert error.error_type == "synthesis"
            assert "Test error" in error.details
    
    def test_is_fallback_mode_returns_correct_state(self):
        """Test is_fallback_mode returns correct state."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Initially should not be in fallback mode (if init succeeds)
        if engine.backend is not None:
            assert engine.is_fallback_mode() is False
        
        # Force fallback mode
        engine.fallback_mode = True
        assert engine.is_fallback_mode() is True
    
    def test_error_count_tracks_errors(self):
        """Test error_count tracks number of errors."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        initial_count = engine.error_count
        
        # Trigger multiple errors
        with patch.object(engine, '_synthesize_complete', side_effect=RuntimeError("Error")):
            for i in range(3):
                list(engine.synthesize(f"Text {i}", streaming=False))
        
        assert engine.error_count == initial_count + 3
    
    def test_user_friendly_message_in_error(self):
        """Test errors contain user-friendly messages."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        with patch.object(engine, '_synthesize_complete', side_effect=RuntimeError("Technical error")):
            text = "Hello world"
            list(engine.synthesize(text, streaming=False))
            
            error = engine.get_last_error()
            assert error is not None
            # User message should be friendly, not technical
            assert "Speech synthesis failed" in error.message
            assert "Audio output unavailable" in error.message


class TestFallbackAudioCreation:
    """Test fallback audio creation."""
    
    def test_create_fallback_audio_returns_silence(self):
        """Test _create_fallback_audio returns silence."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "Hello world"
        voice = VoiceConfig(voice_id="test")
        audio = engine._create_fallback_audio(text, voice)
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert np.all(audio == 0.0)
    
    def test_create_fallback_audio_duration_based_on_text(self):
        """Test fallback audio duration scales with text length."""
        config = TTSConfig(backend=TTSBackend.PIPER, sample_rate=22050)
        engine = TTSEngine(config)
        
        short_text = "Hi"
        long_text = "This is a much longer text that should result in longer audio duration."
        voice = VoiceConfig(voice_id="test", speed=1.0)
        
        short_audio = engine._create_fallback_audio(short_text, voice)
        long_audio = engine._create_fallback_audio(long_text, voice)
        
        # Longer text should produce longer audio
        assert len(long_audio) > len(short_audio)
    
    def test_create_fallback_audio_adjusts_for_speed(self):
        """Test fallback audio duration adjusts for voice speed."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        text = "Hello world"
        voice_normal = VoiceConfig(voice_id="test", speed=1.0)
        voice_fast = VoiceConfig(voice_id="test", speed=2.0)
        
        audio_normal = engine._create_fallback_audio(text, voice_normal)
        audio_fast = engine._create_fallback_audio(text, voice_fast)
        
        # Fast speed should produce shorter audio
        assert len(audio_fast) < len(audio_normal)


class TestGracefulDegradation:
    """Test graceful degradation under various failure scenarios."""
    
    def test_complete_backend_failure_still_returns_audio(self):
        """Test complete backend failure still returns audio."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        
        with patch.object(TTSEngine, '_init_piper_backend', side_effect=RuntimeError("Init failed")):
            engine = TTSEngine(config)
            
            text = "Hello world"
            chunks = list(engine.synthesize(text, streaming=False))
            
            # Should still return audio (fallback)
            assert len(chunks) == 1
            assert isinstance(chunks[0], np.ndarray)
    
    def test_multiple_consecutive_errors_handled(self):
        """Test multiple consecutive errors are handled."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        with patch.object(engine, '_synthesize_complete', side_effect=RuntimeError("Error")):
            # Trigger multiple errors
            for i in range(5):
                chunks = list(engine.synthesize(f"Text {i}", streaming=False))
                assert len(chunks) == 1
                assert isinstance(chunks[0], np.ndarray)
        
        # Should track all errors
        assert engine.error_count >= 5
    
    def test_partial_streaming_failure_returns_partial_results(self):
        """Test partial streaming failure returns what succeeded."""
        config = TTSConfig(backend=TTSBackend.PIPER)
        engine = TTSEngine(config)
        
        # Mock to succeed on first sentence, fail on rest
        call_count = [0]
        def mock_synthesize_sentence(sentence, voice):
            call_count[0] += 1
            if call_count[0] == 1:
                return np.ones(1000, dtype=np.float32)  # Success
            raise RuntimeError("Failed")
        
        with patch.object(engine, '_synthesize_sentence', side_effect=mock_synthesize_sentence):
            text = "First. Second. Third."
            chunks = list(engine.synthesize(text, streaming=True))
            
            # Should get 3 chunks (1 success, 2 fallbacks)
            assert len(chunks) == 3
            # First should be non-zero (success)
            assert not np.all(chunks[0] == 0.0)
            # Rest should be fallback (zeros)
            assert np.all(chunks[1] == 0.0)
            assert np.all(chunks[2] == 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

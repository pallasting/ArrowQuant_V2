"""
Mock backends for Expression Layer testing

Provides mock implementations of TTS and NLG backends for testing without
requiring actual API keys or model files.
"""

import numpy as np
from typing import Iterator, Optional, List, Dict, Any
from unittest.mock import Mock

from llm_compression.expression.expression_types import (
    VoiceConfig,
    NLGConfig,
    ExpressionStyle,
    ExpressionContext,
)


class MockNLGBackend:
    """Mock NLG backend for testing."""
    
    def __init__(self, config: NLGConfig):
        self.config = config
        self.call_count = 0
        self.last_prompt = None
        self.last_system_prompt = None
        self.should_fail = False
        self.failure_message = "Mock NLG failure"
    
    def generate_streaming(
        self,
        prompt: str,
        system_prompt: str
    ) -> Iterator[str]:
        """Generate streaming response."""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        # Return mock response in chunks
        response = f"Mock response to: {prompt[:50]}"
        words = response.split()
        
        for word in words:
            yield word + " "
    
    def generate_complete(
        self,
        prompt: str,
        system_prompt: str
    ) -> str:
        """Generate complete response."""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        return f"Mock response to: {prompt[:50]}"
    
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.last_prompt = None
        self.last_system_prompt = None
        self.should_fail = False


class MockTTSBackend:
    """Mock TTS backend for testing."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.call_count = 0
        self.last_text = None
        self.last_voice_config = None
        self.should_fail = False
        self.failure_message = "Mock TTS failure"
    
    def synthesize(
        self,
        text: str,
        voice_config: VoiceConfig
    ) -> np.ndarray:
        """Synthesize speech from text."""
        self.call_count += 1
        self.last_text = text
        self.last_voice_config = voice_config
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        # Return mock audio (1 second of silence)
        return np.zeros(self.sample_rate, dtype=np.float32)
    
    def synthesize_streaming(
        self,
        text: str,
        voice_config: VoiceConfig
    ) -> Iterator[np.ndarray]:
        """Synthesize speech in streaming mode."""
        self.call_count += 1
        self.last_text = text
        self.last_voice_config = voice_config
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        # Split text into sentences and yield audio for each
        sentences = text.split('.')
        for _ in sentences:
            if _.strip():
                # Return 0.5 seconds of audio per sentence
                yield np.zeros(self.sample_rate // 2, dtype=np.float32)
    
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.last_text = None
        self.last_voice_config = None
        self.should_fail = False


class MockTemplateEngine:
    """Mock template engine for testing."""
    
    def __init__(self):
        self.templates = {
            "greeting_template": "Hello! How can I help you today?",
            "goodbye_template": "Goodbye! Have a great day!",
            "acknowledgment_template": "I understand.",
            "error_template": "I apologize, but I encountered an error.",
        }
        self.call_count = 0
        self.last_prompt = None
        self.last_style = None
    
    def generate(
        self,
        prompt: str,
        style: ExpressionStyle
    ) -> str:
        """Generate response from template."""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_style = style
        
        # Return a simple template-based response
        return f"Template response ({style.value}): {prompt[:30]}"
    
    def get_template(self, template_name: str) -> Optional[str]:
        """Get template by name."""
        return self.templates.get(template_name)
    
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.last_prompt = None
        self.last_style = None


class MockTTSCache:
    """Mock TTS cache for testing."""
    
    def __init__(self, max_size_mb: int = 100):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size_mb = max_size_mb
        self.current_size_mb = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
    
    def get(self, text: str, voice: VoiceConfig) -> Optional[np.ndarray]:
        """Get cached audio."""
        key = self._make_key(text, voice)
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        else:
            self.miss_count += 1
            return None
    
    def put(self, text: str, voice: VoiceConfig, audio: np.ndarray):
        """Cache audio."""
        key = self._make_key(text, voice)
        size_mb = audio.nbytes / (1024 * 1024)
        
        if self.current_size_mb + size_mb > self.max_size_mb:
            self._evict()
        
        self.cache[key] = audio
        self.current_size_mb += size_mb
    
    def _make_key(self, text: str, voice: VoiceConfig) -> str:
        """Generate cache key."""
        import hashlib
        content = f"{text}_{voice.voice_id}_{voice.speed}_{voice.pitch}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _evict(self):
        """Evict oldest entry."""
        if self.cache:
            key = next(iter(self.cache))
            audio = self.cache.pop(key)
            self.current_size_mb -= audio.nbytes / (1024 * 1024)
            self.eviction_count += 1
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.current_size_mb = 0
    
    def reset_stats(self):
        """Reset statistics."""
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return self.hit_count / total


class MockLanguageDetector:
    """Mock language detector for testing."""
    
    def __init__(self):
        self.call_count = 0
        self.last_text = None
        self.override_language = None
    
    def detect_language(self, text: str) -> str:
        """Detect language from text."""
        self.call_count += 1
        self.last_text = text
        
        if self.override_language:
            return self.override_language
        
        # Simple heuristic detection
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return "zh"
        elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            return "ja"
        elif any(char in 'áéíóúñ¿¡' for char in text.lower()):
            return "es"
        else:
            return "en"
    
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.last_text = None
        self.override_language = None


class MockEmotionMapper:
    """Mock emotion mapper for testing."""
    
    def __init__(self):
        self.call_count = 0
        self.last_voice_config = None
        self.emotion_params = {
            "joy": {"speed": 1.1, "pitch": 1.1},
            "sadness": {"speed": 0.9, "pitch": 0.9},
            "anger": {"speed": 1.2, "pitch": 1.15},
            "fear": {"speed": 1.15, "pitch": 1.2},
            "neutral": {"speed": 1.0, "pitch": 1.0},
            "empathetic": {"speed": 0.95, "pitch": 0.98},
            "friendly": {"speed": 1.05, "pitch": 1.05},
        }
    
    def apply_emotion(self, voice: VoiceConfig) -> VoiceConfig:
        """Apply emotion to voice configuration."""
        self.call_count += 1
        self.last_voice_config = voice
        
        params = self.emotion_params.get(voice.emotion, self.emotion_params["neutral"])
        intensity = voice.emotion_intensity
        
        # Create new voice config with adjusted parameters
        adjusted_voice = VoiceConfig(
            voice_id=voice.voice_id,
            language=voice.language,
            speed=1.0 + (params["speed"] - 1.0) * intensity,
            pitch=1.0 + (params["pitch"] - 1.0) * intensity,
            volume=voice.volume,
            emotion=voice.emotion,
            emotion_intensity=voice.emotion_intensity,
            accent=voice.accent,
        )
        
        return adjusted_voice
    
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.last_voice_config = None


def create_mock_nlg_backend(config: NLGConfig = None) -> MockNLGBackend:
    """Create a mock NLG backend for testing."""
    if config is None:
        from llm_compression.expression.expression_types import NLGBackend
        config = NLGConfig(backend=NLGBackend.TEMPLATE)
    return MockNLGBackend(config)


def create_mock_tts_backend(sample_rate: int = 22050) -> MockTTSBackend:
    """Create a mock TTS backend for testing."""
    return MockTTSBackend(sample_rate)


def create_mock_template_engine() -> MockTemplateEngine:
    """Create a mock template engine for testing."""
    return MockTemplateEngine()


def create_mock_tts_cache(max_size_mb: int = 100) -> MockTTSCache:
    """Create a mock TTS cache for testing."""
    return MockTTSCache(max_size_mb)


def create_mock_language_detector() -> MockLanguageDetector:
    """Create a mock language detector for testing."""
    return MockLanguageDetector()


def create_mock_emotion_mapper() -> MockEmotionMapper:
    """Create a mock emotion mapper for testing."""
    return MockEmotionMapper()

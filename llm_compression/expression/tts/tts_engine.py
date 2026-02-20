"""
Text-to-Speech Engine with multiple backend support.

This module implements the core TTS engine that supports multiple backends
(Piper, Coqui, Azure, OpenAI) for speech synthesis with emotion control,
caching, and streaming capabilities.

Requirements: 1.1, 1.3
"""

import logging
from typing import Iterator, Optional, Dict, Any
from pathlib import Path
import numpy as np

from llm_compression.expression.expression_types import (
    TTSConfig,
    VoiceConfig,
    TTSBackend
)

logger = logging.getLogger(__name__)


class TTSCache:
    """
    Cache for TTS outputs to improve performance.
    
    Implements a simple FIFO cache with size limits to store frequently
    used TTS outputs and avoid redundant synthesis.
    
    Requirements: 10.5
    """
    
    def __init__(self, max_size_mb: int = 100):
        """
        Initialize TTS cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size_mb = max_size_mb
        self.current_size_mb = 0.0
        logger.info(f"Initialized TTS cache with max size: {max_size_mb}MB")
    
    def get(self, text: str, voice: VoiceConfig) -> Optional[np.ndarray]:
        """
        Get cached audio for text and voice configuration.
        
        Args:
            text: Input text
            voice: Voice configuration
            
        Returns:
            Cached audio array or None if not found
        """
        key = self._make_key(text, voice)
        audio = self.cache.get(key)
        
        if audio is not None:
            logger.debug(f"Cache hit for text: {text[:50]}...")
        
        return audio
    
    def put(self, text: str, voice: VoiceConfig, audio: np.ndarray):
        """
        Cache audio for text and voice configuration.
        
        Args:
            text: Input text
            voice: Voice configuration
            audio: Generated audio array
        """
        key = self._make_key(text, voice)
        size_mb = audio.nbytes / (1024 * 1024)
        
        # Evict if necessary
        while self.current_size_mb + size_mb > self.max_size_mb and self.cache:
            self._evict()
        
        self.cache[key] = audio
        self.current_size_mb += size_mb
        logger.debug(f"Cached audio for text: {text[:50]}... (size: {size_mb:.2f}MB)")
    
    def _make_key(self, text: str, voice: VoiceConfig) -> str:
        """
        Generate cache key from text and voice configuration.
        
        Args:
            text: Input text
            voice: Voice configuration
            
        Returns:
            Cache key string
        """
        import hashlib
        content = f"{text}_{voice.voice_id}_{voice.speed}_{voice.pitch}_{voice.emotion}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _evict(self):
        """Evict oldest entry from cache (FIFO)."""
        if self.cache:
            key = next(iter(self.cache))
            audio = self.cache.pop(key)
            size_mb = audio.nbytes / (1024 * 1024)
            self.current_size_mb -= size_mb
            logger.debug(f"Evicted cache entry (freed {size_mb:.2f}MB)")
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.current_size_mb = 0.0
        logger.info("Cleared TTS cache")


class EmotionMapper:
    """
    Maps emotions to voice parameters.
    
    Converts emotional states (joy, sadness, anger, etc.) into concrete
    voice parameter adjustments (speed, pitch, volume) with intensity scaling.
    
    Requirements: 3.2, 3.4, 3.5
    """
    
    def __init__(self):
        """Initialize emotion mapper with parameter mappings."""
        self.emotion_params = {
            "joy": {"speed": 1.1, "pitch": 1.1, "volume": 1.0},
            "sadness": {"speed": 0.9, "pitch": 0.9, "volume": 0.9},
            "anger": {"speed": 1.2, "pitch": 1.15, "volume": 1.0},
            "fear": {"speed": 1.15, "pitch": 1.2, "volume": 0.95},
            "surprise": {"speed": 1.1, "pitch": 1.15, "volume": 1.0},
            "disgust": {"speed": 0.95, "pitch": 0.95, "volume": 0.95},
            "trust": {"speed": 1.0, "pitch": 1.0, "volume": 1.0},
            "anticipation": {"speed": 1.05, "pitch": 1.05, "volume": 1.0},
            "neutral": {"speed": 1.0, "pitch": 1.0, "volume": 1.0},
            "empathetic": {"speed": 0.95, "pitch": 0.98, "volume": 0.95},
            "friendly": {"speed": 1.05, "pitch": 1.05, "volume": 1.0},
        }
        logger.info(f"Initialized emotion mapper with {len(self.emotion_params)} emotions")
    
    def apply_emotion(self, voice: VoiceConfig) -> VoiceConfig:
        """
        Apply emotion to voice configuration.
        
        Adjusts voice parameters (speed, pitch, volume) based on the
        specified emotion and intensity.
        
        Args:
            voice: Voice configuration with emotion settings
            
        Returns:
            Voice configuration with adjusted parameters
        """
        params = self.emotion_params.get(
            voice.emotion,
            self.emotion_params["neutral"]
        )
        
        # Apply emotion with intensity scaling
        intensity = voice.emotion_intensity
        
        # Calculate adjusted parameters
        # Formula: current + (target - 1.0) * intensity
        voice.speed = voice.speed * (1.0 + (params["speed"] - 1.0) * intensity)
        voice.pitch = voice.pitch * (1.0 + (params["pitch"] - 1.0) * intensity)
        voice.volume = voice.volume * (1.0 + (params["volume"] - 1.0) * intensity)
        
        # Clamp to valid ranges
        voice.speed = max(0.5, min(2.0, voice.speed))
        voice.pitch = max(0.5, min(2.0, voice.pitch))
        voice.volume = max(0.0, min(1.0, voice.volume))
        
        logger.debug(
            f"Applied emotion '{voice.emotion}' (intensity={intensity:.2f}): "
            f"speed={voice.speed:.2f}, pitch={voice.pitch:.2f}, volume={voice.volume:.2f}"
        )
        
        return voice


class TTSEngine:
    """
    Text-to-Speech engine with multiple backend support.
    
    Supports:
    - Piper (fast, local)
    - Coqui TTS (high quality, local)
    - Azure TTS (cloud, many voices)
    - OpenAI TTS (cloud, natural)
    
    Features:
    - Emotion-aware synthesis
    - Output caching
    - Streaming support
    - Graceful error handling
    
    Requirements: 1.1, 1.3
    """
    
    def __init__(self, config: TTSConfig):
        """
        Initialize TTS engine.
        
        Args:
            config: TTS configuration
        """
        self.config = config
        self.backend = None
        self.cache = TTSCache() if config.cache_enabled else None
        self.emotion_mapper = EmotionMapper()
        
        logger.info(
            f"Initializing TTS engine with backend: {config.backend.value}, "
            f"cache_enabled: {config.cache_enabled}"
        )
        
        # Initialize backend
        self._init_backend()
    
    def _init_backend(self):
        """
        Initialize TTS backend based on configuration.
        
        Loads the appropriate backend (Piper, Coqui, Azure, OpenAI) and
        prepares it for synthesis.
        """
        try:
            if self.config.backend == TTSBackend.PIPER:
                self._init_piper_backend()
            elif self.config.backend == TTSBackend.COQUI:
                self._init_coqui_backend()
            elif self.config.backend == TTSBackend.AZURE:
                self._init_azure_backend()
            elif self.config.backend == TTSBackend.OPENAI:
                self._init_openai_backend()
            else:
                raise ValueError(f"Unsupported TTS backend: {self.config.backend}")
            
            logger.info(f"Successfully initialized {self.config.backend.value} backend")
        
        except Exception as e:
            logger.error(f"Failed to initialize TTS backend: {e}")
            self.backend = None
    
    def _init_piper_backend(self):
        """
        Initialize Piper TTS backend (fast, local).
        
        Piper is a fast, local TTS engine that provides high-quality speech
        synthesis without requiring cloud APIs. It uses ONNX models for
        efficient inference.
        
        Requirements: 1.1, 1.2, 1.3
        """
        try:
            # Import piper-tts if available
            try:
                import piper
                self.piper_available = True
                logger.info("Piper TTS library available")
            except ImportError:
                logger.warning(
                    "piper-tts not installed. Install with: pip install piper-tts\n"
                    "Falling back to mock synthesis for testing."
                )
                self.piper_available = False
            
            # Initialize Piper voice model
            self.piper_voice = None
            self.piper_model_path = None
            
            # Get model path from config or use default
            if hasattr(self.config, 'model_path'):
                self.piper_model_path = Path(self.config.model_path).expanduser()
            else:
                # Default model path
                self.piper_model_path = Path.home() / ".ai-os" / "models" / "piper"
            
            logger.info(f"Piper backend initialized (model path: {self.piper_model_path})")
            self.backend = "piper"
            
        except Exception as e:
            logger.error(f"Failed to initialize Piper backend: {e}")
            raise
    
    def _init_coqui_backend(self):
        """Initialize Coqui TTS backend (high quality, local)."""
        try:
            # Coqui TTS initialization
            # This is a placeholder for the actual Coqui initialization
            logger.info("Coqui backend ready (lazy initialization)")
            self.backend = "coqui"
        except Exception as e:
            logger.error(f"Failed to initialize Coqui backend: {e}")
            raise
    
    def _init_azure_backend(self):
        """Initialize Azure TTS backend (cloud, many voices)."""
        try:
            # Azure Speech SDK initialization
            # This is a placeholder for the actual Azure initialization
            logger.info("Azure backend ready (lazy initialization)")
            self.backend = "azure"
        except Exception as e:
            logger.error(f"Failed to initialize Azure backend: {e}")
            raise
    
    def _init_openai_backend(self):
        """Initialize OpenAI TTS backend (cloud, natural)."""
        try:
            # OpenAI client initialization
            # This is a placeholder for the actual OpenAI initialization
            logger.info("OpenAI backend ready (lazy initialization)")
            self.backend = "openai"
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI backend: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        voice_config: Optional[VoiceConfig] = None,
        streaming: Optional[bool] = None
    ) -> Iterator[np.ndarray]:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            voice_config: Voice configuration (overrides default)
            streaming: Enable streaming (overrides config)
            
        Yields:
            Audio chunks as numpy arrays (float32, sample_rate from config)
            
        Requirements: 1.2, 1.7, 11.1
        """
        voice = voice_config or self.config.voice
        streaming = streaming if streaming is not None else self.config.streaming
        
        logger.info(
            f"Synthesizing text (length={len(text)}, "
            f"voice={voice.voice_id}, streaming={streaming})"
        )
        
        # Check cache first (only for non-streaming)
        if not streaming and self.cache:
            cached = self.cache.get(text, voice)
            if cached is not None:
                logger.info("Using cached audio")
                yield cached
                return
        
        # Apply emotion to voice parameters
        voice = self.emotion_mapper.apply_emotion(voice)
        
        # Generate speech
        try:
            if streaming:
                yield from self._synthesize_streaming(text, voice)
            else:
                audio = self._synthesize_complete(text, voice)
                
                # Cache the result
                if self.cache:
                    self.cache.put(text, voice, audio)
                
                yield audio
        
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            # Return silence as fallback
            silence = np.zeros(self.config.sample_rate, dtype=np.float32)
            yield silence
    
    def _synthesize_streaming(
        self,
        text: str,
        voice: VoiceConfig
    ) -> Iterator[np.ndarray]:
        """
        Synthesize speech in streaming mode.
        
        Splits text into sentences and synthesizes each sentence separately
        for lower latency and continuous output.
        
        Args:
            text: Input text
            voice: Voice configuration
            
        Yields:
            Audio chunks for each sentence
        """
        # Split text into sentences
        sentences = self._split_sentences(text)
        
        logger.debug(f"Streaming synthesis for {len(sentences)} sentences")
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                logger.debug(f"Synthesizing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
                audio = self._synthesize_sentence(sentence, voice)
                yield audio
    
    def _synthesize_complete(
        self,
        text: str,
        voice: VoiceConfig
    ) -> np.ndarray:
        """
        Synthesize complete speech (non-streaming).
        
        Args:
            text: Input text
            voice: Voice configuration
            
        Returns:
            Complete audio as numpy array
        """
        if self.backend is None:
            logger.error("No backend available for synthesis")
            return np.zeros(self.config.sample_rate, dtype=np.float32)
        
        # Backend-specific synthesis
        logger.debug(f"Synthesizing with {self.config.backend.value} backend")
        
        if self.config.backend == TTSBackend.PIPER:
            return self._synthesize_piper(text, voice)
        elif self.config.backend == TTSBackend.COQUI:
            return self._synthesize_coqui(text, voice)
        elif self.config.backend == TTSBackend.AZURE:
            return self._synthesize_azure(text, voice)
        elif self.config.backend == TTSBackend.OPENAI:
            return self._synthesize_openai(text, voice)
        else:
            # Fallback to mock synthesis
            logger.warning(f"Backend {self.config.backend.value} not implemented, using mock")
            return self._synthesize_mock(text, voice)
    
    def _synthesize_sentence(
        self,
        sentence: str,
        voice: VoiceConfig
    ) -> np.ndarray:
        """
        Synthesize single sentence.
        
        Args:
            sentence: Input sentence
            voice: Voice configuration
            
        Returns:
            Audio for sentence as numpy array
        """
        return self._synthesize_complete(sentence, voice)
    
    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences for streaming.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'([.!?]+)', text)
        
        # Recombine sentences with their punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            if sentence:
                result.append(sentence)
        
        # Handle last sentence if no punctuation
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result if result else [text]
    
    def _synthesize_piper(
        self,
        text: str,
        voice: VoiceConfig
    ) -> np.ndarray:
        """
        Synthesize speech using Piper TTS backend.
        
        Piper is a fast, local TTS engine that uses ONNX models for
        efficient inference. It provides high-quality speech synthesis
        without requiring cloud APIs.
        
        Args:
            text: Input text to synthesize
            voice: Voice configuration with speed, pitch, volume settings
            
        Returns:
            Audio as numpy array (float32, sample_rate from config)
            
        Requirements: 1.1, 1.2, 1.3
        """
        try:
            if not self.piper_available:
                logger.warning("Piper not available, using mock synthesis")
                return self._synthesize_mock(text, voice)
            
            # Import piper
            from piper import PiperVoice
            
            # Load voice model if not already loaded
            if self.piper_voice is None:
                # Basic piper voice for English
                voice_name = "en_US-lessac-medium"
                
                # Ensure model directory exists
                self.piper_model_path.mkdir(parents=True, exist_ok=True)
                model_path = self.piper_model_path / f"{voice_name}.onnx"
                config_path = self.piper_model_path / f"{voice_name}.onnx.json"
                
                # Try to find and download voice if needed
                try:
                    if not model_path.exists() or not config_path.exists():
                        import urllib.request
                        logger.info(f"Downloading Piper voice {voice_name}...")
                        base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{voice_name}"
                        urllib.request.urlretrieve(f"{base_url}.onnx", str(model_path))
                        urllib.request.urlretrieve(f"{base_url}.onnx.json", str(config_path))
                        logger.info("Download complete!")
                        
                    # Load voice
                    self.piper_voice = PiperVoice.load(model_path, config_path)
                    logger.info(f"Loaded Piper voice: {voice_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load Piper voice: {e}, using mock synthesis")
                    return self._synthesize_mock(text, voice)
            
            # Synthesize with Piper
            from piper.config import SynthesisConfig
            syn_config = SynthesisConfig(
                speaker_id=None,
                length_scale=1.0 / voice.speed
            )
            
            # Piper yields chunks; we collect the float32 numpy arrays
            arrays = [chunk.audio_float_array for chunk in self.piper_voice.synthesize(text, syn_config=syn_config)]
            if not arrays:
                return self._synthesize_mock(text, voice)
                
            audio = np.concatenate(arrays)
            
            # Apply volume
            audio = audio * voice.volume
            
            # Note: Pitch control is not directly supported by Piper
            # For production, you would need to use additional audio processing
            # libraries like librosa or pyrubberband for pitch shifting
            
            logger.debug(
                f"Piper synthesis complete: {len(audio)} samples, "
                f"{len(audio) / self.config.sample_rate:.2f}s"
            )
            
            return audio
            
        except Exception as e:
            logger.error(f"Piper synthesis failed: {e}")
            return self._synthesize_mock(text, voice)
    
    def _synthesize_coqui(
        self,
        text: str,
        voice: VoiceConfig
    ) -> np.ndarray:
        """
        Synthesize speech using Coqui TTS backend.
        
        Placeholder for Coqui TTS implementation.
        
        Args:
            text: Input text
            voice: Voice configuration
            
        Returns:
            Audio as numpy array
        """
        logger.warning("Coqui TTS not implemented, using mock synthesis")
        return self._synthesize_mock(text, voice)
    
    def _synthesize_azure(
        self,
        text: str,
        voice: VoiceConfig
    ) -> np.ndarray:
        """
        Synthesize speech using Azure TTS backend.
        
        Placeholder for Azure TTS implementation.
        
        Args:
            text: Input text
            voice: Voice configuration
            
        Returns:
            Audio as numpy array
        """
        logger.warning("Azure TTS not implemented, using mock synthesis")
        return self._synthesize_mock(text, voice)
    
    def _synthesize_openai(
        self,
        text: str,
        voice: VoiceConfig
    ) -> np.ndarray:
        """
        Synthesize speech using OpenAI TTS backend.
        
        Placeholder for OpenAI TTS implementation.
        
        Args:
            text: Input text
            voice: Voice configuration
            
        Returns:
            Audio as numpy array
        """
        logger.warning("OpenAI TTS not implemented, using mock synthesis")
        return self._synthesize_mock(text, voice)
    
    def _synthesize_mock(
        self,
        text: str,
        voice: VoiceConfig
    ) -> np.ndarray:
        """
        Mock synthesis for testing and fallback.
        
        Generates silence with appropriate duration based on text length.
        This is used when the actual TTS backend is not available or fails.
        
        Args:
            text: Input text
            voice: Voice configuration
            
        Returns:
            Silence audio as numpy array
        """
        # Estimate duration: ~150 chars/sec at normal speed
        base_duration_seconds = len(text) / 150.0
        
        # Adjust for speed
        duration_seconds = base_duration_seconds / voice.speed
        
        # Generate samples
        samples = int(duration_seconds * self.config.sample_rate)
        audio = np.zeros(samples, dtype=np.float32)
        
        logger.debug(f"Mock synthesis: {samples} samples, {duration_seconds:.2f}s")
        
        return audio

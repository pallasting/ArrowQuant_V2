"""
Text-to-Speech Engine with multiple backend support.

This module implements the core TTS engine that supports multiple backends
(Piper, Coqui, Azure, OpenAI) for speech synthesis with emotion control,
caching, and streaming capabilities.

Requirements: 1.1, 1.3, 11.1, 11.5, 11.6
"""

import logging
from typing import Iterator, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from llm_compression.expression.expression_types import (
    TTSConfig,
    VoiceConfig,
    TTSBackend
)

logger = logging.getLogger(__name__)


@dataclass
class TTSError:
    """
    TTS error information for user-friendly reporting.
    
    Requirements: 11.5, 11.6
    """
    error_type: str  # "backend_init", "synthesis", "timeout", "network"
    message: str  # User-friendly message
    details: str  # Technical details for logging
    fallback_used: bool  # Whether fallback was applied
    original_exception: Optional[Exception] = None


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
    - Graceful error handling with fallbacks
    - User-friendly error messages
    
    Requirements: 1.1, 1.3, 11.1, 11.5, 11.6
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
        self.last_error: Optional[TTSError] = None
        self.error_count = 0
        self.fallback_mode = False
        
        logger.info(
            f"Initializing TTS engine with backend: {config.backend.value}, "
            f"cache_enabled: {config.cache_enabled}"
        )
        
        # Initialize backend with error handling
        self._init_backend()
    
    def _init_backend(self):
        """
        Initialize TTS backend based on configuration.
        
        Loads the appropriate backend (Piper, Coqui, Azure, OpenAI) and
        prepares it for synthesis. Handles initialization errors gracefully.
        
        Requirements: 11.1, 11.5, 11.6
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
            self.fallback_mode = False
        
        except Exception as e:
            error = TTSError(
                error_type="backend_init",
                message=f"Failed to initialize {self.config.backend.value} TTS backend. Text-only mode will be used.",
                details=f"Backend initialization error: {str(e)}",
                fallback_used=True,
                original_exception=e
            )
            self._handle_error(error)
            self.backend = None
            self.fallback_mode = True
    
    def _handle_error(self, error: TTSError):
        """
        Handle TTS errors with logging and user-friendly messages.
        
        Args:
            error: TTSError with error information
            
        Requirements: 11.5, 11.6
        """
        self.last_error = error
        self.error_count += 1
        
        # Log with context
        logger.error(
            f"TTS Error [{error.error_type}]: {error.details}",
            extra={
                "error_type": error.error_type,
                "fallback_used": error.fallback_used,
                "error_count": self.error_count,
                "backend": self.config.backend.value
            },
            exc_info=error.original_exception
        )
        
        # Log user-friendly message at warning level
        if error.fallback_used:
            logger.warning(f"User message: {error.message}")
    
    def get_last_error(self) -> Optional[TTSError]:
        """
        Get the last error that occurred.
        
        Returns:
            Last TTSError or None if no errors
            
        Requirements: 11.6
        """
        return self.last_error
    
    def is_fallback_mode(self) -> bool:
        """
        Check if engine is in fallback mode (text-only).
        
        Returns:
            True if in fallback mode, False otherwise
            
        Requirements: 11.1
        """
        return self.fallback_mode
    
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
        streaming: Optional[bool] = None,
        timeout_seconds: Optional[float] = None
    ) -> Iterator[np.ndarray]:
        """
        Synthesize speech from text with comprehensive error handling.
        
        Args:
            text: Input text to synthesize
            voice_config: Voice configuration (overrides default)
            streaming: Enable streaming (overrides config)
            timeout_seconds: Timeout for synthesis (optional)
            
        Yields:
            Audio chunks as numpy arrays (float32, sample_rate from config)
            Falls back to silence if synthesis fails
            
        Requirements: 1.2, 1.7, 11.1, 11.4, 11.5, 11.6
        """
        voice = voice_config or self.config.voice
        streaming = streaming if streaming is not None else self.config.streaming
        
        logger.info(
            f"Synthesizing text (length={len(text)}, "
            f"voice={voice.voice_id}, streaming={streaming}, "
            f"fallback_mode={self.fallback_mode})"
        )
        
        # If in fallback mode, return silence immediately
        if self.fallback_mode:
            logger.warning("TTS in fallback mode, returning silence")
            yield self._create_fallback_audio(text, voice)
            return
        
        # Check cache first (only for non-streaming)
        if not streaming and self.cache:
            try:
                cached = self.cache.get(text, voice)
                if cached is not None:
                    logger.info("Using cached audio")
                    yield cached
                    return
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}, proceeding with synthesis")
        
        # Apply emotion to voice parameters
        try:
            voice = self.emotion_mapper.apply_emotion(voice)
        except Exception as e:
            logger.warning(f"Emotion mapping failed: {e}, using original voice config")
        
        # Generate speech with error handling
        try:
            if streaming:
                yield from self._synthesize_streaming_safe(text, voice, timeout_seconds)
            else:
                audio = self._synthesize_complete_safe(text, voice, timeout_seconds)
                
                # Cache the result
                if self.cache and audio is not None:
                    try:
                        self.cache.put(text, voice, audio)
                    except Exception as e:
                        logger.warning(f"Failed to cache audio: {e}")
                
                yield audio
        
        except Exception as e:
            error = TTSError(
                error_type="synthesis",
                message="Speech synthesis failed. Audio output unavailable.",
                details=f"Unexpected synthesis error: {str(e)}",
                fallback_used=True,
                original_exception=e
            )
            self._handle_error(error)
            
            # Return fallback audio
            yield self._create_fallback_audio(text, voice)
    
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
    
    def _synthesize_streaming_safe(
        self,
        text: str,
        voice: VoiceConfig,
        timeout_seconds: Optional[float] = None
    ) -> Iterator[np.ndarray]:
        """
        Synthesize speech in streaming mode with error handling.
        
        Args:
            text: Input text
            voice: Voice configuration
            timeout_seconds: Timeout for each sentence
            
        Yields:
            Audio chunks for each sentence, with fallback on errors
            
        Requirements: 11.1, 11.4, 11.5
        """
        sentences = self._split_sentences(text)
        
        logger.debug(f"Safe streaming synthesis for {len(sentences)} sentences")
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            try:
                logger.debug(f"Synthesizing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
                
                # Apply timeout if specified
                if timeout_seconds:
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Synthesis timeout after {timeout_seconds}s")
                    
                    # Set timeout (Unix-like systems only)
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(timeout_seconds))
                        audio = self._synthesize_sentence(sentence, voice)
                        signal.alarm(0)  # Cancel alarm
                    except AttributeError:
                        # Windows doesn't support SIGALRM, use simple approach
                        audio = self._synthesize_sentence(sentence, voice)
                else:
                    audio = self._synthesize_sentence(sentence, voice)
                
                yield audio
                
            except TimeoutError as e:
                error = TTSError(
                    error_type="timeout",
                    message=f"Speech synthesis timed out for sentence {i+1}.",
                    details=f"Timeout after {timeout_seconds}s: {str(e)}",
                    fallback_used=True,
                    original_exception=e
                )
                self._handle_error(error)
                yield self._create_fallback_audio(sentence, voice)
                
            except Exception as e:
                error = TTSError(
                    error_type="synthesis",
                    message=f"Failed to synthesize sentence {i+1}.",
                    details=f"Sentence synthesis error: {str(e)}",
                    fallback_used=True,
                    original_exception=e
                )
                self._handle_error(error)
                yield self._create_fallback_audio(sentence, voice)
    
    def _synthesize_complete_safe(
        self,
        text: str,
        voice: VoiceConfig,
        timeout_seconds: Optional[float] = None
    ) -> np.ndarray:
        """
        Synthesize complete speech with error handling.
        
        Args:
            text: Input text
            voice: Voice configuration
            timeout_seconds: Timeout for synthesis
            
        Returns:
            Complete audio as numpy array, or fallback audio on error
            
        Requirements: 11.1, 11.4, 11.5
        """
        try:
            # Apply timeout if specified
            if timeout_seconds:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Synthesis timeout after {timeout_seconds}s")
                
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout_seconds))
                    audio = self._synthesize_complete(text, voice)
                    signal.alarm(0)
                    return audio
                except AttributeError:
                    # Windows doesn't support SIGALRM
                    return self._synthesize_complete(text, voice)
            else:
                return self._synthesize_complete(text, voice)
                
        except TimeoutError as e:
            error = TTSError(
                error_type="timeout",
                message="Speech synthesis timed out. Partial audio may be available.",
                details=f"Timeout after {timeout_seconds}s: {str(e)}",
                fallback_used=True,
                original_exception=e
            )
            self._handle_error(error)
            return self._create_fallback_audio(text, voice)
            
        except Exception as e:
            error = TTSError(
                error_type="synthesis",
                message="Speech synthesis failed. Audio output unavailable.",
                details=f"Complete synthesis error: {str(e)}",
                fallback_used=True,
                original_exception=e
            )
            self._handle_error(error)
            return self._create_fallback_audio(text, voice)
    
    def _create_fallback_audio(
        self,
        text: str,
        voice: VoiceConfig
    ) -> np.ndarray:
        """
        Create fallback audio (silence) when synthesis fails.
        
        Args:
            text: Input text (for duration estimation)
            voice: Voice configuration (for speed adjustment)
            
        Returns:
            Silence audio with appropriate duration
            
        Requirements: 11.1
        """
        # Estimate duration: ~150 chars/sec at normal speed
        base_duration_seconds = len(text) / 150.0
        duration_seconds = base_duration_seconds / voice.speed
        
        # Generate silence
        samples = int(duration_seconds * self.config.sample_rate)
        audio = np.zeros(samples, dtype=np.float32)
        
        logger.debug(f"Created fallback audio: {samples} samples, {duration_seconds:.2f}s")
        
        return audio
    
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
            
        Requirements: 1.1, 1.2, 1.3, 11.1, 11.5
        """
        try:
            if not self.piper_available:
                logger.warning("Piper not available, using fallback")
                return self._create_fallback_audio(text, voice)
            
            # Import piper
            from piper import PiperVoice
            
            # Load voice model if not already loaded
            if self.piper_voice is None:
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
                    error = TTSError(
                        error_type="backend_init",
                        message="Failed to load Piper voice model.",
                        details=f"Piper voice loading error: {str(e)}",
                        fallback_used=True,
                        original_exception=e
                    )
                    self._handle_error(error)
                    return self._create_fallback_audio(text, voice)
            
            # Synthesize with Piper
            from piper.config import SynthesisConfig
            syn_config = SynthesisConfig(
                speaker_id=None,
                length_scale=1.0 / voice.speed
            )
            
            # Piper yields chunks; we collect the float32 numpy arrays
            arrays = [chunk.audio_float_array for chunk in self.piper_voice.synthesize(text, syn_config=syn_config)]
            if not arrays:
                logger.warning("Piper returned no audio, using fallback")
                return self._create_fallback_audio(text, voice)
                
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
            
        except ImportError as e:
            error = TTSError(
                error_type="backend_init",
                message="Piper TTS library not installed.",
                details=f"Import error: {str(e)}. Install with: pip install piper-tts",
                fallback_used=True,
                original_exception=e
            )
            self._handle_error(error)
            return self._create_fallback_audio(text, voice)
            
        except Exception as e:
            error = TTSError(
                error_type="synthesis",
                message="Piper synthesis failed.",
                details=f"Piper synthesis error: {str(e)}",
                fallback_used=True,
                original_exception=e
            )
            self._handle_error(error)
            return self._create_fallback_audio(text, voice)
    
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
            
        Requirements: 11.1, 11.5
        """
        error = TTSError(
            error_type="backend_init",
            message="Coqui TTS backend not implemented.",
            details="Coqui TTS backend is not yet implemented",
            fallback_used=True
        )
        self._handle_error(error)
        return self._create_fallback_audio(text, voice)
    
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
            
        Requirements: 11.1, 11.5
        """
        error = TTSError(
            error_type="backend_init",
            message="Azure TTS backend not implemented.",
            details="Azure TTS backend is not yet implemented",
            fallback_used=True
        )
        self._handle_error(error)
        return self._create_fallback_audio(text, voice)
    
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
            
        Requirements: 11.1, 11.5
        """
        error = TTSError(
            error_type="backend_init",
            message="OpenAI TTS backend not implemented.",
            details="OpenAI TTS backend is not yet implemented",
            fallback_used=True
        )
        self._handle_error(error)
        return self._create_fallback_audio(text, voice)
    
    def _synthesize_mock(
        self,
        text: str,
        voice: VoiceConfig
    ) -> np.ndarray:
        """
        Mock synthesis for testing and fallback.
        
        Deprecated: Use _create_fallback_audio instead.
        
        Args:
            text: Input text
            voice: Voice configuration
            
        Returns:
            Silence audio as numpy array
        """
        logger.warning("_synthesize_mock is deprecated, use _create_fallback_audio")
        return self._create_fallback_audio(text, voice)

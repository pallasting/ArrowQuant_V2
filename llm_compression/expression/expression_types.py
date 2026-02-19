"""
Core data structures for the Expression & Presentation Layer.

This module defines the fundamental types used throughout the Expression Layer:
- Enums for modalities, styles, and backends
- Dataclasses for configuration and context
- Type hints for all structures

Requirements: 1.1, 2.1, 3.1
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============================================================================
# Enums
# ============================================================================


class OutputModality(Enum):
    """Output modality types."""
    TEXT = "text"
    SPEECH = "speech"
    VISUAL = "visual"
    MULTIMODAL = "multimodal"


class ExpressionStyle(Enum):
    """Expression style types."""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    EMPATHETIC = "empathetic"
    PLAYFUL = "playful"


class TTSBackend(Enum):
    """Text-to-Speech backend types."""
    COQUI = "coqui"
    AZURE = "azure"
    OPENAI = "openai"
    PIPER = "piper"  # Local, fast


class NLGBackend(Enum):
    """Natural Language Generation backend types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    TEMPLATE = "template"


# ============================================================================
# Context and Planning
# ============================================================================


@dataclass
class ExpressionContext:
    """
    Context for expression generation.
    
    Contains all contextual information needed to generate appropriate
    expressions including user state, conversation history, and preferences.
    
    Attributes:
        user_id: Unique user identifier
        conversation_history: List of previous conversation turns
        current_emotion: Current emotional state (neutral, joy, sadness, etc.)
        formality_level: Formality level (0=casual, 1=formal)
        time_of_day: Time context (day, night, morning, evening)
        language: ISO 639-1 language code (en, zh, ja, es, etc.)
        accent: Regional accent (e.g., "en-US", "en-GB", "zh-CN")
        user_preferences: User-specific preferences dictionary
    """
    user_id: str
    conversation_history: List[Dict[str, Any]]
    current_emotion: str = "neutral"
    formality_level: float = 0.5  # 0=casual, 1=formal
    time_of_day: str = "day"
    language: str = "en"  # ISO 639-1 code
    accent: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate context after initialization."""
        if not 0.0 <= self.formality_level <= 1.0:
            raise ValueError(f"formality_level must be between 0 and 1, got {self.formality_level}")
        
        if self.time_of_day not in ["day", "night", "morning", "evening"]:
            raise ValueError(f"Invalid time_of_day: {self.time_of_day}")


@dataclass
class ResponsePlan:
    """
    Plan for multi-modal response generation.
    
    Contains the strategy for generating a response including which modalities
    to use, what style to apply, and how to express emotions.
    
    Attributes:
        modalities: List of output modalities to use
        primary_modality: Primary output modality
        style: Expression style to apply
        emotion: Emotion to express
        emotion_intensity: Emotion intensity (0.0-1.0)
        use_template: Whether to use template-based generation
        template_name: Name of template if use_template is True
        streaming: Whether to use streaming output
        estimated_duration_ms: Estimated generation duration in milliseconds
    """
    modalities: List[OutputModality]
    primary_modality: OutputModality
    style: ExpressionStyle
    emotion: str
    emotion_intensity: float  # 0.0-1.0
    use_template: bool
    template_name: Optional[str]
    streaming: bool
    estimated_duration_ms: int
    
    def __post_init__(self):
        """Validate plan after initialization."""
        if not 0.0 <= self.emotion_intensity <= 1.0:
            raise ValueError(f"emotion_intensity must be between 0 and 1, got {self.emotion_intensity}")
        
        if self.primary_modality not in self.modalities:
            raise ValueError(f"primary_modality {self.primary_modality} not in modalities {self.modalities}")
        
        if self.use_template and not self.template_name:
            raise ValueError("template_name must be provided when use_template is True")


# ============================================================================
# Voice Configuration
# ============================================================================


@dataclass
class VoiceConfig:
    """
    Voice configuration for TTS.
    
    Defines all parameters for voice synthesis including voice selection,
    prosody control, and emotional expression.
    
    Attributes:
        voice_id: Voice identifier (backend-specific)
        language: ISO 639-1 language code
        speed: Speech speed (0.5-2.0, 1.0=normal)
        pitch: Voice pitch (0.5-2.0, 1.0=normal)
        volume: Voice volume (0.0-1.0)
        emotion: Emotion to express
        emotion_intensity: Emotion intensity (0.0-1.0)
        accent: Regional accent (e.g., "en-US", "en-GB", "zh-CN")
    """
    voice_id: str
    language: str = "en"
    speed: float = 1.0  # 0.5-2.0
    pitch: float = 1.0  # 0.5-2.0
    volume: float = 1.0  # 0.0-1.0
    emotion: str = "neutral"
    emotion_intensity: float = 0.5
    accent: Optional[str] = None
    
    def __post_init__(self):
        """Validate voice configuration after initialization."""
        if not 0.5 <= self.speed <= 2.0:
            raise ValueError(f"speed must be between 0.5 and 2.0, got {self.speed}")
        
        if not 0.5 <= self.pitch <= 2.0:
            raise ValueError(f"pitch must be between 0.5 and 2.0, got {self.pitch}")
        
        if not 0.0 <= self.volume <= 1.0:
            raise ValueError(f"volume must be between 0 and 1, got {self.volume}")
        
        if not 0.0 <= self.emotion_intensity <= 1.0:
            raise ValueError(f"emotion_intensity must be between 0 and 1, got {self.emotion_intensity}")


# ============================================================================
# NLG Configuration
# ============================================================================


@dataclass
class NLGConfig:
    """
    NLG engine configuration.
    
    Defines parameters for natural language generation including backend
    selection, model parameters, and generation settings.
    
    Attributes:
        backend: NLG backend to use
        model: Model name (backend-specific)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        streaming: Enable streaming output
        timeout_seconds: API timeout in seconds
    """
    backend: NLGBackend = NLGBackend.OPENAI
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 500
    streaming: bool = True
    timeout_seconds: int = 30
    
    def __post_init__(self):
        """Validate NLG configuration after initialization."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0 and 2, got {self.temperature}")
        
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")


# ============================================================================
# TTS Configuration
# ============================================================================


@dataclass
class TTSConfig:
    """
    TTS engine configuration.
    
    Defines parameters for text-to-speech synthesis including backend
    selection, voice configuration, and caching settings.
    
    Attributes:
        backend: TTS backend to use
        voice: Voice configuration
        sample_rate: Audio sample rate in Hz
        streaming: Enable streaming output
        cache_enabled: Enable output caching
    """
    backend: TTSBackend = TTSBackend.PIPER
    voice: Optional[VoiceConfig] = None
    sample_rate: int = 22050
    streaming: bool = True
    cache_enabled: bool = True
    
    def __post_init__(self):
        """Initialize default voice if not provided."""
        if self.voice is None:
            self.voice = VoiceConfig(voice_id="default")
        
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")


# ============================================================================
# Metrics and Monitoring
# ============================================================================


@dataclass
class ExpressionMetrics:
    """
    Metrics for expression quality monitoring.
    
    Used for tracking expression performance and quality for self-evolution
    and optimization.
    
    Attributes:
        backend_used: Backend identifier
        language: Language used
        latency_ms: Generation latency in milliseconds
        mos_score: Mean Opinion Score for quality (1.0-5.0)
        user_rating: Optional user rating (0.0-5.0)
        emotion_accuracy: Emotion expression accuracy (0.0-1.0)
        context_appropriateness: Context appropriateness score (0.0-1.0)
        timestamp: Timestamp of expression
    """
    backend_used: str
    language: str
    latency_ms: float
    mos_score: float
    user_rating: Optional[float]
    emotion_accuracy: float
    context_appropriateness: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        if not 1.0 <= self.mos_score <= 5.0:
            raise ValueError(f"mos_score must be between 1 and 5, got {self.mos_score}")
        
        if self.user_rating is not None and not 0.0 <= self.user_rating <= 5.0:
            raise ValueError(f"user_rating must be between 0 and 5, got {self.user_rating}")
        
        if not 0.0 <= self.emotion_accuracy <= 1.0:
            raise ValueError(f"emotion_accuracy must be between 0 and 1, got {self.emotion_accuracy}")
        
        if not 0.0 <= self.context_appropriateness <= 1.0:
            raise ValueError(f"context_appropriateness must be between 0 and 1, got {self.context_appropriateness}")

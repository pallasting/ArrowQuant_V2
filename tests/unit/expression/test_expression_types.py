"""
Unit tests for expression_types.py

Tests all core data structures including enums, dataclasses, and validation.
"""

import pytest
from datetime import datetime

from llm_compression.expression.expression_types import (
    OutputModality,
    ExpressionStyle,
    TTSBackend,
    NLGBackend,
    ExpressionContext,
    ResponsePlan,
    VoiceConfig,
    NLGConfig,
    TTSConfig,
    ExpressionMetrics,
)


class TestEnums:
    """Test enum definitions."""
    
    def test_output_modality_values(self):
        """Test OutputModality enum values."""
        assert OutputModality.TEXT.value == "text"
        assert OutputModality.SPEECH.value == "speech"
        assert OutputModality.VISUAL.value == "visual"
        assert OutputModality.MULTIMODAL.value == "multimodal"
    
    def test_expression_style_values(self):
        """Test ExpressionStyle enum values."""
        assert ExpressionStyle.FORMAL.value == "formal"
        assert ExpressionStyle.CASUAL.value == "casual"
        assert ExpressionStyle.TECHNICAL.value == "technical"
        assert ExpressionStyle.EMPATHETIC.value == "empathetic"
        assert ExpressionStyle.PLAYFUL.value == "playful"
    
    def test_tts_backend_values(self):
        """Test TTSBackend enum values."""
        assert TTSBackend.COQUI.value == "coqui"
        assert TTSBackend.AZURE.value == "azure"
        assert TTSBackend.OPENAI.value == "openai"
        assert TTSBackend.PIPER.value == "piper"
    
    def test_nlg_backend_values(self):
        """Test NLGBackend enum values."""
        assert NLGBackend.OPENAI.value == "openai"
        assert NLGBackend.ANTHROPIC.value == "anthropic"
        assert NLGBackend.LOCAL.value == "local"
        assert NLGBackend.TEMPLATE.value == "template"


class TestExpressionContext:
    """Test ExpressionContext dataclass."""
    
    def test_default_creation(self):
        """Test creating context with defaults."""
        ctx = ExpressionContext(
            user_id="user123",
            conversation_history=[]
        )
        
        assert ctx.user_id == "user123"
        assert ctx.conversation_history == []
        assert ctx.current_emotion == "neutral"
        assert ctx.formality_level == 0.5
        assert ctx.time_of_day == "day"
        assert ctx.language == "en"
        assert ctx.accent is None
        assert ctx.user_preferences == {}
    
    def test_custom_values(self):
        """Test creating context with custom values."""
        ctx = ExpressionContext(
            user_id="user456",
            conversation_history=[{"role": "user", "content": "hello"}],
            current_emotion="joy",
            formality_level=0.8,
            time_of_day="night",
            language="zh",
            accent="zh-CN",
            user_preferences={"voice": "female"}
        )
        
        assert ctx.user_id == "user456"
        assert len(ctx.conversation_history) == 1
        assert ctx.current_emotion == "joy"
        assert ctx.formality_level == 0.8
        assert ctx.time_of_day == "night"
        assert ctx.language == "zh"
        assert ctx.accent == "zh-CN"
        assert ctx.user_preferences["voice"] == "female"
    
    def test_formality_validation(self):
        """Test formality level validation."""
        with pytest.raises(ValueError, match="formality_level must be between 0 and 1"):
            ExpressionContext(
                user_id="user123",
                conversation_history=[],
                formality_level=1.5
            )
        
        with pytest.raises(ValueError, match="formality_level must be between 0 and 1"):
            ExpressionContext(
                user_id="user123",
                conversation_history=[],
                formality_level=-0.1
            )
    
    def test_time_of_day_validation(self):
        """Test time_of_day validation."""
        with pytest.raises(ValueError, match="Invalid time_of_day"):
            ExpressionContext(
                user_id="user123",
                conversation_history=[],
                time_of_day="invalid"
            )


class TestResponsePlan:
    """Test ResponsePlan dataclass."""
    
    def test_default_creation(self):
        """Test creating response plan."""
        plan = ResponsePlan(
            modalities=[OutputModality.TEXT, OutputModality.SPEECH],
            primary_modality=OutputModality.TEXT,
            style=ExpressionStyle.CASUAL,
            emotion="neutral",
            emotion_intensity=0.5,
            use_template=False,
            template_name=None,
            streaming=True,
            estimated_duration_ms=1000
        )
        
        assert len(plan.modalities) == 2
        assert plan.primary_modality == OutputModality.TEXT
        assert plan.style == ExpressionStyle.CASUAL
        assert plan.emotion == "neutral"
        assert plan.emotion_intensity == 0.5
        assert not plan.use_template
        assert plan.template_name is None
        assert plan.streaming
        assert plan.estimated_duration_ms == 1000
    
    def test_emotion_intensity_validation(self):
        """Test emotion intensity validation."""
        with pytest.raises(ValueError, match="emotion_intensity must be between 0 and 1"):
            ResponsePlan(
                modalities=[OutputModality.TEXT],
                primary_modality=OutputModality.TEXT,
                style=ExpressionStyle.CASUAL,
                emotion="joy",
                emotion_intensity=1.5,
                use_template=False,
                template_name=None,
                streaming=False,
                estimated_duration_ms=500
            )
    
    def test_primary_modality_validation(self):
        """Test primary modality must be in modalities list."""
        with pytest.raises(ValueError, match="primary_modality .* not in modalities"):
            ResponsePlan(
                modalities=[OutputModality.TEXT],
                primary_modality=OutputModality.SPEECH,
                style=ExpressionStyle.CASUAL,
                emotion="neutral",
                emotion_intensity=0.5,
                use_template=False,
                template_name=None,
                streaming=False,
                estimated_duration_ms=500
            )
    
    def test_template_validation(self):
        """Test template name required when use_template is True."""
        with pytest.raises(ValueError, match="template_name must be provided"):
            ResponsePlan(
                modalities=[OutputModality.TEXT],
                primary_modality=OutputModality.TEXT,
                style=ExpressionStyle.CASUAL,
                emotion="neutral",
                emotion_intensity=0.5,
                use_template=True,
                template_name=None,
                streaming=False,
                estimated_duration_ms=500
            )


class TestVoiceConfig:
    """Test VoiceConfig dataclass."""
    
    def test_default_creation(self):
        """Test creating voice config with defaults."""
        voice = VoiceConfig(voice_id="en_US-lessac-medium")
        
        assert voice.voice_id == "en_US-lessac-medium"
        assert voice.language == "en"
        assert voice.speed == 1.0
        assert voice.pitch == 1.0
        assert voice.volume == 1.0
        assert voice.emotion == "neutral"
        assert voice.emotion_intensity == 0.5
        assert voice.accent is None
    
    def test_custom_values(self):
        """Test creating voice config with custom values."""
        voice = VoiceConfig(
            voice_id="zh-CN-XiaoxiaoNeural",
            language="zh",
            speed=1.2,
            pitch=0.9,
            volume=0.8,
            emotion="joy",
            emotion_intensity=0.7,
            accent="zh-CN"
        )
        
        assert voice.voice_id == "zh-CN-XiaoxiaoNeural"
        assert voice.language == "zh"
        assert voice.speed == 1.2
        assert voice.pitch == 0.9
        assert voice.volume == 0.8
        assert voice.emotion == "joy"
        assert voice.emotion_intensity == 0.7
        assert voice.accent == "zh-CN"
    
    def test_speed_validation(self):
        """Test speed validation."""
        with pytest.raises(ValueError, match="speed must be between 0.5 and 2.0"):
            VoiceConfig(voice_id="test", speed=2.5)
        
        with pytest.raises(ValueError, match="speed must be between 0.5 and 2.0"):
            VoiceConfig(voice_id="test", speed=0.3)
    
    def test_pitch_validation(self):
        """Test pitch validation."""
        with pytest.raises(ValueError, match="pitch must be between 0.5 and 2.0"):
            VoiceConfig(voice_id="test", pitch=2.5)
    
    def test_volume_validation(self):
        """Test volume validation."""
        with pytest.raises(ValueError, match="volume must be between 0 and 1"):
            VoiceConfig(voice_id="test", volume=1.5)


class TestNLGConfig:
    """Test NLGConfig dataclass."""
    
    def test_default_creation(self):
        """Test creating NLG config with defaults."""
        config = NLGConfig()
        
        assert config.backend == NLGBackend.OPENAI
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 500
        assert config.streaming
        assert config.timeout_seconds == 30
    
    def test_custom_values(self):
        """Test creating NLG config with custom values."""
        config = NLGConfig(
            backend=NLGBackend.ANTHROPIC,
            model="claude-3",
            temperature=0.5,
            max_tokens=1000,
            streaming=False,
            timeout_seconds=60
        )
        
        assert config.backend == NLGBackend.ANTHROPIC
        assert config.model == "claude-3"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert not config.streaming
        assert config.timeout_seconds == 60
    
    def test_temperature_validation(self):
        """Test temperature validation."""
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            NLGConfig(temperature=2.5)
    
    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            NLGConfig(max_tokens=0)
    
    def test_timeout_validation(self):
        """Test timeout validation."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            NLGConfig(timeout_seconds=-1)


class TestTTSConfig:
    """Test TTSConfig dataclass."""
    
    def test_default_creation(self):
        """Test creating TTS config with defaults."""
        config = TTSConfig()
        
        assert config.backend == TTSBackend.PIPER
        assert config.voice is not None
        assert config.voice.voice_id == "default"
        assert config.sample_rate == 22050
        assert config.streaming
        assert config.cache_enabled
    
    def test_custom_values(self):
        """Test creating TTS config with custom values."""
        voice = VoiceConfig(voice_id="custom-voice")
        config = TTSConfig(
            backend=TTSBackend.AZURE,
            voice=voice,
            sample_rate=16000,
            streaming=False,
            cache_enabled=False
        )
        
        assert config.backend == TTSBackend.AZURE
        assert config.voice.voice_id == "custom-voice"
        assert config.sample_rate == 16000
        assert not config.streaming
        assert not config.cache_enabled
    
    def test_sample_rate_validation(self):
        """Test sample_rate validation."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            TTSConfig(sample_rate=0)


class TestExpressionMetrics:
    """Test ExpressionMetrics dataclass."""
    
    def test_default_creation(self):
        """Test creating expression metrics."""
        metrics = ExpressionMetrics(
            backend_used="piper",
            language="en",
            latency_ms=150.0,
            mos_score=4.2,
            user_rating=4.5,
            emotion_accuracy=0.85,
            context_appropriateness=0.90
        )
        
        assert metrics.backend_used == "piper"
        assert metrics.language == "en"
        assert metrics.latency_ms == 150.0
        assert metrics.mos_score == 4.2
        assert metrics.user_rating == 4.5
        assert metrics.emotion_accuracy == 0.85
        assert metrics.context_appropriateness == 0.90
        assert isinstance(metrics.timestamp, datetime)
    
    def test_mos_score_validation(self):
        """Test MOS score validation."""
        with pytest.raises(ValueError, match="mos_score must be between 1 and 5"):
            ExpressionMetrics(
                backend_used="test",
                language="en",
                latency_ms=100.0,
                mos_score=5.5,
                user_rating=None,
                emotion_accuracy=0.8,
                context_appropriateness=0.8
            )
    
    def test_user_rating_validation(self):
        """Test user rating validation."""
        with pytest.raises(ValueError, match="user_rating must be between 0 and 5"):
            ExpressionMetrics(
                backend_used="test",
                language="en",
                latency_ms=100.0,
                mos_score=4.0,
                user_rating=6.0,
                emotion_accuracy=0.8,
                context_appropriateness=0.8
            )
    
    def test_emotion_accuracy_validation(self):
        """Test emotion accuracy validation."""
        with pytest.raises(ValueError, match="emotion_accuracy must be between 0 and 1"):
            ExpressionMetrics(
                backend_used="test",
                language="en",
                latency_ms=100.0,
                mos_score=4.0,
                user_rating=None,
                emotion_accuracy=1.5,
                context_appropriateness=0.8
            )
    
    def test_context_appropriateness_validation(self):
        """Test context appropriateness validation."""
        with pytest.raises(ValueError, match="context_appropriateness must be between 0 and 1"):
            ExpressionMetrics(
                backend_used="test",
                language="en",
                latency_ms=100.0,
                mos_score=4.0,
                user_rating=None,
                emotion_accuracy=0.8,
                context_appropriateness=1.5
            )

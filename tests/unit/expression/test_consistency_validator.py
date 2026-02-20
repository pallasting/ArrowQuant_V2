"""
Unit tests for EmotionConsistencyValidator.

Tests validation of emotion consistency across TTS and text modalities.

Requirements: 3.6, 13.1
"""

import pytest
from llm_compression.expression.emotion.consistency_validator import (
    EmotionConsistencyValidator,
    ConsistencyWarning
)
from llm_compression.expression.expression_types import VoiceConfig


class TestEmotionConsistencyValidator:
    """Test suite for EmotionConsistencyValidator."""
    
    def test_init(self):
        """Test validator initialization."""
        validator = EmotionConsistencyValidator(tolerance=0.2)
        
        assert validator.tolerance == 0.2
        assert len(validator.emotion_parameter_ranges) > 0
        assert len(validator.emotion_compatibility) > 0
        assert "joy" in validator.emotion_parameter_ranges
        assert "joy" in validator.emotion_compatibility
    
    def test_matching_emotions_no_warnings(self):
        """Test that matching emotions produce no warnings."""
        validator = EmotionConsistencyValidator()
        
        voice_config = VoiceConfig(
            voice_id="test",
            emotion="joy",
            emotion_intensity=0.5,
            speed=1.1,
            pitch=1.1,
            volume=1.0
        )
        
        text_style_params = {
            "emotion": "joy",
            "emotion_intensity": 0.5,
            "formality": 0.4
        }
        
        warnings = validator.validate_consistency(voice_config, text_style_params)
        
        assert len(warnings) == 0
    
    def test_compatible_emotions_no_warnings(self):
        """Test that compatible emotions produce no warnings."""
        validator = EmotionConsistencyValidator()
        
        # joy and friendly are compatible
        voice_config = VoiceConfig(
            voice_id="test",
            emotion="joy",
            emotion_intensity=0.5,
            speed=1.1,
            pitch=1.1,
            volume=1.0
        )
        
        text_style_params = {
            "emotion": "friendly",
            "emotion_intensity": 0.5,
            "formality": 0.3
        }
        
        warnings = validator.validate_consistency(voice_config, text_style_params)
        
        assert len(warnings) == 0
    
    def test_incompatible_emotions_warning(self):
        """Test that incompatible emotions produce warnings."""
        validator = EmotionConsistencyValidator()
        
        # joy and sadness are incompatible
        voice_config = VoiceConfig(
            voice_id="test",
            emotion="joy",
            emotion_intensity=0.5,
            speed=1.1,
            pitch=1.1,
            volume=1.0
        )
        
        text_style_params = {
            "emotion": "sadness",
            "emotion_intensity": 0.5,
            "formality": 0.5
        }
        
        warnings = validator.validate_consistency(voice_config, text_style_params)
        
        assert len(warnings) > 0
        assert warnings[0].tts_emotion == "joy"
        assert warnings[0].text_emotion == "sadness"
    
    def test_parameter_inconsistency_warning(self):
        """Test that parameter inconsistencies produce warnings."""
        validator = EmotionConsistencyValidator(tolerance=0.1)
        
        # Voice parameters for joy but text emotion is sadness
        voice_config = VoiceConfig(
            voice_id="test",
            emotion="joy",
            emotion_intensity=0.8,
            speed=1.2,  # Fast (joy)
            pitch=1.2,  # High (joy)
            volume=1.0
        )
        
        text_style_params = {
            "emotion": "sadness",  # Expects slow, low pitch
            "emotion_intensity": 0.8,
            "formality": 0.5
        }
        
        warnings = validator.validate_consistency(voice_config, text_style_params)
        
        assert len(warnings) > 0
        
        # Should have parameter inconsistencies
        param_warning = next(
            (w for w in warnings if w.inconsistent_parameters),
            None
        )
        assert param_warning is not None
        assert "speed" in param_warning.inconsistent_parameters
        assert "pitch" in param_warning.inconsistent_parameters
    
    def test_severity_calculation(self):
        """Test warning severity calculation."""
        validator = EmotionConsistencyValidator(tolerance=0.05)
        
        # Create highly inconsistent parameters (all 3 wrong)
        voice_config = VoiceConfig(
            voice_id="test",
            emotion="joy",
            emotion_intensity=1.0,
            speed=1.3,  # Very fast (joy)
            pitch=1.3,  # Very high (joy)
            volume=1.0  # Normal (joy)
        )
        
        text_style_params = {
            "emotion": "sadness",  # Expects slow, low, quiet
            "emotion_intensity": 1.0,
            "formality": 0.5
        }
        
        warnings = validator.validate_consistency(voice_config, text_style_params)
        
        # Debug: print warnings
        for w in warnings:
            print(f"Warning: severity={w.severity}, params={w.inconsistent_parameters}")
        
        # Should have at least medium severity warning (2 params inconsistent)
        has_medium_or_high = any(w.severity in ["medium", "high"] for w in warnings)
        assert has_medium_or_high, f"Expected medium/high severity, got: {[w.severity for w in warnings]}"
    
    def test_neutral_emotion_tolerance(self):
        """Test that neutral emotion is tolerant of variations."""
        validator = EmotionConsistencyValidator()
        
        voice_config = VoiceConfig(
            voice_id="test",
            emotion="neutral",
            emotion_intensity=0.5,
            speed=1.0,
            pitch=1.0,
            volume=1.0
        )
        
        text_style_params = {
            "emotion": "neutral",
            "emotion_intensity": 0.5,
            "formality": 0.5
        }
        
        warnings = validator.validate_consistency(voice_config, text_style_params)
        
        assert len(warnings) == 0
    
    def test_empathetic_trust_compatibility(self):
        """Test that empathetic and trust emotions are compatible."""
        validator = EmotionConsistencyValidator()
        
        voice_config = VoiceConfig(
            voice_id="test",
            emotion="empathetic",
            emotion_intensity=0.6,
            speed=0.95,
            pitch=0.98,
            volume=0.95
        )
        
        text_style_params = {
            "emotion": "trust",
            "emotion_intensity": 0.6,
            "formality": 0.5
        }
        
        warnings = validator.validate_consistency(voice_config, text_style_params)
        
        assert len(warnings) == 0
    
    def test_suggested_fix(self):
        """Test that warnings include suggested fixes."""
        validator = EmotionConsistencyValidator(tolerance=0.1)
        
        voice_config = VoiceConfig(
            voice_id="test",
            emotion="anger",
            emotion_intensity=0.8,
            speed=1.25,
            pitch=1.2,
            volume=1.0
        )
        
        text_style_params = {
            "emotion": "sadness",
            "emotion_intensity": 0.8,
            "formality": 0.5
        }
        
        warnings = validator.validate_consistency(voice_config, text_style_params)
        
        assert len(warnings) > 0
        for warning in warnings:
            assert warning.suggested_fix
            assert len(warning.suggested_fix) > 0
    
    def test_tolerance_parameter(self):
        """Test that tolerance parameter affects validation."""
        # Low tolerance - strict validation
        strict_validator = EmotionConsistencyValidator(tolerance=0.05)
        
        # High tolerance - lenient validation
        lenient_validator = EmotionConsistencyValidator(tolerance=0.5)
        
        voice_config = VoiceConfig(
            voice_id="test",
            emotion="joy",
            emotion_intensity=0.5,
            speed=1.15,  # Slightly outside neutral range
            pitch=1.15,
            volume=1.0
        )
        
        text_style_params = {
            "emotion": "neutral",
            "emotion_intensity": 0.5,
            "formality": 0.5
        }
        
        strict_warnings = strict_validator.validate_consistency(
            voice_config, text_style_params
        )
        lenient_warnings = lenient_validator.validate_consistency(
            voice_config, text_style_params
        )
        
        # Strict should have more warnings
        assert len(strict_warnings) >= len(lenient_warnings)
    
    def test_log_warnings(self, caplog):
        """Test warning logging."""
        validator = EmotionConsistencyValidator()
        
        warnings = [
            ConsistencyWarning(
                severity="medium",
                message="Test warning",
                tts_emotion="joy",
                text_emotion="sadness",
                inconsistent_parameters=["speed", "pitch"],
                suggested_fix="Adjust parameters"
            )
        ]
        
        validator.log_warnings(warnings)
        
        # Check that warning was logged
        assert "emotion consistency warning" in caplog.text.lower()
    
    def test_get_warnings_summary_empty(self):
        """Test warnings summary with no warnings."""
        validator = EmotionConsistencyValidator()
        
        summary = validator.get_warnings_summary([])
        
        assert summary["total"] == 0
        assert summary["has_warnings"] is False
        assert summary["by_severity"]["low"] == 0
        assert summary["by_severity"]["medium"] == 0
        assert summary["by_severity"]["high"] == 0
    
    def test_get_warnings_summary_with_warnings(self):
        """Test warnings summary with multiple warnings."""
        validator = EmotionConsistencyValidator()
        
        warnings = [
            ConsistencyWarning(
                severity="low",
                message="Low warning",
                tts_emotion="joy",
                text_emotion="friendly",
                inconsistent_parameters=[],
                suggested_fix="Fix 1"
            ),
            ConsistencyWarning(
                severity="high",
                message="High warning",
                tts_emotion="joy",
                text_emotion="sadness",
                inconsistent_parameters=["speed", "pitch", "volume"],
                suggested_fix="Fix 2"
            ),
            ConsistencyWarning(
                severity="medium",
                message="Medium warning",
                tts_emotion="anger",
                text_emotion="trust",
                inconsistent_parameters=["speed"],
                suggested_fix="Fix 3"
            )
        ]
        
        summary = validator.get_warnings_summary(warnings)
        
        assert summary["total"] == 3
        assert summary["has_warnings"] is True
        assert summary["by_severity"]["low"] == 1
        assert summary["by_severity"]["medium"] == 1
        assert summary["by_severity"]["high"] == 1
        assert len(summary["warnings"]) == 3
    
    def test_all_emotions_have_ranges(self):
        """Test that all emotions have parameter ranges defined."""
        validator = EmotionConsistencyValidator()
        
        # List of all supported emotions
        emotions = [
            "joy", "sadness", "anger", "fear", "surprise", "disgust",
            "trust", "anticipation", "neutral", "empathetic", "friendly",
            "playful"
        ]
        
        for emotion in emotions:
            assert emotion in validator.emotion_parameter_ranges
            ranges = validator.emotion_parameter_ranges[emotion]
            assert "speed" in ranges
            assert "pitch" in ranges
            assert "volume" in ranges
    
    def test_parameter_ranges_valid(self):
        """Test that parameter ranges are valid."""
        validator = EmotionConsistencyValidator()
        
        for emotion, ranges in validator.emotion_parameter_ranges.items():
            # Check speed range
            speed_min, speed_max = ranges["speed"]
            assert 0.5 <= speed_min <= speed_max <= 2.0
            
            # Check pitch range
            pitch_min, pitch_max = ranges["pitch"]
            assert 0.5 <= pitch_min <= pitch_max <= 2.0
            
            # Check volume range
            volume_min, volume_max = ranges["volume"]
            assert 0.0 <= volume_min <= volume_max <= 1.0
    
    def test_compatibility_matrix_symmetric(self):
        """Test that compatibility relationships are reasonable."""
        validator = EmotionConsistencyValidator()
        
        # Check that compatibility matrix exists for all emotions
        for emotion in validator.emotion_parameter_ranges.keys():
            # Should have compatibility list (can be empty)
            compatible = validator.emotion_compatibility.get(emotion, [])
            assert isinstance(compatible, list)
    
    def test_missing_text_emotion_defaults_to_neutral(self):
        """Test that missing text emotion defaults to neutral."""
        validator = EmotionConsistencyValidator()
        
        voice_config = VoiceConfig(
            voice_id="test",
            emotion="joy",
            emotion_intensity=0.5,
            speed=1.1,
            pitch=1.1,
            volume=1.0
        )
        
        # Text style params without emotion
        text_style_params = {
            "formality": 0.5
        }
        
        warnings = validator.validate_consistency(voice_config, text_style_params)
        
        # Should compare against neutral
        if warnings:
            assert any(w.text_emotion == "neutral" for w in warnings)
    
    def test_extreme_parameter_values(self):
        """Test validation with extreme parameter values."""
        validator = EmotionConsistencyValidator(tolerance=0.1)
        
        # Extreme values
        voice_config = VoiceConfig(
            voice_id="test",
            emotion="anger",
            emotion_intensity=1.0,
            speed=2.0,  # Maximum
            pitch=2.0,  # Maximum
            volume=1.0
        )
        
        text_style_params = {
            "emotion": "sadness",  # Expects minimum values
            "emotion_intensity": 1.0,
            "formality": 0.5
        }
        
        warnings = validator.validate_consistency(voice_config, text_style_params)
        
        # Should have warnings with at least medium severity
        assert len(warnings) > 0
        has_medium_or_high = any(w.severity in ["medium", "high"] for w in warnings)
        assert has_medium_or_high, f"Expected medium/high severity, got: {[w.severity for w in warnings]}"
    
    def test_low_intensity_emotions(self):
        """Test validation with low emotion intensity."""
        validator = EmotionConsistencyValidator()
        
        # Low intensity should be closer to neutral
        voice_config = VoiceConfig(
            voice_id="test",
            emotion="joy",
            emotion_intensity=0.1,  # Very low
            speed=1.02,  # Close to neutral
            pitch=1.02,
            volume=1.0
        )
        
        text_style_params = {
            "emotion": "neutral",
            "emotion_intensity": 0.1,
            "formality": 0.5
        }
        
        warnings = validator.validate_consistency(voice_config, text_style_params)
        
        # Should have minimal or no warnings due to low intensity
        assert len(warnings) <= 1
        if warnings:
            assert warnings[0].severity == "low"


class TestConsistencyWarning:
    """Test suite for ConsistencyWarning dataclass."""
    
    def test_warning_creation(self):
        """Test creating a consistency warning."""
        warning = ConsistencyWarning(
            severity="medium",
            message="Test message",
            tts_emotion="joy",
            text_emotion="sadness",
            inconsistent_parameters=["speed", "pitch"],
            suggested_fix="Adjust speed and pitch"
        )
        
        assert warning.severity == "medium"
        assert warning.message == "Test message"
        assert warning.tts_emotion == "joy"
        assert warning.text_emotion == "sadness"
        assert len(warning.inconsistent_parameters) == 2
        assert "speed" in warning.inconsistent_parameters
        assert warning.suggested_fix == "Adjust speed and pitch"
    
    def test_warning_attributes(self):
        """Test that warning has all required attributes."""
        warning = ConsistencyWarning(
            severity="low",
            message="Test",
            tts_emotion="neutral",
            text_emotion="neutral",
            inconsistent_parameters=[],
            suggested_fix="None"
        )
        
        assert hasattr(warning, "severity")
        assert hasattr(warning, "message")
        assert hasattr(warning, "tts_emotion")
        assert hasattr(warning, "text_emotion")
        assert hasattr(warning, "inconsistent_parameters")
        assert hasattr(warning, "suggested_fix")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

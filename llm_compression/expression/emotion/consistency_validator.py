"""
Emotion Consistency Validator for multi-modal expression.

This module implements validation to ensure emotion consistency across
different output modalities (TTS voice parameters and text style).

Requirements: 3.6
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from llm_compression.expression.expression_types import VoiceConfig

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyWarning:
    """
    Warning about emotion inconsistency.
    
    Attributes:
        severity: Warning severity ("low", "medium", "high")
        message: Human-readable warning message
        tts_emotion: Emotion expressed in TTS
        text_emotion: Emotion expressed in text style
        inconsistent_parameters: List of inconsistent parameters
        suggested_fix: Suggested fix for the inconsistency
    """
    severity: str
    message: str
    tts_emotion: str
    text_emotion: str
    inconsistent_parameters: List[str]
    suggested_fix: str


class EmotionConsistencyValidator:
    """
    Validates emotion consistency across modalities.
    
    Ensures that emotions expressed through TTS voice parameters (speed,
    pitch, volume) are consistent with emotions expressed through text
    generation style (formality, tone, word choice).
    
    Requirements: 3.6
    """
    
    def __init__(self, tolerance: float = 0.3):
        """
        Initialize consistency validator.
        
        Args:
            tolerance: Tolerance for parameter differences (0.0-1.0)
                      Higher values allow more variation before warning
        """
        self.tolerance = tolerance
        
        # Expected parameter ranges for each emotion
        # Format: {emotion: {"speed": (min, max), "pitch": (min, max), ...}}
        self.emotion_parameter_ranges = self._init_parameter_ranges()
        
        # Emotion compatibility matrix
        # Some emotions are compatible (e.g., joy and friendly)
        self.emotion_compatibility = self._init_compatibility_matrix()
        
        logger.info(
            f"Initialized emotion consistency validator "
            f"(tolerance={tolerance:.2f})"
        )
    
    def _init_parameter_ranges(self) -> Dict[str, Dict[str, tuple]]:
        """
        Initialize expected parameter ranges for each emotion.
        
        These ranges define what voice parameters are expected for each
        emotion based on the EmotionMapper's mappings.
        
        Returns:
            Dictionary mapping emotions to parameter ranges
        """
        return {
            "joy": {
                "speed": (1.05, 1.15),
                "pitch": (1.05, 1.15),
                "volume": (0.95, 1.0)
            },
            "sadness": {
                "speed": (0.85, 0.95),
                "pitch": (0.85, 0.95),
                "volume": (0.85, 0.95)
            },
            "anger": {
                "speed": (1.15, 1.25),
                "pitch": (1.10, 1.20),
                "volume": (0.95, 1.0)
            },
            "fear": {
                "speed": (1.10, 1.20),
                "pitch": (1.15, 1.25),
                "volume": (0.90, 1.0)
            },
            "surprise": {
                "speed": (1.05, 1.15),
                "pitch": (1.10, 1.20),
                "volume": (0.95, 1.0)
            },
            "disgust": {
                "speed": (0.90, 1.0),
                "pitch": (0.90, 1.0),
                "volume": (0.90, 1.0)
            },
            "trust": {
                "speed": (0.95, 1.05),
                "pitch": (0.95, 1.05),
                "volume": (0.95, 1.0)
            },
            "anticipation": {
                "speed": (1.00, 1.10),
                "pitch": (1.00, 1.10),
                "volume": (0.95, 1.0)
            },
            "neutral": {
                "speed": (0.95, 1.05),
                "pitch": (0.95, 1.05),
                "volume": (0.95, 1.0)
            },
            "empathetic": {
                "speed": (0.90, 1.0),
                "pitch": (0.93, 1.03),
                "volume": (0.90, 1.0)
            },
            "friendly": {
                "speed": (1.00, 1.10),
                "pitch": (1.00, 1.10),
                "volume": (0.95, 1.0)
            },
            "playful": {
                "speed": (1.05, 1.15),
                "pitch": (1.05, 1.15),
                "volume": (0.95, 1.0)
            },
        }
    
    def _init_compatibility_matrix(self) -> Dict[str, List[str]]:
        """
        Initialize emotion compatibility matrix.
        
        Some emotions are compatible and don't require warnings when
        mixed (e.g., joy and friendly, empathetic and trust).
        
        Returns:
            Dictionary mapping emotions to compatible emotions
        """
        return {
            "joy": ["friendly", "playful", "anticipation", "surprise"],
            "sadness": ["empathetic", "trust"],
            "anger": ["disgust"],
            "fear": ["surprise", "anticipation"],
            "surprise": ["joy", "fear", "anticipation"],
            "disgust": ["anger"],
            "trust": ["empathetic", "neutral", "anticipation"],
            "anticipation": ["joy", "friendly", "trust", "surprise"],
            "neutral": ["trust", "empathetic"],
            "empathetic": ["sadness", "trust", "neutral"],
            "friendly": ["joy", "playful", "anticipation"],
            "playful": ["joy", "friendly", "surprise"],
        }
    
    def validate_consistency(
        self,
        voice_config: VoiceConfig,
        text_style_params: Dict[str, Any]
    ) -> List[ConsistencyWarning]:
        """
        Validate emotion consistency between TTS and text style.
        
        Checks if the emotion expressed through voice parameters matches
        the emotion expressed through text style parameters.
        
        Args:
            voice_config: Voice configuration with emotion settings
            text_style_params: Text style parameters with emotion settings
            
        Returns:
            List of consistency warnings (empty if consistent)
        """
        warnings = []
        
        # Extract emotions
        tts_emotion = voice_config.emotion
        text_emotion = text_style_params.get("emotion", "neutral")
        
        logger.debug(
            f"Validating consistency: TTS={tts_emotion}, Text={text_emotion}"
        )
        
        # Check if emotions match exactly
        if tts_emotion == text_emotion:
            logger.debug("Emotions match exactly - consistent")
            return warnings
        
        # Check if emotions are compatible
        if self._are_emotions_compatible(tts_emotion, text_emotion):
            logger.debug(
                f"Emotions are compatible: {tts_emotion} <-> {text_emotion}"
            )
            return warnings
        
        # Emotions don't match - check parameter consistency
        parameter_warnings = self._validate_parameters(
            voice_config,
            text_style_params,
            tts_emotion,
            text_emotion
        )
        
        if parameter_warnings:
            warnings.extend(parameter_warnings)
        
        # Add general inconsistency warning
        if not warnings:
            # Emotions don't match but parameters are close enough
            warnings.append(ConsistencyWarning(
                severity="low",
                message=(
                    f"Emotion mismatch: TTS uses '{tts_emotion}' but text "
                    f"style uses '{text_emotion}'. Consider using the same "
                    f"emotion for both modalities."
                ),
                tts_emotion=tts_emotion,
                text_emotion=text_emotion,
                inconsistent_parameters=[],
                suggested_fix=(
                    f"Set both emotions to '{tts_emotion}' or '{text_emotion}' "
                    f"for better consistency."
                )
            ))
        
        return warnings
    
    def _are_emotions_compatible(
        self,
        emotion1: str,
        emotion2: str
    ) -> bool:
        """
        Check if two emotions are compatible.
        
        Args:
            emotion1: First emotion
            emotion2: Second emotion
            
        Returns:
            True if emotions are compatible
        """
        # Check both directions
        compatible_with_1 = self.emotion_compatibility.get(emotion1, [])
        compatible_with_2 = self.emotion_compatibility.get(emotion2, [])
        
        return (
            emotion2 in compatible_with_1 or
            emotion1 in compatible_with_2
        )
    
    def _validate_parameters(
        self,
        voice_config: VoiceConfig,
        text_style_params: Dict[str, Any],
        tts_emotion: str,
        text_emotion: str
    ) -> List[ConsistencyWarning]:
        """
        Validate parameter consistency between TTS and text style.
        
        Checks if voice parameters (speed, pitch) are consistent with
        the expected parameters for the text emotion.
        
        Args:
            voice_config: Voice configuration
            text_style_params: Text style parameters
            tts_emotion: TTS emotion
            text_emotion: Text emotion
            
        Returns:
            List of parameter inconsistency warnings
        """
        warnings = []
        
        # Get expected ranges for text emotion
        expected_ranges = self.emotion_parameter_ranges.get(
            text_emotion,
            self.emotion_parameter_ranges["neutral"]
        )
        
        # Check each parameter
        inconsistent_params = []
        
        # Check speed
        speed = voice_config.speed
        speed_range = expected_ranges["speed"]
        if not self._is_in_range(speed, speed_range):
            inconsistent_params.append("speed")
            logger.debug(
                f"Speed inconsistency: {speed:.2f} not in "
                f"expected range {speed_range} for {text_emotion}"
            )
        
        # Check pitch
        pitch = voice_config.pitch
        pitch_range = expected_ranges["pitch"]
        if not self._is_in_range(pitch, pitch_range):
            inconsistent_params.append("pitch")
            logger.debug(
                f"Pitch inconsistency: {pitch:.2f} not in "
                f"expected range {pitch_range} for {text_emotion}"
            )
        
        # Check volume
        volume = voice_config.volume
        volume_range = expected_ranges["volume"]
        if not self._is_in_range(volume, volume_range):
            inconsistent_params.append("volume")
            logger.debug(
                f"Volume inconsistency: {volume:.2f} not in "
                f"expected range {volume_range} for {text_emotion}"
            )
        
        # Create warning if parameters are inconsistent
        if inconsistent_params:
            severity = self._calculate_severity(len(inconsistent_params))
            
            warnings.append(ConsistencyWarning(
                severity=severity,
                message=(
                    f"Voice parameters ({', '.join(inconsistent_params)}) "
                    f"for emotion '{tts_emotion}' are inconsistent with "
                    f"text style emotion '{text_emotion}'. This may create "
                    f"a confusing user experience."
                ),
                tts_emotion=tts_emotion,
                text_emotion=text_emotion,
                inconsistent_parameters=inconsistent_params,
                suggested_fix=self._suggest_fix(
                    inconsistent_params,
                    voice_config,
                    expected_ranges
                )
            ))
        
        return warnings
    
    def _is_in_range(
        self,
        value: float,
        range_tuple: tuple,
        tolerance: Optional[float] = None
    ) -> bool:
        """
        Check if value is within range (with tolerance).
        
        Args:
            value: Value to check
            range_tuple: (min, max) tuple
            tolerance: Optional tolerance override
            
        Returns:
            True if value is in range
        """
        tolerance = tolerance if tolerance is not None else self.tolerance
        min_val, max_val = range_tuple
        
        # Expand range by tolerance
        min_val -= tolerance
        max_val += tolerance
        
        return min_val <= value <= max_val
    
    def _calculate_severity(self, num_inconsistent: int) -> str:
        """
        Calculate warning severity based on number of inconsistent parameters.
        
        Args:
            num_inconsistent: Number of inconsistent parameters
            
        Returns:
            Severity level ("low", "medium", "high")
        """
        if num_inconsistent >= 3:
            return "high"
        elif num_inconsistent == 2:
            return "medium"
        else:
            return "low"
    
    def _suggest_fix(
        self,
        inconsistent_params: List[str],
        voice_config: VoiceConfig,
        expected_ranges: Dict[str, tuple]
    ) -> str:
        """
        Suggest fix for parameter inconsistencies.
        
        Args:
            inconsistent_params: List of inconsistent parameter names
            voice_config: Current voice configuration
            expected_ranges: Expected parameter ranges
            
        Returns:
            Suggested fix description
        """
        suggestions = []
        
        for param in inconsistent_params:
            current_value = getattr(voice_config, param)
            expected_range = expected_ranges[param]
            target_value = sum(expected_range) / 2  # Midpoint
            
            suggestions.append(
                f"{param}: {current_value:.2f} → {target_value:.2f}"
            )
        
        return (
            f"Adjust voice parameters to match text emotion: "
            f"{', '.join(suggestions)}"
        )
    
    def log_warnings(self, warnings: List[ConsistencyWarning]):
        """
        Log consistency warnings.
        
        Args:
            warnings: List of warnings to log
        """
        if not warnings:
            logger.info("No emotion consistency warnings")
            return
        
        logger.warning(
            f"Found {len(warnings)} emotion consistency warning(s)"
        )
        
        for warning in warnings:
            log_func = {
                "low": logger.info,
                "medium": logger.warning,
                "high": logger.error
            }.get(warning.severity, logger.warning)
            
            log_func(
                f"[{warning.severity.upper()}] {warning.message}\n"
                f"  TTS Emotion: {warning.tts_emotion}\n"
                f"  Text Emotion: {warning.text_emotion}\n"
                f"  Inconsistent: {', '.join(warning.inconsistent_parameters)}\n"
                f"  Suggested Fix: {warning.suggested_fix}"
            )
    
    def get_warnings_summary(
        self,
        warnings: List[ConsistencyWarning]
    ) -> Dict[str, Any]:
        """
        Get summary of consistency warnings.
        
        Args:
            warnings: List of warnings
            
        Returns:
            Summary dictionary with counts and details
        """
        if not warnings:
            return {
                "total": 0,
                "by_severity": {"low": 0, "medium": 0, "high": 0},
                "has_warnings": False
            }
        
        severity_counts = {"low": 0, "medium": 0, "high": 0}
        for warning in warnings:
            severity_counts[warning.severity] += 1
        
        return {
            "total": len(warnings),
            "by_severity": severity_counts,
            "has_warnings": True,
            "warnings": [
                {
                    "severity": w.severity,
                    "message": w.message,
                    "tts_emotion": w.tts_emotion,
                    "text_emotion": w.text_emotion,
                    "inconsistent_parameters": w.inconsistent_parameters,
                    "suggested_fix": w.suggested_fix
                }
                for w in warnings
            ]
        }

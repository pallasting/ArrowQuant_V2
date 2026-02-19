"""
Language detection and management for multi-language expression support.

This module provides automatic language detection and language-specific
expression rules for the Expression & Presentation Layer.

Requirements: 15.1, 15.2, 15.3, 15.5
"""

import logging
from typing import Dict, Any
from collections import Counter

from llm_compression.expression.expression_types import ExpressionContext

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detects and manages language context for expression.
    
    Responsibilities:
    - Detect input language automatically
    - Maintain language consistency across conversation
    - Provide language-specific expression rules
    - Support code-switching (multilingual conversations)
    
    Supports 4+ languages: English (en), Chinese (zh), Japanese (ja), Spanish (es)
    
    Requirements: 15.1, 15.2, 15.3, 15.5
    """
    
    def __init__(self):
        """Initialize language detector with default rules."""
        self.language_rules = self._init_language_rules()
        self.current_language = "en"
        self.language_history: list[str] = []
    
    def _init_language_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize language-specific expression rules.
        
        Returns:
            Dictionary mapping language codes to rule dictionaries
            
        Requirements: 15.2, 15.3
        """
        return {
            "en": {
                "formality_default": 0.5,
                "emotion_intensity_multiplier": 1.0,
                "preferred_tts_backend": "piper",
                "voice_id": "en_US-lessac-medium",
            },
            "zh": {
                "formality_default": 0.7,  # Chinese tends to be more formal
                "emotion_intensity_multiplier": 0.8,  # More subtle emotions
                "preferred_tts_backend": "azure",  # Better Chinese support
                "voice_id": "zh-CN-XiaoxiaoNeural",
            },
            "ja": {
                "formality_default": 0.8,  # Japanese is very formal
                "emotion_intensity_multiplier": 0.7,
                "preferred_tts_backend": "azure",
                "voice_id": "ja-JP-NanamiNeural",
            },
            "es": {
                "formality_default": 0.6,
                "emotion_intensity_multiplier": 1.2,  # More expressive
                "preferred_tts_backend": "piper",
                "voice_id": "es_ES-mls_10246-low",
            },
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect language from input text.
        
        Uses simple heuristic-based detection for supported languages.
        In production, consider using langdetect or fasttext for better accuracy.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ISO 639-1 language code (en, zh, ja, es)
            
        Requirements: 15.1
        """
        if not text or not text.strip():
            return self.current_language  # Return current if empty
        
        # Check for Chinese characters (CJK Unified Ideographs)
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return "zh"
        
        # Check for Japanese characters (Hiragana or Katakana)
        if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            return "ja"
        
        # Check for Spanish patterns (accented characters and punctuation)
        spanish_chars = set('áéíóúñ¿¡')
        if any(char in spanish_chars for char in text.lower()):
            return "es"
        
        # Default to English
        return "en"
    
    def update_context(
        self,
        text: str,
        context: ExpressionContext
    ) -> ExpressionContext:
        """
        Update expression context with language information.
        
        Detects language from text, maintains language consistency through
        history tracking (majority vote), and applies language-specific rules.
        
        Args:
            text: Input text
            context: Current expression context
            
        Returns:
            Updated context with language information
            
        Requirements: 15.1, 15.2, 15.3, 15.5
        """
        # Detect language
        detected_lang = self.detect_language(text)
        
        # Update language history (keep last 10 detections)
        self.language_history.append(detected_lang)
        if len(self.language_history) > 10:
            self.language_history.pop(0)
        
        # Determine current language (majority vote from recent history)
        # This provides language consistency across conversation
        lang_counts = Counter(self.language_history)
        self.current_language = lang_counts.most_common(1)[0][0]
        
        # Update context
        context.language = self.current_language
        
        # Apply language-specific rules
        rules = self.language_rules.get(self.current_language, self.language_rules["en"])
        
        # Adjust formality if not explicitly set (still at default value)
        if context.formality_level == 0.5:  # Default value
            context.formality_level = rules["formality_default"]
        
        logger.info(
            f"Language detected: {self.current_language}, "
            f"formality: {context.formality_level}"
        )
        
        return context
    
    def get_preferred_tts_backend(self, language: str) -> str:
        """
        Get preferred TTS backend for language.
        
        Args:
            language: ISO 639-1 language code
            
        Returns:
            Preferred TTS backend name
            
        Requirements: 15.4
        """
        rules = self.language_rules.get(language, self.language_rules["en"])
        return rules["preferred_tts_backend"]
    
    def get_default_voice(self, language: str) -> str:
        """
        Get default voice ID for language.
        
        Args:
            language: ISO 639-1 language code
            
        Returns:
            Default voice ID for the language
            
        Requirements: 15.4
        """
        rules = self.language_rules.get(language, self.language_rules["en"])
        return rules["voice_id"]
    
    def adjust_emotion_intensity(
        self,
        emotion_intensity: float,
        language: str
    ) -> float:
        """
        Adjust emotion intensity based on language culture.
        
        Different cultures express emotions with different intensities.
        This method applies language-specific multipliers to emotion intensity.
        
        Args:
            emotion_intensity: Base emotion intensity (0.0-1.0)
            language: ISO 639-1 language code
            
        Returns:
            Adjusted emotion intensity (0.0-1.0)
            
        Requirements: 15.3
        """
        rules = self.language_rules.get(language, self.language_rules["en"])
        multiplier = rules["emotion_intensity_multiplier"]
        return min(1.0, emotion_intensity * multiplier)
    
    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of ISO 639-1 language codes
        """
        return list(self.language_rules.keys())
    
    def is_language_supported(self, language: str) -> bool:
        """
        Check if a language is supported.
        
        Args:
            language: ISO 639-1 language code
            
        Returns:
            True if language is supported, False otherwise
        """
        return language in self.language_rules

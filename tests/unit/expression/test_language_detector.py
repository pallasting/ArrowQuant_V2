"""
Unit tests for LanguageDetector class.

Tests language detection, language-specific rule application, and
language consistency across conversations.

Requirements: 15.1, 15.2, 15.3, 15.5
"""

import pytest
from llm_compression.expression.language import LanguageDetector
from llm_compression.expression.expression_types import ExpressionContext


class TestLanguageDetector:
    """Test suite for LanguageDetector class."""
    
    def test_init(self):
        """Test LanguageDetector initialization."""
        detector = LanguageDetector()
        
        assert detector.current_language == "en"
        assert detector.language_history == []
        assert len(detector.language_rules) >= 4  # en, zh, ja, es
        assert "en" in detector.language_rules
        assert "zh" in detector.language_rules
        assert "ja" in detector.language_rules
        assert "es" in detector.language_rules
    
    def test_detect_language_english(self):
        """Test English language detection."""
        detector = LanguageDetector()
        
        # Simple English text
        assert detector.detect_language("Hello world") == "en"
        assert detector.detect_language("This is a test") == "en"
        assert detector.detect_language("How are you?") == "en"
    
    def test_detect_language_chinese(self):
        """Test Chinese language detection."""
        detector = LanguageDetector()
        
        # Chinese characters
        assert detector.detect_language("你好世界") == "zh"
        assert detector.detect_language("这是一个测试") == "zh"
        assert detector.detect_language("中文") == "zh"
    
    def test_detect_language_japanese(self):
        """Test Japanese language detection."""
        detector = LanguageDetector()
        
        # Japanese Hiragana
        assert detector.detect_language("こんにちは") == "ja"
        # Japanese Katakana
        assert detector.detect_language("カタカナ") == "ja"
        # Mixed
        assert detector.detect_language("こんにちはカタカナ") == "ja"
    
    def test_detect_language_spanish(self):
        """Test Spanish language detection."""
        detector = LanguageDetector()
        
        # Spanish with accented characters
        assert detector.detect_language("Hola, ¿cómo estás?") == "es"
        assert detector.detect_language("Mañana") == "es"
        assert detector.detect_language("¡Hola!") == "es"
        assert detector.detect_language("Español") == "es"
    
    def test_detect_language_empty_text(self):
        """Test language detection with empty text."""
        detector = LanguageDetector()
        detector.current_language = "zh"
        
        # Empty text should return current language
        assert detector.detect_language("") == "zh"
        assert detector.detect_language("   ") == "zh"
    
    def test_detect_language_mixed_content(self):
        """Test language detection with mixed content."""
        detector = LanguageDetector()
        
        # Chinese characters take precedence
        assert detector.detect_language("Hello 你好") == "zh"
        
        # Japanese characters take precedence
        assert detector.detect_language("Hello こんにちは") == "ja"
        
        # Spanish characters detected
        assert detector.detect_language("Hello señor") == "es"
    
    def test_language_rules_structure(self):
        """Test language rules have correct structure."""
        detector = LanguageDetector()
        
        for lang_code in ["en", "zh", "ja", "es"]:
            rules = detector.language_rules[lang_code]
            
            # Check required keys
            assert "formality_default" in rules
            assert "emotion_intensity_multiplier" in rules
            assert "preferred_tts_backend" in rules
            assert "voice_id" in rules
            
            # Check value ranges
            assert 0.0 <= rules["formality_default"] <= 1.0
            assert rules["emotion_intensity_multiplier"] > 0.0
            assert isinstance(rules["preferred_tts_backend"], str)
            assert isinstance(rules["voice_id"], str)
    
    def test_language_rules_cultural_differences(self):
        """Test language rules reflect cultural differences."""
        detector = LanguageDetector()
        
        # Japanese should be more formal than English
        assert detector.language_rules["ja"]["formality_default"] > \
               detector.language_rules["en"]["formality_default"]
        
        # Spanish should be more emotionally expressive
        assert detector.language_rules["es"]["emotion_intensity_multiplier"] > \
               detector.language_rules["en"]["emotion_intensity_multiplier"]
        
        # Chinese and Japanese should have more subtle emotions
        assert detector.language_rules["zh"]["emotion_intensity_multiplier"] < 1.0
        assert detector.language_rules["ja"]["emotion_intensity_multiplier"] < 1.0
    
    def test_update_context_basic(self):
        """Test basic context update with language detection."""
        detector = LanguageDetector()
        context = ExpressionContext(
            user_id="test_user",
            conversation_history=[]
        )
        
        # Update with English text
        updated = detector.update_context("Hello world", context)
        
        assert updated.language == "en"
        assert len(detector.language_history) == 1
        assert detector.language_history[0] == "en"
    
    def test_update_context_language_consistency(self):
        """Test language consistency across conversation (majority vote)."""
        detector = LanguageDetector()
        context = ExpressionContext(
            user_id="test_user",
            conversation_history=[]
        )
        
        # Mostly English with one Spanish utterance
        detector.update_context("Hello", context)
        detector.update_context("How are you?", context)
        detector.update_context("¿Qué tal?", context)  # Spanish
        detector.update_context("I'm fine", context)
        detector.update_context("Thank you", context)
        
        # Should still be English (majority)
        assert detector.current_language == "en"
        assert context.language == "en"
    
    def test_update_context_language_switch(self):
        """Test language switch when majority changes."""
        detector = LanguageDetector()
        context = ExpressionContext(
            user_id="test_user",
            conversation_history=[]
        )
        
        # Start with English
        for _ in range(3):
            detector.update_context("Hello", context)
        assert detector.current_language == "en"
        
        # Switch to Chinese (majority)
        for _ in range(5):
            detector.update_context("你好", context)
        
        # Should switch to Chinese
        assert detector.current_language == "zh"
        assert context.language == "zh"
    
    def test_update_context_formality_adjustment(self):
        """Test formality adjustment based on language."""
        detector = LanguageDetector()
        
        # English context (default formality)
        en_context = ExpressionContext(
            user_id="test_user",
            conversation_history=[]
        )
        detector.update_context("Hello", en_context)
        en_formality = en_context.formality_level
        
        # Reset detector
        detector = LanguageDetector()
        
        # Japanese context (higher formality)
        ja_context = ExpressionContext(
            user_id="test_user",
            conversation_history=[]
        )
        detector.update_context("こんにちは", ja_context)
        ja_formality = ja_context.formality_level
        
        # Japanese should be more formal
        assert ja_formality > en_formality
    
    def test_update_context_preserves_explicit_formality(self):
        """Test that explicit formality settings are preserved."""
        detector = LanguageDetector()
        context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            formality_level=0.9  # Explicitly set
        )
        
        # Update with English text
        detector.update_context("Hello", context)
        
        # Should preserve explicit formality
        assert context.formality_level == 0.9
    
    def test_update_context_history_limit(self):
        """Test language history is limited to 10 entries."""
        detector = LanguageDetector()
        context = ExpressionContext(
            user_id="test_user",
            conversation_history=[]
        )
        
        # Add 15 entries
        for i in range(15):
            detector.update_context("Hello", context)
        
        # Should only keep last 10
        assert len(detector.language_history) == 10
    
    def test_get_preferred_tts_backend(self):
        """Test getting preferred TTS backend for language."""
        detector = LanguageDetector()
        
        # English and Spanish prefer Piper
        assert detector.get_preferred_tts_backend("en") == "piper"
        assert detector.get_preferred_tts_backend("es") == "piper"
        
        # Chinese and Japanese prefer Azure
        assert detector.get_preferred_tts_backend("zh") == "azure"
        assert detector.get_preferred_tts_backend("ja") == "azure"
    
    def test_get_preferred_tts_backend_unknown_language(self):
        """Test TTS backend fallback for unknown language."""
        detector = LanguageDetector()
        
        # Unknown language should fallback to English
        backend = detector.get_preferred_tts_backend("fr")
        assert backend == "piper"  # English default
    
    def test_get_default_voice(self):
        """Test getting default voice for language."""
        detector = LanguageDetector()
        
        # Each language should have a voice ID
        assert "en_US" in detector.get_default_voice("en")
        assert "zh-CN" in detector.get_default_voice("zh")
        assert "ja-JP" in detector.get_default_voice("ja")
        assert "es_ES" in detector.get_default_voice("es")
    
    def test_get_default_voice_unknown_language(self):
        """Test voice fallback for unknown language."""
        detector = LanguageDetector()
        
        # Unknown language should fallback to English
        voice = detector.get_default_voice("fr")
        assert "en_US" in voice  # English default
    
    def test_adjust_emotion_intensity_english(self):
        """Test emotion intensity adjustment for English."""
        detector = LanguageDetector()
        
        # English has 1.0 multiplier (no change)
        assert detector.adjust_emotion_intensity(0.5, "en") == 0.5
        assert detector.adjust_emotion_intensity(0.8, "en") == 0.8
    
    def test_adjust_emotion_intensity_spanish(self):
        """Test emotion intensity adjustment for Spanish (more expressive)."""
        detector = LanguageDetector()
        
        # Spanish has 1.2 multiplier (more expressive)
        assert detector.adjust_emotion_intensity(0.5, "es") == 0.6
        assert detector.adjust_emotion_intensity(0.8, "es") == 0.96
    
    def test_adjust_emotion_intensity_chinese(self):
        """Test emotion intensity adjustment for Chinese (more subtle)."""
        detector = LanguageDetector()
        
        # Chinese has 0.8 multiplier (more subtle)
        assert detector.adjust_emotion_intensity(0.5, "zh") == 0.4
        assert detector.adjust_emotion_intensity(1.0, "zh") == 0.8
    
    def test_adjust_emotion_intensity_japanese(self):
        """Test emotion intensity adjustment for Japanese (more subtle)."""
        detector = LanguageDetector()
        
        # Japanese has 0.7 multiplier (more subtle)
        assert detector.adjust_emotion_intensity(0.5, "ja") == 0.35
        assert detector.adjust_emotion_intensity(1.0, "ja") == 0.7
    
    def test_adjust_emotion_intensity_clamping(self):
        """Test emotion intensity is clamped to 1.0."""
        detector = LanguageDetector()
        
        # Even with multiplier, should not exceed 1.0
        assert detector.adjust_emotion_intensity(0.9, "es") == 1.0
        assert detector.adjust_emotion_intensity(1.0, "es") == 1.0
    
    def test_adjust_emotion_intensity_unknown_language(self):
        """Test emotion intensity fallback for unknown language."""
        detector = LanguageDetector()
        
        # Unknown language should use English default (1.0 multiplier)
        assert detector.adjust_emotion_intensity(0.5, "fr") == 0.5
    
    def test_get_supported_languages(self):
        """Test getting list of supported languages."""
        detector = LanguageDetector()
        
        supported = detector.get_supported_languages()
        
        assert len(supported) >= 4
        assert "en" in supported
        assert "zh" in supported
        assert "ja" in supported
        assert "es" in supported
    
    def test_is_language_supported(self):
        """Test checking if language is supported."""
        detector = LanguageDetector()
        
        # Supported languages
        assert detector.is_language_supported("en") is True
        assert detector.is_language_supported("zh") is True
        assert detector.is_language_supported("ja") is True
        assert detector.is_language_supported("es") is True
        
        # Unsupported languages
        assert detector.is_language_supported("fr") is False
        assert detector.is_language_supported("de") is False
        assert detector.is_language_supported("invalid") is False
    
    def test_code_switching_scenario(self):
        """Test realistic code-switching scenario (multilingual conversation)."""
        detector = LanguageDetector()
        context = ExpressionContext(
            user_id="test_user",
            conversation_history=[]
        )
        
        # User starts in English
        detector.update_context("Hello, how can I help you?", context)
        assert context.language == "en"
        
        # User switches to Spanish (need majority to switch)
        detector.update_context("Hola, ¿hablas español?", context)
        # Still English (only 1 Spanish utterance)
        assert context.language == "en"
        
        # User continues in Spanish with accented characters
        detector.update_context("Sí, necesito información", context)
        detector.update_context("¿Puedes ayudarme?", context)
        detector.update_context("Está bien, gracias", context)
        
        # Should switch to Spanish (majority: 4 Spanish vs 1 English)
        assert context.language == "es"
    
    def test_language_detection_accuracy(self):
        """Test language detection accuracy with various inputs."""
        detector = LanguageDetector()
        
        test_cases = [
            ("Hello world", "en"),
            ("你好世界", "zh"),
            ("こんにちは", "ja"),
            ("Hola mundo", "en"),  # No accents, defaults to English
            ("¿Cómo estás?", "es"),  # With Spanish punctuation
            # Note: Kanji (日本語) contains Chinese characters, so detected as zh
            # This is a limitation of simple heuristic detection
            ("ひらがな", "ja"),  # Pure Hiragana
            ("カタカナ", "ja"),  # Pure Katakana
            ("中文测试", "zh"),
            ("Testing 123", "en"),
            ("", "en"),  # Empty defaults to current (en)
        ]
        
        for text, expected_lang in test_cases:
            detected = detector.detect_language(text)
            assert detected == expected_lang, \
                f"Failed for '{text}': expected {expected_lang}, got {detected}"

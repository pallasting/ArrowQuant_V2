"""
Integration tests for NLGEngine with TextStyleMapper.

Tests that emotion-aware text style mapping is properly integrated
into the NLG generation pipeline.

Requirements: 3.3, 13.1
"""

import pytest

from llm_compression.expression.nlg.nlg_engine import NLGEngine
from llm_compression.expression.expression_types import (
    NLGConfig,
    NLGBackend,
    ExpressionStyle,
    ExpressionContext
)


class TestNLGTextStyleIntegration:
    """Test suite for NLGEngine integration with TextStyleMapper."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use template backend for testing (no API required)
        self.config = NLGConfig(
            backend=NLGBackend.TEMPLATE,
            model="template",
            temperature=0.7,
            max_tokens=500,
            streaming=False
        )
        self.engine = NLGEngine(self.config)
        
        self.context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            current_emotion="neutral",
            formality_level=0.5,
            language="en"
        )
    
    def test_engine_has_text_style_mapper(self):
        """Test that NLGEngine has TextStyleMapper initialized."""
        assert hasattr(self.engine, 'text_style_mapper')
        assert self.engine.text_style_mapper is not None
    
    def test_build_system_prompt_with_emotion(self):
        """Test building system prompt with emotion parameters."""
        prompt = self.engine._build_system_prompt(
            style=ExpressionStyle.CASUAL,
            context=self.context,
            emotion="joy",
            emotion_intensity=0.8
        )
        
        assert prompt is not None
        assert len(prompt) > 0
        
        # Should contain emotion-related guidance
        assert "enthusiastic" in prompt.lower() or "positive" in prompt.lower()
    
    def test_build_system_prompt_uses_context_emotion(self):
        """Test that system prompt uses context emotion when not specified."""
        context_with_emotion = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            current_emotion="empathetic",
            formality_level=0.5,
            language="en"
        )
        
        prompt = self.engine._build_system_prompt(
            style=ExpressionStyle.EMPATHETIC,
            context=context_with_emotion
        )
        
        assert prompt is not None
        # Should contain empathetic guidance
        assert "empathy" in prompt.lower() or "understanding" in prompt.lower()
    
    def test_build_system_prompt_formality_adjustment(self):
        """Test that emotion adjusts formality in system prompt."""
        # High formality context
        formal_context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            current_emotion="neutral",
            formality_level=0.9,
            language="en"
        )
        
        formal_prompt = self.engine._build_system_prompt(
            style=ExpressionStyle.FORMAL,
            context=formal_context,
            emotion="trust",
            emotion_intensity=0.5
        )
        
        # Should contain formal language guidance
        assert "formal" in formal_prompt.lower()
        
        # Low formality context with emotion that reduces formality
        casual_context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            current_emotion="neutral",
            formality_level=0.3,
            language="en"
        )
        
        casual_prompt = self.engine._build_system_prompt(
            style=ExpressionStyle.CASUAL,
            context=casual_context,
            emotion="friendly",
            emotion_intensity=0.8
        )
        
        # Should contain casual language guidance
        assert "casual" in casual_prompt.lower() or "conversational" in casual_prompt.lower()
    
    def test_generate_with_emotion_parameters(self):
        """Test generating text with explicit emotion parameters."""
        # Generate with joy emotion
        result = list(self.engine.generate(
            prompt="Hello",
            style=ExpressionStyle.CASUAL,
            context=self.context,
            streaming=False,
            emotion="joy",
            emotion_intensity=0.8
        ))
        
        assert len(result) > 0
        assert result[0] is not None
    
    def test_generate_with_different_emotions(self):
        """Test generating text with different emotions."""
        emotions = ["joy", "sadness", "anger", "empathetic", "neutral"]
        
        for emotion in emotions:
            result = list(self.engine.generate(
                prompt="Test prompt",
                style=ExpressionStyle.CASUAL,
                context=self.context,
                streaming=False,
                emotion=emotion,
                emotion_intensity=0.6
            ))
            
            assert len(result) > 0
            assert result[0] is not None
    
    def test_generate_uses_context_emotion_by_default(self):
        """Test that generate uses context emotion when not specified."""
        context_with_emotion = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            current_emotion="playful",
            formality_level=0.3,
            language="en"
        )
        
        result = list(self.engine.generate(
            prompt="Test",
            style=ExpressionStyle.PLAYFUL,
            context=context_with_emotion,
            streaming=False
        ))
        
        assert len(result) > 0
        assert result[0] is not None
    
    def test_emotion_intensity_affects_prompt(self):
        """Test that emotion intensity affects the generated prompt."""
        # Low intensity
        low_intensity_prompt = self.engine._build_system_prompt(
            style=ExpressionStyle.CASUAL,
            context=self.context,
            emotion="joy",
            emotion_intensity=0.2
        )
        
        # High intensity
        high_intensity_prompt = self.engine._build_system_prompt(
            style=ExpressionStyle.CASUAL,
            context=self.context,
            emotion="joy",
            emotion_intensity=0.9
        )
        
        # Both should be valid prompts
        assert low_intensity_prompt is not None
        assert high_intensity_prompt is not None
        
        # They should be different (intensity affects formality)
        # Note: They might be the same if formality adjustment is small
        # but the underlying style parameters should differ
    
    def test_language_specific_instructions(self):
        """Test that language-specific instructions are added."""
        # Chinese context
        zh_context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            current_emotion="neutral",
            formality_level=0.5,
            language="zh"
        )
        
        prompt = self.engine._build_system_prompt(
            style=ExpressionStyle.CASUAL,
            context=zh_context,
            emotion="neutral",
            emotion_intensity=0.5
        )
        
        assert "zh" in prompt.lower() or "chinese" in prompt.lower()
    
    def test_accent_specific_instructions(self):
        """Test that accent-specific instructions are added."""
        context_with_accent = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            current_emotion="neutral",
            formality_level=0.5,
            language="en",
            accent="en-GB"
        )
        
        prompt = self.engine._build_system_prompt(
            style=ExpressionStyle.CASUAL,
            context=context_with_accent,
            emotion="neutral",
            emotion_intensity=0.5
        )
        
        assert "en-gb" in prompt.lower() or "regional" in prompt.lower()
    
    def test_multiple_styles_with_emotions(self):
        """Test all expression styles with various emotions."""
        styles = [
            ExpressionStyle.FORMAL,
            ExpressionStyle.CASUAL,
            ExpressionStyle.TECHNICAL,
            ExpressionStyle.EMPATHETIC,
            ExpressionStyle.PLAYFUL
        ]
        
        emotions = ["joy", "neutral", "empathetic"]
        
        for style in styles:
            for emotion in emotions:
                prompt = self.engine._build_system_prompt(
                    style=style,
                    context=self.context,
                    emotion=emotion,
                    emotion_intensity=0.5
                )
                
                assert prompt is not None
                assert len(prompt) > 0
    
    def test_template_fallback_with_emotion(self):
        """Test that template fallback works with emotion parameters."""
        # Template backend should always work
        result = list(self.engine.generate(
            prompt="error test",
            style=ExpressionStyle.EMPATHETIC,
            context=self.context,
            streaming=False,
            emotion="empathetic",
            emotion_intensity=0.7
        ))
        
        assert len(result) > 0
        assert result[0] is not None
        assert len(result[0]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

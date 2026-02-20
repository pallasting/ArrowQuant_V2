"""
Unit tests for TextStyleMapper.

Tests emotion-to-text-style mapping functionality including formality
adjustments, vocabulary choices, and tone guidance.

Requirements: 3.3, 13.1
"""

import pytest

from llm_compression.expression.emotion import TextStyleMapper, TextStyleParameters
from llm_compression.expression.expression_types import ExpressionStyle, ExpressionContext


class TestTextStyleMapper:
    """Test suite for TextStyleMapper."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = TextStyleMapper()
        self.context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            current_emotion="neutral",
            formality_level=0.5,
            language="en"
        )
    
    def test_initialization(self):
        """Test TextStyleMapper initialization."""
        assert self.mapper is not None
        assert len(self.mapper.emotion_styles) > 0
        
        # Check that all 8 basic emotions + extras are present
        expected_emotions = [
            "joy", "sadness", "anger", "fear",
            "surprise", "disgust", "trust", "anticipation",
            "neutral", "empathetic", "friendly", "playful"
        ]
        for emotion in expected_emotions:
            assert emotion in self.mapper.emotion_styles
    
    def test_apply_emotion_joy(self):
        """Test applying joy emotion to text style."""
        style_params = self.mapper.apply_emotion_to_style(
            emotion="joy",
            emotion_intensity=0.8,
            base_style=ExpressionStyle.CASUAL,
            context=self.context
        )
        
        assert style_params is not None
        assert "formality" in style_params
        assert "tone_descriptors" in style_params
        assert "word_choice_guidance" in style_params
        
        # Joy should reduce formality slightly
        assert style_params["formality"] < self.context.formality_level
        
        # Check tone descriptors
        assert "enthusiastic" in style_params["tone_descriptors"]
        assert "positive" in style_params["tone_descriptors"]
    
    def test_apply_emotion_sadness(self):
        """Test applying sadness emotion to text style."""
        style_params = self.mapper.apply_emotion_to_style(
            emotion="sadness",
            emotion_intensity=0.7,
            base_style=ExpressionStyle.EMPATHETIC,
            context=self.context
        )
        
        assert style_params is not None
        
        # Sadness should not change formality much
        assert abs(style_params["formality"] - self.context.formality_level) < 0.1
        
        # Check tone descriptors
        assert "gentle" in style_params["tone_descriptors"]
        assert "understanding" in style_params["tone_descriptors"]
        
        # Check vocabulary level (should be simpler)
        assert style_params["vocabulary_level"] < 0.5
    
    def test_apply_emotion_anger(self):
        """Test applying anger emotion to text style."""
        style_params = self.mapper.apply_emotion_to_style(
            emotion="anger",
            emotion_intensity=0.6,
            base_style=ExpressionStyle.FORMAL,
            context=self.context
        )
        
        assert style_params is not None
        
        # Anger should reduce formality
        assert style_params["formality"] < self.context.formality_level
        
        # Check tone descriptors
        assert "firm" in style_params["tone_descriptors"]
        assert "direct" in style_params["tone_descriptors"]
        
        # Check sentence length (should be shorter)
        assert style_params["sentence_length"] < 0.5
    
    def test_apply_emotion_neutral(self):
        """Test applying neutral emotion to text style."""
        style_params = self.mapper.apply_emotion_to_style(
            emotion="neutral",
            emotion_intensity=0.5,
            base_style=ExpressionStyle.TECHNICAL,
            context=self.context
        )
        
        assert style_params is not None
        
        # Neutral should not change formality
        assert style_params["formality"] == self.context.formality_level
        
        # Check tone descriptors
        assert "balanced" in style_params["tone_descriptors"]
        assert "objective" in style_params["tone_descriptors"]
    
    def test_emotion_intensity_scaling(self):
        """Test that emotion intensity scales formality adjustments."""
        # Low intensity
        low_intensity = self.mapper.apply_emotion_to_style(
            emotion="joy",
            emotion_intensity=0.2,
            base_style=ExpressionStyle.CASUAL,
            context=self.context
        )
        
        # High intensity
        high_intensity = self.mapper.apply_emotion_to_style(
            emotion="joy",
            emotion_intensity=0.9,
            base_style=ExpressionStyle.CASUAL,
            context=self.context
        )
        
        # High intensity should have larger formality adjustment
        low_adjustment = abs(low_intensity["formality"] - self.context.formality_level)
        high_adjustment = abs(high_intensity["formality"] - self.context.formality_level)
        
        assert high_adjustment > low_adjustment
    
    def test_formality_clamping(self):
        """Test that formality is clamped to valid range [0.0, 1.0]."""
        # Test with very high base formality
        high_context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            formality_level=0.95,
            language="en"
        )
        
        style_params = self.mapper.apply_emotion_to_style(
            emotion="trust",  # Neutral formality adjustment
            emotion_intensity=1.0,
            base_style=ExpressionStyle.FORMAL,
            context=high_context
        )
        
        assert 0.0 <= style_params["formality"] <= 1.0
        
        # Test with very low base formality
        low_context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            formality_level=0.05,
            language="en"
        )
        
        style_params = self.mapper.apply_emotion_to_style(
            emotion="anger",  # Reduces formality
            emotion_intensity=1.0,
            base_style=ExpressionStyle.CASUAL,
            context=low_context
        )
        
        assert 0.0 <= style_params["formality"] <= 1.0
    
    def test_build_style_prompt_formal(self):
        """Test building style prompt for formal context."""
        style_params = self.mapper.apply_emotion_to_style(
            emotion="trust",
            emotion_intensity=0.5,
            base_style=ExpressionStyle.FORMAL,
            context=ExpressionContext(
                user_id="test_user",
                conversation_history=[],
                formality_level=0.9,
                language="en"
            )
        )
        
        base_prompt = "You are a helpful AI assistant."
        enhanced_prompt = self.mapper.build_style_prompt(style_params, base_prompt)
        
        assert base_prompt in enhanced_prompt
        assert "formal" in enhanced_prompt.lower()
        assert "professional" in enhanced_prompt.lower()
    
    def test_build_style_prompt_casual(self):
        """Test building style prompt for casual context."""
        style_params = self.mapper.apply_emotion_to_style(
            emotion="friendly",
            emotion_intensity=0.7,
            base_style=ExpressionStyle.CASUAL,
            context=ExpressionContext(
                user_id="test_user",
                conversation_history=[],
                formality_level=0.2,
                language="en"
            )
        )
        
        base_prompt = "You are a helpful AI assistant."
        enhanced_prompt = self.mapper.build_style_prompt(style_params, base_prompt)
        
        assert base_prompt in enhanced_prompt
        assert "casual" in enhanced_prompt.lower() or "conversational" in enhanced_prompt.lower()
    
    def test_build_style_prompt_with_tone(self):
        """Test that tone descriptors are included in prompt."""
        style_params = self.mapper.apply_emotion_to_style(
            emotion="joy",
            emotion_intensity=0.8,
            base_style=ExpressionStyle.PLAYFUL,
            context=self.context
        )
        
        base_prompt = "You are a helpful AI assistant."
        enhanced_prompt = self.mapper.build_style_prompt(style_params, base_prompt)
        
        # Check that at least one tone descriptor is mentioned
        tone_descriptors = style_params["tone_descriptors"]
        assert any(tone in enhanced_prompt.lower() for tone in tone_descriptors)
    
    def test_build_style_prompt_with_word_choice(self):
        """Test that word choice guidance is included in prompt."""
        style_params = self.mapper.apply_emotion_to_style(
            emotion="empathetic",
            emotion_intensity=0.7,
            base_style=ExpressionStyle.EMPATHETIC,
            context=self.context
        )
        
        base_prompt = "You are a helpful AI assistant."
        enhanced_prompt = self.mapper.build_style_prompt(style_params, base_prompt)
        
        # Word choice guidance should be included
        word_choice = style_params["word_choice_guidance"]
        assert word_choice in enhanced_prompt
    
    def test_get_emotion_formality_adjustment(self):
        """Test getting formality adjustment for emotions."""
        # Joy should reduce formality
        joy_adjustment = self.mapper.get_emotion_formality_adjustment("joy")
        assert joy_adjustment < 0
        
        # Fear should increase formality
        fear_adjustment = self.mapper.get_emotion_formality_adjustment("fear")
        assert fear_adjustment > 0
        
        # Neutral should have no adjustment
        neutral_adjustment = self.mapper.get_emotion_formality_adjustment("neutral")
        assert neutral_adjustment == 0.0
    
    def test_get_supported_emotions(self):
        """Test getting list of supported emotions."""
        emotions = self.mapper.get_supported_emotions()
        
        assert len(emotions) >= 8  # At least 8 basic emotions
        assert "joy" in emotions
        assert "sadness" in emotions
        assert "anger" in emotions
        assert "neutral" in emotions
    
    def test_update_emotion_style(self):
        """Test updating emotion style parameters."""
        custom_params = TextStyleParameters(
            formality_adjustment=0.3,
            vocabulary_level=0.8,
            sentence_length=0.7,
            tone_descriptors=["custom", "test"],
            word_choice_guidance="Use custom language."
        )
        
        self.mapper.update_emotion_style("custom_emotion", custom_params)
        
        # Verify the update
        assert "custom_emotion" in self.mapper.emotion_styles
        assert self.mapper.emotion_styles["custom_emotion"] == custom_params
    
    def test_all_basic_emotions(self):
        """Test that all 8 basic emotions are properly configured."""
        basic_emotions = [
            "joy", "sadness", "anger", "fear",
            "surprise", "disgust", "trust", "anticipation"
        ]
        
        for emotion in basic_emotions:
            style_params = self.mapper.apply_emotion_to_style(
                emotion=emotion,
                emotion_intensity=0.5,
                base_style=ExpressionStyle.CASUAL,
                context=self.context
            )
            
            assert style_params is not None
            assert "formality" in style_params
            assert "tone_descriptors" in style_params
            assert len(style_params["tone_descriptors"]) > 0
            assert "word_choice_guidance" in style_params
    
    def test_unknown_emotion_fallback(self):
        """Test that unknown emotions fall back to neutral."""
        style_params = self.mapper.apply_emotion_to_style(
            emotion="unknown_emotion_xyz",
            emotion_intensity=0.5,
            base_style=ExpressionStyle.CASUAL,
            context=self.context
        )
        
        # Should fall back to neutral parameters
        neutral_params = self.mapper.apply_emotion_to_style(
            emotion="neutral",
            emotion_intensity=0.5,
            base_style=ExpressionStyle.CASUAL,
            context=self.context
        )
        
        assert style_params["formality"] == neutral_params["formality"]
        assert style_params["tone_descriptors"] == neutral_params["tone_descriptors"]
    
    def test_style_params_dataclass(self):
        """Test TextStyleParameters dataclass."""
        params = TextStyleParameters(
            formality_adjustment=0.2,
            vocabulary_level=0.6,
            sentence_length=0.5,
            tone_descriptors=["test", "example"],
            word_choice_guidance="Test guidance"
        )
        
        assert params.formality_adjustment == 0.2
        assert params.vocabulary_level == 0.6
        assert params.sentence_length == 0.5
        assert params.tone_descriptors == ["test", "example"]
        assert params.word_choice_guidance == "Test guidance"
    
    def test_style_params_default_values(self):
        """Test TextStyleParameters default values."""
        params = TextStyleParameters()
        
        assert params.formality_adjustment == 0.0
        assert params.vocabulary_level == 0.5
        assert params.sentence_length == 0.5
        assert params.tone_descriptors == []
        assert params.word_choice_guidance == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

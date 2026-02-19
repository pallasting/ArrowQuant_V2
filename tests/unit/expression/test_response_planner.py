"""
Unit tests for ResponsePlanner class.

Tests the multi-modal response planning functionality including modality selection,
style determination, emotion mapping, and language integration.

Requirements: 13.1, 15.6
"""

import pytest
from unittest.mock import Mock, patch

from llm_compression.expression.response_planner import ResponsePlanner
from llm_compression.expression.expression_types import (
    OutputModality,
    ExpressionStyle,
    ExpressionContext,
    ResponsePlan
)


class TestResponsePlanner:
    """Test suite for ResponsePlanner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planner = ResponsePlanner(
            default_modality=OutputModality.TEXT,
            enable_speech=True,
            enable_visual=False
        )
        
        self.context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            current_emotion="neutral",
            formality_level=0.5,
            time_of_day="day",
            language="en"
        )
    
    def test_init_default_configuration(self):
        """Test ResponsePlanner initialization with default configuration."""
        planner = ResponsePlanner()
        
        assert planner.default_modality == OutputModality.TEXT
        assert planner.enable_speech is True
        assert planner.enable_visual is False
        assert planner.language_detector is not None
        assert len(planner.style_rules) > 0
        assert len(planner.modality_preferences) > 0
    
    def test_init_custom_configuration(self):
        """Test ResponsePlanner initialization with custom configuration."""
        planner = ResponsePlanner(
            default_modality=OutputModality.SPEECH,
            enable_speech=False,
            enable_visual=True
        )
        
        assert planner.default_modality == OutputModality.SPEECH
        assert planner.enable_speech is False
        assert planner.enable_visual is True
    
    def test_style_rules_initialization(self):
        """Test that style rules are properly initialized."""
        rules = self.planner.style_rules
        
        # Check key mappings
        assert rules["error"] == ExpressionStyle.EMPATHETIC
        assert rules["technical_query"] == ExpressionStyle.TECHNICAL
        assert rules["greeting"] == ExpressionStyle.CASUAL
        assert rules["formal_request"] == ExpressionStyle.FORMAL
        assert rules["casual_chat"] == ExpressionStyle.PLAYFUL
    
    def test_modality_preferences_initialization(self):
        """Test that modality preferences are properly initialized."""
        prefs = self.planner.modality_preferences
        
        # Check key mappings
        assert OutputModality.TEXT in prefs["short_response"]
        assert OutputModality.SPEECH in prefs["short_response"]
        assert OutputModality.TEXT in prefs["technical_content"]
        assert OutputModality.VISUAL in prefs["visual_explanation"]
    
    def test_plan_response_basic(self):
        """Test basic response planning functionality."""
        plan = self.planner.plan_response(
            intent="answer_question",
            context=self.context,
            content_length=100
        )
        
        assert isinstance(plan, ResponsePlan)
        assert OutputModality.TEXT in plan.modalities
        assert plan.primary_modality in plan.modalities
        assert isinstance(plan.style, ExpressionStyle)
        assert isinstance(plan.emotion, str)
        assert 0.0 <= plan.emotion_intensity <= 1.0
        assert isinstance(plan.streaming, bool)
        assert plan.estimated_duration_ms > 0
    
    def test_modality_selection_short_content(self):
        """Test modality selection for short content."""
        plan = self.planner.plan_response(
            intent="greet",
            context=self.context,
            content_length=50
        )
        
        # Short content should include both text and speech
        assert OutputModality.TEXT in plan.modalities
        assert OutputModality.SPEECH in plan.modalities
    
    def test_modality_selection_long_content(self):
        """Test modality selection for long content."""
        plan = self.planner.plan_response(
            intent="explain_concept",
            context=self.context,
            content_length=1000
        )
        
        # Long content should prefer text
        assert OutputModality.TEXT in plan.modalities
        # Speech might not be included for very long content
    
    def test_modality_selection_visual_intent(self):
        """Test modality selection for visual intents."""
        planner = ResponsePlanner(enable_visual=True)
        
        plan = planner.plan_response(
            intent="show_diagram",
            context=self.context,
            content_length=200
        )
        
        assert OutputModality.VISUAL in plan.modalities
    
    def test_modality_selection_speech_disabled(self):
        """Test modality selection when speech is disabled."""
        planner = ResponsePlanner(enable_speech=False)
        
        plan = planner.plan_response(
            intent="greet",
            context=self.context,
            content_length=50
        )
        
        assert OutputModality.SPEECH not in plan.modalities
        assert OutputModality.TEXT in plan.modalities
    
    def test_primary_modality_selection_casual(self):
        """Test primary modality selection for casual context."""
        casual_context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            formality_level=0.2  # Very casual
        )
        
        plan = self.planner.plan_response(
            intent="chat",
            context=casual_context,
            content_length=100
        )
        
        # Should prefer speech for casual conversations
        if OutputModality.SPEECH in plan.modalities:
            assert plan.primary_modality == OutputModality.SPEECH
    
    def test_primary_modality_selection_visual(self):
        """Test primary modality selection for visual content."""
        planner = ResponsePlanner(enable_visual=True)
        
        plan = planner.plan_response(
            intent="visualize_data",
            context=self.context,
            content_length=200
        )
        
        if OutputModality.VISUAL in plan.modalities:
            assert plan.primary_modality == OutputModality.VISUAL
    
    def test_style_selection_intent_based(self):
        """Test style selection based on intent patterns."""
        test_cases = [
            ("error_occurred", ExpressionStyle.EMPATHETIC),
            ("technical_query", ExpressionStyle.TECHNICAL),
            ("greeting", ExpressionStyle.CASUAL),
            ("formal_request", ExpressionStyle.FORMAL),
            ("casual_chat", ExpressionStyle.PLAYFUL),
        ]
        
        for intent, expected_style in test_cases:
            plan = self.planner.plan_response(
                intent=intent,
                context=self.context,
                content_length=100
            )
            assert plan.style == expected_style
    
    def test_style_selection_formality_based(self):
        """Test style selection based on formality level."""
        # Formal context
        formal_context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            formality_level=0.8
        )
        
        plan = self.planner.plan_response(
            intent="unknown_intent",
            context=formal_context,
            content_length=100
        )
        assert plan.style == ExpressionStyle.FORMAL
        
        # Casual context
        casual_context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            formality_level=0.2
        )
        
        plan = self.planner.plan_response(
            intent="unknown_intent",
            context=casual_context,
            content_length=100
        )
        assert plan.style == ExpressionStyle.CASUAL
    
    def test_emotion_selection_intent_based(self):
        """Test emotion selection based on intent patterns."""
        test_cases = [
            ("error_occurred", "empathetic", 0.7),
            ("success_message", "joy", 0.6),
            ("greet_user", "friendly", 0.5),
            ("technical_explanation", "neutral", 0.3),
            ("joke_response", "playful", 0.8),
        ]
        
        for intent, expected_emotion, expected_min_intensity in test_cases:
            plan = self.planner.plan_response(
                intent=intent,
                context=self.context,
                content_length=100
            )
            assert plan.emotion == expected_emotion
            assert plan.emotion_intensity >= expected_min_intensity - 0.1  # Allow small variance
    
    def test_emotion_selection_context_based(self):
        """Test emotion selection based on context emotion."""
        emotional_context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            current_emotion="excited"
        )
        
        plan = self.planner.plan_response(
            intent="unknown_intent",
            context=emotional_context,
            content_length=100
        )
        
        # Should use context emotion for unknown intents
        assert plan.emotion == "excited"
    
    def test_template_selection(self):
        """Test template selection for common intents."""
        test_cases = [
            ("greet", True, "greeting_template"),
            ("hello", True, "greeting_template"),
            ("farewell", True, "goodbye_template"),
            ("acknowledge", True, "acknowledgment_template"),
            ("error", True, "error_template"),
            ("unknown_intent", False, None),
        ]
        
        for intent, should_use_template, expected_template in test_cases:
            plan = self.planner.plan_response(
                intent=intent,
                context=self.context,
                content_length=100
            )
            assert plan.use_template == should_use_template
            assert plan.template_name == expected_template
    
    def test_streaming_decision_long_content(self):
        """Test streaming decision for long content."""
        plan = self.planner.plan_response(
            intent="explain",
            context=self.context,
            content_length=500  # Long content
        )
        
        assert plan.streaming is True
    
    def test_streaming_decision_speech_modality(self):
        """Test streaming decision when speech is involved."""
        plan = self.planner.plan_response(
            intent="greet",
            context=self.context,
            content_length=50  # Short content but with speech
        )
        
        if OutputModality.SPEECH in plan.modalities:
            assert plan.streaming is True
    
    def test_streaming_decision_short_text_only(self):
        """Test streaming decision for short text-only content."""
        planner = ResponsePlanner(enable_speech=False)
        
        plan = planner.plan_response(
            intent="acknowledge",
            context=self.context,
            content_length=50  # Short content, no speech
        )
        
        assert plan.streaming is False
    
    def test_duration_estimation_text_only(self):
        """Test duration estimation for text-only responses."""
        planner = ResponsePlanner(enable_speech=False, enable_visual=False)
        
        plan = planner.plan_response(
            intent="explain",
            context=self.context,
            content_length=200
        )
        
        # Should be relatively fast for text only
        assert 0 < plan.estimated_duration_ms < 2000
    
    def test_duration_estimation_with_speech(self):
        """Test duration estimation with speech synthesis."""
        plan = self.planner.plan_response(
            intent="greet",
            context=self.context,
            content_length=100
        )
        
        if OutputModality.SPEECH in plan.modalities:
            # Should include speech synthesis time
            assert plan.estimated_duration_ms > 150  # Base speech time
    
    def test_duration_estimation_with_visual(self):
        """Test duration estimation with visual generation."""
        planner = ResponsePlanner(enable_visual=True)
        
        plan = planner.plan_response(
            intent="show_diagram",
            context=self.context,
            content_length=100
        )
        
        if OutputModality.VISUAL in plan.modalities:
            # Should include visual generation time
            assert plan.estimated_duration_ms > 5000  # Visual takes longer
    
    @patch('llm_compression.expression.response_planner.logger')
    def test_language_integration(self, mock_logger):
        """Test integration with LanguageDetector."""
        # Mock the language detector
        with patch.object(self.planner.language_detector, 'update_context') as mock_update:
            mock_update.return_value = self.context
            
            with patch.object(self.planner.language_detector, 'adjust_emotion_intensity') as mock_adjust:
                mock_adjust.return_value = 0.6
                
                plan = self.planner.plan_response(
                    intent="greet",
                    context=self.context,
                    content_length=100
                )
                
                # Verify language detector methods were called
                mock_update.assert_called_once()
                mock_adjust.assert_called_once()
    
    def test_get_language_detector(self):
        """Test getting the integrated language detector."""
        detector = self.planner.get_language_detector()
        assert detector is self.planner.language_detector
    
    def test_update_style_rules(self):
        """Test updating style selection rules."""
        new_rules = {
            "custom_intent": ExpressionStyle.FORMAL,
            "another_intent": ExpressionStyle.PLAYFUL
        }
        
        original_count = len(self.planner.style_rules)
        self.planner.update_style_rules(new_rules)
        
        assert len(self.planner.style_rules) == original_count + 2
        assert self.planner.style_rules["custom_intent"] == ExpressionStyle.FORMAL
        assert self.planner.style_rules["another_intent"] == ExpressionStyle.PLAYFUL
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = self.planner.get_supported_languages()
        
        assert isinstance(languages, list)
        assert len(languages) >= 4  # At least en, zh, ja, es
        assert "en" in languages
        assert "zh" in languages
        assert "ja" in languages
        assert "es" in languages
    
    def test_response_plan_validation(self):
        """Test that generated response plans are valid."""
        plan = self.planner.plan_response(
            intent="test_intent",
            context=self.context,
            content_length=100
        )
        
        # Validate ResponsePlan constraints
        assert plan.primary_modality in plan.modalities
        assert 0.0 <= plan.emotion_intensity <= 1.0
        assert plan.estimated_duration_ms >= 0
        
        if plan.use_template:
            assert plan.template_name is not None
    
    def test_edge_case_empty_intent(self):
        """Test handling of empty intent."""
        plan = self.planner.plan_response(
            intent="",
            context=self.context,
            content_length=100
        )
        
        # Should still generate a valid plan
        assert isinstance(plan, ResponsePlan)
        assert len(plan.modalities) > 0
    
    def test_edge_case_zero_content_length(self):
        """Test handling of zero content length."""
        plan = self.planner.plan_response(
            intent="greet",
            context=self.context,
            content_length=0
        )
        
        # Should still generate a valid plan
        assert isinstance(plan, ResponsePlan)
        assert plan.estimated_duration_ms >= 0
    
    def test_edge_case_very_long_content(self):
        """Test handling of very long content."""
        plan = self.planner.plan_response(
            intent="explain",
            context=self.context,
            content_length=10000  # Very long
        )
        
        # Should handle gracefully
        assert isinstance(plan, ResponsePlan)
        assert plan.streaming is True  # Should definitely stream
        assert plan.estimated_duration_ms > 1000  # Should take significant time


class TestResponsePlannerIntegration:
    """Integration tests for ResponsePlanner with other components."""
    
    def test_language_detector_integration(self):
        """Test full integration with LanguageDetector."""
        planner = ResponsePlanner()
        
        # Test with different languages
        contexts = [
            ExpressionContext(user_id="user1", conversation_history=[], language="en"),
            ExpressionContext(user_id="user2", conversation_history=[], language="zh"),
            ExpressionContext(user_id="user3", conversation_history=[], language="ja"),
            ExpressionContext(user_id="user4", conversation_history=[], language="es"),
        ]
        
        for context in contexts:
            plan = planner.plan_response(
                intent="greet",
                context=context,
                content_length=100
            )
            
            # Should generate valid plans for all languages
            assert isinstance(plan, ResponsePlan)
            assert len(plan.modalities) > 0
    
    def test_multi_language_emotion_adjustment(self):
        """Test emotion intensity adjustment across languages."""
        planner = ResponsePlanner()
        
        # Create contexts for different languages
        base_context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            current_emotion="joy"
        )
        
        languages = ["en", "zh", "ja", "es"]
        plans = {}
        
        for lang in languages:
            context = ExpressionContext(
                user_id=base_context.user_id,
                conversation_history=base_context.conversation_history,
                current_emotion=base_context.current_emotion,
                language=lang
            )
            
            plan = planner.plan_response(
                intent="success_message",
                context=context,
                content_length=100
            )
            
            plans[lang] = plan
        
        # Verify that emotion intensities are adjusted per language
        # (exact values depend on language rules)
        for lang, plan in plans.items():
            assert 0.0 <= plan.emotion_intensity <= 1.0
            assert plan.emotion == "joy"  # Should maintain emotion type
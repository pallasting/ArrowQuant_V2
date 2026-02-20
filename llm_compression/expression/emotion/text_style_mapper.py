"""
Text Style Mapper for emotion-aware text generation.

This module implements the TextStyleMapper class that maps emotions to text
generation styles, adjusting formality, word choice, and tone based on
emotional context.

Requirements: 3.3
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from llm_compression.expression.expression_types import ExpressionStyle, ExpressionContext

logger = logging.getLogger(__name__)


@dataclass
class TextStyleParameters:
    """
    Parameters for text style generation.
    
    Attributes:
        formality_adjustment: Adjustment to base formality (-1.0 to 1.0)
        vocabulary_level: Vocabulary complexity (0.0=simple, 1.0=complex)
        sentence_length: Preferred sentence length (0.0=short, 1.0=long)
        tone_descriptors: List of tone descriptors for the LLM
        word_choice_guidance: Specific guidance for word selection
    """
    formality_adjustment: float = 0.0
    vocabulary_level: float = 0.5
    sentence_length: float = 0.5
    tone_descriptors: list[str] = None
    word_choice_guidance: str = ""
    
    def __post_init__(self):
        if self.tone_descriptors is None:
            self.tone_descriptors = []


class TextStyleMapper:
    """
    Maps emotions to text generation style parameters.
    
    Converts emotional states into concrete text generation parameters
    including formality adjustments, vocabulary choices, sentence structure,
    and tone guidance for LLM-based text generation.
    
    Requirements: 3.3
    """
    
    def __init__(self):
        """Initialize text style mapper with emotion-to-style mappings."""
        self.emotion_styles = self._init_emotion_styles()
        logger.info(f"Initialized text style mapper with {len(self.emotion_styles)} emotion mappings")
    
    def _init_emotion_styles(self) -> Dict[str, TextStyleParameters]:
        """
        Initialize emotion-to-style parameter mappings.
        
        Maps each emotion to specific text style parameters that guide
        text generation to match the emotional context.
        
        Returns:
            Dictionary mapping emotion names to TextStyleParameters
        """
        return {
            "joy": TextStyleParameters(
                formality_adjustment=-0.1,  # Slightly less formal
                vocabulary_level=0.6,
                sentence_length=0.6,  # Slightly longer, enthusiastic
                tone_descriptors=["enthusiastic", "positive", "upbeat", "cheerful"],
                word_choice_guidance="Use positive, energetic words. Express excitement and happiness."
            ),
            "sadness": TextStyleParameters(
                formality_adjustment=0.0,
                vocabulary_level=0.4,  # Simpler, more direct
                sentence_length=0.4,  # Shorter, more subdued
                tone_descriptors=["gentle", "soft", "understanding", "somber"],
                word_choice_guidance="Use gentle, compassionate words. Avoid overly cheerful language."
            ),
            "anger": TextStyleParameters(
                formality_adjustment=-0.2,  # Less formal, more direct
                vocabulary_level=0.5,
                sentence_length=0.3,  # Short, direct sentences
                tone_descriptors=["firm", "direct", "assertive", "strong"],
                word_choice_guidance="Use clear, direct language. Be assertive but not aggressive."
            ),
            "fear": TextStyleParameters(
                formality_adjustment=0.1,  # Slightly more formal, cautious
                vocabulary_level=0.4,
                sentence_length=0.4,  # Shorter, uncertain
                tone_descriptors=["cautious", "careful", "reassuring", "supportive"],
                word_choice_guidance="Use reassuring, supportive language. Acknowledge concerns."
            ),
            "surprise": TextStyleParameters(
                formality_adjustment=-0.1,
                vocabulary_level=0.5,
                sentence_length=0.5,
                tone_descriptors=["animated", "expressive", "curious", "intrigued"],
                word_choice_guidance="Use expressive language. Show curiosity and interest."
            ),
            "disgust": TextStyleParameters(
                formality_adjustment=0.0,
                vocabulary_level=0.5,
                sentence_length=0.4,
                tone_descriptors=["measured", "careful", "diplomatic", "tactful"],
                word_choice_guidance="Use tactful, diplomatic language. Be respectful."
            ),
            "trust": TextStyleParameters(
                formality_adjustment=0.0,
                vocabulary_level=0.6,
                sentence_length=0.6,  # Longer, more explanatory
                tone_descriptors=["confident", "reliable", "steady", "trustworthy"],
                word_choice_guidance="Use confident, reliable language. Build credibility."
            ),
            "anticipation": TextStyleParameters(
                formality_adjustment=-0.05,
                vocabulary_level=0.6,
                sentence_length=0.6,
                tone_descriptors=["forward-looking", "optimistic", "engaged", "interested"],
                word_choice_guidance="Use forward-looking language. Express interest and engagement."
            ),
            "neutral": TextStyleParameters(
                formality_adjustment=0.0,
                vocabulary_level=0.5,
                sentence_length=0.5,
                tone_descriptors=["balanced", "clear", "informative", "objective"],
                word_choice_guidance="Use clear, balanced language. Be informative and objective."
            ),
            "empathetic": TextStyleParameters(
                formality_adjustment=-0.05,  # Slightly less formal, more personal
                vocabulary_level=0.5,
                sentence_length=0.5,
                tone_descriptors=["warm", "understanding", "compassionate", "supportive"],
                word_choice_guidance="Use warm, understanding language. Show empathy and support."
            ),
            "friendly": TextStyleParameters(
                formality_adjustment=-0.2,  # Less formal, more casual
                vocabulary_level=0.5,
                sentence_length=0.5,
                tone_descriptors=["warm", "approachable", "friendly", "conversational"],
                word_choice_guidance="Use friendly, conversational language. Be approachable."
            ),
            "playful": TextStyleParameters(
                formality_adjustment=-0.3,  # Much less formal
                vocabulary_level=0.6,
                sentence_length=0.5,
                tone_descriptors=["playful", "lighthearted", "fun", "witty"],
                word_choice_guidance="Use playful, lighthearted language. Be creative and fun."
            ),
        }
    
    def apply_emotion_to_style(
        self,
        emotion: str,
        emotion_intensity: float,
        base_style: ExpressionStyle,
        context: ExpressionContext
    ) -> Dict[str, Any]:
        """
        Apply emotion to text generation style.
        
        Converts emotion and intensity into concrete style parameters that
        can be used to guide text generation. Adjusts formality, vocabulary,
        and tone based on the emotional context.
        
        Args:
            emotion: Emotion name (e.g., "joy", "sadness")
            emotion_intensity: Emotion intensity (0.0-1.0)
            base_style: Base expression style
            context: Expression context with user preferences
            
        Returns:
            Dictionary with style parameters for text generation
        """
        # Get emotion-specific style parameters
        emotion_params = self.emotion_styles.get(
            emotion,
            self.emotion_styles["neutral"]
        )
        
        # Calculate adjusted formality
        base_formality = context.formality_level
        formality_adjustment = emotion_params.formality_adjustment * emotion_intensity
        adjusted_formality = max(0.0, min(1.0, base_formality + formality_adjustment))
        
        # Build style parameters
        style_params = {
            "formality": adjusted_formality,
            "vocabulary_level": emotion_params.vocabulary_level,
            "sentence_length": emotion_params.sentence_length,
            "tone_descriptors": emotion_params.tone_descriptors,
            "word_choice_guidance": emotion_params.word_choice_guidance,
            "emotion": emotion,
            "emotion_intensity": emotion_intensity,
            "base_style": base_style.value
        }
        
        logger.debug(
            f"Applied emotion '{emotion}' (intensity={emotion_intensity:.2f}) to style: "
            f"formality={adjusted_formality:.2f}, tone={emotion_params.tone_descriptors}"
        )
        
        return style_params
    
    def build_style_prompt(
        self,
        style_params: Dict[str, Any],
        base_prompt: str
    ) -> str:
        """
        Build enhanced prompt with style guidance.
        
        Constructs a detailed prompt that includes style parameters to guide
        the LLM in generating text that matches the desired emotional tone.
        
        Args:
            style_params: Style parameters from apply_emotion_to_style()
            base_prompt: Base system prompt
            
        Returns:
            Enhanced prompt with style guidance
        """
        # Extract parameters
        formality = style_params["formality"]
        tone_descriptors = style_params["tone_descriptors"]
        word_choice_guidance = style_params["word_choice_guidance"]
        vocabulary_level = style_params["vocabulary_level"]
        sentence_length = style_params["sentence_length"]
        
        # Build style instructions
        style_instructions = []
        
        # Formality instruction
        if formality > 0.7:
            style_instructions.append(
                "Use formal, professional language with proper etiquette and respectful tone."
            )
        elif formality < 0.3:
            style_instructions.append(
                "Use casual, conversational language as if talking to a friend."
            )
        else:
            style_instructions.append(
                "Use balanced, clear language that is neither too formal nor too casual."
            )
        
        # Tone instruction
        if tone_descriptors:
            tone_str = ", ".join(tone_descriptors)
            style_instructions.append(f"Maintain a {tone_str} tone throughout.")
        
        # Word choice instruction
        if word_choice_guidance:
            style_instructions.append(word_choice_guidance)
        
        # Vocabulary instruction
        if vocabulary_level > 0.7:
            style_instructions.append(
                "Use sophisticated vocabulary and complex sentence structures."
            )
        elif vocabulary_level < 0.3:
            style_instructions.append(
                "Use simple, clear vocabulary that is easy to understand."
            )
        
        # Sentence length instruction
        if sentence_length > 0.7:
            style_instructions.append(
                "Use longer, more detailed sentences with rich descriptions."
            )
        elif sentence_length < 0.3:
            style_instructions.append(
                "Use short, concise sentences. Be direct and to the point."
            )
        
        # Combine base prompt with style instructions
        enhanced_prompt = base_prompt
        if style_instructions:
            style_section = " ".join(style_instructions)
            enhanced_prompt = f"{base_prompt} {style_section}"
        
        logger.debug(f"Built style prompt with {len(style_instructions)} instructions")
        
        return enhanced_prompt
    
    def get_emotion_formality_adjustment(self, emotion: str) -> float:
        """
        Get formality adjustment for an emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Formality adjustment value (-1.0 to 1.0)
        """
        emotion_params = self.emotion_styles.get(
            emotion,
            self.emotion_styles["neutral"]
        )
        return emotion_params.formality_adjustment
    
    def get_supported_emotions(self) -> list[str]:
        """
        Get list of supported emotions.
        
        Returns:
            List of emotion names
        """
        return list(self.emotion_styles.keys())
    
    def update_emotion_style(
        self,
        emotion: str,
        style_params: TextStyleParameters
    ):
        """
        Update or add emotion style parameters.
        
        Args:
            emotion: Emotion name
            style_params: New style parameters
        """
        self.emotion_styles[emotion] = style_params
        logger.info(f"Updated style parameters for emotion: {emotion}")

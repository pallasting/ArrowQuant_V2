"""
Response planning for multi-modal expression strategies.

This module implements the ResponsePlanner class which analyzes context and intent
to determine the optimal multi-modal response strategy including modality selection,
style determination, and emotion mapping.

Requirements: 4.1, 8.1, 15.1
"""

import logging
from typing import Dict, List, Optional

from llm_compression.expression.expression_types import (
    OutputModality,
    ExpressionStyle,
    ExpressionContext,
    ResponsePlan
)
from llm_compression.expression.language.language_detector import LanguageDetector

logger = logging.getLogger(__name__)


class ResponsePlanner:
    """
    Plans multi-modal response strategy.
    
    Responsibilities:
    - Analyze context and intent
    - Select appropriate modalities
    - Determine expression style
    - Coordinate generation engines
    - Integrate language detection and adaptation
    
    Requirements: 4.1, 8.1, 15.1
    """
    
    def __init__(
        self,
        default_modality: OutputModality = OutputModality.TEXT,
        enable_speech: bool = True,
        enable_visual: bool = False
    ):
        """
        Initialize ResponsePlanner with configuration.
        
        Args:
            default_modality: Default output modality
            enable_speech: Enable speech synthesis
            enable_visual: Enable visual generation
            
        Requirements: 4.1, 8.1, 15.1
        """
        self.default_modality = default_modality
        self.enable_speech = enable_speech
        self.enable_visual = enable_visual
        
        # Initialize style selection rules
        self.style_rules = self._init_style_rules()
        
        # Initialize modality preferences
        self.modality_preferences = self._init_modality_preferences()
        
        # Integrate LanguageDetector
        self.language_detector = LanguageDetector()
        
        logger.info(
            f"ResponsePlanner initialized: speech={enable_speech}, "
            f"visual={enable_visual}, default={default_modality}"
        )
    
    def _init_style_rules(self) -> Dict[str, ExpressionStyle]:
        """
        Initialize context-to-style mapping rules.
        
        Maps intent patterns to appropriate expression styles.
        
        Returns:
            Dictionary mapping intent patterns to expression styles
            
        Requirements: 8.1
        """
        return {
            "error": ExpressionStyle.EMPATHETIC,
            "fail": ExpressionStyle.EMPATHETIC,
            "technical_query": ExpressionStyle.TECHNICAL,
            "explain": ExpressionStyle.TECHNICAL,
            "debug": ExpressionStyle.TECHNICAL,
            "greeting": ExpressionStyle.CASUAL,
            "hello": ExpressionStyle.CASUAL,
            "hi": ExpressionStyle.CASUAL,
            "formal_request": ExpressionStyle.FORMAL,
            "please": ExpressionStyle.FORMAL,
            "casual_chat": ExpressionStyle.PLAYFUL,
            "joke": ExpressionStyle.PLAYFUL,
            "fun": ExpressionStyle.PLAYFUL,
        }
    
    def _init_modality_preferences(self) -> Dict[str, List[OutputModality]]:
        """
        Initialize modality preferences based on content type.
        
        Returns:
            Dictionary mapping content types to preferred modalities
            
        Requirements: 4.1
        """
        return {
            "short_response": [OutputModality.TEXT, OutputModality.SPEECH],
            "long_response": [OutputModality.TEXT],  # Speech streaming for long content
            "technical_content": [OutputModality.TEXT],
            "casual_conversation": [OutputModality.SPEECH, OutputModality.TEXT],
            "visual_explanation": [OutputModality.TEXT, OutputModality.VISUAL],
            "error_message": [OutputModality.TEXT, OutputModality.SPEECH],
        }
    
    def plan_response(
        self,
        intent: str,
        context: ExpressionContext,
        content_length: int
    ) -> ResponsePlan:
        """
        Plan response strategy based on intent and context.
        
        Analyzes the intent, context, and content to determine the optimal
        multi-modal response strategy including modality selection, style
        determination, and emotion mapping.
        
        Args:
            intent: What to express (e.g., "answer_question", "greet")
            context: Expression context
            content_length: Estimated content length in characters
            
        Returns:
            ResponsePlan with modality and style decisions
            
        Requirements: 4.2, 4.4, 8.2, 15.6
        """
        # Update context with language detection and adaptation
        context = self.language_detector.update_context("", context)
        
        # Determine modalities
        modalities = self._select_modalities(intent, context, content_length)
        
        # Select primary modality
        primary = self._select_primary_modality(modalities, context)
        
        # Determine style
        style = self._select_style(intent, context)
        
        # Determine emotion
        emotion, intensity = self._select_emotion(intent, context)
        
        # Apply language-specific emotion adjustment
        intensity = self.language_detector.adjust_emotion_intensity(
            intensity, context.language
        )
        
        # Check if template applicable
        use_template, template_name = self._check_template(intent)
        
        # Streaming decision
        streaming = self._should_stream(content_length, modalities)
        
        # Estimate duration
        duration = self._estimate_duration(content_length, modalities)
        
        plan = ResponsePlan(
            modalities=modalities,
            primary_modality=primary,
            style=style,
            emotion=emotion,
            emotion_intensity=intensity,
            use_template=use_template,
            template_name=template_name,
            streaming=streaming,
            estimated_duration_ms=duration
        )
        
        logger.info(
            f"Response plan: modalities={[m.value for m in modalities]}, "
            f"style={style.value}, emotion={emotion}({intensity:.2f}), "
            f"language={context.language}"
        )
        
        return plan
    
    def _select_modalities(
        self,
        intent: str,
        context: ExpressionContext,
        content_length: int
    ) -> List[OutputModality]:
        """
        Select appropriate output modalities.
        
        Args:
            intent: Intent to express
            context: Expression context
            content_length: Content length in characters
            
        Returns:
            List of selected output modalities
            
        Requirements: 4.1, 4.2
        """
        modalities = [OutputModality.TEXT]  # Always include text
        
        # Add speech for appropriate content
        if self.enable_speech and self._should_use_speech(intent, context, content_length):
            modalities.append(OutputModality.SPEECH)
        
        # Add visual for appropriate content
        if self.enable_visual and self._should_use_visual(intent):
            modalities.append(OutputModality.VISUAL)
        
        return modalities
    
    def _should_use_speech(
        self,
        intent: str,
        context: ExpressionContext,
        content_length: int
    ) -> bool:
        """
        Determine if speech output is appropriate.
        
        Args:
            intent: Intent to express
            context: Expression context
            content_length: Content length in characters
            
        Returns:
            True if speech should be used
        """
        # Use speech for short content (better user experience)
        if content_length < 500:
            return True
        
        # Use speech for casual conversations
        if context.formality_level < 0.5:
            return True
        
        # Use speech for greetings and acknowledgments
        speech_intents = ["greet", "acknowledge", "farewell", "error"]
        if any(pattern in intent for pattern in speech_intents):
            return True
        
        # Don't use speech for very technical content
        if "technical" in intent and content_length > 200:
            return False
        
        return False
    
    def _should_use_visual(self, intent: str) -> bool:
        """
        Determine if visual output is appropriate.
        
        Args:
            intent: Intent to express
            
        Returns:
            True if visual should be used
            
        Requirements: 4.1
        """
        visual_intents = [
            "explain_concept",
            "show_diagram",
            "illustrate",
            "visualize_data",
            "demonstrate",
            "chart",
            "graph"
        ]
        return any(vi in intent for vi in visual_intents)
    
    def _select_primary_modality(
        self,
        modalities: List[OutputModality],
        context: ExpressionContext
    ) -> OutputModality:
        """
        Select primary output modality.
        
        Args:
            modalities: Available modalities
            context: Expression context
            
        Returns:
            Primary output modality
            
        Requirements: 4.2
        """
        # Prefer speech for casual conversations
        if (OutputModality.SPEECH in modalities and 
            context.formality_level < 0.5):
            return OutputModality.SPEECH
        
        # Prefer visual for visual content
        if OutputModality.VISUAL in modalities:
            return OutputModality.VISUAL
        
        # Default to text
        return OutputModality.TEXT
    
    def _select_style(
        self,
        intent: str,
        context: ExpressionContext
    ) -> ExpressionStyle:
        """
        Select expression style based on context.
        
        Args:
            intent: Intent to express
            context: Expression context
            
        Returns:
            Selected expression style
            
        Requirements: 8.2, 8.3, 15.3
        """
        # Check intent-based rules first
        for pattern, style in self.style_rules.items():
            if pattern in intent.lower():
                return style
        
        # Fallback to formality-based selection
        if context.formality_level > 0.7:
            return ExpressionStyle.FORMAL
        elif context.formality_level < 0.3:
            return ExpressionStyle.CASUAL
        else:
            return ExpressionStyle.TECHNICAL
    
    def _select_emotion(
        self,
        intent: str,
        context: ExpressionContext
    ) -> tuple[str, float]:
        """
        Select emotion and intensity.
        
        Args:
            intent: Intent to express
            context: Expression context
            
        Returns:
            Tuple of (emotion, intensity)
            
        Requirements: 3.1, 3.2, 8.4, 15.3
        """
        # Default to context emotion
        emotion = context.current_emotion
        intensity = 0.5
        
        # Adjust based on intent
        if "error" in intent or "fail" in intent:
            emotion = "empathetic"
            intensity = 0.7
        elif "success" in intent or "complete" in intent:
            emotion = "joy"
            intensity = 0.6
        elif "greet" in intent or "hello" in intent:
            emotion = "friendly"
            intensity = 0.5
        elif "technical" in intent or "explain" in intent:
            emotion = "neutral"
            intensity = 0.3
        elif "joke" in intent or "fun" in intent:
            emotion = "playful"
            intensity = 0.8
        
        return emotion, intensity
    
    def _check_template(self, intent: str) -> tuple[bool, Optional[str]]:
        """
        Check if template should be used.
        
        Args:
            intent: Intent to express
            
        Returns:
            Tuple of (use_template, template_name)
            
        Requirements: 4.4
        """
        template_intents = {
            "greet": "greeting_template",
            "hello": "greeting_template",
            "farewell": "goodbye_template",
            "bye": "goodbye_template",
            "acknowledge": "acknowledgment_template",
            "thanks": "acknowledgment_template",
            "error": "error_template",
            "fail": "error_template"
        }
        
        for pattern, template in template_intents.items():
            if pattern in intent.lower():
                return True, template
        
        return False, None
    
    def _should_stream(
        self,
        content_length: int,
        modalities: List[OutputModality]
    ) -> bool:
        """
        Determine if streaming should be used.
        
        Args:
            content_length: Content length in characters
            modalities: Selected modalities
            
        Returns:
            True if streaming should be used
            
        Requirements: 6.1, 6.2
        """
        # Stream for long content
        if content_length > 200:
            return True
        
        # Stream when speech is involved (better user experience)
        if OutputModality.SPEECH in modalities:
            return True
        
        return False
    
    def _estimate_duration(
        self,
        content_length: int,
        modalities: List[OutputModality]
    ) -> int:
        """
        Estimate response generation duration in milliseconds.
        
        Args:
            content_length: Content length in characters
            modalities: Selected modalities
            
        Returns:
            Estimated duration in milliseconds
            
        Requirements: 10.1, 10.2
        """
        duration = 0
        
        # Text generation: ~50 tokens/sec, ~4 chars/token
        if OutputModality.TEXT in modalities:
            tokens = content_length // 4  # Rough estimate
            duration += (tokens / 50) * 1000  # Convert to milliseconds
        
        # Speech generation: ~150ms base + content_length * 2ms
        if OutputModality.SPEECH in modalities:
            duration += 150 + (content_length * 2)
        
        # Visual generation: ~5-10 seconds
        if OutputModality.VISUAL in modalities:
            duration += 7000
        
        return int(duration)
    
    def get_language_detector(self) -> LanguageDetector:
        """
        Get the integrated language detector.
        
        Returns:
            LanguageDetector instance
        """
        return self.language_detector
    
    def update_style_rules(self, new_rules: Dict[str, ExpressionStyle]) -> None:
        """
        Update style selection rules.
        
        Args:
            new_rules: New style rules to add/update
        """
        self.style_rules.update(new_rules)
        logger.info(f"Updated style rules: {len(new_rules)} rules added")
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of ISO 639-1 language codes
        """
        return self.language_detector.get_supported_languages()
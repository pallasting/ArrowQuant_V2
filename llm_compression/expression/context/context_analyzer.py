"""
Context analyzer for adaptive expression.

Detects conversation context, analyzes formality/technical/emotional tone,
and enables adaptive expression based on context.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from llm_compression.expression.expression_types import ExpressionContext

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of conversation context."""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    EMOTIONAL = "emotional"
    MIXED = "mixed"


@dataclass
class ContextAnalysis:
    """Result of context analysis."""
    context_type: ContextType
    formality_level: float  # 0.0-1.0
    technical_level: float  # 0.0-1.0
    emotional_tone: str  # neutral, joy, sadness, etc.
    emotional_intensity: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    detected_patterns: List[str]


class ContextAnalyzer:
    """
    Analyzes conversation context for adaptive expression.
    
    Responsibilities:
    - Detect conversation context (formal, casual, technical, emotional)
    - Analyze formality level
    - Analyze technical level
    - Analyze emotional tone
    - Update context state
    
    Validates: Requirements 8.1, 8.2
    """
    
    def __init__(self):
        """Initialize context analyzer with pattern rules."""
        self.formal_patterns = self._init_formal_patterns()
        self.technical_patterns = self._init_technical_patterns()
        self.emotional_patterns = self._init_emotional_patterns()
        self.casual_patterns = self._init_casual_patterns()
        
        # Context history for tracking
        self.context_history: List[ContextAnalysis] = []
    
    def _init_formal_patterns(self) -> Dict[str, float]:
        """Initialize formal language patterns with weights."""
        return {
            r'\b(please|kindly|would you|could you)\b': 0.3,
            r'\b(sir|madam|mr\.|mrs\.|dr\.)\b': 0.4,
            r'\b(regarding|concerning|pursuant to|hereby)\b': 0.5,
            r'\b(sincerely|respectfully|cordially)\b': 0.4,
            r'\b(request|inquire|inform|notify)\b': 0.3,
            r'\b(appreciate|grateful|thank you very much)\b': 0.3,
            r'\b(furthermore|moreover|nevertheless|consequently)\b': 0.4,
        }
    
    def _init_technical_patterns(self) -> Dict[str, float]:
        """Initialize technical language patterns with weights."""
        return {
            r'\b(function|class|method|variable|parameter)\b': 0.4,
            r'\b(algorithm|implementation|optimization|complexity)\b': 0.5,
            r'\b(database|query|schema|index|transaction)\b': 0.4,
            r'\b(API|REST|HTTP|JSON|XML)\b': 0.4,
            r'\b(compile|debug|deploy|refactor|test)\b': 0.3,
            r'\b(architecture|framework|library|module)\b': 0.4,
            r'\b(performance|latency|throughput|scalability)\b': 0.5,
            r'[A-Z][a-z]+[A-Z][a-zA-Z]*': 0.2,  # CamelCase
            r'[a-z]+_[a-z_]+': 0.2,  # snake_case
            r'`[^`]+`': 0.3,  # Code snippets
        }
    
    def _init_emotional_patterns(self) -> Dict[str, tuple[str, float]]:
        """Initialize emotional language patterns with emotion and weight."""
        return {
            # Joy patterns
            r'\b(happy|excited|great|wonderful|amazing|love)\b': ('joy', 0.6),
            r'\b(yay|woohoo|awesome|fantastic|brilliant)\b': ('joy', 0.8),
            r'[!]{2,}': ('joy', 0.4),
            r':\)|😊|😄|🎉': ('joy', 0.7),
            
            # Sadness patterns
            r'\b(sad|unhappy|disappointed|sorry|unfortunate)\b': ('sadness', 0.6),
            r'\b(depressed|miserable|terrible|awful)\b': ('sadness', 0.8),
            r':\(|😢|😞': ('sadness', 0.7),
            
            # Anger patterns
            r'\b(angry|furious|annoyed|frustrated|irritated)\b': ('anger', 0.6),
            r'\b(hate|stupid|ridiculous|unacceptable)\b': ('anger', 0.7),
            r'[!]{3,}': ('anger', 0.5),
            
            # Fear patterns
            r'\b(worried|concerned|afraid|scared|anxious)\b': ('fear', 0.6),
            r'\b(nervous|uncertain|unsure|hesitant)\b': ('fear', 0.5),
            
            # Surprise patterns
            r'\b(wow|omg|really|seriously|incredible)\b': ('surprise', 0.6),
            r'\b(unexpected|surprising|shocked|amazed)\b': ('surprise', 0.7),
            r'\?!|!\\?': ('surprise', 0.5),
        }
    
    def _init_casual_patterns(self) -> Dict[str, float]:
        """Initialize casual language patterns with weights."""
        return {
            r'\b(hey|hi|hello|yo|sup)\b': 0.4,
            r'\b(yeah|yep|nope|nah|ok|okay)\b': 0.3,
            r'\b(gonna|wanna|gotta|kinda|sorta)\b': 0.5,
            r'\b(cool|nice|sweet|dude|buddy)\b': 0.4,
            r'\b(lol|lmao|btw|tbh|imo)\b': 0.6,
            r'[.]{3}': 0.2,  # Ellipsis
        }
    
    def analyze_context(
        self,
        text: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> ContextAnalysis:
        """
        Analyze conversation context from text.
        
        Args:
            text: Input text to analyze
            conversation_history: Optional conversation history for context
            
        Returns:
            ContextAnalysis with detected context type and metrics
        """
        text_lower = text.lower()
        
        # Analyze different dimensions
        formality_score = self._analyze_formality(text_lower)
        technical_score = self._analyze_technical_level(text, text_lower)
        emotion, emotion_intensity = self._analyze_emotional_tone(text_lower)
        casual_score = self._analyze_casual_level(text_lower)
        
        # Determine primary context type
        context_type, confidence, patterns = self._determine_context_type(
            formality_score,
            technical_score,
            emotion_intensity,
            casual_score
        )
        
        # Consider conversation history for smoothing
        if conversation_history and self.context_history:
            formality_score = self._smooth_with_history(
                formality_score,
                [c.formality_level for c in self.context_history[-3:]]
            )
            technical_score = self._smooth_with_history(
                technical_score,
                [c.technical_level for c in self.context_history[-3:]]
            )
        
        analysis = ContextAnalysis(
            context_type=context_type,
            formality_level=formality_score,
            technical_level=technical_score,
            emotional_tone=emotion,
            emotional_intensity=emotion_intensity,
            confidence=confidence,
            detected_patterns=patterns
        )
        
        # Store in history
        self.context_history.append(analysis)
        if len(self.context_history) > 10:
            self.context_history.pop(0)
        
        logger.info(
            f"Context analysis: type={context_type.value}, "
            f"formality={formality_score:.2f}, technical={technical_score:.2f}, "
            f"emotion={emotion}({emotion_intensity:.2f})"
        )
        
        return analysis
    
    def _analyze_formality(self, text: str) -> float:
        """
        Analyze formality level of text.
        
        Args:
            text: Lowercase text to analyze
            
        Returns:
            Formality score (0.0=casual, 1.0=formal)
        """
        score = 0.0
        matches = 0
        
        for pattern, weight in self.formal_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                score += weight
                matches += 1
        
        # Check for casual patterns (negative indicator)
        casual_penalty = 0.0
        for pattern, weight in self.casual_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                casual_penalty += weight
        
        # Normalize score
        if matches > 0:
            score = score / max(matches, 1)
        
        # Apply casual penalty
        score = max(0.0, score - casual_penalty * 0.3)
        
        # Sentence structure indicators
        if re.search(r'[.!?]$', text):  # Proper punctuation
            score += 0.1
        if len(text.split()) > 15:  # Longer sentences tend to be formal
            score += 0.1
        
        return min(1.0, score)
    
    def _analyze_technical_level(self, text: str, text_lower: str) -> float:
        """
        Analyze technical level of text.
        
        Args:
            text: Original text (for case-sensitive patterns)
            text_lower: Lowercase text
            
        Returns:
            Technical score (0.0=non-technical, 1.0=highly technical)
        """
        score = 0.0
        matches = 0
        
        for pattern, weight in self.technical_patterns.items():
            # Use original text for case-sensitive patterns
            search_text = text if pattern.startswith('[A-Z]') or '`' in pattern else text_lower
            found = re.findall(pattern, search_text)
            if found:
                score += weight * min(len(found), 3)  # Cap at 3 matches per pattern
                matches += 1
        
        # Normalize
        if matches > 0:
            score = score / (matches * 1.5)  # Adjust denominator for scaling
        
        return min(1.0, score)
    
    def _analyze_emotional_tone(self, text: str) -> tuple[str, float]:
        """
        Analyze emotional tone of text.
        
        Args:
            text: Lowercase text to analyze
            
        Returns:
            Tuple of (emotion_name, intensity)
        """
        emotion_scores: Dict[str, float] = {}
        
        for pattern, (emotion, weight) in self.emotional_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0.0
                emotion_scores[emotion] += weight * min(len(matches), 2)
        
        if not emotion_scores:
            return "neutral", 0.0
        
        # Find dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        emotion_name = dominant_emotion[0]
        intensity = min(1.0, dominant_emotion[1])
        
        return emotion_name, intensity
    
    def _analyze_casual_level(self, text: str) -> float:
        """
        Analyze casual language level.
        
        Args:
            text: Lowercase text to analyze
            
        Returns:
            Casual score (0.0=not casual, 1.0=very casual)
        """
        score = 0.0
        matches = 0
        
        for pattern, weight in self.casual_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                score += weight
                matches += 1
        
        if matches > 0:
            score = score / max(matches, 1)
        
        return min(1.0, score)
    
    def _determine_context_type(
        self,
        formality: float,
        technical: float,
        emotional: float,
        casual: float
    ) -> tuple[ContextType, float, List[str]]:
        """
        Determine primary context type from scores.
        
        Args:
            formality: Formality score
            technical: Technical score
            emotional: Emotional intensity
            casual: Casual score
            
        Returns:
            Tuple of (context_type, confidence, detected_patterns)
        """
        scores = {
            ContextType.FORMAL: formality,
            ContextType.TECHNICAL: technical,
            ContextType.EMOTIONAL: emotional,
            ContextType.CASUAL: casual,
        }
        
        # Find dominant type
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        dominant_type, dominant_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        
        # Check for mixed context
        if dominant_score < 0.4 or (dominant_score - second_score) < 0.2:
            return ContextType.MIXED, 0.5, ["mixed_signals"]
        
        # Confidence based on score strength
        confidence = min(1.0, dominant_score * 1.5)
        
        # Detected patterns
        patterns = []
        if formality > 0.3:
            patterns.append("formal_language")
        if technical > 0.3:
            patterns.append("technical_terms")
        if emotional > 0.3:
            patterns.append("emotional_content")
        if casual > 0.3:
            patterns.append("casual_language")
        
        return dominant_type, confidence, patterns
    
    def _smooth_with_history(
        self,
        current_score: float,
        history_scores: List[float],
        alpha: float = 0.3
    ) -> float:
        """
        Smooth score with historical context using exponential moving average.
        
        Args:
            current_score: Current score
            history_scores: Previous scores
            alpha: Smoothing factor (0.0=all history, 1.0=all current)
            
        Returns:
            Smoothed score
        """
        if not history_scores:
            return current_score
        
        avg_history = sum(history_scores) / len(history_scores)
        return alpha * current_score + (1 - alpha) * avg_history
    
    def update_context(
        self,
        text: str,
        context: ExpressionContext
    ) -> ExpressionContext:
        """
        Update expression context based on analysis.
        
        Args:
            text: Input text to analyze
            context: Current expression context
            
        Returns:
            Updated expression context
        """
        analysis = self.analyze_context(text, context.conversation_history)
        
        # Update context with analysis results
        context.formality_level = analysis.formality_level
        context.current_emotion = analysis.emotional_tone
        
        # Store analysis in user preferences for tracking
        if context.user_preferences is None:
            context.user_preferences = {}
        
        context.user_preferences["last_context_type"] = analysis.context_type.value
        context.user_preferences["last_technical_level"] = analysis.technical_level
        context.user_preferences["last_confidence"] = analysis.confidence
        
        logger.debug(f"Updated context: formality={context.formality_level:.2f}, emotion={context.current_emotion}")
        
        return context
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent context analysis.
        
        Returns:
            Dictionary with context statistics
        """
        if not self.context_history:
            return {"message": "No context history available"}
        
        recent = self.context_history[-5:]
        
        return {
            "total_analyses": len(self.context_history),
            "recent_count": len(recent),
            "avg_formality": sum(c.formality_level for c in recent) / len(recent),
            "avg_technical": sum(c.technical_level for c in recent) / len(recent),
            "dominant_emotion": max(
                set(c.emotional_tone for c in recent),
                key=lambda e: sum(1 for c in recent if c.emotional_tone == e)
            ),
            "context_types": [c.context_type.value for c in recent],
        }

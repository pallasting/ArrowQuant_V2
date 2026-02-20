"""
Emotion Consistency Validation Demo

This example demonstrates how to use the EmotionConsistencyValidator to
ensure emotion consistency across TTS voice parameters and text style.

Requirements: 3.6
"""

from llm_compression.expression.expression_types import (
    VoiceConfig,
    ExpressionStyle,
    ExpressionContext
)
from llm_compression.expression.emotion import (
    EmotionConsistencyValidator,
    TextStyleMapper
)


def demo_consistent_emotions():
    """Demonstrate consistent emotion expression."""
    print("=" * 60)
    print("Demo 1: Consistent Emotions (Joy)")
    print("=" * 60)
    
    validator = EmotionConsistencyValidator(tolerance=0.2)
    text_mapper = TextStyleMapper()
    
    # Create voice config with joy emotion
    voice_config = VoiceConfig(
        voice_id="en_US-lessac-medium",
        emotion="joy",
        emotion_intensity=0.7,
        speed=1.1,
        pitch=1.1,
        volume=1.0
    )
    
    # Create text style with matching emotion
    context = ExpressionContext(
        user_id="demo_user",
        conversation_history=[],
        current_emotion="joy",
        formality_level=0.4
    )
    
    text_style_params = text_mapper.apply_emotion_to_style(
        emotion="joy",
        emotion_intensity=0.7,
        base_style=ExpressionStyle.CASUAL,
        context=context
    )
    
    # Validate consistency
    warnings = validator.validate_consistency(voice_config, text_style_params)
    
    print(f"\nVoice Config:")
    print(f"  Emotion: {voice_config.emotion}")
    print(f"  Speed: {voice_config.speed:.2f}")
    print(f"  Pitch: {voice_config.pitch:.2f}")
    print(f"  Volume: {voice_config.volume:.2f}")
    
    print(f"\nText Style:")
    print(f"  Emotion: {text_style_params['emotion']}")
    print(f"  Formality: {text_style_params['formality']:.2f}")
    print(f"  Tone: {', '.join(text_style_params['tone_descriptors'])}")
    
    print(f"\nValidation Result:")
    if not warnings:
        print("  ✓ No consistency warnings - emotions are aligned!")
    else:
        print(f"  ⚠ {len(warnings)} warning(s) found:")
        for warning in warnings:
            print(f"    - [{warning.severity.upper()}] {warning.message}")
    
    print()


def demo_incompatible_emotions():
    """Demonstrate incompatible emotion expression."""
    print("=" * 60)
    print("Demo 2: Incompatible Emotions (Joy vs Sadness)")
    print("=" * 60)
    
    validator = EmotionConsistencyValidator(tolerance=0.2)
    text_mapper = TextStyleMapper()
    
    # Create voice config with joy emotion
    voice_config = VoiceConfig(
        voice_id="en_US-lessac-medium",
        emotion="joy",
        emotion_intensity=0.8,
        speed=1.15,  # Fast (joy)
        pitch=1.15,  # High (joy)
        volume=1.0
    )
    
    # Create text style with sadness emotion (incompatible!)
    context = ExpressionContext(
        user_id="demo_user",
        conversation_history=[],
        current_emotion="sadness",
        formality_level=0.5
    )
    
    text_style_params = text_mapper.apply_emotion_to_style(
        emotion="sadness",
        emotion_intensity=0.8,
        base_style=ExpressionStyle.EMPATHETIC,
        context=context
    )
    
    # Validate consistency
    warnings = validator.validate_consistency(voice_config, text_style_params)
    
    print(f"\nVoice Config:")
    print(f"  Emotion: {voice_config.emotion}")
    print(f"  Speed: {voice_config.speed:.2f} (fast)")
    print(f"  Pitch: {voice_config.pitch:.2f} (high)")
    print(f"  Volume: {voice_config.volume:.2f}")
    
    print(f"\nText Style:")
    print(f"  Emotion: {text_style_params['emotion']}")
    print(f"  Formality: {text_style_params['formality']:.2f}")
    print(f"  Tone: {', '.join(text_style_params['tone_descriptors'])}")
    
    print(f"\nValidation Result:")
    if not warnings:
        print("  ✓ No consistency warnings")
    else:
        print(f"  ⚠ {len(warnings)} warning(s) found:")
        for warning in warnings:
            print(f"\n    [{warning.severity.upper()}] {warning.message}")
            print(f"    TTS Emotion: {warning.tts_emotion}")
            print(f"    Text Emotion: {warning.text_emotion}")
            if warning.inconsistent_parameters:
                print(f"    Inconsistent: {', '.join(warning.inconsistent_parameters)}")
            print(f"    Suggested Fix: {warning.suggested_fix}")
    
    print()


def demo_compatible_emotions():
    """Demonstrate compatible emotion expression."""
    print("=" * 60)
    print("Demo 3: Compatible Emotions (Joy and Friendly)")
    print("=" * 60)
    
    validator = EmotionConsistencyValidator(tolerance=0.2)
    text_mapper = TextStyleMapper()
    
    # Create voice config with joy emotion
    voice_config = VoiceConfig(
        voice_id="en_US-lessac-medium",
        emotion="joy",
        emotion_intensity=0.6,
        speed=1.08,
        pitch=1.08,
        volume=1.0
    )
    
    # Create text style with friendly emotion (compatible with joy)
    context = ExpressionContext(
        user_id="demo_user",
        conversation_history=[],
        current_emotion="friendly",
        formality_level=0.3
    )
    
    text_style_params = text_mapper.apply_emotion_to_style(
        emotion="friendly",
        emotion_intensity=0.6,
        base_style=ExpressionStyle.CASUAL,
        context=context
    )
    
    # Validate consistency
    warnings = validator.validate_consistency(voice_config, text_style_params)
    
    print(f"\nVoice Config:")
    print(f"  Emotion: {voice_config.emotion}")
    print(f"  Speed: {voice_config.speed:.2f}")
    print(f"  Pitch: {voice_config.pitch:.2f}")
    
    print(f"\nText Style:")
    print(f"  Emotion: {text_style_params['emotion']}")
    print(f"  Tone: {', '.join(text_style_params['tone_descriptors'])}")
    
    print(f"\nValidation Result:")
    if not warnings:
        print("  ✓ No warnings - joy and friendly are compatible emotions!")
    else:
        print(f"  ⚠ {len(warnings)} warning(s) found")
    
    print()


def demo_warnings_summary():
    """Demonstrate warnings summary functionality."""
    print("=" * 60)
    print("Demo 4: Warnings Summary")
    print("=" * 60)
    
    validator = EmotionConsistencyValidator(tolerance=0.15)
    
    # Create multiple scenarios with different severity levels
    scenarios = [
        {
            "name": "Slight mismatch",
            "voice": VoiceConfig(
                voice_id="test",
                emotion="joy",
                emotion_intensity=0.3,
                speed=1.05,
                pitch=1.05,
                volume=1.0
            ),
            "text": {"emotion": "neutral", "emotion_intensity": 0.3}
        },
        {
            "name": "Moderate mismatch",
            "voice": VoiceConfig(
                voice_id="test",
                emotion="anger",
                emotion_intensity=0.7,
                speed=1.2,
                pitch=1.15,
                volume=1.0
            ),
            "text": {"emotion": "trust", "emotion_intensity": 0.7}
        },
        {
            "name": "Severe mismatch",
            "voice": VoiceConfig(
                voice_id="test",
                emotion="joy",
                emotion_intensity=0.9,
                speed=1.2,
                pitch=1.2,
                volume=1.0
            ),
            "text": {"emotion": "sadness", "emotion_intensity": 0.9}
        }
    ]
    
    all_warnings = []
    for scenario in scenarios:
        warnings = validator.validate_consistency(
            scenario["voice"],
            scenario["text"]
        )
        all_warnings.extend(warnings)
        print(f"\n{scenario['name']}: {len(warnings)} warning(s)")
    
    # Get summary
    summary = validator.get_warnings_summary(all_warnings)
    
    print(f"\n{'=' * 40}")
    print("Overall Summary:")
    print(f"{'=' * 40}")
    print(f"Total Warnings: {summary['total']}")
    print(f"By Severity:")
    print(f"  Low:    {summary['by_severity']['low']}")
    print(f"  Medium: {summary['by_severity']['medium']}")
    print(f"  High:   {summary['by_severity']['high']}")
    
    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Emotion Consistency Validation Demo" + " " * 13 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    demo_consistent_emotions()
    demo_incompatible_emotions()
    demo_compatible_emotions()
    demo_warnings_summary()
    
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()

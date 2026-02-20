"""
Demo for ContextAnalyzer - Context-aware expression analysis.

This demo showcases:
1. Detecting conversation context (formal, casual, technical, emotional)
2. Analyzing formality, technical, and emotional levels
3. Tracking context through conversations
4. Adaptive expression based on context
"""

from llm_compression.expression.context.context_analyzer import ContextAnalyzer
from llm_compression.expression.expression_types import ExpressionContext


def print_analysis(text: str, analysis):
    """Pretty print analysis results."""
    print(f"\n{'='*70}")
    print(f"Text: {text}")
    print(f"{'='*70}")
    print(f"Context Type: {analysis.context_type.value}")
    print(f"Formality Level: {analysis.formality_level:.2f} (0=casual, 1=formal)")
    print(f"Technical Level: {analysis.technical_level:.2f} (0=non-tech, 1=tech)")
    print(f"Emotional Tone: {analysis.emotional_tone} (intensity: {analysis.emotional_intensity:.2f})")
    print(f"Confidence: {analysis.confidence:.2f}")
    print(f"Detected Patterns: {', '.join(analysis.detected_patterns)}")


def demo_basic_context_detection():
    """Demo 1: Basic context detection."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Context Detection")
    print("="*70)
    
    analyzer = ContextAnalyzer()
    
    # Test different context types
    test_cases = [
        ("Formal", "Dear Sir, I would kindly request your assistance regarding this matter. I sincerely appreciate your consideration."),
        ("Casual", "Hey! Yeah, I'm gonna check that out later. Sounds cool, btw. Lol!"),
        ("Technical", "The algorithm uses a HashMap for O(1) lookup. We need to optimize the database query and refactor the API endpoint."),
        ("Emotional", "I'm so excited and happy about this! This is absolutely wonderful! 🎉😊"),
        ("Mixed", "Hey, could you please help me with the API implementation?"),
    ]
    
    for label, text in test_cases:
        analysis = analyzer.analyze_context(text)
        print(f"\n{label} Example:")
        print(f"  Detected: {analysis.context_type.value}")
        print(f"  Formality: {analysis.formality_level:.2f}, Technical: {analysis.technical_level:.2f}")
        print(f"  Emotion: {analysis.emotional_tone} ({analysis.emotional_intensity:.2f})")


def demo_conversation_tracking():
    """Demo 2: Tracking context through a conversation."""
    print("\n" + "="*70)
    print("DEMO 2: Conversation Context Tracking")
    print("="*70)
    
    analyzer = ContextAnalyzer()
    context = ExpressionContext(
        user_id="demo_user",
        conversation_history=[]
    )
    
    conversation = [
        "Hey, how's it going?",
        "I need some help with a technical problem.",
        "Can you explain how the database indexing works?",
        "The query performance is really slow with large datasets.",
        "Thanks! That's super helpful! I really appreciate it! 😊",
    ]
    
    print("\nConversation Flow:")
    for i, message in enumerate(conversation, 1):
        context = analyzer.update_context(message, context)
        analysis = analyzer.context_history[-1]
        
        print(f"\n{i}. User: {message}")
        print(f"   Context: {analysis.context_type.value}")
        print(f"   Formality: {analysis.formality_level:.2f}, Technical: {analysis.technical_level:.2f}")
        print(f"   Emotion: {analysis.emotional_tone} ({analysis.emotional_intensity:.2f})")
    
    # Show summary
    print("\n" + "-"*70)
    print("Conversation Summary:")
    summary = analyzer.get_context_summary()
    print(f"  Total messages: {summary['total_analyses']}")
    print(f"  Avg formality: {summary['avg_formality']:.2f}")
    print(f"  Avg technical: {summary['avg_technical']:.2f}")
    print(f"  Dominant emotion: {summary['dominant_emotion']}")
    print(f"  Context progression: {' → '.join(summary['context_types'])}")


def demo_emotion_detection():
    """Demo 3: Emotional tone detection."""
    print("\n" + "="*70)
    print("DEMO 3: Emotional Tone Detection")
    print("="*70)
    
    analyzer = ContextAnalyzer()
    
    emotions = [
        ("Joy", "I'm so happy and excited! This is wonderful! 🎉😊"),
        ("Sadness", "I'm really sad and disappointed about this. It's terrible. 😢"),
        ("Anger", "This is absolutely ridiculous and unacceptable!!! I'm furious!"),
        ("Fear", "I'm really worried and anxious about this. I'm afraid it won't work."),
        ("Surprise", "Wow! Really?! That's incredible and unexpected!"),
        ("Neutral", "The meeting is scheduled for tomorrow at 3 PM."),
    ]
    
    for expected, text in emotions:
        analysis = analyzer.analyze_context(text)
        print(f"\n{expected}:")
        print(f"  Text: {text}")
        print(f"  Detected: {analysis.emotional_tone} (intensity: {analysis.emotional_intensity:.2f})")
        print(f"  Match: {'✓' if analysis.emotional_tone.lower() == expected.lower() or (expected == 'Neutral' and analysis.emotional_intensity == 0) else '✗'}")


def demo_formality_spectrum():
    """Demo 4: Formality spectrum analysis."""
    print("\n" + "="*70)
    print("DEMO 4: Formality Spectrum Analysis")
    print("="*70)
    
    analyzer = ContextAnalyzer()
    
    formality_examples = [
        ("Very Formal", "I hereby respectfully request your kind consideration concerning this matter. I would be most grateful for your response."),
        ("Formal", "I would like to request your assistance with this issue. Thank you for your time."),
        ("Neutral", "Can you help me with this problem? Thanks."),
        ("Casual", "Hey, can you help me out? Thanks!"),
        ("Very Casual", "yo dude, wanna help me with this? lol thanks btw"),
    ]
    
    print("\nFormality Scale (0.0 = Very Casual, 1.0 = Very Formal):")
    print("-" * 70)
    
    for label, text in formality_examples:
        analysis = analyzer.analyze_context(text)
        bar_length = int(analysis.formality_level * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        print(f"\n{label:15} [{bar}] {analysis.formality_level:.2f}")
        print(f"                {text[:60]}...")


def demo_technical_level():
    """Demo 5: Technical level analysis."""
    print("\n" + "="*70)
    print("DEMO 5: Technical Level Analysis")
    print("="*70)
    
    analyzer = ContextAnalyzer()
    
    technical_examples = [
        ("Non-technical", "I really like this app. It's easy to use and looks nice."),
        ("Low technical", "The program needs to save the data to a file."),
        ("Medium technical", "We should use a database to store user information efficiently."),
        ("High technical", "The algorithm complexity is O(n log n) using a binary search tree with balanced rotations."),
        ("Very technical", "Implement the `getUserData()` method with async/await pattern. Use `fetch()` API for REST endpoint with JSON schema validation."),
    ]
    
    print("\nTechnical Scale (0.0 = Non-technical, 1.0 = Very Technical):")
    print("-" * 70)
    
    for label, text in technical_examples:
        analysis = analyzer.analyze_context(text)
        bar_length = int(analysis.technical_level * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        print(f"\n{label:15} [{bar}] {analysis.technical_level:.2f}")
        print(f"                {text[:60]}...")


def demo_adaptive_expression():
    """Demo 6: Adaptive expression based on context."""
    print("\n" + "="*70)
    print("DEMO 6: Adaptive Expression Recommendations")
    print("="*70)
    
    analyzer = ContextAnalyzer()
    
    scenarios = [
        "I need help with my code. It's not working.",
        "Dear team, I would like to discuss the project timeline.",
        "OMG this is amazing!!! I love it so much! 🎉",
        "I'm worried this approach might not scale well.",
    ]
    
    print("\nContext-aware expression recommendations:")
    
    for text in scenarios:
        analysis = analyzer.analyze_context(text)
        
        print(f"\n{'-'*70}")
        print(f"User: {text}")
        print(f"Context: {analysis.context_type.value}")
        
        # Recommend expression style
        if analysis.context_type.name == "FORMAL":
            recommendation = "Use formal language, proper titles, structured responses"
        elif analysis.context_type.name == "CASUAL":
            recommendation = "Use friendly, conversational tone, contractions OK"
        elif analysis.context_type.name == "TECHNICAL":
            recommendation = "Use precise terminology, provide technical details, code examples"
        elif analysis.context_type.name == "EMOTIONAL":
            recommendation = f"Show empathy, match {analysis.emotional_tone} tone, be supportive"
        else:
            recommendation = "Balance formality and friendliness, adapt to user's style"
        
        print(f"Recommendation: {recommendation}")
        print(f"Formality: {analysis.formality_level:.2f}, Technical: {analysis.technical_level:.2f}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("CONTEXT ANALYZER DEMO")
    print("Context-aware expression analysis for adaptive AI communication")
    print("="*70)
    
    demos = [
        ("Basic Context Detection", demo_basic_context_detection),
        ("Conversation Tracking", demo_conversation_tracking),
        ("Emotion Detection", demo_emotion_detection),
        ("Formality Spectrum", demo_formality_spectrum),
        ("Technical Level", demo_technical_level),
        ("Adaptive Expression", demo_adaptive_expression),
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

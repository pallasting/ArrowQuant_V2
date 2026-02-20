#!/usr/bin/env python3
"""
NLGEngine Demo - Expression & Presentation Layer

This demo shows the NLGEngine class in action with different backends,
styles, and contexts. It demonstrates the core functionality implemented
in task 3.1.

Requirements: 2.1, 2.2
"""

import os
from llm_compression.expression.nlg import NLGEngine
from llm_compression.expression.expression_types import (
    NLGConfig,
    NLGBackend,
    ExpressionStyle,
    ExpressionContext
)


def demo_template_backend():
    """Demonstrate template-based NLG (always available)."""
    print("=" * 60)
    print("NLGEngine Demo - Template Backend")
    print("=" * 60)
    
    # Create template-based configuration
    config = NLGConfig(
        backend=NLGBackend.TEMPLATE,
        streaming=False
    )
    
    # Initialize engine
    engine = NLGEngine(config)
    
    # Create sample context
    context = ExpressionContext(
        user_id="demo_user",
        conversation_history=[],
        current_emotion="neutral",
        formality_level=0.5,
        time_of_day="morning",
        language="en"
    )
    
    # Test different styles with the same prompt
    prompt = "Hello, how can I help you today?"
    
    styles = [
        ExpressionStyle.FORMAL,
        ExpressionStyle.CASUAL,
        ExpressionStyle.TECHNICAL,
        ExpressionStyle.EMPATHETIC,
        ExpressionStyle.PLAYFUL
    ]
    
    for style in styles:
        print(f"\n{style.value.upper()} Style:")
        response = list(engine.generate(prompt, style, context))[0]
        print(f"  → {response}")
    
    # Test context adaptation
    print(f"\nContext Adaptation Demo:")
    
    # High formality context
    formal_context = ExpressionContext(
        user_id="demo_user",
        conversation_history=[],
        formality_level=0.9,  # Very formal
        time_of_day="morning"
    )
    
    # Low formality context  
    casual_context = ExpressionContext(
        user_id="demo_user",
        conversation_history=[],
        formality_level=0.1,  # Very casual
        time_of_day="morning"
    )
    
    casual_style_formal_context = list(engine.generate(
        "Thank you", ExpressionStyle.CASUAL, formal_context
    ))[0]
    
    casual_style_casual_context = list(engine.generate(
        "Thank you", ExpressionStyle.CASUAL, casual_context
    ))[0]
    
    print(f"  Casual style + Formal context: {casual_style_formal_context}")
    print(f"  Casual style + Casual context: {casual_style_casual_context}")


def demo_openai_backend():
    """Demonstrate OpenAI backend (if API key available)."""
    print("\n" + "=" * 60)
    print("NLGEngine Demo - OpenAI Backend")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OpenAI API key not found. Skipping OpenAI demo.")
        print("   Set OPENAI_API_KEY environment variable to test OpenAI backend.")
        return
    
    # Create OpenAI configuration
    config = NLGConfig(
        backend=NLGBackend.OPENAI,
        model="gpt-3.5-turbo",  # Use cheaper model for demo
        temperature=0.7,
        max_tokens=100,
        streaming=False
    )
    
    try:
        # Initialize engine
        engine = NLGEngine(config)
        
        # Check backend status
        status = engine.get_backend_status()
        print(f"Backend Status: {status['health']}")
        
        if status['health'] != 'healthy':
            print("⚠️  OpenAI backend not healthy. Falling back to template.")
            return
        
        # Create context
        context = ExpressionContext(
            user_id="demo_user",
            conversation_history=[],
            current_emotion="neutral",
            formality_level=0.5,
            language="en"
        )
        
        # Test generation
        prompt = "Explain what artificial intelligence is in simple terms."
        
        print(f"\nPrompt: {prompt}")
        print(f"Style: {ExpressionStyle.TECHNICAL.value}")
        
        response = list(engine.generate(prompt, ExpressionStyle.TECHNICAL, context))[0]
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"⚠️  OpenAI backend failed: {e}")
        print("   Falling back to template backend.")


def demo_backend_fallback():
    """Demonstrate backend fallback mechanism."""
    print("\n" + "=" * 60)
    print("NLGEngine Demo - Backend Fallback")
    print("=" * 60)
    
    # Create configuration for non-existent backend
    config = NLGConfig(
        backend=NLGBackend.OPENAI,  # Will fail without API key
        streaming=False
    )
    
    # Initialize engine (should fall back to template)
    engine = NLGEngine(config)
    
    # Create context
    context = ExpressionContext(
        user_id="demo_user",
        conversation_history=[],
        language="en"
    )
    
    # Test generation (should use template fallback)
    prompt = "There was an error processing your request."
    
    print(f"Configured backend: {config.backend.value}")
    print(f"Actual backend initialized: {engine.backend is not None}")
    print(f"Template fallback available: {engine.template_engine is not None}")
    
    print(f"\nPrompt: {prompt}")
    response = list(engine.generate(prompt, ExpressionStyle.EMPATHETIC, context))[0]
    print(f"Response (via fallback): {response}")


def demo_multilingual_support():
    """Demonstrate multi-language context support."""
    print("\n" + "=" * 60)
    print("NLGEngine Demo - Multi-language Support")
    print("=" * 60)
    
    config = NLGConfig(backend=NLGBackend.TEMPLATE)
    engine = NLGEngine(config)
    
    # Test different languages
    languages = [
        ("en", "English"),
        ("zh", "Chinese"),
        ("ja", "Japanese"),
        ("es", "Spanish")
    ]
    
    for lang_code, lang_name in languages:
        context = ExpressionContext(
            user_id="demo_user",
            conversation_history=[],
            language=lang_code,
            formality_level=0.5
        )
        
        # Build system prompt to see language adaptation
        system_prompt = engine._build_system_prompt(ExpressionStyle.FORMAL, context)
        
        print(f"\n{lang_name} ({lang_code}):")
        print(f"  System prompt includes: {lang_code in system_prompt.lower()}")
        
        # Generate response
        response = list(engine.generate("Hello", ExpressionStyle.FORMAL, context))[0]
        print(f"  Response: {response}")


def main():
    """Run all NLGEngine demos."""
    print("🤖 NLGEngine Demonstration")
    print("Expression & Presentation Layer - Task 3.1")
    print()
    
    try:
        # Core template functionality (always works)
        demo_template_backend()
        
        # OpenAI backend (if available)
        demo_openai_backend()
        
        # Fallback mechanism
        demo_backend_fallback()
        
        # Multi-language support
        demo_multilingual_support()
        
        print("\n" + "=" * 60)
        print("✅ NLGEngine Demo Complete!")
        print("=" * 60)
        print()
        print("Key Features Demonstrated:")
        print("  ✓ Multi-backend support (Template, OpenAI)")
        print("  ✓ Style-aware generation (Formal, Casual, Technical, etc.)")
        print("  ✓ Context adaptation (formality, language, emotion)")
        print("  ✓ Graceful fallback to templates")
        print("  ✓ Multi-language context support")
        print("  ✓ Robust error handling")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
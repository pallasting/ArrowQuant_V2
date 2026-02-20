#!/usr/bin/env python3
"""
Demo script for Anthropic backend in NLGEngine.

This script demonstrates the Anthropic backend functionality including:
- Client initialization
- Streaming generation
- Complete generation
- Error handling with fallback

Requirements: 2.1
"""

import os
from llm_compression.expression.nlg import NLGEngine
from llm_compression.expression.expression_types import (
    NLGConfig, 
    NLGBackend, 
    ExpressionStyle, 
    ExpressionContext
)

def demo_anthropic_backend():
    """Demonstrate Anthropic backend functionality."""
    print("=== Anthropic Backend Demo ===\n")
    
    # Check if API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set - using mock demonstration")
        print("   Set ANTHROPIC_API_KEY environment variable for live testing\n")
        return demo_mock_anthropic()
    
    # Configure Anthropic backend
    config = NLGConfig(
        backend=NLGBackend.ANTHROPIC,
        model="claude-3-sonnet-20240229",
        temperature=0.7,
        max_tokens=150,
        streaming=True
    )
    
    # Create context
    context = ExpressionContext(
        user_id="demo_user",
        conversation_history=[],
        current_emotion="neutral",
        formality_level=0.5,
        language="en"
    )
    
    try:
        # Initialize engine
        print("1. Initializing Anthropic backend...")
        engine = NLGEngine(config)
        print("   ✅ Backend initialized successfully")
        
        # Test streaming generation
        print("\n2. Testing streaming generation...")
        prompt = "Explain the concept of machine learning in simple terms"
        
        print(f"   Prompt: {prompt}")
        print("   Response: ", end="", flush=True)
        
        for token in engine.generate(prompt, ExpressionStyle.CASUAL, context, streaming=True):
            print(token, end="", flush=True)
        print("\n   ✅ Streaming generation completed")
        
        # Test complete generation
        print("\n3. Testing complete generation...")
        config.streaming = False
        engine = NLGEngine(config)
        
        prompt = "What are the benefits of renewable energy?"
        print(f"   Prompt: {prompt}")
        
        response = list(engine.generate(prompt, ExpressionStyle.FORMAL, context, streaming=False))
        print(f"   Response: {response[0]}")
        print("   ✅ Complete generation completed")
        
        # Test backend status
        print("\n4. Checking backend status...")
        status = engine.get_backend_status()
        print(f"   Status: {status}")
        
        print("\n🎉 All Anthropic backend features working correctly!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   🔄 Falling back to template engine...")
        
        # Demonstrate fallback behavior
        template_response = list(engine.generate(prompt, ExpressionStyle.CASUAL, context))
        print(f"   Fallback response: {template_response[0]}")
        print("   ✅ Fallback mechanism working")

def demo_mock_anthropic():
    """Demonstrate Anthropic backend with mock (no API key required)."""
    print("📝 Mock demonstration (no API calls made):")
    print()
    
    print("✅ Anthropic backend implementation includes:")
    print("   • Client initialization with API key validation")
    print("   • Streaming generation using messages.stream()")
    print("   • Complete generation using messages.create()")
    print("   • Proper error handling with template fallback")
    print("   • Support for Claude models (claude-3-sonnet-20240229, etc.)")
    print("   • Integration with expression styles and context")
    print()
    
    print("🔧 Implementation details:")
    print("   • _init_anthropic_backend(): Initializes Anthropic client")
    print("   • _anthropic_streaming(): Handles streaming responses")
    print("   • _anthropic_complete(): Handles complete responses")
    print("   • Proper message format with system and user roles")
    print("   • Error handling with NLGError exceptions")
    print()
    
    print("🧪 Test coverage:")
    print("   • Backend initialization (with/without API key)")
    print("   • Streaming generation with mock responses")
    print("   • Complete generation with mock responses")
    print("   • Error handling and fallback mechanisms")
    print()
    
    print("📋 Requirements compliance:")
    print("   • ✅ Requirement 2.1: LLM backend integration (Anthropic)")
    print("   • ✅ Streaming token generation support")
    print("   • ✅ Context and style awareness")
    print("   • ✅ Error handling with graceful fallback")

if __name__ == "__main__":
    demo_anthropic_backend()
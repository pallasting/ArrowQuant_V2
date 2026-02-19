#!/usr/bin/env python3
"""Quick test of Ollama API integration"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm_compression.llm_client import LLMClient


async def test_ollama():
    """Test Ollama API integration"""
    print("Testing Ollama API integration...")
    print("-" * 80)
    
    # Initialize client with Ollama endpoint
    client = LLMClient(
        endpoint="http://localhost:11434",
        api_type="ollama",
        timeout=30.0
    )
    
    try:
        # Test basic generation
        print("\nTest 1: Basic generation")
        response = await client.generate(
            prompt="Compress this text into a brief summary: The quick brown fox jumps over the lazy dog.",
            max_tokens=50,
            temperature=0.3
        )
        
        print(f"✅ Response: {response.text[:100]}...")
        print(f"✅ Tokens: {response.tokens_used}")
        print(f"✅ Latency: {response.latency_ms:.1f}ms")
        print(f"✅ Model: {response.model}")
        
        # Test with longer text
        print("\nTest 2: Longer text")
        long_text = "This is a longer test text. " * 20
        response = await client.generate(
            prompt=f"Summarize: {long_text}",
            max_tokens=100,
            temperature=0.3
        )
        
        print(f"✅ Response: {response.text[:100]}...")
        print(f"✅ Latency: {response.latency_ms:.1f}ms")
        
        # Check metrics
        print("\nClient Metrics:")
        metrics = client.get_metrics()
        print(f"  Total requests: {metrics['total_requests']}")
        print(f"  Success rate: {metrics['success_rate']:.1%}")
        print(f"  Avg latency: {metrics['avg_latency_ms']:.1f}ms")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await client.close()


if __name__ == "__main__":
    success = asyncio.run(test_ollama())
    sys.exit(0 if success else 1)

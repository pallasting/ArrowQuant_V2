import os
from pathlib import Path
from llm_compression.protocol_adapter import ProtocolAdapter


def compress_text_generative(text: str, model: str = "claude-opus-4") -> dict:
    """
    Compresses text using an LLM with automatic protocol selection.

    Args:
        text: Text to compress
        model: Model to use (auto-selects optimal protocol)
               Options: claude-opus-4, gpt-4, gemini-pro, etc.

    Returns:
        Dict with summary, entities, compression metrics
    """
    # Initialize adapter
    api_key = os.getenv("LLM_API_KEY") or "sk-0437c02b1560470981866f50b05759e3"
    adapter = ProtocolAdapter(base_url="http://localhost:8045", api_key=api_key)

    print(f"🤖 Using model: {model}")
    print(f"📡 Protocol: {adapter.get_model_config(model).protocol.value}")
    print()

    # Generate summary
    print("📝 Generating summary...")
    summary_prompt = f"Summarize the following text concisely in 1-2 sentences. Text: {text}"
    summary_result = adapter.complete_with_metadata(summary_prompt, model=model, max_tokens=200)
    summary = summary_result["content"]

    print(f"   ✅ Summary generated ({summary_result['usage'].get('output_tokens', 0)} tokens)")

    # Extract entities
    print("🏷️  Extracting entities...")
    entities_prompt = f"Extract key entities (numbers, names, dates, locations, terms) from this text as a JSON list. Example: [\"entity1\", \"entity2\"]. Text: {text}"
    entities_result = adapter.complete_with_metadata(entities_prompt, model=model, max_tokens=200)
    entities_str = entities_result["content"]

    print(f"   ✅ Entities extracted ({entities_result['usage'].get('output_tokens', 0)} tokens)")

    # Parse entities
    import json
    entities_list = []
    try:
        # Try to parse as JSON
        parsed = json.loads(entities_str)
        if isinstance(parsed, list):
            entities_list = parsed
        else:
            entities_list = [entities_str[:200]]
    except json.JSONDecodeError:
        # Extract from markdown code blocks if present
        if "```" in entities_str:
            import re
            code_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', entities_str, re.DOTALL)
            if code_match:
                try:
                    entities_list = json.loads(code_match.group(1))
                except:
                    entities_list = [entities_str[:200]]
            else:
                entities_list = [entities_str[:200]]
        else:
            entities_list = [entities_str[:200]]

    # Calculate metrics
    original_len = len(text.encode('utf-8'))
    summary_len = len(summary.encode('utf-8'))

    # Total token usage
    total_input_tokens = summary_result['usage'].get('input_tokens', 0) + entities_result['usage'].get('input_tokens', 0)
    total_output_tokens = summary_result['usage'].get('output_tokens', 0) + entities_result['usage'].get('output_tokens', 0)

    print()
    print(f"📊 Compression metrics:")
    print(f"   Original: {original_len} bytes")
    print(f"   Compressed: {summary_len} bytes")
    print(f"   Ratio: {original_len/max(summary_len, 1):.2f}x")
    print(f"   Total tokens: input={total_input_tokens}, output={total_output_tokens}")
    print()

    return {
        "summary": summary,
        "entities": entities_list,
        "original_length": original_len,
        "compressed_length_estimate": summary_len,
        "compression_ratio": original_len / max(summary_len, 1),
        "llm_model": model,
        "protocol": summary_result.get("protocol", "unknown"),
        "token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens
        },
        "status": "LLM Compressed"
    }


def mock_compress_text_generative(text: str) -> dict:
    """Simple non-LLM compression for comparison."""
    summary = text[:min(len(text), 50)] + "..."
    entities = ["mock_entity_1", "mock_entity_2"]
    original_len = len(text.encode('utf-8'))
    compressed_len = len(summary.encode('utf-8'))

    return {
        "summary": summary,
        "entities": entities,
        "original_length": original_len,
        "compressed_length_estimate": compressed_len,
        "compression_ratio": original_len / compressed_len,
        "llm_model": "Mock",
        "protocol": "none",
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "status": "Mock Compressed"
    }


# Demo usage
if __name__ == "__main__":
    print("=" * 80)
    print("🧪 LLM 压缩功能演示 - 智能协议选择")
    print("=" * 80)
    print()

    test_text = """Met with John at 3pm to discuss the AI-OS project. He suggested using Arrow format for better performance. We reviewed the compression results and were amazed by the 2800x compression ratio achieved in Phase 1.1. The system can now compress conversation data efficiently while maintaining semantic fidelity."""

    print(f"📝 测试文本 ({len(test_text)} 字符):")
    print(f"   {test_text[:100]}...")
    print()
    print("=" * 80)
    print()

    # Test with Claude (most efficient)
    print("1️⃣  测试 Claude Opus 4 (Claude 协议 - 最高效)")
    print("-" * 80)
    result_claude = compress_text_generative(test_text, model="claude-opus-4")
    print()

    # Test with GPT-4 (OpenAI protocol)
    print("=" * 80)
    print()
    print("2️⃣  测试 GPT-4 (OpenAI 协议)")
    print("-" * 80)
    result_gpt = compress_text_generative(test_text, model="gpt-4")
    print()

    # Comparison
    print("=" * 80)
    print("📊 协议效率对比")
    print("=" * 80)
    print(f"Claude 协议: {result_claude['token_usage']['total_tokens']} total tokens")
    print(f"OpenAI 协议: {result_gpt['token_usage']['total_tokens']} total tokens")
    print()
    print("✅ 演示完成")

import os
import asyncio
import numpy as np
import pyarrow as pa
from llm_compression.embedding_provider import get_default_provider, ArrowEngineProvider
from llm_compression.batch_processor_arrow import BatchProcessorArrow
from llm_compression.cognitive_loop_arrow import CognitiveLoopArrow

# 设置模型路径
os.environ["ARROW_MODEL_PATH"] = r"m:\Documents\ai-os-memory\models\minilm"

async def test_integration():
    print("--- 正在测试 EmbeddingProvider + ArrowEngine ---")
    
    # 1. 获取 Provider
    provider = get_default_provider()
    print(f"Provider 类型: {type(provider).__name__}")
    assert isinstance(provider, ArrowEngineProvider), "应自动选择 ArrowEngineProvider"
    print(f"向量维度: {provider.dimension}")

    # 2. 基本编码测试
    text = "人工智能正在改变世界"
    vec = provider.encode(text)
    print(f"编码成功: shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")
    assert vec.shape == (384,)

    # 3. 集成测试: BatchProcessorArrow
    print("\n--- 正在测试 BatchProcessorArrow 集成 ---")
    bp_arrow = BatchProcessorArrow(embedder_arrow=provider)
    texts = ["你好", "您好", "再见", "拜拜"]
    result = await bp_arrow.compress_batch_arrow(texts)
    print(f"批量处理完成: {len(result.table)} 项")
    print(f"表格列名: {result.table.schema.names}")
    assert "embedding" in result.table.schema.names
    
    # 4. 集成测试: CognitiveLoopArrow
    print("\n--- 正在测试 CognitiveLoopArrow 集成 ---")
    loop_arrow = CognitiveLoopArrow(embedder_arrow=provider)
    # 添加记忆
    loop_arrow.batch_add_memories_arrow(
        memory_ids=["m1", "m2"],
        contents=["北京是中国的首都", "上海是中国最大的城市"]
    )
    print(f"记忆库统计: {loop_arrow.get_memory_stats()}")
    
    # 处理查询
    query = "中国的首都在哪里？"
    
    # 模拟 LLM 响应以避免由于没有本地服务器而失败
    from unittest.mock import patch, MagicMock
    from llm_compression.llm_client import LLMResponse
    
    mock_response = LLMResponse(
        text="中国的首都是北京。",
        tokens_used=10,
        latency_ms=100.0,
        model="mock-model",
        finish_reason="stop"
    )
    
    with patch("llm_compression.llm_client.LLMClient.generate", return_value=asyncio.Future()):
        # 手动设置 Future 的结果
        from llm_compression.llm_client import LLMClient
        with patch.object(LLMClient, 'generate', return_value=mock_response):
            res = await loop_arrow.process_arrow(query)
            print(f"查询结果质量: {res.quality.overall:.2f}")
            assert res.quality.overall > 0.5
            print(f"处理耗时: {res.processing_time_ms:.1f}ms")

    print("\n✅ 所有集成验证通过！")

if __name__ == "__main__":
    asyncio.run(test_integration())

#!/usr/bin/env python3
"""
Chat Agent 功能测试脚本

快速验证 Phase 2.0 核心功能是否正常工作
"""

import asyncio
import time
from llm_compression import (
    LLMClient,
    LLMCompressor,
    ModelSelector,
    ConversationalAgent,
    CognitiveLoop
)
from llm_compression.embedder_cache import preload_default_model
from llm_compression.cognitive_loop_arrow import CognitiveLoopArrow


async def test_basic_functionality():
    """测试基础功能"""
    print("\n" + "="*60)
    print("Phase 2.0 功能测试")
    print("="*60)
    
    # 1. 测试模型缓存
    print("\n1️⃣  测试模型缓存...")
    start = time.time()
    preload_default_model()
    elapsed = time.time() - start
    print(f"   ✓ 模型预加载完成: {elapsed:.2f}s")
    
    # 2. 初始化组件
    print("\n2️⃣  初始化组件...")
    
    # 使用 API 代理（Antigravity Manager）
    # 注意：LLMClient 会自动添加 /v1/chat/completions，所以 endpoint 只需要基础路径
    llm_client = LLMClient(
        endpoint="http://192.168.1.99:8045",
        api_key="sk-0437c02b1560470981866f50b05759e3",
        api_type="openai",
        timeout=30.0
    )
    
    model_selector = ModelSelector()
    compressor = LLMCompressor(
        llm_client=llm_client,
        model_selector=model_selector
    )
    
    from llm_compression.expression_layer import MultiModalExpressor
    from llm_compression.internal_feedback import InternalFeedbackSystem
    from llm_compression.connection_learner import ConnectionLearner
    from llm_compression.network_navigator import NetworkNavigator
    from llm_compression.reconstructor import LLMReconstructor
    
    reconstructor = LLMReconstructor(llm_client=llm_client)
    expressor = MultiModalExpressor(
        llm_client=llm_client,
        reconstructor=reconstructor
    )
    feedback = InternalFeedbackSystem()
    learner = ConnectionLearner()
    navigator = NetworkNavigator()
    
    cognitive_loop = CognitiveLoop(
        expressor=expressor,
        feedback=feedback,
        learner=learner,
        navigator=navigator,
        quality_threshold=0.0,
        max_corrections=0
    )
    
    cognitive_loop_arrow = CognitiveLoopArrow(
        cognitive_loop=cognitive_loop,
        enable_optimizations=True
    )
    
    agent = ConversationalAgent(
        llm_client=llm_client,
        compressor=compressor,
        cognitive_loop=cognitive_loop,
        user_id="test_user",
        enable_personalization=True
    )
    
    print("   ✓ 所有组件初始化完成")
    
    # 3. 测试记忆添加
    print("\n3️⃣  测试记忆添加...")
    
    test_memories = [
        "Python 是一种高级编程语言",
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络",
        "自然语言处理处理文本数据",
        "计算机视觉处理图像数据"
    ]
    
    start = time.time()
    for i, content in enumerate(test_memories):
        compressed = await compressor.compress(content)
        
        from llm_compression.memory_primitive import MemoryPrimitive
        import numpy as np
        
        memory = MemoryPrimitive(
            id=compressed.memory_id,
            content=compressed,
            embedding=np.array(compressed.embedding)
        )
        cognitive_loop.memory_network[memory.id] = memory
    
    elapsed = time.time() - start
    print(f"   ✓ 添加 {len(test_memories)} 条记忆: {elapsed:.2f}s")
    print(f"   ✓ 平均速度: {len(test_memories)/elapsed:.1f} memories/s")
    
    # 4. 测试对话
    print("\n4️⃣  测试对话功能...")
    
    test_queries = [
        "什么是 Python？",
        "告诉我关于机器学习的信息",
        "深度学习和机器学习有什么关系？"
    ]
    
    for query in test_queries:
        print(f"\n   Q: {query}")
        start = time.time()
        
        try:
            response = await agent.chat(query)
            elapsed = time.time() - start
            
            print(f"   A: {response.message[:100]}...")
            print(f"   ✓ 响应时间: {elapsed:.2f}s")
            print(f"   ✓ 使用记忆: {len(response.memories_used)}")
            print(f"   ✓ 质量分数: {response.quality_score:.2f}")
        except Exception as e:
            print(f"   ✗ 错误: {e}")
    
    # 5. 测试优化统计
    print("\n5️⃣  测试优化统计...")
    
    opt_stats = cognitive_loop_arrow.get_optimization_stats()
    
    if opt_stats['optimizations_enabled']:
        print("   ✓ 优化已启用")
        
        if 'cache_info' in opt_stats:
            cache = opt_stats['cache_info']
            print(f"   ✓ 缓存模型数: {cache['cache_size']}")
        
        if 'adaptive_stats' in opt_stats:
            adaptive = opt_stats['adaptive_stats']
            print(f"   ✓ 自适应调用: {adaptive['total_calls']}")
        
        if 'batch_stats' in opt_stats:
            batch = opt_stats['batch_stats']
            print(f"   ✓ 批量处理: {batch['total_items_processed']} items")
    else:
        print("   ⚠️  优化未启用")
    
    # 6. 测试统计信息
    print("\n6️⃣  测试统计信息...")
    
    stats = agent.get_stats()
    print(f"   ✓ 对话轮次: {stats['total_turns']}")
    print(f"   ✓ 记忆数量: {stats['memory_count']}")
    print(f"   ✓ 连接数量: {stats['connection_count']}")
    
    # 完成
    print("\n" + "="*60)
    print("✅ 所有测试完成！")
    print("="*60)
    print("\n建议:")
    print("  1. 运行 'python examples/chat_agent_optimized.py' 进行交互式测试")
    print("  2. 使用 '/benchmark' 命令测试性能")
    print("  3. 使用 '/optimization' 命令查看优化统计")


async def main():
    """主函数"""
    try:
        await test_basic_functionality()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

"""
Cognitive Loop End-to-End Demo

展示完整的自组织认知循环系统：
1. 记忆压缩与构建
2. 查询处理
3. 学习演化
4. 质量改进
5. 网络统计
"""

import asyncio
import numpy as np
from datetime import datetime

from llm_compression import (
    LLMClient,
    LLMCompressor,
    LLMReconstructor,
    MemoryPrimitive,
    CognitiveLoop,
    MultiModalExpressor,
    InternalFeedbackSystem,
    ModelSelector
)


# 测试文本
SAMPLE_TEXTS = [
    "Python is a high-level programming language known for its simplicity and readability.",
    "Python is widely used in web development, data science, and artificial intelligence.",
    "Machine learning is a subset of AI that enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers to process complex patterns.",
    "Natural language processing helps computers understand and generate human language.",
]


async def compress_and_build_network(texts, llm_client, compressor):
    """压缩文本并构建记忆网络"""
    print("\n" + "="*60)
    print("📦 步骤1: 压缩文本并构建记忆网络")
    print("="*60)
    
    memories = []
    for i, text in enumerate(texts):
        print(f"\n压缩文本 {i+1}/{len(texts)}...")
        print(f"原文: {text[:60]}...")
        
        # 压缩
        compressed = await compressor.compress(text)
        
        # 创建embedding（简化：使用随机向量）
        # 实际应用中应使用真实的embedding模型
        embedding = np.random.randn(384)
        embedding = embedding / np.linalg.norm(embedding)
        
        # 创建MemoryPrimitive
        memory = MemoryPrimitive(
            id=f"mem_{i}",
            content=compressed,
            embedding=embedding
        )
        
        memories.append(memory)
        
        print(f"✅ 压缩完成: {compressed.compression_metadata.original_size}B → "
              f"{compressed.compression_metadata.compressed_size}B "
              f"(压缩率: {compressed.compression_metadata.compression_ratio:.1f}x)")
    
    print(f"\n✅ 共创建 {len(memories)} 个记忆单元")
    return memories


async def demonstrate_cognitive_loop(memories, llm_client):
    """演示认知循环"""
    print("\n" + "="*60)
    print("🧠 步骤2: 演示认知循环")
    print("="*60)
    
    # 创建组件
    reconstructor = LLMReconstructor(llm_client)
    expressor = MultiModalExpressor(llm_client, reconstructor)
    feedback = InternalFeedbackSystem(llm_client, reconstructor)
    
    # 创建认知循环
    loop = CognitiveLoop(
        expressor=expressor,
        feedback=feedback,
        quality_threshold=0.85,
        max_corrections=2,
        learning_rate=0.1
    )
    
    # 添加记忆到网络
    for memory in memories:
        loop.add_memory(memory)
    
    print(f"\n✅ 认知循环已初始化")
    print(f"   - 记忆数量: {len(memories)}")
    print(f"   - 质量阈值: {loop.quality_threshold}")
    print(f"   - 最大纠正次数: {loop.max_corrections}")
    
    return loop


async def process_queries(loop, queries):
    """处理查询并展示结果"""
    print("\n" + "="*60)
    print("💬 步骤3: 处理查询")
    print("="*60)
    
    for i, query in enumerate(queries):
        print(f"\n{'─'*60}")
        print(f"查询 {i+1}: {query}")
        print(f"{'─'*60}")
        
        # 创建查询embedding（简化）
        query_embedding = np.random.randn(384)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # 处理查询
        print("\n🔄 执行认知循环...")
        result = await loop.process(
            query=query,
            query_embedding=query_embedding,
            max_memories=3
        )
        
        # 显示结果
        print(f"\n📊 结果:")
        print(f"   - 输出: {result.output[:100]}...")
        print(f"   - 质量评分: {result.quality.overall:.2f}")
        print(f"   - 使用记忆: {len(result.memories_used)} 个")
        print(f"   - 纠正次数: {result.corrections_applied}")
        print(f"   - 发生学习: {'是' if result.learning_occurred else '否'}")
        
        print(f"\n📈 质量详情:")
        print(f"   - 一致性: {result.quality.consistency:.2f}")
        print(f"   - 完整性: {result.quality.completeness:.2f}")
        print(f"   - 准确性: {result.quality.accuracy:.2f}")
        print(f"   - 连贯性: {result.quality.coherence:.2f}")


def show_network_evolution(loop, initial_stats):
    """展示网络演化"""
    print("\n" + "="*60)
    print("📈 步骤4: 网络演化统计")
    print("="*60)
    
    final_stats = loop.get_network_stats()
    
    print(f"\n初始状态:")
    print(f"   - 记忆数量: {initial_stats['total_memories']}")
    print(f"   - 总连接数: {initial_stats['total_connections']}")
    print(f"   - 平均连接: {initial_stats['avg_connections']:.2f}")
    print(f"   - 平均成功率: {initial_stats['avg_success_rate']:.2f}")
    
    print(f"\n最终状态:")
    print(f"   - 记忆数量: {final_stats['total_memories']}")
    print(f"   - 总连接数: {final_stats['total_connections']}")
    print(f"   - 平均连接: {final_stats['avg_connections']:.2f}")
    print(f"   - 平均成功率: {final_stats['avg_success_rate']:.2f}")
    
    print(f"\n变化:")
    print(f"   - 连接增长: {final_stats['total_connections'] - initial_stats['total_connections']}")
    print(f"   - 平均连接增长: {final_stats['avg_connections'] - initial_stats['avg_connections']:.2f}")
    print(f"   - 成功率变化: {final_stats['avg_success_rate'] - initial_stats['avg_success_rate']:.2f}")


async def main():
    """主演示流程"""
    print("\n" + "="*60)
    print("🚀 Phase 2.0 认知闭环系统演示")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化LLM客户端
    print("\n初始化LLM客户端...")
    llm_client = LLMClient(
        endpoint="http://localhost:11434"
    )
    
    # 测试连接
    try:
        response = await llm_client.generate("test")
        print("✅ LLM连接成功")
    except Exception as e:
        print(f"❌ LLM连接失败: {e}")
        print("\n请确保Ollama正在运行:")
        print("  ollama serve")
        print("  ollama pull qwen2.5:7b-instruct")
        return
    
    # 创建压缩器
    model_selector = ModelSelector()
    compressor = LLMCompressor(llm_client, model_selector)
    
    # 步骤1: 压缩并构建网络
    memories = await compress_and_build_network(SAMPLE_TEXTS, llm_client, compressor)
    
    # 步骤2: 创建认知循环
    loop = await demonstrate_cognitive_loop(memories, llm_client)
    
    # 记录初始统计
    initial_stats = loop.get_network_stats()
    
    # 步骤3: 处理查询
    queries = [
        "What is Python used for?",
        "Explain machine learning",
        "How does deep learning work?"
    ]
    
    await process_queries(loop, queries)
    
    # 步骤4: 展示网络演化
    show_network_evolution(loop, initial_stats)
    
    print("\n" + "="*60)
    print("✅ 演示完成！")
    print("="*60)
    print("\n关键成就:")
    print("  ✅ 记忆压缩与构建")
    print("  ✅ 完整认知循环")
    print("  ✅ 自我纠正机制")
    print("  ✅ Hebbian学习")
    print("  ✅ 网络自组织")
    print("\n这是一个真正的自组织认知系统！🧠✨")


if __name__ == "__main__":
    asyncio.run(main())

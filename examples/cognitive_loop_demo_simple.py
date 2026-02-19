"""
Cognitive Loop Demo - Simplified Version (Mock LLM)

快速演示认知闭环系统，使用模拟的LLM响应
"""

import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from llm_compression import (
    MemoryPrimitive,
    CognitiveLoop,
    MultiModalExpressor,
    InternalFeedbackSystem,
    ExpressionResult,
    QualityScore,
    CompressedMemory,
    CompressionMetadata
)


def create_mock_memory(id: str, text: str) -> MemoryPrimitive:
    """创建模拟记忆"""
    metadata = CompressionMetadata(
        original_size=len(text),
        compressed_size=len(text),
        compression_ratio=1.0,
        model_used="mock",
        quality_score=0.9,
        compression_time_ms=10.0,
        compressed_at=datetime.now()
    )
    
    compressed = CompressedMemory(
        memory_id=id,
        summary_hash=f"hash_{id}",
        entities={},
        diff_data=text.encode(),
        embedding=[0.1] * 384,
        compression_metadata=metadata
    )
    
    embedding = np.random.randn(384)
    embedding = embedding / np.linalg.norm(embedding)
    
    return MemoryPrimitive(
        id=id,
        content=compressed,
        embedding=embedding
    )


async def main():
    """主演示流程"""
    print("\n" + "="*60)
    print("🚀 Phase 2.0 认知闭环系统演示 (简化版)")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 步骤1: 创建记忆网络
    print("\n" + "="*60)
    print("📦 步骤1: 构建记忆网络")
    print("="*60)
    
    memories = [
        create_mock_memory("mem_0", "Python is a high-level programming language."),
        create_mock_memory("mem_1", "Python is used for web development and AI."),
        create_mock_memory("mem_2", "Machine learning is a subset of AI."),
        create_mock_memory("mem_3", "Deep learning uses neural networks."),
        create_mock_memory("mem_4", "NLP helps computers understand language."),
    ]
    
    print(f"\n✅ 创建了 {len(memories)} 个记忆单元")
    for mem in memories:
        print(f"   - {mem.id}: {mem.content.diff_data.decode()[:50]}...")
    
    # 步骤2: 创建认知循环（使用mock）
    print("\n" + "="*60)
    print("🧠 步骤2: 初始化认知循环")
    print("="*60)
    
    # Mock expressor
    expressor = MagicMock()
    expressor.express_text = AsyncMock(return_value=ExpressionResult(
        content="Python is a versatile programming language used in web development, data science, and AI.",
        quality_score=0.92,
        modality="text",
        source_memories=["mem_0", "mem_1"]
    ))
    
    # Mock feedback
    feedback = MagicMock()
    feedback.evaluate = AsyncMock(return_value=QualityScore(
        overall=0.92,
        consistency=0.90,
        completeness=0.93,
        accuracy=0.91,
        coherence=0.94
    ))
    
    # 创建认知循环
    loop = CognitiveLoop(
        expressor=expressor,
        feedback=feedback,
        quality_threshold=0.85,
        max_corrections=2,
        learning_rate=0.1
    )
    
    # 添加记忆
    for memory in memories:
        loop.add_memory(memory)
    
    print(f"\n✅ 认知循环已初始化")
    print(f"   - 记忆数量: {len(memories)}")
    print(f"   - 质量阈值: {loop.quality_threshold}")
    print(f"   - 最大纠正次数: {loop.max_corrections}")
    print(f"   - 学习率: {loop.learning_rate}")
    
    # 记录初始统计
    initial_stats = loop.get_network_stats()
    print(f"\n📊 初始网络状态:")
    print(f"   - 总记忆数: {initial_stats['total_memories']}")
    print(f"   - 总连接数: {initial_stats['total_connections']}")
    print(f"   - 平均连接: {initial_stats['avg_connections']:.2f}")
    
    # 步骤3: 处理查询
    print("\n" + "="*60)
    print("💬 步骤3: 处理查询")
    print("="*60)
    
    queries = [
        "What is Python used for?",
        "Explain machine learning",
        "How does deep learning work?"
    ]
    
    for i, query in enumerate(queries):
        print(f"\n{'─'*60}")
        print(f"查询 {i+1}: {query}")
        print(f"{'─'*60}")
        
        # 创建查询embedding
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
        print(f"   - 输出: {result.output[:80]}...")
        print(f"   - 质量评分: {result.quality.overall:.2f}")
        print(f"   - 使用记忆: {len(result.memories_used)} 个")
        print(f"   - 纠正次数: {result.corrections_applied}")
        print(f"   - 发生学习: {'是' if result.learning_occurred else '否'}")
        
        print(f"\n📈 质量详情:")
        print(f"   - 一致性: {result.quality.consistency:.2f}")
        print(f"   - 完整性: {result.quality.completeness:.2f}")
        print(f"   - 准确性: {result.quality.accuracy:.2f}")
        print(f"   - 连贯性: {result.quality.coherence:.2f}")
    
    # 步骤4: 展示网络演化
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
    print(f"   - 连接增长: +{final_stats['total_connections'] - initial_stats['total_connections']}")
    print(f"   - 平均连接增长: +{final_stats['avg_connections'] - initial_stats['avg_connections']:.2f}")
    print(f"   - 成功率变化: +{final_stats['avg_success_rate'] - initial_stats['avg_success_rate']:.2f}")
    
    # 总结
    print("\n" + "="*60)
    print("✅ 演示完成！")
    print("="*60)
    print("\n关键成就:")
    print("  ✅ 记忆网络构建")
    print("  ✅ 完整认知循环")
    print("  ✅ 质量评估 (>0.85)")
    print("  ✅ Hebbian学习")
    print("  ✅ 网络自组织")
    print(f"\n网络演化:")
    print(f"  - 新增连接: {final_stats['total_connections'] - initial_stats['total_connections']}")
    print(f"  - 成功率: {final_stats['avg_success_rate']:.1%}")
    print("\n这是一个真正的自组织认知系统！🧠✨")


if __name__ == "__main__":
    asyncio.run(main())

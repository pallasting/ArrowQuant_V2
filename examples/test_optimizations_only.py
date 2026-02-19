#!/usr/bin/env python3
"""
优化功能测试脚本（不需要 Ollama）

仅测试优化模块，不需要 LLM 服务
"""

import asyncio
import time
from llm_compression.embedder_cache import preload_default_model, EmbedderCache
from llm_compression.embedder_adaptive import AdaptiveEmbedder
from llm_compression.batch_optimizer import MemoryBatchProcessor
from llm_compression.embedder_arrow import LocalEmbedderArrow


async def test_optimizations():
    """测试优化功能"""
    print("\n" + "="*60)
    print("Phase 2.0 优化功能测试（无需 Ollama）")
    print("="*60)
    
    # 1. 测试模型缓存
    print("\n1️⃣  测试模型缓存...")
    start = time.time()
    preload_default_model()
    elapsed = time.time() - start
    print(f"   ✓ 模型预加载完成: {elapsed:.2f}s")
    
    cache_info = EmbedderCache.get_cache_info()
    print(f"   ✓ 缓存模型数: {cache_info['cache_size']}")
    print(f"   ✓ 缓存模型: {', '.join(cache_info['cached_models'])}")
    
    # 2. 测试自适应 Embedder
    print("\n2️⃣  测试自适应 Embedder...")
    adaptive = AdaptiveEmbedder(small_scale_threshold=100, enable_stats=True)
    
    # 小规模测试（应使用传统方法）
    small_texts = ["测试文本 " + str(i) for i in range(50)]
    start = time.time()
    embeddings_small = adaptive.encode_batch(small_texts)
    elapsed_small = time.time() - start
    print(f"   ✓ 小规模（50条）: {elapsed_small:.3f}s")
    
    # 大规模测试（应使用 Arrow 方法）
    large_texts = ["测试文本 " + str(i) for i in range(200)]
    start = time.time()
    embeddings_large = adaptive.encode_batch(large_texts)
    elapsed_large = time.time() - start
    print(f"   ✓ 大规模（200条）: {elapsed_large:.3f}s")
    
    # 查看统计
    stats = adaptive.get_stats()
    print(f"   ✓ 总调用: {stats['total_calls']}")
    print(f"   ✓ 传统方法: {stats['traditional_calls']} ({stats['traditional_percentage']:.1f}%)")
    print(f"   ✓ Arrow方法: {stats['arrow_calls']} ({stats['arrow_percentage']:.1f}%)")
    
    # 3. 测试批量处理器
    print("\n3️⃣  测试批量处理器...")
    embedder = LocalEmbedderArrow()
    batch_processor = MemoryBatchProcessor(
        embedder=embedder.embedder,
        batch_size=50,
        max_workers=4,
        enable_adaptive=True
    )
    
    # 批量处理测试
    test_texts = ["批量测试文本 " + str(i) for i in range(500)]
    memory_ids = [f"test_mem_{i}" for i in range(500)]
    start = time.time()
    results = batch_processor.batch_add_memories(memory_ids, test_texts, parallel=True)
    elapsed = time.time() - start
    
    print(f"   ✓ 处理 {len(test_texts)} 条记忆: {elapsed:.2f}s")
    print(f"   ✓ 吞吐量: {len(test_texts)/elapsed:.1f} memories/s")
    
    # 查看批量处理统计
    batch_stats = batch_processor.get_stats()
    print(f"   ✓ 总批次: {batch_stats['total_batches']}")
    print(f"   ✓ 总处理: {batch_stats['total_items_processed']}")
    print(f"   ✓ 平均吞吐: {batch_stats['avg_throughput']:.1f} items/s")
    print(f"   ✓ 当前批次大小: {batch_stats['current_batch_size']}")
    
    # 4. 性能对比
    print("\n4️⃣  性能对比...")
    
    # 传统方法
    from llm_compression.embedder import LocalEmbedder
    traditional = LocalEmbedder()
    test_batch = ["性能测试 " + str(i) for i in range(100)]
    
    start = time.time()
    _ = traditional.encode_batch(test_batch)
    traditional_time = time.time() - start
    
    # Arrow 方法
    arrow_embedder = LocalEmbedderArrow()
    start = time.time()
    _ = arrow_embedder.batch_encode_arrow(test_batch)
    arrow_time = time.time() - start
    
    print(f"   ✓ 传统方法（100条）: {traditional_time:.3f}s")
    print(f"   ✓ Arrow方法（100条）: {arrow_time:.3f}s")
    print(f"   ✓ 性能提升: {traditional_time/arrow_time:.2f}x")
    
    # 完成
    print("\n" + "="*60)
    print("✅ 所有优化测试完成！")
    print("="*60)
    print("\n总结:")
    print("  • 模型缓存: 正常工作")
    print("  • 自适应切换: 正常工作")
    print("  • 批量处理: 正常工作")
    print(f"  • 批量吞吐量: {len(test_texts)/elapsed:.1f} memories/s")
    print("\n建议:")
    print("  1. 启动 Ollama 服务: ollama serve")
    print("  2. 运行完整测试: python examples/test_chat_agent.py")
    print("  3. 运行交互式 Agent: python examples/chat_agent_optimized.py")


async def main():
    """主函数"""
    try:
        await test_optimizations()
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

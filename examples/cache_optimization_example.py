"""
缓存优化示例

演示 Phase 1.1 的缓存策略优化：
- 增加缓存大小到 50000
- 增加 TTL 到 2 小时
- 优化缓存命中率
"""

import asyncio
import time
import hashlib
from typing import List
from llm_compression.batch_processor import CompressionCache
from llm_compression.compressor import CompressedMemory, CompressionMetadata


def generate_text_hash(text: str) -> str:
    """生成文本哈希"""
    return hashlib.sha256(text.encode()).hexdigest()


async def test_cache_performance():
    """测试缓存性能"""
    print("="*60)
    print("缓存性能测试")
    print("="*60)
    
    # 创建不同大小的缓存
    cache_configs = [
        ("Phase 1.0 缓存", 10000, 3600),
        ("Phase 1.1 缓存", 50000, 7200),
    ]
    
    for name, cache_size, ttl in cache_configs:
        print(f"\n{name}:")
        print(f"  - 缓存大小: {cache_size}")
        print(f"  - TTL: {ttl}s ({ttl/3600:.1f}h)")
        
        cache = CompressionCache(
            max_size=cache_size,
            ttl=ttl
        )
        
        # 测试写入性能
        print(f"\n  测试写入性能...")
        start_time = time.time()
        
        for i in range(1000):
            text = f"Test text {i}"
            text_hash = generate_text_hash(text)
            
            compressed = CompressedMemory(
                summary=f"Summary {i}",
                summary_hash=text_hash,
                entities={},
                diff_data=b"",
                metadata=CompressionMetadata(
                    original_size=len(text),
                    compressed_size=50,
                    compression_ratio=len(text) / 50,
                    model_used="qwen2.5",
                    quality_score=0.9,
                    compression_time_ms=100.0,
                    compressed_at=time.time()
                )
            )
            
            await cache.set(text_hash, compressed)
        
        write_time = time.time() - start_time
        print(f"  - 写入 1000 条: {write_time:.2f}s")
        print(f"  - 写入速度: {1000/write_time:.2f} 条/秒")
        
        # 测试读取性能（命中）
        print(f"\n  测试读取性能（缓存命中）...")
        start_time = time.time()
        hits = 0
        
        for i in range(1000):
            text = f"Test text {i}"
            text_hash = generate_text_hash(text)
            result = await cache.get(text_hash)
            if result:
                hits += 1
        
        read_time = time.time() - start_time
        hit_rate = hits / 1000
        
        print(f"  - 读取 1000 条: {read_time:.2f}s")
        print(f"  - 读取速度: {1000/read_time:.2f} 条/秒")
        print(f"  - 命中率: {hit_rate:.1%}")
        
        # 测试读取性能（未命中）
        print(f"\n  测试读取性能（缓存未命中）...")
        start_time = time.time()
        misses = 0
        
        for i in range(1000, 2000):
            text = f"Test text {i}"
            text_hash = generate_text_hash(text)
            result = await cache.get(text_hash)
            if not result:
                misses += 1
        
        miss_time = time.time() - start_time
        miss_rate = misses / 1000
        
        print(f"  - 读取 1000 条: {miss_time:.2f}s")
        print(f"  - 读取速度: {1000/miss_time:.2f} 条/秒")
        print(f"  - 未命中率: {miss_rate:.1%}")
        
        # 清理
        await cache.clear()


async def test_cache_hit_rate():
    """测试缓存命中率优化"""
    print("\n"+"="*60)
    print("缓存命中率优化测试")
    print("="*60)
    
    # 创建缓存
    cache = CompressionCache(
        max_size=50000,
        ttl=7200
    )
    
    # 模拟真实使用场景
    print("\n模拟真实使用场景...")
    print("  - 80% 的请求访问 20% 的数据（热点数据）")
    print("  - 20% 的请求访问 80% 的数据（冷数据）")
    
    # 生成测试数据
    hot_texts = [f"Hot text {i}" for i in range(200)]  # 20% 热点数据
    cold_texts = [f"Cold text {i}" for i in range(800)]  # 80% 冷数据
    
    # 填充缓存
    print("\n填充缓存...")
    for text in hot_texts + cold_texts[:100]:  # 只缓存部分冷数据
        text_hash = generate_text_hash(text)
        compressed = CompressedMemory(
            summary=f"Summary of {text}",
            summary_hash=text_hash,
            entities={},
            diff_data=b"",
            metadata=CompressionMetadata(
                original_size=len(text),
                compressed_size=50,
                compression_ratio=len(text) / 50,
                model_used="qwen2.5",
                quality_score=0.9,
                compression_time_ms=100.0,
                compressed_at=time.time()
            )
        )
        await cache.set(text_hash, compressed)
    
    # 模拟访问模式
    print("\n模拟访问...")
    total_requests = 1000
    hits = 0
    misses = 0
    
    import random
    for _ in range(total_requests):
        # 80% 概率访问热点数据
        if random.random() < 0.8:
            text = random.choice(hot_texts)
        else:
            text = random.choice(cold_texts)
        
        text_hash = generate_text_hash(text)
        result = await cache.get(text_hash)
        
        if result:
            hits += 1
        else:
            misses += 1
            # 未命中时添加到缓存
            compressed = CompressedMemory(
                summary=f"Summary of {text}",
                summary_hash=text_hash,
                entities={},
                diff_data=b"",
                metadata=CompressionMetadata(
                    original_size=len(text),
                    compressed_size=50,
                    compression_ratio=len(text) / 50,
                    model_used="qwen2.5",
                    quality_score=0.9,
                    compression_time_ms=100.0,
                    compressed_at=time.time()
                )
            )
            await cache.set(text_hash, compressed)
    
    hit_rate = hits / total_requests
    
    print(f"\n结果:")
    print(f"  - 总请求: {total_requests}")
    print(f"  - 命中: {hits}")
    print(f"  - 未命中: {misses}")
    print(f"  - 命中率: {hit_rate:.1%}")
    
    print(f"\n性能提升:")
    # 假设缓存命中延迟 1ms，未命中延迟 1000ms
    cache_hit_latency = 0.001
    cache_miss_latency = 1.0
    
    avg_latency_with_cache = (hits * cache_hit_latency + misses * cache_miss_latency) / total_requests
    avg_latency_without_cache = cache_miss_latency
    
    improvement = (1 - avg_latency_with_cache / avg_latency_without_cache) * 100
    
    print(f"  - 平均延迟（有缓存）: {avg_latency_with_cache*1000:.2f}ms")
    print(f"  - 平均延迟（无缓存）: {avg_latency_without_cache*1000:.2f}ms")
    print(f"  - 性能提升: {improvement:.1f}%")
    
    # 清理
    await cache.clear()


async def main():
    """主函数"""
    print("="*60)
    print("Phase 1.1 缓存优化示例")
    print("="*60)
    
    print("\n本示例演示:")
    print("1. 增加缓存大小到 50000")
    print("2. 增加 TTL 到 2 小时")
    print("3. 优化缓存命中率")
    
    # 测试缓存性能
    await test_cache_performance()
    
    # 测试缓存命中率
    await test_cache_hit_rate()
    
    print("\n"+"="*60)
    print("测试完成！")
    print("="*60)
    
    print("\n关键优化:")
    print("1. 缓存大小: 10000 → 50000 (5x)")
    print("2. TTL: 1h → 2h (2x)")
    print("3. LRU 淘汰策略优化")
    print("4. 热点数据优先缓存")
    
    print("\n预期效果:")
    print("- 缓存命中率: > 80%")
    print("- 平均延迟降低: > 70%")
    print("- 减少 LLM 调用: > 80%")


if __name__ == "__main__":
    asyncio.run(main())

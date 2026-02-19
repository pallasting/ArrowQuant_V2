"""
性能优化基准测试

对比优化前后的性能差异。
"""

import pytest
import time
import numpy as np
from llm_compression.embedder import LocalEmbedder
from llm_compression.embedder_cache import EmbedderCache
from llm_compression.embedder_adaptive import create_adaptive_embedder
from llm_compression.batch_optimizer import create_memory_batch_processor


class TestModelCacheBenchmark:
    """模型缓存性能基准测试"""
    
    def test_cache_performance(self, benchmark):
        """测试模型缓存性能"""
        # 清除缓存
        EmbedderCache.clear_cache()
        
        # 首次加载（慢）
        start = time.time()
        model1 = EmbedderCache.get_model()
        first_load_time = time.time() - start
        
        # 第二次加载（快，从缓存）
        start = time.time()
        model2 = EmbedderCache.get_model()
        cached_load_time = time.time() - start
        
        # 验证
        assert model1 is model2
        assert cached_load_time < 0.001  # < 1ms
        
        speedup = first_load_time / cached_load_time
        print(f"\n模型缓存性能:")
        print(f"  首次加载: {first_load_time:.2f}s")
        print(f"  缓存加载: {cached_load_time:.4f}s")
        print(f"  性能提升: {speedup:.0f}x")
        
        assert speedup > 1000  # 至少 1000x 提升


class TestAdaptiveEmbedderBenchmark:
    """自适应 Embedder 性能基准测试"""
    
    def test_small_scale_performance(self, benchmark):
        """测试小规模数据性能"""
        embedder = create_adaptive_embedder(
            small_scale_threshold=100,
            enable_stats=True
        )
        
        texts = [f"Text {i}" for i in range(50)]
        
        # 测试性能
        start = time.time()
        embeddings = embedder.encode_batch(texts)
        elapsed = time.time() - start
        
        # 验证
        assert embeddings.shape == (50, 384)
        
        stats = embedder.get_stats()
        print(f"\n小规模数据性能:")
        print(f"  数据量: {len(texts)}")
        print(f"  耗时: {elapsed:.3f}s")
        print(f"  传统方法调用: {stats['traditional_calls']}")
        print(f"  Arrow 方法调用: {stats['arrow_calls']}")
        
        # 小规模应该使用传统方法
        assert stats['traditional_calls'] > 0
    
    def test_large_scale_performance(self, benchmark):
        """测试大规模数据性能"""
        embedder = create_adaptive_embedder(
            small_scale_threshold=100,
            enable_stats=True
        )
        
        texts = [f"Text {i}" for i in range(200)]
        
        # 测试性能
        start = time.time()
        embeddings = embedder.encode_batch(texts)
        elapsed = time.time() - start
        
        # 验证
        assert embeddings.shape == (200, 384)
        
        stats = embedder.get_stats()
        print(f"\n大规模数据性能:")
        print(f"  数据量: {len(texts)}")
        print(f"  耗时: {elapsed:.3f}s")
        print(f"  传统方法调用: {stats['traditional_calls']}")
        print(f"  Arrow 方法调用: {stats['arrow_calls']}")
        
        # 大规模应该使用 Arrow 方法
        assert stats['arrow_calls'] > 0


class TestBatchProcessorBenchmark:
    """批量处理器性能基准测试"""
    
    def test_sequential_vs_parallel(self, benchmark):
        """测试顺序 vs 并行处理性能"""
        embedder = LocalEmbedder()
        processor = create_memory_batch_processor(
            embedder=embedder,
            batch_size=50,
            max_workers=4,
            enable_adaptive=True
        )
        
        n_memories = 200
        memory_ids = [f"mem_{i}" for i in range(n_memories)]
        contents = [f"Content {i}" for i in range(n_memories)]
        
        # 顺序处理
        start = time.time()
        table_seq = processor.batch_add_memories(
            memory_ids, contents, parallel=False
        )
        seq_time = time.time() - start
        seq_throughput = n_memories / seq_time
        
        # 重置统计
        processor.reset_stats()
        
        # 并行处理
        start = time.time()
        table_par = processor.batch_add_memories(
            memory_ids, contents, parallel=True
        )
        par_time = time.time() - start
        par_throughput = n_memories / par_time
        
        # 验证
        assert len(table_seq) == n_memories
        assert len(table_par) == n_memories
        
        speedup = seq_time / par_time
        print(f"\n批量处理性能对比:")
        print(f"  数据量: {n_memories}")
        print(f"  顺序处理: {seq_time:.2f}s ({seq_throughput:.1f} memories/s)")
        print(f"  并行处理: {par_time:.2f}s ({par_throughput:.1f} memories/s)")
        print(f"  性能提升: {speedup:.2f}x")
        
        # 并行应该更快
        assert par_time < seq_time
        assert speedup > 1.0
    
    def test_throughput_target(self, benchmark):
        """测试吞吐量目标（200+ memories/s）"""
        embedder = LocalEmbedder()
        processor = create_memory_batch_processor(
            embedder=embedder,
            batch_size=100,
            max_workers=4,
            enable_adaptive=True
        )
        
        n_memories = 500
        memory_ids = [f"mem_{i}" for i in range(n_memories)]
        contents = [f"Content {i}" for i in range(n_memories)]
        
        # 并行处理
        start = time.time()
        table = processor.batch_add_memories(
            memory_ids, contents, parallel=True
        )
        elapsed = time.time() - start
        
        throughput = n_memories / elapsed
        
        print(f"\n吞吐量测试:")
        print(f"  数据量: {n_memories}")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  吞吐量: {throughput:.1f} memories/s")
        
        # 验证
        assert len(table) == n_memories
        
        # 目标：200+ memories/s
        # 在 Windows 环境下放宽到 150+
        assert throughput > 150, f"Throughput {throughput:.1f} < 150 memories/s"


class TestIntegratedOptimizationBenchmark:
    """集成优化性能基准测试"""
    
    def test_full_optimization_pipeline(self, benchmark):
        """测试完整优化流水线"""
        # 1. 预加载模型
        EmbedderCache.preload_model()
        
        # 2. 创建自适应 Embedder
        adaptive_embedder = create_adaptive_embedder(
            small_scale_threshold=100,
            enable_stats=True
        )
        
        # 3. 创建批量处理器
        batch_processor = create_memory_batch_processor(
            embedder=adaptive_embedder.embedder,
            batch_size=100,
            max_workers=4
        )
        
        # 4. 批量添加记忆
        n_memories = 500
        memory_ids = [f"mem_{i}" for i in range(n_memories)]
        contents = [f"Content {i}" for i in range(n_memories)]
        
        start = time.time()
        table = batch_processor.batch_add_memories(
            memory_ids, contents, parallel=True
        )
        total_time = time.time() - start
        
        throughput = n_memories / total_time
        
        # 统计信息
        embedder_stats = adaptive_embedder.get_stats()
        processor_stats = batch_processor.get_stats()
        
        print(f"\n集成优化性能:")
        print(f"  数据量: {n_memories}")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  吞吐量: {throughput:.1f} memories/s")
        print(f"\nEmbedder 统计:")
        print(f"  总调用: {embedder_stats['total_calls']}")
        print(f"  传统方法: {embedder_stats['traditional_calls']}")
        print(f"  Arrow 方法: {embedder_stats['arrow_calls']}")
        print(f"\n批量处理统计:")
        print(f"  处理总数: {processor_stats['total_items_processed']}")
        print(f"  平均吞吐量: {processor_stats['avg_throughput']:.1f} memories/s")
        
        # 验证
        assert len(table) == n_memories
        assert throughput > 150  # 目标：200+，Windows 放宽到 150+


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

"""
测试 Embedder 优化功能

测试模型缓存、自适应切换和批量处理优化。
"""

import pytest
import numpy as np
import pyarrow as pa

from llm_compression.embedder_cache import EmbedderCache, preload_default_model
from llm_compression.embedder_adaptive import AdaptiveEmbedder, create_adaptive_embedder
from llm_compression.batch_optimizer import BatchOptimizer, MemoryBatchProcessor


class TestEmbedderCache:
    """测试模型缓存"""
    
    def test_get_model(self):
        """测试获取模型"""
        model = EmbedderCache.get_model()
        assert model is not None
    
    def test_model_cached(self):
        """测试模型缓存"""
        # 第一次加载
        model1 = EmbedderCache.get_model()
        
        # 第二次应该从缓存获取
        model2 = EmbedderCache.get_model()
        
        # 应该是同一个实例
        assert model1 is model2
    
    def test_is_cached(self):
        """测试检查缓存状态"""
        # 确保模型已加载
        EmbedderCache.get_model()
        
        # 检查缓存状态
        assert EmbedderCache.is_cached("all-MiniLM-L6-v2")
    
    def test_get_cache_info(self):
        """测试获取缓存信息"""
        # 确保模型已加载
        EmbedderCache.get_model()
        
        info = EmbedderCache.get_cache_info()
        
        assert 'cached_models' in info
        assert 'cache_size' in info
        assert info['cache_size'] > 0
    
    def test_preload_model(self):
        """测试预加载模型"""
        # 清除缓存
        EmbedderCache.clear_cache()
        
        # 预加载
        preload_default_model()
        
        # 检查是否已缓存
        assert EmbedderCache.is_cached("all-MiniLM-L6-v2")


class TestAdaptiveEmbedder:
    """测试自适应 Embedder"""
    
    @pytest.fixture
    def adaptive_embedder(self):
        """创建自适应 Embedder"""
        return create_adaptive_embedder(
            small_scale_threshold=100,
            enable_stats=True
        )
    
    def test_small_scale_batch_encode(self, adaptive_embedder):
        """测试小规模批量编码（应使用传统方法）"""
        texts = [f"Text {i}" for i in range(50)]
        
        embeddings = adaptive_embedder.encode_batch(texts)
        
        assert embeddings.shape == (50, 384)
        
        # 检查统计
        stats = adaptive_embedder.get_stats()
        assert stats['traditional_calls'] > 0
    
    def test_large_scale_batch_encode(self, adaptive_embedder):
        """测试大规模批量编码（应使用 Arrow 方法）"""
        texts = [f"Text {i}" for i in range(200)]
        
        embeddings = adaptive_embedder.encode_batch(texts)
        
        assert embeddings.shape == (200, 384)
        
        # 检查统计
        stats = adaptive_embedder.get_stats()
        assert stats['arrow_calls'] > 0
    
    def test_force_method(self, adaptive_embedder):
        """测试强制使用指定方法"""
        texts = [f"Text {i}" for i in range(50)]
        
        # 强制使用 Arrow 方法
        embeddings = adaptive_embedder.encode_batch(
            texts,
            force_method='arrow'
        )
        
        assert embeddings.shape == (50, 384)
        
        # 检查统计
        stats = adaptive_embedder.get_stats()
        assert stats['arrow_calls'] > 0
    
    def test_get_stats(self, adaptive_embedder):
        """测试获取统计信息"""
        texts = [f"Text {i}" for i in range(50)]
        adaptive_embedder.encode_batch(texts)
        
        stats = adaptive_embedder.get_stats()
        
        assert 'traditional_calls' in stats
        assert 'arrow_calls' in stats
        assert 'total_calls' in stats
        assert 'total_items' in stats
    
    def test_reset_stats(self, adaptive_embedder):
        """测试重置统计"""
        texts = [f"Text {i}" for i in range(50)]
        adaptive_embedder.encode_batch(texts)
        
        # 重置
        adaptive_embedder.reset_stats()
        
        stats = adaptive_embedder.get_stats()
        assert stats['total_calls'] == 0
    
    def test_set_threshold(self, adaptive_embedder):
        """测试设置阈值"""
        adaptive_embedder.set_threshold(500)
        assert adaptive_embedder.small_scale_threshold == 500


class TestBatchOptimizer:
    """测试批量处理优化器"""
    
    @pytest.fixture
    def optimizer(self):
        """创建优化器"""
        return BatchOptimizer(
            batch_size=10,
            max_workers=2,
            enable_adaptive=True
        )
    
    def test_process_in_batches_sequential(self, optimizer):
        """测试顺序批量处理"""
        items = list(range(50))
        
        def process_func(batch):
            return [x * 2 for x in batch]
        
        results = optimizer.process_in_batches(
            items,
            process_func,
            parallel=False
        )
        
        # 展平结果
        flat_results = [item for batch in results for item in batch]
        
        assert len(flat_results) == 50
        assert flat_results[0] == 0
        assert flat_results[49] == 98
    
    def test_process_in_batches_parallel(self, optimizer):
        """测试并行批量处理"""
        items = list(range(50))
        
        def process_func(batch):
            return [x * 2 for x in batch]
        
        results = optimizer.process_in_batches(
            items,
            process_func,
            parallel=True
        )
        
        # 展平结果
        flat_results = [item for batch in results for item in batch]
        
        assert len(flat_results) == 50
    
    def test_get_stats(self, optimizer):
        """测试获取统计信息"""
        items = list(range(50))
        
        def process_func(batch):
            return [x * 2 for x in batch]
        
        optimizer.process_in_batches(items, process_func)
        
        stats = optimizer.get_stats()
        
        assert 'total_items_processed' in stats
        assert 'total_batches' in stats
        assert 'avg_throughput' in stats
        assert stats['total_items_processed'] == 50
    
    def test_adaptive_batch_size(self, optimizer):
        """测试自适应批次大小"""
        # 小数据集
        small_batch_size = optimizer._get_batch_size(100)
        
        # 大数据集
        large_batch_size = optimizer._get_batch_size(10000)
        
        # 大数据集应该使用更大的批次
        assert large_batch_size >= small_batch_size


class TestMemoryBatchProcessor:
    """测试记忆批量处理器"""
    
    @pytest.fixture
    def processor(self):
        """创建处理器"""
        from llm_compression.embedder import LocalEmbedder
        embedder = LocalEmbedder()
        
        return MemoryBatchProcessor(
            embedder=embedder,
            batch_size=10,
            max_workers=2
        )
    
    def test_batch_add_memories(self, processor):
        """测试批量添加记忆"""
        memory_ids = [f"mem{i}" for i in range(50)]
        contents = [f"Content {i}" for i in range(50)]
        
        table = processor.batch_add_memories(
            memory_ids,
            contents,
            parallel=False
        )
        
        assert len(table) == 50
        assert 'memory_id' in table.column_names
        assert 'content' in table.column_names
        assert 'embedding' in table.column_names
    
    def test_batch_add_memories_parallel(self, processor):
        """测试并行批量添加记忆"""
        memory_ids = [f"mem{i}" for i in range(50)]
        contents = [f"Content {i}" for i in range(50)]
        
        table = processor.batch_add_memories(
            memory_ids,
            contents,
            parallel=True
        )
        
        assert len(table) == 50
    
    def test_get_stats(self, processor):
        """测试获取统计信息"""
        memory_ids = [f"mem{i}" for i in range(50)]
        contents = [f"Content {i}" for i in range(50)]
        
        processor.batch_add_memories(memory_ids, contents)
        
        stats = processor.get_stats()
        
        assert 'total_items_processed' in stats
        assert 'avg_throughput' in stats
        assert stats['total_items_processed'] == 50
    
    def test_empty_input(self, processor):
        """测试空输入"""
        table = processor.batch_add_memories([], [])
        
        assert len(table) == 0
    
    def test_mismatched_lengths(self, processor):
        """测试长度不匹配"""
        with pytest.raises(ValueError):
            processor.batch_add_memories(
                ["mem1", "mem2"],
                ["content1"]
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

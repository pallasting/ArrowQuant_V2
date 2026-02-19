"""
批量处理优化

优化批量添加记忆的性能，提升吞吐量。

Features:
- 并行批量处理
- 动态批次大小调整
- 内存管理
- 进度监控

Optimization: 提升吞吐量至 200+ memories/s
"""

import logging
from typing import List, Optional, Callable
import numpy as np
import pyarrow as pa
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from llm_compression.embedding_provider import EmbeddingProvider, get_default_provider

logger = logging.getLogger(__name__)


class BatchOptimizer:
    """
    批量处理优化器
    
    提供高效的批量处理策略，优化吞吐量。
    """
    
    # 默认配置
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_MAX_WORKERS = 4
    ADAPTIVE_BATCH_SIZE_MIN = 50
    ADAPTIVE_BATCH_SIZE_MAX = 500
    
    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_workers: int = DEFAULT_MAX_WORKERS,
        enable_adaptive: bool = True,
        enable_progress: bool = False
    ):
        """
        初始化批量处理优化器
        
        Args:
            batch_size: 批次大小
            max_workers: 最大并行工作线程数
            enable_adaptive: 是否启用自适应批次大小
            enable_progress: 是否显示进度
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_adaptive = enable_adaptive
        self.enable_progress = enable_progress
        
        # 统计信息
        self.stats = {
            'total_items_processed': 0,
            'total_batches': 0,
            'total_time_seconds': 0.0,
            'avg_throughput': 0.0,
        }
    
    def process_in_batches(
        self,
        items: List[any],
        process_func: Callable,
        parallel: bool = True,
        **kwargs
    ) -> List[any]:
        """
        批量处理数据
        
        Args:
            items: 待处理的数据项列表
            process_func: 处理函数（接受批次数据）
            parallel: 是否并行处理
            **kwargs: 传递给处理函数的额外参数
        
        Returns:
            处理结果列表
        """
        if not items:
            return []
        
        n_items = len(items)
        start_time = time.time()
        
        # 确定批次大小
        batch_size = self._get_batch_size(n_items)
        
        # 分批
        batches = self._create_batches(items, batch_size)
        n_batches = len(batches)
        
        logger.info(
            f"Processing {n_items} items in {n_batches} batches "
            f"(batch_size={batch_size}, parallel={parallel})"
        )
        
        # 处理批次
        if parallel and n_batches > 1:
            results = self._process_parallel(batches, process_func, **kwargs)
        else:
            results = self._process_sequential(batches, process_func, **kwargs)
        
        # 更新统计
        elapsed_time = time.time() - start_time
        self._update_stats(n_items, n_batches, elapsed_time)
        
        logger.info(
            f"Processed {n_items} items in {elapsed_time:.2f}s "
            f"({n_items / elapsed_time:.1f} items/s)"
        )
        
        return results
    
    def _create_batches(self, items: List[any], batch_size: int) -> List[List[any]]:
        """
        将数据分批
        
        Args:
            items: 数据项列表
            batch_size: 批次大小
        
        Returns:
            批次列表
        """
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def _process_sequential(
        self,
        batches: List[List[any]],
        process_func: Callable,
        **kwargs
    ) -> List[any]:
        """
        顺序处理批次
        
        Args:
            batches: 批次列表
            process_func: 处理函数
            **kwargs: 额外参数
        
        Returns:
            处理结果列表
        """
        results = []
        
        for i, batch in enumerate(batches):
            if self.enable_progress:
                logger.info(f"Processing batch {i+1}/{len(batches)}...")
            
            batch_result = process_func(batch, **kwargs)
            results.append(batch_result)
        
        return results
    
    def _process_parallel(
        self,
        batches: List[List[any]],
        process_func: Callable,
        **kwargs
    ) -> List[any]:
        """
        并行处理批次
        
        Args:
            batches: 批次列表
            process_func: 处理函数
            **kwargs: 额外参数
        
        Returns:
            处理结果列表
        """
        results = [None] * len(batches)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(process_func, batch, **kwargs): i
                for i, batch in enumerate(batches)
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    completed += 1
                    
                    if self.enable_progress:
                        logger.info(
                            f"Completed batch {completed}/{len(batches)}"
                        )
                except Exception as e:
                    logger.error(f"Batch {index} failed: {e}")
                    raise
        
        return results
    
    def _get_batch_size(self, n_items: int) -> int:
        """
        获取批次大小（自适应）
        
        Args:
            n_items: 数据项总数
        
        Returns:
            批次大小
        """
        if not self.enable_adaptive:
            return self.batch_size
        
        # 自适应批次大小
        # 小数据集：使用较小批次
        # 大数据集：使用较大批次
        if n_items < 500:
            return min(self.ADAPTIVE_BATCH_SIZE_MIN, n_items)
        elif n_items < 5000:
            return min(self.batch_size, n_items)
        else:
            return min(self.ADAPTIVE_BATCH_SIZE_MAX, n_items // 10)
    
    def _update_stats(
        self,
        n_items: int,
        n_batches: int,
        elapsed_time: float
    ) -> None:
        """
        更新统计信息
        
        Args:
            n_items: 处理的数据项数量
            n_batches: 批次数量
            elapsed_time: 耗时（秒）
        """
        self.stats['total_items_processed'] += n_items
        self.stats['total_batches'] += n_batches
        self.stats['total_time_seconds'] += elapsed_time
        
        # 计算平均吞吐量
        if self.stats['total_time_seconds'] > 0:
            self.stats['avg_throughput'] = (
                self.stats['total_items_processed'] /
                self.stats['total_time_seconds']
            )
    
    def get_stats(self) -> dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_items_processed': self.stats['total_items_processed'],
            'total_batches': self.stats['total_batches'],
            'total_time_seconds': self.stats['total_time_seconds'],
            'avg_throughput': self.stats['avg_throughput'],
            'current_batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'adaptive_enabled': self.enable_adaptive,
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_items_processed': 0,
            'total_batches': 0,
            'total_time_seconds': 0.0,
            'avg_throughput': 0.0,
        }
        logger.info("Statistics reset")
    
    def __repr__(self) -> str:
        return (
            f"BatchOptimizer("
            f"batch_size={self.batch_size}, "
            f"max_workers={self.max_workers}, "
            f"adaptive={self.enable_adaptive})"
        )


class MemoryBatchProcessor:
    """
    记忆批量处理器
    
    专门用于批量添加记忆的优化处理器。
    """
    
    def __init__(
        self,
        embedder: Optional[EmbeddingProvider] = None,
        batch_size: int = 100,
        max_workers: int = 4,
        enable_adaptive: bool = True
    ):
        """
        初始化记忆批量处理器

        Args:
            embedder: EmbeddingProvider 实例
            batch_size: 批次大小
            max_workers: 最大并行工作线程数
            enable_adaptive: 是否启用自适应批次大小
        """
        self.embedder = embedder or get_default_provider()
        self.optimizer = BatchOptimizer(
            batch_size=batch_size,
            max_workers=max_workers,
            enable_adaptive=enable_adaptive
        )
    
    def batch_add_memories(
        self,
        memory_ids: List[str],
        contents: List[str],
        parallel: bool = True
    ) -> pa.Table:
        """
        批量添加记忆（优化版）
        
        Args:
            memory_ids: 记忆 ID 列表
            contents: 内容列表
            parallel: 是否并行处理
        
        Returns:
            Arrow Table 包含所有记忆
        """
        if len(memory_ids) != len(contents):
            raise ValueError("memory_ids and contents must have same length")
        
        if not memory_ids:
            return pa.table({})
        
        logger.info(f"Batch adding {len(memory_ids)} memories...")
        
        # 准备数据
        items = list(zip(memory_ids, contents))
        
        # 批量处理
        def process_batch(batch_items):
            batch_ids = [item[0] for item in batch_items]
            batch_contents = [item[1] for item in batch_items]
            
            # 批量编码
            embeddings = self.embedder.encode_batch(batch_contents)
            
            # 创建 Arrow Table
            table = pa.table({
                'memory_id': pa.array(batch_ids),
                'content': pa.array(batch_contents),
                'embedding': self._numpy_to_arrow_list(embeddings)
            })
            
            return table
        
        # 处理所有批次
        batch_tables = self.optimizer.process_in_batches(
            items,
            process_batch,
            parallel=parallel
        )
        
        # 合并所有表
        if len(batch_tables) == 1:
            result_table = batch_tables[0]
        else:
            result_table = pa.concat_tables(batch_tables)
        
        logger.info(
            f"Successfully added {len(result_table)} memories "
            f"(throughput: {self.optimizer.stats['avg_throughput']:.1f} memories/s)"
        )
        
        return result_table
    
    def _numpy_to_arrow_list(self, embeddings: np.ndarray) -> pa.Array:
        """
        将 NumPy 数组转换为 Arrow FixedSizeListArray
        
        Args:
            embeddings: NumPy 数组 (shape: [n, d])
        
        Returns:
            Arrow FixedSizeListArray
        """
        n_rows, dim = embeddings.shape
        flat_embeddings = embeddings.flatten()
        values = pa.array(flat_embeddings, type=pa.float32())
        list_array = pa.FixedSizeListArray.from_arrays(values, list_size=dim)
        return list_array
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return self.optimizer.get_stats()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.optimizer.reset_stats()


def create_memory_batch_processor(
    embedder,
    batch_size: int = 100,
    max_workers: int = 4,
    enable_adaptive: bool = True
) -> MemoryBatchProcessor:
    """
    创建记忆批量处理器（便捷函数）
    
    Args:
        embedder: Embedder 实例
        batch_size: 批次大小
        max_workers: 最大并行工作线程数
        enable_adaptive: 是否启用自适应批次大小
    
    Returns:
        MemoryBatchProcessor 实例
    
    Example:
        >>> from llm_compression.batch_optimizer import create_memory_batch_processor
        >>> from llm_compression.embedder import LocalEmbedder
        >>> 
        >>> embedder = LocalEmbedder()
        >>> processor = create_memory_batch_processor(embedder)
        >>> 
        >>> # 批量添加记忆（优化版）
        >>> table = processor.batch_add_memories(memory_ids, contents)
    """
    return MemoryBatchProcessor(
        embedder=embedder,
        batch_size=batch_size,
        max_workers=max_workers,
        enable_adaptive=enable_adaptive
    )

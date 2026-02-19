"""
Embedder 自适应切换逻辑

根据数据规模自动选择最优方法（传统 vs Arrow）。

Features:
- 自动切换逻辑
- 性能阈值配置
- 统计信息收集

Optimization: 小规模用传统方法，大规模用 Arrow
"""

import logging
from typing import List, Union, Optional
import numpy as np
import pyarrow as pa

from llm_compression.embedding_provider import EmbeddingProvider, get_default_provider, LocalEmbedderProvider

logger = logging.getLogger(__name__)


class AdaptiveEmbedder:
    """
    自适应 Embedder
    
    根据数据规模自动选择最优方法：
    - 小规模（<1000）：使用传统方法（更快）
    - 大规模（>=1000）：使用 Arrow 方法（更高效）
    """
    
    # 性能阈值配置
    SMALL_SCALE_THRESHOLD = 1000  # 小规模阈值
    ARROW_OVERHEAD_THRESHOLD = 100  # Arrow 开销阈值
    
    def __init__(
        self,
        embedder: Optional[EmbeddingProvider] = None,
        small_scale_threshold: int = SMALL_SCALE_THRESHOLD,
        enable_stats: bool = True
    ):
        """
        初始化自适应 Embedder

        Args:
            embedder: 基础 EmbeddingProvider 实例
            small_scale_threshold: 小规模阈值
            enable_stats: 是否启用统计信息收集
        """
        self.embedder = embedder or LocalEmbedderProvider()
        # Arrow 版本优先尝试从默认路径加载 ArrowEngineProvider
        self.embedder_arrow = get_default_provider()
        self.small_scale_threshold = small_scale_threshold
        self.enable_stats = enable_stats
        
        # 统计信息
        self.stats = {
            'traditional_calls': 0,
            'arrow_calls': 0,
            'total_items_traditional': 0,
            'total_items_arrow': 0,
        }
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        force_method: Optional[str] = None
    ) -> np.ndarray:
        """
        批量编码（自适应选择方法）
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            normalize: 是否归一化
            force_method: 强制使用的方法（'traditional' 或 'arrow'）
        
        Returns:
            NumPy 数组 (shape: [n, d])
        """
        n_texts = len(texts)
        
        # 决定使用哪种方法
        if force_method:
            use_arrow = (force_method == 'arrow')
        else:
            use_arrow = self._should_use_arrow(n_texts, operation='batch_encode')
        
        # 执行编码
        if use_arrow:
            logger.debug(f"Using Arrow method for {n_texts} texts")
            result = self.embedder_arrow.encode_batch(
                texts,
                batch_size=batch_size,
                normalize=normalize
            )
            
            # 更新统计
            if self.enable_stats:
                self.stats['arrow_calls'] += 1
                self.stats['total_items_arrow'] += n_texts
        else:
            logger.debug(f"Using traditional method for {n_texts} texts")
            result = self.embedder.encode_batch(
                texts,
                batch_size=batch_size,
                normalize=normalize
            )
            
            # 更新统计
            if self.enable_stats:
                self.stats['traditional_calls'] += 1
                self.stats['total_items_traditional'] += n_texts
        
        return result
    
    def find_most_similar(
        self,
        query: Union[str, np.ndarray],
        candidates: Union[List[str], np.ndarray, pa.Table],
        top_k: int = 5,
        force_method: Optional[str] = None
    ) -> List[tuple]:
        """
        找到最相似的候选项（自适应选择方法）
        
        Args:
            query: 查询文本或向量
            candidates: 候选文本列表、向量矩阵或 Arrow Table
            top_k: 返回前 K 个结果
            force_method: 强制使用的方法
        
        Returns:
            [(索引, 相似度分数), ...] 列表
        """
        # 确定候选数量
        if isinstance(candidates, pa.Table):
            n_candidates = len(candidates)
            is_arrow_table = True
        elif isinstance(candidates, list):
            n_candidates = len(candidates)
            is_arrow_table = False
        else:
            n_candidates = len(candidates)
            is_arrow_table = False
        
        # 决定使用哪种方法
        if force_method:
            use_arrow = (force_method == 'arrow')
        else:
            use_arrow = self._should_use_arrow(
                n_candidates,
                operation='similarity_search'
            )
        
        # 执行搜索
        if use_arrow and is_arrow_table:
            logger.debug(f"Using Arrow method for similarity search ({n_candidates} candidates)")
            # 兼容：如果 embedder_arrow 是 LocalEmbedderArrow 包装器
            if hasattr(self.embedder_arrow, 'find_most_similar_arrow'):
                result = self.embedder_arrow.find_most_similar_arrow(
                    query,
                    candidates,
                    top_k=top_k
                )
            else:
                # 使用通用 similarity_matrix 接口
                from llm_compression.arrow_zero_copy import get_embeddings_buffer
                embeddings = get_embeddings_buffer(candidates, 'embedding')
                if isinstance(query, str):
                    query_vec = self.embedder_arrow.encode(query)
                else:
                    query_vec = query
                
                sims = self.embedder_arrow.similarity_matrix(embeddings, query_vec)
                top_indices = np.argsort(sims)[::-1][:top_k]
                result = [(int(idx), float(sims[idx])) for idx in top_indices]
            
            # 更新统计
            if self.enable_stats:
                self.stats['arrow_calls'] += 1
                self.stats['total_items_arrow'] += n_candidates
        else:
            logger.debug(f"Using traditional method for similarity search ({n_candidates} candidates)")
            
            # 如果是 Arrow Table，先转换
            if is_arrow_table:
                from llm_compression.arrow_zero_copy import get_embeddings_buffer
                candidates = get_embeddings_buffer(candidates, 'embedding')
            
            result = self.embedder.find_most_similar(
                query,
                candidates,
                top_k=top_k
            )
            
            # 更新统计
            if self.enable_stats:
                self.stats['traditional_calls'] += 1
                self.stats['total_items_traditional'] += n_candidates
        
        return result
    
    def _should_use_arrow(self, n_items: int, operation: str) -> bool:
        """
        决定是否使用 Arrow 方法
        
        Args:
            n_items: 数据项数量
            operation: 操作类型
        
        Returns:
            是否使用 Arrow 方法
        """
        # 基于数据规模的决策
        if operation == 'batch_encode':
            # 批量编码：小规模用传统方法
            return n_items >= self.small_scale_threshold
        
        elif operation == 'similarity_search':
            # 相似度搜索：小规模用传统方法
            # Arrow 方法在小规模数据集上有额外开销
            return n_items >= self.small_scale_threshold
        
        else:
            # 默认：大规模用 Arrow
            return n_items >= self.small_scale_threshold
    
    def get_stats(self) -> dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        if not self.enable_stats:
            return {'stats_disabled': True}
        
        total_calls = self.stats['traditional_calls'] + self.stats['arrow_calls']
        total_items = self.stats['total_items_traditional'] + self.stats['total_items_arrow']
        
        return {
            'traditional_calls': self.stats['traditional_calls'],
            'arrow_calls': self.stats['arrow_calls'],
            'total_calls': total_calls,
            'traditional_percentage': (
                self.stats['traditional_calls'] / total_calls * 100
                if total_calls > 0 else 0
            ),
            'arrow_percentage': (
                self.stats['arrow_calls'] / total_calls * 100
                if total_calls > 0 else 0
            ),
            'total_items_traditional': self.stats['total_items_traditional'],
            'total_items_arrow': self.stats['total_items_arrow'],
            'total_items': total_items,
            'avg_items_per_call_traditional': (
                self.stats['total_items_traditional'] / self.stats['traditional_calls']
                if self.stats['traditional_calls'] > 0 else 0
            ),
            'avg_items_per_call_arrow': (
                self.stats['total_items_arrow'] / self.stats['arrow_calls']
                if self.stats['arrow_calls'] > 0 else 0
            ),
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'traditional_calls': 0,
            'arrow_calls': 0,
            'total_items_traditional': 0,
            'total_items_arrow': 0,
        }
        logger.info("Statistics reset")
    
    def set_threshold(self, threshold: int) -> None:
        """
        设置小规模阈值
        
        Args:
            threshold: 新的阈值
        """
        old_threshold = self.small_scale_threshold
        self.small_scale_threshold = threshold
        logger.info(
            f"Threshold updated: {old_threshold} -> {threshold}"
        )
    
    def __repr__(self) -> str:
        return (
            f"AdaptiveEmbedder("
            f"threshold={self.small_scale_threshold}, "
            f"stats_enabled={self.enable_stats})"
        )


def create_adaptive_embedder(
    small_scale_threshold: int = AdaptiveEmbedder.SMALL_SCALE_THRESHOLD,
    enable_stats: bool = True
) -> AdaptiveEmbedder:
    """
    创建自适应 Embedder（便捷函数）
    
    Args:
        small_scale_threshold: 小规模阈值
        enable_stats: 是否启用统计
    
    Returns:
        AdaptiveEmbedder 实例
    
    Example:
        >>> from llm_compression.embedder_adaptive import create_adaptive_embedder
        >>> embedder = create_adaptive_embedder()
        >>> # 自动选择最优方法
        >>> embeddings = embedder.encode_batch(texts)
    """
    return AdaptiveEmbedder(
        small_scale_threshold=small_scale_threshold,
        enable_stats=enable_stats
    )

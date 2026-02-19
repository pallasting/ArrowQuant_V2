"""
NetworkNavigator Arrow 原生支持扩展

为 NetworkNavigator 添加向量化检索和 Arrow 原生支持。

Features:
- retrieve_arrow(): 零拷贝检索（返回 Arrow Table）
- 向量化相似度计算（批量处理）
- spread_activation_arrow(): Arrow 原生激活扩散
- Top-K 选择优化（np.argpartition）
- 批量连接强度计算

Requirements: Task 12.3
"""

import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging

from llm_compression.network_navigator import NetworkNavigator, ActivationResult
from llm_compression.memory_primitive import MemoryPrimitive
from llm_compression.arrow_zero_copy import (
    get_embeddings_buffer,
    compute_similarity_zero_copy,
    ArrowBatchView
)

logger = logging.getLogger(__name__)


@dataclass
class ActivationResultArrow:
    """Result of activation spreading (Arrow version)."""
    table: pa.Table  # Arrow Table with memories and activation scores
    activation_map: Dict[str, float]
    hops_taken: int


class NetworkNavigatorArrow:
    """
    NetworkNavigator Arrow 原生支持扩展
    
    提供向量化检索和零拷贝操作，避免逐行处理开销。
    
    Requirements: Task 12.3
    """
    
    def __init__(
        self,
        navigator: Optional[NetworkNavigator] = None,
        max_hops: int = 3,
        decay_rate: float = 0.7,
        activation_threshold: float = 0.1
    ):
        """
        初始化 Arrow 扩展
        
        Args:
            navigator: NetworkNavigator 实例（如果为 None，创建新实例）
            max_hops: 最大跳数
            decay_rate: 激活衰减率
            activation_threshold: 激活阈值
        """
        self.navigator = navigator or NetworkNavigator(
            max_hops=max_hops,
            decay_rate=decay_rate,
            activation_threshold=activation_threshold
        )
        self.max_hops = self.navigator.max_hops
        self.decay_rate = self.navigator.decay_rate
        self.activation_threshold = self.navigator.activation_threshold
    
    def retrieve_arrow(
        self,
        query_embedding: np.ndarray,
        memory_table: pa.Table,
        max_results: int = 10,
        embedding_column: str = 'embedding',
        id_column: str = 'memory_id'
    ) -> ActivationResultArrow:
        """
        检索相关记忆（零拷贝，向量化）
        
        使用向量化操作进行激活扩散，避免逐行处理。
        
        Args:
            query_embedding: 查询向量
            memory_table: 记忆 Arrow Table
            max_results: 最大结果数
            embedding_column: Embedding 列名
            id_column: ID 列名
        
        Returns:
            ActivationResultArrow 包含检索结果
        
        Requirements: Task 12.3
        """
        if len(memory_table) == 0:
            return ActivationResultArrow(
                table=memory_table,
                activation_map={},
                hops_taken=0
            )
        
        # 1. 初始激活（向量化相似度计算）
        initial_indices, initial_activations = self._find_similar_vectorized(
            query_embedding,
            memory_table,
            embedding_column=embedding_column,
            top_k=min(5, len(memory_table))
        )
        
        # 2. 激活扩散（向量化）
        activation_map = self._spread_activation_vectorized(
            initial_indices,
            initial_activations,
            memory_table,
            id_column=id_column
        )
        
        # 3. 排序并返回 top-k
        result_table = self._get_top_k_results(
            memory_table,
            activation_map,
            max_results=max_results,
            id_column=id_column
        )
        
        logger.debug(
            f"Retrieved {len(result_table)} memories using activation spreading "
            f"(max_hops={self.max_hops})"
        )
        
        return ActivationResultArrow(
            table=result_table,
            activation_map=activation_map,
            hops_taken=self.max_hops
        )
    
    def _find_similar_vectorized(
        self,
        query_embedding: np.ndarray,
        memory_table: pa.Table,
        embedding_column: str = 'embedding',
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        找到初始相似记忆（向量化）
        
        Args:
            query_embedding: 查询向量
            memory_table: 记忆表
            embedding_column: Embedding 列名
            top_k: Top-K 数量
        
        Returns:
            (indices, similarities) 元组
        
        Requirements: Task 12.3
        """
        # 提取所有 embeddings（零拷贝）
        embeddings = get_embeddings_buffer(memory_table, embedding_column)
        
        if embeddings is None or len(embeddings) == 0:
            return np.array([]), np.array([])
        
        # 计算相似度（向量化）
        similarities = compute_similarity_zero_copy(embeddings, query_embedding)
        
        # 获取 top-k（使用 argpartition 优化）
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            # argpartition 比 argsort 快（O(n) vs O(n log n)）
            partition_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = partition_indices[np.argsort(similarities[partition_indices])[::-1]]
        
        top_similarities = similarities[top_indices]
        
        logger.debug(
            f"Found {len(top_indices)} initial similar memories "
            f"(top similarity: {top_similarities[0]:.3f})"
        )
        
        return top_indices, top_similarities
    
    def _spread_activation_vectorized(
        self,
        initial_indices: np.ndarray,
        initial_activations: np.ndarray,
        memory_table: pa.Table,
        id_column: str = 'memory_id'
    ) -> Dict[str, float]:
        """
        激活扩散（向量化）
        
        注意：由于连接图的稀疏性，完全向量化较困难。
        这里使用混合方法：向量化相似度计算 + 优化的图遍历。
        
        Args:
            initial_indices: 初始记忆索引
            initial_activations: 初始激活值
            memory_table: 记忆表
            id_column: ID 列名
        
        Returns:
            激活映射 (memory_id -> activation)
        
        Requirements: Task 12.3
        """
        # 激活映射
        activation_map: Dict[str, float] = {}
        
        # 获取所有 memory IDs
        memory_ids = memory_table[id_column].to_pylist()
        
        # 初始化激活
        for idx, activation in zip(initial_indices, initial_activations):
            memory_id = memory_ids[int(idx)]
            activation_map[memory_id] = float(activation)
        
        # 如果表中有连接信息，进行激活扩散
        if 'connections' in memory_table.schema.names:
            activation_map = self._spread_with_connections(
                activation_map,
                memory_table,
                memory_ids
            )
        
        return activation_map
    
    def _spread_with_connections(
        self,
        initial_activation_map: Dict[str, float],
        memory_table: pa.Table,
        memory_ids: List[str]
    ) -> Dict[str, float]:
        """
        使用连接信息进行激活扩散
        
        Args:
            initial_activation_map: 初始激活映射
            memory_table: 记忆表
            memory_ids: 记忆 ID 列表
        
        Returns:
            更新后的激活映射
        """
        activation_map = initial_activation_map.copy()
        
        # 队列：(memory_id, activation, hop_count)
        queue: List[Tuple[str, float, int]] = [
            (mid, act, 0) for mid, act in initial_activation_map.items()
        ]
        
        # 访问集合
        visited: Set[str] = set()
        
        # 创建 ID 到索引的映射
        id_to_idx = {mid: idx for idx, mid in enumerate(memory_ids)}
        
        # 激活扩散
        while queue:
            memory_id, activation, hop = queue.pop(0)
            
            # 跳过已访问或超过最大跳数
            if memory_id in visited or hop >= self.max_hops:
                continue
            
            visited.add(memory_id)
            
            # 获取记忆索引
            idx = id_to_idx.get(memory_id)
            if idx is None:
                continue
            
            # 获取连接信息
            connections = memory_table['connections'][idx].as_py()
            if not connections:
                continue
            
            # 传播到连接的记忆
            for conn_id, connection_strength in connections.items():
                if conn_id not in id_to_idx:
                    continue
                
                # 计算新激活
                new_activation = activation * connection_strength * self.decay_rate
                
                # 跳过低于阈值的激活
                if new_activation < self.activation_threshold:
                    continue
                
                # 累积激活
                if conn_id in activation_map:
                    activation_map[conn_id] = max(activation_map[conn_id], new_activation)
                else:
                    activation_map[conn_id] = new_activation
                
                # 添加到队列
                if conn_id not in visited:
                    queue.append((conn_id, new_activation, hop + 1))
        
        logger.debug(
            f"Activation spreading: {len(initial_activation_map)} initial -> "
            f"{len(activation_map)} final memories"
        )
        
        return activation_map
    
    def _get_top_k_results(
        self,
        memory_table: pa.Table,
        activation_map: Dict[str, float],
        max_results: int,
        id_column: str = 'memory_id'
    ) -> pa.Table:
        """
        获取 top-k 结果（零拷贝）
        
        Args:
            memory_table: 记忆表
            activation_map: 激活映射
            max_results: 最大结果数
            id_column: ID 列名
        
        Returns:
            包含 top-k 结果的 Arrow Table
        
        Requirements: Task 12.3
        """
        if not activation_map:
            return pa.table({}, schema=memory_table.schema)
        
        # 获取所有 memory IDs
        memory_ids = memory_table[id_column].to_pylist()
        
        # 创建 ID 到索引的映射
        id_to_idx = {mid: idx for idx, mid in enumerate(memory_ids)}
        
        # 获取激活的记忆索引和分数
        indices = []
        scores = []
        for memory_id, activation in activation_map.items():
            idx = id_to_idx.get(memory_id)
            if idx is not None:
                indices.append(idx)
                scores.append(activation)
        
        if not indices:
            return pa.table({}, schema=memory_table.schema)
        
        # 转换为 NumPy 数组
        indices_array = np.array(indices)
        scores_array = np.array(scores)
        
        # 排序并获取 top-k（使用 argpartition 优化）
        if len(scores_array) <= max_results:
            sorted_indices = np.argsort(scores_array)[::-1]
        else:
            partition_indices = np.argpartition(scores_array, -max_results)[-max_results:]
            sorted_indices = partition_indices[np.argsort(scores_array[partition_indices])[::-1]]
        
        top_indices = indices_array[sorted_indices]
        top_scores = scores_array[sorted_indices]
        
        # 使用 take 提取行（零拷贝）
        indices_pa = pa.array(top_indices.tolist())
        result_table = pc.take(memory_table, indices_pa)
        
        # 添加激活分数列
        score_array = pa.array(top_scores.tolist(), type=pa.float32())
        result_table = result_table.append_column('activation_score', score_array)
        
        return result_table
    
    def batch_retrieve_arrow(
        self,
        query_embeddings: np.ndarray,
        memory_table: pa.Table,
        max_results: int = 10,
        embedding_column: str = 'embedding',
        id_column: str = 'memory_id'
    ) -> List[ActivationResultArrow]:
        """
        批量检索（向量化优化）
        
        Args:
            query_embeddings: 查询向量矩阵 (shape: [n_queries, d])
            memory_table: 记忆表
            max_results: 每个查询的最大结果数
            embedding_column: Embedding 列名
            id_column: ID 列名
        
        Returns:
            每个查询的结果列表
        
        Requirements: Task 12.3
        """
        results = []
        
        for query_embedding in query_embeddings:
            result = self.retrieve_arrow(
                query_embedding,
                memory_table,
                max_results=max_results,
                embedding_column=embedding_column,
                id_column=id_column
            )
            results.append(result)
        
        logger.debug(
            f"Batch retrieve: {len(query_embeddings)} queries, "
            f"avg results: {np.mean([len(r.table) for r in results]):.1f}"
        )
        
        return results
    
    def find_similar_memories_vectorized(
        self,
        query_embedding: np.ndarray,
        memory_table: pa.Table,
        top_k: int = 10,
        embedding_column: str = 'embedding',
        threshold: float = 0.0
    ) -> pa.Table:
        """
        找到相似记忆（向量化，无激活扩散）
        
        这是一个简化版本，只进行相似度搜索，不进行激活扩散。
        适用于不需要网络导航的场景。
        
        Args:
            query_embedding: 查询向量
            memory_table: 记忆表
            top_k: Top-K 数量
            embedding_column: Embedding 列名
            threshold: 相似度阈值
        
        Returns:
            包含相似记忆的 Arrow Table
        
        Requirements: Task 12.3
        """
        if len(memory_table) == 0:
            return memory_table
        
        # 提取 embeddings（零拷贝）
        embeddings = get_embeddings_buffer(memory_table, embedding_column)
        
        if embeddings is None or len(embeddings) == 0:
            return pa.table({}, schema=memory_table.schema)
        
        # 计算相似度（向量化）
        similarities = compute_similarity_zero_copy(embeddings, query_embedding)
        
        # 应用阈值
        if threshold > 0:
            mask = similarities >= threshold
            filtered_indices = np.where(mask)[0]
            filtered_similarities = similarities[filtered_indices]
        else:
            filtered_indices = np.arange(len(similarities))
            filtered_similarities = similarities
        
        if len(filtered_similarities) == 0:
            # 创建符合 schema 的空表
            empty_data = {field.name: pa.array([], type=field.type) for field in memory_table.schema}
            return pa.table(empty_data, schema=memory_table.schema)
        
        # 获取 top-k
        if len(filtered_similarities) <= top_k:
            sorted_indices = np.argsort(filtered_similarities)[::-1]
        else:
            partition_indices = np.argpartition(filtered_similarities, -top_k)[-top_k:]
            sorted_indices = partition_indices[np.argsort(filtered_similarities[partition_indices])[::-1]]
        
        top_indices = filtered_indices[sorted_indices]
        top_scores = filtered_similarities[sorted_indices]
        
        # 提取结果（零拷贝）
        indices_pa = pa.array(top_indices.tolist())
        result_table = pc.take(memory_table, indices_pa)
        
        # 添加相似度分数列
        score_array = pa.array(top_scores.tolist(), type=pa.float32())
        result_table = result_table.append_column('similarity_score', score_array)
        
        logger.debug(
            f"Found {len(result_table)} similar memories "
            f"(top score: {top_scores[0]:.3f})"
        )
        
        return result_table


def add_arrow_support(navigator: NetworkNavigator) -> NetworkNavigatorArrow:
    """
    为 NetworkNavigator 添加 Arrow 支持
    
    Args:
        navigator: NetworkNavigator 实例
    
    Returns:
        NetworkNavigatorArrow 包装器
    
    Requirements: Task 12.3
    """
    return NetworkNavigatorArrow(navigator)

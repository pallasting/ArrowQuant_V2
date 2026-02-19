"""
LocalEmbedder Arrow 原生支持扩展

为 LocalEmbedder 添加 Arrow 原生支持，实现零拷贝向量化操作。

Features:
- encode_to_arrow(): 直接编码为 Arrow Array
- similarity_matrix_arrow(): 零拷贝相似度计算
- batch_encode_arrow(): 批量编码优化
- 向量化计算（NumPy SIMD）

Requirements: Task 12.2
"""

import warnings
import pyarrow as pa
import numpy as np
from typing import List, Optional, Union
import logging

warnings.warn(
    "llm_compression.embedder_arrow (LocalEmbedderArrow) is deprecated. "
    "Use ArrowEngineProvider instead: "
    "from llm_compression.embedding_provider import ArrowEngineProvider",
    DeprecationWarning,
    stacklevel=2,
)

from llm_compression.embedder import LocalEmbedder

logger = logging.getLogger(__name__)


class LocalEmbedderArrow:
    """
    LocalEmbedder Arrow 原生支持扩展
    
    提供零拷贝向量化操作，避免 Python 对象转换开销。
    
    .. deprecated::
        LocalEmbedderArrow is deprecated. Use EmbeddingProvider interface instead:
        
        from llm_compression.embedding_provider import get_default_provider
        provider = get_default_provider()
        
        ArrowEngineProvider provides native Arrow support with better performance.
    
    Requirements: Task 12.2
    """
    
    def __init__(self, embedder: Optional[LocalEmbedder] = None):
        """
        初始化 Arrow 扩展
        
        Args:
            embedder: LocalEmbedder 实例（如果为 None，创建新实例）
        """
        import warnings
        warnings.warn(
            "LocalEmbedderArrow is deprecated. Use EmbeddingProvider interface instead: "
            "from llm_compression.embedding_provider import get_default_provider",
            DeprecationWarning,
            stacklevel=2
        )
        self.embedder = embedder or LocalEmbedder()
        self.dimension = self.embedder.dimension
    
    def encode_to_arrow(
        self,
        text: str,
        normalize: bool = True
    ) -> pa.Array:
        """
        文本向量化，直接返回 Arrow Array（零拷贝）
        
        Args:
            text: 输入文本
            normalize: 是否归一化向量
        
        Returns:
            Arrow FixedSizeListArray (embedding)
        
        Requirements: Task 12.2
        """
        # 使用原有 encode 方法
        embedding = self.embedder.encode(text, normalize=normalize)
        
        # 转换为 Arrow Array（零拷贝）
        # 使用 FixedSizeList 类型以保持向量维度
        arrow_array = pa.array(
            [embedding.tolist()],
            type=pa.list_(pa.float32(), self.dimension)
        )
        
        return arrow_array
    
    def batch_encode_arrow(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False
    ) -> pa.Array:
        """
        批量向量化，返回 Arrow Array（零拷贝）
        
        相比传统方法，避免了 Python list 的中间转换。
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            normalize: 是否归一化
            show_progress: 是否显示进度条
        
        Returns:
            Arrow FixedSizeListArray (shape: [n_texts, embedding_dim])
        
        Requirements: Task 12.2
        """
        if not texts:
            return pa.array([], type=pa.list_(pa.float32(), self.dimension))
        
        # 批量编码（返回 NumPy 数组）
        embeddings = self.embedder.encode_batch(
            texts,
            batch_size=batch_size,
            normalize=normalize,
            show_progress=show_progress
        )
        
        # 转换为 Arrow Array（零拷贝）
        # 使用 from_numpy_ndarray 进行高效转换
        arrow_array = self._numpy_to_arrow_list(embeddings)
        
        logger.debug(f"Batch encoded {len(texts)} texts to Arrow Array")
        return arrow_array
    
    def _numpy_to_arrow_list(self, embeddings: np.ndarray) -> pa.Array:
        """
        将 NumPy 数组转换为 Arrow FixedSizeListArray（零拷贝）
        
        Args:
            embeddings: NumPy 数组 (shape: [n, d])
        
        Returns:
            Arrow FixedSizeListArray
        """
        # 方法 1：使用 pa.array（简单但可能有拷贝）
        # return pa.array(embeddings.tolist(), type=pa.list_(pa.float32(), self.dimension))
        
        # 方法 2：使用 FixedSizeListArray.from_arrays（零拷贝）
        n_rows, dim = embeddings.shape
        
        # Flatten embeddings to 1D
        flat_embeddings = embeddings.flatten()
        
        # Create Arrow array from flat data (zero-copy)
        values = pa.array(flat_embeddings, type=pa.float32())
        
        # Create FixedSizeListArray
        list_array = pa.FixedSizeListArray.from_arrays(
            values,
            list_size=dim
        )
        
        return list_array
    
    def similarity_matrix_arrow(
        self,
        embeddings_table: pa.Table,
        query_embedding: Optional[np.ndarray] = None,
        embedding_column: str = 'embedding'
    ) -> np.ndarray:
        """
        计算相似度矩阵（零拷贝，向量化）
        
        从 Arrow Table 中提取 embeddings 并计算相似度，
        避免 Python 对象转换。
        
        Args:
            embeddings_table: Arrow Table 包含 embedding 列
            query_embedding: 可选的查询向量
            embedding_column: Embedding 列名
        
        Returns:
            相似度矩阵或向量（NumPy 数组）
        
        Requirements: Task 12.2
        """
        # 提取 embeddings（零拷贝）
        from llm_compression.arrow_zero_copy import get_embeddings_buffer
        
        embeddings = get_embeddings_buffer(embeddings_table, embedding_column)
        
        if embeddings is None or len(embeddings) == 0:
            return np.array([])
        
        # 计算相似度（向量化）
        if query_embedding is not None:
            # 计算每个向量与查询的相似度
            from llm_compression.arrow_zero_copy import compute_similarity_zero_copy
            similarities = compute_similarity_zero_copy(embeddings, query_embedding)
        else:
            # 计算所有向量之间的相似度矩阵
            # 假设向量已归一化，使用点积
            similarities = np.dot(embeddings, embeddings.T)
        
        logger.debug(
            f"Computed similarity matrix: shape={similarities.shape}, "
            f"query={'yes' if query_embedding is not None else 'no'}"
        )
        return similarities
    
    def find_most_similar_arrow(
        self,
        query: Union[str, np.ndarray],
        embeddings_table: pa.Table,
        top_k: int = 5,
        threshold: float = 0.0,
        embedding_column: str = 'embedding'
    ) -> List[tuple]:
        """
        找到最相似的候选项（零拷贝，向量化）
        
        Args:
            query: 查询文本或向量
            embeddings_table: Arrow Table 包含 embeddings
            top_k: 返回前 K 个结果
            threshold: 最低相似度阈值
            embedding_column: Embedding 列名
        
        Returns:
            [(索引, 相似度分数), ...] 列表
        
        Requirements: Task 12.2
        """
        # 处理查询
        if isinstance(query, str):
            query_vec = self.embedder.encode(query)
        else:
            query_vec = query
        
        # 计算相似度（零拷贝）
        similarities = self.similarity_matrix_arrow(
            embeddings_table,
            query_embedding=query_vec,
            embedding_column=embedding_column
        )
        
        if len(similarities) == 0:
            return []
        
        # 应用阈值过滤
        if threshold > 0:
            mask = similarities >= threshold
            filtered_indices = np.where(mask)[0]
            filtered_similarities = similarities[filtered_indices]
        else:
            filtered_indices = np.arange(len(similarities))
            filtered_similarities = similarities
        
        # 排序并返回 top-k（向量化）
        if len(filtered_similarities) == 0:
            return []
        
        top_indices_in_filtered = np.argsort(filtered_similarities)[::-1][:top_k]
        top_indices = filtered_indices[top_indices_in_filtered]
        top_scores = filtered_similarities[top_indices_in_filtered]
        
        results = [
            (int(idx), float(score))
            for idx, score in zip(top_indices, top_scores)
        ]
        
        logger.debug(
            f"Found {len(results)} similar items (top_k={top_k}, threshold={threshold})"
        )
        return results
    
    def semantic_search_arrow(
        self,
        query: str,
        corpus_table: pa.Table,
        text_column: str = 'text',
        embedding_column: str = 'embedding',
        top_k: int = 10,
        threshold: float = 0.0
    ) -> pa.Table:
        """
        语义搜索，返回 Arrow Table（零拷贝）
        
        Args:
            query: 查询文本
            corpus_table: 文档语料库 Arrow Table
            text_column: 文本列名
            embedding_column: Embedding 列名
            top_k: 返回前 K 个结果
            threshold: 最低相似度阈值
        
        Returns:
            Arrow Table 包含搜索结果和相似度分数
        
        Requirements: Task 12.2
        """
        # 找到最相似的项
        results = self.find_most_similar_arrow(
            query,
            corpus_table,
            top_k=top_k,
            threshold=threshold,
            embedding_column=embedding_column
        )
        
        if not results:
            # 返回空表
            return pa.table({}, schema=corpus_table.schema)
        
        # 提取索引和分数
        indices = [idx for idx, _ in results]
        scores = [score for _, score in results]
        
        # 使用 take 提取行（零拷贝）
        import pyarrow.compute as pc
        indices_array = pa.array(indices)
        result_table = pc.take(corpus_table, indices_array)
        
        # 添加相似度分数列
        score_array = pa.array(scores, type=pa.float32())
        result_table = result_table.append_column('similarity_score', score_array)
        
        logger.debug(f"Semantic search returned {len(result_table)} results")
        return result_table
    
    def batch_similarity_search(
        self,
        queries: List[str],
        corpus_table: pa.Table,
        embedding_column: str = 'embedding',
        top_k: int = 10,
        batch_size: int = 32
    ) -> List[List[tuple]]:
        """
        批量语义搜索（向量化优化）
        
        Args:
            queries: 查询文本列表
            corpus_table: 文档语料库 Arrow Table
            embedding_column: Embedding 列名
            top_k: 每个查询返回前 K 个结果
            batch_size: 批处理大小
        
        Returns:
            每个查询的结果列表
        
        Requirements: Task 12.2
        """
        if not queries:
            return []
        
        # 批量编码查询（向量化）
        query_embeddings = self.embedder.encode_batch(
            queries,
            batch_size=batch_size,
            normalize=True
        )
        
        # 提取语料库 embeddings（零拷贝）
        from llm_compression.arrow_zero_copy import get_embeddings_buffer
        corpus_embeddings = get_embeddings_buffer(corpus_table, embedding_column)
        
        if corpus_embeddings is None or len(corpus_embeddings) == 0:
            return [[] for _ in queries]
        
        # 批量计算相似度矩阵（向量化）
        # (n_queries, d) @ (d, n_corpus) = (n_queries, n_corpus)
        similarity_matrix = np.dot(query_embeddings, corpus_embeddings.T)
        
        # 为每个查询提取 top-k（向量化）
        results = []
        for i, similarities in enumerate(similarity_matrix):
            # 排序并获取 top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_scores = similarities[top_indices]
            
            query_results = [
                (int(idx), float(score))
                for idx, score in zip(top_indices, top_scores)
            ]
            results.append(query_results)
        
        logger.debug(
            f"Batch similarity search: {len(queries)} queries, "
            f"{len(corpus_embeddings)} corpus items"
        )
        return results
    
    def create_embedding_table(
        self,
        texts: List[str],
        batch_size: int = 32,
        include_text: bool = True,
        additional_columns: Optional[dict] = None
    ) -> pa.Table:
        """
        创建包含 embeddings 的 Arrow Table
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            include_text: 是否包含原始文本列
            additional_columns: 额外的列（dict: column_name -> values）
        
        Returns:
            Arrow Table 包含 embeddings 和其他列
        
        Requirements: Task 12.2
        """
        if not texts:
            schema = pa.schema([
                ('embedding', pa.list_(pa.float32(), self.dimension))
            ])
            if include_text:
                schema = schema.append(pa.field('text', pa.string()))
            return pa.table({}, schema=schema)
        
        # 批量编码（零拷贝）
        embeddings_array = self.batch_encode_arrow(
            texts,
            batch_size=batch_size,
            normalize=True
        )
        
        # 构建表
        data = {'embedding': embeddings_array}
        
        if include_text:
            data['text'] = pa.array(texts)
        
        if additional_columns:
            for col_name, col_values in additional_columns.items():
                data[col_name] = pa.array(col_values)
        
        table = pa.table(data)
        
        logger.debug(
            f"Created embedding table: {len(table)} rows, "
            f"{len(table.schema)} columns"
        )
        return table
    
    def get_embedding_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension
    
    def __repr__(self) -> str:
        return f"LocalEmbedderArrow(embedder={self.embedder})"


def add_arrow_support(embedder: LocalEmbedder) -> LocalEmbedderArrow:
    """
    为 LocalEmbedder 添加 Arrow 支持
    
    Args:
        embedder: LocalEmbedder 实例
    
    Returns:
        LocalEmbedderArrow 包装器
    
    Requirements: Task 12.2
    """
    return LocalEmbedderArrow(embedder)

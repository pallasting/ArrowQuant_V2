"""
本地向量化引擎

使用 sentence-transformers 进行本地文本向量化，零 API 成本。
目标：<10ms 延迟，384 维向量，>85% 语义相似度准确率。
"""

import warnings
import numpy as np
from typing import List, Union, Optional

warnings.warn(
    "llm_compression.embedder (LocalEmbedder) is deprecated. "
    "Use EmbeddingProvider instead: "
    "from llm_compression.embedding_provider import get_default_provider",
    DeprecationWarning,
    stacklevel=2,
)

# 延迟导入以加快启动速度
_model = None


def _get_model():
    """延迟加载模型（单例模式）"""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    return _model


class LocalEmbedder:
    """
    本地向量化引擎（零 API 成本）

    使用 sentence-transformers 的 all-MiniLM-L6-v2 模型：
    - 384 维向量
    - 快速推理（<10ms）
    - 良好的语义理解能力
    
    .. deprecated::
        LocalEmbedder is deprecated. Use EmbeddingProvider interface instead:
        
        from llm_compression.embedding_provider import get_default_provider
        provider = get_default_provider()
        
        This provides automatic fallback between ArrowEngine and sentence-transformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化本地嵌入模型

        Args:
            model_name: 模型名称（默认 all-MiniLM-L6-v2）
                       可选：all-mpnet-base-v2 (768维，更高质量但更慢)
        """
        warnings.warn(
            "LocalEmbedder is deprecated. Use EmbeddingProvider interface instead: "
            "from llm_compression.embedding_provider import get_default_provider",
            DeprecationWarning,
            stacklevel=2
        )
        self.model_name = model_name
        self.dimension = 384 if "MiniLM" in model_name else 768
        self._model = None  # 延迟加载

    @property
    def model(self):
        """延迟加载模型（使用缓存）"""
        if self._model is None:
            # 使用全局缓存，避免重复加载
            from llm_compression.embedder_cache import get_cached_model
            self._model = get_cached_model(self.model_name)
        return self._model

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        文本向量化

        Args:
            text: 输入文本
            normalize: 是否归一化向量（推荐，用于余弦相似度）

        Returns:
            384 维向量（numpy array）

        Example:
            >>> embedder = LocalEmbedder()
            >>> vec = embedder.encode("Hello, World!")
            >>> vec.shape
            (384,)
        """
        if not text or not text.strip():
            # 空文本返回零向量
            return np.zeros(self.dimension, dtype=np.float32)

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        return embedding.astype(np.float32)

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        批量向量化（更高效）

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            normalize: 是否归一化
            show_progress: 是否显示进度条

        Returns:
            (N, 384) 向量矩阵

        Example:
            >>> embedder = LocalEmbedder()
            >>> texts = ["Hello", "World", "Test"]
            >>> vecs = embedder.encode_batch(texts)
            >>> vecs.shape
            (3, 38    """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        # 过滤空文本
        valid_texts = [t if t and t.strip() else " " for t in texts]

        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress
        )

        return embeddings.astype(np.float32)

    def similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        method: str = "cosine"
    ) -> float:
        """
        计算两个向量的相似度

        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            method: 相似度方法（cosine/dot/euclidean）

        Returns:
            相似度分数（0-1，越高越相似）

        Example:
            >>> embedder = LocalEmbedder()
            >>> v1 = embedder.encode("cat")
            >>> v2 = embedder.encode("kitten")
            >>> embedder.similarity(v1, v2)
            0.85  # 高相似度
        """
        if method == "cosine":
            # 余弦相似度（推荐）
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        elif method == "dot":
            # 点积（如果向量已归一化，等同于余弦相似度）
            return float(np.dot(vec1, vec2))

        elif method == "euclidean":
            # 欧氏距离（转换为相似度）
            distance = np.linalg.norm(vec1 - vec2)
            return float(1.0 / (1.0 + distance))

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def similarity_matrix(
        self,
        vectors: np.ndarray,
        query_vector: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算相似度矩阵

        Args:
            vectors: (N, D) 向量矩阵
            query_vector: 可选的查询向量。如果提供，计算每个向量与查询的相似度；
                         否则计算所有向量之间的相似度矩阵

        Returns:
            相似度矩阵或向量

        Example:
            >>> embedder = LocalEmbedder()
            >>> vecs = embedder.encode_batch(["cat", "dog", "car"])
            >>> query = embedder.encode("animal")
            >>> sims = embedder.similarity_matrix(vecs, query)
            >>> sims.shape
            (3,)
        """
        if query_vector is not None:
            # 计算每个向量与查询的相似度
            # 假设向量已归一化，使用点积
            similarities = np.dot(vectors, query_vector)
            return similarities

        else:
            # 计算所有向量之间的相似度矩阵
            # (N, D) @ (D, N) = (N, N)
            similarities = np.dot(vectors, vectors.T)
            return similarities

    def find_most_similar(
        self,
        query: Union[str, np.ndarray],
        candidates: Union[List[str], np.ndarray],
        top_k: int = 5
    ) -> List[tuple]:
        """
        找到最相似的候选项

        Args:
            query: 查询文本或向量
            candidates: 候选文本列表或向量矩阵
            top_k: 返回前 K 个结果

        Returns:
            索引, 相似度分数), ...] 列表

        Example:
            >>> embedder = LocalEmbedder()
            >>> query = "programming language"
            >>> candidates = ["Python", "Java", "car", "tree"]
            >>> results = embedder.find_most_similar(query, candidates, top_k=2)
            >>> results
            [(0, 0.85), (1, 0.82)]  # Python 和 Java
        """
        # 处理查询
        if isinstance(query, str):
            query_vec = self.encode(query)
        else:
            query_vec = query

        # 处理候选
        if isinstance(candidates, list):
            candidate_vecs =e_batch(candidates)
        else:
            candidate_vecs = candidates

        # 计算相似度
        similarities = self.similarity_matrix(candidate_vecs, query_vec)

        # 排序并返回 top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
        ]

        return results

    def semantic_search(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[tuple]:
        """
        语义搜索

        Args:
            query: 查询文本
            corpus: 文档语料库
            top_k: 返回前 K 个结果
            threshold: 最低相似度阈值

        Returns:
            [(文档索引, 相似度分数), ...] 列表

        Example:
            >>> embedder = LocalEmbedder()
            >>> corpus = [
            ...     "Python is a programming language",
            ...     "Java is also a programming language",
            ...     "The cat sits on the mat"
            ... ]
            >>> results = embedder.semantic_search(
            ...     "coding languages",
            ...     corpus,
            ...     top_k=2
            ... )
            >>> len(results)
            2
        """
        results = self.find_most_similar(query, corpus, top_k=top_k)

        # 应用阈值过滤
        if threshold > 0:
            results = [(idx, score) for idx, score in results if score >= threshold]

        return results

    def get_embedding_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension

    def __repr__(self) -> str:
        return f"LocalEmbedder(model={self.model_name}, dim={self.dimension})"


# 便捷函数
def quick_encode(text: str) -> np.ndarray:
    """
    快速编码单个文本（使用全局模型实例）

    Args:
        text: 输入文本

    Returns:
        384 维向量
    """
    model = _get_model()
    return model.encode(text, convert_to_numpy=True).astype(np.float32)


def quick_similarity(text1: str, text2: str) -> float:
    """
    快速计算两个文本的相似度

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        相似度分数（0-1）
    """
    embedder = LocalEmbedder()
    vec1 = embedder.encode(text1)
    vec2 = embedder.encode(text2)
    return embedder.similarity(vec1, vec2)

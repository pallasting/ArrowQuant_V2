"""
Embedder 模型缓存优化

实现模型预加载和缓存机制，减少首次加载延迟。

Features:
- 全局模型缓存（单例模式）
- 预加载支持
- 多模型缓存
- 内存管理

Optimization: 减少 5-10s 首次延迟
"""

import logging
from typing import Optional, Dict
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

# 全局模型缓存
_model_cache: Dict[str, any] = {}
_cache_lock = threading.Lock()


class EmbedderCache:
    """
    Embedder 模型缓存管理器
    
    提供全局模型缓存，避免重复加载。
    
    .. deprecated::
        EmbedderCache is deprecated. Use EmbeddingProvider interface instead:
        
        from llm_compression.embedding_provider import get_default_provider
        provider = get_default_provider()
        
        The new interface handles caching automatically.
    """
    
    @staticmethod
    def get_model(model_name: str = "all-MiniLM-L6-v2"):
        """
        获取缓存的模型（单例模式）
        
        Args:
            model_name: 模型名称
        
        Returns:
            SentenceTransformer 模型实例
        """
        import warnings
        warnings.warn(
            "EmbedderCache is deprecated. Use EmbeddingProvider interface instead: "
            "from llm_compression.embedding_provider import get_default_provider",
            DeprecationWarning,
            stacklevel=2
        )
        with _cache_lock:
            if model_name not in _model_cache:
                logger.info(f"Loading model '{model_name}' (first time)...")
                try:
                    from sentence_transformers import SentenceTransformer
                    import time
                    
                    start_time = time.time()
                    model = SentenceTransformer(model_name)
                    load_time = time.time() - start_time
                    
                    _model_cache[model_name] = model
                    logger.info(
                        f"Model '{model_name}' loaded successfully "
                        f"in {load_time:.2f}s"
                    )
                except ImportError:
                    raise ImportError(
                        "sentence-transformers not installed. "
                        "Install with: pip install sentence-transformers"
                    )
                except Exception as e:
                    logger.error(f"Failed to load model '{model_name}': {e}")
                    raise
            else:
                logger.debug(f"Using cached model '{model_name}'")
            
            return _model_cache[model_name]
    
    @staticmethod
    def preload_model(model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        预加载模型到缓存
        
        在应用启动时调用，避免首次查询延迟。
        
        Args:
            model_name: 模型名称
        
        Example:
            >>> from llm_compression.embedder_cache import EmbedderCache
            >>> # 在应用启动时预加载
            >>> EmbedderCache.preload_model()
        """
        logger.info(f"Preloading model '{model_name}'...")
        EmbedderCache.get_model(model_name)
        logger.info(f"Model '{model_name}' preloaded successfully")
    
    @staticmethod
    def clear_cache(model_name: Optional[str] = None) -> None:
        """
        清除模型缓存
        
        Args:
            model_name: 模型名称（如果为 None，清除所有缓存）
        """
        with _cache_lock:
            if model_name is None:
                # 清除所有缓存
                count = len(_model_cache)
                _model_cache.clear()
                logger.info(f"Cleared {count} cached models")
            elif model_name in _model_cache:
                # 清除指定模型
                del _model_cache[model_name]
                logger.info(f"Cleared cached model '{model_name}'")
            else:
                logger.warning(f"Model '{model_name}' not in cache")
    
    @staticmethod
    def get_cache_info() -> Dict[str, any]:
        """
        获取缓存信息
        
        Returns:
            缓存统计信息
        """
        with _cache_lock:
            return {
                'cached_models': list(_model_cache.keys()),
                'cache_size': len(_model_cache),
            }
    
    @staticmethod
    def is_cached(model_name: str) -> bool:
        """
        检查模型是否已缓存
        
        Args:
            model_name: 模型名称
        
        Returns:
            是否已缓存
        """
        with _cache_lock:
            return model_name in _model_cache


def preload_default_model() -> None:
    """
    预加载默认模型（便捷函数）
    
    在应用启动时调用。
    
    Example:
        >>> from llm_compression.embedder_cache import preload_default_model
        >>> preload_default_model()
    """
    EmbedderCache.preload_model("all-MiniLM-L6-v2")


def get_cached_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    获取缓存的模型（便捷函数）
    
    Args:
        model_name: 模型名称
    
    Returns:
        SentenceTransformer 模型实例
    """
    return EmbedderCache.get_model(model_name)

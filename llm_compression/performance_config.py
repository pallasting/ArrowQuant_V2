"""
性能优化配置模块

为本地模型部署提供优化的性能配置。
Phase 1.1 性能优化。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PerformanceConfig:
    """性能优化配置"""
    
    # 批量处理优化（Task 26.1）
    batch_size: int = 32  # 增加到 32（Phase 1.1 优化）
    max_concurrent: int = 8  # 增加并发数（本地模型更快）
    similarity_threshold: float = 0.85  # 提高相似度阈值
    
    # 推理性能优化（Task 26.2）
    use_gpu: bool = True  # 启用 GPU 加速
    gpu_memory_fraction: float = 0.9  # GPU 内存使用比例
    enable_model_parallel: bool = False  # 模型并行（多 GPU）
    enable_kv_cache: bool = True  # 启用 KV cache
    
    # 缓存策略优化（Task 26.3）
    cache_size: int = 50000  # 增加缓存大小（Phase 1.1）
    cache_ttl: int = 7200  # 增加 TTL 到 2 小时
    enable_disk_cache: bool = False  # 磁盘缓存（可选）
    disk_cache_path: Optional[str] = None
    
    # 本地模型特定优化
    local_model_batch_size: int = 32  # 本地模型批量大小
    local_model_max_concurrent: int = 8  # 本地模型并发数
    
    # 云端 API 配置（保持不变）
    cloud_batch_size: int = 16
    cloud_max_concurrent: int = 4
    
    @classmethod
    def for_local_model(cls) -> "PerformanceConfig":
        """
        为本地模型创建优化配置
        
        Returns:
            PerformanceConfig: 本地模型优化配置
        """
        return cls(
            batch_size=32,
            max_concurrent=8,
            similarity_threshold=0.85,
            use_gpu=True,
            gpu_memory_fraction=0.9,
            enable_kv_cache=True,
            cache_size=50000,
            cache_ttl=7200
        )
    
    @classmethod
    def for_cloud_api(cls) -> "PerformanceConfig":
        """
        为云端 API 创建配置
        
        Returns:
            PerformanceConfig: 云端 API 配置
        """
        return cls(
            batch_size=16,
            max_concurrent=4,
            similarity_threshold=0.8,
            use_gpu=False,
            cache_size=10000,
            cache_ttl=3600
        )
    
    @classmethod
    def for_hybrid(cls, prefer_local: bool = True) -> "PerformanceConfig":
        """
        为混合模式创建配置
        
        Args:
            prefer_local: 是否优先使用本地模型
            
        Returns:
            PerformanceConfig: 混合模式配置
        """
        if prefer_local:
            return cls.for_local_model()
        else:
            return cls.for_cloud_api()
    
    def get_batch_size(self, is_local: bool) -> int:
        """
        根据模型类型获取批量大小
        
        Args:
            is_local: 是否本地模型
            
        Returns:
            int: 批量大小
        """
        return self.local_model_batch_size if is_local else self.cloud_batch_size
    
    def get_max_concurrent(self, is_local: bool) -> int:
        """
        根据模型类型获取最大并发数
        
        Args:
            is_local: 是否本地模型
            
        Returns:
            int: 最大并发数
        """
        return self.local_model_max_concurrent if is_local else self.cloud_max_concurrent

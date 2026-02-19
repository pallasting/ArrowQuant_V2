"""
健康检查模块

提供系统健康检查功能，检查 LLM 客户端、存储、GPU 和配置状态。

Requirements: 11.7
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from llm_compression.llm_client import LLMClient
from llm_compression.arrow_storage import ArrowStorage
from llm_compression.config import Config


logger = logging.getLogger(__name__)


@dataclass
class ComponentStatus:
    """组件状态"""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    overall_status: str  # "healthy", "degraded", "unhealthy"
    timestamp: float
    components: Dict[str, ComponentStatus]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "status": self.overall_status,
            "timestamp": self.timestamp,
            "components": {
                name: {
                    "status": comp.status,
                    "message": comp.message,
                    "latency_ms": comp.latency_ms,
                    "details": comp.details
                }
                for name, comp in self.components.items()
            }
        }


class HealthChecker:
    """健康检查器"""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        storage: Optional[ArrowStorage] = None,
        config: Optional[Config] = None
    ):
        """
        初始化健康检查器
        
        Args:
            llm_client: LLM 客户端（可选）
            storage: Arrow 存储（可选）
            config: 系统配置（可选）
        """
        self.llm_client = llm_client
        self.storage = storage
        self.config = config or Config()
    
    async def check_health(self) -> HealthCheckResult:
        """
        执行完整的健康检查
        
        检查项：
        1. LLM 客户端连接性
        2. 存储可访问性
        3. GPU 可用性
        4. 配置有效性
        
        Returns:
            HealthCheckResult: 健康检查结果
        """
        components = {}
        
        # 并发执行所有检查
        checks = [
            self._check_llm_client(),
            self._check_storage(),
            self._check_gpu(),
            self._check_config()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        # 收集结果
        check_names = ["llm_client", "storage", "gpu", "config"]
        for name, result in zip(check_names, results):
            if isinstance(result, Exception):
                components[name] = ComponentStatus(
                    name=name,
                    status="unhealthy",
                    message=f"Check failed: {str(result)}"
                )
            else:
                components[name] = result
        
        # 计算总体状态
        overall_status = self._compute_overall_status(components)
        
        return HealthCheckResult(
            overall_status=overall_status,
            timestamp=time.time(),
            components=components
        )
    
    async def _check_llm_client(self) -> ComponentStatus:
        """
        检查 LLM 客户端状态
        
        测试：
        - 连接性（ping 端点）
        - 响应时间
        """
        if not self.llm_client:
            return ComponentStatus(
                name="llm_client",
                status="degraded",
                message="LLM client not configured"
            )
        
        try:
            start_time = time.time()
            
            # 尝试简单的生成请求
            response = await self.llm_client.generate(
                prompt="ping",
                max_tokens=1,
                temperature=0.0
            )
            
            # 使用响应中的延迟（如果可用），否则使用实际测量
            if hasattr(response, 'latency_ms') and response.latency_ms is not None:
                latency_ms = response.latency_ms
            else:
                latency_ms = (time.time() - start_time) * 1000
            
            # 检查响应时间
            if latency_ms > 5000:  # > 5s
                status = "degraded"
                message = f"High latency: {latency_ms:.0f}ms"
            else:
                status = "healthy"
                message = "LLM client operational"
            
            return ComponentStatus(
                name="llm_client",
                status=status,
                message=message,
                latency_ms=latency_ms,
                details={
                    "model": response.model,
                    "endpoint": self.llm_client.endpoint
                }
            )
            
        except Exception as e:
            logger.error(f"LLM client check failed: {e}")
            return ComponentStatus(
                name="llm_client",
                status="unhealthy",
                message=f"Connection failed: {str(e)}"
            )
    
    async def _check_storage(self) -> ComponentStatus:
        """
        检查存储状态
        
        测试：
        - 存储路径可访问
        - 读写权限
        - 磁盘空间
        """
        if not self.storage:
            # 检查配置的存储路径
            storage_path = Path(self.config.storage.storage_path).expanduser()
        else:
            storage_path = self.storage.base_path
        
        try:
            # 检查路径存在
            if not storage_path.exists():
                return ComponentStatus(
                    name="storage",
                    status="degraded",
                    message=f"Storage path does not exist: {storage_path}"
                )
            
            # 检查读写权限
            test_file = storage_path / ".health_check"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                return ComponentStatus(
                    name="storage",
                    status="unhealthy",
                    message=f"No write permission: {str(e)}"
                )
            
            # 检查磁盘空间
            import shutil
            stat = shutil.disk_usage(storage_path)
            free_gb = stat.free / (1024 ** 3)
            
            if free_gb < 1.0:  # < 1GB
                status = "degraded"
                message = f"Low disk space: {free_gb:.1f}GB"
            else:
                status = "healthy"
                message = "Storage accessible"
            
            return ComponentStatus(
                name="storage",
                status=status,
                message=message,
                details={
                    "path": str(storage_path),
                    "free_gb": round(free_gb, 2),
                    "total_gb": round(stat.total / (1024 ** 3), 2)
                }
            )
            
        except Exception as e:
            logger.error(f"Storage check failed: {e}")
            return ComponentStatus(
                name="storage",
                status="unhealthy",
                message=f"Check failed: {str(e)}"
            )
    
    async def _check_gpu(self) -> ComponentStatus:
        """
        检查 GPU 状态
        
        测试：
        - GPU 可用性
        - CUDA 版本
        - 显存使用
        """
        if not TORCH_AVAILABLE:
            return ComponentStatus(
                name="gpu",
                status="degraded",
                message="PyTorch not installed"
            )
        
        try:
            if not torch.cuda.is_available():
                return ComponentStatus(
                    name="gpu",
                    status="degraded",
                    message="No GPU available (CPU mode)"
                )
            
            # 获取 GPU 信息
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            
            # 获取显存信息
            memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            
            memory_usage_pct = (memory_reserved / memory_total) * 100
            
            if memory_usage_pct > 90:
                status = "degraded"
                message = f"High GPU memory usage: {memory_usage_pct:.1f}%"
            else:
                status = "healthy"
                message = "GPU available"
            
            return ComponentStatus(
                name="gpu",
                status=status,
                message=message,
                details={
                    "device_count": device_count,
                    "device_name": device_name,
                    "cuda_version": cuda_version,
                    "memory_allocated_gb": round(memory_allocated, 2),
                    "memory_reserved_gb": round(memory_reserved, 2),
                    "memory_total_gb": round(memory_total, 2),
                    "memory_usage_pct": round(memory_usage_pct, 1)
                }
            )
            
        except Exception as e:
            logger.error(f"GPU check failed: {e}")
            return ComponentStatus(
                name="gpu",
                status="degraded",
                message=f"Check failed: {str(e)}"
            )
    
    async def _check_config(self) -> ComponentStatus:
        """
        检查配置有效性
        
        测试：
        - 配置文件存在
        - 必需字段存在
        - 值在有效范围内
        """
        try:
            # 检查关键配置
            issues = []
            
            # LLM 配置
            if not self.config.llm.cloud_endpoint:
                issues.append("Missing LLM endpoint")
            
            # 存储配置
            storage_path = Path(self.config.storage.storage_path).expanduser()
            if not storage_path.parent.exists():
                issues.append(f"Storage parent directory does not exist: {storage_path.parent}")
            
            # 性能配置
            if self.config.performance.batch_size < 1:
                issues.append("Invalid batch_size (must be >= 1)")
            
            if self.config.performance.max_concurrent < 1:
                issues.append("Invalid max_concurrent (must be >= 1)")
            
            # 压缩配置
            if not (0.0 <= self.config.compression.temperature <= 1.0):
                issues.append("Invalid temperature (must be 0.0-1.0)")
            
            if issues:
                return ComponentStatus(
                    name="config",
                    status="degraded",
                    message=f"Configuration issues: {', '.join(issues)}",
                    details={"issues": issues}
                )
            
            return ComponentStatus(
                name="config",
                status="healthy",
                message="Configuration valid",
                details={
                    "llm_endpoint": self.config.llm.cloud_endpoint,
                    "storage_path": str(storage_path),
                    "batch_size": self.config.performance.batch_size
                }
            )
            
        except Exception as e:
            logger.error(f"Config check failed: {e}")
            return ComponentStatus(
                name="config",
                status="unhealthy",
                message=f"Check failed: {str(e)}"
            )
    
    def _compute_overall_status(self, components: Dict[str, ComponentStatus]) -> str:
        """
        计算总体状态
        
        规则：
        - 任何组件 unhealthy -> 总体 unhealthy
        - 任何组件 degraded -> 总体 degraded
        - 所有组件 healthy -> 总体 healthy
        """
        statuses = [comp.status for comp in components.values()]
        
        if "unhealthy" in statuses:
            return "unhealthy"
        elif "degraded" in statuses:
            return "degraded"
        else:
            return "healthy"

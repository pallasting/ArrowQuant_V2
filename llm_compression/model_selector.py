"""
模型选择器模块

根据记忆类型和性能要求选择最优 LLM 模型。
实现本地模型优先策略、模型降级策略和质量监控。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
import asyncio

from llm_compression.logger import logger


class MemoryType(Enum):
    """记忆类型"""
    TEXT = "text"              # 普通文本
    CODE = "code"              # 代码
    MULTIMODAL = "multimodal"  # 多模态（图文）
    LONG_TEXT = "long_text"    # 长文本（> 500 字）


class QualityLevel(Enum):
    """质量等级"""
    LOW = "low"           # 低质量（快速）
    STANDARD = "standard" # 标准质量
    HIGH = "high"         # 高质量（慢速）


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str              # 模型名称
    endpoint: str                # API 端点
    is_local: bool               # 是否本地模型
    max_tokens: int              # 最大 token 数
    temperature: float           # 采样温度
    expected_latency_ms: float   # 预期延迟
    expected_quality: float      # 预期质量分数


@dataclass
class ModelStats:
    """模型统计信息"""
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    avg_quality_score: float = 0.0
    success_rate: float = 1.0
    total_tokens_used: int = 0
    last_used: float = field(default_factory=time.time)
    
    # 内部统计（用于计算平均值）
    _latency_sum: float = 0.0
    _quality_sum: float = 0.0
    _successful_requests: int = 0


class ModelSelector:
    """模型选择器"""
    
    def __init__(
        self,
        cloud_endpoint: str = "http://localhost:8045",
        local_endpoints: Optional[Dict[str, str]] = None,
        prefer_local: bool = True,
        quality_threshold: float = 0.85,
        ollama_endpoint: str = "http://localhost:11434"
    ):
        """
        初始化模型选择器
        
        Args:
            cloud_endpoint: 云端 API 端点
            local_endpoints: 本地模型端点映射 {"qwen2.5": "http://localhost:11434", ...}
            prefer_local: 是否优先使用本地模型
            quality_threshold: 质量阈值（低于此值建议切换模型）
            ollama_endpoint: Ollama 服务端点（默认 http://localhost:11434）
        """
        self.cloud_endpoint = cloud_endpoint
        self.ollama_endpoint = ollama_endpoint
        
        # 如果没有提供本地端点，使用默认的 Ollama 配置
        if local_endpoints is None:
            local_endpoints = {
                "qwen2.5": ollama_endpoint,  # Qwen2.5-7B (主力模型)
                "llama3.1": ollama_endpoint,  # Llama 3.1 8B (备选)
                "gemma3": ollama_endpoint,    # Gemma 3 4B (轻量级)
            }
        
        self.local_endpoints = local_endpoints
        self.prefer_local = prefer_local
        self.quality_threshold = quality_threshold
        
        # 模型统计信息
        self.model_stats: Dict[str, ModelStats] = {}
        self.stats_lock = asyncio.Lock()
        
        # 模型可用性缓存（避免频繁检查）
        self._availability_cache: Dict[str, bool] = {}
        self._cache_ttl = 60.0  # 缓存 60 秒
        self._cache_timestamps: Dict[str, float] = {}
        
        logger.info(
            f"ModelSelector initialized: cloud={cloud_endpoint}, "
            f"local={list(local_endpoints.keys())}, prefer_local={prefer_local}"
        )
    
    def select_model(
        self,
        memory_type: MemoryType,
        text_length: int,
        quality_requirement: QualityLevel = QualityLevel.STANDARD,
        manual_model: Optional[str] = None
    ) -> ModelConfig:
        """
        选择最优模型
        
        Args:
            memory_type: 记忆类型（TEXT/CODE/MULTIMODAL/LONG_TEXT）
            text_length: 文本长度（字符数）
            quality_requirement: 质量要求（LOW/STANDARD/HIGH）
            manual_model: 手动指定模型（覆盖自动选择）
            
        Returns:
            ModelConfig: 模型配置
        """
        # 如果手动指定模型，直接返回
        if manual_model:
            logger.info(f"Using manually specified model: {manual_model}")
            return self._get_manual_model_config(manual_model)
        
        # 根据规则选择模型
        model_name = self._select_by_rules(memory_type, text_length, quality_requirement)
        
        # 检查模型可用性并降级
        model_config = self._get_model_config_with_fallback(model_name, memory_type, text_length)
        
        logger.info(
            f"Selected model: {model_config.model_name} "
            f"(type={memory_type.value}, length={text_length}, quality={quality_requirement.value})"
        )
        
        return model_config
    
    def _select_by_rules(
        self,
        memory_type: MemoryType,
        text_length: int,
        quality_requirement: QualityLevel
    ) -> str:
        """
        根据规则选择模型
        
        规则（Requirements 3.1, 2.5）：
        - 本地模型优先（Phase 1.1）：
          * 普通文本（< 500 字）→ Qwen2.5-7B (本地) 或 云端 API
          * 长文本（> 500 字）→ Qwen2.5-7B (本地) 或 云端 API
          * 代码记忆 → Qwen2.5-7B (本地) 或 云端 API
        - 高质量要求 → 优先云端 API（Claude/GPT）
        - 降级策略：本地模型 → 云端 API → 简单压缩
        """
        # 高质量要求：优先云端 API
        if quality_requirement == QualityLevel.HIGH:
            return "cloud-api"
        
        # Phase 1.1: 本地模型优先策略
        if self.prefer_local:
            # 优先使用 Qwen2.5-7B（主力本地模型）
            if "qwen2.5" in self.local_endpoints:
                return "qwen2.5"
            
            # 备选：Llama 3.1 8B
            if "llama3.1" in self.local_endpoints:
                return "llama3.1"
            
            # 轻量级选项：Gemma 3 4B
            if "gemma3" in self.local_endpoints:
                return "gemma3"
        
        # 降级到云端 API
        return "cloud-api"
    
    def _get_model_config_with_fallback(
        self,
        model_name: str,
        memory_type: MemoryType,
        text_length: int
    ) -> ModelConfig:
        """
        获取模型配置，如果不可用则降级
        
        降级策略（Requirements 3.3）：
        1. 首选模型
        2. 云端 API（如果首选是本地模型）
        3. 其他可用的本地模型
        4. 简单压缩（返回特殊配置）
        """
        # 尝试首选模型
        if self._is_model_available(model_name):
            return self._get_model_config(model_name)
        
        logger.warning(f"Model {model_name} not available, falling back")
        
        # 如果首选是本地模型，尝试云端 API
        if model_name != "cloud-api" and self._is_model_available("cloud-api"):
            logger.info("Falling back to cloud API")
            return self._get_model_config("cloud-api")
        
        # 尝试其他本地模型
        for local_model in self.local_endpoints.keys():
            if local_model != model_name and self._is_model_available(local_model):
                logger.info(f"Falling back to {local_model}")
                return self._get_model_config(local_model)
        
        # 最后降级到简单压缩（无 LLM）
        logger.warning("All models unavailable, falling back to simple compression")
        return ModelConfig(
            model_name="simple-compression",
            endpoint="",
            is_local=True,
            max_tokens=0,
            temperature=0.0,
            expected_latency_ms=10.0,
            expected_quality=0.7
        )
    
    def _is_model_available(self, model_name: str) -> bool:
        """
        检查模型是否可用（带缓存）
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否可用
        """
        # 检查缓存
        now = time.time()
        if model_name in self._availability_cache:
            cache_time = self._cache_timestamps.get(model_name, 0)
            if now - cache_time < self._cache_ttl:
                return self._availability_cache[model_name]
        
        # 检查实际可用性
        available = self._check_model_availability(model_name)
        
        # 更新缓存
        self._availability_cache[model_name] = available
        self._cache_timestamps[model_name] = now
        
        return available
    
    def _check_model_availability(self, model_name: str) -> bool:
        """
        检查模型实际可用性
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否可用
        """
        # 云端 API 默认可用（实际应该做健康检查）
        if model_name == "cloud-api":
            return True
        
        # 本地模型：检查端点是否配置
        if model_name in self.local_endpoints:
            # 实际应该做 HTTP 健康检查
            # 这里简化为检查配置是否存在
            return True
        
        # 简单压缩总是可用
        if model_name == "simple-compression":
            return True
        
        return False
    
    def _get_model_config(self, model_name: str) -> ModelConfig:
        """
        获取模型配置
        
        Args:
            model_name: 模型名称
            
        Returns:
            ModelConfig: 模型配置
        """
        if model_name == "cloud-api":
            return ModelConfig(
                model_name="cloud-api",
                endpoint=self.cloud_endpoint,
                is_local=False,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=2000.0,
                expected_quality=0.95
            )
        
        # Phase 1.1: 本地模型配置
        elif model_name == "qwen2.5":
            return ModelConfig(
                model_name="qwen2.5:7b-instruct",  # Ollama 模型名称
                endpoint=self.local_endpoints.get("qwen2.5", self.ollama_endpoint),
                is_local=True,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=1500.0,  # 本地模型更快
                expected_quality=0.90
            )
        
        elif model_name == "llama3.1":
            return ModelConfig(
                model_name="llama3.1:8b-instruct-q4_K_M",  # Ollama 模型名称
                endpoint=self.local_endpoints.get("llama3.1", self.ollama_endpoint),
                is_local=True,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=1800.0,
                expected_quality=0.88
            )
        
        elif model_name == "gemma3":
            return ModelConfig(
                model_name="gemma3:4b",  # Ollama 模型名称
                endpoint=self.local_endpoints.get("gemma3", self.ollama_endpoint),
                is_local=True,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=1000.0,  # 更小更快
                expected_quality=0.85
            )
        
        # Phase 1.0 遗留模型（保留兼容性）
        elif model_name == "step-flash":
            return ModelConfig(
                model_name="step-flash",
                endpoint=self.local_endpoints.get("step-flash", ""),
                is_local=True,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=500.0,
                expected_quality=0.85
            )
        
        elif model_name == "minicpm-o":
            return ModelConfig(
                model_name="minicpm-o",
                endpoint=self.local_endpoints.get("minicpm-o", ""),
                is_local=True,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=1500.0,
                expected_quality=0.90
            )
        
        elif model_name == "stable-diffcoder":
            return ModelConfig(
                model_name="stable-diffcoder",
                endpoint=self.local_endpoints.get("stable-diffcoder", ""),
                is_local=True,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=800.0,
                expected_quality=0.88
            )
        
        elif model_name == "intern-s1-pro":
            return ModelConfig(
                model_name="intern-s1-pro",
                endpoint=self.local_endpoints.get("intern-s1-pro", ""),
                is_local=True,
                max_tokens=200,
                temperature=0.3,
                expected_latency_ms=2000.0,
                expected_quality=0.92
            )
        
        else:
            # 未知模型，返回默认配置
            logger.warning(f"Unknown model {model_name}, using default config")
            return ModelConfig(
                model_name=model_name,
                endpoint="",
                is_local=True,
                max_tokens=100,
                temperature=0.3,
                expected_latency_ms=1000.0,
                expected_quality=0.80
            )
    
    def _get_manual_model_config(self, model_name: str) -> ModelConfig:
        """获取手动指定模型的配置"""
        return self._get_model_config(model_name)
    
    async def record_usage(
        self,
        model_name: str,
        latency_ms: float,
        quality_score: float,
        tokens_used: int,
        success: bool = True
    ):
        """
        记录模型使用统计
        
        Args:
            model_name: 模型名称
            latency_ms: 延迟（毫秒）
            quality_score: 质量分数（0-1）
            tokens_used: 使用的 token 数
            success: 是否成功
        """
        async with self.stats_lock:
            if model_name not in self.model_stats:
                self.model_stats[model_name] = ModelStats()
            
            stats = self.model_stats[model_name]
            stats.total_requests += 1
            stats.last_used = time.time()
            
            if success:
                stats._successful_requests += 1
                stats._latency_sum += latency_ms
                stats._quality_sum += quality_score
                stats.total_tokens_used += tokens_used
                
                # 更新平均值
                stats.avg_latency_ms = stats._latency_sum / stats._successful_requests
                stats.avg_quality_score = stats._quality_sum / stats._successful_requests
            
            # 更新成功率
            stats.success_rate = stats._successful_requests / stats.total_requests
    
    def get_model_stats(self) -> Dict[str, ModelStats]:
        """
        获取所有模型的统计信息
        
        Returns:
            Dict[str, ModelStats]: 模型名称 -> 统计信息
        """
        return dict(self.model_stats)
    
    def suggest_model_switch(self, current_model: str) -> Optional[str]:
        """
        建议模型切换（基于质量监控）
        
        Args:
            current_model: 当前使用的模型
            
        Returns:
            Optional[str]: 建议切换的模型名称，如果不需要切换则返回 None
        """
        if current_model not in self.model_stats:
            return None
        
        stats = self.model_stats[current_model]
        
        # 如果质量低于阈值，建议切换到更强大的模型
        if stats.avg_quality_score < self.quality_threshold:
            logger.warning(
                f"Model {current_model} quality {stats.avg_quality_score:.2f} "
                f"below threshold {self.quality_threshold:.2f}"
            )
            
            # 如果当前是本地模型，建议切换到云端 API
            if current_model != "cloud-api":
                return "cloud-api"
            
            # 如果已经是云端 API，无法进一步提升
            logger.warning("Already using cloud API, cannot switch to better model")
            return None
        
        return None
    
    def clear_availability_cache(self):
        """清除可用性缓存（用于强制重新检查）"""
        self._availability_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Availability cache cleared")

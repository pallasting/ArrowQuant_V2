"""
配置管理模块

提供系统配置的加载、验证和访问功能。
支持 YAML 配置文件和环境变量覆盖。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from llm_compression.logger import logger


@dataclass
class LLMConfig:
    """LLM 客户端配置"""
    cloud_endpoint: str = "http://localhost:8045"
    cloud_api_key: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    rate_limit: int = 60  # requests per minute


@dataclass
class ModelConfig:
    """模型选择配置"""
    prefer_local: bool = True
    local_endpoints: Dict[str, str] = field(default_factory=dict)
    quality_threshold: float = 0.85
    ollama_endpoint: str = "http://localhost:11434"  # Ollama 服务端点


@dataclass
class CompressionConfig:
    """压缩配置"""
    min_compress_length: int = 100
    max_tokens: int = 100
    temperature: float = 0.3
    auto_compress_threshold: int = 100


@dataclass
class StorageConfig:
    """存储配置"""
    storage_path: str = "~/.ai-os/memory/"
    compression_level: int = 3  # zstd compression level
    use_float16: bool = True


@dataclass
class PerformanceConfig:
    """性能配置"""
    batch_size: int = 16
    max_concurrent: int = 4
    cache_size: int = 10000
    cache_ttl: int = 3600  # seconds


@dataclass
class MonitoringConfig:
    """监控配置"""
    enable_prometheus: bool = False
    prometheus_port: int = 9090
    alert_quality_threshold: float = 0.85


@dataclass
class Config:
    """系统配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        从 YAML 文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Config: 配置对象
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if not config_dict:
            logger.warning(f"Config file {config_path} is empty, using defaults")
            return cls()
        
        return cls._from_dict(config_dict)
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """从字典创建配置对象"""
        config = cls()
        
        # LLM 配置
        if 'llm' in config_dict:
            llm_dict = config_dict['llm']
            config.llm = LLMConfig(
                cloud_endpoint=llm_dict.get('cloud_endpoint', config.llm.cloud_endpoint),
                cloud_api_key=llm_dict.get('cloud_api_key', config.llm.cloud_api_key),
                timeout=llm_dict.get('timeout', config.llm.timeout),
                max_retries=llm_dict.get('max_retries', config.llm.max_retries),
                rate_limit=llm_dict.get('rate_limit', config.llm.rate_limit)
            )
        
        # 模型配置
        if 'model' in config_dict:
            model_dict = config_dict['model']
            config.model = ModelConfig(
                prefer_local=model_dict.get('prefer_local', config.model.prefer_local),
                local_endpoints=model_dict.get('local_endpoints', config.model.local_endpoints),
                quality_threshold=model_dict.get('quality_threshold', config.model.quality_threshold),
                ollama_endpoint=model_dict.get('ollama_endpoint', config.model.ollama_endpoint)
            )
        
        # 压缩配置
        if 'compression' in config_dict:
            comp_dict = config_dict['compression']
            config.compression = CompressionConfig(
                min_compress_length=comp_dict.get('min_compress_length', config.compression.min_compress_length),
                max_tokens=comp_dict.get('max_tokens', config.compression.max_tokens),
                temperature=comp_dict.get('temperature', config.compression.temperature),
                auto_compress_threshold=comp_dict.get('auto_compress_threshold', config.compression.auto_compress_threshold)
            )
        
        # 存储配置
        if 'storage' in config_dict:
            storage_dict = config_dict['storage']
            config.storage = StorageConfig(
                storage_path=storage_dict.get('storage_path', config.storage.storage_path),
                compression_level=storage_dict.get('compression_level', config.storage.compression_level),
                use_float16=storage_dict.get('use_float16', config.storage.use_float16)
            )
        
        # 性能配置
        if 'performance' in config_dict:
            perf_dict = config_dict['performance']
            config.performance = PerformanceConfig(
                batch_size=perf_dict.get('batch_size', config.performance.batch_size),
                max_concurrent=perf_dict.get('max_concurrent', config.performance.max_concurrent),
                cache_size=perf_dict.get('cache_size', config.performance.cache_size),
                cache_ttl=perf_dict.get('cache_ttl', config.performance.cache_ttl)
            )
        
        # 监控配置
        if 'monitoring' in config_dict:
            mon_dict = config_dict['monitoring']
            config.monitoring = MonitoringConfig(
                enable_prometheus=mon_dict.get('enable_prometheus', config.monitoring.enable_prometheus),
                prometheus_port=mon_dict.get('prometheus_port', config.monitoring.prometheus_port),
                alert_quality_threshold=mon_dict.get('alert_quality_threshold', config.monitoring.alert_quality_threshold)
            )
        
        return config
    
    def apply_env_overrides(self) -> None:
        """
        应用环境变量覆盖
        
        支持的环境变量：
        - LLM_CLOUD_ENDPOINT
        - LLM_CLOUD_API_KEY
        - LLM_TIMEOUT
        - LLM_MAX_RETRIES
        - LLM_RATE_LIMIT
        - MODEL_PREFER_LOCAL
        - OLLAMA_ENDPOINT
        - STORAGE_PATH
        - BATCH_SIZE
        - MAX_CONCURRENT
        """
        # LLM 配置
        if endpoint := os.getenv('LLM_CLOUD_ENDPOINT'):
            self.llm.cloud_endpoint = endpoint
            logger.info(f"Override cloud_endpoint from env: {endpoint}")
        
        if api_key := os.getenv('LLM_CLOUD_API_KEY'):
            self.llm.cloud_api_key = api_key
            logger.info("Override cloud_api_key from env")
        
        if timeout := os.getenv('LLM_TIMEOUT'):
            self.llm.timeout = float(timeout)
            logger.info(f"Override timeout from env: {timeout}")
        
        if max_retries := os.getenv('LLM_MAX_RETRIES'):
            self.llm.max_retries = int(max_retries)
            logger.info(f"Override max_retries from env: {max_retries}")
        
        if rate_limit := os.getenv('LLM_RATE_LIMIT'):
            self.llm.rate_limit = int(rate_limit)
            logger.info(f"Override rate_limit from env: {rate_limit}")
        
        # 模型配置
        if prefer_local := os.getenv('MODEL_PREFER_LOCAL'):
            self.model.prefer_local = prefer_local.lower() in ('true', '1', 'yes')
            logger.info(f"Override prefer_local from env: {self.model.prefer_local}")
        
        if ollama_endpoint := os.getenv('OLLAMA_ENDPOINT'):
            self.model.ollama_endpoint = ollama_endpoint
            logger.info(f"Override ollama_endpoint from env: {ollama_endpoint}")
        
        # 存储配置
        if storage_path := os.getenv('STORAGE_PATH'):
            self.storage.storage_path = storage_path
            logger.info(f"Override storage_path from env: {storage_path}")
        
        # 性能配置
        if batch_size := os.getenv('BATCH_SIZE'):
            self.performance.batch_size = int(batch_size)
            logger.info(f"Override batch_size from env: {batch_size}")
        
        if max_concurrent := os.getenv('MAX_CONCURRENT'):
            self.performance.max_concurrent = int(max_concurrent)
            logger.info(f"Override max_concurrent from env: {max_concurrent}")
    
    def validate(self) -> None:
        """
        验证配置有效性
        
        Raises:
            ValueError: 配置无效时抛出
        """
        # 验证 LLM 配置
        if self.llm.timeout <= 0:
            raise ValueError(f"Invalid timeout: {self.llm.timeout}, must be > 0")
        
        if self.llm.max_retries < 0:
            raise ValueError(f"Invalid max_retries: {self.llm.max_retries}, must be >= 0")
        
        if self.llm.rate_limit <= 0:
            raise ValueError(f"Invalid rate_limit: {self.llm.rate_limit}, must be > 0")
        
        # 验证压缩配置
        if self.compression.min_compress_length < 0:
            raise ValueError(f"Invalid min_compress_length: {self.compression.min_compress_length}, must be >= 0")
        
        if self.compression.max_tokens <= 0:
            raise ValueError(f"Invalid max_tokens: {self.compression.max_tokens}, must be > 0")
        
        if not 0 <= self.compression.temperature <= 1:
            raise ValueError(f"Invalid temperature: {self.compression.temperature}, must be in [0, 1]")
        
        # 验证存储配置
        storage_path = Path(self.storage.storage_path).expanduser()
        if not storage_path.parent.exists():
            logger.warning(f"Storage parent directory does not exist: {storage_path.parent}")
        
        if not 1 <= self.storage.compression_level <= 22:
            raise ValueError(f"Invalid compression_level: {self.storage.compression_level}, must be in [1, 22]")
        
        # 验证性能配置
        if self.performance.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.performance.batch_size}, must be > 0")
        
        if self.performance.max_concurrent <= 0:
            raise ValueError(f"Invalid max_concurrent: {self.performance.max_concurrent}, must be > 0")
        
        if self.performance.cache_size < 0:
            raise ValueError(f"Invalid cache_size: {self.performance.cache_size}, must be >= 0")
        
        if self.performance.cache_ttl < 0:
            raise ValueError(f"Invalid cache_ttl: {self.performance.cache_ttl}, must be >= 0")
        
        # 验证监控配置
        if not 0 <= self.monitoring.alert_quality_threshold <= 1:
            raise ValueError(f"Invalid alert_quality_threshold: {self.monitoring.alert_quality_threshold}, must be in [0, 1]")
        
        logger.info("Configuration validation passed")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    加载配置
    
    Args:
        config_path: 配置文件路径（可选）
        
    Returns:
        Config: 配置对象
    """
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        # 尝试从默认位置加载
        default_paths = [
            'config.yaml',
            'config/config.yaml',
            os.path.expanduser('~/.ai-os/config.yaml')
        ]
        
        config = None
        for path in default_paths:
            if Path(path).exists():
                logger.info(f"Loading config from {path}")
                config = Config.from_yaml(path)
                break
        
        if config is None:
            logger.info("No config file found, using defaults")
            config = Config()
    
    # 应用环境变量覆盖
    config.apply_env_overrides()
    
    # 验证配置
    config.validate()
    
    return config

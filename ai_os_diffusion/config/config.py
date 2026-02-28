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

from ..utils.logger import logger


@dataclass
class DiffusionConfig:
    """Diffusion 模型配置"""
    num_steps: int = 50  # 默认扩散步数
    distilled_steps: int = 4  # 蒸馏后步数
    noise_schedule: str = "cosine"  # 噪声调度策略
    sampler_type: str = "ddpm"  # 采样器类型


@dataclass
class ModelConfig:
    """模型配置"""
    model_path: str = "models/unified-diffusion"
    device: str = "cpu"
    use_quantization: bool = True
    quantization_bits: int = 2  # INT2 量化
    max_batch_size: int = 16


@dataclass
class StorageConfig:
    """存储配置"""
    storage_path: str = "~/.ai-os-diffusion/memory/"
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
class EvolutionConfig:
    """进化配置"""
    uncertainty_threshold: float = 0.7
    enable_l0_composition: bool = True
    enable_l1_controlnet: bool = True
    enable_l2_lora: bool = True
    enable_l3_finetune: bool = False


@dataclass
class Config:
    """系统配置"""
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    
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
        
        # Diffusion 配置
        if 'diffusion' in config_dict:
            diff_dict = config_dict['diffusion']
            config.diffusion = DiffusionConfig(
                num_steps=diff_dict.get('num_steps', config.diffusion.num_steps),
                distilled_steps=diff_dict.get('distilled_steps', config.diffusion.distilled_steps),
                noise_schedule=diff_dict.get('noise_schedule', config.diffusion.noise_schedule),
                sampler_type=diff_dict.get('sampler_type', config.diffusion.sampler_type)
            )
        
        # 模型配置
        if 'model' in config_dict:
            model_dict = config_dict['model']
            config.model = ModelConfig(
                model_path=model_dict.get('model_path', config.model.model_path),
                device=model_dict.get('device', config.model.device),
                use_quantization=model_dict.get('use_quantization', config.model.use_quantization),
                quantization_bits=model_dict.get('quantization_bits', config.model.quantization_bits),
                max_batch_size=model_dict.get('max_batch_size', config.model.max_batch_size)
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
        
        # 进化配置
        if 'evolution' in config_dict:
            evo_dict = config_dict['evolution']
            config.evolution = EvolutionConfig(
                uncertainty_threshold=evo_dict.get('uncertainty_threshold', config.evolution.uncertainty_threshold),
                enable_l0_composition=evo_dict.get('enable_l0_composition', config.evolution.enable_l0_composition),
                enable_l1_controlnet=evo_dict.get('enable_l1_controlnet', config.evolution.enable_l1_controlnet),
                enable_l2_lora=evo_dict.get('enable_l2_lora', config.evolution.enable_l2_lora),
                enable_l3_finetune=evo_dict.get('enable_l3_finetune', config.evolution.enable_l3_finetune)
            )
        
        return config
    
    def apply_env_overrides(self) -> None:
        """应用环境变量覆盖"""
        # 模型配置
        if model_path := os.getenv('MODEL_PATH'):
            self.model.model_path = model_path
            logger.info(f"Override model_path from env: {model_path}")
        
        if device := os.getenv('DEVICE'):
            self.model.device = device
            logger.info(f"Override device from env: {device}")
        
        # 存储配置
        if storage_path := os.getenv('STORAGE_PATH'):
            self.storage.storage_path = storage_path
            logger.info(f"Override storage_path from env: {storage_path}")
        
        # 性能配置
        if batch_size := os.getenv('BATCH_SIZE'):
            self.performance.batch_size = int(batch_size)
            logger.info(f"Override batch_size from env: {batch_size}")
    
    def validate(self) -> None:
        """验证配置有效性"""
        # 验证 Diffusion 配置
        if self.diffusion.num_steps <= 0:
            raise ValueError(f"Invalid num_steps: {self.diffusion.num_steps}, must be > 0")
        
        if self.diffusion.distilled_steps <= 0:
            raise ValueError(f"Invalid distilled_steps: {self.diffusion.distilled_steps}, must be > 0")
        
        # 验证模型配置
        if self.model.quantization_bits not in [2, 4, 8]:
            raise ValueError(f"Invalid quantization_bits: {self.model.quantization_bits}, must be 2, 4, or 8")
        
        # 验证存储配置
        if not 1 <= self.storage.compression_level <= 22:
            raise ValueError(f"Invalid compression_level: {self.storage.compression_level}, must be in [1, 22]")
        
        # 验证性能配置
        if self.performance.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.performance.batch_size}, must be > 0")
        
        logger.info("Configuration validation passed")


def load_config(config_path: Optional[str] = None) -> Config:
    """加载配置"""
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        # 尝试从默认位置加载
        default_paths = [
            'config.yaml',
            'config/config.yaml',
            os.path.expanduser('~/.ai-os-diffusion/config.yaml')
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

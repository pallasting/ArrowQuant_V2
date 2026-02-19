"""
配置模块单元测试
"""

import os
import pytest
from pathlib import Path
import tempfile
import yaml

from llm_compression.config import (
    Config,
    LLMConfig,
    ModelConfig,
    CompressionConfig,
    StorageConfig,
    PerformanceConfig,
    MonitoringConfig,
    load_config
)


class TestConfig:
    """配置类测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = Config()
        
        assert config.llm.cloud_endpoint == "http://localhost:8045"
        assert config.llm.timeout == 30.0
        assert config.llm.max_retries == 3
        assert config.compression.min_compress_length == 100
        assert config.storage.storage_path == "~/.ai-os/memory/"
    
    def test_from_yaml(self):
        """测试从 YAML 加载配置"""
        # 创建临时配置文件
        config_data = {
            'llm': {
                'cloud_endpoint': 'http://test:8045',
                'timeout': 60.0,
                'max_retries': 5
            },
            'compression': {
                'min_compress_length': 200,
                'max_tokens': 150
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config.from_yaml(temp_path)
            
            assert config.llm.cloud_endpoint == 'http://test:8045'
            assert config.llm.timeout == 60.0
            assert config.llm.max_retries == 5
            assert config.compression.min_compress_length == 200
            assert config.compression.max_tokens == 150
        finally:
            os.unlink(temp_path)
    
    def test_from_yaml_nonexistent_file(self):
        """测试加载不存在的配置文件"""
        config = Config.from_yaml('nonexistent.yaml')
        
        # 应该返回默认配置
        assert config.llm.cloud_endpoint == "http://localhost:8045"
    
    def test_env_override(self):
        """测试环境变量覆盖"""
        config = Config()
        
        # 设置环境变量
        os.environ['LLM_CLOUD_ENDPOINT'] = 'http://override:8045'
        os.environ['LLM_TIMEOUT'] = '45.0'
        os.environ['LLM_MAX_RETRIES'] = '10'
        os.environ['BATCH_SIZE'] = '32'
        
        try:
            config.apply_env_overrides()
            
            assert config.llm.cloud_endpoint == 'http://override:8045'
            assert config.llm.timeout == 45.0
            assert config.llm.max_retries == 10
            assert config.performance.batch_size == 32
        finally:
            # 清理环境变量
            for key in ['LLM_CLOUD_ENDPOINT', 'LLM_TIMEOUT', 'LLM_MAX_RETRIES', 'BATCH_SIZE']:
                os.environ.pop(key, None)
    
    def test_validate_success(self):
        """测试配置验证成功"""
        config = Config()
        
        # 不应该抛出异常
        config.validate()
    
    def test_validate_invalid_timeout(self):
        """测试无效的超时配置"""
        config = Config()
        config.llm.timeout = -1
        
        with pytest.raises(ValueError, match="Invalid timeout"):
            config.validate()
    
    def test_validate_invalid_max_retries(self):
        """测试无效的重试次数"""
        config = Config()
        config.llm.max_retries = -1
        
        with pytest.raises(ValueError, match="Invalid max_retries"):
            config.validate()
    
    def test_validate_invalid_temperature(self):
        """测试无效的温度"""
        config = Config()
        config.compression.temperature = 1.5
        
        with pytest.raises(ValueError, match="Invalid temperature"):
            config.validate()
    
    def test_validate_invalid_batch_size(self):
        """测试无效的批量大小"""
        config = Config()
        config.performance.batch_size = 0
        
        with pytest.raises(ValueError, match="Invalid batch_size"):
            config.validate()
    
    def test_validate_invalid_compression_level(self):
        """测试无效的压缩级别"""
        config = Config()
        config.storage.compression_level = 25
        
        with pytest.raises(ValueError, match="Invalid compression_level"):
            config.validate()


class TestLoadConfig:
    """load_config 函数测试"""
    
    def test_load_config_with_path(self):
        """测试指定路径加载配置"""
        config_data = {
            'llm': {
                'cloud_endpoint': 'http://custom:8045'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.llm.cloud_endpoint == 'http://custom:8045'
        finally:
            os.unlink(temp_path)
    
    def test_load_config_default(self):
        """测试默认加载配置"""
        config = load_config()
        
        # 应该返回有效的配置
        assert config is not None
        assert config.llm.cloud_endpoint is not None

"""
Property-based tests for configuration system

Tests configuration loading, environment variable overrides, and validation.
"""

import os
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
import yaml
import pytest

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


# Feature: llm-compression-integration, Property 29: 环境变量覆盖
# **Validates: Requirements 11.2**


# Strategies for generating valid configuration values
@st.composite
def valid_endpoint(draw):
    """Generate valid HTTP endpoint"""
    protocol = draw(st.sampled_from(['http', 'https']))
    host = draw(st.sampled_from(['localhost', '127.0.0.1', 'api.example.com']))
    port = draw(st.integers(min_value=1024, max_value=65535))
    return f"{protocol}://{host}:{port}"


@st.composite
def valid_timeout(draw):
    """Generate valid timeout value"""
    return draw(st.floats(min_value=0.1, max_value=300.0))


@st.composite
def valid_retries(draw):
    """Generate valid retry count"""
    return draw(st.integers(min_value=0, max_value=10))


@st.composite
def valid_rate_limit(draw):
    """Generate valid rate limit"""
    return draw(st.integers(min_value=1, max_value=1000))


@st.composite
def valid_batch_size(draw):
    """Generate valid batch size"""
    return draw(st.integers(min_value=1, max_value=128))


@st.composite
def valid_temperature(draw):
    """Generate valid temperature"""
    return draw(st.floats(min_value=0.0, max_value=1.0))


@st.composite
def valid_compression_level(draw):
    """Generate valid zstd compression level"""
    return draw(st.integers(min_value=1, max_value=22))


class TestEnvironmentVariableOverride:
    """
    Property 29: 环境变量覆盖
    
    Property: 环境变量应该能够覆盖配置文件中的所有关键配置项
    
    Validates: Requirements 11.2
    """
    
    @given(
        endpoint=valid_endpoint(),
        timeout=valid_timeout(),
        max_retries=valid_retries(),
        rate_limit=valid_rate_limit()
    )
    @settings(max_examples=100, deadline=None)
    def test_llm_config_env_override(self, endpoint, timeout, max_retries, rate_limit):
        """
        Property: LLM 配置项应该能够通过环境变量覆盖
        
        Given: 一个基础配置和环境变量
        When: 应用环境变量覆盖
        Then: 配置值应该被环境变量覆盖
        """
        # Create base config
        config = Config()
        original_endpoint = config.llm.cloud_endpoint
        
        # Set environment variables
        env_vars = {
            'LLM_CLOUD_ENDPOINT': endpoint,
            'LLM_TIMEOUT': str(timeout),
            'LLM_MAX_RETRIES': str(max_retries),
            'LLM_RATE_LIMIT': str(rate_limit)
        }
        
        # Apply overrides
        for key, value in env_vars.items():
            os.environ[key] = value
        
        try:
            config.apply_env_overrides()
            
            # Verify overrides
            assert config.llm.cloud_endpoint == endpoint, \
                f"Expected endpoint {endpoint}, got {config.llm.cloud_endpoint}"
            assert config.llm.timeout == timeout, \
                f"Expected timeout {timeout}, got {config.llm.timeout}"
            assert config.llm.max_retries == max_retries, \
                f"Expected max_retries {max_retries}, got {config.llm.max_retries}"
            assert config.llm.rate_limit == rate_limit, \
                f"Expected rate_limit {rate_limit}, got {config.llm.rate_limit}"
            
            # Verify config is still valid after override
            config.validate()
            
        finally:
            # Clean up environment variables
            for key in env_vars.keys():
                os.environ.pop(key, None)
    
    @given(
        prefer_local=st.booleans(),
        storage_path=st.text(min_size=1, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='/-_.'
        ))
    )
    @settings(max_examples=100, deadline=None)
    def test_model_and_storage_env_override(self, prefer_local, storage_path):
        """
        Property: 模型和存储配置应该能够通过环境变量覆盖
        
        Given: 一个基础配置和环境变量
        When: 应用环境变量覆盖
        Then: 配置值应该被环境变量覆盖
        """
        # Filter out invalid paths
        assume(not storage_path.startswith('.'))
        assume('..' not in storage_path)
        
        config = Config()
        
        # Set environment variables
        env_vars = {
            'MODEL_PREFER_LOCAL': 'true' if prefer_local else 'false',
            'STORAGE_PATH': storage_path
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        try:
            config.apply_env_overrides()
            
            # Verify overrides
            assert config.model.prefer_local == prefer_local, \
                f"Expected prefer_local {prefer_local}, got {config.model.prefer_local}"
            assert config.storage.storage_path == storage_path, \
                f"Expected storage_path {storage_path}, got {config.storage.storage_path}"
            
        finally:
            # Clean up
            for key in env_vars.keys():
                os.environ.pop(key, None)
    
    @given(
        batch_size=valid_batch_size(),
        max_concurrent=st.integers(min_value=1, max_value=32)
    )
    @settings(max_examples=100, deadline=None)
    def test_performance_config_env_override(self, batch_size, max_concurrent):
        """
        Property: 性能配置应该能够通过环境变量覆盖
        
        Given: 一个基础配置和环境变量
        When: 应用环境变量覆盖
        Then: 配置值应该被环境变量覆盖
        """
        config = Config()
        
        # Set environment variables
        env_vars = {
            'BATCH_SIZE': str(batch_size),
            'MAX_CONCURRENT': str(max_concurrent)
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        try:
            config.apply_env_overrides()
            
            # Verify overrides
            assert config.performance.batch_size == batch_size, \
                f"Expected batch_size {batch_size}, got {config.performance.batch_size}"
            assert config.performance.max_concurrent == max_concurrent, \
                f"Expected max_concurrent {max_concurrent}, got {config.performance.max_concurrent}"
            
            # Verify config is still valid
            config.validate()
            
        finally:
            # Clean up
            for key in env_vars.keys():
                os.environ.pop(key, None)
    
    @given(
        api_key=st.text(min_size=10, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='-_'
        ))
    )
    @settings(max_examples=100, deadline=None)
    def test_api_key_env_override(self, api_key):
        """
        Property: API 密钥应该能够通过环境变量设置（安全性）
        
        Given: 一个基础配置和 API 密钥环境变量
        When: 应用环境变量覆盖
        Then: API 密钥应该被设置
        """
        config = Config()
        assert config.llm.cloud_api_key is None
        
        os.environ['LLM_CLOUD_API_KEY'] = api_key
        
        try:
            config.apply_env_overrides()
            
            # Verify API key is set
            assert config.llm.cloud_api_key == api_key, \
                f"Expected API key to be set"
            
        finally:
            os.environ.pop('LLM_CLOUD_API_KEY', None)
    
    def test_env_override_precedence(self):
        """
        Property: 环境变量应该优先于配置文件
        
        Given: 配置文件和环境变量都设置了相同的配置项
        When: 加载配置
        Then: 环境变量的值应该优先
        """
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'llm': {
                    'cloud_endpoint': 'http://localhost:8045',
                    'timeout': 30.0
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Set environment variable with different value
            os.environ['LLM_CLOUD_ENDPOINT'] = 'http://localhost:9999'
            os.environ['LLM_TIMEOUT'] = '60.0'
            
            # Load config
            config = Config.from_yaml(config_path)
            config.apply_env_overrides()
            
            # Verify environment variable takes precedence
            assert config.llm.cloud_endpoint == 'http://localhost:9999', \
                "Environment variable should override config file"
            assert config.llm.timeout == 60.0, \
                "Environment variable should override config file"
            
        finally:
            os.environ.pop('LLM_CLOUD_ENDPOINT', None)
            os.environ.pop('LLM_TIMEOUT', None)
            Path(config_path).unlink()


# Feature: llm-compression-integration, Property 30: 配置验证
# **Validates: Requirements 11.4**


class TestConfigurationValidation:
    """
    Property 30: 配置验证
    
    Property: 配置系统应该在启动时验证所有配置项的有效性
    
    Validates: Requirements 11.4
    """
    
    @given(
        timeout=st.floats(min_value=-100.0, max_value=0.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_timeout_rejected(self, timeout):
        """
        Property: 无效的超时值应该被拒绝
        
        Given: 一个负数或零的超时值
        When: 验证配置
        Then: 应该抛出 ValueError
        """
        config = Config()
        config.llm.timeout = timeout
        
        with pytest.raises(ValueError, match="Invalid timeout"):
            config.validate()
    
    @given(
        max_retries=st.integers(max_value=-1)
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_retries_rejected(self, max_retries):
        """
        Property: 负数的重试次数应该被拒绝
        
        Given: 一个负数的重试次数
        When: 验证配置
        Then: 应该抛出 ValueError
        """
        config = Config()
        config.llm.max_retries = max_retries
        
        with pytest.raises(ValueError, match="Invalid max_retries"):
            config.validate()
    
    @given(
        temperature=st.floats(min_value=-1.0, max_value=-0.01) | 
                    st.floats(min_value=1.01, max_value=2.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_temperature_rejected(self, temperature):
        """
        Property: 超出范围的温度值应该被拒绝
        
        Given: 一个不在 [0, 1] 范围内的温度值
        When: 验证配置
        Then: 应该抛出 ValueError
        """
        config = Config()
        config.compression.temperature = temperature
        
        with pytest.raises(ValueError, match="Invalid temperature"):
            config.validate()
    
    @given(
        compression_level=st.integers(max_value=0) | st.integers(min_value=23)
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_compression_level_rejected(self, compression_level):
        """
        Property: 超出范围的压缩级别应该被拒绝
        
        Given: 一个不在 [1, 22] 范围内的压缩级别
        When: 验证配置
        Then: 应该抛出 ValueError
        """
        config = Config()
        config.storage.compression_level = compression_level
        
        with pytest.raises(ValueError, match="Invalid compression_level"):
            config.validate()
    
    @given(
        batch_size=st.integers(max_value=0)
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_batch_size_rejected(self, batch_size):
        """
        Property: 非正数的批量大小应该被拒绝
        
        Given: 一个非正数的批量大小
        When: 验证配置
        Then: 应该抛出 ValueError
        """
        config = Config()
        config.performance.batch_size = batch_size
        
        with pytest.raises(ValueError, match="Invalid batch_size"):
            config.validate()
    
    @given(
        alert_threshold=st.floats(min_value=-1.0, max_value=-0.01) |
                       st.floats(min_value=1.01, max_value=2.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_alert_threshold_rejected(self, alert_threshold):
        """
        Property: 超出范围的告警阈值应该被拒绝
        
        Given: 一个不在 [0, 1] 范围内的告警阈值
        When: 验证配置
        Then: 应该抛出 ValueError
        """
        config = Config()
        config.monitoring.alert_quality_threshold = alert_threshold
        
        with pytest.raises(ValueError, match="Invalid alert_quality_threshold"):
            config.validate()
    
    @given(
        timeout=valid_timeout(),
        max_retries=valid_retries(),
        temperature=valid_temperature(),
        compression_level=valid_compression_level(),
        batch_size=valid_batch_size()
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_config_accepted(self, timeout, max_retries, temperature, 
                                   compression_level, batch_size):
        """
        Property: 所有有效的配置值应该通过验证
        
        Given: 一组有效的配置值
        When: 验证配置
        Then: 不应该抛出异常
        """
        config = Config()
        config.llm.timeout = timeout
        config.llm.max_retries = max_retries
        config.compression.temperature = temperature
        config.storage.compression_level = compression_level
        config.performance.batch_size = batch_size
        
        # Should not raise
        config.validate()
    
    def test_config_validation_on_load(self):
        """
        Property: 配置加载时应该自动验证
        
        Given: 一个包含无效配置的文件
        When: 加载配置
        Then: 应该抛出 ValueError
        """
        # Create a config file with invalid values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'llm': {
                    'timeout': -10.0  # Invalid
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Load config should fail validation
            with pytest.raises(ValueError):
                load_config(config_path)
        finally:
            Path(config_path).unlink()


# Feature: llm-compression-integration, Property 28: 配置项支持完整性
# **Validates: Requirements 1.5, 11.1**


class TestConfigurationCompleteness:
    """
    Property 28: 配置项支持完整性
    
    Property: 配置系统应该支持所有需求中定义的配置项
    
    Validates: Requirements 1.5, 11.1
    """
    
    def test_llm_config_completeness(self):
        """
        Property: LLM 配置应该包含所有必需的配置项
        
        Given: LLM 配置类
        When: 检查配置项
        Then: 应该包含所有必需的字段
        """
        config = LLMConfig()
        
        # Required fields from Requirement 1.5
        assert hasattr(config, 'cloud_endpoint'), "Missing cloud_endpoint"
        assert hasattr(config, 'cloud_api_key'), "Missing cloud_api_key"
        assert hasattr(config, 'timeout'), "Missing timeout"
        assert hasattr(config, 'max_retries'), "Missing max_retries"
        assert hasattr(config, 'rate_limit'), "Missing rate_limit"
    
    def test_model_config_completeness(self):
        """
        Property: 模型配置应该包含所有必需的配置项
        
        Given: 模型配置类
        When: 检查配置项
        Then: 应该包含所有必需的字段
        """
        config = ModelConfig()
        
        # Required fields from Requirement 11.1
        assert hasattr(config, 'prefer_local'), "Missing prefer_local"
        assert hasattr(config, 'local_endpoints'), "Missing local_endpoints"
        assert hasattr(config, 'quality_threshold'), "Missing quality_threshold"
    
    def test_compression_config_completeness(self):
        """
        Property: 压缩配置应该包含所有必需的配置项
        
        Given: 压缩配置类
        When: 检查配置项
        Then: 应该包含所有必需的字段
        """
        config = CompressionConfig()
        
        # Required fields from Requirement 11.1
        assert hasattr(config, 'min_compress_length'), "Missing min_compress_length"
        assert hasattr(config, 'max_tokens'), "Missing max_tokens"
        assert hasattr(config, 'temperature'), "Missing temperature"
        assert hasattr(config, 'auto_compress_threshold'), "Missing auto_compress_threshold"
    
    def test_storage_config_completeness(self):
        """
        Property: 存储配置应该包含所有必需的配置项
        
        Given: 存储配置类
        When: 检查配置项
        Then: 应该包含所有必需的字段
        """
        config = StorageConfig()
        
        # Required fields from Requirement 11.1
        assert hasattr(config, 'storage_path'), "Missing storage_path"
        assert hasattr(config, 'compression_level'), "Missing compression_level"
        assert hasattr(config, 'use_float16'), "Missing use_float16"
    
    def test_performance_config_completeness(self):
        """
        Property: 性能配置应该包含所有必需的配置项
        
        Given: 性能配置类
        When: 检查配置项
        Then: 应该包含所有必需的字段
        """
        config = PerformanceConfig()
        
        # Required fields from Requirement 11.1
        assert hasattr(config, 'batch_size'), "Missing batch_size"
        assert hasattr(config, 'max_concurrent'), "Missing max_concurrent"
        assert hasattr(config, 'cache_size'), "Missing cache_size"
        assert hasattr(config, 'cache_ttl'), "Missing cache_ttl"
    
    def test_monitoring_config_completeness(self):
        """
        Property: 监控配置应该包含所有必需的配置项
        
        Given: 监控配置类
        When: 检查配置项
        Then: 应该包含所有必需的字段
        """
        config = MonitoringConfig()
        
        # Required fields from Requirement 11.1
        assert hasattr(config, 'enable_prometheus'), "Missing enable_prometheus"
        assert hasattr(config, 'prometheus_port'), "Missing prometheus_port"
        assert hasattr(config, 'alert_quality_threshold'), "Missing alert_quality_threshold"
    
    def test_config_yaml_completeness(self):
        """
        Property: config.yaml 模板应该包含所有配置项
        
        Given: config.yaml 文件
        When: 加载配置
        Then: 应该包含所有配置类别
        """
        config_path = Path('config.yaml')
        if not config_path.exists():
            pytest.skip("config.yaml not found")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Check all sections exist
        assert 'llm' in config_data, "Missing llm section"
        assert 'model' in config_data, "Missing model section"
        assert 'compression' in config_data, "Missing compression section"
        assert 'storage' in config_data, "Missing storage section"
        assert 'performance' in config_data, "Missing performance section"
        assert 'monitoring' in config_data, "Missing monitoring section"
    
    def test_config_from_yaml_preserves_all_fields(self):
        """
        Property: 从 YAML 加载配置应该保留所有字段
        
        Given: 一个完整的配置文件
        When: 加载配置
        Then: 所有字段应该被正确加载
        """
        # Create a complete config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'llm': {
                    'cloud_endpoint': 'http://test:8045',
                    'timeout': 45.0,
                    'max_retries': 5,
                    'rate_limit': 120
                },
                'model': {
                    'prefer_local': False,
                    'quality_threshold': 0.9
                },
                'compression': {
                    'min_compress_length': 200,
                    'max_tokens': 150,
                    'temperature': 0.5
                },
                'storage': {
                    'storage_path': '/tmp/test',
                    'compression_level': 5
                },
                'performance': {
                    'batch_size': 32,
                    'max_concurrent': 8
                },
                'monitoring': {
                    'enable_prometheus': True,
                    'prometheus_port': 9091
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = Config.from_yaml(config_path)
            
            # Verify all fields are loaded correctly
            assert config.llm.cloud_endpoint == 'http://test:8045'
            assert config.llm.timeout == 45.0
            assert config.llm.max_retries == 5
            assert config.llm.rate_limit == 120
            
            assert config.model.prefer_local == False
            assert config.model.quality_threshold == 0.9
            
            assert config.compression.min_compress_length == 200
            assert config.compression.max_tokens == 150
            assert config.compression.temperature == 0.5
            
            assert config.storage.storage_path == '/tmp/test'
            assert config.storage.compression_level == 5
            
            assert config.performance.batch_size == 32
            assert config.performance.max_concurrent == 8
            
            assert config.monitoring.enable_prometheus == True
            assert config.monitoring.prometheus_port == 9091
            
        finally:
            Path(config_path).unlink()

"""
配置系统使用示例

演示如何加载、验证和使用配置系统。
"""

import os
from llm_compression.config import Config, load_config
from llm_compression.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)


def example_basic_config_loading():
    """示例 1: 基本配置加载"""
    print("\n=== 示例 1: 基本配置加载 ===")
    
    # 从默认位置加载配置
    config = load_config()
    
    print(f"LLM 端点: {config.llm.cloud_endpoint}")
    print(f"超时时间: {config.llm.timeout}s")
    print(f"批量大小: {config.performance.batch_size}")
    print(f"存储路径: {config.storage.storage_path}")


def example_custom_config_file():
    """示例 2: 从自定义配置文件加载"""
    print("\n=== 示例 2: 从自定义配置文件加载 ===")
    
    # 从指定路径加载配置
    config = load_config('config.yaml')
    
    print(f"模型选择策略: {'优先本地' if config.model.prefer_local else '优先云端'}")
    print(f"质量阈值: {config.model.quality_threshold}")
    print(f"压缩温度: {config.compression.temperature}")


def example_env_override():
    """示例 3: 环境变量覆盖"""
    print("\n=== 示例 3: 环境变量覆盖 ===")
    
    # 设置环境变量
    os.environ['LLM_CLOUD_ENDPOINT'] = 'http://localhost:9999'
    os.environ['BATCH_SIZE'] = '32'
    os.environ['MAX_CONCURRENT'] = '8'
    
    # 加载配置（会自动应用环境变量覆盖）
    config = load_config()
    
    print(f"LLM 端点（被环境变量覆盖）: {config.llm.cloud_endpoint}")
    print(f"批量大小（被环境变量覆盖）: {config.performance.batch_size}")
    print(f"最大并发（被环境变量覆盖）: {config.performance.max_concurrent}")
    
    # 清理环境变量
    os.environ.pop('LLM_CLOUD_ENDPOINT', None)
    os.environ.pop('BATCH_SIZE', None)
    os.environ.pop('MAX_CONCURRENT', None)


def example_config_validation():
    """示例 4: 配置验证"""
    print("\n=== 示例 4: 配置验证 ===")
    
    # 创建配置
    config = Config()
    
    # 修改配置
    config.llm.timeout = 60.0
    config.compression.temperature = 0.5
    config.performance.batch_size = 64
    
    # 验证配置
    try:
        config.validate()
        print("✓ 配置验证通过")
    except ValueError as e:
        print(f"✗ 配置验证失败: {e}")
    
    # 尝试无效配置
    config.llm.timeout = -10.0  # 无效值
    try:
        config.validate()
        print("✓ 配置验证通过")
    except ValueError as e:
        print(f"✗ 配置验证失败（预期）: {e}")


def example_accessing_config_values():
    """示例 5: 访问配置值"""
    print("\n=== 示例 5: 访问配置值 ===")
    
    config = load_config()
    
    # 访问 LLM 配置
    print("\nLLM 配置:")
    print(f"  - 云端端点: {config.llm.cloud_endpoint}")
    print(f"  - 超时时间: {config.llm.timeout}s")
    print(f"  - 最大重试: {config.llm.max_retries}")
    print(f"  - 速率限制: {config.llm.rate_limit} req/min")
    
    # 访问模型配置
    print("\n模型配置:")
    print(f"  - 优先本地: {config.model.prefer_local}")
    print(f"  - 质量阈值: {config.model.quality_threshold}")
    print(f"  - 本地端点: {config.model.local_endpoints}")
    
    # 访问压缩配置
    print("\n压缩配置:")
    print(f"  - 最小压缩长度: {config.compression.min_compress_length}")
    print(f"  - 最大 tokens: {config.compression.max_tokens}")
    print(f"  - 温度: {config.compression.temperature}")
    
    # 访问存储配置
    print("\n存储配置:")
    print(f"  - 存储路径: {config.storage.storage_path}")
    print(f"  - 压缩级别: {config.storage.compression_level}")
    print(f"  - 使用 float16: {config.storage.use_float16}")
    
    # 访问性能配置
    print("\n性能配置:")
    print(f"  - 批量大小: {config.performance.batch_size}")
    print(f"  - 最大并发: {config.performance.max_concurrent}")
    print(f"  - 缓存大小: {config.performance.cache_size}")
    print(f"  - 缓存 TTL: {config.performance.cache_ttl}s")
    
    # 访问监控配置
    print("\n监控配置:")
    print(f"  - 启用 Prometheus: {config.monitoring.enable_prometheus}")
    print(f"  - Prometheus 端口: {config.monitoring.prometheus_port}")
    print(f"  - 告警阈值: {config.monitoring.alert_quality_threshold}")


def example_programmatic_config():
    """示例 6: 编程方式创建配置"""
    print("\n=== 示例 6: 编程方式创建配置 ===")
    
    from llm_compression.config import (
        LLMConfig, ModelConfig, CompressionConfig,
        StorageConfig, PerformanceConfig, MonitoringConfig
    )
    
    # 创建自定义配置
    config = Config(
        llm=LLMConfig(
            cloud_endpoint='http://custom-api:8045',
            timeout=45.0,
            max_retries=5
        ),
        compression=CompressionConfig(
            min_compress_length=200,
            max_tokens=150,
            temperature=0.4
        ),
        performance=PerformanceConfig(
            batch_size=32,
            max_concurrent=8
        )
    )
    
    # 验证配置
    config.validate()
    
    print(f"自定义 LLM 端点: {config.llm.cloud_endpoint}")
    print(f"自定义批量大小: {config.performance.batch_size}")
    print(f"自定义压缩温度: {config.compression.temperature}")


def example_config_in_application():
    """示例 7: 在应用中使用配置"""
    print("\n=== 示例 7: 在应用中使用配置 ===")
    
    # 加载配置
    config = load_config()
    
    # 使用配置初始化组件
    print("\n初始化系统组件...")
    
    # 1. LLM 客户端
    print(f"✓ LLM 客户端: {config.llm.cloud_endpoint}")
    print(f"  - 超时: {config.llm.timeout}s")
    print(f"  - 重试: {config.llm.max_retries}次")
    
    # 2. 压缩器
    print(f"✓ 压缩器:")
    print(f"  - 最小长度: {config.compression.min_compress_length}")
    print(f"  - 温度: {config.compression.temperature}")
    
    # 3. 存储
    print(f"✓ 存储:")
    print(f"  - 路径: {config.storage.storage_path}")
    print(f"  - 压缩级别: {config.storage.compression_level}")
    
    # 4. 批处理器
    print(f"✓ 批处理器:")
    print(f"  - 批量大小: {config.performance.batch_size}")
    print(f"  - 并发数: {config.performance.max_concurrent}")
    
    print("\n所有组件初始化完成！")


def example_config_best_practices():
    """示例 8: 配置最佳实践"""
    print("\n=== 示例 8: 配置最佳实践 ===")
    
    print("\n最佳实践:")
    print("1. 使用配置文件存储默认值")
    print("2. 使用环境变量覆盖敏感信息（如 API 密钥）")
    print("3. 在应用启动时验证配置")
    print("4. 为不同环境使用不同的配置文件")
    print("5. 不要在代码中硬编码配置值")
    
    print("\n示例环境变量设置:")
    print("  export LLM_CLOUD_API_KEY='your-secret-key'")
    print("  export LLM_CLOUD_ENDPOINT='http://production-api:8045'")
    print("  export STORAGE_PATH='/var/lib/ai-os/memory'")
    print("  export BATCH_SIZE=64")
    
    print("\n示例配置文件结构:")
    print("  - config.yaml          # 默认配置")
    print("  - config.dev.yaml      # 开发环境")
    print("  - config.prod.yaml     # 生产环境")
    print("  - config.test.yaml     # 测试环境")


if __name__ == '__main__':
    print("=" * 60)
    print("配置系统使用示例")
    print("=" * 60)
    
    # 运行所有示例
    example_basic_config_loading()
    example_custom_config_file()
    example_env_override()
    example_config_validation()
    example_accessing_config_values()
    example_programmatic_config()
    example_config_in_application()
    example_config_best_practices()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)

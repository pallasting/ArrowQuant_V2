"""
健康检查示例

演示如何使用健康检查系统。

Requirements: 11.7
"""

import asyncio
from llm_compression.health import HealthChecker
from llm_compression.config import Config
from llm_compression.llm_client import LLMClient
from llm_compression.arrow_storage import ArrowStorage


async def main():
    """主函数"""
    print("=" * 60)
    print("LLM Compression System - Health Check Example")
    print("=" * 60)
    print()
    
    # 1. 加载配置
    print("1. Loading configuration...")
    config = Config.from_yaml("config.yaml")
    print(f"   LLM Endpoint: {config.llm.cloud_endpoint}")
    print(f"   Storage Path: {config.storage.storage_path}")
    print()
    
    # 2. 初始化组件
    print("2. Initializing components...")
    
    # LLM 客户端
    llm_client = LLMClient(
        endpoint=config.llm.cloud_endpoint,
        api_key=config.llm.cloud_api_key,
        timeout=config.llm.timeout,
        max_retries=config.llm.max_retries,
        rate_limit=config.llm.rate_limit
    )
    print("   ✓ LLM Client initialized")
    
    # 存储
    storage = ArrowStorage(
        base_path=config.storage.storage_path,
        compression_level=config.storage.compression_level
    )
    print("   ✓ Storage initialized")
    print()
    
    # 3. 创建健康检查器
    print("3. Creating health checker...")
    checker = HealthChecker(
        llm_client=llm_client,
        storage=storage,
        config=config
    )
    print("   ✓ Health checker created")
    print()
    
    # 4. 执行健康检查
    print("4. Running health check...")
    print()
    
    result = await checker.check_health()
    
    # 5. 显示结果
    print("=" * 60)
    print("Health Check Results")
    print("=" * 60)
    print()
    
    # 总体状态
    status_emoji = {
        "healthy": "✅",
        "degraded": "⚠️",
        "unhealthy": "❌"
    }
    
    print(f"Overall Status: {status_emoji.get(result.overall_status, '?')} {result.overall_status.upper()}")
    print(f"Timestamp: {result.timestamp}")
    print()
    
    # 组件详情
    print("Component Details:")
    print("-" * 60)
    
    for name, component in result.components.items():
        emoji = status_emoji.get(component.status, '?')
        print(f"\n{emoji} {name.upper()}")
        print(f"   Status: {component.status}")
        print(f"   Message: {component.message}")
        
        if component.latency_ms is not None:
            print(f"   Latency: {component.latency_ms:.2f}ms")
        
        if component.details:
            print("   Details:")
            for key, value in component.details.items():
                print(f"     - {key}: {value}")
    
    print()
    print("=" * 60)
    
    # 6. JSON 输出
    print()
    print("JSON Output:")
    print("-" * 60)
    import json
    print(json.dumps(result.to_dict(), indent=2))
    
    # 7. 建议
    print()
    print("=" * 60)
    print("Recommendations")
    print("=" * 60)
    print()
    
    if result.overall_status == "healthy":
        print("✅ System is healthy and ready for production use.")
    elif result.overall_status == "degraded":
        print("⚠️  System is operational but has some issues:")
        for name, comp in result.components.items():
            if comp.status == "degraded":
                print(f"   - {name}: {comp.message}")
        print()
        print("   Consider addressing these issues for optimal performance.")
    else:
        print("❌ System is unhealthy and may not function correctly:")
        for name, comp in result.components.items():
            if comp.status == "unhealthy":
                print(f"   - {name}: {comp.message}")
        print()
        print("   Please fix these critical issues before using the system.")
    
    print()


if __name__ == "__main__":
    asyncio.run(main())

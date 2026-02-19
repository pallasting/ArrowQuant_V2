"""
本地模型集成示例

演示如何使用本地部署的 LLM 模型（Ollama）进行压缩和重构。
展示本地模型优先策略和云端 API 降级机制。
"""

import asyncio
from llm_compression.model_selector import ModelSelector, MemoryType, QualityLevel
from llm_compression.config import load_config


async def main():
    """主函数"""
    
    print("=" * 60)
    print("本地模型集成示例")
    print("=" * 60)
    
    # 1. 加载配置
    print("\n1. 加载配置...")
    config = load_config()
    
    # 2. 创建模型选择器（本地模型优先）
    print("\n2. 创建模型选择器（本地模型优先）...")
    selector = ModelSelector(
        cloud_endpoint=config.llm.cloud_endpoint,
        ollama_endpoint=config.model.ollama_endpoint,
        prefer_local=True,  # 优先使用本地模型
        quality_threshold=0.85
    )
    
    print(f"   - 云端 API: {config.llm.cloud_endpoint}")
    print(f"   - Ollama 端点: {config.model.ollama_endpoint}")
    print(f"   - 本地模型优先: {selector.prefer_local}")
    
    # 3. 测试模型选择（不同场景）
    print("\n3. 测试模型选择...")
    
    # 场景 1: 普通文本（< 500 字）
    print("\n   场景 1: 普通文本（< 500 字）")
    model_config = selector.select_model(
        memory_type=MemoryType.TEXT,
        text_length=300,
        quality_requirement=QualityLevel.STANDARD
    )
    print(f"   - 选择模型: {model_config.model_name}")
    print(f"   - 端点: {model_config.endpoint}")
    print(f"   - 是否本地: {model_config.is_local}")
    print(f"   - 预期延迟: {model_config.expected_latency_ms}ms")
    print(f"   - 预期质量: {model_config.expected_quality}")
    
    # 场景 2: 长文本（> 500 字）
    print("\n   场景 2: 长文本（> 500 字）")
    model_config = selector.select_model(
        memory_type=MemoryType.LONG_TEXT,
        text_length=1000,
        quality_requirement=QualityLevel.STANDARD
    )
    print(f"   - 选择模型: {model_config.model_name}")
    print(f"   - 端点: {model_config.endpoint}")
    print(f"   - 是否本地: {model_config.is_local}")
    
    # 场景 3: 高质量要求（应该选择云端 API）
    print("\n   场景 3: 高质量要求")
    model_config = selector.select_model(
        memory_type=MemoryType.TEXT,
        text_length=300,
        quality_requirement=QualityLevel.HIGH
    )
    print(f"   - 选择模型: {model_config.model_name}")
    print(f"   - 端点: {model_config.endpoint}")
    print(f"   - 是否本地: {model_config.is_local}")
    print(f"   - 说明: 高质量要求优先使用云端 API")
    
    # 场景 4: 手动指定模型
    print("\n   场景 4: 手动指定模型（llama3.1）")
    model_config = selector.select_model(
        memory_type=MemoryType.TEXT,
        text_length=300,
        quality_requirement=QualityLevel.STANDARD,
        manual_model="llama3.1"
    )
    print(f"   - 选择模型: {model_config.model_name}")
    print(f"   - 端点: {model_config.endpoint}")
    print(f"   - 是否本地: {model_config.is_local}")
    
    # 4. 测试降级策略
    print("\n4. 测试降级策略...")
    print("   模拟本地模型不可用的情况...")
    
    # 创建一个没有本地模型的选择器
    selector_no_local = ModelSelector(
        cloud_endpoint=config.llm.cloud_endpoint,
        local_endpoints={},  # 没有本地模型
        prefer_local=True,
        quality_threshold=0.85
    )
    
    model_config = selector_no_local.select_model(
        memory_type=MemoryType.TEXT,
        text_length=300,
        quality_requirement=QualityLevel.STANDARD
    )
    print(f"   - 降级到: {model_config.model_name}")
    print(f"   - 端点: {model_config.endpoint}")
    print(f"   - 说明: 本地模型不可用时自动降级到云端 API")
    
    # 5. 查看模型统计
    print("\n5. 模型统计...")
    stats = selector.get_model_stats()
    if stats:
        for model_name, model_stats in stats.items():
            print(f"\n   模型: {model_name}")
            print(f"   - 总请求数: {model_stats.total_requests}")
            print(f"   - 平均延迟: {model_stats.avg_latency_ms:.2f}ms")
            print(f"   - 平均质量: {model_stats.avg_quality_score:.2f}")
            print(f"   - 成功率: {model_stats.success_rate:.2%}")
    else:
        print("   暂无统计数据（需要实际使用后才有数据）")
    
    # 6. 成本对比
    print("\n6. 成本对比（估算）...")
    print("   - 云端 API: ~$0.001/1K tokens")
    print("   - 本地模型: ~$0.0001/1K tokens (电费)")
    print("   - 节省: 90%")
    print("   - 说明: 使用本地模型可以大幅降低运营成本")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    
    print("\n下一步:")
    print("1. 确保 Ollama 服务正在运行: ollama serve")
    print("2. 确保模型已下载: ollama pull qwen2.5:7b-instruct")
    print("3. 运行压缩示例: python examples/compression_example.py")
    print("4. 查看性能对比: python examples/performance_comparison.py")


if __name__ == "__main__":
    asyncio.run(main())

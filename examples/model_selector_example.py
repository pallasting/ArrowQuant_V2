"""
ModelSelector 使用示例

演示如何使用 ModelSelector 选择最优的 LLM 模型。
"""

import asyncio
from llm_compression import (
    ModelSelector,
    MemoryType,
    QualityLevel
)


async def main():
    """主函数"""
    
    print("=" * 60)
    print("ModelSelector 使用示例")
    print("=" * 60)
    print()
    
    # 1. 创建模型选择器（带本地模型）
    print("1. 创建模型选择器（本地优先）")
    selector = ModelSelector(
        cloud_endpoint="http://localhost:8045",
        local_endpoints={
            "step-flash": "http://localhost:8046",
            "minicpm-o": "http://localhost:8047",
            "stable-diffcoder": "http://localhost:8048",
            "intern-s1-pro": "http://localhost:8049"
        },
        prefer_local=True,
        quality_threshold=0.85
    )
    print("✓ 模型选择器已创建")
    print()
    
    # 2. 选择模型 - 短文本
    print("2. 选择模型 - 短文本（< 500 字）")
    config = selector.select_model(
        memory_type=MemoryType.TEXT,
        text_length=200,
        quality_requirement=QualityLevel.STANDARD
    )
    print(f"   选择的模型: {config.model_name}")
    print(f"   是否本地: {config.is_local}")
    print(f"   端点: {config.endpoint}")
    print(f"   预期延迟: {config.expected_latency_ms}ms")
    print(f"   预期质量: {config.expected_quality}")
    print()
    
    # 3. 选择模型 - 长文本
    print("3. 选择模型 - 长文本（> 500 字）")
    config = selector.select_model(
        memory_type=MemoryType.LONG_TEXT,
        text_length=1000,
        quality_requirement=QualityLevel.STANDARD
    )
    print(f"   选择的模型: {config.model_name}")
    print(f"   是否本地: {config.is_local}")
    print()
    
    # 4. 选择模型 - 代码记忆
    print("4. 选择模型 - 代码记忆")
    config = selector.select_model(
        memory_type=MemoryType.CODE,
        text_length=500,
        quality_requirement=QualityLevel.STANDARD
    )
    print(f"   选择的模型: {config.model_name}")
    print(f"   是否本地: {config.is_local}")
    print()
    
    # 5. 选择模型 - 多模态记忆
    print("5. 选择模型 - 多模态记忆")
    config = selector.select_model(
        memory_type=MemoryType.MULTIMODAL,
        text_length=300,
        quality_requirement=QualityLevel.STANDARD
    )
    print(f"   选择的模型: {config.model_name}")
    print(f"   是否本地: {config.is_local}")
    print()
    
    # 6. 选择模型 - 高质量要求
    print("6. 选择模型 - 高质量要求")
    config = selector.select_model(
        memory_type=MemoryType.TEXT,
        text_length=200,
        quality_requirement=QualityLevel.HIGH
    )
    print(f"   选择的模型: {config.model_name}")
    print(f"   是否本地: {config.is_local}")
    print()
    
    # 7. 手动指定模型
    print("7. 手动指定模型")
    config = selector.select_model(
        memory_type=MemoryType.TEXT,
        text_length=200,
        quality_requirement=QualityLevel.STANDARD,
        manual_model="cloud-api"
    )
    print(f"   选择的模型: {config.model_name}")
    print()
    
    # 8. 记录模型使用统计
    print("8. 记录模型使用统计")
    await selector.record_usage(
        model_name="step-flash",
        latency_ms=450.0,
        quality_score=0.92,
        tokens_used=85,
        success=True
    )
    await selector.record_usage(
        model_name="step-flash",
        latency_ms=520.0,
        quality_score=0.88,
        tokens_used=95,
        success=True
    )
    print("✓ 已记录 2 次使用")
    print()
    
    # 9. 查看模型统计
    print("9. 查看模型统计")
    stats = selector.get_model_stats()
    if "step-flash" in stats:
        model_stats = stats["step-flash"]
        print(f"   模型: step-flash")
        print(f"   总请求数: {model_stats.total_requests}")
        print(f"   平均延迟: {model_stats.avg_latency_ms:.1f}ms")
        print(f"   平均质量: {model_stats.avg_quality_score:.2f}")
        print(f"   成功率: {model_stats.success_rate:.2%}")
        print(f"   总 tokens: {model_stats.total_tokens_used}")
    print()
    
    # 10. 质量监控和建议
    print("10. 质量监控和建议")
    # 记录低质量使用
    await selector.record_usage(
        model_name="test-model",
        latency_ms=500.0,
        quality_score=0.75,  # 低于阈值 0.85
        tokens_used=100,
        success=True
    )
    suggestion = selector.suggest_model_switch("test-model")
    if suggestion:
        print(f"   ⚠️  建议切换到: {suggestion}")
    else:
        print("   ✓ 质量达标，无需切换")
    print()
    
    # 11. 仅云端模式
    print("11. 仅云端模式")
    cloud_selector = ModelSelector(
        cloud_endpoint="http://localhost:8045",
        local_endpoints={},
        prefer_local=False
    )
    config = cloud_selector.select_model(
        memory_type=MemoryType.TEXT,
        text_length=200,
        quality_requirement=QualityLevel.STANDARD
    )
    print(f"   选择的模型: {config.model_name}")
    print(f"   是否本地: {config.is_local}")
    print()
    
    print("=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

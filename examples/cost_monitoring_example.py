"""
成本监控示例

演示如何使用 CostMonitor 跟踪和优化成本。
Phase 1.1 成本监控功能。
"""

import asyncio
import time
from llm_compression.cost_monitor import (
    CostMonitor,
    ModelType,
    CostEntry
)
from llm_compression.logger import logger


async def example_basic_cost_tracking():
    """示例 1: 基础成本跟踪"""
    print("\n" + "=" * 60)
    print("示例 1: 基础成本跟踪")
    print("=" * 60)
    
    # 创建成本监控器
    monitor = CostMonitor(
        log_file="cost_logs/cost_tracking.jsonl",
        enable_logging=True
    )
    
    # 模拟一些操作
    print("\n模拟压缩操作...")
    
    # 云端 API 操作
    monitor.record_operation(
        model_type=ModelType.CLOUD_API,
        model_name="gpt-4",
        tokens_used=1500,
        operation="compress",
        success=True
    )
    
    # 本地模型操作
    monitor.record_operation(
        model_type=ModelType.LOCAL_MODEL,
        model_name="qwen2.5:7b-instruct",
        tokens_used=1200,
        operation="compress",
        success=True
    )
    
    monitor.record_operation(
        model_type=ModelType.LOCAL_MODEL,
        model_name="qwen2.5:7b-instruct",
        tokens_used=800,
        operation="reconstruct",
        success=True
    )
    
    # 简单压缩（无 LLM）
    monitor.record_operation(
        model_type=ModelType.SIMPLE_COMPRESSION,
        model_name="zstd",
        tokens_used=0,
        operation="compress",
        success=True
    )
    
    # 获取成本汇总
    summary = monitor.get_summary()
    
    print(f"\n成本汇总:")
    print(f"  总成本: ${summary.total_cost:.6f}")
    print(f"  云端 API 成本: ${summary.cloud_cost:.6f}")
    print(f"  本地模型成本: ${summary.local_cost:.6f}")
    print(f"  总 tokens: {summary.total_tokens:,}")
    print(f"  总操作数: {summary.total_operations}")
    print(f"  成本节省: ${summary.savings:.6f} ({summary.savings_percentage:.1f}%)")


async def example_gpu_cost_tracking():
    """示例 2: GPU 成本跟踪"""
    print("\n" + "=" * 60)
    print("示例 2: GPU 成本跟踪")
    print("=" * 60)
    
    monitor = CostMonitor()
    
    # 开始 GPU 跟踪
    print("\n开始 GPU 跟踪...")
    monitor.start_gpu_tracking()
    
    # 模拟 GPU 工作
    print("模拟 GPU 推理...")
    await asyncio.sleep(0.1)  # 模拟 0.1 秒的 GPU 工作
    
    # 记录一些本地模型操作
    for i in range(10):
        monitor.record_operation(
            model_type=ModelType.LOCAL_MODEL,
            model_name="qwen2.5:7b-instruct",
            tokens_used=1000,
            operation="compress",
            success=True
        )
    
    # 停止 GPU 跟踪
    monitor.stop_gpu_tracking()
    print("GPU 跟踪已停止")
    
    # 获取 GPU 成本
    gpu_cost = monitor.get_gpu_cost()
    print(f"\nGPU 成本: ${gpu_cost:.6f}")
    print(f"GPU 使用时间: {monitor.total_gpu_hours * 3600:.2f} 秒")
    
    # 获取总成本
    summary = monitor.get_summary()
    print(f"\n总成本（包含 GPU）: ${summary.total_cost:.6f}")
    print(f"  本地模型 token 成本: ${summary.local_cost - gpu_cost:.6f}")
    print(f"  GPU 电费成本: ${gpu_cost:.6f}")


async def example_cost_reports():
    """示例 3: 成本报告生成"""
    print("\n" + "=" * 60)
    print("示例 3: 成本报告生成")
    print("=" * 60)
    
    monitor = CostMonitor()
    
    # 模拟一周的操作
    print("\n模拟一周的操作...")
    
    # 第 1-3 天：主要使用云端 API
    for day in range(3):
        for _ in range(50):
            monitor.record_operation(
                model_type=ModelType.CLOUD_API,
                model_name="gpt-4",
                tokens_used=1000,
                operation="compress",
                success=True
            )
    
    # 第 4-7 天：切换到本地模型
    for day in range(4):
        for _ in range(100):
            monitor.record_operation(
                model_type=ModelType.LOCAL_MODEL,
                model_name="qwen2.5:7b-instruct",
                tokens_used=1000,
                operation="compress",
                success=True
            )
    
    # 生成每周报告
    print("\n生成每周成本报告...")
    report = monitor.generate_report(period="week")
    print(report)
    
    # 保存报告到文件
    report_file = "cost_reports/weekly_report.txt"
    monitor.generate_report(period="week", output_file=report_file)
    print(f"\n报告已保存到: {report_file}")


async def example_cost_optimization():
    """示例 4: 成本优化建议"""
    print("\n" + "=" * 60)
    print("示例 4: 成本优化建议")
    print("=" * 60)
    
    monitor = CostMonitor()
    
    # 场景 1: 云端 API 使用过多
    print("\n场景 1: 云端 API 使用过多")
    for _ in range(80):
        monitor.record_operation(
            model_type=ModelType.CLOUD_API,
            model_name="gpt-4",
            tokens_used=1000,
            operation="compress",
            success=True
        )
    
    for _ in range(20):
        monitor.record_operation(
            model_type=ModelType.LOCAL_MODEL,
            model_name="qwen2.5:7b-instruct",
            tokens_used=1000,
            operation="compress",
            success=True
        )
    
    # 获取优化建议
    recommendations = monitor.optimize_model_selection()
    
    print(f"\n当前策略: {recommendations['current_strategy']}")
    print(f"潜在节省: ${recommendations['potential_savings']:.4f}")
    print(f"\n优化建议:")
    for i, rec in enumerate(recommendations['recommendations'], 1):
        print(f"\n  {i}. {rec['type']}")
        print(f"     原因: {rec['reason']}")
        print(f"     行动: {rec['action']}")
        print(f"     潜在节省: ${rec['potential_savings']:.4f}")
    
    # 场景 2: 优化后的使用模式
    print("\n\n场景 2: 优化后的使用模式")
    monitor.clear()
    
    # 主要使用本地模型
    for _ in range(90):
        monitor.record_operation(
            model_type=ModelType.LOCAL_MODEL,
            model_name="qwen2.5:7b-instruct",
            tokens_used=1000,
            operation="compress",
            success=True
        )
    
    # 仅在高质量要求时使用云端 API
    for _ in range(10):
        monitor.record_operation(
            model_type=ModelType.CLOUD_API,
            model_name="gpt-4",
            tokens_used=1000,
            operation="compress",
            success=True
        )
    
    recommendations = monitor.optimize_model_selection()
    summary = monitor.get_summary()
    
    print(f"\n成本汇总:")
    print(f"  总成本: ${summary.total_cost:.6f}")
    print(f"  成本节省: ${summary.savings:.6f} ({summary.savings_percentage:.1f}%)")
    print(f"\n优化建议: {len(recommendations['recommendations'])} 条")
    if not recommendations['recommendations']:
        print("  ✓ 当前策略已优化，无需调整")


async def example_daily_cost_tracking():
    """示例 5: 每日成本跟踪"""
    print("\n" + "=" * 60)
    print("示例 5: 每日成本跟踪")
    print("=" * 60)
    
    monitor = CostMonitor()
    
    # 模拟 7 天的操作
    print("\n模拟 7 天的操作...")
    
    for day in range(7):
        # 每天的操作数量递增（模拟业务增长）
        operations_per_day = 50 + day * 10
        
        # 80% 使用本地模型，20% 使用云端 API
        local_ops = int(operations_per_day * 0.8)
        cloud_ops = operations_per_day - local_ops
        
        for _ in range(local_ops):
            monitor.record_operation(
                model_type=ModelType.LOCAL_MODEL,
                model_name="qwen2.5:7b-instruct",
                tokens_used=1000,
                operation="compress",
                success=True
            )
        
        for _ in range(cloud_ops):
            monitor.record_operation(
                model_type=ModelType.CLOUD_API,
                model_name="gpt-4",
                tokens_used=1000,
                operation="compress",
                success=True
            )
    
    # 获取每日汇总
    daily_summaries = monitor.get_daily_summary(days=7)
    
    print("\n每日成本汇总:")
    print(f"{'日期':<12} {'总成本':>10} {'云端':>10} {'本地':>10} {'节省':>10}")
    print("-" * 60)
    
    for date, summary in sorted(daily_summaries.items()):
        print(
            f"{date:<12} "
            f"${summary.total_cost:>9.4f} "
            f"${summary.cloud_cost:>9.4f} "
            f"${summary.local_cost:>9.4f} "
            f"{summary.savings_percentage:>9.1f}%"
        )
    
    # 总体汇总
    total_summary = monitor.get_summary()
    print("-" * 60)
    print(
        f"{'总计':<12} "
        f"${total_summary.total_cost:>9.4f} "
        f"${total_summary.cloud_cost:>9.4f} "
        f"${total_summary.local_cost:>9.4f} "
        f"{total_summary.savings_percentage:>9.1f}%"
    )


async def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("LLM 压缩系统 - 成本监控示例")
    print("Phase 1.1 成本监控功能演示")
    print("=" * 60)
    
    try:
        # 运行所有示例
        await example_basic_cost_tracking()
        await example_gpu_cost_tracking()
        await example_cost_reports()
        await example_cost_optimization()
        await example_daily_cost_tracking()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        
        print("\n关键要点:")
        print("  1. CostMonitor 跟踪所有模型操作的成本")
        print("  2. 支持云端 API、本地模型和 GPU 成本跟踪")
        print("  3. 提供每日/每周/每月成本报告")
        print("  4. 自动计算成本节省（相比全部使用云端 API）")
        print("  5. 提供成本优化建议")
        print("  6. Phase 1.1 目标: 成本节省 > 90%")
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())

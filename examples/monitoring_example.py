"""
Monitoring and Alerting Example

Demonstrates the monitoring system with quality alerting.

Feature: llm-compression-integration
Requirements: 10.1, 10.4, 10.5, 10.6, 10.7
"""

import asyncio
from llm_compression.performance_monitor import PerformanceMonitor
from llm_compression.monitoring import MonitoringSystem, Alert, AlertSeverity


async def main():
    """Demonstrate monitoring and alerting"""
    
    print("=" * 60)
    print("MONITORING AND ALERTING EXAMPLE")
    print("=" * 60)
    print()
    
    # Create monitoring system
    perf_monitor = PerformanceMonitor(
        quality_alert_threshold=0.85,
        quality_drop_threshold=0.10
    )
    monitoring = MonitoringSystem(
        perf_monitor,
        quality_alert_threshold=0.85,
        quality_drop_threshold=0.10
    )
    
    # Register custom alert callback
    def custom_alert_handler(alert: Alert):
        print(f"\n🚨 CUSTOM ALERT HANDLER:")
        print(f"   Severity: {alert.severity.value.upper()}")
        print(f"   Message: {alert.message}")
        print(f"   Metric: {alert.metric_name}")
        print(f"   Current: {alert.current_value:.3f}")
        print(f"   Threshold: {alert.threshold:.3f}")
    
    monitoring.register_alert_callback(custom_alert_handler)
    
    # Scenario 1: Record high-quality compressions
    print("Scenario 1: Recording high-quality compressions...")
    for i in range(10):
        await monitoring.record_compression(
            latency_ms=100.0,
            compression_ratio=15.0,
            quality_score=0.92,
            model_used="gpt-4",
            tokens_used=100
        )
    print(f"✓ Recorded 10 high-quality compressions (quality=0.92)")
    print()
    
    # Scenario 2: Record degraded quality (should trigger alert)
    print("Scenario 2: Recording degraded quality compressions...")
    for i in range(10):
        await monitoring.record_compression(
            latency_ms=100.0,
            compression_ratio=15.0,
            quality_score=0.75,  # 18% drop from 0.92
            model_used="gpt-4",
            tokens_used=100
        )
    print(f"✓ Recorded 10 degraded compressions (quality=0.75)")
    print()
    
    # Scenario 3: Record low quality (should trigger threshold alert)
    print("Scenario 3: Recording low-quality compression...")
    await monitoring.record_compression(
        latency_ms=100.0,
        compression_ratio=15.0,
        quality_score=0.70,  # Below 0.85 threshold
        model_used="gpt-4",
        tokens_used=100
    )
    print(f"✓ Recorded low-quality compression (quality=0.70)")
    print()
    
    # Get statistics
    print("=" * 60)
    print("PERFORMANCE STATISTICS")
    print("=" * 60)
    stats = monitoring.get_statistics()
    print(f"Total Compressions: {stats.total_compressions}")
    print(f"Avg Quality Score: {stats.avg_quality_score:.3f}")
    print(f"Avg Compression Ratio: {stats.avg_compression_ratio:.2f}x")
    print(f"Avg Latency: {stats.avg_compression_latency_ms:.2f}ms")
    print(f"Low Quality Count: {stats.low_quality_count}")
    print()
    
    # Get model comparison
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    comparison = monitoring.get_model_comparison()
    for model, metrics in comparison.items():
        print(f"{model}:")
        print(f"  Usage: {metrics['total_usage']} compressions")
        print(f"  Avg Quality: {metrics['avg_quality']:.3f}")
        print(f"  Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"  Avg Ratio: {metrics['avg_compression_ratio']:.2f}x")
    print()
    
    # Get cost estimation
    print("=" * 60)
    print("COST ESTIMATION")
    print("=" * 60)
    
    # Simulate storage savings
    for _ in range(21):
        await perf_monitor.record_storage_saved(10000)  # 10KB saved per compression
    
    cost_savings = monitoring.estimate_cost_savings()
    print(f"Storage Saved: {cost_savings['storage_saved_gb']:.4f} GB")
    print(f"Storage Savings: ${cost_savings['storage_savings_per_month_usd']:.4f}/month")
    print(f"API Costs: ${cost_savings['api_cost_usd']:.4f}")
    print(f"Net Savings: ${cost_savings['net_savings_per_month_usd']:.4f}/month")
    print()
    
    # Get alert history
    print("=" * 60)
    print("ALERT HISTORY")
    print("=" * 60)
    alerts = monitoring.get_alert_history()
    print(f"Total Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  [{alert.severity.value.upper()}] {alert.metric_name}: {alert.message}")
    print()
    
    # Export Prometheus metrics
    print("=" * 60)
    print("PROMETHEUS METRICS EXPORT")
    print("=" * 60)
    prometheus_metrics = monitoring.export_prometheus_metrics()
    print(prometheus_metrics[:500] + "...")  # Show first 500 chars
    print()
    
    print("=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

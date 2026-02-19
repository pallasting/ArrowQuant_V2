"""
Property-Based Tests for Monitoring and Alerting

Tests monitoring properties:
- Property 25: Quality alert triggering
- Property 26: Model performance comparison
- Property 27: Cost estimation
- Property 38: Prometheus metrics export

Feature: llm-compression-integration
Requirements: 10.4, 10.5, 10.6, 10.7
"""

import pytest
import asyncio
from hypothesis import given, settings, strategies as st
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from llm_compression.performance_monitor import PerformanceMonitor
from llm_compression.monitoring import (
    MonitoringSystem,
    AlertNotifier,
    Alert,
    AlertSeverity
)


class TestQualityAlertProperties:
    """
    Property 25: Quality Alert Triggering
    
    Feature: llm-compression-integration, Property 25: 质量告警触发
    
    Test: *For any* quality degradation event (average quality score drops > 10%),
    the system should send alert notifications.
    
    Validates: Requirements 10.4
    """
    
    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=None)
    @given(
        initial_quality=st.floats(min_value=0.90, max_value=0.95),
        drop_percentage=st.floats(min_value=0.11, max_value=0.30)
    )
    async def test_quality_drop_triggers_alert(
        self,
        initial_quality,
        drop_percentage
    ):
        """
        Test that quality drops > 10% trigger alerts
        
        Feature: llm-compression-integration, Property 25: 质量告警触发
        Validates: Requirements 10.4
        """
        # Create fresh monitoring system for this test
        perf_monitor = PerformanceMonitor(
            quality_alert_threshold=0.85,
            quality_drop_threshold=0.10
        )
        alert_notifier = AlertNotifier()
        monitoring_system = MonitoringSystem(
            perf_monitor,
            alert_notifier,
            quality_alert_threshold=0.85,
            quality_drop_threshold=0.10
        )
        
        # Record initial high-quality compressions
        for _ in range(10):
            await monitoring_system.record_compression(
                latency_ms=100.0,
                compression_ratio=10.0,
                quality_score=initial_quality,
                model_used="test-model"
            )
        
        # Record degraded quality compressions
        degraded_quality = initial_quality * (1 - drop_percentage)
        for _ in range(10):
            await monitoring_system.record_compression(
                latency_ms=100.0,
                compression_ratio=10.0,
                quality_score=degraded_quality,
                model_used="test-model"
            )
        
        # Check that alert was triggered
        alerts = monitoring_system.get_alert_history()
        
        # Should have at least one quality drop alert
        quality_drop_alerts = [
            a for a in alerts
            if a.metric_name == "quality_drop"
        ]
        
        assert len(quality_drop_alerts) > 0, (
            f"Expected quality drop alert for {drop_percentage * 100:.1f}% drop "
            f"({initial_quality:.3f} -> {degraded_quality:.3f})"
        )
        
        # Verify alert details
        alert = quality_drop_alerts[0]
        assert alert.severity in [AlertSeverity.WARNING, AlertSeverity.ERROR]
        assert alert.current_value < alert.threshold
    
    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=None)
    @given(
        quality_score=st.floats(min_value=0.50, max_value=0.84)
    )
    async def test_low_quality_triggers_alert(
        self,
        quality_score
    ):
        """
        Test that quality below threshold triggers alerts
        
        Feature: llm-compression-integration, Property 25: 质量告警触发
        Validates: Requirements 10.4
        """
        # Create fresh monitoring system
        perf_monitor = PerformanceMonitor(
            quality_alert_threshold=0.85,
            quality_drop_threshold=0.10
        )
        alert_notifier = AlertNotifier()
        monitoring_system = MonitoringSystem(
            perf_monitor,
            alert_notifier,
            quality_alert_threshold=0.85,
            quality_drop_threshold=0.10
        )
        
        # Record low-quality compression
        await monitoring_system.record_compression(
            latency_ms=100.0,
            compression_ratio=10.0,
            quality_score=quality_score,
            model_used="test-model"
        )
        
        # Check that alert was triggered
        alerts = monitoring_system.get_alert_history()
        
        # Should have quality threshold alert
        threshold_alerts = [
            a for a in alerts
            if a.metric_name == "quality_score"
        ]
        
        assert len(threshold_alerts) > 0, (
            f"Expected quality threshold alert for score {quality_score:.3f} "
            f"(threshold=0.85)"
        )
        
        # Verify alert details
        alert = threshold_alerts[0]
        assert alert.severity == AlertSeverity.WARNING
        assert alert.current_value == quality_score
        assert alert.threshold == 0.85
    
    @pytest.mark.asyncio
    async def test_alert_callback_invoked(self):
        """
        Test that custom alert callbacks are invoked
        
        Feature: llm-compression-integration, Property 25: 质量告警触发
        Validates: Requirements 10.4
        """
        # Create fresh monitoring system
        perf_monitor = PerformanceMonitor(
            quality_alert_threshold=0.85,
            quality_drop_threshold=0.10
        )
        alert_notifier = AlertNotifier()
        monitoring_system = MonitoringSystem(
            perf_monitor,
            alert_notifier,
            quality_alert_threshold=0.85,
            quality_drop_threshold=0.10
        )
        
        # Register mock callback
        callback_invoked = []
        
        def mock_callback(alert: Alert):
            callback_invoked.append(alert)
        
        monitoring_system.register_alert_callback(mock_callback)
        
        # Trigger alert with low quality
        await monitoring_system.record_compression(
            latency_ms=100.0,
            compression_ratio=10.0,
            quality_score=0.70,
            model_used="test-model"
        )
        
        # Verify callback was invoked
        assert len(callback_invoked) > 0, "Alert callback should be invoked"
        assert callback_invoked[0].metric_name == "quality_score"


class TestModelComparisonProperties:
    """
    Property 26: Model Performance Comparison
    
    Feature: llm-compression-integration, Property 26: 模型性能对比
    
    Test: *For any* model used, the system should record performance metrics
    (latency, quality, cost) and support model comparison.
    
    Validates: Requirements 3.5, 10.5
    """
    
    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=None)
    @given(
        model_count=st.integers(min_value=2, max_value=5),
        compressions_per_model=st.integers(min_value=5, max_value=20)
    )
    async def test_model_comparison_tracks_all_models(
        self,
        model_count,
        compressions_per_model
    ):
        """
        Test that all models are tracked in comparison
        
        Feature: llm-compression-integration, Property 26: 模型性能对比
        Validates: Requirements 3.5, 10.5
        """
        # Create fresh performance monitor
        performance_monitor = PerformanceMonitor()
        
        models = [f"model-{i}" for i in range(model_count)]
        
        # Record compressions for each model
        for model in models:
            for _ in range(compressions_per_model):
                await performance_monitor.record_compression(
                    latency_ms=100.0 + hash(model) % 50,
                    compression_ratio=10.0 + hash(model) % 5,
                    quality_score=0.90 + (hash(model) % 10) / 100,
                    model_used=model,
                    tokens_used=100
                )
        
        # Get model comparison
        comparison = performance_monitor.get_model_comparison()
        
        # Verify all models are tracked
        assert len(comparison) == model_count, (
            f"Expected {model_count} models in comparison, "
            f"got {len(comparison)}"
        )
        
        for model in models:
            assert model in comparison, f"Model {model} not in comparison"
            
            # Verify metrics are recorded
            stats = comparison[model]
            assert stats['total_usage'] == compressions_per_model
            assert stats['avg_latency_ms'] > 0
            assert stats['avg_quality'] > 0
            assert stats['avg_compression_ratio'] > 0
            assert stats['total_tokens'] > 0
    
    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=None)
    @given(
        latency_values=st.lists(
            st.floats(min_value=50.0, max_value=500.0),
            min_size=10,
            max_size=50
        )
    )
    async def test_model_comparison_calculates_averages(
        self,
        latency_values
    ):
        """
        Test that model comparison calculates correct averages
        
        Feature: llm-compression-integration, Property 26: 模型性能对比
        Validates: Requirements 3.5, 10.5
        """
        # Create fresh performance monitor
        performance_monitor = PerformanceMonitor()
        
        model_name = "test-model"
        
        # Record compressions with varying latencies
        for latency in latency_values:
            await performance_monitor.record_compression(
                latency_ms=latency,
                compression_ratio=10.0,
                quality_score=0.90,
                model_used=model_name,
                tokens_used=100
            )
        
        # Get model comparison
        comparison = performance_monitor.get_model_comparison()
        
        # Verify average calculation
        expected_avg = sum(latency_values) / len(latency_values)
        actual_avg = comparison[model_name]['avg_latency_ms']
        
        # Allow small floating point error
        assert abs(actual_avg - expected_avg) < 0.01, (
            f"Expected avg latency {expected_avg:.2f}, "
            f"got {actual_avg:.2f}"
        )


class TestCostEstimationProperties:
    """
    Property 27: Cost Estimation
    
    Feature: llm-compression-integration, Property 27: 成本估算
    
    Test: *For any* time period, the system should estimate cost savings
    (storage savings - API costs).
    
    Validates: Requirements 10.6
    """
    
    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=None)
    @given(
        original_size=st.integers(min_value=1000, max_value=100000),
        compression_ratio=st.floats(min_value=5.0, max_value=50.0),
        num_compressions=st.integers(min_value=10, max_value=100)
    )
    async def test_cost_estimation_calculates_savings(
        self,
        original_size,
        compression_ratio,
        num_compressions
    ):
        """
        Test that cost estimation calculates storage savings
        
        Feature: llm-compression-integration, Property 27: 成本估算
        Validates: Requirements 10.6
        """
        # Create fresh performance monitor
        performance_monitor = PerformanceMonitor()
        
        # Record compressions with storage savings
        for _ in range(num_compressions):
            compressed_size = original_size / compression_ratio
            bytes_saved = original_size - compressed_size
            
            await performance_monitor.record_compression(
                latency_ms=100.0,
                compression_ratio=compression_ratio,
                quality_score=0.90,
                model_used="test-model",
                tokens_used=100
            )
            
            await performance_monitor.record_storage_saved(int(bytes_saved))
        
        # Get cost estimation
        cost_savings = performance_monitor.estimate_cost_savings()
        
        # Verify storage savings are calculated
        assert cost_savings['storage_saved_gb'] > 0, (
            "Storage savings should be positive"
        )
        
        # Verify cost breakdown exists
        assert 'storage_savings_per_month_usd' in cost_savings
        assert 'api_cost_usd' in cost_savings
        assert 'net_savings_per_month_usd' in cost_savings
        assert 'savings_percentage' in cost_savings
        
        # Verify net savings calculation
        expected_net = (
            cost_savings['storage_savings_per_month_usd'] -
            cost_savings['api_cost_usd']
        )
        assert abs(
            cost_savings['net_savings_per_month_usd'] - expected_net
        ) < 0.01
    
    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=None)
    @given(
        tokens_used=st.integers(min_value=1000, max_value=100000)
    )
    async def test_cost_estimation_tracks_api_costs(
        self,
        tokens_used
    ):
        """
        Test that cost estimation tracks API costs
        
        Feature: llm-compression-integration, Property 27: 成本估算
        Validates: Requirements 10.6
        """
        # Create fresh performance monitor
        performance_monitor = PerformanceMonitor()
        
        # Record API call with tokens
        await performance_monitor.record_api_call(
            latency_ms=100.0,
            success=True,
            tokens_used=tokens_used,
            cost=0.0  # Will be calculated from tokens
        )
        
        # Get cost estimation
        cost_savings = performance_monitor.estimate_cost_savings(
            api_cost_per_1k_tokens=0.002
        )
        
        # Verify API cost is calculated
        expected_cost = (tokens_used / 1000) * 0.002
        assert abs(cost_savings['api_cost_usd'] - expected_cost) < 0.001


class TestPrometheusExportProperties:
    """
    Property 38: Prometheus Metrics Export
    
    Feature: llm-compression-integration, Property 38: Prometheus 指标导出
    
    Test: *For any* Prometheus scrape request, the system should export
    all monitoring metrics in Prometheus format.
    
    Validates: Requirements 10.7
    """
    
    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=None)
    @given(
        num_compressions=st.integers(min_value=1, max_value=50),
        num_reconstructions=st.integers(min_value=1, max_value=50)
    )
    async def test_prometheus_export_includes_all_metrics(
        self,
        num_compressions,
        num_reconstructions
    ):
        """
        Test that Prometheus export includes all metrics
        
        Feature: llm-compression-integration, Property 38: Prometheus 指标导出
        Validates: Requirements 10.7
        """
        # Create fresh performance monitor
        performance_monitor = PerformanceMonitor()
        
        # Record some compressions
        for _ in range(num_compressions):
            await performance_monitor.record_compression(
                latency_ms=100.0,
                compression_ratio=10.0,
                quality_score=0.90,
                model_used="test-model",
                tokens_used=100
            )
        
        # Record some reconstructions
        for _ in range(num_reconstructions):
            await performance_monitor.record_reconstruction(
                latency_ms=50.0,
                quality_score=0.90
            )
        
        # Export Prometheus metrics
        metrics_text = performance_monitor.export_prometheus_metrics()
        
        # Verify required metrics are present
        required_metrics = [
            'llm_compression_total',
            'llm_compression_latency_ms',
            'llm_compression_ratio',
            'llm_reconstruction_total',
            'llm_reconstruction_latency_ms',
            'llm_quality_score',
            'llm_low_quality_total',
            'llm_api_calls_total',
            'llm_api_errors_total',
            'llm_throughput_per_minute'
        ]
        
        for metric in required_metrics:
            assert metric in metrics_text, (
                f"Prometheus export missing metric: {metric}"
            )
        
        # Verify metric values match statistics
        stats = performance_monitor.get_statistics()
        
        assert f"llm_compression_total {stats.total_compressions}" in metrics_text
        assert f"llm_reconstruction_total {stats.total_reconstructions}" in metrics_text
    
    @pytest.mark.asyncio
    async def test_prometheus_export_format_valid(self):
        """
        Test that Prometheus export format is valid
        
        Feature: llm-compression-integration, Property 38: Prometheus 指标导出
        Validates: Requirements 10.7
        """
        # Create fresh performance monitor
        performance_monitor = PerformanceMonitor()
        
        # Record a compression
        await performance_monitor.record_compression(
            latency_ms=100.0,
            compression_ratio=10.0,
            quality_score=0.90,
            model_used="test-model",
            tokens_used=100
        )
        
        # Export metrics
        metrics_text = performance_monitor.export_prometheus_metrics()
        
        # Verify format
        lines = metrics_text.split('\n')
        
        # Should have HELP and TYPE comments
        help_lines = [l for l in lines if l.startswith('# HELP')]
        type_lines = [l for l in lines if l.startswith('# TYPE')]
        
        assert len(help_lines) > 0, "Should have HELP comments"
        assert len(type_lines) > 0, "Should have TYPE comments"
        
        # Verify metric lines format (metric_name value)
        metric_lines = [
            l for l in lines
            if l and not l.startswith('#')
        ]
        
        for line in metric_lines:
            parts = line.split()
            assert len(parts) == 2, f"Invalid metric line format: {line}"
            
            metric_name, value = parts
            assert metric_name.startswith('llm_'), (
                f"Metric should start with 'llm_': {metric_name}"
            )
            
            # Value should be a number
            try:
                float(value)
            except ValueError:
                pytest.fail(f"Metric value should be numeric: {value}")

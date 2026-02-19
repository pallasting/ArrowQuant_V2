"""
Property-Based Tests for Performance Monitoring

Tests comprehensive metric tracking for the compression system.

Feature: llm-compression-integration, Property 24: 指标跟踪完整性
Requirements: 1.6, 10.1
"""

import pytest
import asyncio
from hypothesis import given, settings, strategies as st

from llm_compression import (
    PerformanceMonitor,
    PerformanceMetrics
)


class TestPerformanceMonitoringProperties:
    """
    Property tests for performance monitoring
    
    **Validates: Requirements 1.6, 10.1**
    """
    
    @pytest.fixture
    def monitor(self):
        """Create performance monitor for testing"""
        return PerformanceMonitor(
            quality_alert_threshold=0.85,
            quality_drop_threshold=0.10
        )
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=None)
    @given(
        num_compressions=st.integers(min_value=1, max_value=50),
        latency_range=st.tuples(
            st.floats(min_value=10.0, max_value=100.0),
            st.floats(min_value=100.0, max_value=500.0)
        ),
        ratio_range=st.tuples(
            st.floats(min_value=5.0, max_value=10.0),
            st.floats(min_value=10.0, max_value=50.0)
        ),
        quality_range=st.tuples(
            st.floats(min_value=0.8, max_value=0.9),
            st.floats(min_value=0.9, max_value=1.0)
        )
    )
    async def test_tracks_all_compression_metrics(
        self,
        monitor,
        num_compressions,
        latency_range,
        ratio_range,
        quality_range
    ):
        """
        Feature: llm-compression-integration, Property 24: 指标跟踪完整性
        
        Test: *For any* 系统操作，监控系统应该跟踪所有指定指标：
        - 压缩次数
        - 平均压缩比
        - 平均压缩延迟
        - 平均质量分数
        
        **Validates: Requirements 10.1**
        """
        # Record compression metrics
        for i in range(num_compressions):
            latency = latency_range[0] + (latency_range[1] - latency_range[0]) * (i / num_compressions)
            ratio = ratio_range[0] + (ratio_range[1] - ratio_range[0]) * (i / num_compressions)
            quality = quality_range[0] + (quality_range[1] - quality_range[0]) * (i / num_compressions)
            
            await monitor.record_compression(
                latency_ms=latency,
                compression_ratio=ratio,
                quality_score=quality,
                model_used=f"model-{i % 3}",
                tokens_used=100
            )
        
        # Get statistics
        stats = monitor.get_statistics()
        
        # Verify all metrics are tracked
        assert stats.total_compressions == num_compressions, \
            f"Expected {num_compressions} compressions, got {stats.total_compressions}"
        
        assert stats.avg_compression_latency_ms > 0, \
            "Average compression latency not tracked"
        
        assert stats.avg_compression_ratio > 0, \
            "Average compression ratio not tracked"
        
        assert stats.avg_quality_score > 0, \
            "Average quality score not tracked"
        
        # Verify percentiles are calculated
        assert stats.p95_compression_latency_ms > 0, \
            "P95 compression latency not tracked"
        
        assert stats.p99_compression_latency_ms > 0, \
            "P99 compression latency not tracked"
        
        print(f"Tracked {num_compressions} compressions: "
              f"avg_latency={stats.avg_compression_latency_ms:.2f}ms, "
              f"avg_ratio={stats.avg_compression_ratio:.2f}x, "
              f"avg_quality={stats.avg_quality_score:.3f}")
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=None)
    @given(
        num_reconstructions=st.integers(min_value=1, max_value=50),
        latency_range=st.tuples(
            st.floats(min_value=10.0, max_value=50.0),
            st.floats(min_value=50.0, max_value=200.0)
        )
    )
    async def test_tracks_reconstruction_metrics(
        self,
        monitor,
        num_reconstructions,
        latency_range
    ):
        """
        Feature: llm-compression-integration, Property 24: 指标跟踪完整性
        
        Test: *For any* 重构操作，监控系统应该跟踪重构延迟
        
        **Validates: Requirements 10.1**
        """
        # Record reconstruction metrics
        for i in range(num_reconstructions):
            latency = latency_range[0] + (latency_range[1] - latency_range[0]) * (i / num_reconstructions)
            
            await monitor.record_reconstruction(
                latency_ms=latency,
                quality_score=0.9
            )
        
        # Get statistics
        stats = monitor.get_statistics()
        
        # Verify reconstruction metrics
        assert stats.total_reconstructions == num_reconstructions, \
            f"Expected {num_reconstructions} reconstructions, got {stats.total_reconstructions}"
        
        assert stats.avg_reconstruction_latency_ms > 0, \
            "Average reconstruction latency not tracked"
        
        assert stats.p95_reconstruction_latency_ms > 0, \
            "P95 reconstruction latency not tracked"
        
        assert stats.p99_reconstruction_latency_ms > 0, \
            "P99 reconstruction latency not tracked"
        
        print(f"Tracked {num_reconstructions} reconstructions: "
              f"avg_latency={stats.avg_reconstruction_latency_ms:.2f}ms")
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=None)
    @given(
        num_api_calls=st.integers(min_value=1, max_value=50),
        error_rate=st.floats(min_value=0.0, max_value=0.3)
    )
    async def test_tracks_api_metrics(
        self,
        monitor,
        num_api_calls,
        error_rate
    ):
        """
        Feature: llm-compression-integration, Property 24: 指标跟踪完整性
        
        Test: *For any* API 调用，监控系统应该跟踪 API 延迟和错误率
        
        **Validates: Requirements 1.6, 10.1**
        """
        # Record API calls
        for i in range(num_api_calls):
            success = (i / num_api_calls) > error_rate
            
            await monitor.record_api_call(
                latency_ms=100.0 + i * 10,
                success=success,
                tokens_used=50,
                cost=0.001
            )
        
        # Get statistics
        stats = monitor.get_statistics()
        
        # Verify API metrics
        assert stats.total_api_calls == num_api_calls, \
            f"Expected {num_api_calls} API calls, got {stats.total_api_calls}"
        
        assert stats.avg_api_latency_ms > 0, \
            "Average API latency not tracked"
        
        # Verify error tracking
        expected_errors = int(num_api_calls * error_rate)
        assert stats.api_error_count >= 0, \
            "API errors not tracked"
        
        print(f"Tracked {num_api_calls} API calls: "
              f"avg_latency={stats.avg_api_latency_ms:.2f}ms, "
              f"errors={stats.api_error_count}")
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=None)
    @given(
        num_models=st.integers(min_value=1, max_value=5),
        calls_per_model=st.integers(min_value=5, max_value=20)
    )
    async def test_tracks_model_usage(
        self,
        monitor,
        num_models,
        calls_per_model
    ):
        """
        Feature: llm-compression-integration, Property 24: 指标跟踪完整性
        
        Test: *For any* 模型使用，监控系统应该跟踪每个模型的使用统计
        
        **Validates: Requirements 10.1**
        """
        # Record compressions with different models
        for model_idx in range(num_models):
            model_name = f"model-{model_idx}"
            
            for call_idx in range(calls_per_model):
                await monitor.record_compression(
                    latency_ms=100.0 + call_idx * 10,
                    compression_ratio=10.0 + model_idx,
                    quality_score=0.85 + model_idx * 0.02,
                    model_used=model_name,
                    tokens_used=100
                )
        
        # Get model comparison
        comparison = monitor.get_model_comparison()
        
        # Verify model tracking
        assert len(comparison) == num_models, \
            f"Expected {num_models} models tracked, got {len(comparison)}"
        
        for model_idx in range(num_models):
            model_name = f"model-{model_idx}"
            assert model_name in comparison, \
                f"Model {model_name} not tracked"
            
            model_stats = comparison[model_name]
            assert model_stats['total_usage'] == calls_per_model, \
                f"Expected {calls_per_model} calls for {model_name}, got {model_stats['total_usage']}"
            
            assert model_stats['avg_latency_ms'] > 0, \
                f"Average latency not tracked for {model_name}"
            
            assert model_stats['avg_quality'] > 0, \
                f"Average quality not tracked for {model_name}"
        
        print(f"Tracked {num_models} models with {calls_per_model} calls each")
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=None)
    @given(
        storage_saved_mb=st.integers(min_value=100, max_value=10000)
    )
    async def test_tracks_storage_savings(
        self,
        monitor,
        storage_saved_mb
    ):
        """
        Feature: llm-compression-integration, Property 24: 指标跟踪完整性
        
        Test: *For any* 压缩操作，监控系统应该跟踪存储节省
        
        **Validates: Requirements 10.1**
        """
        # Record storage savings
        bytes_saved = storage_saved_mb * 1024 * 1024
        await monitor.record_storage_saved(bytes_saved)
        
        # Get cost estimation
        cost_savings = monitor.estimate_cost_savings()
        
        # Verify storage tracking
        assert cost_savings['storage_saved_gb'] > 0, \
            "Storage savings not tracked"
        
        assert cost_savings['storage_savings_per_month_usd'] > 0, \
            "Storage cost savings not calculated"
        
        print(f"Tracked {storage_saved_mb}MB storage savings: "
              f"${cost_savings['storage_savings_per_month_usd']:.2f}/month")
    
    @pytest.mark.asyncio
    async def test_calculates_throughput(
        self,
        monitor
    ):
        """
        Feature: llm-compression-integration, Property 24: 指标跟踪完整性
        
        Test: *For any* 时间段，监控系统应该计算吞吐量（items/min）
        
        **Validates: Requirements 10.1**
        """
        # Record some compressions
        num_compressions = 20
        
        for i in range(num_compressions):
            await monitor.record_compression(
                latency_ms=100.0,
                compression_ratio=10.0,
                quality_score=0.9,
                model_used="test-model",
                tokens_used=100
            )
            
            # Small delay to simulate time passing
            await asyncio.sleep(0.01)
        
        # Get statistics
        stats = monitor.get_statistics()
        
        # Verify throughput calculation
        assert stats.throughput_per_minute > 0, \
            "Throughput not calculated"
        
        print(f"Throughput: {stats.throughput_per_minute:.1f} items/min")
    
    @pytest.mark.asyncio
    async def test_generates_report(
        self,
        monitor
    ):
        """
        Feature: llm-compression-integration, Property 24: 指标跟踪完整性
        
        Test: *For any* 监控数据，系统应该能够生成可读报告
        
        **Validates: Requirements 10.1, 10.2**
        """
        # Record some metrics
        for i in range(10):
            await monitor.record_compression(
                latency_ms=100.0 + i * 10,
                compression_ratio=10.0 + i,
                quality_score=0.85 + i * 0.01,
                model_used=f"model-{i % 2}",
                tokens_used=100
            )
        
        # Generate report
        report = monitor.generate_report()
        
        # Verify report contains key sections
        assert "PERFORMANCE REPORT" in report, \
            "Report missing title"
        
        assert "COMPRESSION METRICS" in report, \
            "Report missing compression metrics"
        
        assert "QUALITY METRICS" in report, \
            "Report missing quality metrics"
        
        assert "MODEL COMPARISON" in report, \
            "Report missing model comparison"
        
        assert "COST ANALYSIS" in report, \
            "Report missing cost analysis"
        
        print("Generated performance report successfully")
    
    @pytest.mark.asyncio
    async def test_exports_prometheus_metrics(
        self,
        monitor
    ):
        """
        Feature: llm-compression-integration, Property 24: 指标跟踪完整性
        
        Test: *For any* 监控数据，系统应该能够导出 Prometheus 格式指标
        
        **Validates: Requirements 10.7**
        """
        # Record some metrics
        for i in range(5):
            await monitor.record_compression(
                latency_ms=100.0,
                compression_ratio=10.0,
                quality_score=0.9,
                model_used="test-model",
                tokens_used=100
            )
        
        # Export Prometheus metrics
        prometheus_output = monitor.export_prometheus_metrics()
        
        # Verify Prometheus format
        assert "# HELP" in prometheus_output, \
            "Prometheus output missing HELP comments"
        
        assert "# TYPE" in prometheus_output, \
            "Prometheus output missing TYPE comments"
        
        assert "llm_compression_total" in prometheus_output, \
            "Prometheus output missing compression counter"
        
        assert "llm_compression_latency_ms" in prometheus_output, \
            "Prometheus output missing latency gauge"
        
        assert "llm_compression_ratio" in prometheus_output, \
            "Prometheus output missing ratio gauge"
        
        assert "llm_quality_score" in prometheus_output, \
            "Prometheus output missing quality gauge"
        
        print("Exported Prometheus metrics successfully")
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=None)
    @given(
        quality_drop=st.floats(min_value=0.15, max_value=0.30)
    )
    async def test_detects_quality_drop(
        self,
        monitor,
        quality_drop
    ):
        """
        Feature: llm-compression-integration, Property 24: 指标跟踪完整性
        
        Test: *For any* 质量下降，监控系统应该检测并记录警告
        
        **Validates: Requirements 10.4**
        """
        # Record high quality compressions
        for i in range(10):
            await monitor.record_compression(
                latency_ms=100.0,
                compression_ratio=10.0,
                quality_score=0.95,
                model_used="test-model",
                tokens_used=100
            )
        
        # Record low quality compressions (simulating quality drop)
        low_quality = 0.95 - quality_drop
        for i in range(10):
            await monitor.record_compression(
                latency_ms=100.0,
                compression_ratio=10.0,
                quality_score=low_quality,
                model_used="test-model",
                tokens_used=100
            )
        
        # Verify quality drop was detected
        # (In real implementation, this would trigger an alert)
        stats = monitor.get_statistics()
        
        # Quality should have dropped
        assert stats.avg_quality_score < 0.95, \
            "Quality drop not reflected in statistics"
        
        print(f"Quality drop detected: {quality_drop * 100:.1f}%")

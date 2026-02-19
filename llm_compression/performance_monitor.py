"""
Performance Monitor Module

Tracks and reports performance metrics for the compression system:
- Compression/reconstruction latency
- Compression ratio
- Quality scores
- API costs
- GPU usage
- Throughput

Feature: llm-compression-integration
Requirements: 1.6, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
"""

import asyncio
import time
import statistics
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict

from llm_compression.logger import logger


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    # Compression metrics
    total_compressions: int = 0
    avg_compression_latency_ms: float = 0.0
    p95_compression_latency_ms: float = 0.0
    p99_compression_latency_ms: float = 0.0
    avg_compression_ratio: float = 0.0
    
    # Reconstruction metrics
    total_reconstructions: int = 0
    avg_reconstruction_latency_ms: float = 0.0
    p95_reconstruction_latency_ms: float = 0.0
    p99_reconstruction_latency_ms: float = 0.0
    
    # Quality metrics
    avg_quality_score: float = 0.0
    low_quality_count: int = 0
    
    # API metrics
    total_api_calls: int = 0
    avg_api_latency_ms: float = 0.0
    api_error_count: int = 0
    
    # Model usage
    model_usage: Dict[str, int] = field(default_factory=dict)
    
    # Throughput
    throughput_per_minute: float = 0.0
    
    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class PerformanceMonitor:
    """
    Performance monitoring system
    
    Tracks all system metrics and provides:
    - Real-time metric collection
    - Statistical analysis (mean, median, percentiles)
    - Model performance comparison
    - Cost estimation
    - Quality alerts
    - Prometheus metrics export
    
    Requirements: 1.6, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
    """
    
    def __init__(
        self,
        quality_alert_threshold: float = 0.85,
        quality_drop_threshold: float = 0.10
    ):
        """
        Initialize performance monitor
        
        Args:
            quality_alert_threshold: Quality threshold for alerts
            quality_drop_threshold: Quality drop threshold for alerts (10%)
        
        Requirements: 10.1
        """
        self.quality_alert_threshold = quality_alert_threshold
        self.quality_drop_threshold = quality_drop_threshold
        
        # Metric storage
        self.metrics: Dict[str, List[float]] = {
            'compression_latency': [],
            'reconstruction_latency': [],
            'compression_ratio': [],
            'quality_score': [],
            'api_latency': [],
        }
        
        # Model-specific metrics
        self.model_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {
                'latency': [],
                'quality': [],
                'compression_ratio': [],
                'token_usage': []
            }
        )
        
        # Counters
        self.counters = {
            'total_compressions': 0,
            'total_reconstructions': 0,
            'total_api_calls': 0,
            'api_errors': 0,
            'low_quality_compressions': 0
        }
        
        # Cost tracking
        self.cost_data = {
            'total_tokens': 0,
            'total_api_cost': 0.0,
            'storage_saved_bytes': 0
        }
        
        # Start time
        self.start_time = datetime.now()
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        # Previous quality average for drop detection
        self.previous_quality_avg: Optional[float] = None
        
        logger.info(
            f"PerformanceMonitor initialized: "
            f"quality_threshold={quality_alert_threshold}, "
            f"drop_threshold={quality_drop_threshold}"
        )
    
    async def record_compression(
        self,
        latency_ms: float,
        compression_ratio: float,
        quality_score: float,
        model_used: str,
        tokens_used: int = 0
    ):
        """
        Record compression metrics
        
        Args:
            latency_ms: Compression latency in milliseconds
            compression_ratio: Compression ratio
            quality_score: Quality score (0-1)
            model_used: Model name
            tokens_used: Number of tokens used
        
        Requirements: 10.1
        """
        async with self.lock:
            # Record general metrics
            self.metrics['compression_latency'].append(latency_ms)
            self.metrics['compression_ratio'].append(compression_ratio)
            self.metrics['quality_score'].append(quality_score)
            
            # Record model-specific metrics
            self.model_metrics[model_used]['latency'].append(latency_ms)
            self.model_metrics[model_used]['quality'].append(quality_score)
            self.model_metrics[model_used]['compression_ratio'].append(compression_ratio)
            self.model_metrics[model_used]['token_usage'].append(tokens_used)
            
            # Update counters
            self.counters['total_compressions'] += 1
            
            if quality_score < self.quality_alert_threshold:
                self.counters['low_quality_compressions'] += 1
            
            # Update cost data
            self.cost_data['total_tokens'] += tokens_used
            
            # Check for quality drop
            await self._check_quality_drop()
    
    async def record_reconstruction(
        self,
        latency_ms: float,
        quality_score: Optional[float] = None
    ):
        """
        Record reconstruction metrics
        
        Args:
            latency_ms: Reconstruction latency in milliseconds
            quality_score: Quality score (optional)
        
        Requirements: 10.1
        """
        async with self.lock:
            self.metrics['reconstruction_latency'].append(latency_ms)
            
            if quality_score is not None:
                self.metrics['quality_score'].append(quality_score)
            
            self.counters['total_reconstructions'] += 1
    
    async def record_api_call(
        self,
        latency_ms: float,
        success: bool = True,
        tokens_used: int = 0,
        cost: float = 0.0
    ):
        """
        Record API call metrics
        
        Args:
            latency_ms: API latency in milliseconds
            success: Whether the call succeeded
            tokens_used: Number of tokens used
            cost: API call cost in USD
        
        Requirements: 10.1
        """
        async with self.lock:
            self.metrics['api_latency'].append(latency_ms)
            self.counters['total_api_calls'] += 1
            
            if not success:
                self.counters['api_errors'] += 1
            
            self.cost_data['total_tokens'] += tokens_used
            self.cost_data['total_api_cost'] += cost
    
    async def record_storage_saved(self, bytes_saved: int):
        """
        Record storage savings
        
        Args:
            bytes_saved: Bytes saved by compression
        
        Requirements: 10.6
        """
        async with self.lock:
            self.cost_data['storage_saved_bytes'] += bytes_saved
    
    def get_statistics(self) -> PerformanceMetrics:
        """
        Get performance statistics
        
        Returns:
            PerformanceMetrics: Current performance metrics
        
        Requirements: 10.1, 10.2
        """
        metrics = PerformanceMetrics()
        
        # Compression metrics
        metrics.total_compressions = self.counters['total_compressions']
        if self.metrics['compression_latency']:
            metrics.avg_compression_latency_ms = statistics.mean(
                self.metrics['compression_latency']
            )
            metrics.p95_compression_latency_ms = self._percentile(
                self.metrics['compression_latency'], 0.95
            )
            metrics.p99_compression_latency_ms = self._percentile(
                self.metrics['compression_latency'], 0.99
            )
        
        if self.metrics['compression_ratio']:
            metrics.avg_compression_ratio = statistics.mean(
                self.metrics['compression_ratio']
            )
        
        # Reconstruction metrics
        metrics.total_reconstructions = self.counters['total_reconstructions']
        if self.metrics['reconstruction_latency']:
            metrics.avg_reconstruction_latency_ms = statistics.mean(
                self.metrics['reconstruction_latency']
            )
            metrics.p95_reconstruction_latency_ms = self._percentile(
                self.metrics['reconstruction_latency'], 0.95
            )
            metrics.p99_reconstruction_latency_ms = self._percentile(
                self.metrics['reconstruction_latency'], 0.99
            )
        
        # Quality metrics
        if self.metrics['quality_score']:
            metrics.avg_quality_score = statistics.mean(
                self.metrics['quality_score']
            )
        metrics.low_quality_count = self.counters['low_quality_compressions']
        
        # API metrics
        metrics.total_api_calls = self.counters['total_api_calls']
        if self.metrics['api_latency']:
            metrics.avg_api_latency_ms = statistics.mean(
                self.metrics['api_latency']
            )
        metrics.api_error_count = self.counters['api_errors']
        
        # Model usage
        metrics.model_usage = {
            model: len(data['latency'])
            for model, data in self.model_metrics.items()
        }
        
        # Throughput
        elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        if elapsed_minutes > 0:
            metrics.throughput_per_minute = (
                metrics.total_compressions / elapsed_minutes
            )
        
        # Time range
        metrics.start_time = self.start_time
        metrics.end_time = datetime.now()
        
        return metrics
    
    def get_model_comparison(self) -> Dict[str, Dict[str, float]]:
        """
        Get model performance comparison
        
        Returns:
            Dict mapping model names to performance metrics
        
        Requirements: 10.5
        """
        comparison = {}
        
        for model, data in self.model_metrics.items():
            if not data['latency']:
                continue
            
            comparison[model] = {
                'avg_latency_ms': statistics.mean(data['latency']),
                'avg_quality': statistics.mean(data['quality']) if data['quality'] else 0.0,
                'avg_compression_ratio': statistics.mean(data['compression_ratio']) if data['compression_ratio'] else 0.0,
                'total_usage': len(data['latency']),
                'total_tokens': sum(data['token_usage']),
                'success_rate': 1.0  # TODO: track failures per model
            }
        
        return comparison
    
    def estimate_cost_savings(
        self,
        storage_cost_per_gb_month: float = 0.023,  # AWS S3 standard
        api_cost_per_1k_tokens: float = 0.002  # Approximate
    ) -> Dict[str, float]:
        """
        Estimate cost savings from compression
        
        Args:
            storage_cost_per_gb_month: Storage cost per GB per month
            api_cost_per_1k_tokens: API cost per 1K tokens
        
        Returns:
            Dict with cost breakdown
        
        Requirements: 10.6
        """
        # Storage savings
        storage_saved_gb = self.cost_data['storage_saved_bytes'] / (1024 ** 3)
        storage_savings_per_month = storage_saved_gb * storage_cost_per_gb_month
        
        # API costs
        api_cost = (self.cost_data['total_tokens'] / 1000) * api_cost_per_1k_tokens
        
        # Net savings (storage savings - API costs)
        net_savings = storage_savings_per_month - api_cost
        
        return {
            'storage_saved_gb': storage_saved_gb,
            'storage_savings_per_month_usd': storage_savings_per_month,
            'api_cost_usd': api_cost,
            'net_savings_per_month_usd': net_savings,
            'savings_percentage': (
                (net_savings / storage_savings_per_month * 100)
                if storage_savings_per_month > 0 else 0.0
            )
        }
    
    def generate_report(self) -> str:
        """
        Generate human-readable performance report
        
        Returns:
            str: Formatted report
        
        Requirements: 10.2
        """
        metrics = self.get_statistics()
        model_comparison = self.get_model_comparison()
        cost_savings = self.estimate_cost_savings()
        
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Time Range: {metrics.start_time} to {metrics.end_time}")
        report.append(f"Duration: {(metrics.end_time - metrics.start_time).total_seconds() / 3600:.2f} hours")
        report.append("")
        
        # Compression metrics
        report.append("COMPRESSION METRICS")
        report.append("-" * 60)
        report.append(f"Total Compressions: {metrics.total_compressions}")
        report.append(f"Avg Latency: {metrics.avg_compression_latency_ms:.2f}ms")
        report.append(f"P95 Latency: {metrics.p95_compression_latency_ms:.2f}ms")
        report.append(f"P99 Latency: {metrics.p99_compression_latency_ms:.2f}ms")
        report.append(f"Avg Compression Ratio: {metrics.avg_compression_ratio:.2f}x")
        report.append(f"Throughput: {metrics.throughput_per_minute:.1f} items/min")
        report.append("")
        
        # Reconstruction metrics
        report.append("RECONSTRUCTION METRICS")
        report.append("-" * 60)
        report.append(f"Total Reconstructions: {metrics.total_reconstructions}")
        report.append(f"Avg Latency: {metrics.avg_reconstruction_latency_ms:.2f}ms")
        report.append(f"P95 Latency: {metrics.p95_reconstruction_latency_ms:.2f}ms")
        report.append(f"P99 Latency: {metrics.p99_reconstruction_latency_ms:.2f}ms")
        report.append("")
        
        # Quality metrics
        report.append("QUALITY METRICS")
        report.append("-" * 60)
        report.append(f"Avg Quality Score: {metrics.avg_quality_score:.3f}")
        report.append(f"Low Quality Count: {metrics.low_quality_count}")
        report.append(f"Low Quality Rate: {(metrics.low_quality_count / max(metrics.total_compressions, 1) * 100):.1f}%")
        report.append("")
        
        # API metrics
        report.append("API METRICS")
        report.append("-" * 60)
        report.append(f"Total API Calls: {metrics.total_api_calls}")
        report.append(f"Avg API Latency: {metrics.avg_api_latency_ms:.2f}ms")
        report.append(f"API Errors: {metrics.api_error_count}")
        report.append(f"Error Rate: {(metrics.api_error_count / max(metrics.total_api_calls, 1) * 100):.1f}%")
        report.append("")
        
        # Model comparison
        report.append("MODEL COMPARISON")
        report.append("-" * 60)
        for model, stats in model_comparison.items():
            report.append(f"{model}:")
            report.append(f"  Usage: {stats['total_usage']} compressions")
            report.append(f"  Avg Latency: {stats['avg_latency_ms']:.2f}ms")
            report.append(f"  Avg Quality: {stats['avg_quality']:.3f}")
            report.append(f"  Avg Ratio: {stats['avg_compression_ratio']:.2f}x")
            report.append(f"  Total Tokens: {stats['total_tokens']}")
        report.append("")
        
        # Cost savings
        report.append("COST ANALYSIS")
        report.append("-" * 60)
        report.append(f"Storage Saved: {cost_savings['storage_saved_gb']:.2f} GB")
        report.append(f"Storage Savings: ${cost_savings['storage_savings_per_month_usd']:.2f}/month")
        report.append(f"API Costs: ${cost_savings['api_cost_usd']:.2f}")
        report.append(f"Net Savings: ${cost_savings['net_savings_per_month_usd']:.2f}/month")
        report.append(f"Savings Rate: {cost_savings['savings_percentage']:.1f}%")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format
        
        Returns:
            str: Prometheus-formatted metrics
        
        Requirements: 10.7
        """
        metrics = self.get_statistics()
        lines = []
        
        # Compression metrics
        lines.append("# HELP llm_compression_total Total number of compressions")
        lines.append("# TYPE llm_compression_total counter")
        lines.append(f"llm_compression_total {metrics.total_compressions}")
        
        lines.append("# HELP llm_compression_latency_ms Average compression latency in milliseconds")
        lines.append("# TYPE llm_compression_latency_ms gauge")
        lines.append(f"llm_compression_latency_ms {metrics.avg_compression_latency_ms}")
        
        lines.append("# HELP llm_compression_ratio Average compression ratio")
        lines.append("# TYPE llm_compression_ratio gauge")
        lines.append(f"llm_compression_ratio {metrics.avg_compression_ratio}")
        
        # Reconstruction metrics
        lines.append("# HELP llm_reconstruction_total Total number of reconstructions")
        lines.append("# TYPE llm_reconstruction_total counter")
        lines.append(f"llm_reconstruction_total {metrics.total_reconstructions}")
        
        lines.append("# HELP llm_reconstruction_latency_ms Average reconstruction latency in milliseconds")
        lines.append("# TYPE llm_reconstruction_latency_ms gauge")
        lines.append(f"llm_reconstruction_latency_ms {metrics.avg_reconstruction_latency_ms}")
        
        # Quality metrics
        lines.append("# HELP llm_quality_score Average quality score")
        lines.append("# TYPE llm_quality_score gauge")
        lines.append(f"llm_quality_score {metrics.avg_quality_score}")
        
        lines.append("# HELP llm_low_quality_total Total low quality compressions")
        lines.append("# TYPE llm_low_quality_total counter")
        lines.append(f"llm_low_quality_total {metrics.low_quality_count}")
        
        # API metrics
        lines.append("# HELP llm_api_calls_total Total API calls")
        lines.append("# TYPE llm_api_calls_total counter")
        lines.append(f"llm_api_calls_total {metrics.total_api_calls}")
        
        lines.append("# HELP llm_api_errors_total Total API errors")
        lines.append("# TYPE llm_api_errors_total counter")
        lines.append(f"llm_api_errors_total {metrics.api_error_count}")
        
        # Throughput
        lines.append("# HELP llm_throughput_per_minute Compression throughput per minute")
        lines.append("# TYPE llm_throughput_per_minute gauge")
        lines.append(f"llm_throughput_per_minute {metrics.throughput_per_minute}")
        
        return "\n".join(lines)
    
    async def _check_quality_drop(self):
        """
        Check for quality drop and trigger alert
        
        Requirements: 10.4
        """
        if len(self.metrics['quality_score']) < 10:
            # Need at least 10 samples
            return
        
        # Calculate recent average (last 10 samples)
        recent_quality = statistics.mean(self.metrics['quality_score'][-10:])
        
        if self.previous_quality_avg is None:
            self.previous_quality_avg = recent_quality
            return
        
        # Check for drop
        quality_drop = self.previous_quality_avg - recent_quality
        drop_percentage = quality_drop / self.previous_quality_avg
        
        if drop_percentage > self.quality_drop_threshold:
            logger.warning(
                f"QUALITY ALERT: Quality dropped by {drop_percentage * 100:.1f}% "
                f"({self.previous_quality_avg:.3f} -> {recent_quality:.3f})"
            )
            # TODO: Send alert notification (email, Slack, etc.)
        
        self.previous_quality_avg = recent_quality
    
    def _percentile(self, values: List[float], p: float) -> float:
        """
        Calculate percentile
        
        Args:
            values: List of values
            p: Percentile (0-1)
        
        Returns:
            float: Percentile value
        """
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    async def reset(self):
        """Reset all metrics"""
        async with self.lock:
            for key in self.metrics:
                self.metrics[key].clear()
            
            self.model_metrics.clear()
            
            for key in self.counters:
                self.counters[key] = 0
            
            for key in self.cost_data:
                if isinstance(self.cost_data[key], (int, float)):
                    self.cost_data[key] = 0
            
            self.start_time = datetime.now()
            self.previous_quality_avg = None
            
            logger.info("Performance metrics reset")

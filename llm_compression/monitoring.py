"""
Monitoring and Alerting System

Provides monitoring, alerting, and notification capabilities:
- Quality degradation alerts
- Alert notification system (logging, callbacks)
- Integration with PerformanceMonitor

Feature: llm-compression-integration
Requirements: 10.4
"""

import asyncio
from typing import Callable, List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from llm_compression.logger import logger


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert notification"""
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold: float
    metadata: Dict[str, Any]


class AlertNotifier:
    """
    Alert notification system
    
    Supports multiple notification channels:
    - Logging (always enabled)
    - Custom callbacks (for email, Slack, etc.)
    
    Requirements: 10.4
    """
    
    def __init__(self):
        """Initialize alert notifier"""
        self.callbacks: List[Callable[[Alert], None]] = []
        self.alert_history: List[Alert] = []
        self.lock = asyncio.Lock()
        
        logger.info("AlertNotifier initialized")
    
    def register_callback(self, callback: Callable[[Alert], None]):
        """
        Register a custom alert callback
        
        Args:
            callback: Function to call when alert is triggered
                     Signature: callback(alert: Alert) -> None
        
        Example:
            def send_slack_alert(alert: Alert):
                # Send to Slack
                pass
            
            notifier.register_callback(send_slack_alert)
        """
        self.callbacks.append(callback)
        logger.info(f"Registered alert callback: {callback.__name__}")
    
    async def send_alert(self, alert: Alert):
        """
        Send alert notification
        
        Args:
            alert: Alert to send
        
        Requirements: 10.4
        """
        async with self.lock:
            # Store in history
            self.alert_history.append(alert)
            
            # Log alert
            log_message = (
                f"ALERT [{alert.severity.value.upper()}] {alert.metric_name}: "
                f"{alert.message} (current={alert.current_value:.3f}, "
                f"threshold={alert.threshold:.3f})"
            )
            
            if alert.severity == AlertSeverity.CRITICAL:
                logger.critical(log_message)
            elif alert.severity == AlertSeverity.ERROR:
                logger.error(log_message)
            elif alert.severity == AlertSeverity.WARNING:
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            # Call custom callbacks
            for callback in self.callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def get_alert_history(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        Get alert history
        
        Args:
            severity: Filter by severity (optional)
            limit: Maximum number of alerts to return
        
        Returns:
            List of alerts
        """
        alerts = self.alert_history
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts[-limit:]
    
    async def clear_history(self):
        """Clear alert history"""
        async with self.lock:
            self.alert_history.clear()
            logger.info("Alert history cleared")


class MonitoringSystem:
    """
    Monitoring system with alerting capabilities
    
    Wraps PerformanceMonitor and adds:
    - Quality degradation detection
    - Alert notifications
    - Threshold-based monitoring
    
    Requirements: 10.1, 10.4
    """
    
    def __init__(
        self,
        performance_monitor,
        alert_notifier: Optional[AlertNotifier] = None,
        quality_alert_threshold: float = 0.85,
        quality_drop_threshold: float = 0.10
    ):
        """
        Initialize monitoring system
        
        Args:
            performance_monitor: PerformanceMonitor instance
            alert_notifier: AlertNotifier instance (creates new if None)
            quality_alert_threshold: Quality threshold for alerts
            quality_drop_threshold: Quality drop threshold (10%)
        
        Requirements: 10.1, 10.4
        """
        self.performance_monitor = performance_monitor
        self.alert_notifier = alert_notifier or AlertNotifier()
        self.quality_alert_threshold = quality_alert_threshold
        self.quality_drop_threshold = quality_drop_threshold
        
        logger.info(
            f"MonitoringSystem initialized: "
            f"quality_threshold={quality_alert_threshold}, "
            f"drop_threshold={quality_drop_threshold}"
        )
    
    async def check_quality_degradation(self):
        """
        Check for quality degradation and send alerts
        
        Requirements: 10.4
        """
        metrics = self.performance_monitor.get_statistics()
        
        # Check if quality is below threshold
        if metrics.avg_quality_score < self.quality_alert_threshold:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                message=f"Quality below threshold",
                timestamp=datetime.now(),
                metric_name="quality_score",
                current_value=metrics.avg_quality_score,
                threshold=self.quality_alert_threshold,
                metadata={
                    "low_quality_count": metrics.low_quality_count,
                    "total_compressions": metrics.total_compressions
                }
            )
            await self.alert_notifier.send_alert(alert)
        
        # Check for quality drop
        if (self.performance_monitor.previous_quality_avg is not None and
            len(self.performance_monitor.metrics['quality_score']) >= 10):
            
            recent_quality = sum(
                self.performance_monitor.metrics['quality_score'][-10:]
            ) / 10
            
            quality_drop = (
                self.performance_monitor.previous_quality_avg - recent_quality
            )
            drop_percentage = (
                quality_drop / self.performance_monitor.previous_quality_avg
            )
            
            if drop_percentage > self.quality_drop_threshold:
                alert = Alert(
                    severity=AlertSeverity.ERROR,
                    message=f"Quality dropped by {drop_percentage * 100:.1f}%",
                    timestamp=datetime.now(),
                    metric_name="quality_drop",
                    current_value=recent_quality,
                    threshold=self.performance_monitor.previous_quality_avg,
                    metadata={
                        "drop_percentage": drop_percentage,
                        "previous_avg": self.performance_monitor.previous_quality_avg,
                        "recent_avg": recent_quality
                    }
                )
                await self.alert_notifier.send_alert(alert)
    
    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """
        Register custom alert callback
        
        Args:
            callback: Function to call when alert is triggered
        
        Example:
            def send_email_alert(alert: Alert):
                # Send email
                pass
            
            monitoring.register_alert_callback(send_email_alert)
        """
        self.alert_notifier.register_callback(callback)
    
    def get_alert_history(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get alert history"""
        return self.alert_notifier.get_alert_history(severity, limit)
    
    async def record_compression(
        self,
        latency_ms: float,
        compression_ratio: float,
        quality_score: float,
        model_used: str,
        tokens_used: int = 0
    ):
        """
        Record compression and check for alerts
        
        Args:
            latency_ms: Compression latency
            compression_ratio: Compression ratio
            quality_score: Quality score
            model_used: Model name
            tokens_used: Tokens used
        """
        # Record in performance monitor
        await self.performance_monitor.record_compression(
            latency_ms,
            compression_ratio,
            quality_score,
            model_used,
            tokens_used
        )
        
        # Check for quality issues
        await self.check_quality_degradation()
    
    def get_statistics(self):
        """Get performance statistics"""
        return self.performance_monitor.get_statistics()
    
    def get_model_comparison(self):
        """Get model performance comparison"""
        return self.performance_monitor.get_model_comparison()
    
    def estimate_cost_savings(self, **kwargs):
        """Estimate cost savings"""
        return self.performance_monitor.estimate_cost_savings(**kwargs)
    
    def generate_report(self) -> str:
        """Generate performance report"""
        return self.performance_monitor.generate_report()
    
    def export_prometheus_metrics(self) -> str:
        """Export Prometheus metrics"""
        return self.performance_monitor.export_prometheus_metrics()

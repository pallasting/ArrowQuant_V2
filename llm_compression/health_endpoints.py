"""
Health Check Endpoints - HTTP endpoints for health monitoring

Provides REST API endpoints for health checks, readiness, and liveness probes.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict
import asyncio

from .availability_monitor import (
    AvailabilityMonitor,
    HealthStatus,
    ComponentType
)
from .circuit_breaker import CircuitBreakerRegistry


class HealthEndpoints:
    """
    Health check endpoints for monitoring and orchestration
    
    Provides standard health check endpoints compatible with
    Kubernetes and other orchestration systems.
    """
    
    def __init__(
        self,
        availability_monitor: AvailabilityMonitor,
        circuit_breaker_registry: CircuitBreakerRegistry
    ):
        """
        Initialize health endpoints
        
        Args:
            availability_monitor: Availability monitoring system
            circuit_breaker_registry: Circuit breaker registry
        """
        self.monitor = availability_monitor
        self.circuit_breakers = circuit_breaker_registry
        self.start_time = datetime.now()
    
    async def health(self) -> Dict[str, Any]:
        """
        Basic health check endpoint
        
        Returns:
            Health status and basic metrics
        """
        metrics = self.monitor.get_availability_metrics()
        component_health = self.monitor.get_component_health()
        
        # Determine overall health
        unhealthy_components = [
            comp for comp, status in component_health.items()
            if status == HealthStatus.UNHEALTHY
        ]
        
        degraded_components = [
            comp for comp, status in component_health.items()
            if status == HealthStatus.DEGRADED
        ]
        
        if unhealthy_components:
            overall_status = "unhealthy"
        elif degraded_components:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'availability_percentage': metrics.availability_percentage,
            'components': {
                comp: status.value
                for comp, status in component_health.items()
            },
            'unhealthy_components': unhealthy_components,
            'degraded_components': degraded_components,
            'active_incidents': len(self.monitor.get_active_incidents())
        }
    
    async def readiness(self) -> Dict[str, Any]:
        """
        Readiness probe endpoint
        
        Indicates if the service is ready to accept traffic.
        Returns 200 if ready, 503 if not ready.
        
        Returns:
            Readiness status
        """
        component_health = self.monitor.get_component_health()
        
        # Check critical components
        critical_components = [
            ComponentType.STORAGE.value,
            ComponentType.EMBEDDER.value,
            ComponentType.VECTOR_SEARCH.value
        ]
        
        critical_unhealthy = [
            comp for comp in critical_components
            if component_health.get(comp) == HealthStatus.UNHEALTHY
        ]
        
        ready = len(critical_unhealthy) == 0
        
        return {
            'ready': ready,
            'timestamp': datetime.now().isoformat(),
            'critical_components': {
                comp: component_health.get(comp, HealthStatus.UNKNOWN).value
                for comp in critical_components
            },
            'unhealthy_critical': critical_unhealthy
        }
    
    async def liveness(self) -> Dict[str, Any]:
        """
        Liveness probe endpoint
        
        Indicates if the service is alive and should not be restarted.
        Returns 200 if alive, 503 if dead.
        
        Returns:
            Liveness status
        """
        # Check if monitor is running
        alive = self.monitor._running
        
        # Check for prolonged incidents
        active_incidents = self.monitor.get_active_incidents()
        prolonged_incidents = [
            incident for incident in active_incidents
            if (datetime.now() - incident.start_time).total_seconds() > 300
        ]
        
        # Consider dead if prolonged incidents exist
        if prolonged_incidents:
            alive = False
        
        return {
            'alive': alive,
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'prolonged_incidents': len(prolonged_incidents)
        }
    
    async def metrics(self) -> Dict[str, Any]:
        """
        Detailed metrics endpoint
        
        Returns:
            Comprehensive system metrics
        """
        # Availability metrics
        metrics_24h = self.monitor.get_availability_metrics(
            time_window=timedelta(hours=24)
        )
        metrics_7d = self.monitor.get_availability_metrics(
            time_window=timedelta(days=7)
        )
        metrics_all = self.monitor.get_availability_metrics()
        
        # Circuit breaker metrics
        circuit_metrics = self.circuit_breakers.get_all_metrics()
        
        # Component health
        component_health = self.monitor.get_component_health()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'availability': {
                '24h': {
                    'percentage': metrics_24h.availability_percentage,
                    'uptime_seconds': metrics_24h.uptime_seconds,
                    'downtime_seconds': metrics_24h.downtime_seconds,
                    'mtbf': metrics_24h.mtbf,
                    'mttr': metrics_24h.mttr
                },
                '7d': {
                    'percentage': metrics_7d.availability_percentage,
                    'uptime_seconds': metrics_7d.uptime_seconds,
                    'downtime_seconds': metrics_7d.downtime_seconds,
                    'mtbf': metrics_7d.mtbf,
                    'mttr': metrics_7d.mttr
                },
                'all_time': {
                    'percentage': metrics_all.availability_percentage,
                    'uptime_seconds': metrics_all.uptime_seconds,
                    'downtime_seconds': metrics_all.downtime_seconds,
                    'mtbf': metrics_all.mtbf,
                    'mttr': metrics_all.mttr
                }
            },
            'health_checks': {
                'total': metrics_all.total_checks,
                'successful': metrics_all.successful_checks,
                'failed': metrics_all.failed_checks,
                'success_rate': (
                    metrics_all.successful_checks / metrics_all.total_checks * 100
                    if metrics_all.total_checks > 0 else 0
                )
            },
            'components': {
                comp: status.value
                for comp, status in component_health.items()
            },
            'circuit_breakers': circuit_metrics,
            'incidents': {
                'active': len(self.monitor.get_active_incidents()),
                'total_24h': len(self.monitor.get_incident_history(
                    time_window=timedelta(hours=24)
                )),
                'total_7d': len(self.monitor.get_incident_history(
                    time_window=timedelta(days=7)
                ))
            }
        }
    
    async def incidents(self) -> Dict[str, Any]:
        """
        Incident history endpoint
        
        Returns:
            Recent incidents
        """
        active = self.monitor.get_active_incidents()
        recent = self.monitor.get_incident_history(
            time_window=timedelta(hours=24)
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'active_incidents': [
                {
                    'component': inc.component,
                    'start_time': inc.start_time.isoformat(),
                    'duration_seconds': (
                        datetime.now() - inc.start_time
                    ).total_seconds(),
                    'error': inc.error
                }
                for inc in active
            ],
            'recent_incidents_24h': [
                {
                    'component': inc.component,
                    'start_time': inc.start_time.isoformat(),
                    'end_time': inc.end_time.isoformat() if inc.end_time else None,
                    'duration_seconds': inc.duration_seconds,
                    'error': inc.error,
                    'resolved': inc.resolved
                }
                for inc in recent
            ]
        }

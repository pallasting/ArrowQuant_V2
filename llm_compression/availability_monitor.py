"""
Availability Monitor - System availability tracking and health monitoring

Tracks system uptime, component health, and provides availability metrics
to achieve >99.9% availability target.
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class HealthStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """System component types"""
    STORAGE = "storage"
    EMBEDDER = "embedder"
    SEMANTIC_INDEXER = "semantic_indexer"
    VECTOR_SEARCH = "vector_search"
    MODEL_ROUTER = "model_router"
    COGNITIVE_LOOP = "cognitive_loop"
    API = "api"


@dataclass
class HealthCheck:
    """Health check result"""
    component: str
    status: HealthStatus
    timestamp: datetime
    latency_ms: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AvailabilityMetrics:
    """Availability metrics"""
    uptime_seconds: float
    downtime_seconds: float
    availability_percentage: float
    total_checks: int
    successful_checks: int
    failed_checks: int
    mtbf: float  # Mean Time Between Failures
    mttr: float  # Mean Time To Recovery
    last_incident: Optional[datetime] = None


@dataclass
class Incident:
    """Availability incident"""
    component: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    error: str
    resolved: bool = False


class AvailabilityMonitor:
    """
    System availability monitor
    
    Tracks component health, uptime, and availability metrics
    to ensure >99.9% availability target.
    """
    
    def __init__(
        self,
        check_interval: float = 60.0,
        history_file: Optional[Path] = None
    ):
        """
        Initialize availability monitor
        
        Args:
            check_interval: Health check interval in seconds
            history_file: File to persist availability history
        """
        self.check_interval = check_interval
        self.history_file = history_file or Path("availability_history.jsonl")
        
        # Health tracking
        self.health_checks: List[HealthCheck] = []
        self.component_status: Dict[str, HealthStatus] = {}
        
        # Availability tracking
        self.start_time = datetime.now()
        self.incidents: List[Incident] = []
        self.active_incidents: Dict[str, Incident] = {}
        
        # Metrics
        self.total_checks = 0
        self.successful_checks = 0
        self.failed_checks = 0
        
        # Background task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Load history
        self._load_history()
    
    async def start(self):
        """Start availability monitoring"""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop availability monitoring"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self._save_history()
    
    async def check_health(
        self,
        component: str,
        check_func: callable
    ) -> HealthCheck:
        """
        Perform health check on component
        
        Args:
            component: Component name
            check_func: Async function that performs health check
            
        Returns:
            HealthCheck result
        """
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            # Execute health check with timeout
            await asyncio.wait_for(check_func(), timeout=5.0)
            
            latency_ms = (time.time() - start_time) * 1000
            status = HealthStatus.HEALTHY
            error = None
            
            # Mark as degraded if latency is high
            if latency_ms > 1000:
                status = HealthStatus.DEGRADED
            
            self.successful_checks += 1
            
            # Resolve incident if exists
            if component in self.active_incidents:
                self._resolve_incident(component)
            
        except asyncio.TimeoutError:
            latency_ms = 5000.0
            status = HealthStatus.UNHEALTHY
            error = "Health check timeout"
            self.failed_checks += 1
            self._record_incident(component, error)
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            status = HealthStatus.UNHEALTHY
            error = str(e)
            self.failed_checks += 1
            self._record_incident(component, error)
        
        self.total_checks += 1
        
        # Create health check result
        health_check = HealthCheck(
            component=component,
            status=status,
            timestamp=timestamp,
            latency_ms=latency_ms,
            error=error
        )
        
        # Update tracking
        self.health_checks.append(health_check)
        self.component_status[component] = status
        
        # Keep only recent checks (last 1000)
        if len(self.health_checks) > 1000:
            self.health_checks = self.health_checks[-1000:]
        
        return health_check
    
    def get_availability_metrics(
        self,
        time_window: Optional[timedelta] = None
    ) -> AvailabilityMetrics:
        """
        Calculate availability metrics
        
        Args:
            time_window: Time window for metrics (None = all time)
            
        Returns:
            AvailabilityMetrics
        """
        now = datetime.now()
        
        # Filter incidents by time window
        if time_window:
            cutoff = now - time_window
            incidents = [
                i for i in self.incidents
                if i.start_time >= cutoff
            ]
        else:
            incidents = self.incidents
        
        # Calculate uptime and downtime
        if time_window:
            total_time = time_window.total_seconds()
        else:
            total_time = (now - self.start_time).total_seconds()
        
        downtime = sum(
            i.duration_seconds or 0
            for i in incidents
            if i.resolved
        )
        
        # Add active incident time
        for incident in self.active_incidents.values():
            downtime += (now - incident.start_time).total_seconds()
        
        uptime = total_time - downtime
        availability = (uptime / total_time * 100) if total_time > 0 else 100.0
        
        # Calculate MTBF and MTTR
        resolved_incidents = [i for i in incidents if i.resolved]
        
        if len(resolved_incidents) > 1:
            failure_intervals = []
            for i in range(1, len(resolved_incidents)):
                interval = (
                    resolved_incidents[i].start_time -
                    resolved_incidents[i-1].end_time
                ).total_seconds()
                failure_intervals.append(interval)
            mtbf = sum(failure_intervals) / len(failure_intervals)
        else:
            mtbf = total_time
        
        if resolved_incidents:
            mttr = sum(
                i.duration_seconds for i in resolved_incidents
            ) / len(resolved_incidents)
        else:
            mttr = 0.0
        
        last_incident = incidents[-1].start_time if incidents else None
        
        return AvailabilityMetrics(
            uptime_seconds=uptime,
            downtime_seconds=downtime,
            availability_percentage=availability,
            total_checks=self.total_checks,
            successful_checks=self.successful_checks,
            failed_checks=self.failed_checks,
            mtbf=mtbf,
            mttr=mttr,
            last_incident=last_incident
        )
    
    def get_component_health(self) -> Dict[str, HealthStatus]:
        """Get current health status of all components"""
        return self.component_status.copy()
    
    def get_recent_checks(
        self,
        component: Optional[str] = None,
        limit: int = 10
    ) -> List[HealthCheck]:
        """
        Get recent health checks
        
        Args:
            component: Filter by component (None = all)
            limit: Maximum number of checks to return
            
        Returns:
            List of recent health checks
        """
        checks = self.health_checks
        
        if component:
            checks = [c for c in checks if c.component == component]
        
        return checks[-limit:]
    
    def get_active_incidents(self) -> List[Incident]:
        """Get currently active incidents"""
        return list(self.active_incidents.values())
    
    def get_incident_history(
        self,
        time_window: Optional[timedelta] = None
    ) -> List[Incident]:
        """
        Get incident history
        
        Args:
            time_window: Time window for history (None = all time)
            
        Returns:
            List of incidents
        """
        if time_window:
            cutoff = datetime.now() - time_window
            return [
                i for i in self.incidents
                if i.start_time >= cutoff
            ]
        return self.incidents.copy()
    
    def _record_incident(self, component: str, error: str):
        """Record new incident"""
        if component in self.active_incidents:
            return  # Already tracking this incident
        
        incident = Incident(
            component=component,
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            error=error,
            resolved=False
        )
        
        self.active_incidents[component] = incident
        self.incidents.append(incident)
    
    def _resolve_incident(self, component: str):
        """Resolve active incident"""
        if component not in self.active_incidents:
            return
        
        incident = self.active_incidents[component]
        incident.end_time = datetime.now()
        incident.duration_seconds = (
            incident.end_time - incident.start_time
        ).total_seconds()
        incident.resolved = True
        
        del self.active_incidents[component]
    
    async def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                
                # Periodic save
                if self.total_checks % 10 == 0:
                    self._save_history()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Monitor loop error: {e}")
    
    def _save_history(self):
        """Save availability history to file"""
        try:
            with open(self.history_file, 'a') as f:
                metrics = self.get_availability_metrics()
                record = {
                    'timestamp': datetime.now().isoformat(),
                    'uptime_seconds': metrics.uptime_seconds,
                    'downtime_seconds': metrics.downtime_seconds,
                    'availability_percentage': metrics.availability_percentage,
                    'total_checks': metrics.total_checks,
                    'successful_checks': metrics.successful_checks,
                    'failed_checks': metrics.failed_checks,
                    'mtbf': metrics.mtbf,
                    'mttr': metrics.mttr,
                    'active_incidents': len(self.active_incidents)
                }
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            print(f"Failed to save history: {e}")
    
    def _load_history(self):
        """Load availability history from file"""
        if not self.history_file.exists():
            return
        
        try:
            with open(self.history_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Load last record to restore state
                    last_record = json.loads(lines[-1])
                    self.total_checks = last_record.get('total_checks', 0)
                    self.successful_checks = last_record.get('successful_checks', 0)
                    self.failed_checks = last_record.get('failed_checks', 0)
        except Exception as e:
            print(f"Failed to load history: {e}")

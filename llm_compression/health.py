"""
Health check endpoints for production deployment.
"""
import time
from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from llm_compression.logger import logger


@dataclass
class HealthStatus:
    """Health status information."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime = field(default_factory=datetime.now)
    checks: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    uptime_seconds: float = 0.0


class HealthChecker:
    """Health checker for production monitoring."""
    
    def __init__(self):
        """Initialize health checker."""
        self.start_time = time.time()
        self.last_check_time = None
        self.check_count = 0
        
    def check_health(self) -> HealthStatus:
        """
        Perform health check.
        
        Returns:
            HealthStatus: Current health status
        """
        self.check_count += 1
        self.last_check_time = time.time()
        
        checks = {
            "api_responsive": True,
            "memory_ok": self._check_memory(),
            "disk_ok": self._check_disk(),
        }
        
        # Determine overall status
        if all(checks.values()):
            status = "healthy"
        elif any(checks.values()):
            status = "degraded"
        else:
            status = "unhealthy"
        
        uptime = time.time() - self.start_time
        
        return HealthStatus(
            status=status,
            checks=checks,
            details={
                "check_count": self.check_count,
                "last_check": datetime.fromtimestamp(self.last_check_time).isoformat(),
            },
            uptime_seconds=uptime
        )
    
    def check_readiness(self) -> bool:
        """
        Check if service is ready to accept traffic.
        
        Returns:
            bool: True if ready
        """
        # Service is ready if it's been running for at least 10 seconds
        uptime = time.time() - self.start_time
        return uptime >= 10.0
    
    def _check_memory(self) -> bool:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            # Healthy if less than 90% memory used
            return memory.percent < 90.0
        except ImportError:
            logger.warning("psutil not available, skipping memory check")
            return True
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False
    
    def _check_disk(self) -> bool:
        """Check disk usage."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            # Healthy if less than 90% disk used
            return disk.percent < 90.0
        except ImportError:
            logger.warning("psutil not available, skipping disk check")
            return True
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return False

"""
Tests for availability monitoring system
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from llm_compression.availability_monitor import (
    AvailabilityMonitor,
    HealthStatus,
    ComponentType,
    HealthCheck,
    Incident
)


@pytest.fixture
def temp_history_file():
    """Create temporary history file"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
async def monitor(temp_history_file):
    """Create availability monitor"""
    mon = AvailabilityMonitor(
        check_interval=0.1,
        history_file=temp_history_file
    )
    await mon.start()
    yield mon
    await mon.stop()


@pytest.mark.asyncio
async def test_health_check_success(monitor):
    """Test successful health check"""
    async def healthy_component():
        await asyncio.sleep(0.01)
        return True
    
    check = await monitor.check_health("test_component", healthy_component)
    
    assert check.component == "test_component"
    assert check.status == HealthStatus.HEALTHY
    assert check.error is None
    assert check.latency_ms < 100
    assert monitor.successful_checks == 1
    assert monitor.failed_checks == 0


@pytest.mark.asyncio
async def test_health_check_failure(monitor):
    """Test failed health check"""
    async def failing_component():
        raise Exception("Component failed")
    
    check = await monitor.check_health("test_component", failing_component)
    
    assert check.component == "test_component"
    assert check.status == HealthStatus.UNHEALTHY
    assert check.error == "Component failed"
    assert monitor.successful_checks == 0
    assert monitor.failed_checks == 1


@pytest.mark.asyncio
async def test_health_check_timeout(monitor):
    """Test health check timeout"""
    async def slow_component():
        await asyncio.sleep(10)
        return True
    
    check = await monitor.check_health("test_component", slow_component)
    
    assert check.component == "test_component"
    assert check.status == HealthStatus.UNHEALTHY
    assert check.error == "Health check timeout"
    assert check.latency_ms >= 5000


@pytest.mark.asyncio
async def test_health_check_degraded(monitor):
    """Test degraded health check (slow but successful)"""
    async def slow_component():
        await asyncio.sleep(1.5)
        return True
    
    check = await monitor.check_health("test_component", slow_component)
    
    assert check.component == "test_component"
    assert check.status == HealthStatus.DEGRADED
    assert check.error is None
    assert check.latency_ms > 1000


@pytest.mark.asyncio
async def test_incident_recording(monitor):
    """Test incident recording and resolution"""
    # Trigger failure
    async def failing_component():
        raise Exception("Test failure")
    
    await monitor.check_health("test_component", failing_component)
    
    # Check incident recorded
    active = monitor.get_active_incidents()
    assert len(active) == 1
    assert active[0].component == "test_component"
    assert active[0].resolved is False
    
    # Trigger success to resolve
    async def healthy_component():
        return True
    
    await monitor.check_health("test_component", healthy_component)
    
    # Check incident resolved
    active = monitor.get_active_incidents()
    assert len(active) == 0
    
    history = monitor.get_incident_history()
    assert len(history) == 1
    assert history[0].resolved is True
    assert history[0].duration_seconds is not None


@pytest.mark.asyncio
async def test_availability_metrics(monitor):
    """Test availability metrics calculation"""
    # Perform some checks
    async def healthy():
        return True
    
    async def failing():
        raise Exception("Fail")
    
    # 8 successful, 2 failed
    for _ in range(8):
        await monitor.check_health("component", healthy)
    
    for _ in range(2):
        await monitor.check_health("component", failing)
    
    # Resolve incident
    await monitor.check_health("component", healthy)
    
    metrics = monitor.get_availability_metrics()
    
    assert metrics.total_checks == 11
    assert metrics.successful_checks == 9
    assert metrics.failed_checks == 2
    assert metrics.availability_percentage > 99.0  # Should be high


@pytest.mark.asyncio
async def test_component_status_tracking(monitor):
    """Test component status tracking"""
    async def healthy():
        return True
    
    await monitor.check_health("component_a", healthy)
    await monitor.check_health("component_b", healthy)
    
    status = monitor.get_component_health()
    
    assert "component_a" in status
    assert "component_b" in status
    assert status["component_a"] == HealthStatus.HEALTHY
    assert status["component_b"] == HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_recent_checks(monitor):
    """Test recent checks retrieval"""
    async def healthy():
        return True
    
    # Perform multiple checks
    for i in range(5):
        await monitor.check_health(f"component_{i}", healthy)
    
    # Get all recent checks
    recent = monitor.get_recent_checks(limit=10)
    assert len(recent) == 5
    
    # Get recent checks for specific component
    recent = monitor.get_recent_checks(component="component_0", limit=10)
    assert len(recent) == 1
    assert recent[0].component == "component_0"


@pytest.mark.asyncio
async def test_history_persistence(temp_history_file):
    """Test history persistence"""
    # Create monitor and perform checks
    monitor1 = AvailabilityMonitor(
        check_interval=0.1,
        history_file=temp_history_file
    )
    await monitor1.start()
    
    async def healthy():
        return True
    
    for _ in range(5):
        await monitor1.check_health("component", healthy)
    
    await monitor1.stop()
    
    # Create new monitor and verify state restored
    monitor2 = AvailabilityMonitor(
        check_interval=0.1,
        history_file=temp_history_file
    )
    
    assert monitor2.total_checks == 5
    assert monitor2.successful_checks == 5


@pytest.mark.asyncio
async def test_mtbf_mttr_calculation(monitor):
    """Test MTBF and MTTR calculation"""
    async def healthy():
        return True
    
    async def failing():
        raise Exception("Fail")
    
    # Create pattern: fail -> recover -> fail -> recover
    await monitor.check_health("component", failing)
    await asyncio.sleep(0.1)
    await monitor.check_health("component", healthy)
    
    await asyncio.sleep(0.2)
    
    await monitor.check_health("component", failing)
    await asyncio.sleep(0.1)
    await monitor.check_health("component", healthy)
    
    metrics = monitor.get_availability_metrics()
    
    assert metrics.mtbf > 0  # Mean time between failures
    assert metrics.mttr > 0  # Mean time to recovery
    assert metrics.mttr < metrics.mtbf  # Recovery should be faster


@pytest.mark.asyncio
async def test_time_window_metrics(monitor):
    """Test metrics with time window"""
    async def healthy():
        return True
    
    # Perform checks
    for _ in range(10):
        await monitor.check_health("component", healthy)
    
    # Get metrics for different windows
    metrics_1h = monitor.get_availability_metrics(
        time_window=timedelta(hours=1)
    )
    metrics_all = monitor.get_availability_metrics()
    
    assert metrics_1h.total_checks == metrics_all.total_checks
    assert metrics_1h.availability_percentage == metrics_all.availability_percentage


def test_health_check_dataclass():
    """Test HealthCheck dataclass"""
    check = HealthCheck(
        component="test",
        status=HealthStatus.HEALTHY,
        timestamp=datetime.now(),
        latency_ms=10.5,
        error=None,
        metadata={'key': 'value'}
    )
    
    assert check.component == "test"
    assert check.status == HealthStatus.HEALTHY
    assert check.latency_ms == 10.5
    assert check.metadata['key'] == 'value'


def test_incident_dataclass():
    """Test Incident dataclass"""
    start = datetime.now()
    end = start + timedelta(seconds=30)
    
    incident = Incident(
        component="test",
        start_time=start,
        end_time=end,
        duration_seconds=30.0,
        error="Test error",
        resolved=True
    )
    
    assert incident.component == "test"
    assert incident.duration_seconds == 30.0
    assert incident.resolved is True

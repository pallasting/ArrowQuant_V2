"""
Tests for health check endpoints
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from llm_compression.availability_monitor import (
    AvailabilityMonitor,
    HealthStatus
)
from llm_compression.circuit_breaker import CircuitBreakerRegistry
from llm_compression.health_endpoints import HealthEndpoints


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


@pytest.fixture
def circuit_registry():
    """Create circuit breaker registry"""
    return CircuitBreakerRegistry()


@pytest.fixture
def endpoints(monitor, circuit_registry):
    """Create health endpoints"""
    return HealthEndpoints(monitor, circuit_registry)


@pytest.mark.asyncio
async def test_health_endpoint_healthy(endpoints, monitor):
    """Test health endpoint with healthy system"""
    async def healthy():
        return True
    
    # Perform health checks
    await monitor.check_health("component_a", healthy)
    await monitor.check_health("component_b", healthy)
    
    response = await endpoints.health()
    
    assert response['status'] == 'healthy'
    assert 'timestamp' in response
    assert response['availability_percentage'] > 99.0
    assert len(response['unhealthy_components']) == 0
    assert len(response['degraded_components']) == 0


@pytest.mark.asyncio
async def test_health_endpoint_degraded(endpoints, monitor):
    """Test health endpoint with degraded component"""
    async def healthy():
        return True
    
    async def slow():
        await asyncio.sleep(1.5)
        return True
    
    await monitor.check_health("component_a", healthy)
    await monitor.check_health("component_b", slow)
    
    response = await endpoints.health()
    
    assert response['status'] == 'degraded'
    assert len(response['degraded_components']) == 1
    assert 'component_b' in response['degraded_components']


@pytest.mark.asyncio
async def test_health_endpoint_unhealthy(endpoints, monitor):
    """Test health endpoint with unhealthy component"""
    async def healthy():
        return True
    
    async def failing():
        raise Exception("Component failed")
    
    await monitor.check_health("component_a", healthy)
    await monitor.check_health("component_b", failing)
    
    response = await endpoints.health()
    
    assert response['status'] == 'unhealthy'
    assert len(response['unhealthy_components']) == 1
    assert 'component_b' in response['unhealthy_components']


@pytest.mark.asyncio
async def test_readiness_endpoint_ready(endpoints, monitor):
    """Test readiness endpoint when ready"""
    async def healthy():
        return True
    
    # Check critical components
    await monitor.check_health("storage", healthy)
    await monitor.check_health("embedder", healthy)
    await monitor.check_health("vector_search", healthy)
    
    response = await endpoints.readiness()
    
    assert response['ready'] is True
    assert 'timestamp' in response
    assert len(response['unhealthy_critical']) == 0


@pytest.mark.asyncio
async def test_readiness_endpoint_not_ready(endpoints, monitor):
    """Test readiness endpoint when not ready"""
    async def healthy():
        return True
    
    async def failing():
        raise Exception("Failed")
    
    await monitor.check_health("storage", failing)
    await monitor.check_health("embedder", healthy)
    
    response = await endpoints.readiness()
    
    assert response['ready'] is False
    assert 'storage' in response['unhealthy_critical']


@pytest.mark.asyncio
async def test_liveness_endpoint_alive(endpoints):
    """Test liveness endpoint when alive"""
    response = await endpoints.liveness()
    
    assert response['alive'] is True
    assert 'timestamp' in response
    assert response['uptime_seconds'] >= 0


@pytest.mark.asyncio
async def test_liveness_endpoint_dead(endpoints, monitor):
    """Test liveness endpoint with prolonged incident"""
    # Stop monitor to simulate dead state
    await monitor.stop()
    
    response = await endpoints.liveness()
    
    assert response['alive'] is False


@pytest.mark.asyncio
async def test_metrics_endpoint(endpoints, monitor):
    """Test metrics endpoint"""
    async def healthy():
        return True
    
    # Perform some checks
    for _ in range(10):
        await monitor.check_health("component", healthy)
    
    response = await endpoints.metrics()
    
    assert 'timestamp' in response
    assert 'availability' in response
    assert '24h' in response['availability']
    assert '7d' in response['availability']
    assert 'all_time' in response['availability']
    assert response['health_checks']['total'] == 10
    assert response['health_checks']['successful'] == 10


@pytest.mark.asyncio
async def test_metrics_endpoint_with_failures(endpoints, monitor):
    """Test metrics endpoint with failures"""
    async def healthy():
        return True
    
    async def failing():
        raise Exception("Failed")
    
    # Mix of success and failure
    for _ in range(8):
        await monitor.check_health("component", healthy)
    for _ in range(2):
        await monitor.check_health("component", failing)
    
    response = await endpoints.metrics()
    
    assert response['health_checks']['total'] == 10
    assert response['health_checks']['successful'] == 8
    assert response['health_checks']['failed'] == 2
    assert response['health_checks']['success_rate'] == 80.0


@pytest.mark.asyncio
async def test_incidents_endpoint(endpoints, monitor):
    """Test incidents endpoint"""
    async def healthy():
        return True
    
    async def failing():
        raise Exception("Test failure")
    
    # Create incident
    await monitor.check_health("component", failing)
    
    response = await endpoints.incidents()
    
    assert 'timestamp' in response
    assert len(response['active_incidents']) == 1
    assert response['active_incidents'][0]['component'] == 'component'
    assert response['active_incidents'][0]['error'] == 'Test failure'
    
    # Resolve incident
    await monitor.check_health("component", healthy)
    
    response = await endpoints.incidents()
    assert len(response['active_incidents']) == 0
    assert len(response['recent_incidents_24h']) == 1


@pytest.mark.asyncio
async def test_metrics_with_circuit_breakers(endpoints, circuit_registry):
    """Test metrics endpoint includes circuit breaker data"""
    # Create some circuit breakers
    circuit_registry.get_or_create("breaker1")
    circuit_registry.get_or_create("breaker2")
    
    response = await endpoints.metrics()
    
    assert 'circuit_breakers' in response
    assert len(response['circuit_breakers']) == 2


@pytest.mark.asyncio
async def test_component_health_in_response(endpoints, monitor):
    """Test component health included in responses"""
    async def healthy():
        return True
    
    await monitor.check_health("component_a", healthy)
    await monitor.check_health("component_b", healthy)
    
    response = await endpoints.health()
    
    assert 'components' in response
    assert 'component_a' in response['components']
    assert 'component_b' in response['components']
    assert response['components']['component_a'] == 'healthy'


@pytest.mark.asyncio
async def test_uptime_tracking(endpoints):
    """Test uptime tracking in endpoints"""
    await asyncio.sleep(0.1)
    
    response = await endpoints.health()
    assert response['uptime_seconds'] > 0
    
    response = await endpoints.liveness()
    assert response['uptime_seconds'] > 0

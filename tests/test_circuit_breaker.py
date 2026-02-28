"""
Tests for circuit breaker implementation
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from llm_compression.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerError,
    CircuitBreakerRegistry
)


@pytest.fixture
def config():
    """Create test configuration"""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=1.0,
        reset_timeout=5.0
    )


@pytest.fixture
def breaker(config):
    """Create circuit breaker"""
    return CircuitBreaker("test_breaker", config)


@pytest.mark.asyncio
async def test_circuit_closed_success(breaker):
    """Test successful calls with closed circuit"""
    async def successful_func():
        return "success"
    
    result = await breaker.call(successful_func)
    
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED
    assert breaker.successful_calls == 1
    assert breaker.failed_calls == 0


@pytest.mark.asyncio
async def test_circuit_opens_on_failures(breaker):
    """Test circuit opens after threshold failures"""
    async def failing_func():
        raise Exception("Failure")
    
    # Trigger failures
    for i in range(3):
        with pytest.raises(Exception):
            await breaker.call(failing_func)
    
    assert breaker.state == CircuitState.OPEN
    assert breaker.failed_calls == 3


@pytest.mark.asyncio
async def test_circuit_rejects_when_open(breaker):
    """Test circuit rejects calls when open"""
    async def failing_func():
        raise Exception("Failure")
    
    # Open the circuit
    for _ in range(3):
        with pytest.raises(Exception):
            await breaker.call(failing_func)
    
    assert breaker.state == CircuitState.OPEN
    
    # Try to call - should be rejected
    with pytest.raises(CircuitBreakerError):
        await breaker.call(failing_func)
    
    assert breaker.rejected_calls == 1


@pytest.mark.asyncio
async def test_circuit_half_open_after_timeout(breaker):
    """Test circuit transitions to half-open after timeout"""
    async def failing_func():
        raise Exception("Failure")
    
    async def successful_func():
        return "success"
    
    # Open the circuit
    for _ in range(3):
        with pytest.raises(Exception):
            await breaker.call(failing_func)
    
    assert breaker.state == CircuitState.OPEN
    
    # Wait for timeout
    await asyncio.sleep(1.1)
    
    # Next call should transition to half-open
    result = await breaker.call(successful_func)
    
    assert result == "success"
    assert breaker.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_circuit_closes_from_half_open(breaker):
    """Test circuit closes after successful calls in half-open"""
    async def failing_func():
        raise Exception("Failure")
    
    async def successful_func():
        return "success"
    
    # Open the circuit
    for _ in range(3):
        with pytest.raises(Exception):
            await breaker.call(failing_func)
    
    # Wait and transition to half-open
    await asyncio.sleep(1.1)
    
    # Successful calls to close
    await breaker.call(successful_func)
    assert breaker.state == CircuitState.HALF_OPEN
    
    await breaker.call(successful_func)
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_reopens_on_half_open_failure(breaker):
    """Test circuit reopens if failure occurs in half-open"""
    async def failing_func():
        raise Exception("Failure")
    
    async def successful_func():
        return "success"
    
    # Open the circuit
    for _ in range(3):
        with pytest.raises(Exception):
            await breaker.call(failing_func)
    
    # Wait and transition to half-open
    await asyncio.sleep(1.1)
    await breaker.call(successful_func)
    assert breaker.state == CircuitState.HALF_OPEN
    
    # Failure should reopen
    with pytest.raises(Exception):
        await breaker.call(failing_func)
    
    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_resets_failure_count_on_success(breaker):
    """Test failure count resets after success"""
    async def failing_func():
        raise Exception("Failure")
    
    async def successful_func():
        return "success"
    
    # Partial failures
    with pytest.raises(Exception):
        await breaker.call(failing_func)
    with pytest.raises(Exception):
        await breaker.call(failing_func)
    
    assert breaker.failure_count == 2
    assert breaker.state == CircuitState.CLOSED
    
    # Success resets count
    await breaker.call(successful_func)
    assert breaker.failure_count == 0


def test_circuit_breaker_metrics(breaker):
    """Test circuit breaker metrics"""
    metrics = breaker.get_metrics()
    
    assert metrics['name'] == 'test_breaker'
    assert metrics['state'] == CircuitState.CLOSED.value
    assert metrics['total_calls'] == 0
    assert metrics['successful_calls'] == 0
    assert metrics['failed_calls'] == 0


def test_circuit_breaker_manual_reset(breaker):
    """Test manual circuit breaker reset"""
    breaker.state = CircuitState.OPEN
    breaker.failure_count = 5
    
    breaker.reset()
    
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


@pytest.mark.asyncio
async def test_sync_function_support(breaker):
    """Test circuit breaker with synchronous functions"""
    def sync_func():
        return "sync_result"
    
    result = await breaker.call(sync_func)
    assert result == "sync_result"


def test_circuit_breaker_registry():
    """Test circuit breaker registry"""
    registry = CircuitBreakerRegistry()
    
    # Get or create breakers
    breaker1 = registry.get_or_create("breaker1")
    breaker2 = registry.get_or_create("breaker2")
    breaker1_again = registry.get_or_create("breaker1")
    
    assert breaker1 is breaker1_again
    assert breaker1 is not breaker2
    assert len(registry.breakers) == 2


def test_registry_get_all_metrics():
    """Test getting all metrics from registry"""
    registry = CircuitBreakerRegistry()
    
    registry.get_or_create("breaker1")
    registry.get_or_create("breaker2")
    
    metrics = registry.get_all_metrics()
    
    assert len(metrics) == 2
    assert metrics[0]['name'] in ['breaker1', 'breaker2']
    assert metrics[1]['name'] in ['breaker1', 'breaker2']


def test_registry_reset_all():
    """Test resetting all circuit breakers"""
    registry = CircuitBreakerRegistry()
    
    breaker1 = registry.get_or_create("breaker1")
    breaker2 = registry.get_or_create("breaker2")
    
    # Open circuits
    breaker1.state = CircuitState.OPEN
    breaker2.state = CircuitState.OPEN
    
    # Reset all
    registry.reset_all()
    
    assert breaker1.state == CircuitState.CLOSED
    assert breaker2.state == CircuitState.CLOSED


def test_custom_config():
    """Test circuit breaker with custom configuration"""
    config = CircuitBreakerConfig(
        failure_threshold=10,
        success_threshold=5,
        timeout=30.0,
        reset_timeout=60.0
    )
    
    breaker = CircuitBreaker("custom", config)
    
    assert breaker.config.failure_threshold == 10
    assert breaker.config.success_threshold == 5
    assert breaker.config.timeout == 30.0

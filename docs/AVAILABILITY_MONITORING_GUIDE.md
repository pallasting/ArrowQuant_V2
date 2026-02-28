# Availability Monitoring Guide

## Overview

The LLM Compression System includes comprehensive availability monitoring to achieve and maintain >99.9% system availability. This guide explains the monitoring architecture, components, and how to use them.

## Architecture

### Components

1. **AvailabilityMonitor**: Tracks system uptime, component health, and availability metrics
2. **CircuitBreaker**: Prevents cascading failures through fault isolation
3. **HealthEndpoints**: Provides HTTP endpoints for health checks and monitoring
4. **Component Health Checks**: Validates individual component functionality

### Availability Target

**Target**: >99.9% availability (less than 8.76 hours downtime per year)

## AvailabilityMonitor

### Features

- Real-time component health tracking
- Incident recording and resolution
- Availability metrics calculation (uptime, downtime, MTBF, MTTR)
- Historical data persistence
- Automatic health check scheduling

### Usage

```python
from llm_compression.availability_monitor import AvailabilityMonitor
from pathlib import Path

# Initialize monitor
monitor = AvailabilityMonitor(
    check_interval=60.0,  # Check every 60 seconds
    history_file=Path("availability_history.jsonl")
)

# Start monitoring
await monitor.start()

# Perform health check
async def check_component():
    # Your health check logic
    return True

health_check = await monitor.check_health("component_name", check_component)

# Get availability metrics
metrics = monitor.get_availability_metrics()
print(f"Availability: {metrics.availability_percentage:.2f}%")
print(f"MTBF: {metrics.mtbf:.1f}s")
print(f"MTTR: {metrics.mttr:.1f}s")

# Stop monitoring
await monitor.stop()
```

### Health Check Results

Health checks return one of four statuses:

- **HEALTHY**: Component functioning normally (latency < 1s)
- **DEGRADED**: Component slow but functional (latency > 1s)
- **UNHEALTHY**: Component failed or timed out
- **UNKNOWN**: No health check data available

### Availability Metrics

```python
@dataclass
class AvailabilityMetrics:
    uptime_seconds: float          # Total uptime
    downtime_seconds: float        # Total downtime
    availability_percentage: float # Availability %
    total_checks: int              # Total health checks
    successful_checks: int         # Successful checks
    failed_checks: int             # Failed checks
    mtbf: float                    # Mean Time Between Failures
    mttr: float                    # Mean Time To Recovery
    last_incident: Optional[datetime]
```

## Circuit Breaker

### Purpose

Circuit breakers prevent cascading failures by temporarily blocking requests to failing components, allowing them time to recover.

### States

1. **CLOSED**: Normal operation, requests pass through
2. **OPEN**: Component failing, requests blocked
3. **HALF_OPEN**: Testing recovery, limited requests allowed

### Configuration

```python
from llm_compression.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig
)

config = CircuitBreakerConfig(
    failure_threshold=5,    # Open after 5 failures
    success_threshold=2,    # Close after 2 successes in half-open
    timeout=60.0,          # Try half-open after 60s
    reset_timeout=300.0    # Reset failure count after 300s
)

breaker = CircuitBreaker("storage", config)
```

### Usage

```python
# Wrap operations with circuit breaker
async def risky_operation():
    # Your operation
    return result

try:
    result = await breaker.call(risky_operation)
except CircuitBreakerError:
    # Circuit is open, use fallback
    result = fallback_operation()
```

### Circuit Breaker Registry

Manage multiple circuit breakers:

```python
from llm_compression.circuit_breaker import CircuitBreakerRegistry

registry = CircuitBreakerRegistry()

# Get or create breakers
storage_breaker = registry.get_or_create("storage")
embedder_breaker = registry.get_or_create("embedder")

# Get all metrics
all_metrics = registry.get_all_metrics()

# Reset all breakers
registry.reset_all()
```

## Health Endpoints

### HTTP Endpoints

The system provides standard health check endpoints compatible with Kubernetes and other orchestration systems.

#### 1. Health Check (`/health`)

Basic health status and metrics.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-26T10:00:00",
  "uptime_seconds": 3600.0,
  "availability_percentage": 99.95,
  "components": {
    "storage": "healthy",
    "embedder": "healthy",
    "vector_search": "healthy"
  },
  "unhealthy_components": [],
  "degraded_components": [],
  "active_incidents": 0
}
```

#### 2. Readiness Probe (`/readiness`)

Indicates if service is ready to accept traffic.

**Response**:
```json
{
  "ready": true,
  "timestamp": "2026-02-26T10:00:00",
  "critical_components": {
    "storage": "healthy",
    "embedder": "healthy",
    "vector_search": "healthy"
  },
  "unhealthy_critical": []
}
```

#### 3. Liveness Probe (`/liveness`)

Indicates if service is alive and should not be restarted.

**Response**:
```json
{
  "alive": true,
  "timestamp": "2026-02-26T10:00:00",
  "uptime_seconds": 3600.0,
  "prolonged_incidents": 0
}
```

#### 4. Metrics (`/metrics`)

Comprehensive system metrics.

**Response**:
```json
{
  "timestamp": "2026-02-26T10:00:00",
  "availability": {
    "24h": {
      "percentage": 99.95,
      "uptime_seconds": 86340.0,
      "downtime_seconds": 60.0,
      "mtbf": 43200.0,
      "mttr": 30.0
    },
    "7d": { ... },
    "all_time": { ... }
  },
  "health_checks": {
    "total": 1440,
    "successful": 1438,
    "failed": 2,
    "success_rate": 99.86
  },
  "components": { ... },
  "circuit_breakers": [ ... ],
  "incidents": {
    "active": 0,
    "total_24h": 2,
    "total_7d": 5
  }
}
```

#### 5. Incidents (`/incidents`)

Incident history and active incidents.

**Response**:
```json
{
  "timestamp": "2026-02-26T10:00:00",
  "active_incidents": [],
  "recent_incidents_24h": [
    {
      "component": "embedder",
      "start_time": "2026-02-26T08:00:00",
      "end_time": "2026-02-26T08:00:30",
      "duration_seconds": 30.0,
      "error": "Timeout",
      "resolved": true
    }
  ]
}
```

### Usage

```python
from llm_compression.health_endpoints import HealthEndpoints

endpoints = HealthEndpoints(
    availability_monitor,
    circuit_breaker_registry
)

# Get health status
health = await endpoints.health()

# Check readiness
readiness = await endpoints.readiness()

# Check liveness
liveness = await endpoints.liveness()

# Get metrics
metrics = await endpoints.metrics()

# Get incidents
incidents = await endpoints.incidents()
```

## Kubernetes Integration

### Deployment Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-compression
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: llm-compression
        image: llm-compression:latest
        ports:
        - containerPort: 8000
        
        # Liveness probe
        livenessProbe:
          httpGet:
            path: /liveness
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        # Readiness probe
        readinessProbe:
          httpGet:
            path: /readiness
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Monitoring Integration

### Prometheus Metrics

Export availability metrics to Prometheus:

```python
from prometheus_client import Gauge, Counter

# Define metrics
availability_gauge = Gauge(
    'system_availability_percentage',
    'System availability percentage'
)

health_checks_total = Counter(
    'health_checks_total',
    'Total health checks',
    ['component', 'status']
)

# Update metrics
metrics = monitor.get_availability_metrics()
availability_gauge.set(metrics.availability_percentage)

for check in monitor.get_recent_checks():
    health_checks_total.labels(
        component=check.component,
        status=check.status.value
    ).inc()
```

### Grafana Dashboard

Create a dashboard to visualize:

- Availability percentage over time
- Component health status
- Incident frequency and duration
- MTBF and MTTR trends
- Circuit breaker states

## Best Practices

### 1. Health Check Design

- Keep checks lightweight (< 100ms)
- Test actual functionality, not just process existence
- Include dependency checks (database, external APIs)
- Return detailed error information

### 2. Circuit Breaker Configuration

- Set thresholds based on component characteristics
- Use shorter timeouts for fast-failing components
- Configure longer recovery periods for slow components
- Monitor circuit breaker metrics

### 3. Incident Management

- Investigate all incidents, even brief ones
- Track incident patterns to identify systemic issues
- Implement automated remediation where possible
- Document incident response procedures

### 4. Availability Targets

- Monitor availability continuously
- Set alerts for availability drops below 99.9%
- Track MTBF and MTTR trends
- Conduct regular availability reviews

## Troubleshooting

### Low Availability

If availability drops below 99.9%:

1. Check active incidents: `monitor.get_active_incidents()`
2. Review recent failures: `monitor.get_recent_checks(limit=100)`
3. Examine circuit breaker states: `registry.get_all_metrics()`
4. Analyze MTTR: Are incidents taking too long to resolve?
5. Check component health: Are specific components failing repeatedly?

### Circuit Breaker Stuck Open

If a circuit breaker remains open:

1. Check failure count: `breaker.failure_count`
2. Verify timeout configuration: `breaker.config.timeout`
3. Test component manually to confirm recovery
4. Reset breaker if necessary: `breaker.reset()`

### Missing Health Checks

If health checks aren't running:

1. Verify monitor is started: `monitor._running`
2. Check for exceptions in monitor loop
3. Verify check interval configuration
4. Review system logs for errors

## Example Integration

See `examples/availability_monitoring_integration.py` for a complete integration example.

## Performance Impact

The availability monitoring system is designed for minimal performance impact:

- Health checks: < 1ms overhead per check
- Circuit breakers: < 0.1ms overhead per call
- Metrics calculation: < 10ms for full metrics
- History persistence: Async, non-blocking

## Conclusion

The availability monitoring system provides comprehensive tools to achieve and maintain >99.9% availability. By combining health checks, circuit breakers, and detailed metrics, you can ensure reliable operation of the LLM compression system in production environments.

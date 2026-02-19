# LLM Compression System - Deployment Guide

This guide covers the deployment and health check features of the LLM Compression System.

## Quick Start

### Automated Deployment

The easiest way to deploy the system is using the automated deployment script:

```bash
./deploy.sh
```

This script will:
1. ✅ Check Python version (requires 3.10+)
2. ✅ Verify system requirements (memory, disk space)
3. ✅ Create virtual environment
4. ✅ Install all dependencies
5. ✅ Create default configuration
6. ✅ Set up storage directories
7. ✅ Run health check

### With Tests

To run tests during deployment:

```bash
./deploy.sh --with-tests
```

## Manual Deployment

If you prefer manual deployment:

### 1. Check Requirements

- Python 3.10 or higher
- 8GB+ RAM (recommended)
- 20GB+ disk space (recommended)
- NVIDIA GPU with CUDA 11.8+ (optional, for local models)

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 4. Configure System

Create or edit `config.yaml`:

```yaml
llm:
  cloud_endpoint: "http://localhost:8045"
  cloud_api_key: null
  timeout: 30.0
  max_retries: 3
  rate_limit: 60

storage:
  storage_path: "~/.ai-os/memory/"
  compression_level: 3
  use_float16: true

# ... see config.yaml for full configuration
```

### 5. Create Storage Directories

```bash
mkdir -p ~/.ai-os/memory/{core,working,long-term,shared}
```

## Health Check System

### Overview

The health check system monitors all critical components:

- **LLM Client**: Connection to LLM API, response latency
- **Storage**: Disk accessibility, available space
- **GPU**: CUDA availability, memory usage (if applicable)
- **Config**: Configuration validity

### Health Check API

Start the FastAPI health check server:

```bash
python3 -m llm_compression.api
```

The API will be available at `http://localhost:8000`

### Endpoints

#### GET /health

Main health check endpoint. Returns comprehensive system status.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": 1234567890.123,
  "components": {
    "llm_client": {
      "status": "healthy",
      "message": "LLM client operational",
      "latency_ms": 123.45,
      "details": {
        "model": "gpt-4",
        "endpoint": "http://localhost:8045"
      }
    },
    "storage": {
      "status": "healthy",
      "message": "Storage accessible",
      "details": {
        "path": "/home/user/.ai-os/memory",
        "free_gb": 50.5,
        "total_gb": 100.0
      }
    },
    "gpu": {
      "status": "healthy",
      "message": "GPU available",
      "details": {
        "device_count": 1,
        "device_name": "NVIDIA GeForce RTX 3090",
        "cuda_version": "11.8",
        "memory_total_gb": 24.0,
        "memory_usage_pct": 15.5
      }
    },
    "config": {
      "status": "healthy",
      "message": "Configuration valid",
      "details": {
        "llm_endpoint": "http://localhost:8045",
        "storage_path": "/home/user/.ai-os/memory",
        "batch_size": 16
      }
    }
  }
}
```

**Status Codes:**
- `200`: System is healthy or degraded (but operational)
- `503`: System is unhealthy (not operational)

#### GET /health/live

Liveness probe for Kubernetes. Simple check if the application is running.

**Response:**
```json
{
  "status": "alive"
}
```

#### GET /health/ready

Readiness probe for Kubernetes. Checks if the system is ready to accept traffic.

**Response:**
```json
{
  "status": "ready"
}
```

**Status Codes:**
- `200`: Ready to accept traffic
- `503`: Not ready

### Programmatic Health Check

Use the health checker in your Python code:

```python
import asyncio
from llm_compression.health import HealthChecker
from llm_compression.config import Config

async def check_system():
    config = Config.from_yaml("config.yaml")
    checker = HealthChecker(config=config)
    
    result = await checker.check_health()
    
    print(f"Status: {result.overall_status}")
    for name, component in result.components.items():
        print(f"  {name}: {component.status} - {component.message}")

asyncio.run(check_system())
```

See `examples/health_check_example.py` for a complete example.

## Health Status Levels

### Healthy ✅

All components are functioning normally. System is ready for production use.

### Degraded ⚠️

System is operational but has some issues:
- High LLM latency (> 5s)
- Low disk space (< 1GB)
- High GPU memory usage (> 90%)
- Configuration warnings

The system can still be used, but performance may be affected.

### Unhealthy ❌

Critical issues detected:
- LLM client connection failed
- Storage not accessible
- Invalid configuration

The system should not be used until issues are resolved.

## Monitoring Integration

### Kubernetes

Use the liveness and readiness probes:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: llm-compression
spec:
  containers:
  - name: api
    image: llm-compression:latest
    ports:
    - containerPort: 8000
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 30
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8000
      initialDelaySeconds: 5
      periodSeconds: 10
```

### Prometheus

Enable Prometheus metrics in `config.yaml`:

```yaml
monitoring:
  enable_prometheus: true
  prometheus_port: 9090
```

Metrics will be available at `http://localhost:9090/metrics`

### Docker

Build and run with Docker:

```bash
# Build image
docker build -t llm-compression:latest .

# Run container
docker run -p 8000:8000 -v ~/.ai-os:/root/.ai-os llm-compression:latest
```

## Troubleshooting

### LLM Client Unhealthy

**Symptoms:**
- `llm_client` status is `unhealthy`
- Message: "Connection failed"

**Solutions:**
1. Check if LLM API is running on port 8045
2. Verify `cloud_endpoint` in config.yaml
3. Check network connectivity
4. Verify API key (if required)

### Storage Unhealthy

**Symptoms:**
- `storage` status is `unhealthy`
- Message: "No write permission" or "Storage path does not exist"

**Solutions:**
1. Create storage directory: `mkdir -p ~/.ai-os/memory`
2. Check directory permissions: `chmod 755 ~/.ai-os/memory`
3. Verify `storage_path` in config.yaml

### GPU Degraded

**Symptoms:**
- `gpu` status is `degraded`
- Message: "High GPU memory usage"

**Solutions:**
1. Close other GPU-intensive applications
2. Reduce batch size in config.yaml
3. Use CPU mode (set `prefer_local: false`)

### Config Degraded

**Symptoms:**
- `config` status is `degraded`
- Message: "Configuration issues"

**Solutions:**
1. Review config.yaml for invalid values
2. Check temperature is between 0.0 and 1.0
3. Ensure batch_size >= 1
4. Verify storage paths exist

## Production Deployment

### Recommended Configuration

For production use:

```yaml
llm:
  cloud_endpoint: "http://your-llm-api:8045"
  timeout: 60.0
  max_retries: 5
  rate_limit: 100

performance:
  batch_size: 32
  max_concurrent: 8
  cache_size: 50000

monitoring:
  enable_prometheus: true
  prometheus_port: 9090
  alert_quality_threshold: 0.90
```

### Security Considerations

1. **API Keys**: Store in environment variables, not in config.yaml
2. **Network**: Use HTTPS for LLM API connections
3. **Storage**: Encrypt sensitive data at rest
4. **Access Control**: Restrict health check endpoint access

### High Availability

1. **Load Balancing**: Run multiple instances behind a load balancer
2. **Health Checks**: Configure load balancer to use `/health/ready`
3. **Failover**: Implement automatic failover to backup LLM endpoints
4. **Monitoring**: Set up alerts for degraded/unhealthy status

## Next Steps

1. Review and customize `config.yaml`
2. Start the health check API: `python3 -m llm_compression.api`
3. Test the health endpoint: `curl http://localhost:8000/health`
4. Integrate with your monitoring system
5. Deploy to production

For more information, see:
- [README.md](README.md) - System overview
- [examples/health_check_example.py](examples/health_check_example.py) - Usage examples
- [API Documentation](http://localhost:8000/docs) - Interactive API docs

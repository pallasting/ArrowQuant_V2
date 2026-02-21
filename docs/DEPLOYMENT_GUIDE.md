# LLM Compression System - Production Deployment Guide

**Version**: 1.0  
**Last Updated**: 2026-02-21  
**Status**: Production Ready

---

## Overview

This guide covers deploying the LLM Compression System to production using Docker and Kubernetes.

## Prerequisites

- Docker 20.10+
- Kubernetes 1.24+
- kubectl configured
- Helm 3.0+ (optional)
- 4GB+ RAM per pod
- 10GB+ disk space

---

## Quick Start (Docker Compose)

### 1. Build and Run

```bash
# Build Docker image
docker build -t llm-compression:latest .

# Start services (app + monitoring)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f llm-compression
```

### 2. Access Services

- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 3. Stop Services

```bash
docker-compose down
```

---

## Kubernetes Deployment

### 1. Prepare Cluster

```bash
# Create namespace
kubectl create namespace llm-compression

# Set context
kubectl config set-context --current --namespace=llm-compression
```

### 2. Deploy Application

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Check deployment status
kubectl get pods
kubectl get svc
kubectl get ingress
```

### 3. Deploy Monitoring

```bash
# Create monitoring namespace
kubectl create namespace monitoring

# Deploy Prometheus
kubectl apply -f k8s/monitoring/prometheus-config.yaml -n monitoring

# Deploy Grafana
kubectl apply -f k8s/monitoring/grafana-dashboard.json -n monitoring
```

### 4. Verify Deployment

```bash
# Check pod status
kubectl get pods -l app=llm-compression

# Check logs
kubectl logs -f deployment/llm-compression

# Test health endpoint
kubectl port-forward svc/llm-compression 8000:80
curl http://localhost:8000/health
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model files | `/app/models/minilm` |
| `DEVICE` | Compute device (cpu/cuda) | `cpu` |
| `PORT` | API server port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Resource Limits

**Recommended**:
- Memory: 2-4GB per pod
- CPU: 1-2 cores per pod
- Replicas: 3-10 (auto-scaling)

**Minimum**:
- Memory: 1GB
- CPU: 0.5 cores
- Replicas: 2

---

## Monitoring

### Prometheus Metrics

Available at `/metrics` endpoint:

- `llm_compression_requests_total` - Total requests
- `llm_compression_request_duration_seconds` - Request latency
- `llm_compression_request_errors_total` - Error count
- `llm_compression_ratio` - Compression ratio
- `llm_compression_uptime_seconds` - Service uptime

### Grafana Dashboards

Import dashboard from `k8s/monitoring/grafana-dashboard.json`:

1. Open Grafana (http://localhost:3000)
2. Go to Dashboards → Import
3. Upload `grafana-dashboard.json`
4. Select Prometheus data source

### Alerts

Configured alerts (see `k8s/monitoring/prometheus-config.yaml`):

- **HighErrorRate**: Error rate > 5% for 5 minutes
- **HighLatency**: Latency > 5s for 5 minutes
- **ServiceDown**: Service unavailable for 2 minutes
- **HighMemoryUsage**: Memory > 90% for 5 minutes

---

## CI/CD Pipeline

### GitHub Actions Workflow

Automated pipeline (`.github/workflows/ci-cd.yml`):

1. **Test**: Run unit and integration tests
2. **Lint**: Code quality checks (black, flake8, mypy)
3. **Build**: Build and push Docker image
4. **Deploy Staging**: Auto-deploy to staging (develop branch)
5. **Deploy Production**: Manual deploy to production (releases)

### Deployment Triggers

- **Push to `develop`**: Deploy to staging
- **Push to `main/master`**: Build only
- **Release published**: Deploy to production

### Required Secrets

Configure in GitHub repository settings:

- `KUBECONFIG_STAGING`: Base64-encoded kubeconfig for staging
- `KUBECONFIG_PRODUCTION`: Base64-encoded kubeconfig for production

---

## Health Checks

### Liveness Probe

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-21T14:00:00Z",
  "checks": {
    "api_responsive": true,
    "memory_ok": true,
    "disk_ok": true
  },
  "uptime_seconds": 3600.0
}
```

### Readiness Probe

```bash
curl http://localhost:8000/ready
```

Returns `200 OK` when service is ready to accept traffic.

---

## Scaling

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment llm-compression --replicas=5

# Check status
kubectl get pods -l app=llm-compression
```

### Auto-Scaling (HPA)

Configured in `k8s/hpa.yaml`:

- **Min replicas**: 3
- **Max replicas**: 10
- **CPU target**: 70%
- **Memory target**: 80%

```bash
# Check HPA status
kubectl get hpa llm-compression-hpa

# View scaling events
kubectl describe hpa llm-compression-hpa
```

---

## Troubleshooting

### Pod Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>

# Check resource limits
kubectl top pod <pod-name>
```

### High Memory Usage

```bash
# Check memory usage
kubectl top pods -l app=llm-compression

# Increase memory limits in deployment.yaml
# Then apply changes
kubectl apply -f k8s/deployment.yaml
```

### Service Unavailable

```bash
# Check service endpoints
kubectl get endpoints llm-compression

# Check ingress
kubectl describe ingress llm-compression

# Test service directly
kubectl port-forward svc/llm-compression 8000:80
curl http://localhost:8000/health
```

---

## Backup and Recovery

### Model Files

```bash
# Backup model files
kubectl exec -it <pod-name> -- tar czf /tmp/models.tar.gz /app/models
kubectl cp <pod-name>:/tmp/models.tar.gz ./models-backup.tar.gz

# Restore model files
kubectl cp ./models-backup.tar.gz <pod-name>:/tmp/models.tar.gz
kubectl exec -it <pod-name> -- tar xzf /tmp/models.tar.gz -C /
```

### Configuration

```bash
# Backup ConfigMaps
kubectl get configmap -o yaml > configmaps-backup.yaml

# Restore ConfigMaps
kubectl apply -f configmaps-backup.yaml
```

---

## Security

### Network Policies

```bash
# Apply network policies (if available)
kubectl apply -f k8s/network-policy.yaml
```

### Secrets Management

```bash
# Create secret for API keys
kubectl create secret generic llm-compression-secrets \
  --from-literal=api-key=<your-api-key>

# Use in deployment
# env:
#   - name: API_KEY
#     valueFrom:
#       secretKeyRef:
#         name: llm-compression-secrets
#         key: api-key
```

---

## Performance Tuning

### CPU Optimization

- Use CPU-optimized Docker image
- Enable multi-threading
- Adjust worker count based on CPU cores

### Memory Optimization

- Use memory-mapped model loading
- Enable model caching
- Adjust batch sizes

### Network Optimization

- Enable HTTP/2
- Use connection pooling
- Configure appropriate timeouts

---

## Maintenance

### Rolling Updates

```bash
# Update image
kubectl set image deployment/llm-compression \
  llm-compression=llm-compression:v2.0.0

# Monitor rollout
kubectl rollout status deployment/llm-compression

# Rollback if needed
kubectl rollout undo deployment/llm-compression
```

### Log Rotation

Logs are automatically rotated by Kubernetes. Configure retention:

```yaml
# In deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: llm-compression
        env:
        - name: LOG_MAX_SIZE
          value: "100MB"
        - name: LOG_MAX_AGE
          value: "7d"
```

---

## Support

For issues and questions:

- GitHub Issues: https://github.com/your-org/llm-compression/issues
- Documentation: https://docs.example.com
- Email: support@example.com

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-21  
**Maintained By**: DevOps Team

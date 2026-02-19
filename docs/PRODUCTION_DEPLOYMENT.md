# ArrowEngine 生产环境部署指南 (v1.0)

## 1. 简介

本指南旨在帮助运维团队将 ArrowEngine 嵌入服务部署到生产环境。系统已经过全面优化，支持高并发、低延迟的文本嵌入任务。

## 2. 系统架构

- **API 层**: FastAPI (异步处理，Request ID 追踪)
- **安全层**: API Key 鉴权 + 令牌桶限流 (50 RPS/IP)
- **推理层**: ArrowEngine (PyArrow 零拷贝 + Rust Tokenizer)
- **监控层**: Prometheus Metrics (/metrics 端点)
- **部署层**: Docker 容器化 + uvicorn

## 3. 快速部署 (Docker)

### 3.1 前置要求
- Docker Engine >= 20.10
- Docker Compose >= 2.0
- 至少 2GB 可用内存

### 3.2 启动服务

```bash
# 1. 克隆代码库
git clone https://github.com/ai-os/llm-compression.git
cd llm-compression

# 2. 设置环境变量 (可选，推荐生产环境开启鉴权)
export ARROW_API_KEY="your-secret-key-here"

# 3. 启动服务
docker-compose up -d
```

### 3.3 验证状态

```bash
# 检查容器状态
docker-compose ps

# 检查服务健康 (应返回 200 OK)
curl http://localhost:8000/health
```

## 4. 监控与运维

### 4.1 Prometheus 监控
服务在 `http://localhost:8000/metrics` 暴露标准 Prometheus 指标。

**关键指标说明：**
| 指标名称 | 类型 | 说明 | 告警阈值建议 |
| :--- | :--- | :--- | :--- |
| `arrowengine_request_total` | Counter | 请求总数 (标签: status, endpoint) | 5xx 占比 > 1% |
| `arrowengine_inference_latency_seconds` | Histogram | 推理延迟分布 | P99 > 50ms |
| `arrowengine_model_loaded` | Gauge | 模型加载状态 (1=成功, 0=失败) | 值 == 0 |

### 4.2 日志审计
日志输出为标准 JSON 格式，包含 `request_id` 全链路追踪。

**示例日志：**
```json
{
  "timestamp": "2023-10-27 10:00:01,123",
  "level": "INFO",
  "message": "Access Log",
  "request_id": "a1b2c3d4-...",
  "path": "/embed",
  "status": "200",
  "duration_ms": 8.5
}
```

## 5. 性能与容量规划

### 5.1 资源基准
| 规格 | RPS 上限 (est.) | P99 延迟 | 推荐用途 |
| :--- | :--- | :--- | :--- |
| 2 CPU / 2GB RAM | 500+ | < 20ms | 标准生产节点 |
| 4 CPU / 4GB RAM | 1000+ | < 15ms | 高负载节点 |

### 5.2 压力测试
使用内置 Locust 脚本进行验证：
```bash
pip install locust
# 模拟 100 用户并发
locust -f tests/load/locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10
```

## 6. 安全配置

### 6.1 启用鉴权
设置环境变量 `ARROW_API_KEY` 即可开启鉴权。开启后，所有请求必须包含请求头：
`X-API-Key: <your-key>`

### 6.2 速率限制
默认限流策略为 **50 请求/秒 (突发 100)**。
如需调整，请修改 `llm_compression/server/security.py` 中的 `RATE_LIMIT_RPS` 常量（后续版本将支持环境变量配置）。

## 7. 故障排查

- **服务起不来**：检查 `docker-compose logs`，确认模型路径 `MODEL_PATH` 是否正确挂载。
- **性能下降**：检查 `/metrics` 中的延迟指标，确认是否触发 CPU 限制 (Throttling)。
- **429 错误**：客户端请求频率超过限流阈值，建议客户端增加退避重试 (Backoff)。

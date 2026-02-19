"""
ArrowEngine 服务的 Prometheus 监控模块。

定义并暴露以下指标：
- arrowengine_request_total (Counter): 总请求数 (按方法、路径和状态过滤)
- arrowengine_inference_latency_seconds (Histogram): 推理延迟
- arrowengine_tokens_processed_total (Counter): 已处理的总词元数（吞吐量）
- arrowengine_cache_hits_total (Counter): 服务器端缓存命中（如果后续实现）
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
from typing import Callable, Any

# 创建独立的注册表
REGISTRY = CollectorRegistry()

# 1. 请求总数
REQUEST_COUNT = Counter(
    "arrowengine_request_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
    registry=REGISTRY
)

# 2. 推理延迟
INFERENCE_LATENCY = Histogram(
    "arrowengine_inference_latency_seconds",
    "Time spent performing inference",
    ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY
)

# 3. 词元吞吐量
TOKENS_PROCESSED = Counter(
    "arrowengine_tokens_processed_total",
    "Total number of tokens processed by the engine",
    ["endpoint"],
    registry=REGISTRY
)

# 4. 模型加载状态 (Gauge)
MODEL_LOAD_STATUS = Gauge(
    "arrowengine_model_loaded",
    "Model load status (1 for loaded, 0 for not)",
    registry=REGISTRY
)

def track_latency(endpoint: str):
    """用于统计推理延迟的装饰器"""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                INFERENCE_LATENCY.labels(endpoint=endpoint).observe(duration)
        return wrapper
    return decorator

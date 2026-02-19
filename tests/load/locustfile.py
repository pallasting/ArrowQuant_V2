"""
ArrowEngine 负载测试脚本 (Locust).

目标：验证系统在 1000+ RPS 下的稳定性。
场景：
1. 混合负载：80% 单句嵌入，20% 批量嵌入 (batch_size=8)。
2. 健康检查：独立用户模拟监控系统每 10s 探测一次。

使用方法：
    pip install locust
    locust -f tests/load/locustfile.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between, events
import random
import os

# 从环境变量获取 API Key（如果已启用鉴权）
API_KEY = os.getenv("ARROW_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

class ArrowEngineUser(HttpUser):
    # 思考时间 0.1~0.5秒，模拟高频调用
    wait_time = between(0.1, 0.5)
    
    @task(4)
    def embed_single(self):
        """模拟单句嵌入请求 (权重 4)"""
        text = f"This is a test sentence for load testing {random.randint(1, 10000)}"
        self.client.post(
            "/embed",
            json={"texts": [text]},
            headers=HEADERS,
            name="/embed (single)"
        )

    @task(1)
    def embed_batch(self):
        """模拟批量嵌入请求 (权重 1)"""
        batch_size = 8
        texts = [f"Batch text {i} {random.randint(1, 10000)}" for i in range(batch_size)]
        self.client.post(
            "/embed",
            json={"texts": texts},
            headers=HEADERS,
            name="/embed (batch)"
        )

class HealthCheckUser(HttpUser):
    """模拟监控系统，固定频率检查健康状态"""
    wait_time = between(10, 10)
    
    @task
    def health_check(self):
        self.client.get("/health", name="/health")

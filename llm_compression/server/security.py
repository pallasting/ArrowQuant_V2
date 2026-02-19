"""
ArrowEngine 安全模块。

提供 API Key 鉴权和基于内存的速率限制功能。
"""

import os
import time
from typing import Dict, Tuple
from fastapi import HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN, HTTP_429_TOO_MANY_REQUESTS

# 定义 API Key 来源
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 简单的内存限流器存储 (IP -> (tokens, last_refill_time))
# 注意：在多进程 (Gunicorn) 模式下，这是进程本地的。
# 生产环境建议使用 Redis，但为了保持无依赖，这里使用内存实现。
_rate_limit_store: Dict[str, Tuple[float, float]] = {}

# 限流配置：每秒允许的请求数 (RPS) 和 突发容量 (Burst)
RATE_LIMIT_RPS = 50.0
RATE_LIMIT_BURST = 100.0

async def get_api_key(
    api_key_header: str = Security(api_key_header),
):
    """
    验证 API Key。
    
    如果环境变量 ARROW_API_KEY 未设置，则默认禁用鉴权（开发模式）。
    如果设置了，则必须匹配。
    """
    expected_key = os.getenv("ARROW_API_KEY")
    
    # 如果未配置 Key，则视为开发模式，允许所有请求
    if not expected_key:
        return None
        
    if api_key_header == expected_key:
        return api_key_header
        
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Could not validate credentials"
    )

async def check_rate_limit(request: Request):
    """
    检查速率限制 (Token Bucket 算法)。
    基于客户端 IP 进行限流。
    """
    # 如果未启用限流（RPS <= 0），直接通过
    if RATE_LIMIT_RPS <= 0:
        return
        
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    # 获取当前状态 (tokens, last_refill)
    tokens, last_refill = _rate_limit_store.get(client_ip, (RATE_LIMIT_BURST, now))
    
    # 计算需要补充的令牌数
    elapsed = now - last_refill
    refill = elapsed * RATE_LIMIT_RPS
    tokens = min(RATE_LIMIT_BURST, tokens + refill)
    
    # 检查是否有足够的令牌
    if tokens < 1.0:
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    # 消耗一个令牌并更新状态
    _rate_limit_store[client_ip] = (tokens - 1.0, now)
    
    # 定期清理过期的 IP (可选，防止内存泄漏)
    if len(_rate_limit_store) > 10000:
        _cleanup_rate_limit_store(now)

def _cleanup_rate_limit_store(now: float):
    """清理长时间未活跃的 IP"""
    keys_to_remove = []
    for ip, (_, last_refill) in _rate_limit_store.items():
        if now - last_refill > 60: # 1分钟无活动
            keys_to_remove.append(ip)
    for key in keys_to_remove:
        del _rate_limit_store[key]

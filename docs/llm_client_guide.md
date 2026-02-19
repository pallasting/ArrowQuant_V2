# LLM 客户端使用指南

## 概述

LLM 客户端提供统一的接口来访问云端 API 和本地模型，支持连接池、重试机制、速率限制和指标记录。

## 特性

- ✅ **OpenAI 兼容 API**: 支持标准的 OpenAI API 格式
- ✅ **连接池管理**: 高效管理 HTTP 连接，避免连接泄漏
- ✅ **自动重试**: 使用指数退避策略自动重试失败的请求
- ✅ **速率限制**: 防止超过 API 限流限制
- ✅ **指标跟踪**: 记录延迟、token 使用量、成功率等指标
- ✅ **批量处理**: 支持并发批量请求
- ✅ **异步支持**: 完全异步实现，高性能

## 快速开始

### 基础使用

```python
import asyncio
from llm_compression import LLMClient

async def main():
    # 创建客户端
    client = LLMClient(
        endpoint="http://localhost:8045",
        timeout=30.0,
        max_retries=3,
        rate_limit=60  # 请求/分钟
    )
    
    try:
        # 生成文本
        response = await client.generate(
            prompt="Summarize: The quick brown fox jumps over the lazy dog.",
            max_tokens=50,
            temperature=0.3
        )
        
        print(f"响应: {response.text}")
        print(f"Tokens: {response.tokens_used}")
        print(f"延迟: {response.latency_ms}ms")
        
    finally:
        await client.close()

asyncio.run(main())
```

### 批量请求

```python
async def batch_example():
    client = LLMClient(endpoint="http://localhost:8045")
    
    try:
        prompts = [
            "Summarize: Text 1",
            "Summarize: Text 2",
            "Summarize: Text 3"
        ]
        
        responses = await client.batch_generate(
            prompts=prompts,
            max_tokens=50,
            temperature=0.3
        )
        
        for i, response in enumerate(responses):
            print(f"响应 {i+1}: {response.text}")
            
    finally:
        await client.close()
```

### 使用配置文件

```python
from llm_compression import LLMClient, Config

async def config_example():
    # 加载配置
    config = Config.from_yaml("config.yaml")
    
    # 创建客户端
    client = LLMClient(
        endpoint=config.llm.cloud_endpoint,
        api_key=config.llm.cloud_api_key,
        timeout=config.llm.timeout,
        max_retries=config.llm.max_retries,
        rate_limit=config.llm.rate_limit
    )
    
    try:
        response = await client.generate(prompt="Hello!")
        print(response.text)
    finally:
        await client.close()
```

## 配置选项

### LLMClient 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `endpoint` | str | 必需 | API 端点 URL |
| `api_key` | str | None | API 密钥（可选） |
| `timeout` | float | 30.0 | 请求超时时间（秒） |
| `max_retries` | int | 3 | 最大重试次数 |
| `rate_limit` | int | 60 | 速率限制（请求/分钟） |
| `pool_size` | int | 10 | 连接池大小 |
| `max_concurrent` | int | 10 | 最大并发请求数（批量处理） |
| `eager_init` | bool | True | 是否立即初始化连接池 |

### generate 方法参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | str | 必需 | 输入提示 |
| `max_tokens` | int | 100 | 最大生成 token 数 |
| `temperature` | float | 0.3 | 采样温度（0-1） |
| `stop_sequences` | List[str] | None | 停止序列 |

## 错误处理

```python
from llm_compression.llm_client import LLMAPIError, LLMTimeoutError

async def error_handling():
    client = LLMClient(endpoint="http://localhost:8045")
    
    try:
        response = await client.generate(prompt="Test")
        print(response.text)
        
    except LLMTimeoutError as e:
        print(f"请求超时: {e}")
        
    except LLMAPIError as e:
        print(f"API 错误: {e}")
        
    finally:
        await client.close()
```

## 指标跟踪

```python
async def metrics_example():
    client = LLMClient(endpoint="http://localhost:8045")
    
    try:
        # 执行一些请求
        for i in range(10):
            await client.generate(prompt=f"Request {i}")
        
        # 获取指标
        metrics = client.get_metrics()
        
        print(f"总请求数: {metrics['total_requests']}")
        print(f"成功率: {metrics['success_rate']:.2%}")
        print(f"平均延迟: {metrics['avg_latency_ms']:.2f}ms")
        print(f"平均 tokens: {metrics['avg_tokens_per_request']:.2f}")
        
    finally:
        await client.close()
```

## 高级功能

### 连接池管理

连接池自动管理 HTTP 连接，提高性能并避免连接泄漏：

```python
client = LLMClient(
    endpoint="http://localhost:8045",
    pool_size=20  # 增加连接池大小以支持更高并发
)

# 连接池会自动初始化和管理
# 无需手动操作
```

### 重试策略

客户端使用指数退避策略自动重试失败的请求：

```python
client = LLMClient(
    endpoint="http://localhost:8045",
    max_retries=5  # 最多重试 5 次
)

# 重试延迟：1s, 2s, 4s, 8s, 16s
# 自动处理临时网络错误和 API 限流
```

### 速率限制

防止超过 API 限流限制：

```python
client = LLMClient(
    endpoint="http://localhost:8045",
    rate_limit=100  # 每分钟最多 100 个请求
)

# 客户端会自动限制请求速率
# 超过限制时会等待
```

### 并发请求

使用 asyncio 实现高效并发（带并发控制）：

```python
async def concurrent_requests():
    client = LLMClient(
        endpoint="http://localhost:8045",
        pool_size=20,
        rate_limit=200,
        max_concurrent=15  # 限制最大并发数
    )
    
    try:
        # 创建并发任务
        tasks = [
            client.generate(prompt=f"Request {i}")
            for i in range(50)
        ]
        
        # 并发执行（自动控制并发数）
        responses = await asyncio.gather(*tasks)
        
        print(f"完成 {len(responses)} 个请求")
        
    finally:
        await client.close()
```

### 健康检查

```python
async def health_check_example():
    client = LLMClient(endpoint="http://localhost:8045")
    
    try:
        # 检查客户端健康状态
        health = await client.health_check()
        
        print(f"健康状态: {health['healthy']}")
        print(f"可用连接: {health['connection_pool_available']}/{health['connection_pool_size']}")
        print(f"成功率: {health['metrics']['success_rate']:.2%}")
        
    finally:
        await client.close()
```

## 性能优化建议

1. **调整连接池大小**: 根据并发需求调整 `pool_size`
2. **合理设置速率限制**: 避免触发 API 限流
3. **使用批量请求**: 对于多个独立请求，使用 `batch_generate`
4. **监控指标**: 定期检查 `get_metrics()` 以优化性能
5. **复用客户端**: 避免频繁创建和销毁客户端实例

## 最佳实践

### 使用上下文管理器（推荐）

```python
async def best_practice():
    # 推荐：使用 async with 自动管理资源
    async with LLMClient(endpoint="http://localhost:8045") as client:
        response = await client.generate(prompt="Test")
        return response
    # 客户端会自动关闭
```

### 传统方式

```python
async def traditional_way():
    client = LLMClient(endpoint="http://localhost:8045")
    
    try:
        # 使用客户端
        response = await client.generate(prompt="Test")
        return response
    finally:
        # 确保关闭
        await client.close()
```

### 2. 配置文件管理

使用配置文件管理不同环境的设置：

```yaml
# config.yaml
llm:
  cloud_endpoint: "http://localhost:8045"
  timeout: 30.0
  max_retries: 3
  rate_limit: 60
```

### 3. 错误处理

始终处理可能的异常：

```python
try:
    response = await client.generate(prompt="Test")
except LLMTimeoutError:
    # 处理超时
    pass
except LLMAPIError:
    # 处理 API 错误
    pass
```

### 4. 监控和日志

定期检查指标并记录日志：

```python
import logging

logger = logging.getLogger(__name__)

# 定期记录指标
metrics = client.get_metrics()
logger.info(f"LLM Client Metrics: {metrics}")
```

## 故障排查

### 问题：连接超时

**原因**: API 服务器响应慢或不可达

**解决方案**:
- 增加 `timeout` 值
- 检查网络连接
- 验证 API 端点是否正确

### 问题：速率限制错误

**原因**: 超过 API 限流限制

**解决方案**:
- 降低 `rate_limit` 值
- 增加请求间隔
- 使用批量请求减少请求次数

### 问题：连接池耗尽

**原因**: 并发请求过多

**解决方案**:
- 增加 `pool_size`
- 设置 `max_concurrent` 限制并发数
- 使用批量请求的并发控制
- 检查是否有连接泄漏

### 问题：首次请求延迟高

**原因**: 连接池延迟初始化

**解决方案**:
- 使用 `eager_init=True`（默认）立即初始化
- 或在首次请求前手动调用 `await client.connection_pool.initialize()`

### 问题：内存使用过高

**原因**: 指标记录过多

**解决方案**:
- 客户端自动限制最近 1000 个延迟记录
- 定期重置客户端
- 监控内存使用

## API 参考

### LLMClient

```python
class LLMClient:
    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        rate_limit: int = 60,
        pool_size: int = 10,
        max_concurrent: int = 10,
        eager_init: bool = True
    )
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.3,
        stop_sequences: Optional[List[str]] = None
    ) -> LLMResponse
    
    async def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.3
    ) -> List[LLMResponse]
    
    def get_metrics(self) -> Dict[str, Any]
    
    async def health_check(self) -> Dict[str, Any]
    
    async def close(self)
    
    async def __aenter__(self)
    
    async def __aexit__(self, exc_type, exc_val, exc_tb)
```

### LLMResponse

```python
@dataclass
class LLMResponse:
    text: str                    # 生成的文本
    tokens_used: int             # 使用的 token 数
    latency_ms: float            # 延迟（毫秒）
    model: str                   # 使用的模型
    finish_reason: str           # 完成原因
    metadata: Dict[str, Any]     # 额外元数据
```

## 相关文档

- [配置管理指南](./configuration.md)
- [错误处理指南](./error_handling.md)
- [性能优化指南](./performance.md)
- [API 参考](./api_reference.md)

## 支持

如有问题或建议，请联系 AI-OS 团队或提交 Issue。

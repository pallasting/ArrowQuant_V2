"""
LLM 客户端模块

提供统一的 LLM 访问接口，支持云端 API 和本地模型。
实现连接池、重试机制、速率限制和指标记录。
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import aiohttp

from llm_compression.logger import logger
from llm_compression.errors import LLMAPIError, LLMTimeoutError


@dataclass
class LLMResponse:
    """LLM 响应"""
    text: str                    # 生成的文本
    tokens_used: int             # 使用的 token 数
    latency_ms: float            # 延迟（毫秒）
    model: str                   # 使用的模型
    finish_reason: str           # 完成原因（stop/length/error）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


# Remove old error definitions - now imported from errors module


class RetryPolicy:
    """重试策略"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    async def execute_with_retry(self, func, *args, **kwargs) -> Any:
        """
        执行函数并在失败时重试
        
        使用指数退避策略：
        - 第 1 次重试：等待 1s
        - 第 2 次重试：等待 2s
        - 第 3 次重试：等待 4s
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except (LLMAPIError, LLMTimeoutError, aiohttp.ClientError) as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}, "
                        f"retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} retries failed")
        
        raise last_exception


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_times: List[float] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """
        获取请求许可
        
        使用滑动窗口算法：
        - 记录最近 1 分钟内的所有请求时间
        - 如果超过限制，等待直到可以发送
        """
        async with self.lock:
            now = time.time()
            
            # 清理 1 分钟前的记录
            self.request_times = [
                t for t in self.request_times
                if now - t < 60
            ]
            
            # 检查是否超过限制
            if len(self.request_times) >= self.requests_per_minute:
                # 计算需要等待的时间
                oldest = self.request_times[0]
                wait_time = 60 - (now - oldest)
                
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    now = time.time()
            
            # 记录本次请求
            self.request_times.append(now)


class LLMConnectionPool:
    """LLM 连接池"""
    
    def __init__(
        self,
        endpoint: str,
        pool_size: int = 10,
        timeout: float = 30.0
    ):
        self.endpoint = endpoint
        self.pool_size = pool_size
        self.timeout = timeout
        self.sessions: List[aiohttp.ClientSession] = []
        self.available: asyncio.Queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """初始化连接池"""
        if self._initialized:
            return
        
        async with self.lock:
            if self._initialized:
                return
            
            for _ in range(self.pool_size):
                session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                )
                self.sessions.append(session)
                await self.available.put(session)
            
            self._initialized = True
            logger.info(f"Connection pool initialized with {self.pool_size} connections")
    
    async def acquire(self) -> aiohttp.ClientSession:
        """获取连接"""
        if not self._initialized:
            await self.initialize()
        return await self.available.get()
    
    async def release(self, session: aiohttp.ClientSession):
        """释放连接"""
        await self.available.put(session)
    
    async def close(self):
        """关闭所有连接"""
        for session in self.sessions:
            await session.close()
        self._initialized = False
        logger.info("Connection pool closed")


class LLMClient:
    """统一的 LLM 客户端接口"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        rate_limit: int = 60,  # requests per minute
        pool_size: int = 10,
        max_concurrent: int = 10,  # 最大并发请求数
        eager_init: bool = True,  # 是否立即初始化连接池
        api_type: str = "auto"  # API 类型: "auto", "openai", "ollama"
    ):
        """
        初始化 LLM 客户端
        
        Args:
            endpoint: API 端点 (e.g., "http://localhost:8045")
            api_key: API 密钥（云端 API 需要）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            rate_limit: 速率限制（请求/分钟）
            pool_size: 连接池大小
            max_concurrent: 最大并发请求数（用于批量处理）
            eager_init: 是否立即初始化连接池（False 则延迟到首次使用）
            api_type: API 类型 ("auto", "openai", "ollama")
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self._closed = False

        # 兼容性处理：如果 endpoint 不是字符串（例如传入了 Config 对象）
        if not isinstance(endpoint, str) and endpoint is not None:
            if hasattr(endpoint, 'llm') and hasattr(endpoint.llm, 'cloud_endpoint'):
                endpoint = endpoint.llm.cloud_endpoint
            elif hasattr(endpoint, 'cloud_endpoint'):
                endpoint = endpoint.cloud_endpoint
            else:
                endpoint = str(endpoint)

        # 确保 endpoint 是字符串
        if endpoint is None:
            endpoint = "http://localhost:8045"

        self.endpoint = endpoint

        # 检测 API 类型
        if api_type == "auto":
            # 根据端点自动检测
            if "11434" in endpoint or "ollama" in endpoint.lower():
                self.api_type = "ollama"
            else:
                self.api_type = "openai"
        else:
            self.api_type = api_type
        
        # 初始化组件
        self.retry_policy = RetryPolicy(max_retries=max_retries)
        self.rate_limiter = RateLimiter(requests_per_minute=rate_limit)
        self.connection_pool = LLMConnectionPool(
            endpoint=endpoint,
            pool_size=pool_size,
            timeout=timeout
        )
        
        # 指标记录（限制内存使用）
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_latency_ms': 0.0,
            'latencies': []  # 只保留最近 1000 个
        }
        self.metrics_lock = asyncio.Lock()
        self._max_latency_records = 1000
        
        # 立即初始化连接池（避免首次请求延迟）
        if eager_init:
            asyncio.create_task(self.connection_pool.initialize())
        
        logger.info(f"LLMClient initialized with endpoint: {endpoint}, api_type: {self.api_type}")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.3,
        stop_sequences: Optional[List[str]] = None
    ) -> LLMResponse:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成 token 数
            temperature: 采样温度（0-1）
            stop_sequences: 停止序列
            
        Returns:
            LLMResponse: 包含生成文本和元数据
            
        Raises:
            LLMAPIError: API 调用失败
            LLMTimeoutError: 请求超时
        """
        # 速率限制
        await self.rate_limiter.acquire()
        
        # 使用重试策略执行请求
        try:
            response = await self.retry_policy.execute_with_retry(
                self._make_request,
                prompt,
                max_tokens,
                temperature,
                stop_sequences
            )
            
            # 记录成功指标
            await self._record_metrics(response, success=True)
            
            return response
        except (LLMAPIError, LLMTimeoutError) as e:
            # 记录失败指标
            await self._record_metrics(None, success=False)
            raise
    
    async def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.3
    ) -> List[LLMResponse]:
        """
        批量生成（并发处理，带并发控制）
        
        Args:
            prompts: 提示列表
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            
        Returns:
            List[LLMResponse]: 响应列表
            
        Note:
            使用 semaphore 控制并发数，避免耗尽连接池
        """
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def generate_with_semaphore(prompt: str):
            async with semaphore:
                return await self.generate(prompt, max_tokens, temperature)
        
        tasks = [
            generate_with_semaphore(prompt)
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        responses = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch generation failed: {result}")
                # 记录失败
                await self._record_metrics(None, success=False)
            else:
                responses.append(result)
        
        return responses
    
    async def _make_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop_sequences: Optional[List[str]]
    ) -> LLMResponse:
        """执行实际的 API 请求"""
        if self.api_type == "ollama":
            return await self._make_ollama_request(prompt, max_tokens, temperature, stop_sequences)
        else:
            return await self._make_openai_request(prompt, max_tokens, temperature, stop_sequences)
    
    async def _make_ollama_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop_sequences: Optional[List[str]]
    ) -> LLMResponse:
        """执行 Ollama API 请求"""
        start_time = time.time()
        
        # 构建 Ollama 请求
        request_data = {
            "model": "qwen2.5:7b-instruct",  # 默认模型
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        if stop_sequences:
            request_data["options"]["stop"] = stop_sequences
        
        # 构建请求头
        headers = {
            "Content-Type": "application/json"
        }
        
        # 获取连接
        session = await self.connection_pool.acquire()
        
        try:
            async with session.post(
                f"{self.endpoint}/api/generate",
                json=request_data,
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMAPIError(
                        f"Ollama API returned status {response.status}: {error_text}"
                    )
                
                data = await response.json()
                
                # 解析 Ollama 响应
                text = data.get('response', '')
                finish_reason = data.get('done_reason', 'stop')
                
                # Ollama 不直接返回 token 数，估算
                tokens_used = len(text.split()) + len(prompt.split())
                
                latency_ms = (time.time() - start_time) * 1000
                
                return LLMResponse(
                    text=text,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    model=data.get('model', 'unknown'),
                    finish_reason=finish_reason,
                    metadata=data
                )
        
        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(f"Request timed out after {self.timeout}s") from e
        except aiohttp.ClientError as e:
            raise LLMAPIError(f"Client error: {e}") from e
        finally:
            # 释放连接
            await self.connection_pool.release(session)
    
    async def _make_openai_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop_sequences: Optional[List[str]]
    ) -> LLMResponse:
        """执行 OpenAI 兼容的 API 请求"""
        start_time = time.time()
        
        # 构建 OpenAI 兼容的请求
        request_data = {
            "model": "gpt-3.5-turbo",  # 默认模型
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if stop_sequences:
            request_data["stop"] = stop_sequences
        
        # 构建请求头
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # 获取连接
        session = await self.connection_pool.acquire()
        
        try:
            async with session.post(
                f"{self.endpoint}/v1/chat/completions",
                json=request_data,
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMAPIError(
                        f"API returned status {response.status}: {error_text}"
                    )
                
                data = await response.json()
                
                # 解析响应
                choice = data['choices'][0]
                text = choice['message']['content']
                finish_reason = choice.get('finish_reason', 'stop')
                
                usage = data.get('usage', {})
                tokens_used = usage.get('total_tokens', 0)
                
                latency_ms = (time.time() - start_time) * 1000
                
                return LLMResponse(
                    text=text,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    model=data.get('model', 'unknown'),
                    finish_reason=finish_reason,
                    metadata=data
                )
        
        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(f"Request timed out after {self.timeout}s") from e
        except aiohttp.ClientError as e:
            raise LLMAPIError(f"Client error: {e}") from e
        finally:
            # 释放连接
            await self.connection_pool.release(session)
    
    async def _record_metrics(self, response: Optional[LLMResponse], success: bool):
        """记录指标（限制内存使用）"""
        async with self.metrics_lock:
            self.metrics['total_requests'] += 1
            
            if success and response:
                self.metrics['successful_requests'] += 1
                self.metrics['total_tokens'] += response.tokens_used
                self.metrics['total_latency_ms'] += response.latency_ms
                self.metrics['latencies'].append(response.latency_ms)
                
                # 限制延迟记录数量，防止内存无限增长
                if len(self.metrics['latencies']) > self._max_latency_records:
                    self.metrics['latencies'] = self.metrics['latencies'][-self._max_latency_records:]
            else:
                self.metrics['failed_requests'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取客户端指标（延迟、token 使用量等）"""
        total_requests = self.metrics['total_requests']
        successful_requests = self.metrics['successful_requests']
        
        avg_latency = 0.0
        if successful_requests > 0:
            avg_latency = self.metrics['total_latency_ms'] / successful_requests
        
        avg_tokens = 0.0
        if successful_requests > 0:
            avg_tokens = self.metrics['total_tokens'] / successful_requests
        
        success_rate = 0.0
        if total_requests > 0:
            success_rate = successful_requests / total_requests
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': success_rate,
            'total_tokens': self.metrics['total_tokens'],
            'avg_tokens_per_request': avg_tokens,
            'avg_latency_ms': avg_latency,
            'recent_latencies': self.metrics['latencies'][-10:]  # 最近 10 个
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            Dict: 健康状态信息
                - healthy: bool - 是否健康
                - connection_pool_available: int - 可用连接数
                - connection_pool_size: int - 连接池总大小
                - metrics: Dict - 性能指标
        """
        try:
            # 检查连接池状态
            pool_available = self.connection_pool.available.qsize() if self.connection_pool._initialized else 0
            pool_size = self.connection_pool.pool_size
            
            # 获取指标
            metrics = self.get_metrics()
            
            # 判断健康状态
            healthy = (
                not self._closed and
                pool_available > 0 and
                (metrics['success_rate'] > 0.5 or metrics['total_requests'] == 0)
            )
            
            return {
                'healthy': healthy,
                'connection_pool_available': pool_available,
                'connection_pool_size': pool_size,
                'metrics': metrics,
                'endpoint': self.endpoint
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e)
            }
    
    async def __aenter__(self):
        """上下文管理器入口"""
        # 确保连接池已初始化
        await self.connection_pool.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        await self.close()
        return False
    
    async def close(self):
        """关闭客户端"""
        if self._closed:
            return
        
        self._closed = True
        await self.connection_pool.close()
        logger.info("LLMClient closed")


class ArrowLLMClient(LLMClient):
    """
    Arrow 引擎原生客户端。
    不再依赖外部 HTTP API，而是直接调用底层的 ArrowEngine 进行 Zero-Copy 的快速推理。
    """
    def __init__(self, arrow_engine):
        # 绕过连接池和外部API类型检测
        super().__init__(endpoint="local://arrow_memory", eager_init=False, api_type="arrow")
        self.arrow_engine = arrow_engine

    async def _make_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop_sequences: Optional[List[str]]
    ) -> LLMResponse:
        """执行原生内存的生成直接穿透 ArrowEngine"""
        start_time = time.time()
        
        try:
            # CPU/GPU 纯本地运算，脱离网络协议栈
            text = self.arrow_engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            latency_ms = (time.time() - start_time) * 1000
            tokens_used = len(text.split()) + len(prompt.split())  # 简单估算
            
            return LLMResponse(
                text=text,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                model="arrow-native-llm",
                finish_reason="stop",
                metadata={"source": "ArrowEngine"}
            )
        except Exception as e:
            logger.error(f"ArrowEngine native generation failed: {e}")
            raise LLMAPIError(f"Arrow Native Inference Error: {e}") from e

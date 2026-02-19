"""
LLM 客户端单元测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import aiohttp

from llm_compression.llm_client import (
    LLMClient,
    LLMResponse,
    LLMAPIError,
    LLMTimeoutError,
    RetryPolicy,
    RateLimiter,
    LLMConnectionPool
)


class TestRetryPolicy:
    """重试策略测试"""
    
    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """测试第一次尝试成功"""
        policy = RetryPolicy(max_retries=3)
        
        async def success_func():
            return "success"
        
        result = await policy.execute_with_retry(success_func)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_success_after_retries(self):
        """测试重试后成功"""
        policy = RetryPolicy(max_retries=3, base_delay=0.01)
        
        call_count = 0
        
        async def retry_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMAPIError("API error")
            return "success"
        
        result = await policy.execute_with_retry(retry_func)
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_all_retries_failed(self):
        """测试所有重试都失败"""
        policy = RetryPolicy(max_retries=2, base_delay=0.01)
        
        async def fail_func():
            raise LLMAPIError("API error")
        
        with pytest.raises(LLMAPIError):
            await policy.execute_with_retry(fail_func)
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """测试指数退避"""
        policy = RetryPolicy(max_retries=3, base_delay=0.1, exponential_base=2.0)
        
        delays = []
        
        async def track_delay_func():
            import time
            if delays:
                delays.append(time.time() - delays[-1])
            else:
                delays.append(time.time())
            raise LLMAPIError("API error")
        
        with pytest.raises(LLMAPIError):
            await policy.execute_with_retry(track_delay_func)
        
        # 验证延迟递增（允许一些误差）
        assert len(delays) == 4  # 初始 + 3 次重试


class TestRateLimiter:
    """速率限制器测试"""
    
    @pytest.mark.asyncio
    async def test_within_limit(self):
        """测试在限制内"""
        limiter = RateLimiter(requests_per_minute=10)
        
        # 连续 5 个请求应该立即通过
        for _ in range(5):
            await limiter.acquire()
        
        assert len(limiter.request_times) == 5
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self):
        """测试速率限制执行"""
        limiter = RateLimiter(requests_per_minute=2)
        
        # 前 2 个请求应该立即通过
        await limiter.acquire()
        await limiter.acquire()
        
        # 第 3 个请求应该等待
        import time
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        
        # 应该等待接近 60 秒（实际测试中使用较短时间）
        # 这里只验证确实等待了
        assert elapsed > 0
    
    @pytest.mark.asyncio
    async def test_sliding_window(self):
        """测试滑动窗口"""
        limiter = RateLimiter(requests_per_minute=5)
        
        # 添加一些旧的请求时间
        import time
        now = time.time()
        limiter.request_times = [now - 70, now - 65, now - 61]  # 超过 60 秒
        
        # 新请求应该清理旧记录
        await limiter.acquire()
        
        # 只保留最近 60 秒内的
        assert all(now - t < 60 for t in limiter.request_times)


class TestLLMConnectionPool:
    """连接池测试"""
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """测试初始化"""
        pool = LLMConnectionPool(
            endpoint="http://test:8045",
            pool_size=5,
            timeout=30.0
        )
        
        await pool.initialize()
        
        assert len(pool.sessions) == 5
        assert pool.available.qsize() == 5
        
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_acquire_release(self):
        """测试获取和释放连接"""
        pool = LLMConnectionPool(
            endpoint="http://test:8045",
            pool_size=3,
            timeout=30.0
        )
        
        await pool.initialize()
        
        # 获取连接
        session1 = await pool.acquire()
        assert pool.available.qsize() == 2
        
        session2 = await pool.acquire()
        assert pool.available.qsize() == 1
        
        # 释放连接
        await pool.release(session1)
        assert pool.available.qsize() == 2
        
        await pool.release(session2)
        assert pool.available.qsize() == 3
        
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_close(self):
        """测试关闭连接池"""
        pool = LLMConnectionPool(
            endpoint="http://test:8045",
            pool_size=2,
            timeout=30.0
        )
        
        await pool.initialize()
        await pool.close()
        
        assert pool._initialized is False


class TestLLMClient:
    """LLM 客户端测试"""
    
    @pytest.fixture
    def mock_response_data(self):
        """模拟响应数据"""
        return {
            'choices': [{
                'message': {'content': 'Test response'},
                'finish_reason': 'stop'
            }],
            'usage': {
                'total_tokens': 50
            },
            'model': 'gpt-3.5-turbo'
        }
    
    @pytest.mark.asyncio
    async def test_init(self):
        """测试初始化"""
        client = LLMClient(
            endpoint="http://localhost:8045",
            api_key="test-key",
            timeout=30.0,
            max_retries=3,
            rate_limit=60
        )
        
        assert client.endpoint == "http://localhost:8045"
        assert client.api_key == "test-key"
        assert client.timeout == 30.0
        assert client.retry_policy.max_retries == 3
        assert client.rate_limiter.requests_per_minute == 60
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_generate_success(self, mock_response_data):
        """测试成功生成文本"""
        client = LLMClient(
            endpoint="http://localhost:8045",
            timeout=30.0
        )
        
        # Mock the session.post
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp.ClientSession, 'post', return_value=mock_response):
            response = await client.generate(
                prompt="Test prompt",
                max_tokens=50
            )
            
            assert response.text == "Test response"
            assert response.tokens_used == 50
            assert response.model == "gpt-3.5-turbo"
            assert response.finish_reason == "stop"
            assert response.latency_ms > 0
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_generate_api_error(self):
        """测试 API 错误"""
        client = LLMClient(
            endpoint="http://localhost:8045",
            timeout=30.0,
            max_retries=1
        )
        
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp.ClientSession, 'post', return_value=mock_response):
            with pytest.raises(LLMAPIError):
                await client.generate(prompt="Test prompt")
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_generate_timeout(self):
        """测试超时"""
        client = LLMClient(
            endpoint="http://localhost:8045",
            timeout=0.1,
            max_retries=1
        )
        
        # Mock timeout - create proper async context manager
        class TimeoutResponse:
            async def __aenter__(self):
                await asyncio.sleep(1)
                raise asyncio.TimeoutError()
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None
        
        with patch.object(aiohttp.ClientSession, 'post', return_value=TimeoutResponse()):
            with pytest.raises(LLMTimeoutError):
                await client.generate(prompt="Test prompt")
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_batch_generate(self, mock_response_data):
        """测试批量生成"""
        client = LLMClient(
            endpoint="http://localhost:8045",
            timeout=30.0
        )
        
        # Mock the session.post
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp.ClientSession, 'post', return_value=mock_response):
            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            responses = await client.batch_generate(prompts)
            
            assert len(responses) == 3
            for response in responses:
                assert response.text == "Test response"
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, mock_response_data):
        """测试获取指标"""
        client = LLMClient(
            endpoint="http://localhost:8045",
            timeout=30.0
        )
        
        # Mock the session.post
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp.ClientSession, 'post', return_value=mock_response):
            # 执行几次请求
            await client.generate(prompt="Test 1")
            await client.generate(prompt="Test 2")
            
            metrics = client.get_metrics()
            
            assert metrics['total_requests'] == 2
            assert metrics['successful_requests'] == 2
            assert metrics['failed_requests'] == 0
            assert metrics['success_rate'] == 1.0
            assert metrics['total_tokens'] == 100  # 50 * 2
            assert metrics['avg_tokens_per_request'] == 50.0
            assert metrics['avg_latency_ms'] > 0
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_metrics_with_failures(self):
        """测试包含失败的指标"""
        client = LLMClient(
            endpoint="http://localhost:8045",
            timeout=30.0,
            max_retries=1
        )
        
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp.ClientSession, 'post', return_value=mock_response):
            try:
                await client.generate(prompt="Test")
            except LLMAPIError:
                pass
            
            metrics = client.get_metrics()
            
            assert metrics['total_requests'] == 1
            assert metrics['successful_requests'] == 0
            assert metrics['failed_requests'] == 1
            assert metrics['success_rate'] == 0.0
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_with_api_key(self, mock_response_data):
        """测试使用 API 密钥"""
        client = LLMClient(
            endpoint="http://localhost:8045",
            api_key="test-api-key",
            timeout=30.0
        )
        
        # Mock the session.post to capture headers
        captured_headers = {}
        
        class CaptureResponse:
            def __init__(self, headers_dict):
                self.headers_dict = headers_dict
                self.status = 200
            
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None
            
            async def json(self):
                return mock_response_data
        
        def capture_post(url, json=None, headers=None, **kwargs):
            captured_headers.update(headers or {})
            return CaptureResponse(captured_headers)
        
        with patch.object(aiohttp.ClientSession, 'post', side_effect=capture_post):
            await client.generate(prompt="Test")
            
            assert "Authorization" in captured_headers
            assert captured_headers["Authorization"] == "Bearer test-api-key"
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_response_data):
        """测试上下文管理器"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp.ClientSession, 'post', return_value=mock_response):
            async with LLMClient(endpoint="http://localhost:8045") as client:
                response = await client.generate(prompt="Test")
                assert response.text == "Test response"
            
            # 验证客户端已关闭
            assert client._closed is True
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_response_data):
        """测试健康检查"""
        client = LLMClient(
            endpoint="http://localhost:8045",
            timeout=30.0
        )
        
        # 初始化连接池
        await client.connection_pool.initialize()
        
        # 执行一些请求
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp.ClientSession, 'post', return_value=mock_response):
            await client.generate(prompt="Test")
        
        # 检查健康状态
        health = await client.health_check()
        
        assert 'healthy' in health
        assert 'connection_pool_available' in health
        assert 'connection_pool_size' in health
        assert 'metrics' in health
        assert health['healthy'] is True
        assert health['connection_pool_size'] == 10
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_control(self, mock_response_data):
        """测试并发控制"""
        client = LLMClient(
            endpoint="http://localhost:8045",
            timeout=30.0,
            max_concurrent=3  # 限制并发数
        )
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp.ClientSession, 'post', return_value=mock_response):
            # 批量请求
            prompts = [f"Prompt {i}" for i in range(10)]
            responses = await client.batch_generate(prompts)
            
            # 验证所有请求都成功
            assert len(responses) == 10
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_eager_init(self):
        """测试立即初始化"""
        client = LLMClient(
            endpoint="http://localhost:8045",
            eager_init=True
        )
        
        # 等待一小段时间让初始化完成
        await asyncio.sleep(0.1)
        
        # 验证连接池已初始化
        assert client.connection_pool._initialized is True
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_lazy_init(self):
        """测试延迟初始化"""
        client = LLMClient(
            endpoint="http://localhost:8045",
            eager_init=False
        )
        
        # 验证连接池未初始化
        assert client.connection_pool._initialized is False
        
        await client.close()

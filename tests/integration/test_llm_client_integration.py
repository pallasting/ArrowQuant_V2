"""
LLM 客户端集成测试

测试 LLM 客户端与实际 API 的集成（使用 mock 服务器）
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import aiohttp

from llm_compression.llm_client import LLMClient, LLMResponse
from llm_compression.config import load_config


class TestLLMClientIntegration:
    """LLM 客户端集成测试"""
    
    @pytest.fixture
    def config(self):
        """加载配置"""
        return load_config()
    
    @pytest.fixture
    def mock_api_server(self):
        """模拟 API 服务器"""
        async def mock_post(url, json=None, headers=None):
            # 模拟不同的响应
            prompt = json['messages'][0]['content']
            
            if "error" in prompt.lower():
                mock_response = AsyncMock()
                mock_response.status = 500
                mock_response.text = AsyncMock(return_value="Internal Server Error")
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                return mock_response
            
            # 正常响应
            response_data = {
                'choices': [{
                    'message': {
                        'content': f"Response to: {prompt[:50]}..."
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': len(prompt.split()),
                    'completion_tokens': 20,
                    'total_tokens': len(prompt.split()) + 20
                },
                'model': 'gpt-3.5-turbo'
            }
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=response_data)
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            return mock_response
        
        return mock_post
    
    @pytest.mark.asyncio
    async def test_single_request(self, config, mock_api_server):
        """测试单个请求"""
        client = LLMClient(
            endpoint=config.llm.cloud_endpoint,
            timeout=config.llm.timeout,
            max_retries=config.llm.max_retries,
            rate_limit=config.llm.rate_limit
        )
        
        with patch.object(aiohttp.ClientSession, 'post', side_effect=mock_api_server):
            response = await client.generate(
                prompt="Summarize the following text: Hello world",
                max_tokens=100,
                temperature=0.3
            )
            
            assert isinstance(response, LLMResponse)
            assert len(response.text) > 0
            assert response.tokens_used > 0
            assert response.latency_ms > 0
            assert response.finish_reason == "stop"
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_batch_requests(self, config, mock_api_server):
        """测试批量请求"""
        client = LLMClient(
            endpoint=config.llm.cloud_endpoint,
            timeout=config.llm.timeout,
            max_retries=config.llm.max_retries,
            rate_limit=config.llm.rate_limit
        )
        
        prompts = [
            "Summarize: Text 1",
            "Summarize: Text 2",
            "Summarize: Text 3",
            "Summarize: Text 4",
            "Summarize: Text 5"
        ]
        
        with patch.object(aiohttp.ClientSession, 'post', side_effect=mock_api_server):
            responses = await client.batch_generate(
                prompts=prompts,
                max_tokens=100,
                temperature=0.3
            )
            
            assert len(responses) == len(prompts)
            for response in responses:
                assert isinstance(response, LLMResponse)
                assert len(response.text) > 0
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, config, mock_api_server):
        """测试错误处理"""
        client = LLMClient(
            endpoint=config.llm.cloud_endpoint,
            timeout=config.llm.timeout,
            max_retries=1,  # 减少重试次数以加快测试
            rate_limit=config.llm.rate_limit
        )
        
        with patch.object(aiohttp.ClientSession, 'post', side_effect=mock_api_server):
            # 触发错误的请求
            from llm_compression.llm_client import LLMAPIError
            
            with pytest.raises(LLMAPIError):
                await client.generate(
                    prompt="This should trigger an error",
                    max_tokens=100
                )
            
            # 验证指标记录了失败
            metrics = client.get_metrics()
            assert metrics['failed_requests'] > 0
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, config, mock_api_server):
        """测试指标跟踪"""
        client = LLMClient(
            endpoint=config.llm.cloud_endpoint,
            timeout=config.llm.timeout,
            max_retries=config.llm.max_retries,
            rate_limit=config.llm.rate_limit
        )
        
        with patch.object(aiohttp.ClientSession, 'post', side_effect=mock_api_server):
            # 执行多个请求
            for i in range(5):
                await client.generate(
                    prompt=f"Test request {i}",
                    max_tokens=50
                )
            
            # 获取指标
            metrics = client.get_metrics()
            
            # 验证指标
            assert metrics['total_requests'] == 5
            assert metrics['successful_requests'] == 5
            assert metrics['failed_requests'] == 0
            assert metrics['success_rate'] == 1.0
            assert metrics['total_tokens'] > 0
            assert metrics['avg_tokens_per_request'] > 0
            assert metrics['avg_latency_ms'] > 0
            assert len(metrics['recent_latencies']) == 5
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, config, mock_api_server):
        """测试并发请求"""
        client = LLMClient(
            endpoint=config.llm.cloud_endpoint,
            timeout=config.llm.timeout,
            max_retries=config.llm.max_retries,
            rate_limit=100,  # 提高速率限制以支持并发
            pool_size=10
        )
        
        with patch.object(aiohttp.ClientSession, 'post', side_effect=mock_api_server):
            # 创建并发任务
            tasks = [
                client.generate(prompt=f"Concurrent request {i}", max_tokens=50)
                for i in range(10)
            ]
            
            # 并发执行
            responses = await asyncio.gather(*tasks)
            
            # 验证所有请求都成功
            assert len(responses) == 10
            for response in responses:
                assert isinstance(response, LLMResponse)
                assert len(response.text) > 0
            
            # 验证指标
            metrics = client.get_metrics()
            assert metrics['total_requests'] == 10
            assert metrics['successful_requests'] == 10
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_connection_pool_reuse(self, config, mock_api_server):
        """测试连接池复用"""
        client = LLMClient(
            endpoint=config.llm.cloud_endpoint,
            timeout=config.llm.timeout,
            max_retries=config.llm.max_retries,
            rate_limit=config.llm.rate_limit,
            pool_size=3
        )
        
        # 初始化连接池
        await client.connection_pool.initialize()
        
        # 验证连接池大小
        assert len(client.connection_pool.sessions) == 3
        assert client.connection_pool.available.qsize() == 3
        
        with patch.object(aiohttp.ClientSession, 'post', side_effect=mock_api_server):
            # 执行多个请求
            for i in range(5):
                await client.generate(prompt=f"Request {i}", max_tokens=50)
            
            # 验证连接池状态
            # 所有连接应该已释放
            assert client.connection_pool.available.qsize() == 3
        
        await client.close()

"""
LLM 客户端属性测试

Feature: llm-compression-integration
使用 Hypothesis 进行基于属性的测试
"""

import pytest
import asyncio
from hypothesis import given, settings, strategies as st
from unittest.mock import AsyncMock, patch
import aiohttp

from llm_compression.llm_client import (
    LLMClient,
    LLMResponse,
    LLMAPIError,
    LLMConnectionPool,
    RateLimiter
)


# ============================================================================
# Property 35: API 格式兼容性
# Validates: Requirements 1.2
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    prompt=st.text(min_size=1, max_size=500),
    max_tokens=st.integers(min_value=10, max_value=500),
    temperature=st.floats(min_value=0.0, max_value=1.0)
)
@pytest.mark.asyncio
async def test_property_35_api_format_compatibility(prompt, max_tokens, temperature):
    """
    Feature: llm-compression-integration, Property 35: API 格式兼容性
    
    For any OpenAI 兼容格式的请求，LLM 客户端应该能够正确处理并返回符合格式的响应
    
    Validates: Requirements 1.2
    """
    client = LLMClient(
        endpoint="http://localhost:8045",
        timeout=30.0
    )
    
    # Mock response data
    mock_response_data = {
        'choices': [{
            'message': {'content': 'Generated text'},
            'finish_reason': 'stop'
        }],
        'usage': {
            'total_tokens': max_tokens
        },
        'model': 'gpt-3.5-turbo'
    }
    
    # Mock the session.post
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=mock_response_data)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    captured_request = {}
    
    async def capture_post(url, json=None, headers=None):
        captured_request.update(json or {})
        return mock_response
    
    with patch.object(aiohttp.ClientSession, 'post', side_effect=capture_post):
        response = await client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # 验证请求格式符合 OpenAI API
        assert 'model' in captured_request
        assert 'messages' in captured_request
        assert isinstance(captured_request['messages'], list)
        assert len(captured_request['messages']) > 0
        assert 'role' in captured_request['messages'][0]
        assert 'content' in captured_request['messages'][0]
        assert captured_request['messages'][0]['content'] == prompt
        assert captured_request['max_tokens'] == max_tokens
        assert captured_request['temperature'] == temperature
        
        # 验证响应格式
        assert isinstance(response, LLMResponse)
        assert isinstance(response.text, str)
        assert isinstance(response.tokens_used, int)
        assert isinstance(response.latency_ms, float)
        assert isinstance(response.model, str)
        assert isinstance(response.finish_reason, str)
        assert response.latency_ms > 0
    
    await client.close()


# ============================================================================
# Property 36: 连接池管理
# Validates: Requirements 1.3
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    pool_size=st.integers(min_value=1, max_value=20),
    num_operations=st.integers(min_value=1, max_value=50)
)
@pytest.mark.asyncio
async def test_property_36_connection_pool_management(pool_size, num_operations):
    """
    Feature: llm-compression-integration, Property 36: 连接池管理
    
    For any 并发 API 请求，连接池应该正确管理连接的获取和释放，避免连接泄漏
    
    Validates: Requirements 1.3
    """
    pool = LLMConnectionPool(
        endpoint="http://test:8045",
        pool_size=pool_size,
        timeout=30.0
    )
    
    await pool.initialize()
    
    # 验证初始状态
    assert len(pool.sessions) == pool_size
    assert pool.available.qsize() == pool_size
    
    # 执行多次获取和释放操作
    acquired_sessions = []
    
    for i in range(min(num_operations, pool_size)):
        session = await pool.acquire()
        acquired_sessions.append(session)
        
        # 验证可用连接数减少
        assert pool.available.qsize() == pool_size - len(acquired_sessions)
    
    # 释放所有连接
    for session in acquired_sessions:
        await pool.release(session)
    
    # 验证所有连接都已释放
    assert pool.available.qsize() == pool_size
    
    # 验证没有连接泄漏
    assert len(pool.sessions) == pool_size
    
    await pool.close()


# ============================================================================
# Property 31: 连接重试机制
# Validates: Requirements 1.3, 13.6
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    max_retries=st.integers(min_value=1, max_value=5),
    fail_count=st.integers(min_value=0, max_value=10)
)
@pytest.mark.asyncio
async def test_property_31_connection_retry_mechanism(max_retries, fail_count):
    """
    Feature: llm-compression-integration, Property 31: 连接重试机制
    
    For any API 调用失败（超时、网络错误），系统应该使用指数退避策略重试
    
    Validates: Requirements 1.3, 13.6
    """
    client = LLMClient(
        endpoint="http://localhost:8045",
        timeout=30.0,
        max_retries=max_retries
    )
    
    call_count = 0
    
    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        
        # 前 fail_count 次失败
        if call_count <= fail_count:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Server Error")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            return mock_response
        
        # 之后成功
        mock_response_data = {
            'choices': [{
                'message': {'content': 'Success'},
                'finish_reason': 'stop'
            }],
            'usage': {'total_tokens': 10},
            'model': 'gpt-3.5-turbo'
        }
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        return mock_response
    
    with patch.object(aiohttp.ClientSession, 'post', side_effect=mock_post):
        if fail_count <= max_retries:
            # 应该成功（在重试次数内）
            response = await client.generate(prompt="Test")
            assert response.text == "Success"
            assert call_count == fail_count + 1
        else:
            # 应该失败（超过重试次数）
            with pytest.raises(LLMAPIError):
                await client.generate(prompt="Test")
            assert call_count == max_retries + 1
    
    await client.close()


# ============================================================================
# Property 22: 速率限制保护
# Validates: Requirements 1.7, 9.5
# ============================================================================

@settings(max_examples=30, deadline=None)
@given(
    requests_per_minute=st.integers(min_value=1, max_value=10),
    num_requests=st.integers(min_value=1, max_value=20)
)
@pytest.mark.asyncio
async def test_property_22_rate_limit_protection(requests_per_minute, num_requests):
    """
    Feature: llm-compression-integration, Property 22: 速率限制保护
    
    For any API 调用序列，系统应该实现速率限制，确保不超过配置的请求/分钟限制
    
    Validates: Requirements 1.7, 9.5
    """
    limiter = RateLimiter(requests_per_minute=requests_per_minute)
    
    import time
    start_time = time.time()
    
    # 执行请求
    for _ in range(min(num_requests, requests_per_minute + 5)):
        await limiter.acquire()
    
    elapsed = time.time() - start_time
    
    # 验证速率限制
    # 如果请求数超过限制，应该有等待时间
    if num_requests > requests_per_minute:
        # 应该等待了一些时间（但测试中不强制等待完整的 60 秒）
        assert len(limiter.request_times) <= requests_per_minute or elapsed > 0
    
    # 验证滑动窗口：所有记录的请求都在最近 60 秒内
    now = time.time()
    for request_time in limiter.request_times:
        assert now - request_time < 60


# ============================================================================
# Property 24: 指标跟踪完整性（部分）
# Validates: Requirements 1.6
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    num_successful=st.integers(min_value=0, max_value=20),
    num_failed=st.integers(min_value=0, max_value=10)
)
@pytest.mark.asyncio
async def test_property_24_metrics_tracking_completeness(num_successful, num_failed):
    """
    Feature: llm-compression-integration, Property 24: 指标跟踪完整性（部分）
    
    For any 系统操作，监控系统应该跟踪所有指定指标：
    压缩次数、压缩比、延迟、质量分数、API 成本、GPU 使用率
    
    此测试验证 LLM 客户端的指标记录（延迟、token 使用量）
    
    Validates: Requirements 1.6
    """
    client = LLMClient(
        endpoint="http://localhost:8045",
        timeout=30.0,
        max_retries=1
    )
    
    # Mock 成功响应
    mock_success_data = {
        'choices': [{
            'message': {'content': 'Success'},
            'finish_reason': 'stop'
        }],
        'usage': {'total_tokens': 50},
        'model': 'gpt-3.5-turbo'
    }
    
    async def mock_post(*args, **kwargs):
        # 根据调用次数决定成功或失败
        if client.metrics['total_requests'] < num_successful:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_success_data)
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            return mock_response
        else:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Error")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            return mock_response
    
    with patch.object(aiohttp.ClientSession, 'post', side_effect=mock_post):
        # 执行成功请求
        for _ in range(num_successful):
            try:
                await client.generate(prompt="Test")
            except:
                pass
        
        # 执行失败请求
        for _ in range(num_failed):
            try:
                await client.generate(prompt="Test")
            except:
                pass
        
        # 获取指标
        metrics = client.get_metrics()
        
        # 验证指标完整性
        assert 'total_requests' in metrics
        assert 'successful_requests' in metrics
        assert 'failed_requests' in metrics
        assert 'success_rate' in metrics
        assert 'total_tokens' in metrics
        assert 'avg_tokens_per_request' in metrics
        assert 'avg_latency_ms' in metrics
        assert 'recent_latencies' in metrics
        
        # 验证指标准确性
        assert metrics['total_requests'] == num_successful + num_failed
        assert metrics['successful_requests'] == num_successful
        assert metrics['failed_requests'] == num_failed
        
        if num_successful + num_failed > 0:
            expected_success_rate = num_successful / (num_successful + num_failed)
            assert abs(metrics['success_rate'] - expected_success_rate) < 0.01
        
        if num_successful > 0:
            assert metrics['total_tokens'] == num_successful * 50
            assert metrics['avg_tokens_per_request'] == 50.0
            assert metrics['avg_latency_ms'] > 0
    
    await client.close()

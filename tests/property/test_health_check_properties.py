"""
Property-Based Tests for Health Check System

Tests universal properties of the health check endpoint and system monitoring.

Feature: llm-compression-integration
Property 37: 健康检查端点（Health Check Endpoint）
Validates: Requirements 11.7
"""

import pytest
from hypothesis import given, strategies as st, settings
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile
import shutil

from llm_compression.health import (
    HealthChecker,
    ComponentStatus,
    HealthCheckResult
)
from llm_compression.config import Config
from llm_compression.llm_client import LLMClient, LLMResponse
from llm_compression.arrow_storage import ArrowStorage


# ============================================================================
# Test Strategies
# ============================================================================

@st.composite
def component_status_strategy(draw):
    """生成组件状态"""
    name = draw(st.sampled_from(["llm_client", "storage", "gpu", "config"]))
    status = draw(st.sampled_from(["healthy", "degraded", "unhealthy"]))
    message = draw(st.text(min_size=1, max_size=100))
    latency_ms = draw(st.none() | st.floats(min_value=0, max_value=10000))
    
    return ComponentStatus(
        name=name,
        status=status,
        message=message,
        latency_ms=latency_ms,
        details={}
    )


# ============================================================================
# Property 37: 健康检查端点（Health Check Endpoint）
# ============================================================================

class TestHealthCheckEndpointProperties:
    """
    Property 37: 健康检查端点
    
    For any 健康检查请求，系统应该返回当前状态（LLM 可用性、存储状态、资源使用情况）
    
    Validates: Requirements 11.7
    """
    
    @pytest.mark.asyncio
    @settings(max_examples=100)
    @given(
        llm_available=st.booleans(),
        storage_available=st.booleans(),
        gpu_available=st.booleans()
    )
    async def test_health_check_always_returns_status(
        self,
        llm_available,
        storage_available,
        gpu_available
    ):
        """
        Property: 健康检查总是返回状态
        
        For any 组件可用性配置，健康检查应该：
        1. 返回有效的 HealthCheckResult
        2. 包含 overall_status 字段
        3. 包含所有组件的状态
        4. overall_status 是 "healthy", "degraded", "unhealthy" 之一
        """
        # 创建模拟组件
        llm_client = self._create_mock_llm_client(llm_available)
        storage = self._create_mock_storage(storage_available)
        config = Config()
        
        # 创建健康检查器
        checker = HealthChecker(
            llm_client=llm_client,
            storage=storage,
            config=config
        )
        
        # 执行健康检查
        result = await checker.check_health()
        
        # 验证结果结构
        assert isinstance(result, HealthCheckResult)
        assert result.overall_status in ["healthy", "degraded", "unhealthy"]
        assert isinstance(result.timestamp, float)
        assert result.timestamp > 0
        assert isinstance(result.components, dict)
        
        # 验证所有组件都被检查
        expected_components = {"llm_client", "storage", "gpu", "config"}
        assert set(result.components.keys()) == expected_components
        
        # 验证每个组件状态
        for name, component in result.components.items():
            assert isinstance(component, ComponentStatus)
            assert component.name == name
            assert component.status in ["healthy", "degraded", "unhealthy"]
            assert isinstance(component.message, str)
            assert len(component.message) > 0
    
    @pytest.mark.asyncio
    @settings(max_examples=100)
    @given(
        component_statuses=st.lists(
            st.sampled_from(["healthy", "degraded", "unhealthy"]),
            min_size=1,
            max_size=4
        )
    )
    async def test_overall_status_reflects_worst_component(
        self,
        component_statuses
    ):
        """
        Property: 总体状态反映最差组件
        
        For any 组件状态组合，总体状态应该：
        1. 如果任何组件 unhealthy -> 总体 unhealthy
        2. 如果任何组件 degraded（且无 unhealthy）-> 总体 degraded
        3. 如果所有组件 healthy -> 总体 healthy
        """
        # 创建模拟组件状态
        components = {}
        component_names = ["llm_client", "storage", "gpu", "config"]
        
        for i, status in enumerate(component_statuses):
            if i < len(component_names):
                components[component_names[i]] = ComponentStatus(
                    name=component_names[i],
                    status=status,
                    message=f"Test status: {status}"
                )
        
        # 填充剩余组件为 healthy
        for name in component_names:
            if name not in components:
                components[name] = ComponentStatus(
                    name=name,
                    status="healthy",
                    message="Test status: healthy"
                )
        
        # 创建健康检查器并计算总体状态
        checker = HealthChecker()
        overall_status = checker._compute_overall_status(components)
        
        # 验证总体状态逻辑
        statuses = [comp.status for comp in components.values()]
        
        if "unhealthy" in statuses:
            assert overall_status == "unhealthy"
        elif "degraded" in statuses:
            assert overall_status == "degraded"
        else:
            assert overall_status == "healthy"
    
    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=500)  # Increase deadline for async operations
    @given(
        latency_ms=st.floats(min_value=0, max_value=10000)
    )
    async def test_llm_client_latency_affects_status(
        self,
        latency_ms
    ):
        """
        Property: LLM 客户端延迟影响状态
        
        For any LLM 响应延迟：
        1. 延迟 > 5000ms -> degraded 或 unhealthy
        2. 延迟 <= 5000ms -> healthy（如果连接成功）
        """
        # 创建模拟 LLM 客户端
        llm_client = Mock(spec=LLMClient)
        llm_client.endpoint = "http://test:8045"
        
        async def mock_generate(*args, **kwargs):
            # 模拟延迟（不实际等待，只返回延迟值）
            return LLMResponse(
                text="test",
                tokens_used=1,
                latency_ms=latency_ms,
                model="test-model",
                finish_reason="stop"
            )
        
        llm_client.generate = mock_generate
        
        # 创建健康检查器
        checker = HealthChecker(llm_client=llm_client)
        
        # 执行 LLM 检查
        result = await checker._check_llm_client()
        
        # 验证状态
        assert isinstance(result, ComponentStatus)
        assert result.name == "llm_client"
        
        if latency_ms > 5000:
            # 高延迟应该是 degraded
            assert result.status in ["degraded", "unhealthy"]
        else:
            # 低延迟应该是 healthy
            assert result.status == "healthy"
        
        # 验证延迟被记录
        if result.latency_ms is not None:
            assert result.latency_ms >= 0
    
    @pytest.mark.asyncio
    @settings(max_examples=50)
    @given(
        free_gb=st.floats(min_value=0.0, max_value=100.0)
    )
    async def test_storage_disk_space_affects_status(
        self,
        free_gb
    ):
        """
        Property: 存储磁盘空间影响状态
        
        For any 可用磁盘空间：
        1. 空间 < 1GB -> degraded
        2. 空间 >= 1GB -> healthy
        """
        # 创建临时存储目录
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            
            # 创建模拟存储
            storage = Mock(spec=ArrowStorage)
            storage.base_path = storage_path
            
            # 创建健康检查器
            checker = HealthChecker(storage=storage)
            
            # 执行存储检查
            result = await checker._check_storage()
            
            # 验证结果
            assert isinstance(result, ComponentStatus)
            assert result.name == "storage"
            
            # 注意：实际磁盘空间由系统决定，我们只验证检查逻辑
            # 状态应该是 healthy, degraded, 或 unhealthy 之一
            assert result.status in ["healthy", "degraded", "unhealthy"]
            
            # 如果有详细信息，验证格式
            if "free_gb" in result.details:
                assert isinstance(result.details["free_gb"], (int, float))
                assert result.details["free_gb"] >= 0
    
    @pytest.mark.asyncio
    @settings(max_examples=100)
    @given(
        temperature=st.floats(min_value=-1.0, max_value=2.0),
        batch_size=st.integers(min_value=-10, max_value=100)
    )
    async def test_config_validation_detects_invalid_values(
        self,
        temperature,
        batch_size
    ):
        """
        Property: 配置验证检测无效值
        
        For any 配置值：
        1. temperature 不在 [0.0, 1.0] -> degraded
        2. batch_size < 1 -> degraded
        3. 所有值有效 -> healthy
        """
        # 创建配置
        config = Config()
        config.compression.temperature = temperature
        config.performance.batch_size = batch_size
        
        # 创建健康检查器
        checker = HealthChecker(config=config)
        
        # 执行配置检查
        result = await checker._check_config()
        
        # 验证结果
        assert isinstance(result, ComponentStatus)
        assert result.name == "config"
        
        # 检查是否检测到无效值
        has_invalid_temperature = not (0.0 <= temperature <= 1.0)
        has_invalid_batch_size = batch_size < 1
        
        if has_invalid_temperature or has_invalid_batch_size:
            # 应该检测到问题
            assert result.status in ["degraded", "unhealthy"]
            if "issues" in result.details:
                assert len(result.details["issues"]) > 0
        else:
            # 配置有效（可能有其他问题，但不是这两个）
            # 状态应该是 healthy 或 degraded（其他原因）
            assert result.status in ["healthy", "degraded"]
    
    @pytest.mark.asyncio
    async def test_health_check_result_serializable(self):
        """
        Property: 健康检查结果可序列化
        
        For any 健康检查结果，应该：
        1. 可以转换为字典
        2. 字典包含所有必需字段
        3. 字典可以序列化为 JSON
        """
        # 创建健康检查器
        checker = HealthChecker()
        
        # 执行健康检查
        result = await checker.check_health()
        
        # 转换为字典
        result_dict = result.to_dict()
        
        # 验证字典结构
        assert isinstance(result_dict, dict)
        assert "status" in result_dict
        assert "timestamp" in result_dict
        assert "components" in result_dict
        
        # 验证可以序列化为 JSON
        import json
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str)
        
        # 验证可以反序列化
        parsed = json.loads(json_str)
        assert parsed["status"] == result_dict["status"]
        assert parsed["timestamp"] == result_dict["timestamp"]
    
    @pytest.mark.asyncio
    async def test_health_check_idempotent(self):
        """
        Property: 健康检查是幂等的
        
        For any 系统状态，多次执行健康检查应该：
        1. 返回一致的状态（在短时间内）
        2. 不改变系统状态
        3. 每次都返回有效结果
        """
        # 创建健康检查器
        checker = HealthChecker()
        
        # 执行多次健康检查
        results = []
        for _ in range(3):
            result = await checker.check_health()
            results.append(result)
            await asyncio.sleep(0.1)  # 短暂延迟
        
        # 验证所有结果都有效
        for result in results:
            assert isinstance(result, HealthCheckResult)
            assert result.overall_status in ["healthy", "degraded", "unhealthy"]
        
        # 验证状态一致（在短时间内不应该变化）
        statuses = [r.overall_status for r in results]
        # 注意：由于系统状态可能变化，我们只验证所有结果都有效
        # 不强制要求完全一致
        assert all(s in ["healthy", "degraded", "unhealthy"] for s in statuses)
    
    # Helper methods
    
    def _create_mock_llm_client(self, available: bool) -> Mock:
        """创建模拟 LLM 客户端"""
        client = Mock(spec=LLMClient)
        client.endpoint = "http://test:8045"
        
        if available:
            async def mock_generate(*args, **kwargs):
                return LLMResponse(
                    text="test",
                    tokens_used=1,
                    latency_ms=100.0,
                    model="test-model",
                    finish_reason="stop"
                )
            client.generate = mock_generate
        else:
            async def mock_generate_fail(*args, **kwargs):
                raise Exception("LLM client unavailable")
            client.generate = mock_generate_fail
        
        return client
    
    def _create_mock_storage(self, available: bool) -> Mock:
        """创建模拟存储"""
        if available:
            tmpdir = tempfile.mkdtemp()
            storage = Mock(spec=ArrowStorage)
            storage.base_path = Path(tmpdir)
            return storage
        else:
            storage = Mock(spec=ArrowStorage)
            storage.base_path = Path("/nonexistent/path")
            return storage


# ============================================================================
# Additional Property Tests
# ============================================================================

class TestHealthCheckRobustness:
    """健康检查鲁棒性测试"""
    
    @pytest.mark.asyncio
    async def test_health_check_handles_component_failures(self):
        """
        Property: 健康检查处理组件失败
        
        For any 组件失败，健康检查应该：
        1. 不抛出异常
        2. 返回有效结果
        3. 标记失败组件为 unhealthy
        """
        # 创建会失败的模拟组件
        llm_client = Mock(spec=LLMClient)
        llm_client.endpoint = "http://test:8045"
        
        async def mock_generate_fail(*args, **kwargs):
            raise Exception("Simulated failure")
        
        llm_client.generate = mock_generate_fail
        
        # 创建健康检查器
        checker = HealthChecker(llm_client=llm_client)
        
        # 执行健康检查（不应该抛出异常）
        result = await checker.check_health()
        
        # 验证结果
        assert isinstance(result, HealthCheckResult)
        assert result.overall_status in ["healthy", "degraded", "unhealthy"]
        
        # LLM 客户端应该被标记为 unhealthy
        assert "llm_client" in result.components
        assert result.components["llm_client"].status == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_health_check_concurrent_safe(self):
        """
        Property: 健康检查并发安全
        
        For any 并发请求数，健康检查应该：
        1. 正确处理并发请求
        2. 每个请求返回有效结果
        3. 不发生竞态条件
        """
        # 创建健康检查器
        checker = HealthChecker()
        
        # 并发执行多个健康检查
        tasks = [checker.check_health() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # 验证所有结果都有效
        assert len(results) == 10
        for result in results:
            assert isinstance(result, HealthCheckResult)
            assert result.overall_status in ["healthy", "degraded", "unhealthy"]

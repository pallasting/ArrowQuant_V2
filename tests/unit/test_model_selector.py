"""
模型选择器单元测试

测试 ModelSelector 的基本功能和边缘情况。
"""

import pytest
import asyncio

from llm_compression.model_selector import (
    ModelSelector,
    MemoryType,
    QualityLevel,
    ModelConfig,
    ModelStats
)


class TestModelSelector:
    """ModelSelector 基础功能测试"""
    
    @pytest.fixture
    def selector(self):
        """创建测试用的选择器"""
        return ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={
                "step-flash": "http://localhost:8046",
                "minicpm-o": "http://localhost:8047",
                "stable-diffcoder": "http://localhost:8048",
                "intern-s1-pro": "http://localhost:8049"
            },
            prefer_local=True,
            quality_threshold=0.85
        )
    
    def test_initialization(self, selector):
        """测试初始化"""
        assert selector.cloud_endpoint == "http://localhost:8045"
        assert len(selector.local_endpoints) == 4
        assert selector.prefer_local is True
        assert selector.quality_threshold == 0.85
        assert len(selector.model_stats) == 0
    
    def test_select_model_returns_config(self, selector):
        """测试选择模型返回配置"""
        config = selector.select_model(
            memory_type=MemoryType.TEXT,
            text_length=200,
            quality_requirement=QualityLevel.STANDARD
        )
        
        assert isinstance(config, ModelConfig)
        assert config.model_name is not None
        assert config.endpoint is not None
    
    def test_short_text_selects_step_flash(self, selector):
        """测试短文本选择 step-flash"""
        config = selector.select_model(
            memory_type=MemoryType.TEXT,
            text_length=200,
            quality_requirement=QualityLevel.STANDARD
        )
        
        # 本地优先，应该选择 step-flash
        assert config.model_name == "step-flash"
        assert config.is_local is True
    
    def test_long_text_selects_intern(self, selector):
        """测试长文本选择 intern-s1-pro"""
        config = selector.select_model(
            memory_type=MemoryType.LONG_TEXT,
            text_length=1000,
            quality_requirement=QualityLevel.STANDARD
        )
        
        # 本地优先，应该选择 intern-s1-pro
        assert config.model_name == "intern-s1-pro"
        assert config.is_local is True
    
    def test_code_memory_selects_diffcoder(self, selector):
        """测试代码记忆选择 stable-diffcoder"""
        config = selector.select_model(
            memory_type=MemoryType.CODE,
            text_length=500,
            quality_requirement=QualityLevel.STANDARD
        )
        
        # 本地优先，应该选择 stable-diffcoder
        assert config.model_name == "stable-diffcoder"
        assert config.is_local is True
    
    def test_multimodal_selects_minicpm(self, selector):
        """测试多模态选择 minicpm-o"""
        config = selector.select_model(
            memory_type=MemoryType.MULTIMODAL,
            text_length=300,
            quality_requirement=QualityLevel.STANDARD
        )
        
        # 本地优先，应该选择 minicpm-o
        assert config.model_name == "minicpm-o"
        assert config.is_local is True
    
    def test_high_quality_selects_cloud(self, selector):
        """测试高质量要求选择云端 API"""
        config = selector.select_model(
            memory_type=MemoryType.TEXT,
            text_length=200,
            quality_requirement=QualityLevel.HIGH
        )
        
        # 高质量要求应该选择云端 API
        assert config.model_name == "cloud-api"
        assert config.is_local is False
    
    def test_manual_model_override(self, selector):
        """测试手动指定模型"""
        config = selector.select_model(
            memory_type=MemoryType.TEXT,
            text_length=200,
            quality_requirement=QualityLevel.STANDARD,
            manual_model="cloud-api"
        )
        
        # 应该使用手动指定的模型
        assert config.model_name == "cloud-api"
    
    def test_cloud_only_selector(self):
        """测试仅云端的选择器"""
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={},
            prefer_local=False
        )
        
        config = selector.select_model(
            memory_type=MemoryType.TEXT,
            text_length=200,
            quality_requirement=QualityLevel.STANDARD
        )
        
        # 没有本地模型，应该使用云端 API
        assert config.model_name == "cloud-api"
    
    @pytest.mark.asyncio
    async def test_record_usage(self, selector):
        """测试记录使用统计"""
        model_name = "test-model"
        
        await selector.record_usage(
            model_name=model_name,
            latency_ms=500.0,
            quality_score=0.9,
            tokens_used=100,
            success=True
        )
        
        stats = selector.get_model_stats()
        assert model_name in stats
        
        model_stats = stats[model_name]
        assert model_stats.total_requests == 1
        assert model_stats.avg_latency_ms == 500.0
        assert model_stats.avg_quality_score == 0.9
        assert model_stats.total_tokens_used == 100
        assert model_stats.success_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_record_multiple_usage(self, selector):
        """测试记录多次使用"""
        model_name = "test-model"
        
        # 记录 3 次使用
        await selector.record_usage(
            model_name=model_name,
            latency_ms=100.0,
            quality_score=0.8,
            tokens_used=50,
            success=True
        )
        await selector.record_usage(
            model_name=model_name,
            latency_ms=200.0,
            quality_score=0.9,
            tokens_used=100,
            success=True
        )
        await selector.record_usage(
            model_name=model_name,
            latency_ms=300.0,
            quality_score=1.0,
            tokens_used=150,
            success=True
        )
        
        stats = selector.get_model_stats()
        model_stats = stats[model_name]
        
        assert model_stats.total_requests == 3
        assert model_stats.avg_latency_ms == 200.0  # (100 + 200 + 300) / 3
        assert model_stats.avg_quality_score == 0.9  # (0.8 + 0.9 + 1.0) / 3
        assert model_stats.total_tokens_used == 300  # 50 + 100 + 150
        assert model_stats.success_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_record_failed_usage(self, selector):
        """测试记录失败的使用"""
        model_name = "test-model"
        
        # 记录 1 次成功，1 次失败
        await selector.record_usage(
            model_name=model_name,
            latency_ms=100.0,
            quality_score=0.9,
            tokens_used=50,
            success=True
        )
        await selector.record_usage(
            model_name=model_name,
            latency_ms=0.0,
            quality_score=0.0,
            tokens_used=0,
            success=False
        )
        
        stats = selector.get_model_stats()
        model_stats = stats[model_name]
        
        assert model_stats.total_requests == 2
        assert model_stats._successful_requests == 1
        assert model_stats.success_rate == 0.5  # 1 / 2
    
    @pytest.mark.asyncio
    async def test_suggest_model_switch_low_quality(self, selector):
        """测试低质量建议切换"""
        model_name = "step-flash"
        
        # 记录低质量使用
        await selector.record_usage(
            model_name=model_name,
            latency_ms=500.0,
            quality_score=0.7,  # 低于阈值 0.85
            tokens_used=100,
            success=True
        )
        
        suggestion = selector.suggest_model_switch(model_name)
        assert suggestion == "cloud-api"
    
    @pytest.mark.asyncio
    async def test_suggest_model_switch_high_quality(self, selector):
        """测试高质量不建议切换"""
        model_name = "step-flash"
        
        # 记录高质量使用
        await selector.record_usage(
            model_name=model_name,
            latency_ms=500.0,
            quality_score=0.95,  # 高于阈值 0.85
            tokens_used=100,
            success=True
        )
        
        suggestion = selector.suggest_model_switch(model_name)
        assert suggestion is None
    
    @pytest.mark.asyncio
    async def test_suggest_model_switch_cloud_api(self, selector):
        """测试云端 API 低质量不建议切换"""
        model_name = "cloud-api"
        
        # 记录低质量使用
        await selector.record_usage(
            model_name=model_name,
            latency_ms=2000.0,
            quality_score=0.7,  # 低于阈值 0.85
            tokens_used=100,
            success=True
        )
        
        # 已经是云端 API，无法进一步提升
        suggestion = selector.suggest_model_switch(model_name)
        assert suggestion is None
    
    def test_clear_availability_cache(self, selector):
        """测试清除可用性缓存"""
        # 触发缓存
        selector._is_model_available("cloud-api")
        assert len(selector._availability_cache) > 0
        
        # 清除缓存
        selector.clear_availability_cache()
        assert len(selector._availability_cache) == 0
        assert len(selector._cache_timestamps) == 0
    
    def test_model_config_values(self, selector):
        """测试模型配置值"""
        # 测试云端 API 配置
        config = selector._get_model_config("cloud-api")
        assert config.model_name == "cloud-api"
        assert config.endpoint == "http://localhost:8045"
        assert config.is_local is False
        assert config.max_tokens == 100
        assert config.temperature == 0.3
        
        # 测试本地模型配置
        config = selector._get_model_config("step-flash")
        assert config.model_name == "step-flash"
        assert config.endpoint == "http://localhost:8046"
        assert config.is_local is True
    
    def test_unknown_model_config(self, selector):
        """测试未知模型配置"""
        config = selector._get_model_config("unknown-model")
        assert config.model_name == "unknown-model"
        assert config.is_local is True
        assert config.expected_quality == 0.80


class TestModelStats:
    """ModelStats 数据类测试"""
    
    def test_default_values(self):
        """测试默认值"""
        stats = ModelStats()
        assert stats.total_requests == 0
        assert stats.avg_latency_ms == 0.0
        assert stats.avg_quality_score == 0.0
        assert stats.success_rate == 1.0
        assert stats.total_tokens_used == 0
        assert stats._latency_sum == 0.0
        assert stats._quality_sum == 0.0
        assert stats._successful_requests == 0


class TestEnums:
    """枚举类型测试"""
    
    def test_memory_type_values(self):
        """测试 MemoryType 枚举值"""
        assert MemoryType.TEXT.value == "text"
        assert MemoryType.CODE.value == "code"
        assert MemoryType.MULTIMODAL.value == "multimodal"
        assert MemoryType.LONG_TEXT.value == "long_text"
    
    def test_quality_level_values(self):
        """测试 QualityLevel 枚举值"""
        assert QualityLevel.LOW.value == "low"
        assert QualityLevel.STANDARD.value == "standard"
        assert QualityLevel.HIGH.value == "high"

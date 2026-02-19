"""
模型选择器属性测试

使用 Hypothesis 进行基于属性的测试，验证模型选择规则的一致性。
"""

import pytest
from hypothesis import given, settings, strategies as st

from llm_compression.model_selector import (
    ModelSelector,
    MemoryType,
    QualityLevel,
    ModelConfig
)


# 测试策略
memory_types = st.sampled_from([
    MemoryType.TEXT,
    MemoryType.CODE,
    MemoryType.MULTIMODAL,
    MemoryType.LONG_TEXT
])

quality_levels = st.sampled_from([
    QualityLevel.LOW,
    QualityLevel.STANDARD,
    QualityLevel.HIGH
])

text_lengths = st.integers(min_value=0, max_value=10000)


class TestModelSelectionRuleConsistency:
    """
    Property 8: 模型选择规则一致性
    
    验证模型选择器根据预定义规则返回合适的模型配置
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        memory_type=memory_types,
        text_length=text_lengths,
        quality_level=quality_levels
    )
    def test_model_selection_returns_valid_config(
        self,
        memory_type,
        text_length,
        quality_level
    ):
        """
        Feature: llm-compression-integration, Property 8: 模型选择规则一致性
        
        For any 记忆类型和文本长度，模型选择器应该返回有效的模型配置
        """
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={
                "step-flash": "http://localhost:8046",
                "minicpm-o": "http://localhost:8047",
                "stable-diffcoder": "http://localhost:8048",
                "intern-s1-pro": "http://localhost:8049"
            },
            prefer_local=True
        )
        
        config = selector.select_model(
            memory_type=memory_type,
            text_length=text_length,
            quality_requirement=quality_level
        )
        
        # 验证返回的配置有效
        assert isinstance(config, ModelConfig)
        assert config.model_name is not None
        assert len(config.model_name) > 0
        assert config.max_tokens >= 0
        assert 0 <= config.temperature <= 1
        assert config.expected_latency_ms >= 0
        assert 0 <= config.expected_quality <= 1
    
    @settings(max_examples=100, deadline=None)
    @given(
        text_length=st.integers(min_value=0, max_value=499)
    )
    def test_short_text_selection(self, text_length):
        """
        Feature: llm-compression-integration, Property 8: 模型选择规则一致性
        
        文本 < 500 字 → Step 3.5 Flash 或云端 API
        """
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={
                "step-flash": "http://localhost:8046",
                "minicpm-o": "http://localhost:8047",
                "stable-diffcoder": "http://localhost:8048",
                "intern-s1-pro": "http://localhost:8049"
            },
            prefer_local=True
        )
        
        config = selector.select_model(
            memory_type=MemoryType.TEXT,
            text_length=text_length,
            quality_requirement=QualityLevel.STANDARD
        )
        
        # 应该选择 step-flash（本地优先）或 cloud-api
        assert config.model_name in ["step-flash", "cloud-api"]
    
    @settings(max_examples=100, deadline=None)
    @given(
        text_length=st.integers(min_value=500, max_value=10000)
    )
    def test_long_text_selection(self, text_length):
        """
        Feature: llm-compression-integration, Property 8: 模型选择规则一致性
        
        长文本 > 500 字 → Intern-S1-Pro 或云端 API
        """
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={
                "step-flash": "http://localhost:8046",
                "minicpm-o": "http://localhost:8047",
                "stable-diffcoder": "http://localhost:8048",
                "intern-s1-pro": "http://localhost:8049"
            },
            prefer_local=True
        )
        
        config = selector.select_model(
            memory_type=MemoryType.LONG_TEXT,
            text_length=text_length,
            quality_requirement=QualityLevel.STANDARD
        )
        
        # 应该选择 intern-s1-pro（本地优先）或 cloud-api
        assert config.model_name in ["intern-s1-pro", "cloud-api"]
    
    @settings(max_examples=100, deadline=None)
    @given(
        text_length=text_lengths
    )
    def test_code_memory_selection(self, text_length):
        """
        Feature: llm-compression-integration, Property 8: 模型选择规则一致性
        
        代码记忆 → Stable-DiffCoder 或云端 API
        """
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={
                "step-flash": "http://localhost:8046",
                "minicpm-o": "http://localhost:8047",
                "stable-diffcoder": "http://localhost:8048",
                "intern-s1-pro": "http://localhost:8049"
            },
            prefer_local=True
        )
        
        config = selector.select_model(
            memory_type=MemoryType.CODE,
            text_length=text_length,
            quality_requirement=QualityLevel.STANDARD
        )
        
        # 应该选择 stable-diffcoder（本地优先）或 cloud-api
        assert config.model_name in ["stable-diffcoder", "cloud-api"]
    
    @settings(max_examples=100, deadline=None)
    @given(
        text_length=text_lengths
    )
    def test_multimodal_memory_selection(self, text_length):
        """
        Feature: llm-compression-integration, Property 8: 模型选择规则一致性
        
        多模态记忆 → MiniCPM-o 4.5 或云端 API
        """
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={
                "step-flash": "http://localhost:8046",
                "minicpm-o": "http://localhost:8047",
                "stable-diffcoder": "http://localhost:8048",
                "intern-s1-pro": "http://localhost:8049"
            },
            prefer_local=True
        )
        
        config = selector.select_model(
            memory_type=MemoryType.MULTIMODAL,
            text_length=text_length,
            quality_requirement=QualityLevel.STANDARD
        )
        
        # 应该选择 minicpm-o（本地优先）或 cloud-api
        assert config.model_name in ["minicpm-o", "cloud-api"]
    
    @settings(max_examples=100, deadline=None)
    @given(
        memory_type=memory_types,
        text_length=text_lengths
    )
    def test_high_quality_prefers_cloud(self, memory_type, text_length):
        """
        Feature: llm-compression-integration, Property 8: 模型选择规则一致性
        
        高质量要求 → 优先云端 API（Claude/GPT）
        """
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={
                "step-flash": "http://localhost:8046",
                "minicpm-o": "http://localhost:8047",
                "stable-diffcoder": "http://localhost:8048",
                "intern-s1-pro": "http://localhost:8049"
            },
            prefer_local=True
        )
        
        config = selector.select_model(
            memory_type=memory_type,
            text_length=text_length,
            quality_requirement=QualityLevel.HIGH
        )
        
        # 高质量要求应该优先选择云端 API
        assert config.model_name == "cloud-api"
        assert not config.is_local
    
    @settings(max_examples=100, deadline=None)
    @given(
        memory_type=memory_types,
        text_length=text_lengths,
        quality_level=quality_levels
    )
    def test_cloud_only_always_returns_cloud(
        self,
        memory_type,
        text_length,
        quality_level
    ):
        """
        Feature: llm-compression-integration, Property 8: 模型选择规则一致性
        
        当没有本地模型时，应该总是返回云端 API
        """
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={},
            prefer_local=False
        )
        
        config = selector.select_model(
            memory_type=memory_type,
            text_length=text_length,
            quality_requirement=quality_level
        )
        
        # 没有本地模型时应该使用云端 API
        assert config.model_name == "cloud-api"
        assert config.endpoint == "http://localhost:8045"
    
    @settings(max_examples=100, deadline=None)
    @given(
        memory_type=memory_types,
        text_length=text_lengths,
        quality_level=quality_levels,
        manual_model=st.sampled_from([
            "cloud-api",
            "step-flash",
            "minicpm-o",
            "stable-diffcoder",
            "intern-s1-pro"
        ])
    )
    def test_manual_model_override(
        self,
        memory_type,
        text_length,
        quality_level,
        manual_model
    ):
        """
        Feature: llm-compression-integration, Property 8: 模型选择规则一致性
        
        手动指定模型应该覆盖自动选择
        """
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={
                "step-flash": "http://localhost:8046",
                "minicpm-o": "http://localhost:8047",
                "stable-diffcoder": "http://localhost:8048",
                "intern-s1-pro": "http://localhost:8049"
            },
            prefer_local=True
        )
        
        config = selector.select_model(
            memory_type=memory_type,
            text_length=text_length,
            quality_requirement=quality_level,
            manual_model=manual_model
        )
        
        # 应该使用手动指定的模型
        assert config.model_name == manual_model
    
    @settings(max_examples=100, deadline=None)
    @given(
        memory_type=memory_types,
        text_length=text_lengths,
        quality_level=quality_levels
    )
    def test_selection_is_deterministic(
        self,
        memory_type,
        text_length,
        quality_level
    ):
        """
        Feature: llm-compression-integration, Property 8: 模型选择规则一致性
        
        相同输入应该返回相同的模型选择（确定性）
        """
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={
                "step-flash": "http://localhost:8046",
                "minicpm-o": "http://localhost:8047",
                "stable-diffcoder": "http://localhost:8048",
                "intern-s1-pro": "http://localhost:8049"
            },
            prefer_local=True
        )
        
        config1 = selector.select_model(
            memory_type=memory_type,
            text_length=text_length,
            quality_requirement=quality_level
        )
        
        config2 = selector.select_model(
            memory_type=memory_type,
            text_length=text_length,
            quality_requirement=quality_level
        )
        
        # 相同输入应该返回相同的模型
        assert config1.model_name == config2.model_name
        assert config1.endpoint == config2.endpoint
        assert config1.is_local == config2.is_local


class TestModelStatistics:
    """测试模型统计记录"""
    
    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=None)
    @given(
        latency=st.floats(min_value=10.0, max_value=5000.0),
        quality=st.floats(min_value=0.0, max_value=1.0),
        tokens=st.integers(min_value=1, max_value=1000)
    )
    async def test_usage_recording(self, latency, quality, tokens):
        """
        Feature: llm-compression-integration, Property 8: 模型选择规则一致性
        
        模型使用统计应该正确记录
        """
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={"step-flash": "http://localhost:8046"},
            prefer_local=True
        )
        
        model_name = "test-model"
        
        # 记录使用
        await selector.record_usage(
            model_name=model_name,
            latency_ms=latency,
            quality_score=quality,
            tokens_used=tokens,
            success=True
        )
        
        # 验证统计信息
        stats = selector.get_model_stats()
        assert model_name in stats
        
        model_stats = stats[model_name]
        assert model_stats.total_requests == 1
        assert model_stats._successful_requests == 1
        assert model_stats.avg_latency_ms == latency
        assert model_stats.avg_quality_score == quality
        assert model_stats.total_tokens_used == tokens
        assert model_stats.success_rate == 1.0
    
    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=None)
    @given(
        latencies=st.lists(
            st.floats(min_value=10.0, max_value=5000.0),
            min_size=2,
            max_size=10
        ),
        qualities=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=2,
            max_size=10
        )
    )
    async def test_average_calculation(self, latencies, qualities):
        """
        Feature: llm-compression-integration, Property 8: 模型选择规则一致性
        
        平均值计算应该正确
        """
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={"step-flash": "http://localhost:8046"},
            prefer_local=True
        )
        
        # 确保列表长度相同
        min_len = min(len(latencies), len(qualities))
        latencies = latencies[:min_len]
        qualities = qualities[:min_len]
        
        model_name = "test-model"
        
        # 记录多次使用
        for lat, qual in zip(latencies, qualities):
            await selector.record_usage(
                model_name=model_name,
                latency_ms=lat,
                quality_score=qual,
                tokens_used=100,
                success=True
            )
        
        # 验证平均值
        stats = selector.get_model_stats()
        model_stats = stats[model_name]
        
        expected_avg_latency = sum(latencies) / len(latencies)
        expected_avg_quality = sum(qualities) / len(qualities)
        
        assert abs(model_stats.avg_latency_ms - expected_avg_latency) < 0.01
        assert abs(model_stats.avg_quality_score - expected_avg_quality) < 0.01
        assert model_stats.total_requests == len(latencies)
        assert model_stats.success_rate == 1.0


class TestQualityMonitoring:
    """测试质量监控和建议"""
    
    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=None)
    @given(
        quality=st.floats(min_value=0.0, max_value=0.84)
    )
    async def test_low_quality_suggests_switch(self, quality):
        """
        Feature: llm-compression-integration, Property 26: 模型性能对比（部分）
        
        当质量低于阈值时，应该建议切换到更强大的模型
        """
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={"step-flash": "http://localhost:8046"},
            prefer_local=True,
            quality_threshold=0.85
        )
        
        model_name = "step-flash"
        
        # 记录低质量使用
        await selector.record_usage(
            model_name=model_name,
            latency_ms=500.0,
            quality_score=quality,
            tokens_used=100,
            success=True
        )
        
        # 应该建议切换
        suggestion = selector.suggest_model_switch(model_name)
        assert suggestion is not None
        assert suggestion == "cloud-api"
    
    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=None)
    @given(
        quality=st.floats(min_value=0.85, max_value=1.0)
    )
    async def test_high_quality_no_switch(self, quality):
        """
        Feature: llm-compression-integration, Property 26: 模型性能对比（部分）
        
        当质量达标时，不应该建议切换
        """
        selector = ModelSelector(
            cloud_endpoint="http://localhost:8045",
            local_endpoints={"step-flash": "http://localhost:8046"},
            prefer_local=True,
            quality_threshold=0.85
        )
        
        model_name = "step-flash"
        
        # 记录高质量使用
        await selector.record_usage(
            model_name=model_name,
            latency_ms=500.0,
            quality_score=quality,
            tokens_used=100,
            success=True
        )
        
        # 不应该建议切换
        suggestion = selector.suggest_model_switch(model_name)
        assert suggestion is None

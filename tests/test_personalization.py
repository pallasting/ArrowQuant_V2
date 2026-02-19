"""
PersonalizationEngine 单元测试
"""

import pytest
from datetime import datetime

from llm_compression import PersonalizationEngine, UserProfile


@pytest.fixture
def engine():
    """创建PersonalizationEngine实例"""
    return PersonalizationEngine(
        user_id="test_user",
        learning_rate=0.1,
        decay_rate=0.01
    )


def test_initialization(engine):
    """测试初始化"""
    assert engine.user_id == "test_user"
    assert engine.learning_rate == 0.1
    assert engine.profile.user_id == "test_user"
    assert engine.profile.total_interactions == 0


def test_track_preference(engine):
    """测试偏好追踪"""
    engine.track_preference("python", sentiment=0.8)
    
    assert "python" in engine.profile.topic_interests
    assert engine.profile.topic_interests["python"] > 0.5
    assert engine.profile.total_interactions == 1
    assert len(engine.profile.preference_history) == 1


def test_multiple_preferences(engine):
    """测试多个偏好"""
    engine.track_preference("python", sentiment=0.9)
    engine.track_preference("javascript", sentiment=0.3)
    engine.track_preference("rust", sentiment=0.7)
    
    assert len(engine.profile.topic_interests) == 3
    assert engine.profile.total_interactions == 3


def test_preference_learning(engine):
    """测试偏好学习"""
    # 多次正面反馈
    for _ in range(5):
        engine.track_preference("ai", sentiment=0.9)
    
    interest = engine.get_topic_interest("ai")
    assert interest > 0.6  # 应该学到较高兴趣（考虑衰减）


def test_get_topic_interest(engine):
    """测试获取话题兴趣"""
    engine.track_preference("ml", sentiment=0.8)
    
    assert engine.get_topic_interest("ml") > 0.5
    assert engine.get_topic_interest("unknown") == 0.5  # 默认中性


def test_get_top_interests(engine):
    """测试获取最感兴趣话题"""
    engine.track_preference("topic1", sentiment=0.9)
    engine.track_preference("topic2", sentiment=0.7)
    engine.track_preference("topic3", sentiment=0.5)
    
    top = engine.get_top_interests(n=2)
    
    assert len(top) == 2
    assert top[0][0] == "topic1"  # 最高兴趣
    assert top[0][1] > top[1][1]  # 降序


def test_update_style(engine):
    """测试风格更新"""
    engine.update_style("formality", 0.8)
    
    assert engine.style_dimensions["formality"] > 0.5


def test_style_clipping(engine):
    """测试风格值裁剪"""
    # 尝试设置超出范围的值
    engine.update_style("formality", 1.5, weight=10.0)
    
    assert engine.style_dimensions["formality"] <= 1.0


def test_personalize_response_friendly(engine):
    """测试友好风格个性化"""
    # 设置高友好度
    engine.style_dimensions["friendliness"] = 0.8
    engine.profile.total_interactions = 10
    
    response = engine.personalize_response("Hello!")
    
    assert "😊" in response


def test_personalize_response_formal(engine):
    """测试正式风格个性化"""
    engine.style_dimensions["formality"] = 0.8
    
    response = engine.personalize_response("你好")
    
    assert "您" in response


def test_personalize_response_casual(engine):
    """测试随意风格个性化"""
    engine.style_dimensions["formality"] = 0.2
    
    response = engine.personalize_response("您好")
    
    assert "你" in response


def test_get_profile(engine):
    """测试获取用户画像"""
    engine.track_preference("test", sentiment=0.8)
    
    profile = engine.get_profile()
    
    assert isinstance(profile, UserProfile)
    assert profile.user_id == "test_user"
    assert profile.total_interactions == 1


def test_get_style_summary(engine):
    """测试获取风格摘要"""
    summary = engine.get_style_summary()
    
    assert "formality" in summary
    assert "verbosity" in summary
    assert "technicality" in summary
    assert "friendliness" in summary


def test_preference_decay(engine):
    """测试偏好衰减"""
    engine.track_preference("topic", sentiment=0.9)
    initial = engine.get_topic_interest("topic")
    
    # 多次衰减
    for _ in range(10):
        engine._decay_preferences()
    
    final = engine.get_topic_interest("topic")
    
    # 应该向0.5衰减
    assert abs(final - 0.5) < abs(initial - 0.5)


def test_reset(engine):
    """测试重置"""
    engine.track_preference("test", sentiment=0.9)
    engine.update_style("formality", 0.9)
    
    engine.reset()
    
    assert engine.profile.total_interactions == 0
    assert len(engine.profile.topic_interests) == 0
    assert engine.style_dimensions["formality"] == 0.5


def test_preference_history(engine):
    """测试偏好历史"""
    before = datetime.now()
    engine.track_preference("test", sentiment=0.8)
    after = datetime.now()
    
    history = engine.profile.preference_history
    assert len(history) == 1
    
    topic, sentiment, timestamp = history[0]
    assert topic == "test"
    assert sentiment == 0.8
    assert before <= timestamp <= after


def test_invalid_style_dimension(engine):
    """测试无效风格维度"""
    engine.update_style("invalid_dimension", 0.8)
    
    # 不应该添加新维度
    assert "invalid_dimension" not in engine.style_dimensions

"""
个性化引擎

追踪用户偏好、话题兴趣、交互风格，实现个性化回复
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    topic_interests: Dict[str, float] = field(default_factory=dict)  # 话题 -> 兴趣度
    interaction_style: Dict[str, float] = field(default_factory=dict)  # 风格特征
    preference_history: List[tuple] = field(default_factory=list)  # (话题, 情感, 时间戳)
    total_interactions: int = 0


class PersonalizationEngine:
    """个性化引擎"""
    
    def __init__(
        self,
        user_id: str = "default_user",
        learning_rate: float = 0.1,
        decay_rate: float = 0.01
    ):
        self.user_id = user_id
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.profile = UserProfile(user_id=user_id)
        
        # 风格维度
        self.style_dimensions = {
            "formality": 0.5,      # 正式 vs 随意
            "verbosity": 0.5,      # 简洁 vs 详细
            "technicality": 0.5,   # 技术 vs 通俗
            "friendliness": 0.7    # 友好度
        }
    
    def track_preference(
        self,
        topic: str,
        sentiment: float = 0.5,
        weight: float = 1.0
    ):
        """
        追踪用户偏好
        
        Args:
            topic: 话题关键词
            sentiment: 情感分数 (0-1, 0.5为中性)
            weight: 权重
        """
        # 更新话题兴趣
        current = self.profile.topic_interests.get(topic, 0.5)
        new_interest = current + self.learning_rate * (sentiment - current) * weight
        self.profile.topic_interests[topic] = np.clip(new_interest, 0.0, 1.0)
        
        # 记录历史
        from datetime import datetime
        self.profile.preference_history.append((topic, sentiment, datetime.now()))
        self.profile.total_interactions += 1
        
        # 衰减旧偏好
        self._decay_preferences()
    
    def update_style(
        self,
        dimension: str,
        value: float,
        weight: float = 1.0
    ):
        """
        更新交互风格
        
        Args:
            dimension: 风格维度 (formality/verbosity/technicality/friendliness)
            value: 目标值 (0-1)
            weight: 权重
        """
        if dimension not in self.style_dimensions:
            return
        
        current = self.style_dimensions[dimension]
        new_value = current + self.learning_rate * (value - current) * weight
        self.style_dimensions[dimension] = np.clip(new_value, 0.0, 1.0)
    
    def get_topic_interest(self, topic: str) -> float:
        """获取话题兴趣度"""
        return self.profile.topic_interests.get(topic, 0.5)
    
    def get_top_interests(self, n: int = 5) -> List[tuple]:
        """获取最感兴趣的话题"""
        sorted_topics = sorted(
            self.profile.topic_interests.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_topics[:n]
    
    def personalize_response(
        self,
        response: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        个性化回复（简单版本）
        
        Args:
            response: 原始回复
            context: 上下文信息
            
        Returns:
            个性化后的回复
        """
        # 根据风格调整
        formality = self.style_dimensions["formality"]
        friendliness = self.style_dimensions["friendliness"]
        
        # 添加个性化前缀（基于友好度）
        if friendliness > 0.7 and self.profile.total_interactions > 5:
            response = f"😊 {response}"
        
        # 简单的风格调整（实际应用中可以用LLM重写）
        if formality < 0.3:
            # 更随意
            response = response.replace("您", "你")
        elif formality > 0.7:
            # 更正式
            response = response.replace("你", "您")
        
        return response
    
    def get_profile(self) -> UserProfile:
        """获取用户画像"""
        return self.profile
    
    def get_style_summary(self) -> Dict[str, float]:
        """获取风格摘要"""
        return self.style_dimensions.copy()
    
    def _decay_preferences(self):
        """衰减旧偏好（自然遗忘）"""
        for topic in self.profile.topic_interests:
            current = self.profile.topic_interests[topic]
            # 向中性值(0.5)衰减
            self.profile.topic_interests[topic] = (
                current + self.decay_rate * (0.5 - current)
            )
    
    def reset(self):
        """重置用户画像"""
        self.profile = UserProfile(user_id=self.user_id)
        self.style_dimensions = {
            "formality": 0.5,
            "verbosity": 0.5,
            "technicality": 0.5,
            "friendliness": 0.7
        }

"""
对话Agent

整合认知循环、对话记忆、个性化引擎，实现持续学习的对话系统
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from .cognitive_loop import CognitiveLoop
from .conversation_memory import ConversationMemory, ConversationTurn
from .personalization import PersonalizationEngine
from .compressor import LLMCompressor
from .llm_client import LLMClient


@dataclass
class AgentResponse:
    """Agent响应"""
    message: str
    quality_score: float
    memories_used: List[str]
    learning_occurred: bool
    personalized: bool


class ConversationalAgent:
    """对话Agent"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        compressor: LLMCompressor,
        cognitive_loop: Optional[CognitiveLoop] = None,
        user_id: str = "default_user",
        enable_personalization: bool = True
    ):
        self.llm_client = llm_client
        self.compressor = compressor
        
        # 创建或使用认知循环
        self.cognitive_loop = cognitive_loop or CognitiveLoop()
        
        # 对话记忆
        self.conversation_memory = ConversationMemory(
            compressor=compressor,
            cognitive_loop=self.cognitive_loop
        )
        
        # 个性化引擎
        self.enable_personalization = enable_personalization
        if enable_personalization:
            self.personalization = PersonalizationEngine(user_id=user_id)
        else:
            self.personalization = None
    
    async def chat(
        self,
        user_message: str,
        max_context_turns: int = 5,
        system_prompt: Optional[str] = None
    ) -> AgentResponse:
        """
        处理用户消息
        
        Args:
            user_message: 用户消息
            max_context_turns: 最大上下文轮次
            
        Returns:
            Agent响应
        """
        # 1. 检索相关对话历史
        context_turns = await self.conversation_memory.get_context(
            query=user_message,
            max_turns=max_context_turns
        )
        
        # 2. 构建上下文
        context = self._build_context(context_turns)
        
        # 3. 生成回复
        query_embedding = await self.compressor.get_embedding(user_message)
        result = await self.cognitive_loop.process(
            query=user_message,
            query_embedding=query_embedding,
            max_memories=5,
            system_prompt=system_prompt
        )
        
        # 4. 个性化回复
        response_text = result.output
        personalized = False
        
        if self.enable_personalization and self.personalization:
            response_text = self.personalization.personalize_response(
                response_text,
                context={"user_message": user_message}
            )
            personalized = True
        
        # 5. 存储对话轮次
        await self.conversation_memory.add_turn(
            user_message=user_message,
            agent_reply=response_text
        )
        
        # 6. 更新个性化（基于质量反馈）
        if self.enable_personalization and self.personalization:
            self._update_personalization(
                user_message=user_message,
                quality=result.quality.overall
            )
        
        return AgentResponse(
            message=response_text,
            quality_score=result.quality.overall,
            memories_used=result.memories_used,
            learning_occurred=result.learning_occurred,
            personalized=personalized
        )
    
    def _build_context(self, turns: List[ConversationTurn]) -> str:
        """构建上下文字符串"""
        if not turns:
            return ""
        
        context_parts = []
        for turn in turns:
            context_parts.append(f"User: {turn.user_message}")
            context_parts.append(f"Agent: {turn.agent_reply}")
        
        return "\n".join(context_parts)
    
    def _update_personalization(
        self,
        user_message: str,
        quality: float
    ):
        """更新个性化（简单版本）"""
        if not self.personalization:
            return
        
        # 提取话题（简单：取前3个词）
        words = user_message.lower().split()[:3]
        topic = " ".join(words) if words else "general"
        
        # 基于质量更新偏好
        sentiment = quality  # 质量高 = 正面反馈
        self.personalization.track_preference(topic, sentiment=sentiment)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = self.conversation_memory.get_stats()
        
        if self.personalization:
            stats["user_profile"] = {
                "total_interactions": self.personalization.profile.total_interactions,
                "top_interests": self.personalization.get_top_interests(n=3),
                "style": self.personalization.get_style_summary()
            }
        
        return stats
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_memory.clear_history()
        if self.personalization:
            self.personalization.reset()

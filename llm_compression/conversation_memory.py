"""
对话记忆管理模块

管理对话历史的压缩、存储和检索，支持：
- 对话轮次压缩
- 上下文检索
- 时间戳追踪
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import numpy as np

from .compressor import LLMCompressor
from .cognitive_loop import CognitiveLoop
from .memory_primitive import MemoryPrimitive


@dataclass
class ConversationTurn:
    """对话轮次"""
    turn_id: str
    user_message: str
    agent_reply: str
    timestamp: datetime
    memory_id: Optional[str] = None


class ConversationMemory:
    """对话记忆管理器"""
    
    def __init__(
        self,
        compressor: LLMCompressor,
        cognitive_loop: CognitiveLoop,
        max_history: int = 100
    ):
        self.compressor = compressor
        self.cognitive_loop = cognitive_loop
        self.max_history = max_history
        self.turns: List[ConversationTurn] = []
        self._turn_counter = 0
    
    async def add_turn(
        self,
        user_message: str,
        agent_reply: str
    ) -> str:
        """
        添加对话轮次
        
        Args:
            user_message: 用户消息
            agent_reply: Agent回复
            
        Returns:
            记忆ID
        """
        # 生成turn ID
        turn_id = f"turn_{self._turn_counter}"
        self._turn_counter += 1
        
        # 压缩对话轮次
        turn_text = f"User: {user_message}\nAgent: {agent_reply}"
        compressed = await self.compressor.compress(turn_text)
        
        # 创建记忆单元
        memory = MemoryPrimitive(
            id=compressed.memory_id,
            content=compressed,
            embedding=np.array(compressed.embedding)
        )
        
        # 添加到认知循环
        self.cognitive_loop.add_memory(memory)
        
        # 记录轮次
        turn = ConversationTurn(
            turn_id=turn_id,
            user_message=user_message,
            agent_reply=agent_reply,
            timestamp=datetime.now(),
            memory_id=compressed.memory_id
        )
        self.turns.append(turn)
        
        # 限制历史长度
        if len(self.turns) > self.max_history:
            self.turns.pop(0)
        
        return compressed.memory_id
    
    async def get_context(
        self,
        query: str,
        max_turns: int = 5
    ) -> List[ConversationTurn]:
        """
        检索相关对话历史
        
        Args:
            query: 查询文本
            max_turns: 最大返回轮次数
            
        Returns:
            相关对话轮次列表
        """
        if not self.turns:
            return []
        
        # 获取查询embedding
        query_embedding = await self.compressor.get_embedding(query)
        
        # 使用认知循环检索
        result = await self.cognitive_loop.process(
            query=query,
            query_embedding=query_embedding,
            max_memories=max_turns
        )
        
        # 匹配记忆ID到对话轮次
        memory_ids = set(result.memories_used)
        relevant_turns = [
            turn for turn in self.turns
            if turn.memory_id in memory_ids
        ]
        
        # 按时间排序
        relevant_turns.sort(key=lambda t: t.timestamp)
        
        return relevant_turns[-max_turns:]
    
    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """获取最近N轮对话"""
        return self.turns[-n:] if self.turns else []
    
    def get_all_turns(self) -> List[ConversationTurn]:
        """获取所有对话轮次"""
        return self.turns.copy()
    
    def clear_history(self):
        """清空对话历史"""
        self.turns.clear()
        self._turn_counter = 0
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_turns": len(self.turns),
            "memory_count": len(self.cognitive_loop.memory_network),
            "connection_count": sum(
                len(mem.connections)
                for mem in self.cognitive_loop.memory_network.values()
            ) // 2,  # 双向连接
            "avg_connections": sum(
                len(mem.connections)
                for mem in self.cognitive_loop.memory_network.values()
            ) / max(len(self.cognitive_loop.memory_network), 1)
        }

"""
Cognitive Loop - Self-Organizing Cognitive System (Task 42)

完整的认知闭环，整合所有Phase 2.0组件：
- 记忆检索 (NetworkNavigator)
- 输出生成 (MultiModalExpressor)
- 质量评估 (InternalFeedbackSystem)
- 连接学习 (ConnectionLearner)
- 自我纠正 (Feedback Loop)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from llm_compression.memory_primitive import MemoryPrimitive
from llm_compression.connection_learner import ConnectionLearner
from llm_compression.network_navigator import NetworkNavigator, ActivationResult
from llm_compression.expression_layer import MultiModalExpressor, ExpressionResult
from llm_compression.internal_feedback import InternalFeedbackSystem, QualityScore, Correction, CorrectionType
from llm_compression.compressor import CompressedMemory


@dataclass
class CognitiveResult:
    """认知循环结果"""
    output: str
    quality: QualityScore
    memories_used: List[str]
    corrections_applied: int
    learning_occurred: bool


class CognitiveLoop:
    """
    自组织认知闭环系统
    
    完整的感知-行动-学习循环：
    1. 检索相关记忆 (Navigation)
    2. 生成输出 (Expression)
    3. 评估质量 (Reflection)
    4. 自我纠正 (Correction)
    5. 学习连接 (Learning)
    """
    
    def __init__(
        self,
        expressor: Optional[MultiModalExpressor] = None,
        feedback: Optional[InternalFeedbackSystem] = None,
        learner: Optional[ConnectionLearner] = None,
        navigator: Optional[NetworkNavigator] = None,
        quality_threshold: float = 0.85,
        max_corrections: int = 2,
        learning_rate: float = 0.1
    ):
        """
        初始化认知闭环
        
        Args:
            expressor: 表达层（可选，用于依赖注入）
            feedback: 反馈系统（可选，用于依赖注入）
            learner: 连接学习器（可选，用于依赖注入）
            navigator: 网络导航器（可选，用于依赖注入）
            quality_threshold: 质量阈值
            max_corrections: 最大纠正次数
            learning_rate: 学习率
        """
        self.memory_network: Dict[str, MemoryPrimitive] = {}
        self.learner = learner or ConnectionLearner()
        self.navigator = navigator or NetworkNavigator()
        self.expressor = expressor
        self.feedback = feedback
        
        self.quality_threshold = quality_threshold
        self.max_corrections = max_corrections
        self.learning_rate = learning_rate
    
    async def process(
        self,
        query: str,
        query_embedding: np.ndarray,
        max_memories: int = 5
    ) -> CognitiveResult:
        """
        完整认知循环处理
        
        Args:
            query: 查询文本
            query_embedding: 查询向量
            max_memories: 最大检索记忆数
            
        Returns:
            CognitiveResult: 认知结果
        """
        corrections_applied = 0
        
        # 1. 检索相关记忆 (Navigation)
        retrieval = self.navigator.retrieve(
            query_embedding=query_embedding,
            memory_network=self.memory_network,
            max_results=max_memories
        )
        
        # 2. 生成初始输出 (Expression)
        output = await self._generate_output(query, retrieval)
        
        # 3. 评估质量 (Reflection)
        quality = await self.feedback.evaluate(
            output.content,
            query,
            [m.content for m in retrieval.memories]
        )
        
        # 4. 自我纠正循环 (Correction)
        while quality.overall < self.quality_threshold and corrections_applied < self.max_corrections:
            correction = self.feedback.suggest_correction(quality)
            output = await self._apply_correction(correction, query, retrieval)
            
            # 重新评估
            quality = await self.feedback.evaluate(
                output.content,
                query,
                [m.content for m in retrieval.memories]
            )
            corrections_applied += 1
        
        # 5. 学习连接 (Learning)
        learning_occurred = self._learn_from_interaction(
            retrieval.memories,
            quality
        )
        
        return CognitiveResult(
            output=output.content,
            quality=quality,
            memories_used=[m.id for m in retrieval.memories],
            corrections_applied=corrections_applied,
            learning_occurred=learning_occurred
        )
    
    async def _generate_output(
        self,
        query: str,
        retrieval: ActivationResult
    ) -> ExpressionResult:
        """生成输出"""
        # 即使没有记忆，也让LLM生成回复
        return await self.expressor.express_text(
            memories=[m.content for m in retrieval.memories] if retrieval.memories else [],
            query=query
        )
    
    async def _apply_correction(
        self,
        correction: Correction,
        query: str,
        retrieval: ActivationResult
    ) -> ExpressionResult:
        """应用纠正策略"""
        if correction.type == CorrectionType.SUPPLEMENT:
            # 补充：检索更多记忆
            extended_retrieval = self.navigator.retrieve(
                query_embedding=retrieval.memories[0].embedding,
                memory_network=self.memory_network,
                max_results=len(retrieval.memories) + 3
            )
            return await self._generate_output(query, extended_retrieval)
        
        elif correction.type == CorrectionType.RECTIFY:
            # 纠正：重新生成（带约束）
            return await self.expressor.express_text(
                memories=[m.content for m in retrieval.memories],
                query=f"{query}\n[Constraint: Focus on accuracy and factual correctness]"
            )
        
        elif correction.type == CorrectionType.RESTRUCTURE:
            # 重构：重新生成（带结构）
            return await self.expressor.express_text(
                memories=[m.content for m in retrieval.memories],
                query=f"{query}\n[Constraint: Provide clear structure and logical flow]"
            )
        
        return await self._generate_output(query, retrieval)
    
    def _learn_from_interaction(
        self,
        memories: List[MemoryPrimitive],
        quality: QualityScore
    ) -> bool:
        """
        从交互中学习
        
        Hebbian学习：共同激活的记忆强化连接
        成功记录：根据质量更新记忆成功率
        
        Returns:
            bool: 是否发生学习
        """
        if not memories:
            return False
        
        learning_occurred = False
        
        # Hebbian学习：强化共同激活的记忆间连接
        for i, mem_a in enumerate(memories):
            for mem_b in memories[i+1:]:
                self.learner.hebbian_learning(
                    mem_a,
                    mem_b,
                    learning_rate=self.learning_rate
                )
                learning_occurred = True
        
        # 记录成功/失败
        success = quality.overall >= self.quality_threshold
        for memory in memories:
            memory.activate(1.0)  # 增加access_count
            if success:
                memory.record_success()
        
        return learning_occurred
    
    def add_memory(self, memory: MemoryPrimitive) -> None:
        """添加记忆到网络"""
        self.memory_network[memory.id] = memory
    
    def get_memory(self, memory_id: str) -> Optional[MemoryPrimitive]:
        """获取记忆"""
        return self.memory_network.get(memory_id)
    
    def get_network_stats(self) -> Dict:
        """获取网络统计信息"""
        if not self.memory_network:
            return {
                "total_memories": 0,
                "total_connections": 0,
                "avg_connections": 0.0,
                "avg_success_rate": 0.0
            }
        
        total_connections = sum(
            len(m.connections) for m in self.memory_network.values()
        )
        
        success_rates = [
            m.get_success_rate() for m in self.memory_network.values()
            if m.access_count > 0
        ]
        
        return {
            "total_memories": len(self.memory_network),
            "total_connections": total_connections,
            "avg_connections": total_connections / len(self.memory_network),
            "avg_success_rate": np.mean(success_rates) if success_rates else 0.0
        }

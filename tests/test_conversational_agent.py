"""
ConversationalAgent 单元测试
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from llm_compression import (
    ConversationalAgent,
    AgentResponse,
    LLMClient,
    LLMCompressor,
    CognitiveLoop,
    CompressedMemory,
    CompressionMetadata,
    CognitiveResult,
    QualityScore
)


@pytest.fixture
def mock_llm_client():
    """Mock LLM客户端"""
    client = MagicMock()
    return client


@pytest.fixture
def mock_compressor():
    """Mock压缩器"""
    compressor = MagicMock()
    
    async def mock_compress(text):
        metadata = CompressionMetadata(
            original_size=len(text),
            compressed_size=len(text),
            compression_ratio=1.0,
            model_used="mock",
            quality_score=0.9,
            compression_time_ms=10.0,
            compressed_at=datetime.now()
        )
        return CompressedMemory(
            memory_id=f"mem_{hash(text) % 1000}",
            summary_hash=f"hash_{hash(text) % 1000}",
            entities={},
            diff_data=text.encode(),
            embedding=[0.1] * 384,
            compression_metadata=metadata
        )
    
    compressor.compress = AsyncMock(side_effect=mock_compress)
    
    async def mock_embedding(text):
        emb = np.random.randn(384)
        return emb / np.linalg.norm(emb)
    
    compressor.get_embedding = AsyncMock(side_effect=mock_embedding)
    
    return compressor


@pytest.fixture
def mock_cognitive_loop():
    """Mock认知循环"""
    loop = MagicMock()
    loop.memory_network = {}
    loop.add_memory = MagicMock()
    
    async def mock_process(query, query_embedding, max_memories=5):
        return CognitiveResult(
            output=f"Response to: {query}",
            quality=QualityScore(
                overall=0.9,
                consistency=0.9,
                completeness=0.9,
                accuracy=0.9,
                coherence=0.9
            ),
            memories_used=["mem_1", "mem_2"],
            corrections_applied=0,
            learning_occurred=True
        )
    
    loop.process = AsyncMock(side_effect=mock_process)
    
    return loop


@pytest.fixture
def agent(mock_llm_client, mock_compressor, mock_cognitive_loop):
    """创建ConversationalAgent实例"""
    return ConversationalAgent(
        llm_client=mock_llm_client,
        compressor=mock_compressor,
        cognitive_loop=mock_cognitive_loop,
        user_id="test_user",
        enable_personalization=True
    )


@pytest.mark.asyncio
async def test_chat_basic(agent):
    """测试基本对话"""
    response = await agent.chat("Hello!")
    
    assert isinstance(response, AgentResponse)
    assert "Hello!" in response.message
    assert response.quality_score > 0.0
    assert response.learning_occurred


@pytest.mark.asyncio
async def test_chat_with_context(agent):
    """测试带上下文的对话"""
    # 第一轮
    await agent.chat("My name is Alice")
    
    # 第二轮（应该有上下文）
    response = await agent.chat("What's my name?")
    
    assert response.message is not None
    assert len(response.memories_used) > 0


@pytest.mark.asyncio
async def test_personalization(agent):
    """测试个性化"""
    response = await agent.chat("Hello!")
    
    assert response.personalized is True
    assert agent.personalization.profile.total_interactions > 0


@pytest.mark.asyncio
async def test_without_personalization(mock_llm_client, mock_compressor, mock_cognitive_loop):
    """测试禁用个性化"""
    agent = ConversationalAgent(
        llm_client=mock_llm_client,
        compressor=mock_compressor,
        cognitive_loop=mock_cognitive_loop,
        enable_personalization=False
    )
    
    response = await agent.chat("Hello!")
    
    assert response.personalized is False
    assert agent.personalization is None


@pytest.mark.asyncio
async def test_memory_storage(agent):
    """测试记忆存储"""
    await agent.chat("First message")
    await agent.chat("Second message")
    
    stats = agent.get_stats()
    
    assert stats["total_turns"] == 2


@pytest.mark.asyncio
async def test_get_stats(agent):
    """测试统计信息"""
    await agent.chat("Test message")
    
    stats = agent.get_stats()
    
    assert "total_turns" in stats
    assert "memory_count" in stats
    assert "user_profile" in stats


def test_clear_history(agent):
    """测试清空历史"""
    agent.conversation_memory.turns = [MagicMock()]
    agent.personalization.profile.total_interactions = 5
    
    agent.clear_history()
    
    assert len(agent.conversation_memory.turns) == 0
    assert agent.personalization.profile.total_interactions == 0


@pytest.mark.asyncio
async def test_quality_feedback(agent):
    """测试质量反馈"""
    # 高质量对话
    await agent.chat("Good question")
    
    # 检查个性化是否更新
    assert agent.personalization.profile.total_interactions > 0


@pytest.mark.asyncio
async def test_multiple_turns(agent):
    """测试多轮对话"""
    messages = ["Hello", "How are you?", "Tell me about AI"]
    
    for msg in messages:
        response = await agent.chat(msg)
        assert response.message is not None
    
    stats = agent.get_stats()
    assert stats["total_turns"] == 3


@pytest.mark.asyncio
async def test_context_building(agent):
    """测试上下文构建"""
    from llm_compression import ConversationTurn
    
    turns = [
        ConversationTurn("t1", "Hello", "Hi", datetime.now(), "mem_1"),
        ConversationTurn("t2", "How are you?", "Good", datetime.now(), "mem_2")
    ]
    
    context = agent._build_context(turns)
    
    assert "User: Hello" in context
    assert "Agent: Hi" in context


def test_empty_context(agent):
    """测试空上下文"""
    context = agent._build_context([])
    assert context == ""

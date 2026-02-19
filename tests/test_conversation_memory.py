"""
ConversationMemory 单元测试
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from llm_compression import (
    ConversationMemory,
    ConversationTurn,
    LLMCompressor,
    CognitiveLoop,
    CompressedMemory,
    CompressionMetadata
)


@pytest.fixture
def mock_compressor():
    """Mock压缩器"""
    compressor = MagicMock()
    
    # Mock compress
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
    
    # Mock get_embedding
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
    
    # Mock process
    from llm_compression import CognitiveResult, QualityScore, ExpressionResult
    
    async def mock_process(query, query_embedding, max_memories=5):
        return CognitiveResult(
            output="Mock response",
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
def conversation_memory(mock_compressor, mock_cognitive_loop):
    """创建ConversationMemory实例"""
    return ConversationMemory(
        compressor=mock_compressor,
        cognitive_loop=mock_cognitive_loop,
        max_history=10
    )


@pytest.mark.asyncio
async def test_add_turn(conversation_memory):
    """测试添加对话轮次"""
    memory_id = await conversation_memory.add_turn(
        user_message="Hello!",
        agent_reply="Hi there!"
    )
    
    assert memory_id is not None
    assert len(conversation_memory.turns) == 1
    assert conversation_memory.turns[0].user_message == "Hello!"
    assert conversation_memory.turns[0].agent_reply == "Hi there!"


@pytest.mark.asyncio
async def test_multiple_turns(conversation_memory):
    """测试多轮对话"""
    for i in range(5):
        await conversation_memory.add_turn(
            user_message=f"Message {i}",
            agent_reply=f"Reply {i}"
        )
    
    assert len(conversation_memory.turns) == 5
    assert conversation_memory._turn_counter == 5


@pytest.mark.asyncio
async def test_max_history_limit(conversation_memory):
    """测试历史长度限制"""
    # 添加超过max_history的轮次
    for i in range(15):
        await conversation_memory.add_turn(
            user_message=f"Message {i}",
            agent_reply=f"Reply {i}"
        )
    
    # 应该只保留最近10轮
    assert len(conversation_memory.turns) == 10
    assert conversation_memory.turns[0].user_message == "Message 5"


@pytest.mark.asyncio
async def test_get_context(conversation_memory):
    """测试上下文检索"""
    # 添加几轮对话
    await conversation_memory.add_turn("Hello", "Hi")
    await conversation_memory.add_turn("How are you?", "I'm good")
    await conversation_memory.add_turn("What's your name?", "I'm Agent")
    
    # 检索上下文
    context = await conversation_memory.get_context("greeting", max_turns=2)
    
    assert len(context) <= 2
    assert all(isinstance(turn, ConversationTurn) for turn in context)


@pytest.mark.asyncio
async def test_get_recent_turns(conversation_memory):
    """测试获取最近轮次"""
    # 添加对话
    for i in range(5):
        await conversation_memory.add_turn(f"Msg {i}", f"Reply {i}")
    
    recent = conversation_memory.get_recent_turns(3)
    
    assert len(recent) == 3
    assert recent[0].user_message == "Msg 2"
    assert recent[-1].user_message == "Msg 4"


def test_get_all_turns(conversation_memory):
    """测试获取所有轮次"""
    # 空历史
    assert conversation_memory.get_all_turns() == []


def test_clear_history(conversation_memory):
    """测试清空历史"""
    conversation_memory.turns = [
        ConversationTurn("t1", "msg", "reply", datetime.now())
    ]
    conversation_memory._turn_counter = 5
    
    conversation_memory.clear_history()
    
    assert len(conversation_memory.turns) == 0
    assert conversation_memory._turn_counter == 0


def test_get_stats(conversation_memory, mock_cognitive_loop):
    """测试统计信息"""
    # 添加一些数据
    conversation_memory.turns = [
        ConversationTurn("t1", "msg1", "reply1", datetime.now(), "mem_1"),
        ConversationTurn("t2", "msg2", "reply2", datetime.now(), "mem_2")
    ]
    
    # Mock memory network
    from llm_compression import MemoryPrimitive
    mem1 = MemoryPrimitive("mem_1", "content1", np.random.randn(384))
    mem2 = MemoryPrimitive("mem_2", "content2", np.random.randn(384))
    mem1.add_connection("mem_2", 0.8)
    mem2.add_connection("mem_1", 0.8)
    
    mock_cognitive_loop.memory_network = {
        "mem_1": mem1,
        "mem_2": mem2
    }
    
    stats = conversation_memory.get_stats()
    
    assert stats["total_turns"] == 2
    assert stats["memory_count"] == 2
    assert stats["connection_count"] == 1  # 双向连接算1个
    assert stats["avg_connections"] == 1.0


@pytest.mark.asyncio
async def test_turn_timestamp(conversation_memory):
    """测试时间戳记录"""
    before = datetime.now()
    await conversation_memory.add_turn("Hello", "Hi")
    after = datetime.now()
    
    turn = conversation_memory.turns[0]
    assert before <= turn.timestamp <= after


@pytest.mark.asyncio
async def test_empty_context(conversation_memory):
    """测试空历史的上下文检索"""
    context = await conversation_memory.get_context("query")
    assert context == []


def test_get_recent_turns_empty(conversation_memory):
    """测试空历史的最近轮次"""
    recent = conversation_memory.get_recent_turns(5)
    assert recent == []

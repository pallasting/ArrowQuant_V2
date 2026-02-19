#!/usr/bin/env python3
"""
ArrowEngine Agent-Chat 实际环境测试

测试 ArrowEngine 在真实 Agent-Chat 会话场景中的表现。
验证：精度、性能、稳定性、内存使用。
"""

import asyncio
import time
import sys
from datetime import datetime
from typing import List, Dict

# 添加项目路径
sys.path.insert(0, '/memory/Documents/ai-os-memory')

from llm_compression.embedding_provider import get_default_provider
from llm_compression.semantic_indexer import SemanticIndexer
from llm_compression.semantic_index_db import SemanticIndexDB
from llm_compression.arrow_storage import ArrowStorage
from llm_compression.memory_search import MemorySearch, SearchMode
from llm_compression.vector_search import VectorSearch
from llm_compression.background_queue import BackgroundQueue
from llm_compression.logger import logger


class ChatMemorySystem:
    """Chat 记忆系统（使用 ArrowEngine）"""
    
    def __init__(self, data_dir: str = "./test_chat_data"):
        """初始化系统"""
        self.data_dir = data_dir
        
        # 初始化组件
        logger.info("Initializing ArrowEngine-based memory system...")
        self.provider = get_default_provider()
        self.storage = ArrowStorage(f"{data_dir}/memories.parquet")
        self.index_db = SemanticIndexDB(f"{data_dir}/index")
        self.indexer = SemanticIndexer(self.provider, self.storage, self.index_db)
        self.vector_search = VectorSearch(self.provider, self.storage, self.index_db)
        self.memory_search = MemorySearch(self.vector_search, self.storage)
        self.background_queue = BackgroundQueue(self.indexer, batch_size=16)
        
        # 统计信息
        self.stats = {
            'memories_added': 0,
            'searches_performed': 0,
            'total_search_time': 0.0,
            'total_index_time': 0.0
        }
        
        logger.info(f"System initialized with {type(self.provider).__name__}")
    
    async def start(self):
        """启动系统"""
        await self.background_queue.start()
        logger.info("Background queue started")
    
    async def stop(self):
        """停止系统"""
        await self.background_queue.stop()
        logger.info("Background queue stopped")
    
    async def add_chat_message(
        self,
        message_id: str,
        role: str,
        content: str,
        metadata: Dict = None
    ):
        """添加聊天消息到记忆"""
        start = time.time()
        
        memory = {
            'memory_id': message_id,
            'category': 'chat',
            'context': f"[{role}] {content}",
            'timestamp': datetime.now(),
            'embedding': None,
            'metadata': metadata or {}
        }
        
        # 异步索引（非阻塞）
        await self.background_queue.submit(memory)
        
        elapsed = time.time() - start
        self.stats['memories_added'] += 1
        self.stats['total_index_time'] += elapsed
        
        return memory
    
    async def search_relevant_context(
        self,
        query: str,
        top_k: int = 5,
        mode: SearchMode = SearchMode.SEMANTIC
    ) -> List:
        """搜索相关上下文"""
        start = time.time()
        
        results = self.memory_search.search(
            query=query,
            category="chat",
            mode=mode,
            top_k=top_k
        )
        
        elapsed = time.time() - start
        self.stats['searches_performed'] += 1
        self.stats['total_search_time'] += elapsed
        
        return results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        avg_search_time = (
            self.stats['total_search_time'] / self.stats['searches_performed']
            if self.stats['searches_performed'] > 0 else 0
        )
        avg_index_time = (
            self.stats['total_index_time'] / self.stats['memories_added']
            if self.stats['memories_added'] > 0 else 0
        )
        
        return {
            **self.stats,
            'avg_search_time_ms': avg_search_time * 1000,
            'avg_index_time_ms': avg_index_time * 1000,
            'provider_type': type(self.provider).__name__,
            'embedding_dimension': self.provider.dimension
        }


async def simulate_chat_session(system: ChatMemorySystem):
    """模拟真实的 Chat 会话"""
    
    print("\n" + "="*70)
    print("🤖 ArrowEngine Agent-Chat 实际环境测试")
    print("="*70)
    
    # 1. 模拟对话历史
    print("\n📝 Phase 1: 添加对话历史...")
    
    conversation_history = [
        ("user", "你好，我想了解一下机器学习"),
        ("assistant", "你好！机器学习是人工智能的一个分支，它让计算机能够从数据中学习并做出预测。"),
        ("user", "机器学习有哪些主要类型？"),
        ("assistant", "机器学习主要分为三类：监督学习、无监督学习和强化学习。监督学习使用标记数据，无监督学习发现数据中的模式，强化学习通过试错来学习。"),
        ("user", "什么是深度学习？"),
        ("assistant", "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的复杂表示。它在图像识别、自然语言处理等领域表现出色。"),
        ("user", "Python 在机器学习中的作用是什么？"),
        ("assistant", "Python 是机器学习最流行的编程语言，因为它有丰富的库（如 scikit-learn、TensorFlow、PyTorch）和简洁的语法。"),
        ("user", "能给我推荐一些学习资源吗？"),
        ("assistant", "当然！我推荐：1) Andrew Ng 的机器学习课程，2) fast.ai 的深度学习课程，3) Kaggle 的实践项目。"),
    ]
    
    start_time = time.time()
    
    for i, (role, content) in enumerate(conversation_history):
        message_id = f"msg_{i+1}"
        await system.add_chat_message(
            message_id=message_id,
            role=role,
            content=content,
            metadata={'turn': i+1}
        )
        print(f"  ✓ [{role:10s}] {content[:60]}...")
    
    # 等待索引完成
    await system.background_queue.wait_until_empty(timeout=30.0)
    
    elapsed = time.time() - start_time
    print(f"\n  ⏱️  添加 {len(conversation_history)} 条消息: {elapsed:.2f}s")
    print(f"  📊 平均速度: {len(conversation_history)/elapsed:.1f} messages/s")
    
    # 2. 测试语义搜索
    print("\n🔍 Phase 2: 测试语义搜索...")
    
    test_queries = [
        "深度学习和神经网络",
        "Python 编程语言",
        "学习资源推荐",
        "机器学习的分类"
    ]
    
    for query in test_queries:
        print(f"\n  Query: {query}")
        
        results = await system.search_relevant_context(
            query=query,
            top_k=3,
            mode=SearchMode.SEMANTIC
        )
        
        if results:
            print(f"  Found {len(results)} results:")
            for j, result in enumerate(results[:3], 1):
                print(f"    {j}. [{result.memory_id}] similarity={result.similarity:.3f}")
                if result.memory:
                    context = result.memory.get('context', '')
                    print(f"       {context[:80]}...")
        else:
            print("  ⚠️  No results found")
    
    # 3. 测试混合搜索
    print("\n🔀 Phase 3: 测试混合搜索...")
    
    results = await system.search_relevant_context(
        query="机器学习 Python",
        top_k=5,
        mode=SearchMode.HYBRID
    )
    
    print(f"  Found {len(results)} results with HYBRID mode")
    for i, result in enumerate(results[:3], 1):
        print(f"    {i}. similarity={result.similarity:.3f}")
    
    # 4. 性能统计
    print("\n📊 Phase 4: 性能统计...")
    
    stats = system.get_stats()
    
    print(f"\n  Provider: {stats['provider_type']}")
    print(f"  Embedding Dimension: {stats['embedding_dimension']}")
    print(f"\n  Memories Added: {stats['memories_added']}")
    print(f"  Searches Performed: {stats['searches_performed']}")
    print(f"\n  Avg Index Time: {stats['avg_index_time_ms']:.2f}ms")
    print(f"  Avg Search Time: {stats['avg_search_time_ms']:.2f}ms")
    print(f"\n  Total Index Time: {stats['total_index_time']:.2f}s")
    print(f"  Total Search Time: {stats['total_search_time']:.2f}s")
    
    # 5. 索引统计
    print("\n📈 Phase 5: 索引统计...")
    
    index_size = system.index_db.get_category_size("chat")
    categories = system.index_db.get_categories()
    
    print(f"  Indexed Memories: {index_size}")
    print(f"  Categories: {categories}")
    
    # 6. 压力测试（可选）
    print("\n💪 Phase 6: 压力测试...")
    
    print("  Adding 50 more messages...")
    stress_start = time.time()
    
    for i in range(50):
        await system.add_chat_message(
            message_id=f"stress_msg_{i}",
            role="user" if i % 2 == 0 else "assistant",
            content=f"Stress test message {i}: This is a test message for performance evaluation.",
            metadata={'stress_test': True}
        )
    
    await system.background_queue.wait_until_empty(timeout=60.0)
    stress_elapsed = time.time() - stress_start
    
    print(f"  ✓ Added 50 messages in {stress_elapsed:.2f}s")
    print(f"  ✓ Throughput: {50/stress_elapsed:.1f} messages/s")
    
    # 7. 最终统计
    print("\n📊 Phase 7: 最终统计...")
    
    final_stats = system.get_stats()
    final_index_size = system.index_db.get_category_size("chat")
    
    print(f"\n  Total Memories: {final_stats['memories_added']}")
    print(f"  Total Searches: {final_stats['searches_performed']}")
    print(f"  Index Size: {final_index_size}")
    print(f"\n  Overall Avg Index Time: {final_stats['avg_index_time_ms']:.2f}ms")
    print(f"  Overall Avg Search Time: {final_stats['avg_search_time_ms']:.2f}ms")
    
    # 完成
    print("\n" + "="*70)
    print("✅ 测试完成！")
    print("="*70)
    
    # 评估结果
    print("\n📋 评估结果:")
    
    if stats['provider_type'] == 'ArrowEngineProvider':
        print("  ✅ 使用 ArrowEngine (高性能模式)")
    else:
        print("  ⚠️  使用 SentenceTransformerProvider (回退模式)")
    
    if final_stats['avg_search_time_ms'] < 100:
        print(f"  ✅ 搜索性能优秀 ({final_stats['avg_search_time_ms']:.2f}ms < 100ms)")
    elif final_stats['avg_search_time_ms'] < 500:
        print(f"  ✓  搜索性能良好 ({final_stats['avg_search_time_ms']:.2f}ms < 500ms)")
    else:
        print(f"  ⚠️  搜索性能需要优化 ({final_stats['avg_search_time_ms']:.2f}ms)")
    
    if final_stats['avg_index_time_ms'] < 50:
        print(f"  ✅ 索引性能优秀 ({final_stats['avg_index_time_ms']:.2f}ms < 50ms)")
    elif final_stats['avg_index_time_ms'] < 200:
        print(f"  ✓  索引性能良好 ({final_stats['avg_index_time_ms']:.2f}ms < 200ms)")
    else:
        print(f"  ⚠️  索引性能需要优化 ({final_stats['avg_index_time_ms']:.2f}ms)")
    
    if final_index_size == final_stats['memories_added']:
        print(f"  ✅ 索引完整性验证通过 ({final_index_size}/{final_stats['memories_added']})")
    else:
        print(f"  ⚠️  索引完整性问题 ({final_index_size}/{final_stats['memories_added']})")
    
    print("\n💡 建议:")
    if stats['provider_type'] != 'ArrowEngineProvider':
        print("  - 运行 'python scripts/convert_and_validate.py' 转换模型以使用 ArrowEngine")
    print("  - 在生产环境中使用 GPU 以获得更好的性能")
    print("  - 根据实际负载调整 batch_size 参数")
    print("  - 监控内存使用和索引大小")


async def main():
    """主函数"""
    system = None
    
    try:
        # 创建系统
        system = ChatMemorySystem(data_dir="./test_chat_data")
        
        # 启动系统
        await system.start()
        
        # 运行测试
        await simulate_chat_session(system)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # 清理
        if system:
            await system.stop()
            print("\n🧹 系统已清理")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

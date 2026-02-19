# Phase 2.0 快速开始指南

**版本**: 2.0  
**最后更新**: 2026-02-17

---

## 目录

1. [环境要求](#环境要求)
2. [安装步骤](#安装步骤)
3. [快速验证](#快速验证)
4. [基础使用](#基础使用)
5. [高级功能](#高级功能)
6. [常见问题](#常见问题)

---

## 环境要求

### 系统要求

- **操作系统**: Windows / Linux / macOS
- **Python**: 3.10+
- **内存**: 4GB+ (推荐 8GB+)
- **存储**: 2GB+ 可用空间

### 依赖服务

- **Ollama** (可选): 本地 LLM 服务
  - 下载: https://ollama.ai/
  - 默认端口: 11434

---

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/your-org/ai-os-memory.git
cd ai-os-memory
```

### 2. 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 安装为可编辑包（开发模式）
pip install -e .
```

### 3. 验证安装

```bash
# 运行测试
pytest tests/unit/ -v

# 查看版本
python -c "import llm_compression; print(llm_compression.__version__)"
```

---

## 快速验证

### 运行功能测试

```bash
python examples/test_chat_agent.py
```

**预期输出**:
```
==============================================================
Phase 2.0 功能测试
==============================================================

1️⃣  测试模型缓存...
   ✓ 模型预加载完成: 25.43s

2️⃣  初始化组件...
   ✓ 所有组件初始化完成

3️⃣  测试记忆添加...
   ✓ 添加 5 条记忆: 0.52s
   ✓ 平均速度: 9.6 memories/s

4️⃣  测试对话功能...
   Q: 什么是 Python？
   A: Python 是一种高级编程语言...
   ✓ 响应时间: 1.23s
   ✓ 使用记忆: 3
   ✓ 质量分数: 0.85

5️⃣  测试优化统计...
   ✓ 优化已启用
   ✓ 缓存模型数: 1
   ✓ 自适应调用: 5
   ✓ 批量处理: 5 items

6️⃣  测试统计信息...
   ✓ 对话轮次: 3
   ✓ 记忆数量: 5
   ✓ 连接数量: 10

==============================================================
✅ 所有测试完成！
==============================================================
```

---

## 基础使用

### 1. 启动交互式对话

```bash
python examples/chat_agent_optimized.py
```

### 2. 基础对话

```
💬 You: 你好！

🤖 Agent: 你好！我是一个具有持续学习能力的对话助手。
   📊 Quality: 0.85 | Memories: 0 | Learning: ✅ | Time: 0.52s

💬 You: 什么是机器学习？

🤖 Agent: 机器学习是人工智能的一个分支，它使计算机能够从数据中学习...
   📊 Quality: 0.92 | Memories: 2 | Learning: ✅ | Time: 1.15s
```

### 3. 使用命令

```
💬 You: /help

📖 Available Commands:
  /help              - Show this help message
  /stats             - Show conversation statistics
  /optimization      - Show optimization statistics
  /benchmark         - Run performance benchmark
  /import <file>     - Import file as memories
  /clear             - Clear conversation history
  /quit, /exit       - Exit the chat
```

---

## 高级功能

### 1. 导入文件作为记忆

```
💬 You: /import docs/README.md

📥 Importing: docs/README.md
  📄 Found 15 chunks
  ✅ Imported 15 new memories
  ✅ Total memories: 20
```

### 2. 查看统计信息

```
💬 You: /stats

📊 Conversation Statistics:
  • Total turns: 10
  • Memory count: 20
  • Connections: 45
  • Avg connections: 2.25
  • Session time: 125.3s

  User Profile:
    • Total interactions: 10
    • Top interests:
      - machine learning: 0.85
      - python: 0.72
      - ai: 0.68
```

### 3. 查看优化统计

```
💬 You: /optimization

⚡ Optimization Statistics:

  Model Cache:
    • Cached models: 1
    • Models: all-MiniLM-L6-v2

  Adaptive Embedder:
    • Total calls: 25
    • Traditional: 15 (60.0%)
    • Arrow: 10 (40.0%)
    • Total items: 250

  Batch Processor:
    • Items processed: 250
    • Batches: 5
    • Avg throughput: 1285.3 items/s
    • Batch size: 50
    • Workers: 4
```

### 4. 运行性能基准测试

```
💬 You: /benchmark

🏃 Running Performance Benchmark...
  This will test memory operations performance

  Test 1: Batch Memory Addition
    ✓ Added 100 memories in 0.78s
    ✓ Throughput: 128.2 memories/s

  Test 2: Memory Retrieval
    ✓ 10 retrievals in 0.35s
    ✓ Avg retrieval time: 35.2ms

  ✅ Benchmark Complete!
```

---

## 编程使用

### 基础示例

```python
import asyncio
from llm_compression import (
    LLMClient,
    LLMCompressor,
    ModelSelector,
    ConversationalAgent,
    CognitiveLoop
)
from llm_compression.embedder_cache import preload_default_model

async def main():
    # 1. 预加载模型（优化）
    preload_default_model()
    
    # 2. 初始化组件
    llm_client = LLMClient(endpoint="http://localhost:11434")
    model_selector = ModelSelector()
    compressor = LLMCompressor(llm_client, model_selector)
    
    # 3. 创建认知循环
    from llm_compression.expression_layer import MultiModalExpressor
    from llm_compression.internal_feedback import InternalFeedbackSystem
    from llm_compression.connection_learner import ConnectionLearner
    from llm_compression.network_navigator import NetworkNavigator
    from llm_compression.reconstructor import LLMReconstructor
    
    reconstructor = LLMReconstructor(llm_client=llm_client)
    expressor = MultiModalExpressor(llm_client, reconstructor)
    feedback = InternalFeedbackSystem()
    learner = ConnectionLearner()
    navigator = NetworkNavigator()
    
    cognitive_loop = CognitiveLoop(
        expressor=expressor,
        feedback=feedback,
        learner=learner,
        navigator=navigator
    )
    
    # 4. 创建对话 Agent
    agent = ConversationalAgent(
        llm_client=llm_client,
        compressor=compressor,
        cognitive_loop=cognitive_loop,
        user_id="user_001"
    )
    
    # 5. 对话
    response = await agent.chat("你好！")
    print(f"Agent: {response.message}")
    
    response = await agent.chat("什么是机器学习？")
    print(f"Agent: {response.message}")
    
    # 6. 查看统计
    stats = agent.get_stats()
    print(f"Total turns: {stats['total_turns']}")
    print(f"Memory count: {stats['memory_count']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 使用优化功能

```python
from llm_compression.cognitive_loop_arrow import CognitiveLoopArrow

# 创建优化版认知循环
cognitive_loop_arrow = CognitiveLoopArrow(
    cognitive_loop=cognitive_loop,
    enable_optimizations=True,  # 启用所有优化
    adaptive_threshold=1000,    # 自适应阈值
    batch_size=100,             # 批次大小
    max_workers=4               # 并行线程数
)

# 查看优化统计
opt_stats = cognitive_loop_arrow.get_optimization_stats()
print(f"Optimizations enabled: {opt_stats['optimizations_enabled']}")
```

---

## 常见问题

### Q1: 首次运行很慢？

**A**: 首次运行需要下载 embedding 模型（~500MB），大约需要 30-60 秒。后续运行会使用缓存，速度会快很多。

**解决方案**:
```python
# 在应用启动时预加载模型
from llm_compression.embedder_cache import preload_default_model
preload_default_model()
```

---

### Q2: Ollama 连接失败？

**A**: 确保 Ollama 服务正在运行。

**检查方法**:
```bash
# 检查 Ollama 是否运行
curl http://localhost:11434/api/tags

# 启动 Ollama（如果未运行）
ollama serve
```

---

### Q3: 内存占用过高？

**A**: 可以调整批次大小和并行线程数。

**优化方法**:
```python
cognitive_loop_arrow = CognitiveLoopArrow(
    enable_optimizations=True,
    batch_size=50,      # 减小批次大小
    max_workers=2       # 减少并行线程
)
```

---

### Q4: 如何提升性能？

**A**: 启用所有优化功能。

**优化清单**:
1. ✅ 预加载模型（减少首次延迟）
2. ✅ 使用 CognitiveLoopArrow（Arrow 零拷贝）
3. ✅ 启用自适应切换（自动选择最优方法）
4. ✅ 启用批量处理（提升吞吐量）
5. ✅ 调整批次大小和线程数

---

### Q5: 如何持久化记忆？

**A**: 使用 Arrow 存储保存记忆。

**示例**:
```python
from llm_compression.arrow_storage_zero_copy import ArrowStorageZeroCopy

# 保存记忆
storage = ArrowStorageZeroCopy()
storage.save(cognitive_loop_arrow.memory_table, "memories.parquet")

# 加载记忆
memory_table = storage.load("memories.parquet")
cognitive_loop_arrow.load_memories_from_table(memory_table)
```

---

## 下一步

- 📖 阅读 [API 参考文档](API_REFERENCE.md)
- 📐 阅读 [架构设计文档](ARCHITECTURE.md)
- 📚 阅读 [用户使用手册](USER_GUIDE.md)
- 🚀 查看 [性能优化指南](PHASE_2.0_OPTIMIZATION_COMPLETION_REPORT.md)

---

## 获取帮助

- **文档**: `docs/` 目录
- **示例**: `examples/` 目录
- **测试**: `tests/` 目录
- **问题**: GitHub Issues

---

**文档版本**: 1.0  
**最后更新**: 2026-02-17  
**维护者**: AI-OS 团队

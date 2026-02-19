#!/usr/bin/env python3
"""
对话Agent CLI - 交互式命令行界面

展示Phase 2.0+认知系统的持续学习能力
"""

import asyncio
import sys
from pathlib import Path

from llm_compression import (
    LLMClient,
    LLMCompressor,
    ModelSelector,
    ConversationalAgent,
    MemoryVisualizer,
    CognitiveLoop
)


class ChatCLI:
    """对话Agent命令行界面"""
    
    def __init__(self, agent: ConversationalAgent):
        self.agent = agent
        self.running = True
    
    async def run(self):
        """运行交互式对话"""
        self.print_welcome()
        
        while self.running:
            try:
                # 获取用户输入
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.startswith("/"):
                    await self.handle_command(user_input)
                    continue
                
                # 处理对话
                print("🤔 Agent is thinking...")
                response = await self.agent.chat(user_input)
                
                # 显示回复
                print(f"\n🤖 Agent: {response.message}")
                print(f"   📊 Quality: {response.quality_score:.2f} | "
                      f"Memories: {len(response.memories_used)} | "
                      f"Learning: {'✅' if response.learning_occurred else '❌'}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    async def handle_command(self, cmd: str):
        """处理特殊命令"""
        parts = cmd.split()
        command = parts[0].lower()
        
        if command == "/help":
            self.print_help()
        
        elif command == "/stats":
            self.print_stats()
        
        elif command == "/visualize":
            output_path = parts[1] if len(parts) > 1 else "memory_network.html"
            await self.visualize_network(output_path)
        
        elif command == "/import":
            if len(parts) < 2:
                print("❌ Usage: /import <file_path>")
                return
            file_path = " ".join(parts[1:])  # 支持带空格的路径
            await self.import_file(file_path)
        
        elif command == "/cleanup":
            await self.cleanup_memories()
        
        elif command == "/clear":
            self.agent.clear_history()
            print("✅ History cleared")
        
        elif command == "/quit" or command == "/exit":
            self.running = False
            print("👋 Goodbye!")
        
        else:
            print(f"❌ Unknown command: {command}")
            print("   Type /help for available commands")
    
    def print_welcome(self):
        """打印欢迎信息"""
        print("\n" + "="*60)
        print("🧠 Phase 2.0+ Conversational Agent")
        print("="*60)
        print("\n✨ Features:")
        print("  • Continuous learning (Hebbian)")
        print("  • Self-organizing memory network")
        print("  • Personalized responses")
        print("  • Network visualization")
        print("  • File import (knowledge seeding)")
        print("\n💡 Commands:")
        print("  /help      - Show help")
        print("  /import    - Import file as memories")
        print("  /stats     - Show statistics")
        print("  /visualize - Generate network visualization")
        print("  /clear     - Clear history")
        print("  /quit      - Exit")
        print("\n" + "="*60)
    
    def print_help(self):
        """打印帮助信息"""
        print("\n📖 Available Commands:")
        print("  /help              - Show this help message")
        print("  /import <file>     - Import file as memories (txt, md, py, json, yaml)")
        print("  /cleanup           - Remove unused/isolated memories")
        print("  /stats             - Show conversation statistics")
        print("  /visualize [path]  - Generate HTML visualization (default: memory_network.html)")
        print("  /clear             - Clear conversation history")
        print("  /quit, /exit       - Exit the chat")
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.agent.get_stats()
        
        print("\n📊 Statistics:")
        print(f"  Conversation:")
        print(f"    • Total turns: {stats['total_turns']}")
        print(f"    • Memory count: {stats['memory_count']}")
        print(f"    • Connections: {stats['connection_count']}")
        print(f"    • Avg connections: {stats['avg_connections']:.2f}")
        
        if "user_profile" in stats:
            profile = stats["user_profile"]
            print(f"\n  User Profile:")
            print(f"    • Total interactions: {profile['total_interactions']}")
            
            if profile["top_interests"]:
                print(f"    • Top interests:")
                for topic, score in profile["top_interests"]:
                    print(f"      - {topic}: {score:.2f}")
            
            print(f"    • Style:")
            for dim, value in profile["style"].items():
                print(f"      - {dim}: {value:.2f}")
    
    async def visualize_network(self, output_path: str):
        """生成网络可视化"""
        try:
            memory_network = self.agent.cognitive_loop.memory_network
            
            if not memory_network:
                print("⚠️  No memories to visualize yet")
                return
            
            visualizer = MemoryVisualizer(memory_network)
            visualizer.export_html(output_path, title="Conversation Memory Network")
            
            print(f"✅ Visualization saved to: {output_path}")
            print(f"   Open in browser to view interactive network")
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
    
    async def import_file(self, file_path: str):
        """导入文件为记忆"""
        try:
            from llm_compression.file_importer import FileImporter
            from llm_compression.memory_primitive import MemoryPrimitive
            import numpy as np
            import hashlib
            
            print(f"📥 Importing: {file_path}")
            
            # 创建导入器
            importer = FileImporter(self.agent.compressor)
            
            # 导入文件（返回 CompressedMemory 对象）
            compressed_memories = await importer.import_file(file_path)
            
            print(f"✅ Imported {len(compressed_memories)} memory chunks")
            
            # 关键：将导入的记忆添加到认知循环的记忆网络（带去重）
            print(f"  📌 Adding to memory network...", flush=True)
            
            memory_network = self.agent.cognitive_loop.memory_network
            
            added = 0
            skipped = 0
            
            for compressed in compressed_memories:
                # 计算内容哈希（用于去重）
                content_hash = compressed.summary_hash
                
                # 检查是否已存在相同内容
                duplicate = False
                for existing_id, existing_mem in memory_network.items():
                    if hasattr(existing_mem.content, 'summary_hash'):
                        if existing_mem.content.summary_hash == content_hash:
                            duplicate = True
                            skipped += 1
                            break
                
                if not duplicate:
                    # 创建 MemoryPrimitive
                    memory = MemoryPrimitive(
                        id=compressed.memory_id,
                        content=compressed,
                        embedding=np.array(compressed.embedding)
                    )
                    
                    # 添加到记忆网络（dict）
                    memory_network[memory.id] = memory
                    added += 1
            
            print(f"  ✓ Added {added} new memories, skipped {skipped} duplicates")
            print(f"  ✓ Memory network: {len(memory_network)} total memories")
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
        except Exception as e:
            print(f"❌ Import failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def cleanup_memories(self):
        """清理未使用的记忆"""
        print("🧹 Cleaning up unused memories...")
        
        memory_network = self.agent.cognitive_loop.memory_network
        
        if not memory_network:
            print("  ℹ️  No memories to clean")
            return
        
        # 统计每个记忆的激活次数（从连接强度推断）
        # 注意：当前实现中没有直接的激活计数，这里用连接数作为代理
        
        # 简单策略：删除没有任何连接的孤立记忆
        isolated = []
        for mem_id, memory in list(memory_network.items()):
            # 检查是否有连接
            has_connections = False
            for other_id in memory_network:
                if other_id != mem_id:
                    conn_key = tuple(sorted([mem_id, other_id]))
                    if conn_key in self.agent.cognitive_loop.learner.connections:
                        has_connections = True
                        break
            
            if not has_connections:
                isolated.append(mem_id)
        
        if isolated:
            print(f"  🗑️  Found {len(isolated)} isolated memories")
            confirm = input(f"  ⚠️  Delete {len(isolated)} memories? (y/N): ")
            if confirm.lower() == 'y':
                for mem_id in isolated:
                    del memory_network[mem_id]
                print(f"  ✓ Deleted {len(isolated)} memories")
                print(f"  ✓ Remaining: {len(memory_network)} memories")
            else:
                print("  ✗ Cleanup cancelled")
        else:
            print("  ✓ No isolated memories found")
            print(f"  ℹ️  All {len(memory_network)} memories have connections")


async def main():
    """主函数"""
    # 使用 HF 镜像（国内访问快）
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    print("🚀 Initializing Agent...")
    
    # 初始化LLM客户端
    # Initialize LLM client
    llm_client = LLMClient(
        endpoint="http://localhost:11434",
        timeout=30.0
    )
    
    # 初始化压缩器
    model_selector = ModelSelector()
    compressor = LLMCompressor(
        llm_client=llm_client,
        model_selector=model_selector
    )
    
    # 预热 embedding 模型（避免首次导入时卡住）
    print("  ⏳ Loading embedding model (first time ~50s)...")
    import asyncio
    _ = await compressor.get_embedding("预热")
    print("  ✓ Embedding model ready")
    
    # 初始化认知循环组件
    from llm_compression.expression_layer import MultiModalExpressor
    from llm_compression.internal_feedback import InternalFeedbackSystem
    from llm_compression.connection_learner import ConnectionLearner
    from llm_compression.network_navigator import NetworkNavigator
    from llm_compression.reconstructor import LLMReconstructor
    
    reconstructor = LLMReconstructor(llm_client=llm_client)
    expressor = MultiModalExpressor(
        llm_client=llm_client,
        reconstructor=reconstructor
    )
    feedback = InternalFeedbackSystem()
    learner = ConnectionLearner()
    navigator = NetworkNavigator()
    
    # 创建认知循环
    cognitive_loop = CognitiveLoop(
        expressor=expressor,
        feedback=feedback,
        learner=learner,
        navigator=navigator,
        quality_threshold=0.0,  # 禁用自我纠正
        max_corrections=0
    )
    
    # 创建对话Agent
    agent = ConversationalAgent(
        llm_client=llm_client,
        compressor=compressor,
        cognitive_loop=cognitive_loop,
        user_id="demo_user",
        enable_personalization=True
    )
    
    # 运行CLI
    cli = ChatCLI(agent)
    await cli.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)

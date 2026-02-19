#!/usr/bin/env python3
"""
快速启动版本 - 延迟加载所有模型
"""
import asyncio
import sys

from llm_compression import (
    LLMClient,
    LLMCompressor,
    ModelSelector,
    ConversationalAgent,
)


async def main():
    """主函数"""
    # 设置离线模式（避免HuggingFace连接超时）
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    print("🚀 Quick Start (延迟加载模式)...")
    
    # 1. 初始化LLM客户端（不连接）
    print("  ✓ LLM Client")
    llm_client = LLMClient(
        endpoint="http://localhost:11434",
        timeout=30.0
    )
    
    # 2. 初始化压缩器（不加载embedding）
    print("  ✓ Compressor (embedding延迟加载)")
    model_selector = ModelSelector()
    compressor = LLMCompressor(
        llm_client=llm_client,
        model_selector=model_selector,
        prewarm_embedding=False  # 关键：不预热
    )
    
    # 3. 初始化认知循环（最小配置）
    print("  ✓ Cognitive Loop")
    from llm_compression.expression_layer import MultiModalExpressor
    from llm_compression.internal_feedback import InternalFeedbackSystem
    from llm_compression.connection_learner import ConnectionLearner
    from llm_compression.network_navigator import NetworkNavigator
    from llm_compression.reconstructor import LLMReconstructor
    from llm_compression.cognitive_loop import CognitiveLoop
    
    reconstructor = LLMReconstructor(llm_client=llm_client)
    expressor = MultiModalExpressor(llm_client=llm_client, reconstructor=reconstructor)
    feedback = InternalFeedbackSystem()
    learner = ConnectionLearner()
    navigator = NetworkNavigator()
    
    cognitive_loop = CognitiveLoop(
        expressor=expressor,
        feedback=feedback,
        learner=learner,
        navigator=navigator,
        quality_threshold=0.0,
        max_corrections=0
    )
    
    # 4. 创建Agent
    print("  ✓ Agent")
    agent = ConversationalAgent(
        llm_client=llm_client,
        compressor=compressor,
        cognitive_loop=cognitive_loop,
        enable_personalization=False  # 暂时禁用个性化
    )
    
    print("\n✅ 初始化完成！开始对话...\n")
    print("=" * 60)
    
    # 简单对话循环
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input in ["/quit", "/exit"]:
                print("👋 Goodbye!")
                break
            
            print("🤔 Agent is thinking...")
            
            # 首次调用时会加载embedding（这里才会卡）
            response = await agent.chat(user_input)
            
            print(f"\n🤖 Agent: {response.message}")
            print(f"   📊 Quality: {response.quality_score:.2f} | "
                  f"Memories: {len(response.memories_used)}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

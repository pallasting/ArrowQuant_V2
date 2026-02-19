#!/usr/bin/env python3
"""
诊断记忆检索问题
"""
import asyncio
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

async def diagnose():
    print("=== 记忆检索诊断 ===\n")
    
    from llm_compression import LLMClient, LLMCompressor, ModelSelector
    from llm_compression.cognitive_loop import CognitiveLoop
    from llm_compression.expression_layer import MultiModalExpressor
    from llm_compression.internal_feedback import InternalFeedbackSystem
    from llm_compression.connection_learner import ConnectionLearner
    from llm_compression.network_navigator import NetworkNavigator
    from llm_compression.reconstructor import LLMReconstructor
    from llm_compression.memory_primitive import MemoryPrimitive
    import numpy as np
    
    # 初始化
    client = LLMClient("http://localhost:11434", 30.0)
    selector = ModelSelector()
    compressor = LLMCompressor(client, selector)
    
    # 预热
    print("1. 预热 embedding...")
    await compressor.get_embedding("test")
    print("   ✓ 完成\n")
    
    # 导入文件
    print("2. 导入文件...")
    from llm_compression.file_importer import FileImporter
    importer = FileImporter(compressor)
    
    file_path = "/Data/CascadeProjects/TalkingWithU/AI_Memory_Palace_Framework.md"
    compressed_memories = await importer.import_file(file_path)
    print(f"   ✓ 导入 {len(compressed_memories)} 块\n")
    
    # 创建记忆网络
    print("3. 构建记忆网络...")
    memory_network = {}
    for compressed in compressed_memories:
        memory = MemoryPrimitive(
            id=compressed.memory_id,
            content=compressed,
            embedding=np.array(compressed.embedding)
        )
        memory_network[memory.id] = memory
    print(f"   ✓ {len(memory_network)} 个记忆\n")
    
    # 测试查询
    query = "Memory Palace 框架"
    print(f"4. 测试查询: '{query}'")
    
    # 生成查询 embedding
    query_emb = await compressor.get_embedding(query)
    print(f"   查询 embedding: {len(query_emb)} 维")
    
    # 计算相似度
    print("\n   记忆相似度排名:")
    similarities = []
    for mem_id, memory in memory_network.items():
        sim = np.dot(query_emb, memory.embedding) / (
            np.linalg.norm(query_emb) * np.linalg.norm(memory.embedding)
        )
        similarities.append((mem_id, sim, memory))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    for i, (mem_id, sim, memory) in enumerate(similarities[:5]):
        print(f"   [{i+1}] 相似度: {sim:.3f} | ID: {mem_id[:16]}...")
        
        # 重构内容
        reconstructor = LLMReconstructor(client)
        reconstructed = await reconstructor.reconstruct(memory.content)
        
        preview = reconstructed.full_text[:100].replace('\n', ' ')
        print(f"       内容预览: {preview}...")
        print()
    
    # 测试完整流程
    print("\n5. 测试完整认知流程...")
    
    reconstructor = LLMReconstructor(client)
    expressor = MultiModalExpressor(client, reconstructor)
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
    
    # 手动添加记忆到认知循环
    cognitive_loop.memory_network = memory_network
    
    # 处理查询
    result = await cognitive_loop.process(
        query=query,
        query_embedding=query_emb,
        max_memories=3
    )
    
    print(f"   使用记忆数: {len(result.memories_used)}")
    print(f"   质量分数: {result.quality.overall:.2f}")
    print(f"\n   生成回复:")
    print(f"   {result.output[:200]}...")
    
    print("\n=== 诊断完成 ===")

if __name__ == "__main__":
    asyncio.run(diagnose())

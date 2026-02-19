#!/usr/bin/env python3
"""
诊断导入卡死问题
"""
import asyncio
import time
import os

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

async def test_import():
    print("=== 导入诊断测试 ===\n")
    
    # 1. 测试 LLM 连接
    print("1. 测试 Ollama 连接...")
    from llm_compression import LLMClient
    client = LLMClient("http://localhost:11434", 30.0)
    
    try:
        start = time.time()
        response = await client.generate("测试", max_tokens=10)
        print(f"   ✓ Ollama 响应正常 ({time.time()-start:.2f}s)")
        print(f"   响应: {response.text[:50]}...")
    except Exception as e:
        print(f"   ✗ Ollama 错误: {e}")
        return
    
    # 2. 测试 Embedding
    print("\n2. 测试 Embedding 加载...")
    from llm_compression import LLMCompressor, ModelSelector
    
    start = time.time()
    selector = ModelSelector()
    compressor = LLMCompressor(client, selector, prewarm_embedding=False)
    print(f"   ✓ Compressor 初始化 ({time.time()-start:.2f}s)")
    
    print("\n3. 测试首次 Embedding 生成...")
    start = time.time()
    emb = await compressor.get_embedding("测试文本")
    print(f"   ✓ Embedding 生成 ({time.time()-start:.2f}s, 维度={len(emb)})")
    
    # 4. 测试压缩
    print("\n4. 测试文本压缩...")
    test_text = "这是一段测试文本。" * 20  # ~200字符
    
    start = time.time()
    compressed = await compressor.compress(test_text)
    elapsed = time.time() - start
    print(f"   ✓ 压缩完成 ({elapsed:.2f}s)")
    print(f"   原始: {len(test_text)} 字符")
    print(f"   压缩: {len(compressed.diff_data)} 字节")
    print(f"   比率: {compressed.compression_metadata.compression_ratio:.1f}x")
    
    # 5. 测试文件导入
    print("\n5. 测试小文件导入...")
    from llm_compression.file_importer import FileImporter
    
    # 创建测试文件
    test_file = "/tmp/test_import.txt"
    with open(test_file, 'w') as f:
        f.write("测试内容。\n" * 100)  # ~500字符
    
    importer = FileImporter(compressor)
    
    start = time.time()
    print(f"   导入文件: {test_file} (~500字符)")
    
    # 逐块显示进度
    memory_ids = []
    chunks = importer._chunk_text(open(test_file).read(), 1000, 200)
    print(f"   分块数: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        chunk_start = time.time()
        annotated = f"[Source: test, Part {i+1}/{len(chunks)}]\n{chunk}"
        compressed = await compressor.compress(annotated)
        memory_ids.append(compressed.memory_id)
        print(f"   块 {i+1}/{len(chunks)}: {time.time()-chunk_start:.2f}s")
    
    elapsed = time.time() - start
    print(f"   ✓ 导入完成 ({elapsed:.2f}s, 平均 {elapsed/len(chunks):.2f}s/块)")
    
    # 清理
    os.remove(test_file)
    
    print("\n=== 诊断完成 ===")
    print(f"\n结论:")
    print(f"  - Ollama 工作正常")
    print(f"  - Embedding 工作正常")
    print(f"  - 压缩工作正常")
    print(f"  - 小文件导入工作正常")
    print(f"\n如果大文件卡死，可能是:")
    print(f"  1. 分块太多，LLM 调用超时")
    print(f"  2. 内存不足")
    print(f"  3. Ollama 并发限制")

if __name__ == "__main__":
    try:
        asyncio.run(test_import())
    except KeyboardInterrupt:
        print("\n\n中断测试")
    finally:
        # 清理连接
        import gc
        gc.collect()

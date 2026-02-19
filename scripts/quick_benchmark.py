#!/usr/bin/env python3
"""
快速基准测试脚本

快速测试本地模型和云端 API 的性能对比。
用于验证 Phase 1.1 性能目标。
"""

import asyncio
import time
from typing import List, Dict

from llm_compression.config import Config
from llm_compression.llm_client import LLMClient
from llm_compression.model_selector import ModelSelector
from llm_compression.compressor import LLMCompressor
from llm_compression.reconstructor import LLMReconstructor
from llm_compression.quality_evaluator import QualityEvaluator
from llm_compression.logger import logger


async def quick_benchmark():
    """快速基准测试"""
    print("\n" + "=" * 60)
    print("快速基准测试 - Phase 1.1 性能验证")
    print("=" * 60)
    
    # 初始化组件
    config = Config.load()
    llm_client = LLMClient(endpoint=config.llm_endpoint)
    model_selector = ModelSelector(
        cloud_endpoint=config.llm_endpoint,
        prefer_local=True,
        ollama_endpoint=config.ollama_endpoint
    )
    quality_evaluator = QualityEvaluator()
    
    compressor = LLMCompressor(
        llm_client=llm_client,
        model_selector=model_selector,
        quality_evaluator=quality_evaluator
    )
    
    reconstructor = LLMReconstructor(llm_client=llm_client)
    
    # 测试文本
    test_text = """
    On January 15, 2024, Dr. Alice Johnson from Stanford University announced a major breakthrough 
    in quantum computing. The research team successfully demonstrated a 100-qubit quantum processor 
    that achieved quantum supremacy in solving complex optimization problems. The processor, named 
    "Quantum-X", operates at temperatures near absolute zero (-273.15°C) and uses superconducting 
    qubits. This achievement represents a significant milestone in the field, as it demonstrates 
    practical applications for quantum computing in areas such as drug discovery, financial modeling, 
    and cryptography. The project received $50 million in funding from the National Science Foundation 
    and involved collaboration with researchers from MIT, Caltech, and IBM Research. Dr. Johnson stated 
    that this technology could revolutionize computing within the next 5-10 years.
    """ * 2  # 约 1000 字符
    
    print(f"\n测试文本长度: {len(test_text)} 字符")
    
    # 测试本地模型
    print("\n" + "-" * 60)
    print("测试 1: 本地模型 (Qwen2.5-7B)")
    print("-" * 60)
    
    try:
        start = time.time()
        compressed = await compressor.compress(test_text, manual_model="qwen2.5")
        compression_time = time.time() - start
        
        if compressed:
            print(f"✓ 压缩成功")
            print(f"  - 压缩比: {compressed.metadata.compression_ratio:.2f}x")
            print(f"  - 压缩延迟: {compression_time * 1000:.0f}ms")
            print(f"  - 原始大小: {compressed.metadata.original_size} 字节")
            print(f"  - 压缩后大小: {compressed.metadata.compressed_size} 字节")
            
            # 测试重构
            start = time.time()
            reconstructed = await reconstructor.reconstruct(compressed)
            reconstruction_time = time.time() - start
            
            if reconstructed:
                print(f"✓ 重构成功")
                print(f"  - 重构延迟: {reconstruction_time * 1000:.0f}ms")
                
                # 质量评估
                quality = await quality_evaluator.evaluate(
                    original_text=test_text,
                    reconstructed_text=reconstructed.full_text,
                    compressed_memory=compressed
                )
                print(f"  - 语义相似度: {quality.semantic_similarity:.3f}")
                print(f"  - 实体准确率: {quality.entity_accuracy:.3f}")
                
                # Phase 1.1 目标验证
                print(f"\nPhase 1.1 目标验证:")
                print(f"  - 压缩延迟 < 2s: {'✓' if compression_time < 2 else '✗'} ({compression_time:.2f}s)")
                print(f"  - 重构延迟 < 500ms: {'✓' if reconstruction_time < 0.5 else '✗'} ({reconstruction_time * 1000:.0f}ms)")
                print(f"  - 压缩比 > 10x: {'✓' if compressed.metadata.compression_ratio > 10 else '✗'} ({compressed.metadata.compression_ratio:.2f}x)")
                print(f"  - 质量 > 0.85: {'✓' if quality.semantic_similarity > 0.85 else '✗'} ({quality.semantic_similarity:.3f})")
    
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        logger.error(f"Local model test failed", exc_info=True)
    
    # 测试云端 API（如果可用）
    print("\n" + "-" * 60)
    print("测试 2: 云端 API")
    print("-" * 60)
    
    try:
        start = time.time()
        compressed = await compressor.compress(test_text, manual_model="cloud-api")
        compression_time = time.time() - start
        
        if compressed:
            print(f"✓ 压缩成功")
            print(f"  - 压缩比: {compressed.metadata.compression_ratio:.2f}x")
            print(f"  - 压缩延迟: {compression_time * 1000:.0f}ms")
            
            # 测试重构
            start = time.time()
            reconstructed = await reconstructor.reconstruct(compressed)
            reconstruction_time = time.time() - start
            
            if reconstructed:
                print(f"✓ 重构成功")
                print(f"  - 重构延迟: {reconstruction_time * 1000:.0f}ms")
                
                # 质量评估
                quality = await quality_evaluator.evaluate(
                    original_text=test_text,
                    reconstructed_text=reconstructed.full_text,
                    compressed_memory=compressed
                )
                print(f"  - 语义相似度: {quality.semantic_similarity:.3f}")
                print(f"  - 实体准确率: {quality.entity_accuracy:.3f}")
    
    except Exception as e:
        print(f"✗ 云端 API 不可用或测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("快速基准测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(quick_benchmark())

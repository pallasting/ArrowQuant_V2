"""
优化的批量处理示例

演示 Phase 1.1 的性能优化：
- 增加批量大小到 32
- 提高并发数到 8
- 优化分组算法
- 使用本地模型加速
"""

import asyncio
import time
from typing import List
from llm_compression.batch_processor import BatchProcessor
from llm_compression.compressor import LLMCompressor
from llm_compression.llm_client import LLMClient
from llm_compression.model_selector import ModelSelector
from llm_compression.quality_evaluator import QualityEvaluator
from llm_compression.performance_config import PerformanceConfig
from llm_compression.config import load_config


async def generate_test_texts(count: int = 100) -> List[str]:
    """生成测试文本"""
    texts = []
    
    # 生成不同类型的文本
    for i in range(count):
        if i % 3 == 0:
            # 技术文档
            text = f"""
            技术文档 #{i}: 本系统实现了基于 LLM 的语义压缩算法。
            主要特性包括：
            1. 10-50x 压缩比
            2. 语义相似度 > 0.85
            3. 实体准确率 > 0.95
            4. 支持批量处理
            5. 本地模型部署
            
            性能指标：
            - 压缩延迟: < 2s
            - 重构延迟: < 500ms
            - 吞吐量: > 100/min
            
            技术栈：Python 3.10+, PyArrow, Ollama, ROCm
            """
        elif i % 3 == 1:
            # 会议记录
            text = f"""
            会议记录 #{i}: 2026-02-15 团队会议
            参与者: Alice, Bob, Charlie
            
            讨论内容:
            1. Phase 1.0 已完成，所有验收标准达成
            2. Phase 1.1 进展顺利，本地模型已部署
            3. 性能优化正在进行中
            
            行动项:
            - Alice: 完成性能基准测试
            - Bob: 更新文档
            - Charlie: 准备演示
            
            下次会议: 2026-02-22
            """
        else:
            # 用户反馈
            text = f"""
            用户反馈 #{i}: 系统使用体验
            
            优点:
            - 压缩效果很好，节省了大量存储空间
            - 重构质量高，几乎无损
            - 本地模型部署降低了成本
            
            改进建议:
            - 希望支持更多模型
            - 希望提供更详细的文档
            - 希望有更好的监控界面
            
            总体评分: 9/10
            """
        
        texts.append(text)
    
    return texts


async def benchmark_batch_processing(
    texts: List[str],
    config: PerformanceConfig,
    use_local: bool = True
):
    """
    批量处理性能基准测试
    
    Args:
        texts: 测试文本列表
        config: 性能配置
        use_local: 是否使用本地模型
    """
    print(f"\n{'='*60}")
    print(f"批量处理性能测试 ({'本地模型' if use_local else '云端 API'})")
    print(f"{'='*60}")
    
    # 创建组件
    system_config = load_config()
    
    llm_client = LLMClient(
        endpoint=system_config.model.ollama_endpoint if use_local else system_config.llm.cloud_endpoint,
        timeout=system_config.llm.timeout,
        max_retries=system_config.llm.max_retries
    )
    
    model_selector = ModelSelector(
        cloud_endpoint=system_config.llm.cloud_endpoint,
        ollama_endpoint=system_config.model.ollama_endpoint,
        prefer_local=use_local
    )
    
    quality_evaluator = QualityEvaluator()
    
    compressor = LLMCompressor(
        llm_client=llm_client,
        model_selector=model_selector,
        quality_evaluator=quality_evaluator
    )
    
    # 创建批量处理器
    batch_size = config.get_batch_size(use_local)
    max_concurrent = config.get_max_concurrent(use_local)
    
    batch_processor = BatchProcessor(
        compressor=compressor,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        similarity_threshold=config.similarity_threshold
    )
    
    print(f"\n配置:")
    print(f"  - 批量大小: {batch_size}")
    print(f"  - 最大并发: {max_concurrent}")
    print(f"  - 相似度阈值: {config.similarity_threshold}")
    print(f"  - 文本数量: {len(texts)}")
    
    # 执行批量压缩
    print(f"\n开始批量压缩...")
    start_time = time.time()
    
    try:
        results = await batch_processor.compress_batch(
            texts=texts,
            checkpoint_id=f"benchmark_{'local' if use_local else 'cloud'}"
        )
        
        elapsed_time = time.time() - start_time
        
        # 统计结果
        successful = sum(1 for r in results if r is not None)
        failed = len(results) - successful
        
        if successful > 0:
            avg_compression_ratio = sum(
                r.metadata.compression_ratio for r in results if r is not None
            ) / successful
            
            avg_quality = sum(
                r.metadata.quality_score for r in results if r is not None
            ) / successful
        else:
            avg_compression_ratio = 0
            avg_quality = 0
        
        # 输出结果
        print(f"\n结果:")
        print(f"  - 总耗时: {elapsed_time:.2f}s")
        print(f"  - 成功: {successful}/{len(texts)}")
        print(f"  - 失败: {failed}")
        print(f"  - 吞吐量: {successful / elapsed_time:.2f} 条/秒")
        print(f"  - 平均压缩比: {avg_compression_ratio:.2f}x")
        print(f"  - 平均质量: {avg_quality:.2f}")
        
        # 性能指标
        throughput_per_min = (successful / elapsed_time) * 60
        print(f"\n性能指标:")
        print(f"  - 吞吐量: {throughput_per_min:.2f} 条/分钟")
        print(f"  - 平均延迟: {elapsed_time / len(texts):.2f}s/条")
        
        # Phase 1.1 目标检查
        print(f"\nPhase 1.1 目标检查:")
        print(f"  - 吞吐量 > 100/min: {'✓' if throughput_per_min > 100 else '✗'} ({throughput_per_min:.2f})")
        print(f"  - 压缩比 > 30x: {'✓' if avg_compression_ratio > 30 else '✗'} ({avg_compression_ratio:.2f})")
        print(f"  - 质量 > 0.85: {'✓' if avg_quality > 0.85 else '✗'} ({avg_quality:.2f})")
        
        return {
            "elapsed_time": elapsed_time,
            "successful": successful,
            "failed": failed,
            "throughput_per_min": throughput_per_min,
            "avg_compression_ratio": avg_compression_ratio,
            "avg_quality": avg_quality
        }
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return None


async def compare_configurations():
    """对比不同配置的性能"""
    print("="*60)
    print("性能配置对比测试")
    print("="*60)
    
    # 生成测试数据
    print("\n生成测试数据...")
    texts = await generate_test_texts(count=50)  # 使用较小的数据集进行快速测试
    print(f"生成了 {len(texts)} 条测试文本")
    
    # 测试配置
    configs = [
        ("Phase 1.0 配置", PerformanceConfig.for_cloud_api(), False),
        ("Phase 1.1 配置（本地模型）", PerformanceConfig.for_local_model(), True),
    ]
    
    results = {}
    
    for name, config, use_local in configs:
        print(f"\n\n{'='*60}")
        print(f"测试: {name}")
        print(f"{'='*60}")
        
        result = await benchmark_batch_processing(texts, config, use_local)
        if result:
            results[name] = result
        
        # 等待一下，避免过载
        await asyncio.sleep(2)
    
    # 对比结果
    if len(results) >= 2:
        print(f"\n\n{'='*60}")
        print("性能对比")
        print(f"{'='*60}")
        
        phase_10 = results.get("Phase 1.0 配置")
        phase_11 = results.get("Phase 1.1 配置（本地模型）")
        
        if phase_10 and phase_11:
            print(f"\n吞吐量提升:")
            improvement = (phase_11["throughput_per_min"] / phase_10["throughput_per_min"] - 1) * 100
            print(f"  Phase 1.0: {phase_10['throughput_per_min']:.2f} 条/分钟")
            print(f"  Phase 1.1: {phase_11['throughput_per_min']:.2f} 条/分钟")
            print(f"  提升: {improvement:+.1f}%")
            
            print(f"\n延迟改进:")
            latency_10 = phase_10["elapsed_time"] / 50
            latency_11 = phase_11["elapsed_time"] / 50
            improvement = (1 - latency_11 / latency_10) * 100
            print(f"  Phase 1.0: {latency_10:.2f}s/条")
            print(f"  Phase 1.1: {latency_11:.2f}s/条")
            print(f"  改进: {improvement:+.1f}%")


async def main():
    """主函数"""
    print("="*60)
    print("Phase 1.1 性能优化示例")
    print("="*60)
    
    print("\n本示例演示:")
    print("1. 增加批量大小到 32")
    print("2. 提高并发数到 8")
    print("3. 优化分组算法")
    print("4. 使用本地模型加速")
    
    # 运行对比测试
    await compare_configurations()
    
    print("\n"+"="*60)
    print("测试完成！")
    print("="*60)
    
    print("\n关键优化:")
    print("1. 批量大小: 16 → 32 (2x)")
    print("2. 并发数: 4 → 8 (2x)")
    print("3. 缓存大小: 10000 → 50000 (5x)")
    print("4. 本地模型: 降低延迟和成本")
    
    print("\n预期改进:")
    print("- 吞吐量: > 100 条/分钟")
    print("- 延迟: < 2s/条")
    print("- 成本: 节省 90%")


if __name__ == "__main__":
    asyncio.run(main())

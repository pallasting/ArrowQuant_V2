"""
Performance Optimization Example

Demonstrates batch processing, checkpoint/resume, and performance monitoring.

Feature: llm-compression-integration
Requirements: 9.1, 9.3, 9.4, 9.6, 9.7, 10.1
"""

import asyncio
import tempfile
from pathlib import Path

from llm_compression import (
    LLMClient,
    ModelSelector,
    LLMCompressor,
    BatchProcessor,
    PerformanceMonitor,
    MemoryType,
    Config
)


async def main():
    """Demonstrate performance optimization features"""
    
    print("=" * 60)
    print("PERFORMANCE OPTIMIZATION EXAMPLE")
    print("=" * 60)
    print()
    
    # Load configuration
    config = Config.from_yaml("config.yaml")
    
    # Initialize components
    llm_client = LLMClient(
        endpoint=config.llm_endpoint,
        timeout=config.llm_timeout,
        max_retries=config.llm_max_retries
    )
    
    model_selector = ModelSelector(
        cloud_endpoint=config.llm_endpoint,
        prefer_local=False
    )
    
    compressor = LLMCompressor(
        llm_client=llm_client,
        model_selector=model_selector,
        min_compress_length=100
    )
    
    # Create temporary checkpoint directory
    checkpoint_dir = tempfile.mkdtemp()
    print(f"Checkpoint directory: {checkpoint_dir}")
    print()
    
    # Initialize batch processor
    batch_processor = BatchProcessor(
        compressor=compressor,
        batch_size=16,
        max_concurrent=4,
        similarity_threshold=0.8,
        checkpoint_dir=checkpoint_dir
    )
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(
        quality_alert_threshold=0.85,
        quality_drop_threshold=0.10
    )
    
    # Example 1: Batch Processing
    print("=" * 60)
    print("EXAMPLE 1: BATCH PROCESSING")
    print("=" * 60)
    print()
    
    # Generate sample texts
    texts = [
        f"This is a sample memory about topic {i % 3}. "
        f"It contains information about subject {i % 3} and related concepts. "
        f"Memory number {i}."
        for i in range(20)
    ]
    
    print(f"Processing {len(texts)} texts in batch...")
    
    # Process batch
    import time
    start_time = time.time()
    
    results = await batch_processor.compress_batch(
        texts=texts,
        memory_type=MemoryType.TEXT,
        checkpoint_id="example_batch_1"
    )
    
    elapsed_time = time.time() - start_time
    throughput = (len(texts) / elapsed_time) * 60
    
    print(f"✓ Batch processing complete:")
    print(f"  - Processed: {len(results)} texts")
    print(f"  - Time: {elapsed_time:.2f}s")
    print(f"  - Throughput: {throughput:.1f} items/min")
    print()
    
    # Record metrics
    for result in results:
        await monitor.record_compression(
            latency_ms=result.compression_metadata.compression_time_ms,
            compression_ratio=result.compression_metadata.compression_ratio,
            quality_score=result.compression_metadata.quality_score,
            model_used=result.compression_metadata.model_used,
            tokens_used=100  # Approximate
        )
    
    # Example 2: Checkpoint/Resume
    print("=" * 60)
    print("EXAMPLE 2: CHECKPOINT/RESUME")
    print("=" * 60)
    print()
    
    # Simulate a batch that might fail
    more_texts = [
        f"Additional memory {i} for checkpoint testing"
        for i in range(10)
    ]
    
    print(f"Processing {len(more_texts)} texts with checkpoint support...")
    
    results2 = await batch_processor.compress_batch(
        texts=more_texts,
        checkpoint_id="example_batch_2"
    )
    
    print(f"✓ Checkpoint-enabled processing complete:")
    print(f"  - Processed: {len(results2)} texts")
    print(f"  - Checkpoint ID: example_batch_2")
    print()
    
    # Example 3: Performance Monitoring
    print("=" * 60)
    print("EXAMPLE 3: PERFORMANCE MONITORING")
    print("=" * 60)
    print()
    
    # Get statistics
    stats = monitor.get_statistics()
    
    print("Performance Statistics:")
    print(f"  - Total Compressions: {stats.total_compressions}")
    print(f"  - Avg Compression Latency: {stats.avg_compression_latency_ms:.2f}ms")
    print(f"  - Avg Compression Ratio: {stats.avg_compression_ratio:.2f}x")
    print(f"  - Avg Quality Score: {stats.avg_quality_score:.3f}")
    print(f"  - Throughput: {stats.throughput_per_minute:.1f} items/min")
    print()
    
    # Get model comparison
    model_comparison = monitor.get_model_comparison()
    
    print("Model Comparison:")
    for model, model_stats in model_comparison.items():
        print(f"  {model}:")
        print(f"    - Usage: {model_stats['total_usage']} compressions")
        print(f"    - Avg Latency: {model_stats['avg_latency_ms']:.2f}ms")
        print(f"    - Avg Quality: {model_stats['avg_quality']:.3f}")
        print(f"    - Avg Ratio: {model_stats['avg_compression_ratio']:.2f}x")
    print()
    
    # Generate full report
    print("=" * 60)
    print("FULL PERFORMANCE REPORT")
    print("=" * 60)
    print()
    
    report = monitor.generate_report()
    print(report)
    print()
    
    # Export Prometheus metrics
    print("=" * 60)
    print("PROMETHEUS METRICS")
    print("=" * 60)
    print()
    
    prometheus_metrics = monitor.export_prometheus_metrics()
    print(prometheus_metrics)
    print()
    
    # Cleanup
    import shutil
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    
    print("=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

"""
Performance benchmark suite for ArrowEngine vs sentence-transformers.

This benchmark validates that ArrowEngine achieves the performance targets:
- Model load time: < 100ms (vs 2-5s for sentence-transformers)
- Single inference latency: < 5ms (vs 10-20ms)
- Batch throughput: > 2000 req/s (vs 500-800 req/s)
- Memory usage: < 100MB (vs ~180MB)

Usage:
    python benchmarks/arrowengine_benchmark.py
    
    # With specific model path:
    ARROW_MODEL_PATH=./models/minilm python benchmarks/arrowengine_benchmark.py
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_compression.logger import logger


@dataclass
class BenchmarkResult:
    """Performance benchmark results."""
    
    model_load_time_ms: float
    single_inference_latency_ms: float
    batch_throughput_rps: float
    memory_usage_mb: float
    
    # Additional metrics
    batch_latency_ms: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    
    def __str__(self) -> str:
        """Format results for display."""
        return (
            f"Performance Benchmark Results:\n"
            f"  Model Load Time:     {self.model_load_time_ms:>8.2f} ms\n"
            f"  Single Latency (P50): {self.p50_latency_ms or self.single_inference_latency_ms:>8.2f} ms\n"
            f"  Single Latency (P95): {self.p95_latency_ms or 0:>8.2f} ms\n"
            f"  Single Latency (P99): {self.p99_latency_ms or 0:>8.2f} ms\n"
            f"  Batch Throughput:    {self.batch_throughput_rps:>8.1f} req/s\n"
            f"  Memory Usage:        {self.memory_usage_mb:>8.1f} MB"
        )
    
    def meets_targets(self) -> bool:
        """Check if results meet performance targets."""
        return (
            self.model_load_time_ms < 100 and
            self.single_inference_latency_ms < 5 and
            self.batch_throughput_rps > 2000 and
            self.memory_usage_mb < 100
        )


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        logger.warning("psutil not installed, memory measurement unavailable")
        return 0.0


def benchmark_arrowengine(model_path: str) -> BenchmarkResult:
    """
    Benchmark ArrowEngine performance.
    
    Args:
        model_path: Path to converted ArrowEngine model
        
    Returns:
        BenchmarkResult with performance metrics
    """
    from llm_compression.inference.arrow_engine import ArrowEngine
    
    logger.info("=" * 60)
    logger.info("ArrowEngine Performance Benchmark")
    logger.info("=" * 60)
    
    # === 1. Model Loading Time ===
    logger.info("\n1. Measuring model load time...")
    start = time.time()
    engine = ArrowEngine(model_path)
    load_time_ms = (time.time() - start) * 1000
    logger.info(f"   Load time: {load_time_ms:.2f} ms")
    
    # === 2. Single Inference Latency ===
    logger.info("\n2. Measuring single inference latency...")
    test_sentence = "Machine learning is a subset of artificial intelligence."
    
    # Warmup
    for _ in range(10):
        engine.encode(test_sentence)
    
    # Measure
    latencies = []
    num_iterations = 100
    for _ in range(num_iterations):
        start = time.time()
        engine.encode(test_sentence)
        latencies.append((time.time() - start) * 1000)
    
    latencies = np.array(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    logger.info(f"   P50 latency: {p50_latency:.2f} ms")
    logger.info(f"   P95 latency: {p95_latency:.2f} ms")
    logger.info(f"   P99 latency: {p99_latency:.2f} ms")
    
    # === 3. Batch Throughput ===
    logger.info("\n3. Measuring batch throughput...")
    batch_size = 32
    num_batches = 100
    test_sentences = [test_sentence] * batch_size
    
    # Warmup
    for _ in range(5):
        engine.encode(test_sentences)
    
    # Measure
    start = time.time()
    for _ in range(num_batches):
        engine.encode(test_sentences)
    elapsed = time.time() - start
    
    total_requests = batch_size * num_batches
    throughput = total_requests / elapsed
    batch_latency_ms = (elapsed / num_batches) * 1000
    
    logger.info(f"   Throughput: {throughput:.1f} req/s")
    logger.info(f"   Batch latency: {batch_latency_ms:.2f} ms (batch_size={batch_size})")
    
    # === 4. Memory Usage ===
    logger.info("\n4. Measuring memory usage...")
    memory_mb = get_memory_usage_mb()
    logger.info(f"   Memory usage: {memory_mb:.1f} MB")
    
    return BenchmarkResult(
        model_load_time_ms=load_time_ms,
        single_inference_latency_ms=p50_latency,
        batch_throughput_rps=throughput,
        memory_usage_mb=memory_mb,
        batch_latency_ms=batch_latency_ms,
        p50_latency_ms=p50_latency,
        p95_latency_ms=p95_latency,
        p99_latency_ms=p99_latency,
    )


def benchmark_sentence_transformers() -> BenchmarkResult:
    """
    Benchmark sentence-transformers performance for comparison.
    
    Returns:
        BenchmarkResult with performance metrics
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed, skipping baseline")
        return None
    
    logger.info("\n" + "=" * 60)
    logger.info("sentence-transformers Baseline Benchmark")
    logger.info("=" * 60)
    
    # === 1. Model Loading Time ===
    logger.info("\n1. Measuring model load time...")
    start = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    load_time_ms = (time.time() - start) * 1000
    logger.info(f"   Load time: {load_time_ms:.2f} ms")
    
    # === 2. Single Inference Latency ===
    logger.info("\n2. Measuring single inference latency...")
    test_sentence = "Machine learning is a subset of artificial intelligence."
    
    # Warmup
    for _ in range(10):
        model.encode(test_sentence)
    
    # Measure
    latencies = []
    num_iterations = 100
    for _ in range(num_iterations):
        start = time.time()
        model.encode(test_sentence)
        latencies.append((time.time() - start) * 1000)
    
    latencies = np.array(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    logger.info(f"   P50 latency: {p50_latency:.2f} ms")
    logger.info(f"   P95 latency: {p95_latency:.2f} ms")
    logger.info(f"   P99 latency: {p99_latency:.2f} ms")
    
    # === 3. Batch Throughput ===
    logger.info("\n3. Measuring batch throughput...")
    batch_size = 32
    num_batches = 100
    test_sentences = [test_sentence] * batch_size
    
    # Warmup
    for _ in range(5):
        model.encode(test_sentences, batch_size=batch_size)
    
    # Measure
    start = time.time()
    for _ in range(num_batches):
        model.encode(test_sentences, batch_size=batch_size)
    elapsed = time.time() - start
    
    total_requests = batch_size * num_batches
    throughput = total_requests / elapsed
    batch_latency_ms = (elapsed / num_batches) * 1000
    
    logger.info(f"   Throughput: {throughput:.1f} req/s")
    logger.info(f"   Batch latency: {batch_latency_ms:.2f} ms (batch_size={batch_size})")
    
    # === 4. Memory Usage ===
    logger.info("\n4. Measuring memory usage...")
    memory_mb = get_memory_usage_mb()
    logger.info(f"   Memory usage: {memory_mb:.1f} MB")
    
    return BenchmarkResult(
        model_load_time_ms=load_time_ms,
        single_inference_latency_ms=p50_latency,
        batch_throughput_rps=throughput,
        memory_usage_mb=memory_mb,
        batch_latency_ms=batch_latency_ms,
        p50_latency_ms=p50_latency,
        p95_latency_ms=p95_latency,
        p99_latency_ms=p99_latency,
    )


def print_comparison(arrow_result: BenchmarkResult, st_result: Optional[BenchmarkResult]):
    """Print comparison table."""
    logger.info("\n" + "=" * 60)
    logger.info("Performance Comparison")
    logger.info("=" * 60)
    
    if st_result is None:
        logger.info("\n" + str(arrow_result))
        logger.info("\n(sentence-transformers baseline not available)")
        return
    
    # Calculate speedups
    load_speedup = st_result.model_load_time_ms / arrow_result.model_load_time_ms
    latency_speedup = st_result.single_inference_latency_ms / arrow_result.single_inference_latency_ms
    throughput_speedup = arrow_result.batch_throughput_rps / st_result.batch_throughput_rps
    memory_reduction = (1 - arrow_result.memory_usage_mb / st_result.memory_usage_mb) * 100
    
    print("\n" + "=" * 80)
    print(f"{'Metric':<30} {'ArrowEngine':>15} {'ST Baseline':>15} {'Improvement':>15}")
    print("=" * 80)
    print(f"{'Model Load Time':<30} {arrow_result.model_load_time_ms:>13.2f} ms {st_result.model_load_time_ms:>13.2f} ms {load_speedup:>13.1f}x")
    print(f"{'Single Latency (P50)':<30} {arrow_result.p50_latency_ms:>13.2f} ms {st_result.p50_latency_ms:>13.2f} ms {latency_speedup:>13.1f}x")
    print(f"{'Single Latency (P95)':<30} {arrow_result.p95_latency_ms:>13.2f} ms {st_result.p95_latency_ms:>13.2f} ms")
    print(f"{'Batch Throughput':<30} {arrow_result.batch_throughput_rps:>11.1f} req/s {st_result.batch_throughput_rps:>11.1f} req/s {throughput_speedup:>13.1f}x")
    print(f"{'Memory Usage':<30} {arrow_result.memory_usage_mb:>13.1f} MB {st_result.memory_usage_mb:>13.1f} MB {memory_reduction:>12.1f}%")
    print("=" * 80)
    
    # Check targets
    print("\nPerformance Targets:")
    print(f"  Model Load Time < 100ms:     {'✅ PASS' if arrow_result.model_load_time_ms < 100 else '❌ FAIL'} ({arrow_result.model_load_time_ms:.2f} ms)")
    print(f"  Single Latency < 5ms:        {'✅ PASS' if arrow_result.single_inference_latency_ms < 5 else '❌ FAIL'} ({arrow_result.single_inference_latency_ms:.2f} ms)")
    print(f"  Batch Throughput > 2000 rps: {'✅ PASS' if arrow_result.batch_throughput_rps > 2000 else '❌ FAIL'} ({arrow_result.batch_throughput_rps:.1f} rps)")
    print(f"  Memory Usage < 100MB:        {'✅ PASS' if arrow_result.memory_usage_mb < 100 else '❌ FAIL'} ({arrow_result.memory_usage_mb:.1f} MB)")
    
    if arrow_result.meets_targets():
        print("\n🎉 All performance targets met!")
    else:
        print("\n⚠️  Some performance targets not met")


def main():
    """Run benchmark suite."""
    # Get model path
    model_path = os.environ.get("ARROW_MODEL_PATH", "./models/minilm")
    
    if not Path(model_path).exists():
        logger.error(f"ArrowEngine model not found at {model_path}")
        logger.error("Please convert the model first:")
        logger.error(
            f"python -m llm_compression.tools.cli convert "
            f"--model sentence-transformers/all-MiniLM-L6-v2 "
            f"--output {model_path} --float16 --validate"
        )
        sys.exit(1)
    
    # Run benchmarks
    try:
        arrow_result = benchmark_arrowengine(model_path)
    except Exception as e:
        logger.error(f"ArrowEngine benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        st_result = benchmark_sentence_transformers()
    except Exception as e:
        logger.warning(f"sentence-transformers benchmark failed: {e}")
        st_result = None
    
    # Print comparison
    print_comparison(arrow_result, st_result)
    
    # Exit with appropriate code
    if arrow_result.meets_targets():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

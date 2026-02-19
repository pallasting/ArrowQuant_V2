"""
Phase 1: End-to-End Performance Benchmark

Measures ArrowEngine performance across all key metrics:
- Model load time (vs sentence-transformers)
- Inference latency (single sequence)
- Throughput (batch processing)
- Memory usage

Usage:
    python tests/performance_benchmark.py
"""

import sys
import time
import tempfile
import tracemalloc
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SINGLE_TEXT = "The quick brown fox jumps over the lazy dog."
BATCH_TEXTS_32 = [f"Sample sentence number {i} for throughput testing." for i in range(32)]
BATCH_TEXTS_100 = [f"Sample sentence number {i} for throughput testing." for i in range(100)]


def benchmark_sentence_transformers():
    """Benchmark sentence-transformers as reference."""
    print("\n--- sentence-transformers Reference ---")
    from sentence_transformers import SentenceTransformer

    # Load time
    t0 = time.perf_counter()
    st = SentenceTransformer(MODEL_NAME)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Load time:       {load_ms:8.1f} ms")

    # Single inference latency (warmup + 10 runs)
    st.encode([SINGLE_TEXT])  # warmup
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        st.encode([SINGLE_TEXT])
        times.append((time.perf_counter() - t0) * 1000)
    lat_ms = np.mean(times)
    print(f"  Latency (1 seq): {lat_ms:8.2f} ms  (mean of 10)")

    # Throughput batch=32
    t0 = time.perf_counter()
    for _ in range(5):
        st.encode(BATCH_TEXTS_32)
    thr_ms = (time.perf_counter() - t0) * 1000 / 5
    thr_qps = 32 / (thr_ms / 1000)
    print(f"  Throughput b=32: {thr_qps:8.0f} seq/s  ({thr_ms:.1f} ms/batch)")

    # Memory
    tracemalloc.start()
    st2 = SentenceTransformer(MODEL_NAME)
    st2.encode(BATCH_TEXTS_32)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Peak memory:     {peak / 1024 / 1024:8.1f} MB")

    return {
        'load_ms': load_ms,
        'latency_ms': lat_ms,
        'throughput_qps': thr_qps,
        'peak_memory_mb': peak / 1024 / 1024,
    }


def convert_model(output_dir: str) -> bool:
    """Convert model to Arrow format."""
    from llm_compression.tools.model_converter import ModelConverter, ConversionConfig
    conv = ModelConverter(ConversionConfig(use_float16=True))
    result = conv.convert(MODEL_NAME, output_dir, 'sentence-transformers')
    if result.success:
        print(f"  Converted: {result.file_size_mb:.1f} MB, ratio {result.compression_ratio:.1f}x")
    return result.success


def benchmark_arrow_engine(model_dir: str):
    """Benchmark ArrowEngine."""
    print("\n--- ArrowEngine ---")
    from llm_compression.inference.arrow_engine import ArrowEngine

    # Load time
    t0 = time.perf_counter()
    engine = ArrowEngine(model_path=model_dir, device='cpu', normalize_embeddings=True)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Load time:       {load_ms:8.1f} ms")

    # Single inference latency (warmup + 10 runs)
    engine.encode([SINGLE_TEXT])  # warmup
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        engine.encode([SINGLE_TEXT])
        times.append((time.perf_counter() - t0) * 1000)
    lat_ms = np.mean(times)
    print(f"  Latency (1 seq): {lat_ms:8.2f} ms  (mean of 10)")

    # Throughput batch=32
    t0 = time.perf_counter()
    for _ in range(5):
        engine.encode(BATCH_TEXTS_32)
    thr_ms = (time.perf_counter() - t0) * 1000 / 5
    thr_qps = 32 / (thr_ms / 1000)
    print(f"  Throughput b=32: {thr_qps:8.0f} seq/s  ({thr_ms:.1f} ms/batch)")

    # Throughput batch=100
    t0 = time.perf_counter()
    for _ in range(3):
        engine.encode(BATCH_TEXTS_100)
    thr_ms_100 = (time.perf_counter() - t0) * 1000 / 3
    thr_qps_100 = 100 / (thr_ms_100 / 1000)
    print(f"  Throughput b=100:{thr_qps_100:8.0f} seq/s  ({thr_ms_100:.1f} ms/batch)")

    # Memory
    tracemalloc.start()
    engine2 = ArrowEngine(model_path=model_dir, device='cpu', normalize_embeddings=True)
    engine2.encode(BATCH_TEXTS_32)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Peak memory:     {peak / 1024 / 1024:8.1f} MB")

    return {
        'load_ms': load_ms,
        'latency_ms': lat_ms,
        'throughput_qps': thr_qps,
        'throughput_qps_100': thr_qps_100,
        'peak_memory_mb': peak / 1024 / 1024,
    }


def run_benchmark():
    print("=" * 55)
    print("  ArrowEngine Performance Benchmark — Phase 1")
    print("=" * 55)
    print(f"  Model: {MODEL_NAME}")

    # Reference
    st_metrics = benchmark_sentence_transformers()

    # Convert model
    print("\n--- Model Conversion ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = str(Path(tmpdir) / 'minilm')
        Path(model_dir).mkdir(parents=True)

        if not convert_model(model_dir):
            print("Conversion failed")
            return

        # ArrowEngine benchmark
        ae_metrics = benchmark_arrow_engine(model_dir)

    # Comparison
    print("\n" + "=" * 55)
    print("  COMPARISON SUMMARY")
    print("=" * 55)
    print(f"  {'Metric':<25} {'ST':>10} {'Arrow':>10} {'Speedup':>10}")
    print(f"  {'-'*55}")

    load_speedup = st_metrics['load_ms'] / ae_metrics['load_ms']
    lat_speedup = st_metrics['latency_ms'] / ae_metrics['latency_ms']
    thr_speedup = ae_metrics['throughput_qps'] / st_metrics['throughput_qps']

    print(f"  {'Load time (ms)':<25} {st_metrics['load_ms']:>10.1f} {ae_metrics['load_ms']:>10.1f} {load_speedup:>9.1f}x")
    print(f"  {'Latency/seq (ms)':<25} {st_metrics['latency_ms']:>10.2f} {ae_metrics['latency_ms']:>10.2f} {lat_speedup:>9.1f}x")
    print(f"  {'Throughput b=32 (seq/s)':<25} {st_metrics['throughput_qps']:>10.0f} {ae_metrics['throughput_qps']:>10.0f} {thr_speedup:>9.1f}x")
    print(f"  {'Peak memory (MB)':<25} {st_metrics['peak_memory_mb']:>10.1f} {ae_metrics['peak_memory_mb']:>10.1f}")

    # Targets
    print("\n  Targets:")
    targets = [
        ("Load time < 100ms", ae_metrics['load_ms'] < 100, f"{ae_metrics['load_ms']:.1f}ms"),
        ("Latency < 5ms", ae_metrics['latency_ms'] < 5, f"{ae_metrics['latency_ms']:.2f}ms"),
        ("Throughput > 2000 seq/s", ae_metrics['throughput_qps'] > 2000, f"{ae_metrics['throughput_qps']:.0f} seq/s"),
    ]
    for name, passed, value in targets:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {value}")

    print("=" * 55)


if __name__ == "__main__":
    run_benchmark()

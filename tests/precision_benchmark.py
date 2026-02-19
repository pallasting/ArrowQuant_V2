"""
Task 0.3: Precision Benchmark — ArrowEngine vs sentence-transformers

Validates that ArrowEngine produces embeddings with cosine similarity ≥ 0.99
compared to the reference sentence-transformers implementation.

Usage:
    python tests/precision_benchmark.py

Steps:
    1. Convert all-MiniLM-L6-v2 to Arrow/Parquet format
    2. Load with ArrowEngine
    3. Encode same texts with both engines
    4. Compute cosine similarity and report
"""

import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language for data science.",
    "The Eiffel Tower is located in Paris, France.",
    "Neural networks are inspired by the human brain.",
    "Climate change is one of the most pressing global issues.",
    "The stock market experienced significant volatility today.",
    "Quantum computing promises to revolutionize cryptography.",
    "The Amazon rainforest is home to millions of species.",
    "Space exploration has advanced significantly in recent decades.",
    "Renewable energy sources include solar, wind, and hydropower.",
    "The human genome contains approximately 3 billion base pairs.",
    "Artificial intelligence is transforming healthcare diagnostics.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Deep learning models require large amounts of training data.",
    "The Mediterranean diet is associated with longevity.",
    "Blockchain technology enables decentralized transactions.",
    "The James Webb Space Telescope captures stunning cosmic images.",
    "Electric vehicles are becoming increasingly affordable.",
    "Mindfulness meditation can reduce stress and anxiety.",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between corresponding rows of a and b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.sum(a_norm * b_norm, axis=1)


def encode_with_sentence_transformers(texts: list) -> np.ndarray:
    """Encode texts using sentence-transformers (reference implementation)."""
    print("\n[1] Loading sentence-transformers model...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(MODEL_NAME)
    print(f"    Loaded in {(time.time()-t0)*1000:.1f}ms")

    print("[1] Encoding with sentence-transformers...")
    t0 = time.time()
    embeddings = st_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    elapsed = (time.time() - t0) * 1000
    print(f"    Encoded {len(texts)} texts in {elapsed:.1f}ms ({elapsed/len(texts):.2f}ms/text)")
    print(f"    Embedding shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    return embeddings


def convert_model_to_arrow(output_dir: str) -> bool:
    """Convert the model to Arrow/Parquet format."""
    print(f"\n[2] Converting {MODEL_NAME} to Arrow format...")
    from llm_compression.tools.model_converter import ModelConverter, ConversionConfig

    config = ConversionConfig(
        compression="lz4",
        use_float16=True,
        extract_tokenizer=True,
        validate_output=True,
    )
    converter = ModelConverter(config=config)

    t0 = time.time()
    result = converter.convert(
        model_name_or_path=MODEL_NAME,
        output_dir=output_dir,
        model_type="sentence-transformers",
    )
    elapsed = time.time() - t0

    if result.success:
        print(f"    Conversion SUCCESS in {elapsed:.2f}s")
        print(f"    Parquet: {result.parquet_path} ({result.file_size_mb:.2f} MB)")
        print(f"    Parameters: {result.total_parameters:,}")
        print(f"    Compression ratio: {result.compression_ratio:.2f}x")
        print(f"    Validation: {'PASSED' if result.validation_passed else 'FAILED'}")
        return True
    else:
        print(f"    Conversion FAILED: {result.error_message}")
        return False


def encode_with_arrow_engine(texts: list, model_dir: str) -> np.ndarray:
    """Encode texts using ArrowEngine."""
    print("\n[3] Loading ArrowEngine...")
    from llm_compression.inference.arrow_engine import ArrowEngine

    t0 = time.time()
    engine = ArrowEngine(
        model_path=model_dir,
        device="cpu",
        normalize_embeddings=True,
    )
    load_time = (time.time() - t0) * 1000
    print(f"    Loaded in {load_time:.1f}ms")
    print(f"    Embedding dimension: {engine.get_embedding_dimension()}")

    print("[3] Encoding with ArrowEngine...")
    t0 = time.time()
    embeddings = engine.encode(texts)
    elapsed = (time.time() - t0) * 1000
    print(f"    Encoded {len(texts)} texts in {elapsed:.1f}ms ({elapsed/len(texts):.2f}ms/text)")

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    print(f"    Embedding shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    return embeddings


def run_benchmark():
    """Run the full precision benchmark."""
    print("=" * 60)
    print("  ArrowEngine Precision Benchmark — Task 0.3")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Test texts: {len(TEST_TEXTS)}")

    # Step 1: Get reference embeddings from sentence-transformers
    st_embeddings = encode_with_sentence_transformers(TEST_TEXTS)

    # Step 2: Convert model to Arrow format
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = str(Path(tmpdir) / "minilm")
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        if not convert_model_to_arrow(model_dir):
            print("\n❌ Conversion failed — cannot proceed with benchmark")
            return False

        # Step 3: Get ArrowEngine embeddings
        try:
            arrow_embeddings = encode_with_arrow_engine(TEST_TEXTS, model_dir)
        except Exception as e:
            print(f"\n❌ ArrowEngine encoding failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Step 4: Compute similarity metrics
    print("\n[4] Computing precision metrics...")

    # Cosine similarity between corresponding embeddings
    similarities = cosine_similarity(st_embeddings, arrow_embeddings)

    mean_sim = float(np.mean(similarities))
    min_sim = float(np.min(similarities))
    max_sim = float(np.max(similarities))
    std_sim = float(np.std(similarities))

    # L2 distance
    l2_distances = np.linalg.norm(st_embeddings - arrow_embeddings, axis=1)
    mean_l2 = float(np.mean(l2_distances))

    print("\n" + "=" * 60)
    print("  PRECISION RESULTS")
    print("=" * 60)
    print(f"  Cosine Similarity:")
    print(f"    Mean:  {mean_sim:.6f}  (target: ≥ 0.99)")
    print(f"    Min:   {min_sim:.6f}")
    print(f"    Max:   {max_sim:.6f}")
    print(f"    Std:   {std_sim:.6f}")
    print(f"  L2 Distance (mean): {mean_l2:.6f}")

    # Per-text breakdown
    print(f"\n  Per-text cosine similarities:")
    for i, (text, sim) in enumerate(zip(TEST_TEXTS, similarities)):
        status = "✅" if sim >= 0.99 else "⚠️" if sim >= 0.95 else "❌"
        print(f"    [{i+1:2d}] {status} {sim:.6f}  {text[:50]}...")

    # Final verdict
    target = 0.99
    passed = mean_sim >= target
    texts_above_target = int(np.sum(similarities >= target))

    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)
    print(f"  Mean cosine similarity: {mean_sim:.6f}")
    print(f"  Texts ≥ {target}: {texts_above_target}/{len(TEST_TEXTS)}")
    print(f"  Result: {'✅ PASSED' if passed else '❌ FAILED'} (target: mean ≥ {target})")
    print("=" * 60)

    return passed


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)

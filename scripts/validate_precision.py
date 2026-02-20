
"""
Precision Validation Script for AI-OS.
Compares optimized embeddings with a baseline to ensure quality is maintained.
"""

import torch
import numpy as np
from pathlib import Path
from llm_compression.inference import ArrowEngine
from llm_compression.logger import logger

def cosine_similarity(a, b):
    """Compute cosine similarity between two sets of embeddings."""
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    # Average cosine similarity across batch
    return np.mean(np.sum(a_norm * b_norm, axis=-1))

def validate_model_precision(model_path: str, model_name: str):
    logger.info(f"--- Validating Precision for {model_name} ---")
    
    test_sentences = [
        "Self-evolving memory systems are the future of AI operating systems.",
        "The quick brown fox jumps over the lazy dog.",
        "Precision and recall are key metrics in information retrieval.",
        "Multimodal fusion allows integration of vision and text data."
    ]
    
    # 1. Baseline: Eager loading on CPU, no extra optimizations
    engine_baseline = ArrowEngine(model_path, device="cpu", enable_intel_optimizations=False, lazy_load=False)
    # We force eager and no intel opt to get a "pure" baseline
    emb_baseline = engine_baseline.encode(test_sentences, normalize=True)
    
    logger.info(f"Baseline Embedding Shape: {emb_baseline.shape}")
    logger.info(f"Baseline Sample (first 5): {emb_baseline[0, :5]}")
    
    # 2. Optimized: Torch.compile + Intel Opts + Lazy Loading
    engine_optimized = ArrowEngine(model_path, device="cpu", enable_intel_optimizations=True, lazy_load=True)
    emb_optimized = engine_optimized.encode(test_sentences, normalize=True)
    
    logger.info(f"Optimized Embedding Shape: {emb_optimized.shape}")
    logger.info(f"Optimized Sample (first 5): {emb_optimized[0, :5]}")
    
    similarity = cosine_similarity(emb_baseline, emb_optimized)
    
    logger.info(f"Cosine Similarity (Baseline vs Optimized): {similarity:.6f}")
    
    if similarity > 0.999:
        logger.info("PASS: Precision is high.")
    elif similarity > 0.99:
        logger.warning("SUCCESS: Precision is within acceptable limits (>0.99).")
    else:
        logger.error(f"FAILURE: Precision drift detected! Similarity: {similarity:.6f}")
        
    return similarity

def main():
    # Use original worktree paths where weights actually exist
    original_models = Path("/Media/Ubuntu/Documents/Surface-Memory/Documents/ai-os-memory/models")
    results = {}
    
    # Validate MiniLM
    minilm_path = original_models / "minilm"
    if minilm_path.exists():
        results["minilm"] = validate_model_precision(str(minilm_path), "MiniLM-L6")
        
    # Validate Qwen-0.5B
    qwen_path = original_models / "qwen2.5-0.5b-arrow"
    if qwen_path.exists():
        results["qwen_0.5b"] = validate_model_precision(str(qwen_path), "Qwen2.5-0.5B")
        
    logger.info("Precision validation complete.")

if __name__ == "__main__":
    main()

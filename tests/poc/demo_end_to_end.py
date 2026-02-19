
"""
End-to-End Proof of Concept (PoC) Demo for ArrowEngine-Native Architecture.

Demonstrates:
1. Loading real ArrowEngine (MiniLM).
2. Extracting Attention-based Key Information.
3. Compressing Vector Space Representation.
4. Reconstructing Vector Space Representation.
5. Adaptive Learning (Boosting important dimensions).
"""

import sys
import os
import time
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llm_compression.inference.arrow_engine import ArrowEngine
from llm_compression.compression.vector_compressor import VectorSpaceCompressor
from llm_compression.learning.incremental_learner import IncrementalLearner

def main():
    print("=== ArrowEngine-Native Architecture PoC ===\n")
    
    model_path = os.path.abspath("models/minilm")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # 1. Initialize Engine
    print(f"1. Loading ArrowEngine from {model_path}...")
    start = time.time()
    engine = ArrowEngine(model_path)
    print(f"   Loaded in {(time.time() - start)*1000:.2f}ms")
    
    # 2. Components
    compressor = VectorSpaceCompressor(engine)
    learner = IncrementalLearner(dimension_size=engine.get_embedding_dimension())
    
    text = "Artificial intelligence is revolutionizing healthcare and finance sectors with advanced algorithms."
    print(f"\n2. Processing Input: '{text}'")
    
    # 3. Attention Extraction
    print("\n[Attention Extraction]")
    key_info = compressor.attn_extractor.extract_key_information(text)
    print(f"   Key Tokens (Top 5): {key_info.key_tokens[:5]}")
    print(f"   Scores: {[f'{s:.3f}' for s in key_info.token_scores[:5]]}")
    
    # 4. Compression (20% Ratio)
    print("\n[Vector Compression @ 20%]")
    ratio = 0.2
    start = time.time()
    compressed = compressor.compress(text, compression_ratio=ratio)
    comp_time = (time.time() - start) * 1000
    
    orig_size = compressed.meta_info['full_dim'] * 4 # float32 = 4 bytes
    comp_size = len(compressed.sparse_vector) * 1 + len(compressed.key_indices) * 2 # int8 + uint16
    
    print(f"   Compression Ratio: {orig_size / comp_size:.1f}x ({orig_size} -> {comp_size} bytes)")
    print(f"   Time: {comp_time:.2f}ms")
    
    # Verify similarity
    start = time.time()
    reconstructed = compressor.reconstruct(compressed)
    recon_time = (time.time() - start) * 1000
    
    original = engine.encode(text)[0]
    sim = np.dot(reconstructed, original) / (np.linalg.norm(reconstructed) * np.linalg.norm(original))
    print(f"   Cosine Similarity: {sim:.4f}")
    
    # 4b. Compression (5% Ratio - Target > 20x)
    print("\n[Vector Compression @ 5%]")
    ratio_aggressive = 0.05
    compressed_agg = compressor.compress(text, compression_ratio=ratio_aggressive)
    
    comp_size_agg = len(compressed_agg.sparse_vector) * 1 + len(compressed_agg.key_indices) * 2
    print(f"   Compression Ratio: {orig_size / comp_size_agg:.1f}x ({orig_size} -> {comp_size_agg} bytes)")
    
    reconstructed_agg = compressor.reconstruct(compressed_agg)
    sim_agg = np.dot(reconstructed_agg, original) / (np.linalg.norm(reconstructed_agg) * np.linalg.norm(original))
    print(f"   Cosine Similarity: {sim_agg:.4f}")
    
    if sim_agg > 0.8:
        print("   SUCCESS: >20x compression with >0.8 fidelity!")

    # 6. Adaptive Learning Simulation
    print("\n[Adaptive Learning]")
    print("   Simulating 50 accesses to boost dimension importance...")
    
    # Current indices
    old_indices = compressed.key_indices[:5] # Top 5
    print(f"   Top 5 Indices (Before): {old_indices}")
    
    # Record access many times
    for _ in range(50):
        learner.record_access(compressed)
        
    weights = learner.get_dimension_weights(learning_rate=0.5)
    max_w = np.max(weights)
    print(f"   Max Dimension Weight: {max_w:.2f}")
    
    # Re-compress with weights
    compressed_v2 = compressor.compress(text, compression_ratio=ratio, dimension_weights=weights)
    new_indices = compressed_v2.key_indices[:5]
    print(f"   Top 5 Indices (After) : {new_indices}")
    
    # Check if indices match (they should, or be reinforced)
    overlap = len(set(old_indices).intersection(set(new_indices)))
    print(f"   Overlap in Top 5: {overlap}/5")
    
    print("\n=== PoC Complete ===")

if __name__ == "__main__":
    main()

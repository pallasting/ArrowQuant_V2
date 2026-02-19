
"""
Localized Memory System Fidelity Benchmark.

Compares reconstruction fidelity and associative recall accuracy 
for the ArrowEngine-Native architecture.
"""

import sys
import time
import tempfile
import numpy as np
import torch
import shutil
from pathlib import Path
from typing import List, Dict

from llm_compression.arrow_storage import ArrowStorage
from llm_compression.vector_search import VectorSearch
from llm_compression.knowledge_graph import KnowledgeGraphManager, HybridNavigator
from llm_compression.arrow_native_compressor import ArrowNativeCompressor
from llm_compression.compression.vector_compressor import VectorSpaceCompressor
from llm_compression.compression.attention_extractor import AttentionBasedExtractor

# Test Data: Related clusters
BENCHMARK_DATA = [
    {
        "cluster": "AI & ML",
        "texts": [
            "Machine learning involves training models on large datasets.",
            "Neural networks are a fundamental part of deep learning.",
            "Supervised learning requires labeled data for training.",
            "Backpropagation is the core algorithm for training neural nets.",
            "Natural Language Processing uses transformers for text analysis."
        ]
    },
    {
        "cluster": "Programming",
        "texts": [
            "Python is a versatile language for data science and web development.",
            "JavaScript is primarily used for frontend web interactivity.",
            "Go is known for its concurrency features and efficiency.",
            "Rust provides memory safety without a garbage collector.",
            "SQL is the standard language for querying relational databases."
        ]
    }
]

def run_fidelity_benchmark():
    print("=" * 60)
    print("  Localized Memory Fidelity Benchmark")
    print("=" * 60)
    
    # 1. Setup Environment
    tmp_dir = tempfile.mkdtemp()
    test_path = Path(tmp_dir)
    
    # We need a real ArrowEngine for attention weights if we want realistic results
    # But for a fast benchmark with mock, we simulate.
    # Actually, let's use the actual default provider if available.
    from llm_compression.embedding_provider import get_default_provider
    try:
        provider = get_default_provider()
        # provider is usually ArrowEngineProvider -> engine
        engine = provider.engine if hasattr(provider, 'engine') else provider
    except:
        print("❌ Failed to load default provider. Skipping real benchmark.")
        return

    kg_manager = KnowledgeGraphManager(test_path)
    storage = ArrowStorage(storage_path=tmp_dir, kg_manager=kg_manager)
    vector_search = VectorSearch(provider, storage)
    extractor = AttentionBasedExtractor(engine)
    compressor = ArrowNativeCompressor(engine)
    hybrid = HybridNavigator(vector_search, kg_manager, extractor)
    
    vec_compressor = VectorSpaceCompressor(engine)
    
    print(f"  Provider: {provider}")
    print(f"  Storage: {tmp_dir}")
    
    # 2. Add memories
    all_mids = []
    print("\n[1] Compressing and Indexing Memories...")
    t0 = time.time()
    for cluster in BENCHMARK_DATA:
        for text in cluster['texts']:
            # Metadata can include cluster info
            compressed = compressor.compress(text)
            storage.save(compressed)
            all_mids.append(compressed.memory_id)
    
    elapsed = time.time() - t0
    print(f"    Indexed {len(all_mids)} memories in {elapsed:.2f}s")
    
    # 3. Precision Benchmark (Vector Reconstruction)
    print("\n[2] Evaluating 4-bit Reconstruction Fidelity...")
    avg_sim = 0.0
    count = 0
    for i, mid in enumerate(all_mids):
        mem = storage.load(mid)
        # Reconstruct vector
        reconstructed = vec_compressor.reconstruct(mem) # Returns numpy array
        original = np.array(mem.embedding, dtype=np.float32)
        
        # Cosine similarity
        sim = np.dot(original, reconstructed) / (np.linalg.norm(original) * np.linalg.norm(reconstructed) + 1e-9)
        avg_sim += sim
        count += 1
        
        if i == 0:
            print(f"      - Sample 0: norm_orig={np.linalg.norm(original):.4f}, norm_recon={np.linalg.norm(reconstructed):.4f}, sim={sim:.4f}")
            print(f"      - Original (first 10): {original[:10]}")
            print(f"      - Recon (first 10):    {reconstructed[:10]}")
    
    print(f"    Mean Reconstruction Similarity (4-bit): {avg_sim/count:.4f}")
    
    # 4. Association Benchmark (Associative Recall)
    # Scenario: Query for "Deep Learning" (which is in Cluster 0)
    # Should find other items in Cluster 0 via "neural networks" or "learning" association 
    # even if they don't have the words "deep learning".
    print("\n[3] Evaluating Associative Recall (Memetic Retrieval)...")
    
    test_query = "deep learning"
    # Find results from Hybrid search
    results = hybrid.search(test_query, top_k=5, alpha=0.3) # Low alpha to favor association
    
    print(f"    Query: '{test_query}'")
    found_clusters = []
    for r in results:
        # Check which cluster it belongs to
        diff_data = r.memory.diff_data
        try:
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            text = dctx.decompress(diff_data).decode('utf-8')
        except:
            text = str(r.memory.diff_data)
            
        cluster_found = "Unknown"
        for c in BENCHMARK_DATA:
            if any(t[:20] in text for t in c['texts']):
                cluster_found = c['cluster']
                break
        found_clusters.append(cluster_found)
        print(f"      - Score: {r.similarity:.4f} | {cluster_found:12} | {text[:50]}...")

    # Calculate intra-cluster recall@5
    target_cluster = "AI & ML"
    hits = found_clusters.count(target_cluster)
    print(f"\n    Intra-cluster Recall@5: {hits}/5 ({(hits/5)*100:.1f}%)")

    # Clean up
    shutil.rmtree(tmp_dir)
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)
    print(f"  Fidelity: {'[PASSED]' if avg_sim/count > 0.85 else '[LOW]'}")
    print(f"  Recall: {'[EXCELLENT]' if hits >= 4 else '[MODERATE]' if hits >= 2 else '[POOR]'}")
    print("=" * 60)

if __name__ == "__main__":
    run_fidelity_benchmark()

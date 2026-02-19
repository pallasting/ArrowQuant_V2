
import os
import sys
import time
import tempfile
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from llm_compression.embedding_provider import get_default_provider
from llm_compression.compression.vector_compressor import VectorSpaceCompressor
from llm_compression.knowledge_graph.manager import KnowledgeGraphManager
from llm_compression.arrow_storage import ArrowStorage

def run_full_pipeline_benchmark():
    print("=" * 60)
    print("  ArrowEngine-Native: Full Pipeline Benchmark")
    print("=" * 60)
    
    # Setup
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_path = Path(tmp_dir)
        kg_manager = KnowledgeGraphManager(storage_path)
        storage = ArrowStorage(storage_path, kg_manager=kg_manager)
        
        provider = get_default_provider()
        engine = provider.engine if hasattr(provider, 'engine') else provider
        compressor = VectorSpaceCompressor(engine)
        
        print(f"Device: {engine.device}")
        
        # Test Data
        corpus = [
            "The artificial intelligence revolution is transforming the global economy.",
            "Machine learning algorithms require large amounts of high-quality data for training.",
            "Neural networks and deep learning are the core technologies behind modern AI.",
            "Natural language processing allows computers to understand and generate human text.",
            "Vector databases are essential for storing and retrieving high-dimensional embeddings efficiently."
        ] * 10 # 50 documents
        
        print(f"\n[1] Processing {len(corpus)} documents...")
        
        # Iteration benchmark
        start_total = time.perf_counter()
        
        for i, text in enumerate(corpus):
            # A. Compression & Relation Extraction
            compressed = compressor.compress(text, use_4bit=True)
            
            # B. Storage & KG Update (triggers automatically in storage.save)
            # Fill required mock fields for storage compatibility
            compressed.memory_id = f"mem_{i}"
            compressed.summary_hash = f"hash_{i}"
            compressed.entities = {"keywords": compressed.key_tokens, "persons": [], "locations": [], "dates": [], "numbers": []}
            compressed.diff_data = b""
            compressed.embedding = [0.0] * 384
            from datetime import datetime
            from llm_compression.compressor import CompressionMetadata
            compressed.compression_metadata = CompressionMetadata(
                original_size=len(text), compressed_size=10, compression_ratio=1.0,
                model_used="native", quality_score=1.0, compression_time_ms=0,
                compressed_at=datetime.now()
            )
            compressed.original_fields = {}
            
            storage.save(compressed, category='experiences')
            
        duration_total = time.perf_counter() - start_total
        avg_per_doc = (duration_total / len(corpus)) * 1000
        
        print(f"Full pipeline time: {duration_total:.2f}s")
        print(f"Average time per document: {avg_per_doc:.2f}ms")
        
        # [2] Reasoning Benchmark
        print("\n[2] Reasoning Query Benchmark...")
        queries = ["artificial intelligence", "machine learning", "vector databases"]
        
        start_reason = time.perf_counter()
        for q in queries:
            kg_manager.find_related_memories([q], max_hops=2)
        duration_reason = (time.perf_counter() - start_reason) / len(queries) * 1000
        
        print(f"Average reasoning latency (2-hop): {duration_reason:.2f}ms")
        
        # Graph Stats
        print(f"\n[3] Final Knowledge Graph Stats:")
        print(f"  Nodes: {kg_manager.graph.number_of_nodes()}")
        print(f"  Edges: {kg_manager.graph.number_of_edges()}")
        
        print("\n[PASS] Pipeline benchmark completed.")

if __name__ == "__main__":
    run_full_pipeline_benchmark()

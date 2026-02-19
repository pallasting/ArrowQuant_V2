"""
ArrowStorage Integration Test

Test if ArrowStorage correctly handles both CompressedMemory and StoredMemory types.
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

def test_arrow_storage_integration():
    """Test ArrowStorage integration with both memory types"""
    print("\n" + "=" * 60)
    print("ArrowStorage Integration Test")
    print("=" * 60)
    
    # Use temporary test directory
    test_storage_dir = Path("./test_storage_validation")
    
    # Clean up old test data
    if test_storage_dir.exists():
        print(f"\nCleaning up old test data...")
        shutil.rmtree(test_storage_dir)
    
    try:
        from llm_compression.embedding_provider import get_default_provider
        from llm_compression.arrow_storage import ArrowStorage
        from llm_compression.stored_memory import StoredMemory, SemanticIndex, Entity
        from llm_compression.compressor import CompressedMemory, CompressionMetadata
        
        # Initialize provider and storage
        print(f"\nInitializing EmbeddingProvider and ArrowStorage...")
        provider = get_default_provider()
        storage = ArrowStorage(storage_path=test_storage_dir)
        
        print(f"v EmbeddingProvider: {type(provider).__name__}")
        print(f"v Embedding dimension: {provider.dimension}")
        print(f"v Storage path: {test_storage_dir}")
        
        # Test 1: Create and save StoredMemory
        print(f"\n--- Test 1: StoredMemory ---")
        test_text = "Machine learning is a subset of artificial intelligence."
        
        # Create embedding
        embedding = provider.encode(test_text)
        print(f"v Generated embedding: shape={embedding.shape}")
        
        # Create semantic index
        semantic_index = SemanticIndex(
            summary="ML is part of AI",
            entities=[
                Entity(name="machine learning", type="TECH", confidence=0.95),
                Entity(name="artificial intelligence", type="TECH", confidence=0.90)
            ],
            topics=["AI", "ML", "technology"],
            indexed_at=datetime.now(),
            model_used="test",
            quality_score=0.85
        )
        
        # Create StoredMemory
        stored_memory = StoredMemory(
            id="test_stored_001",
            created_at=datetime.now(),
            original_compressed=test_text.encode('utf-8'),
            semantic_index=semantic_index,
            embedding=embedding,
            metadata={"source": "test", "category": "tech"},
            key_tokens=["machine", "learning", "artificial", "intelligence"]
        )
        
        print(f"v Created StoredMemory: {stored_memory.id}")
        
        # Save to ArrowStorage
        storage.save(stored_memory, category='experiences')
        print(f"v Saved StoredMemory to ArrowStorage")
        
        # Load back
        loaded = storage.load(stored_memory.id, category='experiences')
        if loaded:
            print(f"v Loaded memory: {loaded.memory_id}")
            print(f"  - Original size: {loaded.compression_metadata.original_size} bytes")
            print(f"  - Embedding dim: {len(loaded.embedding)}")
            print(f"  - Key tokens: {loaded.key_tokens}")
        else:
            print(f"X Failed to load memory")
            return 1
        
        # Test 2: Create and save CompressedMemory
        print(f"\n--- Test 2: CompressedMemory ---")
        
        compressed_memory = CompressedMemory(
            memory_id="test_compressed_001",
            summary_hash="abc123",
            entities={
                'persons': ['Alice', 'Bob'],
                'keywords': ['test', 'compression']
            },
            diff_data=b"compressed diff data",
            embedding=embedding.tolist(),
            compression_metadata=CompressionMetadata(
                original_size=100,
                compressed_size=50,
                compression_ratio=2.0,
                model_used="test_model",
                quality_score=0.9,
                compression_time_ms=10.0,
                compressed_at=datetime.now()
            ),
            original_fields={"source": "test"},
            key_tokens=["test", "compression"],
            token_scores=[0.8, 0.7]
        )
        
        print(f"v Created CompressedMemory: {compressed_memory.memory_id}")
        
        # Save to ArrowStorage
        storage.save(compressed_memory, category='experiences')
        print(f"v Saved CompressedMemory to ArrowStorage")
        
        # Load back
        loaded2 = storage.load(compressed_memory.memory_id, category='experiences')
        if loaded2:
            print(f"v Loaded memory: {loaded2.memory_id}")
            print(f"  - Compression ratio: {loaded2.compression_metadata.compression_ratio:.2f}x")
            print(f"  - Entities: {loaded2.entities}")
        else:
            print(f"X Failed to load memory")
            return 1
        
        # Test 3: Query all memories
        print(f"\n--- Test 3: Query All Memories ---")
        all_memories = storage.query(category='experiences')
        print(f"v Found {len(all_memories)} memories in storage")
        
        for mem in all_memories:
            print(f"  - {mem.memory_id}: {mem.compression_metadata.model_used}")
        
        # Test 4: Similarity search
        print(f"\n--- Test 4: Similarity Search ---")
        query_text = "What is AI?"
        query_embedding = provider.encode(query_text).tolist()
        
        similar = storage.query_by_similarity(
            category='experiences',
            query_embedding=query_embedding,
            top_k=2
        )
        
        print(f"v Found {len(similar)} similar memories")
        for mem, score in similar:
            print(f"  - {mem.memory_id}: similarity={score:.4f}")
        
        print(f"\n" + "=" * 60)
        print("v ArrowStorage integration test completed")
        print(f"\nVerified functionality:")
        print(f"  v StoredMemory save/load")
        print(f"  v CompressedMemory save/load")
        print(f"  v Query all memories")
        print(f"  v Similarity search")
        
        # Clean up test data
        if test_storage_dir.exists():
            shutil.rmtree(test_storage_dir)
            print(f"\nv Cleaned up test data")
        
        return 0
        
    except Exception as e:
        print(f"\nX Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up test data
        if test_storage_dir.exists():
            try:
                shutil.rmtree(test_storage_dir)
            except:
                pass
        
        return 1

if __name__ == "__main__":
    sys.exit(test_arrow_storage_integration())

"""
Arrow Storage Example

Demonstrates the Arrow/Parquet-based storage layer for compressed memories.

Features demonstrated:
- Saving and loading compressed memories
- Querying with filters and sorting
- Time-based queries
- Entity-based queries
- Similarity-based queries
- Summary deduplication
- Multiple memory categories

Requirements: 8.1-8.6
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from llm_compression import (
    ArrowStorage,
    CompressedMemory,
    CompressionMetadata,
    LLMClient,
    LLMCompressor,
    ModelSelector
)


async def main():
    """Demonstrate Arrow storage functionality"""
    
    print("=" * 80)
    print("Arrow Storage Example")
    print("=" * 80)
    print()
    
    # Create temporary storage for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ArrowStorage(storage_path=tmpdir)
        
        print(f"✓ Initialized storage at: {tmpdir}")
        print()
        
        # ====================================================================
        # 1. Create and save compressed memories
        # ====================================================================
        print("1. Creating and saving compressed memories...")
        print("-" * 80)
        
        memories = []
        for i in range(5):
            memory = CompressedMemory(
                memory_id=f"memory_{i:03d}",
                summary_hash=f"hash_{i % 3}",  # Some duplicates for dedup demo
                entities={
                    'persons': [f'Person_{i}', f'Colleague_{i}'],
                    'dates': [f'2024-01-{15+i:02d}', f'{10+i}am'],
                    'numbers': [f'{100+i*10}', f'${50+i*5}'],
                    'locations': ['Office', 'Conference Room'][i % 2],
                    'keywords': ['meeting', 'project', 'deadline', 'review'][i % 4]
                },
                diff_data=f"Additional details for memory {i}".encode('utf-8'),
                embedding=[0.1 * (i + 1)] * 384,
                compression_metadata=CompressionMetadata(
                    original_size=1000 + i * 100,
                    compressed_size=100 + i * 10,
                    compression_ratio=(1000 + i * 100) / (100 + i * 10),
                    model_used="gpt-4" if i % 2 == 0 else "claude-3",
                    quality_score=0.90 + i * 0.01,
                    compression_time_ms=50.0 + i * 5,
                    compressed_at=datetime.now() - timedelta(days=i)
                ),
                original_fields={
                    'intent': f'intent_{i}',
                    'action': f'action_{i}',
                    'outcome': f'outcome_{i}',
                    'success': i % 2 == 0,
                    'related_memories': [f'memory_{j:03d}' for j in range(max(0, i-2), i)]
                }
            )
            memories.append(memory)
            storage.save(memory, category='experiences')
        
        print(f"✓ Saved {len(memories)} memories to 'experiences' category")
        print()
        
        # ====================================================================
        # 2. Load individual memory
        # ====================================================================
        print("2. Loading individual memory...")
        print("-" * 80)
        
        loaded = storage.load("memory_002", category='experiences')
        if loaded:
            print(f"✓ Loaded memory: {loaded.memory_id}")
            print(f"  - Summary hash: {loaded.summary_hash}")
            print(f"  - Entities: {len(loaded.entities['persons'])} persons, "
                  f"{len(loaded.entities['dates'])} dates")
            print(f"  - Compression ratio: {loaded.compression_metadata.compression_ratio:.2f}x")
            print(f"  - Quality score: {loaded.compression_metadata.quality_score:.2f}")
        print()
        
        # ====================================================================
        # 3. Query with filters
        # ====================================================================
        print("3. Querying with filters...")
        print("-" * 80)
        
        # Query all compressed memories
        results = storage.query(
            category='experiences',
            filters={'is_compressed': True}
        )
        print(f"✓ Found {len(results)} compressed memories")
        
        # Query with limit
        results = storage.query(
            category='experiences',
            limit=3
        )
        print(f"✓ Limited query returned {len(results)} results")
        print()
        
        # ====================================================================
        # 4. Query with sorting
        # ====================================================================
        print("4. Querying with sorting...")
        print("-" * 80)
        
        results = storage.query(
            category='experiences',
            sort_by='memory_id',
            ascending=True,
            limit=3
        )
        print(f"✓ Sorted query returned {len(results)} results:")
        for r in results:
            print(f"  - {r.memory_id}")
        print()
        
        # ====================================================================
        # 5. Time-based query
        # ====================================================================
        print("5. Time-based query...")
        print("-" * 80)
        
        now = datetime.now()
        results = storage.query_by_time_range(
            category='experiences',
            start_time=now - timedelta(days=2),
            end_time=now,
            limit=10
        )
        print(f"✓ Found {len(results)} memories from last 2 days")
        for r in results:
            age_days = (now - r.compression_metadata.compressed_at).days
            print(f"  - {r.memory_id}: {age_days} days ago")
        print()
        
        # ====================================================================
        # 6. Entity-based query
        # ====================================================================
        print("6. Entity-based query...")
        print("-" * 80)
        
        results = storage.query_by_entity(
            category='experiences',
            entity_type='persons',
            entity_value='Person_2',
            limit=5
        )
        print(f"✓ Found {len(results)} memories mentioning 'Person_2'")
        for r in results:
            print(f"  - {r.memory_id}: {r.entities['persons']}")
        print()
        
        # ====================================================================
        # 7. Similarity-based query
        # ====================================================================
        print("7. Similarity-based query...")
        print("-" * 80)
        
        # Query using embedding from memory_001
        query_embedding = [0.2] * 384
        results = storage.query_by_similarity(
            category='experiences',
            query_embedding=query_embedding,
            top_k=3,
            threshold=0.5
        )
        print(f"✓ Found {len(results)} similar memories:")
        for memory, similarity in results:
            print(f"  - {memory.memory_id}: similarity={similarity:.4f}")
        print()
        
        # ====================================================================
        # 8. Summary deduplication
        # ====================================================================
        print("8. Summary deduplication...")
        print("-" * 80)
        
        # Count unique summary hashes
        unique_hashes = set(m.summary_hash for m in memories)
        print(f"✓ {len(memories)} memories use {len(unique_hashes)} unique summaries")
        print(f"  - Deduplication ratio: {len(memories) / len(unique_hashes):.2f}x")
        print()
        
        # ====================================================================
        # 9. Multiple categories
        # ====================================================================
        print("9. Multiple memory categories...")
        print("-" * 80)
        
        # Save to different categories
        categories = ['identity', 'preferences', 'context']
        for category in categories:
            memory = CompressedMemory(
                memory_id=f"{category}_001",
                summary_hash=f"{category}_hash",
                entities={'persons': [], 'dates': [], 'numbers': [], 'locations': [], 'keywords': []},
                diff_data=f"{category} data".encode('utf-8'),
                embedding=[0.5] * 384,
                compression_metadata=CompressionMetadata(
                    original_size=500,
                    compressed_size=50,
                    compression_ratio=10.0,
                    model_used="gpt-4",
                    quality_score=0.95,
                    compression_time_ms=30.0,
                    compressed_at=datetime.now()
                ),
                original_fields={}
            )
            storage.save(memory, category=category)
            print(f"✓ Saved memory to '{category}' category")
        print()
        
        # ====================================================================
        # 10. Storage statistics
        # ====================================================================
        print("10. Storage statistics...")
        print("-" * 80)
        
        # Calculate total storage
        total_original = sum(m.compression_metadata.original_size for m in memories)
        total_compressed = sum(m.compression_metadata.compressed_size for m in memories)
        avg_ratio = total_original / total_compressed if total_compressed > 0 else 0
        
        print(f"✓ Storage statistics:")
        print(f"  - Total memories: {len(memories)}")
        print(f"  - Original size: {total_original:,} bytes")
        print(f"  - Compressed size: {total_compressed:,} bytes")
        print(f"  - Average compression ratio: {avg_ratio:.2f}x")
        print(f"  - Space saved: {(1 - total_compressed/total_original)*100:.1f}%")
        print()
        
        print("=" * 80)
        print("✓ Arrow Storage Example Complete!")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

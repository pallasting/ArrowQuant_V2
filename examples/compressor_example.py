"""
LLM Compressor Example

Demonstrates how to use the LLMCompressor for semantic compression.

Feature: llm-compression-integration
Requirements: 5.1-5.7
"""

import asyncio
from llm_compression.compressor import LLMCompressor
from llm_compression.llm_client import LLMClient
from llm_compression.model_selector import ModelSelector, MemoryType, QualityLevel
from llm_compression.logger import logger


async def basic_compression_example():
    """Basic compression example"""
    print("\n" + "="*80)
    print("Example 1: Basic Compression")
    print("="*80)
    
    # Initialize components
    llm_client = LLMClient(
        endpoint="http://localhost:8045",
        timeout=30.0,
        max_retries=3
    )
    
    model_selector = ModelSelector(
        cloud_endpoint="http://localhost:8045",
        prefer_local=False
    )
    
    compressor = LLMCompressor(
        llm_client=llm_client,
        model_selector=model_selector,
        min_compress_length=100
    )
    
    # Sample text to compress
    text = """
    John Smith met Mary Johnson at 3pm on 2024-01-15 to discuss the quarterly project review.
    They reviewed the budget of $125,000 and discussed the timeline for Q1 2024.
    The meeting took place at the downtown office and lasted approximately 2 hours.
    Key decisions included hiring 3 new developers and extending the deadline by 2 weeks.
    """
    
    print(f"\nOriginal text ({len(text)} chars):")
    print(text)
    
    try:
        # Compress the text
        compressed = await compressor.compress(text, memory_type=MemoryType.TEXT)
        
        print(f"\nCompression Results:")
        print(f"  Memory ID: {compressed.memory_id}")
        print(f"  Summary Hash: {compressed.summary_hash}")
        print(f"  Original Size: {compressed.compression_metadata.original_size} bytes")
        print(f"  Compressed Size: {compressed.compression_metadata.compressed_size} bytes")
        print(f"  Compression Ratio: {compressed.compression_metadata.compression_ratio:.2f}x")
        print(f"  Model Used: {compressed.compression_metadata.model_used}")
        print(f"  Compression Time: {compressed.compression_metadata.compression_time_ms:.2f}ms")
        
        print(f"\nExtracted Entities:")
        for entity_type, entities in compressed.entities.items():
            if entities:
                print(f"  {entity_type}: {entities}")
        
        print(f"\nEmbedding: {len(compressed.embedding)} dimensions")
        print(f"  First 5 values: {compressed.embedding[:5]}")
        
    except Exception as e:
        print(f"\nError during compression: {e}")
    finally:
        await llm_client.close()


async def short_text_example():
    """Short text handling example"""
    print("\n" + "="*80)
    print("Example 2: Short Text Handling")
    print("="*80)
    
    # Initialize components
    llm_client = LLMClient(
        endpoint="http://localhost:8045",
        timeout=30.0
    )
    
    model_selector = ModelSelector(
        cloud_endpoint="http://localhost:8045"
    )
    
    compressor = LLMCompressor(
        llm_client=llm_client,
        model_selector=model_selector,
        min_compress_length=100  # Texts shorter than this won't be compressed
    )
    
    # Short text (< 100 chars)
    short_text = "This is a short text that won't be compressed."
    
    print(f"\nShort text ({len(short_text)} chars):")
    print(short_text)
    
    try:
        # Compress (will store uncompressed)
        compressed = await compressor.compress(short_text)
        
        print(f"\nResults:")
        print(f"  Model Used: {compressed.compression_metadata.model_used}")
        print(f"  Compression Ratio: {compressed.compression_metadata.compression_ratio:.2f}x")
        print(f"  Summary Hash: '{compressed.summary_hash}' (empty for uncompressed)")
        
        if compressed.compression_metadata.model_used == "uncompressed":
            print("\n✓ Text was stored uncompressed (too short)")
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        await llm_client.close()


async def batch_compression_example():
    """Batch compression example"""
    print("\n" + "="*80)
    print("Example 3: Batch Compression")
    print("="*80)
    
    # Initialize components
    llm_client = LLMClient(
        endpoint="http://localhost:8045",
        timeout=30.0
    )
    
    model_selector = ModelSelector(
        cloud_endpoint="http://localhost:8045"
    )
    
    compressor = LLMCompressor(
        llm_client=llm_client,
        model_selector=model_selector,
        min_compress_length=50
    )
    
    # Multiple texts to compress
    texts = [
        "Alice called Bob at 5pm on 2024-01-16 to review the code changes. " * 2,
        "Charlie emailed David at 9am on 2024-01-17 to plan the sprint. " * 2,
        "Eve met Frank at 2pm on 2024-01-18 to discuss the architecture. " * 2,
    ]
    
    print(f"\nCompressing {len(texts)} texts...")
    
    try:
        # Batch compress
        import time
        start_time = time.time()
        
        compressed_list = await compressor.compress_batch(texts, memory_type=MemoryType.TEXT)
        
        elapsed_time = time.time() - start_time
        
        print(f"\nBatch Compression Complete:")
        print(f"  Total Time: {elapsed_time*1000:.2f}ms")
        print(f"  Average Time per Text: {elapsed_time*1000/len(texts):.2f}ms")
        
        print(f"\nResults:")
        for i, compressed in enumerate(compressed_list):
            print(f"\n  Text {i+1}:")
            print(f"    Original: {compressed.compression_metadata.original_size} bytes")
            print(f"    Compressed: {compressed.compression_metadata.compressed_size} bytes")
            print(f"    Ratio: {compressed.compression_metadata.compression_ratio:.2f}x")
            print(f"    Entities: {sum(len(v) for v in compressed.entities.values())} total")
        
        # Calculate average compression ratio
        avg_ratio = sum(c.compression_metadata.compression_ratio for c in compressed_list) / len(compressed_list)
        print(f"\n  Average Compression Ratio: {avg_ratio:.2f}x")
        
    except Exception as e:
        print(f"\nError during batch compression: {e}")
    finally:
        await llm_client.close()


async def entity_extraction_example():
    """Entity extraction example"""
    print("\n" + "="*80)
    print("Example 4: Entity Extraction")
    print("="*80)
    
    # Initialize components
    llm_client = LLMClient(
        endpoint="http://localhost:8045",
        timeout=30.0
    )
    
    model_selector = ModelSelector(
        cloud_endpoint="http://localhost:8045"
    )
    
    compressor = LLMCompressor(
        llm_client=llm_client,
        model_selector=model_selector
    )
    
    # Text with various entity types
    text = """
    On January 15, 2024, John Smith and Mary Johnson met at the San Francisco office.
    They discussed the Q1 budget of $250,000 and reviewed the timeline.
    The meeting started at 3:30pm and ended at 5pm.
    Key metrics included 95% completion rate and 1,234 active users.
    """
    
    print(f"\nText:")
    print(text)
    
    # Extract entities (without full compression)
    entities = compressor._extract_entities(text)
    
    print(f"\nExtracted Entities:")
    print(f"  Persons: {entities['persons']}")
    print(f"  Dates: {entities['dates']}")
    print(f"  Numbers: {entities['numbers']}")
    print(f"  Locations: {entities['locations']}")
    print(f"  Keywords: {entities['keywords']}")
    
    await llm_client.close()


async def compression_metadata_example():
    """Compression metadata example"""
    print("\n" + "="*80)
    print("Example 5: Compression with Metadata")
    print("="*80)
    
    # Initialize components
    llm_client = LLMClient(
        endpoint="http://localhost:8045",
        timeout=30.0
    )
    
    model_selector = ModelSelector(
        cloud_endpoint="http://localhost:8045"
    )
    
    compressor = LLMCompressor(
        llm_client=llm_client,
        model_selector=model_selector
    )
    
    # Text with custom metadata
    text = "Project meeting to discuss Q1 goals and objectives. " * 5
    
    from datetime import datetime
    metadata = {
        "timestamp": datetime.now(),
        "intent": "meeting_notes",
        "context": "quarterly_review",
        "success": True
    }
    
    print(f"\nCompressing text with metadata...")
    print(f"Metadata: {metadata}")
    
    try:
        compressed = await compressor.compress(
            text,
            memory_type=MemoryType.TEXT,
            metadata=metadata
        )
        
        print(f"\nCompression Complete:")
        print(f"  Memory ID: {compressed.memory_id}")
        print(f"  Compression Ratio: {compressed.compression_metadata.compression_ratio:.2f}x")
        
        print(f"\nOriginal Fields (metadata):")
        for key, value in compressed.original_fields.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        await llm_client.close()


async def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("LLM Compressor Examples")
    print("="*80)
    print("\nThese examples demonstrate the LLMCompressor functionality.")
    print("Make sure the LLM API is running on http://localhost:8045")
    print("\nPress Ctrl+C to stop at any time.")
    
    try:
        # Run examples
        await basic_compression_example()
        await short_text_example()
        await batch_compression_example()
        await entity_extraction_example()
        await compression_metadata_example()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

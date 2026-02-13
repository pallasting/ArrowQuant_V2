"""
Test Arrow storage engine
"""
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_core import MemoryStore


@pytest.fixture
def memory_store(tmp_path):
    """Create temporary memory store"""
    return MemoryStore(storage_path=str(tmp_path / "test_memories"))


def test_store_and_retrieve(memory_store):
    """Test basic store and retrieve"""
    
    # Store a memory
    memory_store.store(
        memory_id="test-001",
        content="Had a great conversation about AI memory systems",
        context="Meeting with team",
        metadata={"importance": "high"}
    )
    
    # Retrieve it
    memories = memory_store.retrieve(memory_id="test-001")
    
    assert len(memories) == 1
    assert memories[0]['id'] == "test-001"
    assert "AI memory systems" in memories[0]['content']
    assert memories[0]['metadata']['importance'] == "high"


def test_multiple_memories(memory_store):
    """Test storing multiple memories"""
    
    # Store multiple memories
    for i in range(5):
        memory_store.store(
            memory_id=f"test-{i:03d}",
            content=f"Memory content {i}",
            context=f"Context {i}"
        )
    
    # Retrieve all
    memories = memory_store.retrieve()
    
    assert len(memories) == 5


def test_storage_stats(memory_store):
    """Test storage statistics"""
    
    # Initially empty
    stats = memory_store.get_stats()
    assert stats['total_memories'] == 0
    
    # Add some memories
    for i in range(10):
        memory_store.store(
            memory_id=f"test-{i:03d}",
            content=f"Memory content {i}" * 100,  # Make it bigger
        )
    
    # Check stats
    stats = memory_store.get_stats()
    assert stats['total_memories'] == 10
    assert stats['storage_size_bytes'] > 0
    
    print(f"\nStorage stats: {stats}")


def test_embedding_storage(memory_store):
    """Test storing embeddings"""
    
    # Create fake embedding
    embedding = [0.1] * 1536
    
    memory_store.store(
        memory_id="test-embedding",
        content="Test with embedding",
        embedding=embedding
    )
    
    memories = memory_store.retrieve(memory_id="test-embedding")
    
    assert len(memories) == 1
    assert len(memories[0]['embedding']) == 1536
    assert memories[0]['embedding'][0] == pytest.approx(0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

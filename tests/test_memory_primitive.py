"""
Unit tests for MemoryPrimitive (Task 33)
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from llm_compression.memory_primitive import MemoryPrimitive
from llm_compression.compressor import CompressedMemory


@pytest.fixture
def sample_compressed_memory():
    """Create a sample compressed memory for testing."""
    from llm_compression.compressor import CompressionMetadata
    from datetime import datetime
    
    metadata = CompressionMetadata(
        original_size=100,
        compressed_size=10,
        compression_ratio=10.0,
        model_used="test_model",
        quality_score=0.9,
        compression_time_ms=100.0,
        compressed_at=datetime.now()
    )
    
    return CompressedMemory(
        memory_id="test_memory_001",
        summary_hash="abc123",
        entities={},
        diff_data=b"test diff data",
        embedding=[0.1] * 384,
        compression_metadata=metadata
    )


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    return np.random.rand(384)  # Standard BERT embedding size


@pytest.fixture
def memory_primitive(sample_compressed_memory, sample_embedding):
    """Create a sample MemoryPrimitive for testing."""
    return MemoryPrimitive(
        id="mem_001",
        content=sample_compressed_memory,
        embedding=sample_embedding
    )


class TestMemoryPrimitiveCreation:
    """Test MemoryPrimitive creation and initialization."""
    
    def test_create_memory_primitive(self, memory_primitive):
        """Test basic creation."""
        assert memory_primitive.id == "mem_001"
        assert memory_primitive.activation == 0.0
        assert memory_primitive.access_count == 0
        assert memory_primitive.success_count == 0
        assert len(memory_primitive.connections) == 0
        assert memory_primitive.created_at is not None
    
    def test_embedding_shape(self, memory_primitive):
        """Test embedding is properly stored."""
        assert memory_primitive.embedding.shape == (384,)
        assert isinstance(memory_primitive.embedding, np.ndarray)


class TestActivation:
    """Test activation and decay mechanisms."""
    
    def test_activate_memory(self, memory_primitive):
        """Test activating a memory."""
        memory_primitive.activate(0.5)
        
        assert memory_primitive.activation == 0.5
        assert memory_primitive.access_count == 1
        assert memory_primitive.last_access is not None
    
    def test_multiple_activations(self, memory_primitive):
        """Test multiple activations accumulate."""
        memory_primitive.activate(0.3)
        memory_primitive.activate(0.4)
        
        assert memory_primitive.activation == 0.7
        assert memory_primitive.access_count == 2
    
    def test_activation_cap(self, memory_primitive):
        """Test activation is capped at 1.0."""
        memory_primitive.activate(0.8)
        memory_primitive.activate(0.5)
        
        assert memory_primitive.activation == 1.0
    
    def test_decay(self, memory_primitive):
        """Test activation decay."""
        memory_primitive.activate(1.0)
        memory_primitive.decay(rate=0.1)
        
        assert memory_primitive.activation == pytest.approx(0.9)
    
    def test_decay_floor(self, memory_primitive):
        """Test decay doesn't go below 0."""
        memory_primitive.activation = 0.05
        memory_primitive.decay(rate=0.5)
        
        assert memory_primitive.activation >= 0.0


class TestSuccessTracking:
    """Test success rate tracking."""
    
    def test_initial_success_rate(self, memory_primitive):
        """Test initial success rate is 0."""
        assert memory_primitive.get_success_rate() == 0.0
    
    def test_record_success(self, memory_primitive):
        """Test recording success."""
        memory_primitive.activate(0.5)
        memory_primitive.record_success()
        
        assert memory_primitive.success_count == 1
        assert memory_primitive.get_success_rate() == 1.0
    
    def test_partial_success_rate(self, memory_primitive):
        """Test partial success rate."""
        memory_primitive.activate(0.5)
        memory_primitive.record_success()
        
        memory_primitive.activate(0.5)
        # No success recorded for second access
        
        assert memory_primitive.get_success_rate() == 0.5
    
    def test_success_rate_calculation(self, memory_primitive):
        """Test success rate with multiple accesses."""
        for _ in range(10):
            memory_primitive.activate(0.5)
        
        for _ in range(7):
            memory_primitive.record_success()
        
        assert memory_primitive.get_success_rate() == 0.7


class TestConnections:
    """Test connection management."""
    
    def test_add_connection(self, memory_primitive):
        """Test adding a connection."""
        memory_primitive.add_connection("mem_002", 0.5)
        
        assert "mem_002" in memory_primitive.connections
        assert memory_primitive.connections["mem_002"] == 0.5
    
    def test_strengthen_connection(self, memory_primitive):
        """Test strengthening existing connection."""
        memory_primitive.add_connection("mem_002", 0.3)
        memory_primitive.add_connection("mem_002", 0.4)
        
        assert memory_primitive.connections["mem_002"] == 0.7
    
    def test_connection_cap(self, memory_primitive):
        """Test connection strength is capped at 1.0."""
        memory_primitive.add_connection("mem_002", 0.8)
        memory_primitive.add_connection("mem_002", 0.5)
        
        assert memory_primitive.connections["mem_002"] == 1.0
    
    def test_get_connection_strength(self, memory_primitive):
        """Test getting connection strength."""
        memory_primitive.add_connection("mem_002", 0.6)
        
        assert memory_primitive.get_connection_strength("mem_002") == 0.6
        assert memory_primitive.get_connection_strength("mem_999") == 0.0
    
    def test_multiple_connections(self, memory_primitive):
        """Test managing multiple connections."""
        memory_primitive.add_connection("mem_002", 0.5)
        memory_primitive.add_connection("mem_003", 0.7)
        memory_primitive.add_connection("mem_004", 0.3)
        
        assert len(memory_primitive.connections) == 3
        assert memory_primitive.get_connection_strength("mem_003") == 0.7


class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_realistic_usage_pattern(self, memory_primitive):
        """Test realistic usage pattern."""
        # First access - successful
        memory_primitive.activate(0.8)
        memory_primitive.record_success()
        memory_primitive.add_connection("mem_002", 0.5)
        
        # Decay over time
        memory_primitive.decay(0.1)
        
        # Second access - not successful
        memory_primitive.activate(0.6)
        
        # Verify state
        assert memory_primitive.access_count == 2
        assert memory_primitive.success_count == 1
        assert memory_primitive.get_success_rate() == 0.5
        assert memory_primitive.activation > 0.0
        assert len(memory_primitive.connections) == 1

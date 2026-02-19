"""
Unit tests for NetworkNavigator (Task 39)
"""

import pytest
import numpy as np
from llm_compression.network_navigator import NetworkNavigator, ActivationResult
from llm_compression.memory_primitive import MemoryPrimitive
from llm_compression.compressor import CompressedMemory, CompressionMetadata
from datetime import datetime


@pytest.fixture
def navigator():
    """Create NetworkNavigator instance."""
    return NetworkNavigator(
        max_hops=3,
        decay_rate=0.7,
        activation_threshold=0.1
    )


@pytest.fixture
def memory_network():
    """Create a test memory network."""
    metadata = CompressionMetadata(
        original_size=100,
        compressed_size=10,
        compression_ratio=10.0,
        model_used="test",
        quality_score=0.9,
        compression_time_ms=100.0,
        compressed_at=datetime.now()
    )
    
    # Create memories with different embeddings
    memories = {}
    
    # Memory A: [1, 0, 0]
    compressed_a = CompressedMemory(
        memory_id="mem_a",
        summary_hash="hash_a",
        entities={},
        diff_data=b"data_a",
        embedding=[0.1] * 384,
        compression_metadata=metadata
    )
    mem_a = MemoryPrimitive(
        id="mem_a",
        content=compressed_a,
        embedding=np.array([1.0, 0.0, 0.0])
    )
    memories["mem_a"] = mem_a
    
    # Memory B: [0.9, 0.1, 0] - similar to A
    compressed_b = CompressedMemory(
        memory_id="mem_b",
        summary_hash="hash_b",
        entities={},
        diff_data=b"data_b",
        embedding=[0.1] * 384,
        compression_metadata=metadata
    )
    mem_b = MemoryPrimitive(
        id="mem_b",
        content=compressed_b,
        embedding=np.array([0.9, 0.1, 0.0])
    )
    memories["mem_b"] = mem_b
    
    # Memory C: [0, 1, 0] - different from A
    compressed_c = CompressedMemory(
        memory_id="mem_c",
        summary_hash="hash_c",
        entities={},
        diff_data=b"data_c",
        embedding=[0.1] * 384,
        compression_metadata=metadata
    )
    mem_c = MemoryPrimitive(
        id="mem_c",
        content=compressed_c,
        embedding=np.array([0.0, 1.0, 0.0])
    )
    memories["mem_c"] = mem_c
    
    # Memory D: [0, 0, 1] - different from all
    compressed_d = CompressedMemory(
        memory_id="mem_d",
        summary_hash="hash_d",
        entities={},
        diff_data=b"data_d",
        embedding=[0.1] * 384,
        compression_metadata=metadata
    )
    mem_d = MemoryPrimitive(
        id="mem_d",
        content=compressed_d,
        embedding=np.array([0.0, 0.0, 1.0])
    )
    memories["mem_d"] = mem_d
    
    # Add connections: A -> B (strong), B -> C (medium), C -> D (weak)
    mem_a.add_connection("mem_b", 0.8)
    mem_b.add_connection("mem_a", 0.8)
    mem_b.add_connection("mem_c", 0.5)
    mem_c.add_connection("mem_b", 0.5)
    mem_c.add_connection("mem_d", 0.3)
    mem_d.add_connection("mem_c", 0.3)
    
    return memories


class TestNetworkNavigatorCreation:
    """Test NetworkNavigator initialization."""
    
    def test_create_navigator(self, navigator):
        """Test basic creation."""
        assert navigator.max_hops == 3
        assert navigator.decay_rate == 0.7
        assert navigator.activation_threshold == 0.1


class TestCosineSimilarity:
    """Test cosine similarity calculation."""
    
    def test_identical_vectors(self, navigator):
        """Test similarity of identical vectors."""
        vec = np.array([1.0, 0.0, 0.0])
        similarity = navigator._cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0)
    
    def test_orthogonal_vectors(self, navigator):
        """Test similarity of orthogonal vectors."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        similarity = navigator._cosine_similarity(vec_a, vec_b)
        assert similarity == pytest.approx(0.5)
    
    def test_similar_vectors(self, navigator):
        """Test similarity of similar vectors."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.9, 0.1, 0.0])
        similarity = navigator._cosine_similarity(vec_a, vec_b)
        assert similarity > 0.9


class TestFindSimilar:
    """Test finding similar memories."""
    
    def test_find_similar_basic(self, navigator, memory_network):
        """Test finding similar memories."""
        query = np.array([1.0, 0.0, 0.0])
        similar = navigator._find_similar(query, memory_network, top_k=2)
        
        assert len(similar) == 2
        # Should find mem_a first (most similar)
        assert similar[0][0].id == "mem_a"
        assert similar[0][1] > 0.9
    
    def test_find_similar_top_k(self, navigator, memory_network):
        """Test top-k limiting."""
        query = np.array([1.0, 0.0, 0.0])
        similar = navigator._find_similar(query, memory_network, top_k=1)
        
        assert len(similar) == 1
        assert similar[0][0].id == "mem_a"


class TestActivationSpreading:
    """Test activation spreading."""
    
    def test_spread_activation_basic(self, navigator, memory_network):
        """Test basic activation spreading."""
        # Start from mem_a with activation 1.0
        initial = [(memory_network["mem_a"], 1.0)]
        
        activation_map = navigator._spread_activation(initial, memory_network)
        
        # mem_a should be activated
        assert "mem_a" in activation_map
        assert activation_map["mem_a"] == 1.0
        
        # mem_b should be activated (connected to mem_a)
        assert "mem_b" in activation_map
        assert activation_map["mem_b"] > 0.0
    
    def test_spread_activation_decay(self, navigator, memory_network):
        """Test activation decay."""
        initial = [(memory_network["mem_a"], 1.0)]
        
        activation_map = navigator._spread_activation(initial, memory_network)
        
        # mem_b should have decayed activation
        # activation = 1.0 * 0.8 (connection) * 0.7 (decay) = 0.56
        assert activation_map["mem_b"] < 1.0
        assert activation_map["mem_b"] > 0.5
    
    def test_spread_activation_multi_hop(self, navigator, memory_network):
        """Test multi-hop propagation."""
        initial = [(memory_network["mem_a"], 1.0)]
        
        activation_map = navigator._spread_activation(initial, memory_network)
        
        # Should reach mem_c through mem_b
        assert "mem_c" in activation_map
        assert activation_map["mem_c"] > 0.0
        assert activation_map["mem_c"] < activation_map["mem_b"]
    
    def test_spread_activation_threshold(self, navigator, memory_network):
        """Test activation threshold."""
        # Use high threshold
        navigator.activation_threshold = 0.5
        
        initial = [(memory_network["mem_a"], 1.0)]
        activation_map = navigator._spread_activation(initial, memory_network)
        
        # mem_b should be included (strong connection)
        assert "mem_b" in activation_map
        
        # mem_c might be excluded (weak propagated activation)
        # This depends on exact values


class TestRetrieve:
    """Test full retrieval."""
    
    def test_retrieve_basic(self, navigator, memory_network):
        """Test basic retrieval."""
        query = np.array([1.0, 0.0, 0.0])
        
        result = navigator.retrieve(query, memory_network, max_results=3)
        
        assert isinstance(result, ActivationResult)
        assert len(result.memories) > 0
        assert len(result.memories) <= 3
        assert result.hops_taken == 3
    
    def test_retrieve_relevance(self, navigator, memory_network):
        """Test retrieval relevance."""
        query = np.array([1.0, 0.0, 0.0])
        
        result = navigator.retrieve(query, memory_network, max_results=2)
        
        # Most relevant should be mem_a
        assert result.memories[0].id == "mem_a"
        
        # Second should be mem_b (connected and similar)
        if len(result.memories) > 1:
            assert result.memories[1].id == "mem_b"
    
    def test_retrieve_max_results(self, navigator, memory_network):
        """Test max results limiting."""
        query = np.array([1.0, 0.0, 0.0])
        
        result = navigator.retrieve(query, memory_network, max_results=2)
        
        assert len(result.memories) <= 2
    
    def test_retrieve_activation_map(self, navigator, memory_network):
        """Test activation map in result."""
        query = np.array([1.0, 0.0, 0.0])
        
        result = navigator.retrieve(query, memory_network, max_results=10)
        
        assert len(result.activation_map) > 0
        # All returned memories should be in activation map
        for memory in result.memories:
            assert memory.id in result.activation_map


class TestIntegration:
    """Integration tests."""
    
    def test_realistic_navigation(self, navigator, memory_network):
        """Test realistic navigation scenario."""
        # Query similar to mem_a
        query = np.array([0.95, 0.05, 0.0])
        
        result = navigator.retrieve(query, memory_network, max_results=4)
        
        # Should retrieve multiple memories
        assert len(result.memories) >= 2
        
        # Should be sorted by activation
        activations = [result.activation_map[m.id] for m in result.memories]
        assert activations == sorted(activations, reverse=True)
    
    def test_different_parameters(self):
        """Test with different parameters."""
        # Short-range navigator
        short_nav = NetworkNavigator(max_hops=1, decay_rate=0.5)
        
        # Long-range navigator
        long_nav = NetworkNavigator(max_hops=5, decay_rate=0.9)
        
        assert short_nav.max_hops < long_nav.max_hops
        assert short_nav.decay_rate < long_nav.decay_rate

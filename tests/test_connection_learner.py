"""
Unit tests for ConnectionLearner (Task 34)
"""

import pytest
import numpy as np
from llm_compression.connection_learner import ConnectionLearner
from llm_compression.memory_primitive import MemoryPrimitive
from llm_compression.compressor import CompressedMemory, CompressionMetadata
from datetime import datetime


@pytest.fixture
def connection_learner():
    """Create a ConnectionLearner instance."""
    return ConnectionLearner(
        co_activation_weight=0.3,
        similarity_weight=0.3,
        decay_rate=0.01
    )


@pytest.fixture
def memory_a():
    """Create first test memory."""
    metadata = CompressionMetadata(
        original_size=100,
        compressed_size=10,
        compression_ratio=10.0,
        model_used="test",
        quality_score=0.9,
        compression_time_ms=100.0,
        compressed_at=datetime.now()
    )
    
    compressed = CompressedMemory(
        memory_id="mem_a",
        summary_hash="hash_a",
        entities={},
        diff_data=b"data_a",
        embedding=[0.1] * 384,
        compression_metadata=metadata
    )
    
    return MemoryPrimitive(
        id="mem_a",
        content=compressed,
        embedding=np.array([1.0, 0.0, 0.0])  # Simple vector
    )


@pytest.fixture
def memory_b():
    """Create second test memory (similar to A)."""
    metadata = CompressionMetadata(
        original_size=100,
        compressed_size=10,
        compression_ratio=10.0,
        model_used="test",
        quality_score=0.9,
        compression_time_ms=100.0,
        compressed_at=datetime.now()
    )
    
    compressed = CompressedMemory(
        memory_id="mem_b",
        summary_hash="hash_b",
        entities={},
        diff_data=b"data_b",
        embedding=[0.1] * 384,
        compression_metadata=metadata
    )
    
    return MemoryPrimitive(
        id="mem_b",
        content=compressed,
        embedding=np.array([0.9, 0.1, 0.0])  # Similar to A
    )


@pytest.fixture
def memory_c():
    """Create third test memory (different from A and B)."""
    metadata = CompressionMetadata(
        original_size=100,
        compressed_size=10,
        compression_ratio=10.0,
        model_used="test",
        quality_score=0.9,
        compression_time_ms=100.0,
        compressed_at=datetime.now()
    )
    
    compressed = CompressedMemory(
        memory_id="mem_c",
        summary_hash="hash_c",
        entities={},
        diff_data=b"data_c",
        embedding=[0.1] * 384,
        compression_metadata=metadata
    )
    
    return MemoryPrimitive(
        id="mem_c",
        content=compressed,
        embedding=np.array([0.0, 0.0, 1.0])  # Orthogonal to A
    )


class TestConnectionLearnerCreation:
    """Test ConnectionLearner initialization."""
    
    def test_create_learner(self, connection_learner):
        """Test basic creation."""
        assert connection_learner.co_activation_weight == 0.3
        assert connection_learner.similarity_weight == 0.3
        assert connection_learner.decay_rate == 0.01
        assert len(connection_learner.co_activation_history) == 0


class TestSimilarityCalculation:
    """Test embedding similarity calculation."""
    
    def test_identical_vectors(self, connection_learner):
        """Test similarity of identical vectors."""
        vec = np.array([1.0, 0.0, 0.0])
        similarity = connection_learner._calculate_similarity(vec, vec)
        assert similarity == pytest.approx(1.0)
    
    def test_orthogonal_vectors(self, connection_learner):
        """Test similarity of orthogonal vectors."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        similarity = connection_learner._calculate_similarity(vec_a, vec_b)
        assert similarity == pytest.approx(0.5)  # Normalized to [0, 1]
    
    def test_opposite_vectors(self, connection_learner):
        """Test similarity of opposite vectors."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([-1.0, 0.0, 0.0])
        similarity = connection_learner._calculate_similarity(vec_a, vec_b)
        assert similarity == pytest.approx(0.0)  # Normalized to [0, 1]
    
    def test_similar_vectors(self, connection_learner, memory_a, memory_b):
        """Test similarity of similar vectors."""
        similarity = connection_learner._calculate_similarity(
            memory_a.embedding,
            memory_b.embedding
        )
        assert 0.9 < similarity <= 1.0  # Very similar
    
    def test_zero_vector(self, connection_learner):
        """Test handling of zero vectors."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 0.0, 0.0])
        similarity = connection_learner._calculate_similarity(vec_a, vec_b)
        assert similarity == 0.0


class TestCoActivation:
    """Test co-activation tracking."""
    
    def test_initial_co_activation(self, connection_learner, memory_a, memory_b):
        """Test initial co-activation is 0."""
        strength = connection_learner.get_co_activation_strength(memory_a, memory_b)
        assert strength == 0.0
    
    def test_record_co_activation(self, connection_learner, memory_a, memory_b):
        """Test recording co-activation."""
        connection_learner.record_co_activation(memory_a, memory_b)
        strength = connection_learner.get_co_activation_strength(memory_a, memory_b)
        assert strength == 0.1
    
    def test_multiple_co_activations(self, connection_learner, memory_a, memory_b):
        """Test multiple co-activations accumulate."""
        connection_learner.record_co_activation(memory_a, memory_b)
        connection_learner.record_co_activation(memory_a, memory_b)
        connection_learner.record_co_activation(memory_a, memory_b)
        
        strength = connection_learner.get_co_activation_strength(memory_a, memory_b)
        assert strength == pytest.approx(0.3)
    
    def test_co_activation_cap(self, connection_learner, memory_a, memory_b):
        """Test co-activation is capped at 1.0."""
        for _ in range(20):
            connection_learner.record_co_activation(memory_a, memory_b)
        
        strength = connection_learner.get_co_activation_strength(memory_a, memory_b)
        assert strength == 1.0
    
    def test_co_activation_symmetric(self, connection_learner, memory_a, memory_b):
        """Test co-activation is symmetric (A-B == B-A)."""
        connection_learner.record_co_activation(memory_a, memory_b)
        
        strength_ab = connection_learner.get_co_activation_strength(memory_a, memory_b)
        strength_ba = connection_learner.get_co_activation_strength(memory_b, memory_a)
        
        assert strength_ab == strength_ba


class TestCoActivationDecay:
    """Test co-activation decay."""
    
    def test_decay(self, connection_learner, memory_a, memory_b):
        """Test co-activation decays over time."""
        connection_learner.record_co_activation(memory_a, memory_b)
        initial = connection_learner.get_co_activation_strength(memory_a, memory_b)
        
        connection_learner.decay_co_activations()
        after_decay = connection_learner.get_co_activation_strength(memory_a, memory_b)
        
        assert after_decay < initial
        assert after_decay == pytest.approx(initial * 0.99)
    
    def test_weak_connections_removed(self, connection_learner, memory_a, memory_b):
        """Test very weak connections are removed."""
        # Set very weak connection
        key = connection_learner._make_key(memory_a.id, memory_b.id)
        connection_learner.co_activation_history[key] = 0.005
        
        connection_learner.decay_co_activations()
        
        assert key not in connection_learner.co_activation_history


class TestConnectionLearning:
    """Test connection strength learning."""
    
    def test_learn_connection_no_history(self, connection_learner, memory_a, memory_b):
        """Test learning connection with no co-activation history."""
        strength = connection_learner.learn_connection(memory_a, memory_b)
        
        # Only similarity contributes (high similarity)
        assert 0.0 < strength <= 1.0
    
    def test_learn_connection_with_co_activation(self, connection_learner, memory_a, memory_b):
        """Test learning connection with co-activation history."""
        # Record co-activations
        for _ in range(5):
            connection_learner.record_co_activation(memory_a, memory_b)
        
        strength = connection_learner.learn_connection(memory_a, memory_b)
        
        # Both co-activation and similarity contribute
        assert 0.3 < strength <= 1.0
    
    def test_learn_connection_dissimilar(self, connection_learner, memory_a, memory_c):
        """Test learning connection between dissimilar memories."""
        strength = connection_learner.learn_connection(memory_a, memory_c)
        
        # Low similarity, no co-activation
        assert 0.0 <= strength < 0.3
    
    def test_connection_strength_range(self, connection_learner, memory_a, memory_b):
        """Test connection strength is always in [0, 1]."""
        for _ in range(10):
            connection_learner.record_co_activation(memory_a, memory_b)
        
        strength = connection_learner.learn_connection(memory_a, memory_b)
        
        assert 0.0 <= strength <= 1.0


class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_realistic_learning_scenario(self, connection_learner, memory_a, memory_b, memory_c):
        """Test realistic learning scenario."""
        # A and B are activated together multiple times
        for _ in range(3):
            connection_learner.record_co_activation(memory_a, memory_b)
        
        # Learn connections
        strength_ab = connection_learner.learn_connection(memory_a, memory_b)
        strength_ac = connection_learner.learn_connection(memory_a, memory_c)
        
        # A-B should be stronger (similar + co-activated)
        assert strength_ab > strength_ac
        
        # Decay
        connection_learner.decay_co_activations()
        
        # Strength should decrease slightly
        strength_ab_after = connection_learner.learn_connection(memory_a, memory_b)
        assert strength_ab_after < strength_ab
    
    def test_multiple_memories(self, connection_learner, memory_a, memory_b, memory_c):
        """Test learning with multiple memories."""
        # Create network: A-B strong, B-C medium, A-C weak
        for _ in range(5):
            connection_learner.record_co_activation(memory_a, memory_b)
        for _ in range(2):
            connection_learner.record_co_activation(memory_b, memory_c)
        
        strength_ab = connection_learner.learn_connection(memory_a, memory_b)
        strength_bc = connection_learner.learn_connection(memory_b, memory_c)
        strength_ac = connection_learner.learn_connection(memory_a, memory_c)
        
        assert strength_ab > strength_bc > strength_ac


class TestHebbianLearning:
    """Test Hebbian learning (Task 36)."""
    
    def test_hebbian_learning_basic(self, connection_learner, memory_a, memory_b):
        """Test basic Hebbian learning."""
        # Initial state
        assert memory_a.get_connection_strength(memory_b.id) == 0.0
        assert memory_b.get_connection_strength(memory_a.id) == 0.0
        
        # Apply Hebbian learning
        connection_learner.hebbian_learning(memory_a, memory_b, learning_rate=0.1)
        
        # Connections should be strengthened
        strength_ab = memory_a.get_connection_strength(memory_b.id)
        strength_ba = memory_b.get_connection_strength(memory_a.id)
        
        assert strength_ab > 0.0
        assert strength_ba > 0.0
    
    def test_hebbian_learning_bidirectional(self, connection_learner, memory_a, memory_b):
        """Test bidirectional symmetry."""
        connection_learner.hebbian_learning(memory_a, memory_b, learning_rate=0.1)
        
        strength_ab = memory_a.get_connection_strength(memory_b.id)
        strength_ba = memory_b.get_connection_strength(memory_a.id)
        
        # Should be symmetric
        assert strength_ab == strength_ba
    
    def test_hebbian_learning_multiple_applications(self, connection_learner, memory_a, memory_b):
        """Test multiple Hebbian learning applications."""
        # Apply multiple times
        for _ in range(5):
            connection_learner.hebbian_learning(memory_a, memory_b, learning_rate=0.1)
        
        strength = memory_a.get_connection_strength(memory_b.id)
        
        # Should accumulate but not exceed 1.0
        assert 0.0 < strength <= 1.0
    
    def test_hebbian_learning_rate_effect(self, connection_learner, memory_a, memory_b, memory_c):
        """Test learning rate effect."""
        # Learn with different rates
        connection_learner.hebbian_learning(memory_a, memory_b, learning_rate=0.2)
        connection_learner.hebbian_learning(memory_a, memory_c, learning_rate=0.05)
        
        strength_ab = memory_a.get_connection_strength(memory_b.id)
        strength_ac = memory_a.get_connection_strength(memory_c.id)
        
        # Higher learning rate should result in stronger connection
        assert strength_ab > strength_ac
    
    def test_hebbian_learning_co_activation_tracking(self, connection_learner, memory_a, memory_b):
        """Test that Hebbian learning records co-activation."""
        initial_co_activation = connection_learner.get_co_activation_strength(memory_a, memory_b)
        
        connection_learner.hebbian_learning(memory_a, memory_b, learning_rate=0.1)
        
        after_co_activation = connection_learner.get_co_activation_strength(memory_a, memory_b)
        
        # Co-activation should increase
        assert after_co_activation > initial_co_activation

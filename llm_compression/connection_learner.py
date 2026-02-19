"""
Connection Learner - Hebbian learning for memory connections

Implements connection learning between memories based on:
1. Co-activation (memories activated together)
2. Similarity (embedding similarity)
"""

from typing import Dict, Tuple
import numpy as np

from .memory_primitive import MemoryPrimitive


class ConnectionLearner:
    """
    Learn connections between memories using Hebbian learning.
    
    "Neurons that fire together, wire together" - applied to memories.
    Connection strength = weighted combination of co-activation and similarity.
    """
    
    def __init__(
        self,
        co_activation_weight: float = 0.3,
        similarity_weight: float = 0.3,
        decay_rate: float = 0.01
    ):
        """
        Initialize connection learner.
        
        Args:
            co_activation_weight: Weight for co-activation score (0-1)
            similarity_weight: Weight for similarity score (0-1)
            decay_rate: Decay rate for co-activation history
        """
        self.co_activation_weight = co_activation_weight
        self.similarity_weight = similarity_weight
        self.decay_rate = decay_rate
        
        # Track co-activations: {(id_a, id_b): count}
        self.co_activation_history: Dict[Tuple[str, str], float] = {}
    
    def learn_connection(
        self,
        memory_a: MemoryPrimitive,
        memory_b: MemoryPrimitive
    ) -> float:
        """
        Calculate connection strength between two memories.
        
        Args:
            memory_a: First memory
            memory_b: Second memory
            
        Returns:
            Connection strength (0.0-1.0)
        """
        # 1. Co-activation score
        co_activation = self._calculate_co_activation(memory_a, memory_b)
        
        # 2. Similarity score
        similarity = self._calculate_similarity(
            memory_a.embedding,
            memory_b.embedding
        )
        
        # 3. Weighted combination
        connection_strength = (
            self.co_activation_weight * co_activation +
            self.similarity_weight * similarity
        )
        
        # Normalize to [0, 1]
        return min(1.0, max(0.0, connection_strength))
    
    def record_co_activation(
        self,
        memory_a: MemoryPrimitive,
        memory_b: MemoryPrimitive,
        strength: float = 0.1
    ):
        """
        Record that two memories were activated together.
        
        Args:
            memory_a: First memory
            memory_b: Second memory
            strength: Increment strength (default: 0.1)
        """
        key = self._make_key(memory_a.id, memory_b.id)
        current = self.co_activation_history.get(key, 0.0)
        self.co_activation_history[key] = min(1.0, current + strength)
    
    def decay_co_activations(self):
        """Decay all co-activation history (forgetting)."""
        for key in list(self.co_activation_history.keys()):
            self.co_activation_history[key] *= (1.0 - self.decay_rate)
            
            # Remove very weak connections
            if self.co_activation_history[key] < 0.01:
                del self.co_activation_history[key]
    
    def get_co_activation_strength(
        self,
        memory_a: MemoryPrimitive,
        memory_b: MemoryPrimitive
    ) -> float:
        """
        Get co-activation strength between two memories.
        
        Args:
            memory_a: First memory
            memory_b: Second memory
            
        Returns:
            Co-activation strength (0.0-1.0)
        """
        key = self._make_key(memory_a.id, memory_b.id)
        return self.co_activation_history.get(key, 0.0)
    
    def _calculate_co_activation(
        self,
        memory_a: MemoryPrimitive,
        memory_b: MemoryPrimitive
    ) -> float:
        """Calculate co-activation score."""
        return self.get_co_activation_strength(memory_a, memory_b)
    
    def _calculate_similarity(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding_a: First embedding vector
            embedding_b: Second embedding vector
            
        Returns:
            Cosine similarity (-1.0 to 1.0, normalized to 0.0-1.0)
        """
        # Cosine similarity
        dot_product = np.dot(embedding_a, embedding_b)
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm_a * norm_b)
        
        # Normalize from [-1, 1] to [0, 1]
        return (cosine_sim + 1.0) / 2.0
    
    def _make_key(self, id_a: str, id_b: str) -> Tuple[str, str]:
        """Create sorted key for memory pair."""
        return tuple(sorted([id_a, id_b]))
    
    def hebbian_learning(
        self,
        memory_a: MemoryPrimitive,
        memory_b: MemoryPrimitive,
        learning_rate: float = 0.1
    ):
        """
        Hebbian learning: "Neurons that fire together, wire together"
        
        Strengthens bidirectional connections when memories are co-activated.
        
        Args:
            memory_a: First memory
            memory_b: Second memory
            learning_rate: Learning rate (0.0-1.0)
        """
        # Calculate connection strength
        connection_strength = self.learn_connection(memory_a, memory_b)
        
        # Strengthen with learning rate
        new_strength = min(1.0, connection_strength + learning_rate)
        
        # Update bidirectional connections
        memory_a.add_connection(memory_b.id, new_strength)
        memory_b.add_connection(memory_a.id, new_strength)
        
        # Record co-activation
        self.record_co_activation(memory_a, memory_b, learning_rate)

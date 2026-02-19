"""
Network Navigator - Navigate self-organized memory network

Implements activation spreading algorithm for memory retrieval:
1. Initial activation (similarity-based)
2. Activation spreading (connection-based)
3. Multi-hop propagation with decay
"""

from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import numpy as np

from .memory_primitive import MemoryPrimitive


@dataclass
class ActivationResult:
    """Result of activation spreading."""
    memories: List[MemoryPrimitive]
    activation_map: Dict[str, float]
    hops_taken: int


class NetworkNavigator:
    """
    Navigate memory network using activation spreading.
    
    Implements spreading activation algorithm:
    - Start from query-similar memories
    - Spread activation along connections
    - Decay activation with distance
    - Return top-k activated memories
    """
    
    def __init__(
        self,
        max_hops: int = 3,
        decay_rate: float = 0.7,
        activation_threshold: float = 0.1
    ):
        """
        Initialize network navigator.
        
        Args:
            max_hops: Maximum hops for activation spreading
            decay_rate: Activation decay per hop (0-1)
            activation_threshold: Minimum activation to propagate
        """
        self.max_hops = max_hops
        self.decay_rate = decay_rate
        self.activation_threshold = activation_threshold
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        memory_network: Dict[str, MemoryPrimitive],
        max_results: int = 10
    ) -> ActivationResult:
        """
        Retrieve relevant memories using activation spreading.
        
        Args:
            query_embedding: Query embedding vector
            memory_network: Dictionary of memory_id -> MemoryPrimitive
            max_results: Maximum number of results
            
        Returns:
            ActivationResult with retrieved memories
        """
        # 1. Initial activation (similarity-based)
        initial_memories = self._find_similar(
            query_embedding,
            memory_network,
            top_k=min(5, len(memory_network))
        )
        
        # 2. Spread activation
        activation_map = self._spread_activation(
            initial_memories,
            memory_network
        )
        
        # 3. Sort by activation and return top-k
        sorted_memories = sorted(
            [memory_network[mid] for mid in activation_map.keys()],
            key=lambda m: activation_map[m.id],
            reverse=True
        )[:max_results]
        
        return ActivationResult(
            memories=sorted_memories,
            activation_map=activation_map,
            hops_taken=self.max_hops
        )
    
    def _find_similar(
        self,
        query_embedding: np.ndarray,
        memory_network: Dict[str, MemoryPrimitive],
        top_k: int = 5
    ) -> List[Tuple[MemoryPrimitive, float]]:
        """
        Find initial similar memories.
        
        Args:
            query_embedding: Query embedding
            memory_network: Memory network
            top_k: Number of initial memories
            
        Returns:
            List of (memory, similarity) tuples
        """
        similarities = []
        
        for memory in memory_network.values():
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            similarities.append((memory, similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _spread_activation(
        self,
        initial_memories: List[Tuple[MemoryPrimitive, float]],
        memory_network: Dict[str, MemoryPrimitive]
    ) -> Dict[str, float]:
        """
        Spread activation along connections.
        
        Args:
            initial_memories: List of (memory, initial_activation) tuples
            memory_network: Memory network
            
        Returns:
            Dictionary of memory_id -> final_activation
        """
        # Activation map: memory_id -> activation
        activation_map: Dict[str, float] = {}
        
        # Queue: (memory_id, activation, hop_count)
        queue: List[Tuple[str, float, int]] = []
        
        # Initialize with initial memories
        for memory, activation in initial_memories:
            activation_map[memory.id] = activation
            queue.append((memory.id, activation, 0))
        
        # Visited set to avoid cycles
        visited: Set[str] = set()
        
        # Spread activation
        while queue:
            memory_id, activation, hop = queue.pop(0)
            
            # Skip if already visited or max hops reached
            if memory_id in visited or hop >= self.max_hops:
                continue
            
            visited.add(memory_id)
            
            # Get memory
            memory = memory_network.get(memory_id)
            if not memory:
                continue
            
            # Propagate to connected memories
            for conn_id, connection_strength in memory.connections.items():
                if conn_id not in memory_network:
                    continue
                
                # Calculate new activation
                new_activation = activation * connection_strength * self.decay_rate
                
                # Skip if below threshold
                if new_activation < self.activation_threshold:
                    continue
                
                # Accumulate activation
                if conn_id in activation_map:
                    activation_map[conn_id] = max(activation_map[conn_id], new_activation)
                else:
                    activation_map[conn_id] = new_activation
                
                # Add to queue for further propagation
                if conn_id not in visited:
                    queue.append((conn_id, new_activation, hop + 1))
        
        return activation_map
    
    def _cosine_similarity(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity.
        
        Args:
            embedding_a: First embedding
            embedding_b: Second embedding
            
        Returns:
            Cosine similarity (0-1)
        """
        dot_product = np.dot(embedding_a, embedding_b)
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm_a * norm_b)
        
        # Normalize to [0, 1]
        return (cosine_sim + 1.0) / 2.0

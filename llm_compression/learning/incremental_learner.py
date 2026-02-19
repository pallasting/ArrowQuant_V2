"""
Incremental Learning System for ArrowEngine Native Memory.
"""

import numpy as np
import threading
from typing import Dict, List, Optional
from llm_compression.compression.vector_compressor import CompressedMemory

class IncrementalLearner:
    """
    Tracks memory usage and learns which semantic dimensions are most important
    over time, enabling adaptive compression.
    """
    
    def __init__(self, dimension_size: int = 384):
        self.dimension_size = dimension_size
        # Count how many times each dimension was retrieved/activated
        self.usage_counts = np.zeros(self.dimension_size, dtype=np.float32)
        # Total retrievals
        self.total_accesses = 0
        # Thread lock
        self._lock = threading.Lock()
        
    def record_access(self, compressed_memory: CompressedMemory):
        """
        Record that a memory was accessed. Increase weights for its active dimensions.
        """
        with self._lock:
            # Increment counts for active dimensions
            # We add 1.0 to each active dimension
            # Or we could add the magnitude of the value? Let's stick to frequency for now.
            np.add.at(self.usage_counts, compressed_memory.key_indices, 1.0)
            self.total_accesses += 1
            
    def get_dimension_weights(self, base_weight: float = 1.0, learning_rate: float = 0.1) -> np.ndarray:
        """
        Calculate current dimension weights based on usage history.
        
        Args:
            base_weight: Starting weight for all dimensions (e.g. 1.0)
            learning_rate: How much history influences the weight
            
        Returns:
            Weights array (size,)
        """
        with self._lock:
            if self.total_accesses == 0:
                return np.ones(self.dimension_size, dtype=np.float32)
            
            # Frequency = count / total
            frequency = self.usage_counts / self.total_accesses
            
            # Weight = 1.0 + (freq * relevance_boost)
            # If a dimension is used in EVERY retrieval, it gets high weight.
            # If never used, it stays at base_weight (or could decay).
            
            # Let's say we want to boost frequent dimensions by up to 2x or 3x.
            # Max boost factor = 5.0
            
            boost = frequency * 5.0 * learning_rate
            
            weights = base_weight + boost
            
            return weights

    def reset(self):
        with self._lock:
            self.usage_counts.fill(0)
            self.total_accesses = 0

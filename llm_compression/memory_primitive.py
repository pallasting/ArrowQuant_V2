"""
Memory Primitive - Fundamental memory unit for Phase 2.0 Cognitive Loop

This module implements the basic building block of the self-organizing memory system.
Each MemoryPrimitive stores compressed content with self-organizing properties.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import numpy as np

from .compressor import CompressedMemory


@dataclass
class MemoryPrimitive:
    """
    Minimal memory unit - foundation of self-organizing cognitive system.
    
    Combines Phase 1.1 compression with Phase 2.0 self-organizing properties:
    - Activation tracking (how "active" this memory is)
    - Connection learning (relationships to other memories)
    - Success rate (quality feedback)
    """
    
    # Core data
    id: str
    content: CompressedMemory  # From Phase 1.1
    embedding: np.ndarray      # Feature vector for similarity
    
    # Self-organizing properties
    connections: Dict[str, float] = field(default_factory=dict)  # {memory_id: strength}
    activation: float = 0.0  # Current activation level (0.0-1.0)
    
    # Statistics for learning
    access_count: int = 0
    success_count: int = 0
    last_access: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def activate(self, strength: float):
        """
        Activate this memory with given strength.
        
        Args:
            strength: Activation strength (0.0-1.0)
        """
        self.activation = min(1.0, self.activation + strength)
        self.access_count += 1
        self.last_access = datetime.now()
    
    def decay(self, rate: float = 0.1):
        """
        Decay activation over time (forgetting).
        
        Args:
            rate: Decay rate (0.0-1.0)
        """
        self.activation = max(0.0, self.activation * (1.0 - rate))
    
    def record_success(self, success: bool = True):
        """
        Record usage result for this memory.

        Args:
            success: Whether the usage was successful (default: True)
        """
        self.access_count += 1
        self.last_access = datetime.now()
        if success:
            self.success_count += 1
    
    def get_success_rate(self) -> float:
        """
        Calculate success rate of this memory.
        
        Returns:
            Success rate (0.0-1.0)
        """
        if self.access_count == 0:
            return 0.0
        return self.success_count / self.access_count
    
    def add_connection(self, memory_id: str, strength: float):
        """
        Add or strengthen connection to another memory.
        
        Args:
            memory_id: ID of connected memory
            strength: Connection strength (0.0-1.0)
        """
        if memory_id in self.connections:
            # Strengthen existing connection
            self.connections[memory_id] = min(1.0, self.connections[memory_id] + strength)
        else:
            # Create new connection
            self.connections[memory_id] = strength
    
    def get_connection_strength(self, memory_id: str) -> float:
        """
        Get connection strength to another memory.
        
        Args:
            memory_id: ID of connected memory
            
        Returns:
            Connection strength (0.0 if not connected)
        """
        return self.connections.get(memory_id, 0.0)

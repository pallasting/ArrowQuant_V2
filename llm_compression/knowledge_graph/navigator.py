
"""
Graph Navigator - Algorithms for traversing the memory knowledge graph.

Implements Spreading Activation for associative retrieval.
"""

from typing import List, Dict, Set, Tuple, Optional
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class GraphNavigator:
    """
    Navigates the Knowledge Graph to find associations.
    """
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        
    def spread_activation(
        self,
        start_concepts: List[str],
        max_hops: int = 2,
        decay: float = 0.5,
        min_activation: float = 0.01
    ) -> List[Tuple[str, float]]:
        """
        Perform spreading activation from start concepts.
        
        Args:
            start_concepts: List of concept strings to activate
            max_hops: Maximum distance to spread
            decay: Decay factor per hop (0.0 to 1.0)
            min_activation: Minimum activation to propagate
            
        Returns:
            List of (node_id, activation_score) tuples, sorted by score.
            Includes both definitions (concepts) and memories.
        """
        # 1. Initialize activations
        activations: Dict[str, float] = {}
        queue: List[Tuple[str, float, int]] = [] # (node, activation, depth)
        
        for concept in start_concepts:
            c = concept.lower().strip()
            if self.graph.has_node(c):
                activations[c] = 1.0
                queue.append((c, 1.0, 0))
            else:
                # Optional: fuzzy match or skip
                pass
                
        visited: Set[str] = set()
        
        # 2. Spread
        while queue:
            current_node, activation, depth = queue.pop(0)
            
            if depth >= max_hops:
                continue
                
            if current_node in visited:
                continue
            visited.add(current_node)
            
            # Get neighbors
            neighbors = list(self.graph.neighbors(current_node))
            
            # Distribute activation to neighbors
            for neighbor in neighbors:
                # Calculate edge weight influence
                edge_data = self.graph.get_edge_data(current_node, neighbor)
                weight = edge_data.get('weight', 0.5) if edge_data else 0.5
                
                # New activation
                input_excitation = activation * weight * decay
                
                # Add to existing activation ("Energy accumulation")
                current_val = activations.get(neighbor, 0.0)
                # Simple additive accumulation, bounded? Or max?
                # Let's use max for path finding, sum for "resonance"
                # Using sum usually allows multi-path reinforcement
                new_val = current_val + input_excitation
                
                if new_val >= min_activation:
                    activations[neighbor] = new_val
                    # Don't add to queue if we already processed it at this depth or lower?
                    # BFS nature handles filtering. Just add if not visited deep enough.
                    if neighbor not in visited:
                        queue.append((neighbor, input_excitation, depth + 1))
                        
        # 3. Filter and Sort
        # We might only want memory nodes? Or both?
        # Let's return all, let caller filter.
        results = sorted(activations.items(), key=lambda x: x[1], reverse=True)
        return results

    def get_related_memories(
        self, 
        concepts: List[str], 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find specific memories related to concepts.
        Wraps spread_activation filtering for type='memory'.
        """
        all_activated = self.spread_activation(concepts)
        
        memories = []
        for node_id, score in all_activated:
            node_data = self.graph.nodes.get(node_id)
            if node_data and node_data.get('type') == 'memory':
                memories.append((node_id, score))
                
        return memories[:top_k]

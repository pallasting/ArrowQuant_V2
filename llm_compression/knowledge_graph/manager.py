
"""
Knowledge Graph Manager for ArrowEngine-Native Memory.

Handles storage and updates of the memetic knowledge graph using NetworkX.
"""

import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
import pickle

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    """
    Manages the memory knowledge graph.
    
    Nodes:
    - Memories (MemoryPrimitive IDs)
    - Concepts/Entities (Key Tokens)
    
    Edges:
    - Memory -> Concept (contains)
    - Concept -> Concept (co-occurrence)
    - Memory -> Memory (similarity/reference - TBD)
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.graph_file = self.storage_path / "knowledge_graph.gpickle"
        self.graph = nx.Graph()
        self._load()
        
    def _load(self):
        """Load graph from disk."""
        if self.graph_file.exists():
            try:
                with open(self.graph_file, 'rb') as f:
                    self.graph = pickle.load(f)
                print(f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes.")
            except Exception as e:
                print(f"Failed to load knowledge graph: {e}")
                self.graph = nx.Graph()
        else:
            print("Initialized new knowledge graph.")
            
    def save(self):
        """Save graph to disk."""
        try:
            # Ensure directory exists
            self.storage_path.mkdir(parents=True, exist_ok=True) 
            with open(self.graph_file, 'wb') as f:
                pickle.dump(self.graph, f)
            print("Saved knowledge graph.")
        except Exception as e:
            print(f"Failed to save knowledge graph: {e}")
            
    def add_memory_concepts(
        self, 
        memory_id: str, 
        concepts: List[str], 
        scores: List[float],
        top_k_concepts: int = 5
    ):
        """
        Add memory and its related concepts to the graph.
        
        Args:
            memory_id: Unique memory ID
            concepts: List of extracted key tokens/concepts
            scores: Importance scores for each concept
            top_k_concepts: Max number of top concepts to link for co-occurrence checks
        """
        if not concepts:
            return
            
        # Add memory node
        self.graph.add_node(memory_id, type="memory", label=f"Memory:{memory_id[:8]}")
        
        # Identify top concepts by score
        concept_pairs = sorted(zip(concepts, scores), key=lambda x: x[1], reverse=True)
        top_concepts = concept_pairs[:top_k_concepts]
        
        # Add concept nodes and link to memory
        added_concepts = []
        for concept, score in top_concepts:
            concept = concept.lower().strip()
            if not concept or len(concept) < 2: continue # filter too short
            
            # Add or update concept node
            if self.graph.has_node(concept):
                curr_w = self.graph.nodes[concept].get('importance', 0.0)
                self.graph.nodes[concept]['importance'] = curr_w + score
                self.graph.nodes[concept]['frequency'] = self.graph.nodes[concept].get('frequency', 0) + 1
            else:
                self.graph.add_node(concept, type="concept", importance=score, frequency=1, label=concept)
                
            # Link Memory -> Concept
            self.graph.add_edge(memory_id, concept, weight=score, relation="contains")
            added_concepts.append((concept, score))
            
        # Add Concept -> Concept co-occurrence links (Clique)
        # Only link within the top-k concepts of this memory
        for i in range(len(added_concepts)):
            for j in range(i+1, len(added_concepts)):
                c1, s1 = added_concepts[i]
                c2, s2 = added_concepts[j]
                
                # Edge weight = combined importance in this context
                w = (s1 + s2) / 2.0
                
                if self.graph.has_edge(c1, c2):
                    curr_w = self.graph.edges[c1, c2].get('weight', 0.0)
                    count = self.graph.edges[c1, c2].get('count', 0)
                    self.graph.edges[c1, c2]['weight'] = curr_w + w
                    self.graph.edges[c1, c2]['count'] = count + 1
                else:
                    self.graph.add_edge(c1, c2, weight=w, count=1, relation="related")
                    
    def add_memory(self, memory_id: str, concepts: List[str], text: str = ""):
        """
        Add a memory to the graph. Simplified wrapper around add_memory_concepts.
        """
        if not concepts:
            return
            
        # Assign default scores based on order if not provided
        scores = [1.0 / (i + 1) for i in range(len(concepts))]
        self.add_memory_concepts(memory_id, concepts, scores)
        
        # Store metadata if useful (optional)
        if text:
            self.graph.nodes[memory_id]['text_preview'] = text[:50]

    def add_concept_relations(self, relations: List[Tuple[str, str, float]]):
        """
        Add direct edges between concepts based on attention flow.
        """
        if not relations:
            return
            
        for c1, c2, weight in relations:
            c1 = c1.lower().strip()
            c2 = c2.lower().strip()
            if not c1 or not c2 or c1 == c2:
                continue
                
            # Ensure concept nodes exist
            for c in [c1, c2]:
                if not self.graph.has_node(c):
                    self.graph.add_node(c, type="concept", importance=0.0, frequency=1, label=c)
            
            # In undirected nx.Graph, this adds/updates the weight between concepts
            if self.graph.has_edge(c1, c2):
                curr_w = self.graph.edges[c1, c2].get('weight', 0.0)
                self.graph.edges[c1, c2]['weight'] = curr_w + weight
                # Promote to functional dependency
                self.graph.edges[c1, c2]['relation'] = "attends_to"
            else:
                # Store as 'attends_to' to indicate functional dependency from attention
                self.graph.add_edge(c1, c2, weight=weight, relation="attends_to")
                    
    def find_related_concepts(self, start_concept: str, max_depth: int = 1, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find related concepts using multi-hop traversal.
        
        Returns:
            List of dicts with {'concept', 'distance', 'relevance'}
        """
        start = start_concept.lower().strip()
        if not self.graph.has_node(start):
            return []
            
        results = []
        # Use Dijkstra-like weighted BFS for relevance
        # For simplicity, use simple BFS distance here
        seen = {start}
        queue = [(start, 0, 1.0)] # (concept, depth, relevance)
        
        while queue:
            curr, depth, relevance = queue.pop(0)
            if depth > 0:
                results.append({
                    "concept": curr,
                    "depth": depth,
                    "relevance": relevance
                })
            
            if depth >= max_depth:
                continue
                
            # Get neighbors that are concepts
            for neighbor in self.graph.neighbors(curr):
                if neighbor not in seen and self.graph.nodes[neighbor].get('type') == 'concept':
                    seen.add(neighbor)
                    edge_w = self.graph.edges[curr, neighbor].get('weight', 1.0)
                    # Decay relevance by distance and edge weight
                    new_rel = relevance * (min(1.0, edge_w)) * 0.8
                    queue.append((neighbor, depth + 1, new_rel))
        
        # Sort results by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:top_k]

    def find_related_memories(self, query_concepts: List[str], max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Find memories related to a set of query concepts through graph paths.
        Implement Multi-Hop Reasoning.
        """
        query_nodes = [c.lower().strip() for c in query_concepts if self.graph.has_node(c.lower().strip())]
        if not query_nodes:
            return []
            
        memory_scores: Dict[str, float] = {}
        
        # Traverse from each query concept
        for start_node in query_nodes:
            seen = {start_node}
            queue = [(start_node, 0, 1.0)] # (node, depth, score)
            
            while queue:
                curr, depth, score = queue.pop(0)
                
                # If current node is a memory, add to scores
                if self.graph.nodes[curr].get('type') == 'memory':
                    memory_scores[curr] = memory_scores.get(curr, 0.0) + score
                
                if depth >= max_hops:
                    continue
                    
                # Continue traversal
                for neighbor in self.graph.neighbors(curr):
                    if neighbor not in seen:
                        # Edge weight as traversal probability
                        w = self.graph.edges[curr, neighbor].get('weight', 0.5)
                        # Normalize weight if too large
                        w = min(1.0, w / 5.0) if w > 1.0 else w 
                        
                        new_score = score * w * 0.9 # Depth decay
                        queue.append((neighbor, depth + 1, new_score))
                        # We don't mark memories as 'seen' globally to allow multiple paths
                        if self.graph.nodes[neighbor].get('type') == 'concept':
                            seen.add(neighbor)
                            
        # Convert to list and sort
        results = [
            {"memory_id": mid, "relevance": round(score, 4)} 
            for mid, score in memory_scores.items()
        ]
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results

    def find_memories_by_concept(self, concept: str) -> List[str]:
        """Simple neighbor lookup for memories."""
        c = concept.lower().strip()
        if not self.graph.has_node(c):
            return []
            
        memories = [n for n in self.graph.neighbors(c) 
                   if self.graph.nodes[n].get('type') == 'memory']
                   
        return memories

    def add_image_node(self, image_id: str, concepts: List[str], scores: List[float] = None):
        """
        Add an image node and link it to concepts.
        """
        if not concepts:
            return
            
        self.graph.add_node(image_id, type="image", label=f"Image:{image_id[:8]}")
        
        if scores is None:
            scores = [1.0] * len(concepts)
            
        for concept, score in zip(concepts, scores):
            concept = concept.lower().strip()
            if not self.graph.has_node(concept):
                self.graph.add_node(concept, type="concept", importance=0.1, frequency=1, label=concept)
                
            # Edge: Image -> Concept (depicts)
            if self.graph.has_edge(image_id, concept):
                self.graph.edges[image_id, concept]['weight'] = max(score, self.graph.edges[image_id, concept]['weight'])
            else:
                self.graph.add_edge(image_id, concept, weight=score, relation="depicts")

    def find_related_images(self, query: str, max_depth: int = 1) -> List[str]:
        """Find image IDs related to a concept query."""
        start = query.lower().strip()
        if not self.graph.has_node(start):
            return []
            
        images = []
        # Simple neighbor check for now
        for neighbor in self.graph.neighbors(start):
             if self.graph.nodes[neighbor].get('type') == 'image':
                 images.append(neighbor)
                 
        return images

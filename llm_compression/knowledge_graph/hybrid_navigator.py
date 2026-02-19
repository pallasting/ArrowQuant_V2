
"""
Hybrid Navigator - Combines semantic vector search with knowledge graph spreading activation.

Implements the advanced "Memetic Retrieval" algorithm.
"""

from typing import List, Dict, Set, Tuple, Optional, Any
import logging
import numpy as np

from llm_compression.vector_search import VectorSearch, SearchResult
from llm_compression.knowledge_graph.navigator import GraphNavigator
from llm_compression.knowledge_graph.manager import KnowledgeGraphManager
from llm_compression.compression.attention_extractor import AttentionBasedExtractor

logger = logging.getLogger(__name__)

class HybridNavigator:
    """
    Combines semantic search and graph-based associative retrieval.
    """
    
    def __init__(
        self,
        vector_search: VectorSearch,
        kg_manager: KnowledgeGraphManager,
        attention_extractor: AttentionBasedExtractor
    ):
        self.vector_search = vector_search
        self.kg_manager = kg_manager
        self.graph_navigator = GraphNavigator(kg_manager.graph)
        self.attention_extractor = attention_extractor
        
    def search(
        self,
        query: str,
        category: str = 'experiences',
        top_k: int = 10,
        alpha: float = 0.5 # Balance between semantic (vector) and associative (KG)
    ) -> List[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Query string
            category: Memory category
            top_k: Total results to return
            alpha: Weighting factor (alpha * semantic + (1-alpha) * associative)
            
        Returns:
            Ranked list of SearchResults
        """
        logger.info(f"Hybrid search for: {query}")
        
        # 1. Semantic Search (Dense Retrieval)
        vector_results = self.vector_search.search(query, category, top_k=top_k*2)
        semantic_scores = {r.memory_id: r.similarity for r in vector_results}
        
        # 2. Key Token Extraction from Query
        # This helps activate the KG from the query's concepts
        query_info = self.attention_extractor.extract_key_information(query)
        query_concepts = query_info.key_tokens
        
        # 3. Associative Retrieval (KG Spreading Activation)
        # We activate based on query concepts AND top semantic results' concepts
        seed_concepts = set(query_concepts)
        
        # Add concepts from top semantic results (Context Expansion)
        for res in vector_results[:3]: # Take top 3 for expansion
            if hasattr(res.memory, 'key_tokens') and res.memory.key_tokens:
                seed_concepts.update(res.memory.key_tokens)
                
        # Spread activation in KG
        activated_nodes = self.graph_navigator.spread_activation(list(seed_concepts))
        
        # Extract memory scores from graph results
        associative_scores: Dict[str, float] = {}
        for node_id, score in activated_nodes:
            # Check if node is a memory
            node_data = self.kg_manager.graph.nodes.get(node_id)
            if node_data and node_data.get('type') == 'memory':
                associative_scores[node_id] = score
                
        # 4. Hybrid Scoring & Re-ranking
        # Normalize associative scores to [0, 1] range if needed
        # (Spreading activation scores can exceed 1.0 depending on accumulation logic)
        if associative_scores:
            max_assoc = max(associative_scores.values())
            for mid in associative_scores:
                associative_scores[mid] /= max_assoc
                
        # Combine all unique memory IDs
        all_ids = set(semantic_scores.keys()) | set(associative_scores.keys())
        
        combined_results = []
        for mid in all_ids:
            s_score = semantic_scores.get(mid, 0.0)
            a_score = associative_scores.get(mid, 0.0)
            
            # Hybrid Score
            final_score = alpha * s_score + (1.0 - alpha) * a_score
            
            # Load memory object if not already loaded in vector_results
            memory_obj = None
            for res in vector_results:
                if res.memory_id == mid:
                    memory_obj = res.memory
                    break
            
            if memory_obj is None:
                memory_obj = self.vector_search.storage.load(mid, category)
                
            combined_results.append(SearchResult(
                memory_id=mid,
                similarity=final_score,
                memory=memory_obj
            ))
            
        # 5. Sort and Return
        combined_results.sort(key=lambda x: x.similarity, reverse=True)
        return combined_results[:top_k]

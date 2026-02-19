
import os
import sys
import tempfile
from pathlib import Path
import networkx as nx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from llm_compression.knowledge_graph.manager import KnowledgeGraphManager

def test_multi_hop_reasoning():
    print("=" * 60)
    print("  Knowledge Graph: Multi-Hop Reasoning Test")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_path = Path(tmp_dir)
        kg = KnowledgeGraphManager(storage_path)
        
        # 1. Create a chain of memories
        # Memory A: [AI, Machine Learning]
        # Memory B: [Machine Learning, Neural Networks]
        # Memory C: [Neural Networks, Transformers]
        
        print("[1] Building knowledge chain...")
        kg.add_memory_concepts("mem_a", ["AI", "Machine Learning"], [0.9, 0.8])
        kg.add_memory_concepts("mem_b", ["Machine Learning", "Neural Networks"], [0.7, 0.9])
        kg.add_memory_concepts("mem_c", ["Neural Networks", "Transformers"], [0.6, 1.0])
        
        # 2. Add some direct attention relations
        # Transformers -> Attention Mechanism -> Transformers
        kg.add_concept_relations([("Transformers", "Attention", 0.95)])
        # Memory D: [Attention, Scaling]
        kg.add_memory_concepts("mem_d", ["Attention", "Scaling"], [0.9, 0.5])
        
        print(f"Graph nodes: {kg.graph.number_of_nodes()}")
        print(f"Graph edges: {kg.graph.number_of_edges()}")
        
        # 3. Test Multi-Hop Search
        # Query: "AI"
        # Expectation: 
        # - Hop 1: AI (concept), mem_a (memory), Machine Learning (concept)
        # - Hop 2: mem_b (via ML)
        # - Hop 3: Neural Networks (via mem_b)
        
        print("\n[2] Testing Multi-Hop Memory Retrieval for 'AI'...")
        results = kg.find_related_memories(["AI"], max_hops=4)
        
        print("Related Memories found:")
        for res in results:
            print(f"  - {res['memory_id']} (Relevance: {res['relevance']})")
            
        mids = [r['memory_id'] for r in results]
        assert "mem_a" in mids
        assert "mem_b" in mids
        
        # 4. Test Multi-Hop Concept Discovery
        # Query from "AI"
        print("\n[3] Testing Multi-Hop Concept Discovery for 'AI'...")
        concepts = kg.find_related_concepts("AI", max_depth=3)
        print("Related Concepts discovered:")
        for c in concepts:
            print(f"  - {c['concept']} (Depth: {c['depth']}, Relevance: {c['relevance']:.4f})")
            
        # Check if "Neural Networks" is discovered through AI -> ML -> mem_b -> NN or similar
        found_nn = any(c['concept'] == 'neural networks' for c in concepts)
        print(f"Found 'Neural Networks' from 'AI': {found_nn}")
        
        # 5. Complex reasoning: "AI" -> "Transformers"
        print("\n[4] Reasoning: Path from 'AI' to 'Transformers'?")
        try:
            path = nx.shortest_path(kg.graph, "ai", "transformers")
            print(f"  Reasoning Path: {' -> '.join(path)}")
        except nx.NetworkXNoPath:
            print("  No path found.")

        if "mem_c" in mids or any(c['concept'] == 'transformers' for c in concepts):
             print("\n✅ Success: Cross-memory multi-hop reasoning working.")
        else:
             print("\n⚠️ Note: Limited reach with current weights/hops.")

if __name__ == "__main__":
    test_multi_hop_reasoning()

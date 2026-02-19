
import unittest
import networkx as nx
from llm_compression.knowledge_graph.navigator import GraphNavigator

class TestGraphNavigator(unittest.TestCase):
    def setUp(self):
        # Create a sample graph
        self.g = nx.Graph()
        
        # Concepts
        self.g.add_node("python", type="concept")
        self.g.add_node("code", type="concept")
        self.g.add_node("ai", type="concept")
        
        # Memories
        self.g.add_node("mem_1", type="memory") # Related to python, code
        self.g.add_node("mem_2", type="memory") # Related to python, ai
        
        # Edges
        self.g.add_edge("python", "mem_1", weight=0.8)
        self.g.add_edge("code", "mem_1", weight=0.5)
        self.g.add_edge("python", "mem_2", weight=0.6)
        self.g.add_edge("ai", "mem_2", weight=0.9)
        self.g.add_edge("python", "code", weight=0.3)
        self.g.add_edge("python", "ai", weight=0.4)
        
        self.navigator = GraphNavigator(self.g)
        
    def test_spread_activation(self):
        # Start from "python"
        results = self.navigator.spread_activation(["python"])
        
        # python -> mem_1 (0.8)
        # python -> mem_2 (0.6)
        # python -> code (0.3)
        # python -> ai (0.4)
        
        # Verify activations exist
        found = {n[0]: n[1] for n in results}
        self.assertIn("mem_1", found)
        self.assertIn("mem_2", found)
        
        # mem_1 should be > mem_2 due to weight (0.8 vs 0.6)?
        # Actually decay * weight.
        # Check order
        self.assertGreater(found["python"], found["mem_1"]) 
        
    def test_related_memories(self):
        # Query: "code" -> expecting mem_1 strongly
        memories = self.navigator.get_related_memories(["code"])
        top_mem = memories[0]
        self.assertEqual(top_mem[0], "mem_1")
        
    def test_multi_hop_association(self):
        # Query: "code"
        # code -> python (0.3) -> mem_2 (0.6)
        # mem_2 is related to code via python, even if not directly connected strongly?
        # Let's say code -> mem_2 is NOT directly connected.
        # But code -> python -> mem_2 exists.
        
        # Let's remove any direct edge if any
        if self.g.has_edge("code", "mem_2"):
             self.g.remove_edge("code", "mem_2")
             
        # "code" activates "python" (0.3) which activates "mem_2" (0.6 * 0.3 * decay)
        results = self.navigator.get_related_memories(["code"], top_k=5)
        memory_ids = [m[0] for m in results]
        
        self.assertIn("mem_2", memory_ids) # Found via association

if __name__ == "__main__":
    unittest.main()

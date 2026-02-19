
import unittest
import shutil
import tempfile
import networkx as nx
from pathlib import Path
from llm_compression.knowledge_graph.manager import KnowledgeGraphManager

class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.tmp_dir)
        self.kg = KnowledgeGraphManager(self.storage_path)
        
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        
    def test_add_memory(self):
        memory_id = "mem_123"
        concepts = ["Machine Learning", "Python", "Data Structure"]
        scores = [0.9, 0.8, 0.5]
        
        self.kg.add_memory_concepts(memory_id, concepts, scores)
        
        # Verify nodes exist (lowercase)
        self.assertTrue(self.kg.graph.has_node("python"))
        self.assertTrue(self.kg.graph.has_node("mem_123")) 
        
        # Verify edge Memory -> Concept
        self.assertTrue(self.kg.graph.has_edge("mem_123", "machine learning"))
        
        # Verify Concept -> Concept (Python related to Machine Learning)
        self.assertTrue(self.kg.graph.has_edge("python", "machine learning"))
        
    def test_find_related(self):
        # Memory 1: AI, Python
        self.kg.add_memory_concepts("m1", ["AI", "Python"], [0.9, 0.8])
        # Memory 2: Python, Code
        self.kg.add_memory_concepts("m2", ["Python", "Code"], [0.8, 0.7])
        
        # Python should relate to AI and Code
        related = self.kg.find_related_concepts("Python")
        print(f"Related to Python: {related}")
        
        self.assertIn("ai", related)
        self.assertIn("code", related)
        
    def test_persistence(self):
        self.kg.add_memory_concepts("m1", ["Test"], [1.0])
        self.kg.save()
        
        # Reload
        new_kg = KnowledgeGraphManager(self.storage_path)
        self.assertTrue(new_kg.graph.has_node("test"))
        self.assertEqual(len(new_kg.graph.nodes), 2) # m1 and test

if __name__ == "__main__":
    unittest.main()

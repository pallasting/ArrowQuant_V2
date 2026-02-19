
import unittest
import tempfile
import shutil
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock

from llm_compression.arrow_storage import ArrowStorage
from llm_compression.vector_search import VectorSearch
from llm_compression.knowledge_graph.manager import KnowledgeGraphManager
from llm_compression.knowledge_graph.hybrid_navigator import HybridNavigator
from llm_compression.compression.attention_extractor import AttentionBasedExtractor, KeyInformation
from llm_compression.arrow_native_compressor import ArrowNativeCompressor

class MockEngine:
    def __init__(self):
        self.tokenizer = MagicMock()
        self.tokenizer.convert_ids_to_tokens.return_value = []
        
    def encode(self, text, output_attentions=False, normalize=True):
        # Return deterministic vector based on content for simple similarity
        if "neural" in text.lower():
            v = np.array([1.0] + [0.0]*383, dtype=np.float32)
        elif "machine" in text.lower():
            v = np.array([0.0, 1.0] + [0.0]*382, dtype=np.float32)
        else:
            v = np.array([0.0]*384, dtype=np.float32)
            
        if output_attentions:
            return v.reshape(1, -1), [tuple([torch.zeros(1,1,1,1)]*12)]
        return v.reshape(1, -1)

class TestHybridRetrieval(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.path = Path(self.test_dir)
        
        self.engine = MockEngine()
        self.kg_manager = KnowledgeGraphManager(self.path)
        self.storage = ArrowStorage(storage_path=self.test_dir, kg_manager=self.kg_manager)
        self.vector_search = VectorSearch(MagicMock(), self.storage)
        # Mock vector search embedder
        self.vector_search.embedder.encode.side_effect = lambda q, normalize=True: self.engine.encode(q)[0]
        
        self.extractor = MagicMock()
        self.hybrid = HybridNavigator(self.vector_search, self.kg_manager, self.extractor)
        
        self.native_compressor = ArrowNativeCompressor(self.engine)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_associative_recall(self):
        # 1. Prepare Memories
        # Memory 1: Related to Machine Learning
        m1_text = "Machine learning models are trained on large datasets."
        self.extractor.extract_key_information.return_value = KeyInformation(
            key_tokens=["machine", "learning", "models"],
            token_scores=[0.9, 0.8, 0.7]
        )
        c1 = self.native_compressor.compress(m1_text)
        # Manually fix key_tokens and token_scores since we mocked extractor later
        c1.key_tokens = ["machine", "learning", "models"]
        c1.token_scores = [0.9, 0.8, 0.7]
        self.storage.save(c1)
        
        # Memory 2: Related to Neural Networks & Machine Learning
        m2_text = "Neural networks are a key part of machine learning."
        self.extractor.extract_key_information.return_value = KeyInformation(
            key_tokens=["neural", "networks", "machine", "learning"],
            token_scores=[0.95, 0.9, 0.6, 0.5]
        )
        c2 = self.native_compressor.compress(m2_text)
        c2.key_tokens = ["neural", "networks", "machine", "learning"]
        c2.token_scores = [0.95, 0.9, 0.6, 0.5]
        self.storage.save(c2)
        
        # Verify KG state
        self.assertTrue(self.kg_manager.graph.has_edge("machine", "learning"))
        self.assertTrue(self.kg_manager.graph.has_edge(c1.memory_id, "machine"))
        self.assertTrue(self.kg_manager.graph.has_edge(c2.memory_id, "machine"))
        
        # 2. Query "neural networks"
        # Pure semantic search should find m2 (similarity ~1.0) but NOT m1 (similarity ~0.0)
        query = "neural networks"
        
        # Mock extractor for the query
        self.extractor.extract_key_information.return_value = KeyInformation(
            key_tokens=["neural", "networks"],
            token_scores=[1.0, 1.0]
        )
        
        # Execute Hybrid Search
        results = self.hybrid.search(query, alpha=0.5)
        
        ids = [r.memory_id for r in results]
        print(f"Hybrid Results IDs: {ids}")
        
        # Result 1 should be m2 (Semantic + Assoc)
        self.assertEqual(ids[0], c2.memory_id)
        
        # Result 2 should be m1 (Assoc via "machine")
        # Even though semantic similarity is low, associative activation should bring it up.
        self.assertIn(c1.memory_id, ids)
        
        print("Hybrid Associative Recall Test Passed!")

if __name__ == "__main__":
    unittest.main()

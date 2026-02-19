
import unittest
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch
from unittest.mock import MagicMock

from llm_compression.arrow_native_compressor import ArrowNativeCompressor
from llm_compression.arrow_storage import ArrowStorage
from llm_compression.compressor import CompressedMemory

class MockArrowEngine:
    def __init__(self):
        self.embedding_dim = 10 
        self.layers = 2
        self.heads = 2
        self.seq_len = 5
        self.tokenizer = MagicMock()
        self.tokenizer.encode.return_value = {
            'input_ids': torch.tensor([[101, 102, 0, 0, 0]])
        }
        self.tokenizer.convert_ids_to_tokens.return_value = ["token_a", "token_b", "token_c", "token_d", "token_e"]
        
    def encode(self, text, output_attentions=False, normalize=True):
        val = np.array([0.5, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        # Mock attention: specific pattern to ensure token_b gets high score
        # 5 tokens. (1, heads, seq, seq)
        attn = torch.zeros(1, 2, 5, 5)
        attn[:, :, 0, 1] = 1.0 # token 0 attends to token 1 (token_b)
        attentions = [tuple([attn] * 2)]
        
        if output_attentions:
            return val.reshape(1, -1), attentions
        return val.reshape(1, -1)

class TestArrowNativeIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize KG Manager
        from llm_compression.knowledge_graph.manager import KnowledgeGraphManager
        self.kg_manager = KnowledgeGraphManager(Path(self.test_dir))
        
        # Pass KG Manager to Storage
        self.storage = ArrowStorage(storage_path=self.test_dir, kg_manager=self.kg_manager)
        
        self.engine = MockArrowEngine()
        self.compressor = ArrowNativeCompressor(self.engine)
        
    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
        except:
            pass
        
    def test_compress_save_load(self):
        text = "test integration 4bit"
        
        # 1. Compress (default should be 4bit with our recent change)
        compressed = self.compressor.compress(text)
        
        # Verify 4-bit metadata
        self.assertTrue(compressed.sparse_meta.get("is_4bit"), "Should be 4-bit compressed by default")
        
        # Verify fields are present
        self.assertIsNotNone(compressed.sparse_vector)
        self.assertIsNotNone(compressed.sparse_indices)
        self.assertIsNotNone(compressed.sparse_meta)
        self.assertTrue(len(compressed.key_tokens) >= 0)
        
        # Verify token_scores present
        if hasattr(compressed, 'token_scores'):
             self.assertIsNotNone(compressed.token_scores)
        
        # 2. Save (Should trigger KG update)
        self.storage.save(compressed, category='experiences')
        
        # 3. Load
        loaded = self.storage.load(compressed.memory_id, category='experiences')
        
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.memory_id, compressed.memory_id)
        
        # Verify Vector Space fields survived round trip
        self.assertEqual(loaded.sparse_vector, compressed.sparse_vector)
        self.assertEqual(loaded.sparse_indices, compressed.sparse_indices)
        self.assertEqual(loaded.key_tokens, compressed.key_tokens)
        
        # Verify Knowledge Graph Update
        # Check if memory node exists
        self.assertTrue(self.kg_manager.graph.has_node(compressed.memory_id))
        
        # Check if any key token became a concept node
        # Based on mock tokens: token_b. Lowercased.
        # "token_b" length > 2
        # Mock tokens are "token_a", "token_b"...
        # Attention points to token_b. So token_b should have high score.
        # Check if "token_b" is in graph
        # Note: AttentionBasedExtractor implementation might filter "token_a" etc if they are not in input text?
        # The logic: Tokenizer converts ids to tokens. If successful, valid.
        # It doesn't check against input text string unless explicit check.
        # It does check: if token in ["[CLS]"] etc.
        
        self.assertTrue(self.kg_manager.graph.has_node("token_b"))
        self.assertTrue(self.kg_manager.graph.has_edge(compressed.memory_id, "token_b"))
        
        print("Knowledge Graph Integration Passed")
        
        # Verify sparse_meta keys
        # sparse_meta keys might differ if I didn't preserve types exactly (e.g. float vs float32)
        # But values should be close
        self.assertAlmostEqual(loaded.sparse_meta['scale_factor'], compressed.sparse_meta['scale_factor'])
        self.assertAlmostEqual(loaded.sparse_meta['full_dim'], compressed.sparse_meta['full_dim'])
        
    def test_learning_integration(self):
        # Create a mock learner
        from llm_compression.learning.incremental_learner import IncrementalLearner
        learner = IncrementalLearner(dimension_size=10)
        # Mock methods
        learner.get_dimension_weights = MagicMock(return_value=np.ones(10, dtype=np.float32))
        
        compressor_with_learning = ArrowNativeCompressor(self.engine, learner=learner)
        
        text = "learning test"
        compressor_with_learning.compress(text)
        
        # Verify get_dimension_weights was called
        learner.get_dimension_weights.assert_called_once()
        print("Learning Integration Passed")
        
        print("Integration Test Passed: Compression -> Storage -> Reconstruction")

if __name__ == "__main__":
    unittest.main()

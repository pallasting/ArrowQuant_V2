
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
from llm_compression.compression.vector_compressor import VectorSpaceCompressor, CompressedMemory
from llm_compression.learning.incremental_learner import IncrementalLearner

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
        self.tokenizer.convert_ids_to_tokens.return_value = ["a", "b", "c", "d", "e"]
        
    def encode(self, text, output_attentions=False, normalize=True):
        # Deterministic but with specific pattern to test learning
        # Let's say indices 0 and 1 have moderate values, others low.
        val = np.array([0.5, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
       
        # If we "learn", we might boost index 1 to be higher than 0 if index 1 is used often
        # (Assuming initially index 0 > index 1)
        
        attentions = [tuple([torch.zeros(1, 2, 5, 5)] * 2)]
        
        if output_attentions:
            return val.reshape(1, -1), attentions
        return val.reshape(1, -1)

class TestIncrementalLearning(unittest.TestCase):
    def setUp(self):
        self.engine = MockArrowEngine()
        self.compressor = VectorSpaceCompressor(self.engine)
        self.learner = IncrementalLearner(dimension_size=10)
        
    def test_learning_cycle(self):
        # 1. Compress a text
        text = "test"
        # Ratio 0.2 -> Keep 2 dimensions out of 10
        # Initially values are [0.5, 0.4, ...] so indices 0 and 1 should be kept.
        
        c1 = self.compressor.compress(text, compression_ratio=0.2)
        indices = c1.key_indices
        print(f"Initial indices: {indices}")
        self.assertTrue(0 in indices)
        self.assertTrue(1 in indices)
        
        # 2. Simulate access to a memory that uses index 2 HEAVILY
        # We manually construct a compressed memory that uses index 2
        fake_mem = CompressedMemory(
            sparse_vector=np.array([100], dtype=np.int8),
            key_indices=np.array([2], dtype=np.uint16),
            original_norm=1.0,
            meta_info={}
        )
        
        # 3. Feed to learner many times to boost index 2
        for _ in range(50):
            self.learner.record_access(fake_mem)
            
        # Get weights
        weights = self.learner.get_dimension_weights(learning_rate=2.0)
        print(f"Weights: {weights}")
        self.assertGreater(weights[2], weights[0])
        
        # 4. Now compress again with weights
        # Original val at idx 2 is 0.1. Weight at idx 2 is high (e.g. 1 + 1*5*2 = 11).
        # Weighted val at idx 2 = 0.1 * 11 = 1.1 > 0.5 (idx 0).
        # So index 2 should now be selected.
        
        c2 = self.compressor.compress(text, compression_ratio=0.2, dimension_weights=weights)
        new_indices = c2.key_indices
        print(f"New indices: {new_indices}")
        
        self.assertTrue(2 in new_indices) # Should now include 2
        
if __name__ == "__main__":
    unittest.main()

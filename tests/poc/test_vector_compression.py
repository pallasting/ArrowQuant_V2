
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
from llm_compression.compression.vector_compressor import VectorSpaceCompressor, CompressedMemory

class MockArrowEngine:
    def __init__(self):
        self.embedding_dim = 384
        self.layers = 6
        self.heads = 12
        self.seq_len = 10
        
        self.tokenizer = MagicMock()
        self.tokenizer.encode.return_value = {
            'input_ids': torch.tensor([[101, 102, 103] + [0]*7])
        }
        self.tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "test", "[SEP]"] + ["[PAD]"]*7
        
    def encode(self, text, output_attentions=False, normalize=True):
        # Deterministic random embedding based on text
        seed = hash(text) % (2**32)
        rng = np.random.RandomState(seed)
        
        # Return random embedding
        emb = rng.randn(1, self.embedding_dim).astype(np.float32)
        if normalize:
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            
        if not output_attentions:
            return emb
            
        # Mock attentions: List[Tuple[Tensor]]
        # 1 batch, 1 item
        # Tuple of layers
        # Each tensor: (batch=1, heads=12, seq=10, seq=10)
        attentions = []
        batch_atts = []
        for _ in range(self.layers):
            layer_att = torch.rand(1, self.heads, self.seq_len, self.seq_len)
            # Softmax to make it look like attention
            layer_att = torch.nn.functional.softmax(layer_att, dim=-1)
            batch_atts.append(layer_att)
            
        attentions = [tuple(batch_atts)]
        
        return emb, attentions

class TestVectorCompression(unittest.TestCase):
    def setUp(self):
        self.engine = MockArrowEngine()
        self.compressor = VectorSpaceCompressor(self.engine)
        
    def test_compression_structure(self):
        text = "Hello world this is a test"
        compressed = self.compressor.compress(text, compression_ratio=0.5)
        
        self.assertIsInstance(compressed, CompressedMemory)
        self.assertEqual(len(compressed.sparse_vector), int(384 * 0.5))
        self.assertEqual(len(compressed.key_indices), int(384 * 0.5))
        self.assertIn("key_tokens", compressed.meta_info)
        self.assertIn("token_scores", compressed.meta_info)
        
    def test_reconstruction(self):
        text = "Test reconstruction"
        # 1. Compress
        compressed = self.compressor.compress(text, compression_ratio=0.8) # Keep 80%
        
        # 2. Reconstruct
        reconstructed = self.compressor.reconstruct(compressed)
        
        # 3. Check shape
        self.assertEqual(reconstructed.shape, (384,))
        
        # 4. Check similarity (should be high as we kept 80% of dimensions)
        # Original (we need to get it to compare)
        original, _ = self.engine.encode(text, output_attentions=True)
        original = original[0]
        
        # Cosine similarity
        sim = np.dot(reconstructed, original) / (np.linalg.norm(reconstructed) * np.linalg.norm(original))
        print(f"Reconstruction Similarity (80% kept): {sim:.4f}")
        self.assertGreater(sim, 0.70) # Random vectors might lose info fast but 80% should be good
        
    def test_compression_ratio_calculation(self):
        # Original size: 384 * 4 bytes = 1536 bytes
        # Compressed: 
        #   sparse_vector: k * 1 byte (int8)
        #   key_indices: k * 2 bytes (uint16)
        #   metadata: ~small
        # For k=38 (10%): 38*3 = 114 bytes
        # Ratio: 1536 / 114 ~= 13.5x
        
        text = "Compression test"
        ratio = 0.1 # Keep 10%
        compressed = self.compressor.compress(text, compression_ratio=ratio)
        
        k = len(compressed.sparse_vector)
        size_bytes = k * 1 + k * 2 # values + indices
        
        print(f"Compressed size (bytes): {size_bytes} vs Original: {384*4}")
        self.assertLess(size_bytes, 384*4 * 0.2) # Should be < 20% of original

if __name__ == "__main__":
    unittest.main()


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
        self.tokenizer.convert_ids_to_tokens.return_value = ["a", "b", "c", "d", "e"]
        
    def encode(self, text, output_attentions=False, normalize=True):
        val = np.array([0.5, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        attentions = [tuple([torch.zeros(1, 2, 5, 5)] * 2)]
        
        if output_attentions:
            return val.reshape(1, -1), attentions
        return val.reshape(1, -1)

class TestArrowNativeIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage = ArrowStorage(storage_path=self.test_dir)
        self.engine = MockArrowEngine()
        self.compressor = ArrowNativeCompressor(self.engine)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
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
        
        # 2. Save
        self.storage.save(compressed, category='experiences')
        
        # 3. Load
        loaded = self.storage.load(compressed.memory_id, category='experiences')
        
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.memory_id, compressed.memory_id)
        
        
        # Verify Vector Space fields survived round trip
        print(f"Original Vector Type: {type(compressed.sparse_vector)}")
        print(f"Loaded Vector Type: {type(loaded.sparse_vector)}")
        print(f"Original Vector Len: {len(compressed.sparse_vector) if compressed.sparse_vector else 'None'}")
        print(f"Loaded Vector Len: {len(loaded.sparse_vector) if loaded.sparse_vector else 'None'}")
        
        self.assertEqual(loaded.sparse_vector, compressed.sparse_vector)
        self.assertEqual(loaded.sparse_indices, compressed.sparse_indices)
        self.assertEqual(loaded.key_tokens, compressed.key_tokens)
        
        # Verify sparse_meta
        # sparse_meta keys might differ if I didn't preserve types exactly (e.g. float vs float32)
        # But values should be close
        self.assertAlmostEqual(loaded.sparse_meta['scale_factor'], compressed.sparse_meta['scale_factor'])
        self.assertAlmostEqual(loaded.sparse_meta['full_dim'], compressed.sparse_meta['full_dim'])
        
        print("Integration Test Passed: Compression -> Storage -> Reconstruction")

if __name__ == "__main__":
    unittest.main()

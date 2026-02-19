
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
from llm_compression.compression.attention_extractor import AttentionBasedExtractor, KeyInformation

class MockArrowEngine:
    def __init__(self):
        self.embedding_dim = 384
        self.layers = 6
        self.heads = 12
        self.seq_len = 5 # Small for testing
        
        # Mock tokenizer methods
        self.tokenizer = MagicMock()
        self.tokenizer.encode.return_value = {
            'input_ids': torch.tensor([[101, 2023, 1045, 2038, 102]]) # [CLS] this is test [SEP]
        }
        self.tokenizer.convert_ids_to_tokens.return_value = [
            "[CLS]", "this", "is", "test", "[SEP]"
        ]
        
    def encode(self, text, output_attentions=False, normalize=True):
        # Return dummy embedding
        emb = np.random.randn(1, self.embedding_dim).astype(np.float32)
        
        if not output_attentions:
            return emb
            
        # Mock attentions
        # Shape: (1, 12, 5, 5) per layer
        # Make "test" token (idx 3) receive high attention
        layer_att = torch.rand(1, self.heads, self.seq_len, self.seq_len)
        
        # Manually increase attention to index 3 (test)
        layer_att[:, :, :, 3] += 10.0
        
        attentions = [tuple([layer_att] * self.layers)]
        
        return emb, attentions

class TestAttentionExtraction(unittest.TestCase):
    def setUp(self):
        self.engine = MockArrowEngine()
        self.extractor = AttentionBasedExtractor(self.engine)
        
    def test_key_token_extraction(self):
        text = "this is test"
        
        info = self.extractor.extract_key_information(text, top_k=2)
        
        self.assertIsInstance(info, KeyInformation)
        self.assertTrue(len(info.key_tokens) > 0)
        
        # Since we mocked high attention to "test" (idx 3), it should be top
        self.assertEqual(info.key_tokens[0], "test")
        
        # Verify scores are sorted desc
        scores = info.token_scores
        self.assertTrue(all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))
        
    def test_special_tokens_excluded(self):
        text = "dummy"
        info = self.extractor.extract_key_information(text)
        
        self.assertNotIn("[CLS]", info.key_tokens)
        self.assertNotIn("[SEP]", info.key_tokens)
        self.assertNotIn("[PAD]", info.key_tokens)

if __name__ == "__main__":
    unittest.main()

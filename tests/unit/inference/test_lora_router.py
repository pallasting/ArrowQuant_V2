
import unittest
import numpy as np

from llm_compression.inference.lora_router import LoRARouter
from llm_compression.inference.lora_manager import LoRAManager
from llm_compression.inference.lora_format import LoRACard

class TestLoRARouter(unittest.TestCase):
    def test_router_logic(self):
        """Test basic LoRA intent matching."""
        
        # Mock Embedder
        # Simply maps keyword -> vector
        vocab = {
            "code": np.array([1.0, 0.0, 0.0]),
            "python": np.array([0.9, 0.1, 0.0]),
            "write": np.array([0.5, 0.5, 0.0]),
            "art": np.array([0.0, 1.0, 0.0]),
            "paint": np.array([0.0, 0.9, 0.1])
        }
        
        def embed(text: str) -> np.ndarray:
            # Simple BoW for test
            vec = np.zeros(3)
            words = text.lower().replace(":", "").split()
            count = 0
            for w in words:
                if w in vocab:
                    vec += vocab[w]
                    count += 1
            if count > 0:
                vec /= count
            return vec
            
        # Mock Manager
        class MockManager:
            pass
            
        manager = MockManager()
        router = LoRARouter(manager, embedder_func=embed)
        
        # Create Dummy Cards
        card_code = LoRACard(
            name="coding_expert",
            rank=4, alpha=16, target_modules=[], weights_A={}, weights_B={},
            metadata={"description": "expert at writing python code"}
        )
        
        card_art = LoRACard(
            name="art_generator",
            rank=4, alpha=16, target_modules=[], weights_A={}, weights_B={},
            metadata={"description": "generates beautiful art and paintings"}
        )
        
        # Register
        router.register_card(card_code)
        router.register_card(card_art)
        
        # Test Query 1: "write python code"
        # Should match coding_expert
        selected = router.select("write python code", threshold=0.5)
        self.assertIn("coding_expert", selected)
        self.assertNotIn("art_generator", selected)
        
        # Test Query 2: "paint art"
        # Should match art_generator
        selected = router.select("paint art", threshold=0.5)
        self.assertIn("art_generator", selected)
        self.assertNotIn("coding_expert", selected)
        
if __name__ == "__main__":
    unittest.main()

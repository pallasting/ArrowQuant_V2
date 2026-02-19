
import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os

from llm_compression.inference.lora_layer import LoRALinear
from llm_compression.inference.lora_format import LoRACard, LoRAFormat
from llm_compression.inference.lora_manager import LoRAManager
from llm_compression.inference.inference_core import InferenceCore

class TestLoRA(unittest.TestCase):
    def test_lora_layer_logic(self):
        """Test LoRALinear forward pass."""
        in_dim, out_dim = 10, 5
        rank = 2
        
        original = nn.Linear(in_dim, out_dim)
        lora = LoRALinear(original, rank=rank, alpha=32.0)
        
        # 1. Forward pass should work
        x = torch.randn(1, 1, in_dim)
        y = lora(x)
        self.assertEqual(y.shape, (1, 1, out_dim))
        
        # 2. Forward pass result should differ from original (non-zero init B)
        # But wait, B is init to zeros, so output should be IDENTICAL initially!
        y_orig = original(x)
        self.assertTrue(torch.allclose(y, y_orig), "Initialized LoRA should match original (B=0)")
        
        # 3. Change B weights, output should change
        with torch.no_grad():
            lora.lora_B.data.fill_(1.0)
            
        y_new = lora(x)
        self.assertFalse(torch.allclose(y_new, y_orig), "Modified LoRA should differ from original")
        
        # 4. Disable logic
        lora.enabled = False
        y_disabled = lora(x)
        self.assertTrue(torch.allclose(y_disabled, y_orig), "Disabled LoRA should match original")

    def test_lora_format_io(self):
        """Test saving and loading LoRACard."""
        card = LoRACard(
            name="test_lora",
            rank=4,
            alpha=16.0,
            target_modules=["q_proj"],
            weights_A={"layer.0.q_proj": np.random.rand(4, 10).astype(np.float32)},
            weights_B={"layer.0.q_proj": np.random.rand(10, 4).astype(np.float32)},
            metadata={"description": "unit test"}
        )
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
            
        try:
            LoRAFormat.save(card, path)
            loaded = LoRAFormat.load(path)
            
            self.assertEqual(loaded.name, card.name)
            self.assertEqual(loaded.rank, card.rank)
            self.assertEqual(loaded.target_modules, card.target_modules)
            self.assertTrue("layer.0.q_proj" in loaded.weights_A)
            
            # Check values
            np.testing.assert_array_equal(loaded.weights_A["layer.0.q_proj"], card.weights_A["layer.0.q_proj"])
            
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_lora_manager_injection(self):
        """Test injecting LoRA into dummy InferenceCore."""
        # Mock InferenceCore
        in_dim, out_dim = 32, 32
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(in_dim, out_dim)
        
        class MockCore(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MockModel()
                
        core = MockCore()
        manager = LoRAManager(core)
        
        # Create Dummy Card targeting "layer"
        card = LoRACard(
            name="mock_lora",
            rank=4,
            alpha=16.0,
            target_modules=["layer"], # The linear layer is named "layer"
            weights_A={"layer": np.random.randn(4, 32).astype(np.float32)},
            weights_B={"layer": np.random.randn(32, 4).astype(np.float32)}
        )
        
        # Apply
        manager.apply_card(card)
        
        # Check injection
        self.assertIsInstance(core.model.layer, LoRALinear)
        self.assertEqual(core.model.layer.rank, 4)
        
        # Remove
        manager.remove_card("mock_lora")
        self.assertIsInstance(core.model.layer, nn.Linear)
        self.assertNotIsInstance(core.model.layer, LoRALinear)

if __name__ == "__main__":
    unittest.main()

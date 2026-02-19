
import unittest
import torch
import numpy as np
from llm_compression.inference.vision_core import VisionInferenceCore
from llm_compression.multimodal.image_processor import ImageProcessor

class TestVisionCore(unittest.TestCase):
    def setUp(self):
        # 1. Define a tiny config for testing (Micro-ViT)
        self.config = {
            "image_size": 32,          # Small image
            "patch_size": 8,           # 4x4 patches
            "hidden_size": 64,         # Small hidden dim
            "num_layers": 2,           # 2 layers
            "num_attention_heads": 2,  # 2 heads
            "intermediate_size": 256,
            "layer_norm_eps": 1e-5,
            "projection_dim": 32       # Output dim
        }
        
        # 2. Generate Random Weights matching the config
        # This simulates what convert_clip.py would produce
        self.weights = {}
        
        # Patch Embedding [Out, In, kH, kW]
        self.weights['vision_model.embeddings.patch_embedding.weight'] = torch.randn(64, 3, 8, 8)
        
        # Class + Pos Embeddings
        num_patches = (32 // 8) ** 2
        self.weights['vision_model.embeddings.class_embedding'] = torch.randn(64)
        self.weights['vision_model.embeddings.position_embedding.weight'] = torch.randn(num_patches + 1, 64)
        self.weights['vision_model.pre_layrnorm.weight'] = torch.ones(64)
        self.weights['vision_model.pre_layrnorm.bias'] = torch.zeros(64)
        
        # Layers (0 and 1)
        for i in range(2):
            p = f"vision_model.encoder.layers.{i}"
            # Attn
            self.weights[f"{p}.self_attn.q_proj.weight"] = torch.randn(64, 64)
            self.weights[f"{p}.self_attn.q_proj.bias"] = torch.zeros(64)
            self.weights[f"{p}.self_attn.k_proj.weight"] = torch.randn(64, 64)
            self.weights[f"{p}.self_attn.k_proj.bias"] = torch.zeros(64)
            self.weights[f"{p}.self_attn.v_proj.weight"] = torch.randn(64, 64)
            self.weights[f"{p}.self_attn.v_proj.bias"] = torch.zeros(64)
            self.weights[f"{p}.self_attn.out_proj.weight"] = torch.randn(64, 64)
            self.weights[f"{p}.self_attn.out_proj.bias"] = torch.zeros(64)
            self.weights[f"{p}.layer_norm1.weight"] = torch.ones(64)
            self.weights[f"{p}.layer_norm1.bias"] = torch.zeros(64)
            # MLP
            self.weights[f"{p}.mlp.fc1.weight"] = torch.randn(256, 64)
            self.weights[f"{p}.mlp.fc1.bias"] = torch.zeros(256)
            self.weights[f"{p}.mlp.fc2.weight"] = torch.randn(64, 256)
            self.weights[f"{p}.mlp.fc2.bias"] = torch.zeros(64)
            self.weights[f"{p}.layer_norm2.weight"] = torch.ones(64)
            self.weights[f"{p}.layer_norm2.bias"] = torch.zeros(64)
            
        # Post
        self.weights['vision_model.post_layernorm.weight'] = torch.ones(64)
        self.weights['vision_model.post_layernorm.bias'] = torch.zeros(64)
        self.weights['visual_projection.weight'] = torch.randn(32, 64)

    def test_forward_pass(self):
        """Test if the Vision Core can process a preprocessed tensor."""
        # Initialize Core
        core = VisionInferenceCore(self.weights, self.config)
        
        # Create dummy input (Batch=1, Channels=3, H=32, W=32)
        # ImageProcessor output is (224,224,3) usually, but we config'd core for 32x32
        # PyTorch expects (B, 3, H, W) usually? 
        # Wait, VisionInferenceCore.forward docstring says (Batch, 3, H, W)
        # ImageProcessor output is (H, W, 3). We need to transpose.
        
        dummy_input = torch.randn(1, 3, 32, 32)
        
        # Run inference
        output = core(dummy_input)
        
        print("\n[VisionCore] Output shape:", output.shape)
        
        # Check shape: (Batch, ProjectionDim) -> (1, 32)
        self.assertEqual(output.shape, (1, 32))
        self.assertTrue(torch.is_tensor(output))

    def test_integration_with_processor(self):
        """Test full flow: ImageProcessor -> Transpose -> VisionCore"""
        # 1. ImageProcessor (configured for 32x32)
        processor = ImageProcessor(image_size=32)
        
        # 2. Fake Image (H, W, 3)
        fake_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 3. Preprocess -> (32, 32, 3) Normalized
        processed = processor.preprocess(fake_img)
        self.assertEqual(processed.shape, (32, 32, 3))
        
        # 4. Prepare for PyTorch (Numpy -> Tensor -> NHWC to NCHW)
        tensor_input = torch.from_numpy(processed).unsqueeze(0) # (1, 32, 32, 3)
        tensor_input = tensor_input.permute(0, 3, 1, 2) # (1, 3, 32, 32)
        
        # 5. Inference
        core = VisionInferenceCore(self.weights, self.config)
        embedding = core(tensor_input)
        
        print(f"\n[FullFlow] Input: {fake_img.shape} -> Processed: {processed.shape} -> Embedding: {embedding.shape}")
        self.assertEqual(embedding.shape, (1, 32))

if __name__ == "__main__":
    unittest.main()

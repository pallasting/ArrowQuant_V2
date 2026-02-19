
import unittest
import numpy as np
from llm_compression.compression.vector_compressor import VectorSpaceCompressor, CompressedMemory
from llm_compression.inference.arrow_engine import ArrowEngine

class MockArrowEngine:
    def __init__(self):
        self.device = "cpu"

class TestVisualCompression(unittest.TestCase):
    def setUp(self):
        self.engine = MockArrowEngine()
        self.compressor = VectorSpaceCompressor(self.engine)
        
    def test_visual_vector_compression(self):
        # 1. Simulate a 512-dim visual vector (float32)
        # Create a vector with some structure (not pure random) to test selection
        vector = np.zeros(512, dtype=np.float32)
        # Add 50 sparse peaks
        indices = np.random.choice(512, 50, replace=False)
        vector[indices] = np.random.randn(50) * 2.0 # High value peaks
        # Add low noise
        vector += np.random.randn(512) * 0.1
        
        original_size = vector.nbytes
        print(f"\n[VisualCompression] Original Size: {original_size} bytes")
        
        # 2. Compress (Keep 20% dimensions, 4-bit)
        compressed = self.compressor.compress_vector(
            vector, 
            compression_ratio=0.2, 
            use_4bit=True
        )
        
        # Check size
        sparse_vec_size = len(compressed.sparse_vector)
        sparse_idx_size = len(compressed.sparse_indices)
        total_size = sparse_vec_size + sparse_idx_size
        print(f"  Compressed Size: {total_size} bytes (Vec={sparse_vec_size}, Idx={sparse_idx_size})")
        print(f"  Ratio: {original_size / total_size:.1f}x")
        
        # 3. Reconstruct
        reconstructed = self.compressor.reconstruct(compressed)
        
        # 4. Check Fidelity (Cosine sim)
        sim = np.dot(vector, reconstructed) / (np.linalg.norm(vector) * np.linalg.norm(reconstructed))
        print(f"  Reconstruction Fidelity: {sim:.4f}")
        
        self.assertGreater(sim, 0.85) # Should maintain high fidelity for retrieval
        
    def test_pure_noise_robustness(self):
        # Even pure noise (like our Noise Image test) should be compressible
        vector = np.random.randn(512).astype(np.float32)
        compressed = self.compressor.compress_vector(vector, compression_ratio=0.5, use_4bit=True)
        reconstructed = self.compressor.reconstruct(compressed)
        
        sim = np.dot(vector, reconstructed) / (np.linalg.norm(vector) * np.linalg.norm(reconstructed))
        print(f"\n[NoiseTest] Fidelity: {sim:.4f}")
        self.assertGreater(sim, 0.6) # Noise is harder to compress (entropy high), but should be positive

if __name__ == "__main__":
    unittest.main()

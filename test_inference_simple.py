"""
Simple test script for AI-OS Diffusion inference module.
Run from project root directory.
"""

import sys
from pathlib import Path

# Add ai-os-diffusion to path
sys.path.insert(0, str(Path(__file__).parent / "ai-os-diffusion"))

print("=" * 60)
print("AI-OS Diffusion Inference Module - Simple Test")
print("=" * 60)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from inference import (
        ArrowEngine,
        WeightLoader,
        FastTokenizer,
        InferenceCore,
        get_optimal_device,
    )
    from config import Config
    from utils.logger import logger
    from utils.errors import DiffusionError, ModelLoadError
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Device detection
print("\n2. Testing device detection...")
try:
    device = get_optimal_device()
    print(f"✅ Detected device: {device}")
except Exception as e:
    print(f"❌ Device detection failed: {e}")

# Test 3: Configuration
print("\n3. Testing configuration...")
try:
    config = Config()
    print(f"✅ Config created: log_level={config.log_level}, device={config.device}")
except Exception as e:
    print(f"❌ Config failed: {e}")

# Test 4: Logger
print("\n4. Testing logger...")
try:
    logger.info("Test info message")
    logger.debug("Test debug message")
    print("✅ Logger working")
except Exception as e:
    print(f"❌ Logger failed: {e}")

# Test 5: Quantization schema
print("\n5. Testing quantization schema...")
try:
    from inference.quantization_schema import WEIGHT_SCHEMA_V1, WEIGHT_SCHEMA_V2
    print(f"✅ Schema V1: {len(WEIGHT_SCHEMA_V1)} columns")
    print(f"✅ Schema V2: {len(WEIGHT_SCHEMA_V2)} columns")
except Exception as e:
    print(f"❌ Quantization schema failed: {e}")

# Test 6: Decoder layers
print("\n6. Testing decoder layers...")
try:
    import torch
    from inference.decoder_layers import RMSNorm, precompute_freqs_cis
    
    norm = RMSNorm(dim=384)
    x = torch.randn(1, 10, 384)
    output = norm(x)
    print(f"✅ RMSNorm: {x.shape} -> {output.shape}")
    
    freqs_cis = precompute_freqs_cis(dim=64, end=512)
    print(f"✅ RoPE freqs_cis: {freqs_cis.shape}")
except Exception as e:
    print(f"❌ Decoder layers failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: InferenceCore with dummy weights
print("\n7. Testing InferenceCore...")
try:
    import torch
    from inference.inference_core import InferenceCore
    
    # Create tiny model
    hidden_size = 128
    num_layers = 2
    vocab_size = 1000
    
    weights = {}
    # Embeddings
    weights['embeddings.word_embeddings.weight'] = torch.randn(vocab_size, hidden_size)
    weights['embeddings.position_embeddings.weight'] = torch.randn(64, hidden_size)
    weights['embeddings.token_type_embeddings.weight'] = torch.randn(2, hidden_size)
    weights['embeddings.LayerNorm.weight'] = torch.ones(hidden_size)
    weights['embeddings.LayerNorm.bias'] = torch.zeros(hidden_size)
    
    # Encoder layers
    for i in range(num_layers):
        prefix = f"encoder.layer.{i}"
        weights[f"{prefix}.attention.self.query.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"{prefix}.attention.self.query.bias"] = torch.zeros(hidden_size)
        weights[f"{prefix}.attention.self.key.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"{prefix}.attention.self.key.bias"] = torch.zeros(hidden_size)
        weights[f"{prefix}.attention.self.value.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"{prefix}.attention.self.value.bias"] = torch.zeros(hidden_size)
        weights[f"{prefix}.attention.output.dense.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"{prefix}.attention.output.dense.bias"] = torch.zeros(hidden_size)
        weights[f"{prefix}.attention.output.LayerNorm.weight"] = torch.ones(hidden_size)
        weights[f"{prefix}.attention.output.LayerNorm.bias"] = torch.zeros(hidden_size)
        weights[f"{prefix}.intermediate.dense.weight"] = torch.randn(hidden_size * 4, hidden_size)
        weights[f"{prefix}.intermediate.dense.bias"] = torch.zeros(hidden_size * 4)
        weights[f"{prefix}.output.dense.weight"] = torch.randn(hidden_size, hidden_size * 4)
        weights[f"{prefix}.output.dense.bias"] = torch.zeros(hidden_size)
        weights[f"{prefix}.output.LayerNorm.weight"] = torch.ones(hidden_size)
        weights[f"{prefix}.output.LayerNorm.bias"] = torch.zeros(hidden_size)
    
    config = {
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_attention_heads': 4,
        'vocab_size': vocab_size,
    }
    
    core = InferenceCore(weights, config, device='cpu')
    print(f"✅ InferenceCore created: {core}")
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (2, 10))
    attention_mask = torch.ones(2, 10)
    
    with torch.no_grad():
        embeddings = core(input_ids, attention_mask)
    
    print(f"✅ Forward pass: {input_ids.shape} -> {embeddings.shape}")
    
except Exception as e:
    print(f"❌ InferenceCore failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🎉 All basic tests completed!")
print("=" * 60)

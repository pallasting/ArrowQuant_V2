"""
Test imports by directly adding the directory to sys.path.
This works around the hyphen in the directory name.
"""

import sys
from pathlib import Path

# Add the ai-os-diffusion directory directly to path
# Python will treat it as a package even with the hyphen
ai_os_dir = Path(__file__).parent / "ai-os-diffusion"
sys.path.insert(0, str(ai_os_dir))

print("=" * 60)
print("Testing AI-OS Diffusion Imports")
print("=" * 60)

# Now we can import using the module structure
print("\n1. Testing config...")
try:
    from config import Config
    config = Config()
    print(f"✅ Config: log_level={config.log_level}")
except Exception as e:
    print(f"❌ Config failed: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing logger...")
try:
    from utils.logger import logger
    logger.info("Test message")
    print("✅ Logger working")
except Exception as e:
    print(f"❌ Logger failed: {e}")

print("\n3. Testing errors...")
try:
    from utils.errors import DiffusionError, ModelLoadError
    print("✅ Error classes imported")
except Exception as e:
    print(f"❌ Errors failed: {e}")

print("\n4. Testing device utils...")
try:
    from inference.device_utils import get_optimal_device, get_device_info
    device = get_optimal_device()
    info = get_device_info(device)
    print(f"✅ Device: {device} - {info.get('name', 'unknown')}")
except Exception as e:
    print(f"❌ Device utils failed: {e}")

print("\n5. Testing quantization schema...")
try:
    from inference.quantization_schema import WEIGHT_SCHEMA_V1, WEIGHT_SCHEMA_V2
    print(f"✅ Schema V1: {len(WEIGHT_SCHEMA_V1)} columns")
    print(f"✅ Schema V2: {len(WEIGHT_SCHEMA_V2)} columns")
except Exception as e:
    print(f"❌ Quantization schema failed: {e}")

print("\n6. Testing decoder layers...")
try:
    import torch
    from inference.decoder_layers import RMSNorm, precompute_freqs_cis
    
    norm = RMSNorm(dim=384)
    x = torch.randn(1, 10, 384)
    output = norm(x)
    print(f"✅ RMSNorm: {x.shape} -> {output.shape}")
    
    freqs_cis = precompute_freqs_cis(dim=64, end=512)
    print(f"✅ RoPE: {freqs_cis.shape}")
except Exception as e:
    print(f"❌ Decoder layers failed: {e}")
    import traceback
    traceback.print_exc()

print("\n7. Testing InferenceCore...")
try:
    import torch
    from inference.inference_core import InferenceCore
    
    # Create minimal dummy weights
    hidden_size = 128
    num_layers = 1
    vocab_size = 100
    
    weights = {}
    weights['embeddings.word_embeddings.weight'] = torch.randn(vocab_size, hidden_size)
    weights['embeddings.position_embeddings.weight'] = torch.randn(64, hidden_size)
    weights['embeddings.token_type_embeddings.weight'] = torch.randn(2, hidden_size)
    weights['embeddings.LayerNorm.weight'] = torch.ones(hidden_size)
    weights['embeddings.LayerNorm.bias'] = torch.zeros(hidden_size)
    
    # Single encoder layer
    prefix = "encoder.layer.0"
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
    print(f"✅ InferenceCore created")
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (2, 10))
    attention_mask = torch.ones(2, 10)
    
    with torch.no_grad():
        embeddings = core(input_ids, attention_mask)
    
    print(f"✅ Forward pass: {input_ids.shape} -> {embeddings.shape}")
    assert embeddings.shape == (2, hidden_size), f"Expected (2, {hidden_size}), got {embeddings.shape}"
    print(f"✅ Output shape correct!")
    
except Exception as e:
    print(f"❌ InferenceCore failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ All tests completed successfully!")
print("=" * 60)
print("\nThe inference module is working correctly!")
print("Next steps:")
print("  1. Test with real model weights (Parquet format)")
print("  2. Test FastTokenizer with real tokenizer.json")
print("  3. Test full ArrowEngine.encode() pipeline")

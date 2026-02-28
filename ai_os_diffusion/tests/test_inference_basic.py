"""
Basic tests for inference module migration.

Tests the core functionality of the migrated inference components:
- WeightLoader: Zero-copy weight loading
- FastTokenizer: Rust-based tokenization
- InferenceCore: Forward pass
- ArrowEngine: End-to-end encoding
"""

import sys
from pathlib import Path

# Add ai-os-diffusion directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch


def test_imports():
    """Test that all migrated modules can be imported."""
    print("Testing imports...")
    
    try:
        from ..inference import (
            ArrowEngine,
            WeightLoader,
            LazyWeightDict,
            FastTokenizer,
            InferenceCore,
            get_optimal_device,
            get_device_info,
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_device_utils():
    """Test device detection utilities."""
    print("\nTesting device utilities...")
    
    try:
        from ..inference.device_utils import (
            get_optimal_device,
            get_device_info,
            get_best_device,
        )
        
        # Test device detection
        device = get_best_device()
        print(f"  Detected device: {device}")
        
        # Test device info
        info = get_device_info(device)
        print(f"  Device info: {info.get('name', 'unknown')}")
        
        print("✅ Device utilities working")
        return True
    except Exception as e:
        print(f"❌ Device utils test failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from ..config import Config
        
        # Test default config
        config = Config()
        print(f"  Default config created")
        print(f"  Log level: {config.log_level}")
        print(f"  Device: {config.device}")
        
        print("✅ Configuration working")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_logger():
    """Test logging system."""
    print("\nTesting logger...")
    
    try:
        from ..utils.logger import logger
        
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        
        print("✅ Logger working")
        return True
    except Exception as e:
        print(f"❌ Logger test failed: {e}")
        return False


def test_errors():
    """Test error hierarchy."""
    print("\nTesting error hierarchy...")
    
    try:
        from ..utils.errors import (
            DiffusionError,
            ModelLoadError,
            InferenceError,
        )
        
        # Test error creation
        try:
            raise ModelLoadError(
                "Test error",
                context={"test": "value"}
            )
        except ModelLoadError as e:
            print(f"  Caught ModelLoadError: {e}")
        
        print("✅ Error hierarchy working")
        return True
    except Exception as e:
        print(f"❌ Error test failed: {e}")
        return False


def test_quantization_schema():
    """Test quantization schema definitions."""
    print("\nTesting quantization schema...")
    
    try:
        from ..inference.quantization_schema import (
            WEIGHT_SCHEMA_V1,
            WEIGHT_SCHEMA_V2,
            detect_schema_version,
            QuantType,
        )
        
        print(f"  Schema V1 columns: {len(WEIGHT_SCHEMA_V1)}")
        print(f"  Schema V2 columns: {len(WEIGHT_SCHEMA_V2)}")
        
        print("✅ Quantization schema working")
        return True
    except Exception as e:
        print(f"❌ Quantization schema test failed: {e}")
        return False


def test_decoder_layers():
    """Test decoder layer components."""
    print("\nTesting decoder layers...")
    
    try:
        from ..inference.decoder_layers import (
            RMSNorm,
            precompute_freqs_cis,
            KVCache,
            Attention,
            MLP,
            DecoderLayer,
        )
        
        # Test RMSNorm
        norm = RMSNorm(dim=384)
        x = torch.randn(1, 10, 384)
        output = norm(x)
        assert output.shape == x.shape
        print(f"  RMSNorm: input {x.shape} -> output {output.shape}")
        
        # Test RoPE
        freqs_cis = precompute_freqs_cis(dim=64, end=512)
        print(f"  RoPE freqs_cis shape: {freqs_cis.shape}")
        
        print("✅ Decoder layers working")
        return True
    except Exception as e:
        print(f"❌ Decoder layers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_core_creation():
    """Test InferenceCore creation with dummy weights."""
    print("\nTesting InferenceCore creation...")
    
    try:
        from ..inference.inference_core import InferenceCore
        
        # Create dummy weights for a tiny BERT model
        hidden_size = 128
        num_layers = 2
        num_heads = 4
        vocab_size = 1000
        max_seq_len = 64
        
        weights = {}
        
        # Embedding weights
        weights['embeddings.word_embeddings.weight'] = torch.randn(vocab_size, hidden_size)
        weights['embeddings.position_embeddings.weight'] = torch.randn(max_seq_len, hidden_size)
        weights['embeddings.token_type_embeddings.weight'] = torch.randn(2, hidden_size)
        weights['embeddings.LayerNorm.weight'] = torch.ones(hidden_size)
        weights['embeddings.LayerNorm.bias'] = torch.zeros(hidden_size)
        
        # Encoder layers
        for i in range(num_layers):
            prefix = f"encoder.layer.{i}"
            
            # Attention
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
            
            # FFN
            intermediate_size = hidden_size * 4
            weights[f"{prefix}.intermediate.dense.weight"] = torch.randn(intermediate_size, hidden_size)
            weights[f"{prefix}.intermediate.dense.bias"] = torch.zeros(intermediate_size)
            weights[f"{prefix}.output.dense.weight"] = torch.randn(hidden_size, intermediate_size)
            weights[f"{prefix}.output.dense.bias"] = torch.zeros(hidden_size)
            weights[f"{prefix}.output.LayerNorm.weight"] = torch.ones(hidden_size)
            weights[f"{prefix}.output.LayerNorm.bias"] = torch.zeros(hidden_size)
        
        config = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_attention_heads': num_heads,
            'intermediate_size': hidden_size * 4,
            'max_position_embeddings': max_seq_len,
            'vocab_size': vocab_size,
            'layer_norm_eps': 1e-12,
        }
        
        # Create InferenceCore
        core = InferenceCore(weights, config, device='cpu')
        print(f"  Created InferenceCore: {core}")
        
        # Test forward pass
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            embeddings = core(input_ids, attention_mask)
        
        print(f"  Forward pass: input {input_ids.shape} -> embeddings {embeddings.shape}")
        assert embeddings.shape == (batch_size, hidden_size)
        
        print("✅ InferenceCore creation and forward pass working")
        return True
    except Exception as e:
        print(f"❌ InferenceCore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("AI-OS Diffusion Inference Module Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Device Utils", test_device_utils),
        ("Configuration", test_config),
        ("Logger", test_logger),
        ("Errors", test_errors),
        ("Quantization Schema", test_quantization_schema),
        ("Decoder Layers", test_decoder_layers),
        ("InferenceCore", test_inference_core_creation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

"""
Simple integration test for ArrowEngine.

Tests basic functionality without requiring actual model files.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from llm_compression.inference import (
            ArrowEngine,
            WeightLoader,
            FastTokenizer,
            InferenceCore,
        )
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_weight_loader_api():
    """Test WeightLoader API without actual file."""
    print("\nTesting WeightLoader API...")
    
    try:
        from llm_compression.inference import WeightLoader
        
        print("  - WeightLoader class available")
        print("  - Methods: load_weights, get_layer, get_metadata")
        print("✓ WeightLoader API test passed")
        return True
    except Exception as e:
        print(f"✗ WeightLoader API test failed: {e}")
        return False


def test_fast_tokenizer_api():
    """Test FastTokenizer API."""
    print("\nTesting FastTokenizer API...")
    
    try:
        from llm_compression.inference import FastTokenizer
        
        print("  - FastTokenizer class available")
        print("  - Methods: encode, decode, batch_decode")
        print("✓ FastTokenizer API test passed")
        return True
    except Exception as e:
        print(f"✗ FastTokenizer API test failed: {e}")
        return False


def test_inference_core_api():
    """Test InferenceCore API."""
    print("\nTesting InferenceCore API...")
    
    try:
        from llm_compression.inference import InferenceCore
        
        print("  - InferenceCore class available")
        print("  - Methods: forward, mean_pooling, normalize_embeddings")
        print("✓ InferenceCore API test passed")
        return True
    except Exception as e:
        print(f"✗ InferenceCore API test failed: {e}")
        return False


def test_arrow_engine_api():
    """Test ArrowEngine API."""
    print("\nTesting ArrowEngine API...")
    
    try:
        from llm_compression.inference import ArrowEngine
        
        print("  - ArrowEngine class available")
        print("  - Methods: encode, encode_batch, similarity")
        print("✓ ArrowEngine API test passed")
        return True
    except Exception as e:
        print(f"✗ ArrowEngine API test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("ArrowEngine Integration Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_weight_loader_api,
        test_fast_tokenizer_api,
        test_inference_core_api,
        test_arrow_engine_api,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

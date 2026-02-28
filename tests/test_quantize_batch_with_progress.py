"""
Test quantize_batch_with_progress API with progress callback support.

This test verifies that:
1. Progress callbacks are invoked for each layer
2. Progress values increase monotonically from 0.0 to 1.0
3. Callback errors are handled gracefully without failing quantization
4. Results are identical to quantize_batch (no functional difference)
5. Progress reporting works correctly with parallel processing

Validates: Requirement 2.1 - Progress callback support for batch operations
"""

import numpy as np
import pytest


def test_quantize_batch_with_progress_basic():
    """Test basic batch quantization with progress callback."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Track progress calls
    progress_calls = []
    
    def progress_callback(layer_name: str, progress: float):
        progress_calls.append((layer_name, progress))
    
    # Create test data with multiple layers
    weights_dict = {
        "layer.0.weight": np.random.randn(1000).astype(np.float32),
        "layer.1.weight": np.random.randn(2000).astype(np.float32),
        "layer.2.weight": np.random.randn(1500).astype(np.float32),
    }
    
    # Quantize batch with progress
    results = quantizer.quantize_batch_with_progress(
        weights_dict, 
        bit_width=4,
        progress_callback=progress_callback
    )
    
    # Verify all layers processed
    assert len(results) == 3
    assert "layer.0.weight" in results
    assert "layer.1.weight" in results
    assert "layer.2.weight" in results
    
    # Verify progress callback was called for each layer
    assert len(progress_calls) == 3
    
    # Verify progress values are between 0 and 1
    for layer_name, progress in progress_calls:
        assert 0.0 <= progress <= 1.0
        assert layer_name in weights_dict
    
    # Verify final progress is 1.0
    final_progress = max(p for _, p in progress_calls)
    assert final_progress == 1.0


def test_quantize_batch_with_progress_monotonic():
    """Test that progress values increase monotonically."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Track progress values
    progress_values = []
    
    def progress_callback(layer_name: str, progress: float):
        progress_values.append(progress)
    
    # Create test data with many layers
    np.random.seed(42)
    weights_dict = {
        f"layer.{i}.weight": np.random.randn(500).astype(np.float32)
        for i in range(10)
    }
    
    # Quantize batch with progress
    results = quantizer.quantize_batch_with_progress(
        weights_dict,
        bit_width=4,
        progress_callback=progress_callback
    )
    
    # Verify all layers processed
    assert len(results) == 10
    
    # Verify progress values increase monotonically
    assert len(progress_values) == 10
    for i in range(1, len(progress_values)):
        assert progress_values[i] >= progress_values[i-1], \
            f"Progress decreased: {progress_values[i-1]} -> {progress_values[i]}"


def test_quantize_batch_with_progress_no_callback():
    """Test batch quantization without progress callback (should work normally)."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Create test data
    weights_dict = {
        "layer.0.weight": np.random.randn(1000).astype(np.float32),
        "layer.1.weight": np.random.randn(1000).astype(np.float32),
    }
    
    # Quantize without callback (should work fine)
    results = quantizer.quantize_batch_with_progress(
        weights_dict,
        bit_width=4,
        progress_callback=None
    )
    
    # Verify results
    assert len(results) == 2
    assert "layer.0.weight" in results
    assert "layer.1.weight" in results


def test_quantize_batch_with_progress_callback_error():
    """Test that callback errors are handled gracefully without failing quantization."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Create a callback that raises an error
    def faulty_callback(layer_name: str, progress: float):
        raise RuntimeError("Callback error!")
    
    # Create test data
    weights_dict = {
        "layer.0.weight": np.random.randn(1000).astype(np.float32),
        "layer.1.weight": np.random.randn(1000).astype(np.float32),
    }
    
    # Quantization should succeed despite callback errors
    # (errors are logged but don't fail the operation)
    results = quantizer.quantize_batch_with_progress(
        weights_dict,
        bit_width=4,
        progress_callback=faulty_callback
    )
    
    # Verify quantization succeeded
    assert len(results) == 2
    assert "layer.0.weight" in results
    assert "layer.1.weight" in results
    
    # Verify results are valid
    for layer_name, result in results.items():
        assert "quantized_data" in result
        assert len(result["quantized_data"]) > 0


def test_quantize_batch_with_progress_vs_without():
    """Test that results are identical with and without progress callback."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Create test data
    np.random.seed(42)
    weights_dict = {
        "layer.0.weight": np.random.randn(1000).astype(np.float32),
        "layer.1.weight": np.random.randn(1000).astype(np.float32),
        "layer.2.weight": np.random.randn(1000).astype(np.float32),
    }
    
    # Quantize without progress callback
    results_without = quantizer.quantize_batch(weights_dict, bit_width=4)
    
    # Quantize with progress callback
    def progress_callback(layer_name: str, progress: float):
        pass  # Do nothing
    
    results_with = quantizer.quantize_batch_with_progress(
        weights_dict,
        bit_width=4,
        progress_callback=progress_callback
    )
    
    # Verify results are identical
    assert set(results_without.keys()) == set(results_with.keys())
    
    for layer_name in results_without.keys():
        r_without = results_without[layer_name]
        r_with = results_with[layer_name]
        
        # Compare quantized data
        assert r_without["quantized_data"] == r_with["quantized_data"]
        
        # Compare scales
        scales_without = np.array(r_without["scales"])
        scales_with = np.array(r_with["scales"])
        np.testing.assert_allclose(scales_without, scales_with, rtol=1e-6)


def test_quantize_batch_with_progress_empty():
    """Test batch quantization with empty dictionary and progress callback."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Track progress calls
    progress_calls = []
    
    def progress_callback(layer_name: str, progress: float):
        progress_calls.append((layer_name, progress))
    
    # Empty dictionary should return empty results
    results = quantizer.quantize_batch_with_progress(
        {},
        bit_width=4,
        progress_callback=progress_callback
    )
    
    assert len(results) == 0
    # No progress calls for empty batch
    assert len(progress_calls) == 0


def test_quantize_batch_with_progress_large_scale():
    """Test progress reporting with many layers (stress test)."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Track progress
    progress_calls = []
    
    def progress_callback(layer_name: str, progress: float):
        progress_calls.append((layer_name, progress))
    
    # Create 50 layers
    np.random.seed(42)
    weights_dict = {
        f"layer.{i}.weight": np.random.randn(500).astype(np.float32)
        for i in range(50)
    }
    
    # Quantize batch with progress
    results = quantizer.quantize_batch_with_progress(
        weights_dict,
        bit_width=4,
        progress_callback=progress_callback
    )
    
    # Verify all layers processed
    assert len(results) == 50
    
    # Verify progress callback called for each layer
    assert len(progress_calls) == 50
    
    # Verify progress reaches 1.0
    final_progress = max(p for _, p in progress_calls)
    assert final_progress == 1.0
    
    # Verify all layer names are present
    reported_layers = {name for name, _ in progress_calls}
    assert reported_layers == set(weights_dict.keys())


def test_quantize_batch_with_progress_invalid_bit_width():
    """Test that invalid bit width raises error even with progress callback."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    def progress_callback(layer_name: str, progress: float):
        pass
    
    weights_dict = {
        "layer.0.weight": np.random.randn(1000).astype(np.float32),
    }
    
    # Invalid bit width should raise error
    with pytest.raises(ValueError, match="Invalid bit_width"):
        quantizer.quantize_batch_with_progress(
            weights_dict,
            bit_width=3,
            progress_callback=progress_callback
        )


def test_quantize_batch_with_progress_callback_signature():
    """Test that callback receives correct arguments."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Track callback arguments
    callback_args = []
    
    def progress_callback(layer_name: str, progress: float):
        # Verify argument types
        assert isinstance(layer_name, str)
        assert isinstance(progress, float)
        callback_args.append((layer_name, progress))
    
    # Create test data
    weights_dict = {
        "layer.0.weight": np.random.randn(1000).astype(np.float32),
        "layer.1.weight": np.random.randn(1000).astype(np.float32),
    }
    
    # Quantize with progress
    results = quantizer.quantize_batch_with_progress(
        weights_dict,
        bit_width=4,
        progress_callback=progress_callback
    )
    
    # Verify callback was called with correct arguments
    assert len(callback_args) == 2
    
    for layer_name, progress in callback_args:
        assert layer_name in weights_dict
        assert 0.0 <= progress <= 1.0


def test_quantize_batch_with_progress_partial_callback():
    """Test callback that only processes some calls (e.g., for UI updates)."""
    from arrow_quant_v2 import ArrowQuantV2
    
    quantizer = ArrowQuantV2("diffusion")
    
    # Callback that only logs every 5th layer
    logged_layers = []
    
    def selective_callback(layer_name: str, progress: float):
        if len(logged_layers) % 5 == 0:
            logged_layers.append(layer_name)
    
    # Create test data with 20 layers
    np.random.seed(42)
    weights_dict = {
        f"layer.{i}.weight": np.random.randn(500).astype(np.float32)
        for i in range(20)
    }
    
    # Quantize with selective callback
    results = quantizer.quantize_batch_with_progress(
        weights_dict,
        bit_width=4,
        progress_callback=selective_callback
    )
    
    # Verify all layers processed
    assert len(results) == 20
    
    # Verify callback was selective (not all layers logged)
    assert len(logged_layers) < 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

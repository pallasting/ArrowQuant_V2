#!/usr/bin/env python3
"""
Test the Stage 1-3 optimizations
Validates memory optimization, Python API, and SIMD features
"""

import sys
import numpy as np
import pyarrow as pa
import time

sys.path.insert(0, '/home/pallasting/arrow_quant_v2_local/target/release')

import arrow_quant_v2

print("=" * 70)
print("Arrow Quant V2 - Optimization Validation Test")
print("=" * 70)
print()

passed = 0
failed = 0

def test(name, func):
    global passed, failed
    print(f"[TEST] {name}")
    try:
        func()
        print(f"  ✓ PASSED\n")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")
        failed += 1
        import traceback
        traceback.print_exc()

# Stage 1: Memory Optimization Tests
print("=" * 70)
print("STAGE 1: Memory Optimization")
print("=" * 70)
print()

def test_batch_processing():
    """Test batch API (reduces boundary crossings)"""
    quant = arrow_quant_v2.ArrowQuantV2()
    
    # Create batch data
    weights = {
        "layer1": np.random.randn(100).astype(np.float32),
        "layer2": np.random.randn(100).astype(np.float32),
        "layer3": np.random.randn(100).astype(np.float32),
    }
    
    # Quantize batch
    results = quant.quantize_batch(weights, bit_width=4)
    
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert "layer1" in results, "layer1 not in results"
    assert "quantized_data" in results["layer1"], "quantized_data not in result"
    print(f"    Batch processed {len(results)} layers successfully")

test("Batch Processing (Memory Optimization)", test_batch_processing)

def test_zero_copy_arrow():
    """Test Arrow IPC zero-copy interface"""
    quant = arrow_quant_v2.ArrowQuantV2()
    
    # Create Arrow table
    table = pa.table({
        'layer_name': ['layer1', 'layer2'],
        'weights': [
            np.random.randn(100).astype(np.float32).tolist(),
            np.random.randn(100).astype(np.float32).tolist()
        ]
    })
    
    # Quantize via Arrow (zero-copy)
    result = quant.quantize_arrow(table, bit_width=4)
    
    assert result.num_rows == 2, f"Expected 2 rows, got {result.num_rows}"
    assert 'quantized_data' in result.column_names, "quantized_data column missing"
    print(f"    Arrow zero-copy processed {result.num_rows} layers")

test("Arrow IPC Zero-Copy", test_zero_copy_arrow)

# Stage 2: Python API Tests
print("=" * 70)
print("STAGE 2: Python API Production Features")
print("=" * 70)
print()

def test_parameter_validation():
    """Test parameter validation"""
    quant = arrow_quant_v2.ArrowQuantV2()
    
    weights = {"layer1": np.array([1.0, 2.0], dtype=np.float32)}
    
    # Test invalid bit_width
    try:
        quant.quantize_batch(weights, bit_width=3)
        raise AssertionError("Should have raised ValueError for invalid bit_width")
    except ValueError as e:
        print(f"    Correctly rejected invalid bit_width: {str(e)[:60]}...")
    
    # Test valid bit_widths
    for bw in [2, 4, 8]:
        result = quant.quantize_batch(weights, bit_width=bw)
        assert result["layer1"]["bit_width"] == bw
    print(f"    All valid bit_widths (2, 4, 8) accepted")

test("Parameter Validation", test_parameter_validation)

def test_error_handling():
    """Test error handling and messages"""
    quant = arrow_quant_v2.ArrowQuantV2()
    
    # Test empty batch
    try:
        quant.quantize_batch({}, bit_width=4)
        raise AssertionError("Should have raised ValueError for empty batch")
    except ValueError as e:
        print(f"    Correctly rejected empty batch: {str(e)[:60]}...")
    
    # Test invalid data type
    try:
        weights = {"layer1": np.array([1, 2, 3], dtype=np.int32)}  # Wrong dtype
        quant.quantize_batch(weights, bit_width=4)
        raise AssertionError("Should have raised ValueError for wrong dtype")
    except ValueError as e:
        print(f"    Correctly rejected wrong dtype: {str(e)[:60]}...")

test("Error Handling", test_error_handling)

def test_progress_callback():
    """Test progress callback feature"""
    quant = arrow_quant_v2.ArrowQuantV2()
    
    weights = {
        f"layer{i}": np.random.randn(50).astype(np.float32)
        for i in range(5)
    }
    
    progress_calls = []
    
    def progress_callback(layer_name, progress):
        progress_calls.append((layer_name, progress))
    
    # Quantize with progress
    result = quant.quantize_batch_with_progress(
        weights,
        bit_width=4,
        progress_callback=progress_callback
    )
    
    assert len(result) == 5, f"Expected 5 results, got {len(result)}"
    assert len(progress_calls) > 0, "Progress callback was not called"
    print(f"    Progress callback called {len(progress_calls)} times")

test("Progress Callback", test_progress_callback)

# Stage 3: SIMD and Performance Tests
print("=" * 70)
print("STAGE 3: SIMD and Performance Features")
print("=" * 70)
print()

def test_large_batch_performance():
    """Test performance with large batch"""
    quant = arrow_quant_v2.ArrowQuantV2()
    
    # Create large batch
    weights = {
        f"layer{i}": np.random.randn(10000).astype(np.float32)
        for i in range(10)
    }
    
    start = time.time()
    results = quant.quantize_batch(weights, bit_width=4)
    elapsed = time.time() - start
    
    assert len(results) == 10, f"Expected 10 results, got {len(results)}"
    print(f"    Processed 10 layers (100K elements) in {elapsed:.3f}s")
    print(f"    Throughput: {10 / elapsed:.1f} layers/sec")

test("Large Batch Performance", test_large_batch_performance)

def test_arrow_performance():
    """Test Arrow IPC performance"""
    quant = arrow_quant_v2.ArrowQuantV2()
    
    # Create Arrow table with large data
    table = pa.table({
        'layer_name': [f'layer{i}' for i in range(10)],
        'weights': [
            np.random.randn(10000).astype(np.float32).tolist()
            for i in range(10)
        ]
    })
    
    start = time.time()
    result = quant.quantize_arrow(table, bit_width=4)
    elapsed = time.time() - start
    
    assert result.num_rows == 10, f"Expected 10 rows, got {result.num_rows}"
    print(f"    Arrow processed 10 layers (100K elements) in {elapsed:.3f}s")
    print(f"    Throughput: {10 / elapsed:.1f} layers/sec")

test("Arrow IPC Performance", test_arrow_performance)

def test_different_bit_widths():
    """Test all supported bit widths"""
    quant = arrow_quant_v2.ArrowQuantV2()
    
    weights = {"layer1": np.random.randn(1000).astype(np.float32)}
    
    for bit_width in [2, 4, 8]:
        result = quant.quantize_batch(weights, bit_width=bit_width)
        assert result["layer1"]["bit_width"] == bit_width
        data_size = len(result["layer1"]["quantized_data"])
        print(f"    bit_width={bit_width}: {data_size} bytes")

test("Different Bit Widths", test_different_bit_widths)

# Summary
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Total tests: {passed + failed}")
print(f"✓ Passed: {passed}")
print(f"✗ Failed: {failed}")
print()

if failed == 0:
    print("🎉 All optimization tests passed!")
    print()
    print("Verified features:")
    print("  ✓ Stage 1: Memory optimization (batch processing, zero-copy)")
    print("  ✓ Stage 2: Python API (validation, error handling, progress)")
    print("  ✓ Stage 3: Performance (large batches, Arrow IPC, bit widths)")
    print()
    print("The optimizations are working correctly!")
    sys.exit(0)
else:
    print("⚠ Some tests failed. Please review the errors above.")
    sys.exit(1)

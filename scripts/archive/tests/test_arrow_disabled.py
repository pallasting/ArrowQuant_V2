#!/usr/bin/env python3
"""Test that Arrow IPC is properly disabled"""

import sys
import numpy as np
import pyarrow as pa

sys.path.insert(0, '/home/pallasting/arrow_quant_v2_local/target/release')

import arrow_quant_v2

print("=" * 60)
print("Testing Arrow IPC Disabled Status")
print("=" * 60)
print()

# Test 1: Verify Arrow methods are disabled
print("[1/2] Testing that Arrow methods are disabled...")
try:
    quant = arrow_quant_v2.ArrowQuantV2()
    
    table = pa.table({
        'layer_name': ['layer1'],
        'weights': [[1.0, 2.0, 3.0]]
    })
    
    try:
        result = quant.quantize_arrow(table, bit_width=4)
        print("  ✗ Arrow method should be disabled!")
        sys.exit(1)
    except NotImplementedError as e:
        print(f"  ✓ Arrow method properly disabled")
        print(f"  ✓ Error message: {str(e)[:80]}...")
except Exception as e:
    print(f"  ✗ Unexpected error: {e}")
    sys.exit(1)

# Test 2: Verify batch API still works
print("\n[2/2] Testing that batch API still works...")
try:
    quant = arrow_quant_v2.ArrowQuantV2()
    
    weights = {
        "layer1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "layer2": np.array([4.0, 5.0, 6.0], dtype=np.float32),
    }
    
    results = quant.quantize_batch(weights, bit_width=4)
    
    assert len(results) == 2
    print(f"  ✓ Batch API works correctly")
    print(f"  ✓ Processed {len(results)} layers")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("✓ Arrow IPC properly disabled, batch API working!")
print("=" * 60)
print()
print("Status:")
print("  ✓ Arrow methods disabled (no crashes)")
print("  ✓ Batch API fully functional")
print("  ✓ Users can use quantize_batch() instead")

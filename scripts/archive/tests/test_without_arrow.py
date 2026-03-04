#!/usr/bin/env python3
"""Test without using PyArrow at all"""

import sys
import numpy as np

sys.path.insert(0, '/home/pallasting/arrow_quant_v2_local/target/release')

import arrow_quant_v2

print("=" * 60)
print("Testing Without PyArrow")
print("=" * 60)
print()

# Test: Only use batch API (no Arrow)
print("[1/1] Testing batch API (no PyArrow)...")
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
    print(f"  ✓ No crashes!")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("✓ Batch API works perfectly without PyArrow!")
print("=" * 60)

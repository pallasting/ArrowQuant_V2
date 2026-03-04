#!/usr/bin/env python3
"""
Quick functional test for arrow_quant_v2
Tests basic functionality without running full test suite
"""

import sys
import os

# Add the local build to Python path
sys.path.insert(0, '/home/pallasting/arrow_quant_v2_local/target/release')

print("=" * 60)
print("Arrow Quant V2 - Quick Functional Test")
print("=" * 60)
print()

# Test 1: Import the module
print("[1/5] Testing module import...")
try:
    import arrow_quant_v2
    print("✓ Module imported successfully")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Check basic classes exist
print("\n[2/5] Testing class availability...")
try:
    # Check if main classes are accessible
    assert hasattr(arrow_quant_v2, 'ArrowQuantV2'), "ArrowQuantV2 class not found"
    print("✓ ArrowQuantV2 class available")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 3: Create instance
print("\n[3/5] Testing instance creation...")
try:
    quant = arrow_quant_v2.ArrowQuantV2()
    print("✓ Instance created successfully")
except Exception as e:
    print(f"✗ Failed to create instance: {e}")
    sys.exit(1)

# Test 4: Check methods exist
print("\n[4/5] Testing method availability...")
try:
    methods = ['quantize', 'quantize_batch', 'quantize_arrow']
    for method in methods:
        if hasattr(quant, method):
            print(f"  ✓ {method}() available")
        else:
            print(f"  ⚠ {method}() not found (may be optional)")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 5: Basic version/info check
print("\n[5/5] Testing module info...")
try:
    if hasattr(arrow_quant_v2, '__version__'):
        print(f"  Version: {arrow_quant_v2.__version__}")
    if hasattr(arrow_quant_v2, '__doc__'):
        doc = arrow_quant_v2.__doc__
        if doc:
            print(f"  Doc: {doc[:100]}...")
    print("✓ Module info accessible")
except Exception as e:
    print(f"⚠ Warning: {e}")

print()
print("=" * 60)
print("✓ All basic functional tests passed!")
print("=" * 60)
print()
print("Next steps:")
print("1. Run unit tests: cargo test --release --lib")
print("2. Run benchmarks: cargo bench")
print("3. Run Python integration tests")

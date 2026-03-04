#!/usr/bin/env python3
"""Inspect the actual API of arrow_quant_v2"""

import sys
import inspect

sys.path.insert(0, '/home/pallasting/arrow_quant_v2_local/target/release')

import arrow_quant_v2

print("=" * 60)
print("Arrow Quant V2 - API Inspection")
print("=" * 60)
print()

# Get all public methods
quant = arrow_quant_v2.ArrowQuantV2()
methods = [m for m in dir(quant) if not m.startswith('_')]

print(f"Available methods ({len(methods)}):")
for method in sorted(methods):
    print(f"  - {method}")

print()
print("=" * 60)
print("Method Signatures:")
print("=" * 60)

for method_name in ['quantize', 'quantize_batch', 'quantize_arrow']:
    if hasattr(quant, method_name):
        method = getattr(quant, method_name)
        print(f"\n{method_name}:")
        print(f"  {method.__doc__}")
        try:
            sig = inspect.signature(method)
            print(f"  Signature: {sig}")
        except:
            print("  (Signature not available for Rust methods)")

#!/usr/bin/env python3
"""Check what methods are actually exported from ArrowQuantV2"""

import arrow_quant_v2

print("Module members:")
for name in sorted(dir(arrow_quant_v2)):
    if not name.startswith('_'):
        print(f"  {name}")

print("\nArrowQuantV2 class methods:")
q = arrow_quant_v2.ArrowQuantV2()
methods = [name for name in dir(q) if not name.startswith('_')]
for name in sorted(methods):
    print(f"  {name}")

print(f"\nTotal methods: {len(methods)}")

# Check for specific methods we expect
expected = [
    'new',
    'quantize',
    'quantize_diffusion_model',
    'quantize_from_safetensors',
    'validate_quality',
    'get_markov_metrics',
    'simple_test',
    'test_method',
    'quantize_arrow',
    'quantize_arrow_batch',
    'quantize_batch',
    'quantize_batch_with_progress',
]

print("\nExpected methods status:")
for method in expected:
    status = "✓" if hasattr(q, method) else "✗"
    print(f"  {status} {method}")

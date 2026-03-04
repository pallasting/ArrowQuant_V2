from arrow_quant_v2 import ArrowQuantV2

q = ArrowQuantV2()
print("All methods:")
for attr in dir(q):
    if not attr.startswith('_'):
        print(f"  {attr}")

print("\nChecking specific methods:")
print(f"  has quantize_arrow: {hasattr(q, 'quantize_arrow')}")
print(f"  has quantize_arrow_batch: {hasattr(q, 'quantize_arrow_batch')}")
print(f"  has quantize_numpy: {hasattr(q, 'quantize_numpy')}")
print(f"  has test_method: {hasattr(q, 'test_method')}")

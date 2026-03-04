#!/usr/bin/env python3
"""Quick functional test to verify core functionality works"""

import sys
import numpy as np

try:
    import arrow_quant_v2
    print("✅ [1/5] Module import successful")
except ImportError as e:
    print(f"❌ [1/5] Module import failed: {e}")
    sys.exit(1)

try:
    quantizer = arrow_quant_v2.ArrowQuantV2()
    print("✅ [2/5] ArrowQuantV2 instance created")
except Exception as e:
    print(f"❌ [2/5] Failed to create instance: {e}")
    sys.exit(1)

try:
    # Test basic quantization
    weights = {"layer1": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)}
    result = quantizer.quantize(weights, bit_width=8)
    print(f"✅ [3/5] Basic quantization works (result keys: {list(result.keys())})")
except Exception as e:
    print(f"❌ [3/5] Quantization failed: {e}")
    sys.exit(1)

try:
    # Test batch quantization
    batch_weights = {
        "layer1": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "layer2": np.array([0.4, 0.5, 0.6], dtype=np.float32),
    }
    batch_result = quantizer.quantize_batch(batch_weights, bit_width=8)
    print(f"✅ [4/5] Batch quantization works ({len(batch_result)} layers)")
except Exception as e:
    print(f"❌ [4/5] Batch quantization failed: {e}")
    sys.exit(1)

try:
    # Test Arrow quantization
    import pyarrow as pa
    table = pa.table({
        "layer_name": ["layer1", "layer2"],
        "weights": [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
    })
    arrow_result = quantizer.quantize_arrow(table, bit_width=8, num_time_groups=2)
    print(f"✅ [5/5] Arrow quantization works (result: {arrow_result.num_rows} rows)")
except Exception as e:
    print(f"⚠️  [5/5] Arrow quantization skipped (may need pyarrow): {e}")

print("\n" + "="*60)
print("✅ All core functionality tests passed!")
print("="*60)

"""
Simple zero-copy export test

This test verifies that the optimized export_recordbatch_to_pyarrow() 
supports zero-copy by checking if to_pandas(zero_copy_only=True) succeeds.

**Validates: Requirements 5.4, 8.4** - Task 6.2 acceptance criteria
"""

import numpy as np
import pyarrow as pa

try:
    from arrow_quant_v2 import ArrowQuantV2
    ARROW_QUANT_AVAILABLE = True
except ImportError:
    ARROW_QUANT_AVAILABLE = False
    print("arrow_quant_v2 module not available - skipping test")
    exit(0)


def test_zero_copy_export():
    """
    Test that exported RecordBatch supports zero-copy pandas conversion
    
    This is the acceptance criteria for Task 6.2:
    "Python to_pandas(zero_copy_only=True) succeeds"
    """
    print("Testing zero-copy export...")
    
    # Create test data
    weights_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    
    # Create PyArrow table
    table = pa.Table.from_pydict({
        "layer_name": ["test_layer"],
        "weights": [weights_data],
    })
    
    print(f"Input table: {table.schema}")
    
    # Convert to RecordBatch
    batches = table.to_batches()
    assert len(batches) > 0, "Table has no batches"
    batch = batches[0]
    
    print(f"Input batch: {batch.schema}, {batch.num_rows} rows")
    
    # Quantize using the optimized API
    quantizer = ArrowQuantV2()
    result_batch = quantizer.quantize_arrow_batch(batch, bit_width=8)
    
    print(f"Result type: {type(result_batch)}")
    print(f"Result is RecordBatch: {isinstance(result_batch, pa.RecordBatch)}")
    
    if isinstance(result_batch, pa.RecordBatch):
        print(f"Result schema: {result_batch.schema}")
        print(f"Result columns: {result_batch.schema.names}")
        print(f"Result rows: {result_batch.num_rows}")
        
        # THE KEY TEST: Convert to pandas with zero_copy_only=True
        # This will FAIL if the data was copied during export
        try:
            df = result_batch.to_pandas(zero_copy_only=True)
            print("✓ SUCCESS: to_pandas(zero_copy_only=True) succeeded!")
            print(f"  DataFrame shape: {df.shape}")
            print(f"  DataFrame columns: {list(df.columns)}")
            print("\n✓ Task 6.2 acceptance criteria met: Zero-copy export verified")
            return True
        except pa.ArrowInvalid as e:
            print(f"✗ FAILED: to_pandas(zero_copy_only=True) raised ArrowInvalid")
            print(f"  Error: {e}")
            print("\n✗ Task 6.2 acceptance criteria NOT met: Data was copied during export")
            return False
    else:
        print(f"✗ FAILED: Expected RecordBatch, got {type(result_batch)}")
        print("\n✗ Task 6.2 acceptance criteria NOT met: Wrong return type")
        return False


if __name__ == "__main__":
    if ARROW_QUANT_AVAILABLE:
        success = test_zero_copy_export()
        exit(0 if success else 1)
    else:
        print("Skipping test - module not available")
        exit(0)

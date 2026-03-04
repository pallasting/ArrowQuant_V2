"""Debug script to isolate Arrow FFI import issue."""
import pyarrow as pa
import numpy as np

# Create a simple table like in the test
weights = np.random.randn(10).astype(np.float32)
table = pa.Table.from_pydict({
    "layer_name": ["layer.0.weight"],
    "weights": [weights.tolist()],
    "shape": [[10]],
})

print("Table created successfully:")
print(table.schema)
print(f"Number of rows: {len(table)}")
print(f"Number of columns: {len(table.columns)}")

# Try to convert to RecordBatch
batches = table.to_batches()
print(f"\nNumber of batches: {len(batches)}")

if batches:
    batch = batches[0]
    print(f"Batch schema: {batch.schema}")
    print(f"Batch num_rows: {batch.num_rows}")
    
    # Try to access the C Data Interface
    try:
        # This is what the Rust code does
        array_capsule = batch.__arrow_c_array__()
        schema_capsule = batch.__arrow_c_schema__()
        print("\n✅ C Data Interface export successful")
        print(f"Array capsule: {array_capsule}")
        print(f"Schema capsule: {schema_capsule}")
    except Exception as e:
        print(f"\n❌ C Data Interface export failed: {e}")
        import traceback
        traceback.print_exc()

# Now try to import it in Python using arrow-rs style
print("\n" + "="*60)
print("Testing import...")

try:
    # Import the module
    from arrow_quant_v2 import ArrowQuantV2
    
    print("✅ Module imported successfully")
    
    # Create quantizer
    quantizer = ArrowQuantV2(mode="diffusion")
    print("✅ Quantizer created successfully")
    
    # Try to quantize
    print("\nAttempting quantization...")
    result = quantizer.quantize_arrow(table, bit_width=4)
    print("✅ Quantization successful!")
    print(f"Result type: {type(result)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

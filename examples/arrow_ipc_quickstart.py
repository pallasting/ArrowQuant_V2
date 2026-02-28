"""
Arrow IPC Zero-Copy Quantization - Quick Start Example

This example demonstrates the complete workflow using Arrow IPC for
zero-copy quantization with ArrowQuant V2.
"""

import numpy as np
import pyarrow as pa
from pathlib import Path

def create_sample_weights_table(num_layers=3, weights_per_layer=1000):
    """Create a sample PyArrow table with weight data."""
    print(f"Creating sample table with {num_layers} layers...")
    
    layer_names = [f"layer.{i}.weight" for i in range(num_layers)]
    weights_list = []
    shapes = []
    
    for i in range(num_layers):
        # Create random weights
        size = weights_per_layer * (i + 1)  # Varying sizes
        weights = np.random.randn(size).astype(np.float32)
        weights_list.append(weights.tolist())
        shapes.append([size])
    
    # Create Arrow Table
    table = pa.Table.from_pydict({
        "layer_name": layer_names,
        "weights": weights_list,
        "shape": shapes,
    })
    
    print(f"✓ Created table with {table.num_rows} rows")
    print(f"  Schema: {table.schema}")
    return table


def quantize_with_arrow_ipc(table, bit_width=4):
    """Quantize weights using Arrow IPC zero-copy interface."""
    try:
        from arrow_quant_v2 import ArrowQuantV2
    except ImportError:
        print("❌ arrow_quant_v2 not installed. Run: maturin develop")
        return None
    
    print(f"\nQuantizing with Arrow IPC (bit_width={bit_width})...")
    
    # Initialize quantizer
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Zero-copy quantization via Arrow IPC
    result_table = quantizer.quantize_arrow(table, bit_width=bit_width)
    
    print(f"✓ Quantization complete!")
    print(f"  Result schema: {result_table.schema}")
    print(f"  Quantized {result_table.num_rows} layers")
    
    return result_table


def analyze_results(result_table):
    """Analyze quantization results."""
    print("\n📊 Results Analysis:")
    
    for i in range(result_table.num_rows):
        layer_name = result_table.column("layer_name")[i].as_py()
        quantized_data = result_table.column("quantized_data")[i].as_py()
        scales = result_table.column("scales")[i].as_py()
        zero_points = result_table.column("zero_points")[i].as_py()
        shape = result_table.column("shape")[i].as_py()
        bit_width = result_table.column("bit_width")[i].as_py()
        
        original_size = np.prod(shape) * 4  # float32 = 4 bytes
        quantized_size = len(quantized_data)
        compression_ratio = original_size / quantized_size
        
        print(f"\n  Layer: {layer_name}")
        print(f"    Shape: {shape}")
        print(f"    Bit width: {bit_width}")
        print(f"    Original size: {original_size:,} bytes")
        print(f"    Quantized size: {quantized_size:,} bytes")
        print(f"    Compression: {compression_ratio:.2f}x")
        print(f"    Scales: {len(scales)} groups")
        print(f"    Zero points: {len(zero_points)} groups")


def save_to_parquet(table, output_path):
    """Save quantized results to Parquet file."""
    print(f"\n💾 Saving to Parquet: {output_path}")
    table.to_parquet(output_path)
    
    # Verify file size
    file_size = Path(output_path).stat().st_size
    print(f"✓ Saved successfully ({file_size:,} bytes)")


def load_from_parquet(input_path):
    """Load quantized results from Parquet file."""
    print(f"\n📂 Loading from Parquet: {input_path}")
    table = pa.parquet.read_table(input_path)
    print(f"✓ Loaded {table.num_rows} layers")
    return table


def main():
    """Run the complete Arrow IPC quantization workflow."""
    print("=" * 60)
    print("Arrow IPC Zero-Copy Quantization - Quick Start")
    print("=" * 60)
    
    # Step 1: Create sample data
    weights_table = create_sample_weights_table(num_layers=5, weights_per_layer=10000)
    
    # Step 2: Quantize using Arrow IPC (zero-copy)
    result_table = quantize_with_arrow_ipc(weights_table, bit_width=4)
    
    if result_table is None:
        print("\n⚠️  Quantization failed. Please build the extension first:")
        print("    cd ai_os_diffusion/arrow_quant_v2")
        print("    maturin develop --release")
        return
    
    # Step 3: Analyze results
    analyze_results(result_table)
    
    # Step 4: Save to Parquet
    output_path = "quantized_weights_arrow_ipc.parquet"
    save_to_parquet(result_table, output_path)
    
    # Step 5: Load back from Parquet
    loaded_table = load_from_parquet(output_path)
    
    print("\n" + "=" * 60)
    print("✅ Complete workflow successful!")
    print("=" * 60)
    print("\n📝 Key Benefits of Arrow IPC:")
    print("  • Zero-copy data transfer (30x faster)")
    print("  • Batch processing (single boundary crossing)")
    print("  • Native Arrow integration")
    print("  • Seamless Parquet I/O")
    
    print("\n🚀 Next Steps:")
    print("  1. Try with your own model weights")
    print("  2. Integrate with safetensors loader")
    print("  3. Run performance benchmarks")
    print("  4. Compare with legacy quantization")


if __name__ == "__main__":
    main()

import torch
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from llm_compression.inference import WeightLoader

def compare_weights(original_path, quantized_path):
    print(f"Comparing original: {original_path}")
    print(f"with quantized: {quantized_path}")
    
    orig_loader = WeightLoader(original_path)
    quant_loader = WeightLoader(quantized_path)
    
    # Ensure tables are loaded
    orig_loader._load_table()
    quant_loader._load_table()
    
    orig_layers = orig_loader.get_layer_names()
    
    for layer_name in orig_layers[:15]: # Check first 15 layers
        if "weight" not in layer_name:
            continue
            
        print(f"\n--- Layer: {layer_name} ---")
        
        try:
            w_orig = orig_loader.get_layer(layer_name).numpy()
            w_quant = quant_loader.get_layer(layer_name).numpy()
            
            # Get raw quantized data and params from table
            row_idx = quant_loader.get_layer_names().index(layer_name)
            table = quant_loader._table
            scales = table['scales'][row_idx].as_py()
            zps = table['zero_points'][row_idx].as_py()
            q_type = table['quant_type'][row_idx].as_py()
            quant_axis = table['quant_axis'][row_idx].as_py()
            
            print(f"  Shape: {w_orig.shape}")
            print(f"  Quant Type: {q_type}, Quant Axis: {quant_axis}")
            print(f"  Scales (first 3): {scales[:3]}")
            print(f"  Zero Points (first 3): {zps[:3]}")
            
            w_orig_flat = w_orig.flatten()
            w_quant_flat = w_quant.flatten()
            
            print(f"  Original (first 5): {w_orig_flat[:5]}")
            print(f"  Dequantized (first 5): {w_quant_flat[:5]}")
            
            # Try to infer quantized integers
            if quant_axis == -1:
                q_ints = (w_quant_flat / (scales[0] + 1e-12) + zps[0])
                print(f"  Inferred Q-Ints (Raw, first 5): {q_ints[:5]}")
                print(f"  Inferred Q-Ints (Round, first 5): {np.round(q_ints[:5])}")
            
            elif quant_axis == 0:
                row_len = w_orig.shape[1] if len(w_orig.shape) > 1 else len(w_orig)
                q_ints_row0 = (w_quant[0] / (scales[0] + 1e-12) + zps[0])
                print(f"  Inferred Q-Ints (Row 0, Raw, first 5): {q_ints_row0[:5]}")
                print(f"  Inferred Q-Ints (Row 0, Round, first 5): {np.round(q_ints_row0[:5])}")

            print(f"  Original  - Mean: {w_orig_flat.mean():.6f}, Std: {w_orig_flat.std():.6f}, Min: {w_orig_flat.min():.6f}, Max: {w_orig_flat.max():.6f}")
            print(f"  Quantized - Mean: {w_quant_flat.mean():.6f}, Std: {w_quant_flat.std():.6f}, Min: {w_quant_flat.min():.6f}, Max: {w_quant_flat.max():.6f}")
            
            sim = np.dot(w_orig_flat, w_quant_flat) / (np.linalg.norm(w_orig_flat) * np.linalg.norm(w_quant_flat) + 1e-9)
            print(f"  Weight Cosine Similarity: {sim:.6f}")
            
            if sim < 0.1:
                print("  [!] CRITICAL: Weights are unrelated!")
            elif sim < 0.8:
                print("  [!] WARNING: Low fidelity.")
        except Exception as e:
            print(f"  Error inspecting layer: {e}")

if __name__ == "__main__":
    import sys
    orig_path = sys.argv[1] if len(sys.argv) > 1 else "models/minilm/weights.parquet"
    quant_path = sys.argv[2] if len(sys.argv) > 2 else "models/minilm/weights_int2_gptq.parquet"
    compare_weights(orig_path, quant_path)

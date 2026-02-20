
import os
import torch
import numpy as np
from transformers import CLIPVisionModel
from llm_compression.multimodal.vision_encoder import VisionEncoder
from llm_compression.inference.weight_loader import WeightLoader

def debug_layers():
    print("=== Starting Layer-by-Layer Debugging ===")

    model_path = "llm_compression/models/clip-vit-base-patch32"
    hf_model_name = "openai/clip-vit-base-patch32"

    # 1. Load HF Model
    print(f"Loading HF model: {hf_model_name}...")
    hf_model = CLIPVisionModel.from_pretrained(hf_model_name)
    hf_model.eval()

    # 2. Load Arrow Model
    print(f"Loading Arrow model: {model_path}...")
    # Manually load core to access internals
    loader = WeightLoader(f"{model_path}/weights.parquet")
    weights = loader.load_weights()

    # Initialize VisionInferenceCore directly
    from llm_compression.inference.vision_core import VisionInferenceCore
    config = {
        "image_size": 224,
        "patch_size": 32,
        "hidden_size": 768,
        "num_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-5
    }
    arrow_core = VisionInferenceCore(weights, config)
    arrow_core.eval()

    # 3. Create Input
    batch_size = 1
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    print(f"Input shape: {pixel_values.shape}")

    # 4. Compare Embeddings
    print("\n--- Comparing Embeddings ---")

    # HF Embeddings
    with torch.no_grad():
        hf_embeddings = hf_model.vision_model.embeddings(pixel_values)

    # Arrow Embeddings
    # Access patch_embedding and positional embeddings manually
    with torch.no_grad():
        # Replicate arrow_core forward logic part 1
        x = arrow_core.patch_embedding(pixel_values)  # (B, 768, 7, 7)
        x = x.flatten(2).transpose(1, 2) # (B, 49, 768)

        # Add CLS token
        cls_token = arrow_core.class_embedding.expand(batch_size, 1, -1)
        x = torch.cat((cls_token, x), dim=1) # (B, 50, 768)

        # Add positional embedding
        x = x + arrow_core.position_embedding.weight[:x.size(1)]

        # Pre-Layernorm
        arrow_embeddings = arrow_core.pre_layernorm(x)

    compare_tensors("Embeddings (Post-LN)", hf_embeddings, arrow_embeddings)

    # 5. Compare Encoder Layers
    print("\n--- Comparing Encoder Layers ---")

    hf_hidden = hf_embeddings
    arrow_hidden = arrow_embeddings

    for i in range(12):
        print(f"\nLayer {i}:")

        # HF Layer
        with torch.no_grad():
            # HF outputs: (hidden_states,)
            hf_layer_out = hf_model.vision_model.encoder.layers[i](hf_hidden)[0]

        # Arrow Layer
        with torch.no_grad():
            # Arrow layer: self_attn -> layer_norm1 -> mlp -> layer_norm2 (Post-LN?)
            # Or Pre-LN? CLIP is usually Pre-LN.
            # Let's verify Arrow implementation structure vs HF structure.
            arrow_layer_out = arrow_core.encoder_layers[i](arrow_hidden)

        compare_tensors(f"Layer {i} Output", hf_layer_out, arrow_layer_out)

        # Update inputs for next layer
        hf_hidden = hf_layer_out
        arrow_hidden = arrow_layer_out

        # Stop if divergence is too high
        if (hf_hidden - arrow_hidden).abs().max() > 1.0:
            print("🛑 Divergence too high, stopping.")
            break

def compare_tensors(name, t1, t2):
    if t1.shape != t2.shape:
        print(f"{name}: ❌ SHAPE MISMATCH {t1.shape} vs {t2.shape}")
        return

    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    status = "✅" if max_diff < 1e-4 else "❌"
    print(f"{name}: {status} Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")

if __name__ == "__main__":
    debug_layers()

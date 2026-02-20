
import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor
from llm_compression.multimodal.image_processor import ImageProcessor

def debug_preprocessing():
    print("=== Starting Preprocessing Debug ===")

    # 1. Create a dummy image
    # Random RGB image 256x256
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    print("Created dummy image: 256x256 RGB")

    # 2. HuggingFace Processor
    print("\n--- HuggingFace Processor ---")
    hf_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    hf_inputs = hf_processor(images=image, return_tensors="pt")
    hf_pixel_values = hf_inputs['pixel_values']

    # HF config
    print(f"HF Mean: {hf_processor.image_processor.image_mean}")
    print(f"HF Std:  {hf_processor.image_processor.image_std}")
    print(f"HF Size: {hf_processor.image_processor.size}")

    print(f"Output Shape: {hf_pixel_values.shape}")
    print(f"Output Mean: {hf_pixel_values.mean().item():.4f}")
    print(f"Output Std:  {hf_pixel_values.std().item():.4f}")
    print(f"Output Range: [{hf_pixel_values.min().item():.4f}, {hf_pixel_values.max().item():.4f}]")

    # 3. ArrowEngine Processor
    print("\n--- ArrowEngine Processor ---")
    # Assuming ArrowProcessor mimics CLIP default: 224x224
    arrow_processor = ImageProcessor(image_size=224)

    # Process
    # Check if process returns numpy or tensor, handle batch dim
    arrow_output = arrow_processor.preprocess(image)

    # Normalize output to tensor (1, 3, 224, 224)
    if isinstance(arrow_output, np.ndarray):
        arrow_pixel_values = torch.from_numpy(arrow_output)
    else:
        arrow_pixel_values = arrow_output

    if arrow_pixel_values.dim() == 3:
        arrow_pixel_values = arrow_pixel_values.unsqueeze(0)

    # Check shape match before printing stats
    if arrow_pixel_values.shape != hf_pixel_values.shape:
        # Maybe channel last?
        if arrow_pixel_values.shape[-1] == 3:
             arrow_pixel_values = arrow_pixel_values.permute(0, 3, 1, 2)

    print(f"Arrow Mean Config: {getattr(arrow_processor, 'mean', 'Unknown')}")
    print(f"Arrow Std Config:  {getattr(arrow_processor, 'std', 'Unknown')}")

    print(f"Output Shape: {arrow_pixel_values.shape}")
    print(f"Output Mean: {arrow_pixel_values.mean().item():.4f}")
    print(f"Output Std:  {arrow_pixel_values.std().item():.4f}")
    print(f"Output Range: [{arrow_pixel_values.min().item():.4f}, {arrow_pixel_values.max().item():.4f}]")

    # 4. Compare
    print("\n--- Comparison ---")
    if hf_pixel_values.shape != arrow_pixel_values.shape:
        print(f"❌ SHAPE MISMATCH! HF={hf_pixel_values.shape}, Arrow={arrow_pixel_values.shape}")
        return

    diff = (hf_pixel_values - arrow_pixel_values).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max Difference: {max_diff:.6f}")
    print(f"Mean Difference: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("✅ Preprocessing matches! (Diff < 1e-4)")
    elif max_diff < 0.1:
        print("⚠️ Minor difference (Likely resizing interpolation or float precision)")
        # Check center crop area specifically?
    else:
        print("❌ MAJOR MISMATCH in Preprocessing")
        # Check if normalization is applied
        if arrow_pixel_values.max() > 1.0 and hf_pixel_values.max() < 3.0:
             print("   -> Arrow output seems unnormalized (0-255 range?)")
        elif arrow_pixel_values.min() >= 0.0 and hf_pixel_values.min() < 0.0:
             print("   -> Arrow output seems unnormalized (0-1 range?)")

if __name__ == "__main__":
    debug_preprocessing()

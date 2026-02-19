#!/usr/bin/env python3
"""Debug vision encoder pooled output (before projection)."""

import numpy as np
import torch
from PIL import Image

from llm_compression.logger import logger

# Create a simple test image
test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

# Load HuggingFace model
logger.info("Loading HuggingFace CLIP...")
from transformers import CLIPModel, CLIPProcessor
hf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
hf_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
hf_model.eval()

# Encode with HuggingFace (get pooled output before projection)
logger.info("Encoding with HuggingFace...")
pil_image = Image.fromarray(test_image)
inputs = hf_processor(images=pil_image, return_tensors="pt")

with torch.no_grad():
    vision_outputs = hf_model.vision_model(**inputs)
    pooled = vision_outputs.pooler_output  # (1, 768)
    
logger.info(f"HF pooled output shape: {pooled.shape}")
logger.info(f"HF pooled stats: min={pooled.min():.4f}, max={pooled.max():.4f}, mean={pooled.mean():.4f}")

# Now load our VisionInferenceCore directly and check pooled output
logger.info("\nLoading ArrowEngine VisionInferenceCore...")
from llm_compression.inference.weight_loader import WeightLoader
from llm_compression.inference.vision_core import VisionInferenceCore

weight_loader = WeightLoader("D:/ai-models/clip-vit-b32/weights.parquet")
weights = weight_loader.load_weights()

config = {
    "image_size": 224,
    "patch_size": 32,
    "hidden_size": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-5,
    "projection_dim": 512,
}

core = VisionInferenceCore(weights=weights, config=config, device="cpu")

# Preprocess image the same way as HF
from llm_compression.multimodal.image_processor import ImageProcessor
processor = ImageProcessor(image_size=224)
preprocessed = processor.preprocess(test_image)
image_tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)

logger.info(f"Input tensor shape: {image_tensor.shape}")
logger.info(f"Input tensor stats: min={image_tensor.min():.4f}, max={image_tensor.max():.4f}, mean={image_tensor.mean():.4f}")

# Get output from our core
with torch.no_grad():
    arrow_output = core(image_tensor)

logger.info(f"\nArrow output shape: {arrow_output.shape}")
logger.info(f"Arrow output stats: min={arrow_output.min():.4f}, max={arrow_output.max():.4f}, mean={arrow_output.mean():.4f}")

# Compare with HF projected output
with torch.no_grad():
    hf_projected = hf_model.visual_projection(pooled)

logger.info(f"\nHF projected shape: {hf_projected.shape}")
logger.info(f"HF projected stats: min={hf_projected.min():.4f}, max={hf_projected.max():.4f}, mean={hf_projected.mean():.4f}")

# Compute similarity
arrow_np = arrow_output.cpu().numpy()[0]
hf_np = hf_projected.cpu().numpy()[0]

similarity = np.dot(arrow_np, hf_np) / (np.linalg.norm(arrow_np) * np.linalg.norm(hf_np))
logger.info(f"\nCosine similarity: {similarity:.6f}")

logger.info(f"\nFirst 10 dimensions:")
logger.info(f"Arrow: {arrow_np[:10]}")
logger.info(f"HF:    {hf_np[:10]}")

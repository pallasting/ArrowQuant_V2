#!/usr/bin/env python3
"""Debug vision encoder outputs to understand the mismatch."""

import numpy as np
import torch
from PIL import Image

from llm_compression.logger import logger
from llm_compression.multimodal.vision_encoder import VisionEncoder

# Load models
logger.info("Loading ArrowEngine VisionEncoder...")
arrow_encoder = VisionEncoder("D:/ai-models/clip-vit-b32")

logger.info("Loading HuggingFace CLIP...")
from transformers import CLIPModel, CLIPProcessor
hf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
hf_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
hf_model.eval()

# Create a simple test image
test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

# Encode with ArrowEngine
logger.info("Encoding with ArrowEngine...")
arrow_emb = arrow_encoder.encode(test_image, normalize=False)
logger.info(f"Arrow output shape: {arrow_emb.shape}")
logger.info(f"Arrow output stats: min={arrow_emb.min():.4f}, max={arrow_emb.max():.4f}, mean={arrow_emb.mean():.4f}, std={arrow_emb.std():.4f}")
logger.info(f"Arrow L2 norm: {np.linalg.norm(arrow_emb):.4f}")

# Encode with HuggingFace
logger.info("Encoding with HuggingFace...")
pil_image = Image.fromarray(test_image)
inputs = hf_processor(images=pil_image, return_tensors="pt")

with torch.no_grad():
    # Get vision model output (before projection)
    vision_outputs = hf_model.vision_model(**inputs)
    pooled = vision_outputs.pooler_output
    logger.info(f"HF pooled output shape: {pooled.shape}")
    logger.info(f"HF pooled stats: min={pooled.min():.4f}, max={pooled.max():.4f}, mean={pooled.mean():.4f}, std={pooled.std():.4f}")
    
    # Apply projection
    projected = hf_model.visual_projection(pooled)
    logger.info(f"HF projected shape: {projected.shape}")
    logger.info(f"HF projected stats: min={projected.min():.4f}, max={projected.max():.4f}, mean={projected.mean():.4f}, std={projected.std():.4f}")
    logger.info(f"HF L2 norm: {projected.norm():.4f}")

hf_emb = projected.cpu().numpy().astype(np.float32)[0]

# Compare
logger.info("\nComparison:")
logger.info(f"Shape match: {arrow_emb.shape == hf_emb.shape}")
logger.info(f"Cosine similarity (unnormalized): {np.dot(arrow_emb, hf_emb) / (np.linalg.norm(arrow_emb) * np.linalg.norm(hf_emb)):.6f}")

# Normalize and compare
arrow_norm = arrow_emb / np.linalg.norm(arrow_emb)
hf_norm = hf_emb / np.linalg.norm(hf_emb)
logger.info(f"Cosine similarity (normalized): {np.dot(arrow_norm, hf_norm):.6f}")

# Check first few dimensions
logger.info(f"\nFirst 10 dimensions:")
logger.info(f"Arrow: {arrow_emb[:10]}")
logger.info(f"HF:    {hf_emb[:10]}")

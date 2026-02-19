#!/usr/bin/env python3
"""Debug preprocessing differences."""

import numpy as np
import torch
from PIL import Image

from llm_compression.logger import logger

# Create a simple test image
test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
logger.info(f"Test image shape: {test_image.shape}, dtype: {test_image.dtype}")
logger.info(f"Test image stats: min={test_image.min()}, max={test_image.max()}, mean={test_image.mean():.2f}")

# HuggingFace preprocessing
logger.info("\n=== HuggingFace Preprocessing ===")
from transformers import CLIPProcessor
hf_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

pil_image = Image.fromarray(test_image)
hf_inputs = hf_processor(images=pil_image, return_tensors="pt")
hf_tensor = hf_inputs['pixel_values'][0]  # (3, 224, 224)

logger.info(f"HF tensor shape: {hf_tensor.shape}")
logger.info(f"HF tensor stats: min={hf_tensor.min():.4f}, max={hf_tensor.max():.4f}, mean={hf_tensor.mean():.4f}, std={hf_tensor.std():.4f}")

# ArrowEngine preprocessing
logger.info("\n=== ArrowEngine Preprocessing ===")
from llm_compression.multimodal.image_processor import ImageProcessor
arrow_processor = ImageProcessor(image_size=224)

arrow_preprocessed = arrow_processor.preprocess(test_image)
logger.info(f"Arrow preprocessed shape: {arrow_preprocessed.shape}")
logger.info(f"Arrow preprocessed stats: min={arrow_preprocessed.min():.4f}, max={arrow_preprocessed.max():.4f}, mean={arrow_preprocessed.mean():.4f}, std={arrow_preprocessed.std():.4f}")

# Convert to tensor format
arrow_tensor = torch.from_numpy(arrow_preprocessed).permute(2, 0, 1)  # (3, 224, 224)
logger.info(f"Arrow tensor shape: {arrow_tensor.shape}")
logger.info(f"Arrow tensor stats: min={arrow_tensor.min():.4f}, max={arrow_tensor.max():.4f}, mean={arrow_tensor.mean():.4f}, std={arrow_tensor.std():.4f}")

# Check CLIP's normalization parameters
logger.info("\n=== CLIP Normalization Parameters ===")
logger.info(f"HF processor image_mean: {hf_processor.image_processor.image_mean}")
logger.info(f"HF processor image_std: {hf_processor.image_processor.image_std}")

# Manually apply CLIP normalization to our preprocessing
logger.info("\n=== Manual CLIP Normalization ===")
clip_mean = np.array([0.48145466, 0.4578275, 0.40821073])
clip_std = np.array([0.26862954, 0.26130258, 0.27577711])

# Our preprocessing: (H, W, C) float32 in [0, 1]
manual_normalized = (arrow_preprocessed - clip_mean) / clip_std
manual_tensor = torch.from_numpy(manual_normalized).permute(2, 0, 1)

logger.info(f"Manual tensor shape: {manual_tensor.shape}")
logger.info(f"Manual tensor stats: min={manual_tensor.min():.4f}, max={manual_tensor.max():.4f}, mean={manual_tensor.mean():.4f}, std={manual_tensor.std():.4f}")

# Compare
logger.info("\n=== Comparison ===")
logger.info(f"HF vs Arrow difference: {torch.abs(hf_tensor - arrow_tensor).mean():.6f}")
logger.info(f"HF vs Manual difference: {torch.abs(hf_tensor - manual_tensor).mean():.6f}")

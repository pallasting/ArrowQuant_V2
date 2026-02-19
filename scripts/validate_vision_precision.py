#!/usr/bin/env python3
"""
Vision Encoder Precision Validation

Validates ArrowEngine VisionEncoder precision against HuggingFace CLIP.
Computes cosine similarities and verifies average similarity > 0.95.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from llm_compression.logger import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate VisionEncoder precision against HuggingFace CLIP"
    )
    parser.add_argument(
        "--arrow-model",
        type=str,
        default="D:/ai-models/clip-vit-b32",
        help="Path to ArrowEngine vision model"
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model name"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of test images"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Minimum average cosine similarity threshold"
    )
    return parser.parse_args()


def load_arrow_encoder(model_path: str):
    """Load ArrowEngine VisionEncoder."""
    from llm_compression.multimodal.vision_encoder import VisionEncoder
    
    logger.info(f"Loading ArrowEngine VisionEncoder from {model_path}")
    start_time = time.time()
    encoder = VisionEncoder(model_path)
    load_time = time.time() - start_time
    
    logger.info(f"ArrowEngine loaded in {load_time:.3f}s")
    return encoder, load_time


def load_hf_encoder(model_name: str):
    """Load HuggingFace CLIP vision encoder."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise ImportError(
            "transformers not installed. Install with: pip install transformers"
        )
    
    logger.info(f"Loading HuggingFace CLIP from {model_name}")
    start_time = time.time()
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    load_time = time.time() - start_time
    
    model.eval()
    logger.info(f"HuggingFace CLIP loaded in {load_time:.3f}s")
    return model, processor, load_time


def generate_test_images(num_images: int, seed: int = 42) -> np.ndarray:
    """
    Generate diverse test images.
    
    Creates synthetic images with different patterns:
    - Solid colors
    - Gradients
    - Checkerboards
    - Random noise
    
    Args:
        num_images: Number of images to generate
        seed: Random seed for reproducibility
        
    Returns:
        Images array: (num_images, 224, 224, 3) uint8
    """
    np.random.seed(seed)
    images = []
    
    patterns_per_type = num_images // 4
    
    # Pattern 1: Solid colors
    for i in range(patterns_per_type):
        color = np.random.randint(0, 256, 3, dtype=np.uint8)
        img = np.full((224, 224, 3), color, dtype=np.uint8)
        images.append(img)
    
    # Pattern 2: Horizontal gradients
    for i in range(patterns_per_type):
        gradient = np.linspace(0, 255, 224, dtype=np.uint8)
        img = np.repeat(gradient[np.newaxis, :, np.newaxis], 224, axis=0)
        img = np.repeat(img, 3, axis=2)
        images.append(img)
    
    # Pattern 3: Checkerboards
    for i in range(patterns_per_type):
        checker_size = np.random.randint(8, 32)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        for y in range(0, 224, checker_size):
            for x in range(0, 224, checker_size):
                if (x // checker_size + y // checker_size) % 2 == 0:
                    img[y:y+checker_size, x:x+checker_size] = 255
        images.append(img)
    
    # Pattern 4: Random noise
    remaining = num_images - len(images)
    for i in range(remaining):
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        images.append(img)
    
    images = np.stack(images, axis=0)
    logger.info(f"Generated {num_images} test images with diverse patterns")
    return images


def encode_with_arrow(
    encoder,
    images: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Encode images with ArrowEngine VisionEncoder.
    
    Args:
        encoder: VisionEncoder instance
        images: (n, 224, 224, 3) uint8 array
        
    Returns:
        (embeddings, inference_time)
    """
    start_time = time.time()
    embeddings = encoder.encode(images, normalize=True)
    inference_time = time.time() - start_time
    
    # Ensure 2D array
    if embeddings.ndim == 1:
        embeddings = embeddings[np.newaxis, :]
    
    logger.info(
        f"ArrowEngine encoded {len(images)} images in {inference_time:.3f}s "
        f"({inference_time/len(images)*1000:.1f}ms per image)"
    )
    return embeddings, inference_time


def encode_with_hf(
    model,
    processor,
    images: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Encode images with HuggingFace CLIP.
    
    Args:
        model: CLIP model
        processor: CLIP processor
        images: (n, 224, 224, 3) uint8 array
        
    Returns:
        (embeddings, inference_time)
    """
    from PIL import Image
    
    # Convert numpy to PIL images
    pil_images = [Image.fromarray(img) for img in images]
    
    # Process images
    inputs = processor(images=pil_images, return_tensors="pt")
    
    # Encode
    start_time = time.time()
    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
        image_features = model.visual_projection(vision_outputs.pooler_output)
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    inference_time = time.time() - start_time
    
    embeddings = image_features.cpu().numpy().astype(np.float32)
    
    logger.info(
        f"HuggingFace encoded {len(images)} images in {inference_time:.3f}s "
        f"({inference_time/len(images)*1000:.1f}ms per image)"
    )
    return embeddings, inference_time


def compute_cosine_similarities(
    arrow_embeddings: np.ndarray,
    hf_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise cosine similarities.
    
    Args:
        arrow_embeddings: (n, d) ArrowEngine embeddings
        hf_embeddings: (n, d) HuggingFace embeddings
        
    Returns:
        similarities: (n,) cosine similarities
    """
    # Compute dot product (cosine similarity for normalized vectors)
    similarities = np.sum(arrow_embeddings * hf_embeddings, axis=1)
    return similarities


def validate_precision(
    arrow_embeddings: np.ndarray,
    hf_embeddings: np.ndarray,
    threshold: float = 0.95
) -> Dict:
    """
    Validate precision by comparing embeddings.
    
    Args:
        arrow_embeddings: ArrowEngine embeddings
        hf_embeddings: HuggingFace embeddings
        threshold: Minimum average similarity
        
    Returns:
        Validation results dictionary
    """
    logger.info("Computing cosine similarities...")
    
    similarities = compute_cosine_similarities(arrow_embeddings, hf_embeddings)
    
    results = {
        "num_samples": len(similarities),
        "mean_similarity": float(np.mean(similarities)),
        "std_similarity": float(np.std(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "median_similarity": float(np.median(similarities)),
        "threshold": threshold,
        "passed": np.mean(similarities) >= threshold,
    }
    
    # Log results
    logger.info("=" * 60)
    logger.info("Precision Validation Results:")
    logger.info(f"  Samples: {results['num_samples']}")
    logger.info(f"  Mean similarity: {results['mean_similarity']:.6f}")
    logger.info(f"  Std similarity: {results['std_similarity']:.6f}")
    logger.info(f"  Min similarity: {results['min_similarity']:.6f}")
    logger.info(f"  Max similarity: {results['max_similarity']:.6f}")
    logger.info(f"  Median similarity: {results['median_similarity']:.6f}")
    logger.info(f"  Threshold: {results['threshold']:.2f}")
    logger.info(f"  Status: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
    logger.info("=" * 60)
    
    return results


def main():
    """Main validation function."""
    args = parse_args()
    
    logger.info("Starting Vision Encoder Precision Validation")
    logger.info(f"ArrowEngine model: {args.arrow_model}")
    logger.info(f"HuggingFace model: {args.hf_model}")
    logger.info(f"Test images: {args.num_images}")
    logger.info(f"Threshold: {args.threshold}")
    
    try:
        # Step 1: Load models
        arrow_encoder, arrow_load_time = load_arrow_encoder(args.arrow_model)
        hf_model, hf_processor, hf_load_time = load_hf_encoder(args.hf_model)
        
        logger.info(f"Load time comparison: Arrow={arrow_load_time:.3f}s, HF={hf_load_time:.3f}s")
        logger.info(f"Speedup: {hf_load_time/arrow_load_time:.2f}x faster")
        
        # Step 2: Generate test images
        test_images = generate_test_images(args.num_images)
        
        # Step 3: Encode with both models
        arrow_embeddings, arrow_time = encode_with_arrow(arrow_encoder, test_images)
        hf_embeddings, hf_time = encode_with_hf(hf_model, hf_processor, test_images)
        
        logger.info(f"Inference time comparison: Arrow={arrow_time:.3f}s, HF={hf_time:.3f}s")
        logger.info(f"Speedup: {hf_time/arrow_time:.2f}x faster")
        
        # Step 4: Validate precision
        results = validate_precision(arrow_embeddings, hf_embeddings, args.threshold)
        
        # Step 5: Summary
        logger.info("=" * 60)
        logger.info("Validation Summary:")
        logger.info(f"  Precision: {results['mean_similarity']:.4f} (threshold: {args.threshold})")
        logger.info(f"  Load speedup: {hf_load_time/arrow_load_time:.2f}x")
        logger.info(f"  Inference speedup: {hf_time/arrow_time:.2f}x")
        logger.info(f"  Status: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        logger.info("=" * 60)
        
        return 0 if results['passed'] else 1
        
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

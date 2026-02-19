#!/usr/bin/env python3
"""
CLIP Engine Precision Validation

Validates that ArrowEngine CLIP produces text-image similarities
that correlate highly (Pearson > 0.95) with HuggingFace CLIP.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_compression.logger import logger
from llm_compression.multimodal.clip_engine import CLIPEngine


def generate_test_data(num_samples: int = 20) -> Tuple[List[str], List[np.ndarray]]:
    """
    Generate diverse test text-image pairs.
    
    Args:
        num_samples: Number of test samples to generate
        
    Returns:
        Tuple of (texts, images)
    """
    # Diverse text descriptions
    texts = [
        "a photo of a cat",
        "a dog playing in the park",
        "a beautiful sunset over the ocean",
        "a person riding a bicycle",
        "a red sports car on the highway",
    ][:num_samples]
    
    # Generate random images (in practice, use real images)
    # For validation, we use synthetic images to test the pipeline
    images = []
    for i in range(min(num_samples, len(texts))):
        # Create diverse synthetic images
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        # Add some structure to make images more realistic
        img[:, :, 0] = ((img[:, :, 0].astype(np.int32) + i * 10) % 256).astype(np.uint8)  # Vary red channel
        img[:, :, 1] = ((img[:, :, 1].astype(np.int32) + i * 5) % 256).astype(np.uint8)   # Vary green channel
        images.append(img)
    
    return texts, images


def validate_clip_precision(
    vision_model_path: str,
    num_samples: int = 20,
    correlation_threshold: float = 0.95
) -> dict:
    """
    Validate CLIP-style text-image similarity against HuggingFace.
    
    This validates that our vision encoder produces embeddings that,
    when combined with text embeddings, correlate highly with HuggingFace CLIP.
    
    Args:
        vision_model_path: Path to converted vision model (CLIP vision encoder)
        num_samples: Number of test samples
        correlation_threshold: Minimum Pearson correlation required
        
    Returns:
        Validation results dictionary
    """
    logger.info(f"Loading HuggingFace CLIP model")
    from transformers import CLIPModel, CLIPProcessor
    from llm_compression.multimodal.vision_encoder import VisionEncoder
    
    # Load HuggingFace CLIP
    hf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    hf_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    hf_model.eval()
    
    logger.info(f"Loading ArrowEngine Vision Encoder from: {vision_model_path}")
    arrow_vision = VisionEncoder(vision_model_path)
    
    # Generate test data
    logger.info(f"Generating {num_samples} test samples")
    texts, images = generate_test_data(num_samples)
    
    # Compute similarities with HuggingFace
    logger.info("Computing HuggingFace CLIP similarities")
    with torch.no_grad():
        hf_inputs = hf_processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )
        hf_outputs = hf_model(**hf_inputs)
        hf_similarities = hf_outputs.logits_per_text.cpu().numpy()
    
    # Compute similarities with ArrowEngine
    # We'll use HF text encoder + Arrow vision encoder
    logger.info("Computing ArrowEngine vision + HF text similarities")
    with torch.no_grad():
        # Get HF text embeddings
        text_inputs = hf_processor(text=texts, return_tensors="pt", padding=True)
        hf_text_features = hf_model.get_text_features(**text_inputs)
        hf_text_features = hf_text_features / hf_text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Get Arrow vision embeddings
        arrow_vision_features = []
        for img in images:
            emb = arrow_vision.encode(img)
            arrow_vision_features.append(emb)
        arrow_vision_features = torch.from_numpy(np.stack(arrow_vision_features))
        arrow_vision_features = arrow_vision_features / torch.norm(arrow_vision_features, p=2, dim=-1, keepdim=True)
        
        # Compute similarity (text x vision)
        # Scale by logit_scale like CLIP does
        logit_scale = hf_model.logit_scale.exp()
        arrow_similarities = (logit_scale * hf_text_features @ arrow_vision_features.T).cpu().numpy()
    
    # Flatten similarity matrices for correlation
    hf_flat = hf_similarities.flatten()
    arrow_flat = arrow_similarities.flatten()
    
    # Compute Pearson correlation
    correlation, p_value = pearsonr(hf_flat, arrow_flat)
    
    # Compute statistics
    diff = hf_flat - arrow_flat
    abs_diff = np.abs(diff)
    
    results = {
        "num_samples": num_samples,
        "num_pairs": len(hf_flat),
        "correlation": correlation,
        "p_value": p_value,
        "threshold": correlation_threshold,
        "passed": correlation >= correlation_threshold,
        "statistics": {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std()),
            "max_abs_diff": float(abs_diff.max()),
            "hf_mean": float(hf_flat.mean()),
            "hf_std": float(hf_flat.std()),
            "arrow_mean": float(arrow_flat.mean()),
            "arrow_std": float(arrow_flat.std()),
        }
    }
    
    return results


def print_results(results: dict) -> None:
    """Print validation results in a formatted way."""
    print("\n" + "="*70)
    print("  CLIP Engine Precision Validation Results")
    print("="*70)
    print()
    print(f"Test Samples:    {results['num_samples']}")
    print(f"Total Pairs:     {results['num_pairs']}")
    print()
    print("--- Correlation Analysis ---")
    print(f"Pearson Correlation: {results['correlation']:.6f}")
    print(f"P-value:             {results['p_value']:.2e}")
    print(f"Threshold:           {results['threshold']:.6f}")
    print()
    print("--- Similarity Statistics ---")
    stats = results['statistics']
    print(f"HuggingFace - Mean: {stats['hf_mean']:.6f}, Std: {stats['hf_std']:.6f}")
    print(f"ArrowEngine - Mean: {stats['arrow_mean']:.6f}, Std: {stats['arrow_std']:.6f}")
    print(f"Difference  - Mean: {stats['mean_diff']:.6f}, Std: {stats['std_diff']:.6f}")
    print(f"Max Abs Diff:       {stats['max_abs_diff']:.6f}")
    print()
    print("="*70)
    if results['passed']:
        print("✅ PASSED: Correlation >= threshold")
    else:
        print("❌ FAILED: Correlation < threshold")
    print("="*70)
    print()


def main():
    """Main validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate CLIP-style vision encoder precision against HuggingFace"
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default="D:/ai-models/clip-vit-b32",
        help="Path to converted vision model directory"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of test samples (default: 5)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Minimum Pearson correlation threshold (default: 0.95)"
    )
    
    args = parser.parse_args()
    
    # Run validation
    results = validate_clip_precision(
        vision_model_path=args.vision_model,
        num_samples=args.samples,
        correlation_threshold=args.threshold
    )
    
    # Print results
    print_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if results['passed'] else 1)


if __name__ == "__main__":
    main()

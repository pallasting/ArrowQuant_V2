#!/usr/bin/env python3
"""
Model Conversion Validation Script

Validates converted models by comparing embeddings between original HuggingFace
models and converted ArrowEngine models. Computes cosine similarity and reports
compression metrics.

Usage:
    python scripts/validate_model_conversion.py --model openai/clip-vit-base-patch32 --converted D:/ai-models/clip-vit-b32 --type clip
    python scripts/validate_model_conversion.py --model openai/whisper-base --converted D:/ai-models/whisper-base --type whisper
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_compression.logger import logger


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
    return float(np.dot(vec1_norm, vec2_norm))


def validate_clip_conversion(
    model_name: str,
    converted_path: Path,
    num_samples: int = 10
) -> Dict[str, Any]:
    """
    Validate CLIP model conversion by comparing embeddings.
    
    Args:
        model_name: HuggingFace model name
        converted_path: Path to converted model
        num_samples: Number of test samples
        
    Returns:
        Validation results dictionary
    """
    try:
        from transformers import CLIPModel, CLIPProcessor
        from llm_compression.multimodal.vision_encoder import VisionEncoder
    except ImportError as e:
        logger.error(f"Required library not installed: {e}")
        return {"success": False, "error": str(e)}
    
    logger.info(f"Loading original CLIP model: {model_name}")
    hf_model = CLIPModel.from_pretrained(model_name)
    hf_processor = CLIPProcessor.from_pretrained(model_name)
    hf_model.eval()
    
    logger.info(f"Loading converted model from: {converted_path}")
    arrow_encoder = VisionEncoder(str(converted_path))
    
    # Generate test images
    logger.info(f"Generating {num_samples} test images")
    test_images = []
    for i in range(num_samples):
        # Create random RGB images (224x224)
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        test_images.append(img)
    
    # Compute embeddings with both models
    similarities = []
    
    with torch.no_grad():
        for i, img in enumerate(test_images):
            # HuggingFace embedding
            inputs = hf_processor(images=img, return_tensors="pt")
            hf_output = hf_model.get_image_features(**inputs)
            hf_embedding = hf_output.cpu().numpy().flatten()
            
            # ArrowEngine embedding
            arrow_embedding = arrow_encoder.encode(img)
            
            # Compute similarity
            similarity = compute_cosine_similarity(hf_embedding, arrow_embedding)
            similarities.append(similarity)
            
            logger.info(f"Sample {i+1}/{num_samples}: similarity = {similarity:.6f}")
    
    # Compute statistics
    avg_similarity = float(np.mean(similarities))
    min_similarity = float(np.min(similarities))
    max_similarity = float(np.max(similarities))
    std_similarity = float(np.std(similarities))
    
    # Check file sizes
    parquet_path = converted_path / "weights.parquet"
    if parquet_path.exists():
        converted_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    else:
        converted_size_mb = 0.0
    
    # Estimate original size (rough approximation)
    total_params = sum(p.numel() for p in hf_model.vision_model.parameters())
    original_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
    compression_ratio = original_size_mb / converted_size_mb if converted_size_mb > 0 else 0.0
    
    results = {
        "success": True,
        "model_type": "CLIP",
        "model_name": model_name,
        "num_samples": num_samples,
        "avg_similarity": avg_similarity,
        "min_similarity": min_similarity,
        "max_similarity": max_similarity,
        "std_similarity": std_similarity,
        "passed": avg_similarity >= 0.95,
        "threshold": 0.95,
        "original_size_mb": original_size_mb,
        "converted_size_mb": converted_size_mb,
        "compression_ratio": compression_ratio,
    }
    
    return results


def validate_whisper_conversion(
    model_name: str,
    converted_path: Path,
    num_samples: int = 10
) -> Dict[str, Any]:
    """
    Validate Whisper model conversion by comparing embeddings.
    
    Args:
        model_name: HuggingFace model name
        converted_path: Path to converted model
        num_samples: Number of test samples
        
    Returns:
        Validation results dictionary
    """
    try:
        from transformers import WhisperModel, WhisperProcessor
        from llm_compression.multimodal.audio_encoder import AudioEncoder, AudioConfig
        import json
    except ImportError as e:
        logger.error(f"Required library not installed: {e}")
        return {"success": False, "error": str(e)}
    
    logger.info(f"Loading original Whisper model: {model_name}")
    hf_model = WhisperModel.from_pretrained(model_name)
    hf_processor = WhisperProcessor.from_pretrained(model_name)
    hf_model.eval()
    
    logger.info(f"Loading converted model from: {converted_path}")
    
    # Load metadata to get correct configuration
    metadata_path = converted_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        config_dict = metadata.get("config", {})
        config = AudioConfig(
            n_mels=config_dict.get("n_mels", 80),
            hidden_size=config_dict.get("hidden_size", 512),
            num_layers=config_dict.get("num_layers", 6),
            num_attention_heads=config_dict.get("num_attention_heads", 8),
            intermediate_size=config_dict.get("intermediate_size", 2048),
            layer_norm_eps=config_dict.get("layer_norm_eps", 1e-5),
            max_positions=config_dict.get("max_positions", 1500),
        )
        logger.info(f"Loaded config from metadata: hidden_size={config.hidden_size}, num_layers={config.num_layers}")
    else:
        config = None
        logger.warning("No metadata.json found, using default config")
    
    arrow_encoder = AudioEncoder(str(converted_path), config=config)
    
    # Generate test audio (random waveforms)
    logger.info(f"Generating {num_samples} test audio samples")
    sample_rate = 16000
    duration = 3.0  # 3 seconds
    num_samples_audio = int(sample_rate * duration)
    
    similarities = []
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate random audio
            audio = np.random.randn(num_samples_audio).astype(np.float32)
            
            # HuggingFace embedding
            inputs = hf_processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            hf_output = hf_model.encoder(**inputs)
            # Mean pool over time dimension
            hf_embedding = hf_output.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            
            # ArrowEngine embedding
            arrow_embedding = arrow_encoder.encode(audio)
            
            # Compute similarity
            similarity = compute_cosine_similarity(hf_embedding, arrow_embedding)
            similarities.append(similarity)
            
            logger.info(f"Sample {i+1}/{num_samples}: similarity = {similarity:.6f}")
    
    # Compute statistics
    avg_similarity = float(np.mean(similarities))
    min_similarity = float(np.min(similarities))
    max_similarity = float(np.max(similarities))
    std_similarity = float(np.std(similarities))
    
    # Check file sizes
    parquet_path = converted_path / "weights.parquet"
    if parquet_path.exists():
        converted_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    else:
        converted_size_mb = 0.0
    
    # Estimate original size
    total_params = sum(p.numel() for p in hf_model.encoder.parameters())
    original_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
    compression_ratio = original_size_mb / converted_size_mb if converted_size_mb > 0 else 0.0
    
    results = {
        "success": True,
        "model_type": "Whisper",
        "model_name": model_name,
        "num_samples": num_samples,
        "avg_similarity": avg_similarity,
        "min_similarity": min_similarity,
        "max_similarity": max_similarity,
        "std_similarity": std_similarity,
        "passed": avg_similarity >= 0.95,
        "threshold": 0.95,
        "original_size_mb": original_size_mb,
        "converted_size_mb": converted_size_mb,
        "compression_ratio": compression_ratio,
    }
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print validation results in a formatted way."""
    print("\n" + "=" * 70)
    print("  Model Conversion Validation Results")
    print("=" * 70)
    
    if not results.get("success", False):
        print(f"\n❌ FAILED: {results.get('error', 'Unknown error')}")
        return
    
    print(f"\nModel Type:      {results['model_type']}")
    print(f"Model Name:      {results['model_name']}")
    print(f"Test Samples:    {results['num_samples']}")
    
    print(f"\n--- Embedding Similarity ---")
    print(f"Average:         {results['avg_similarity']:.6f}")
    print(f"Minimum:         {results['min_similarity']:.6f}")
    print(f"Maximum:         {results['max_similarity']:.6f}")
    print(f"Std Dev:         {results['std_similarity']:.6f}")
    print(f"Threshold:       {results['threshold']:.6f}")
    
    print(f"\n--- Compression Metrics ---")
    print(f"Original Size:   {results['original_size_mb']:.2f} MB")
    print(f"Converted Size:  {results['converted_size_mb']:.2f} MB")
    print(f"Compression:     {results['compression_ratio']:.2f}x")
    
    print("\n" + "=" * 70)
    
    if results['passed']:
        print("✅ PASSED: Average similarity >= threshold")
    else:
        print(f"❌ FAILED: Average similarity {results['avg_similarity']:.6f} < threshold {results['threshold']:.6f}")
    
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate model conversion by comparing embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate CLIP conversion
  python scripts/validate_model_conversion.py \\
      --model openai/clip-vit-base-patch32 \\
      --converted D:/ai-models/clip-vit-b32 \\
      --type clip

  # Validate Whisper conversion
  python scripts/validate_model_conversion.py \\
      --model openai/whisper-base \\
      --converted D:/ai-models/whisper-base \\
      --type whisper \\
      --samples 20
        """
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--converted",
        required=True,
        help="Path to converted model directory"
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["clip", "whisper"],
        help="Model type"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of test samples (default: 10)"
    )
    
    args = parser.parse_args()
    
    converted_path = Path(args.converted)
    if not converted_path.exists():
        logger.error(f"Converted model path does not exist: {converted_path}")
        sys.exit(1)
    
    # Run validation
    if args.type == "clip":
        results = validate_clip_conversion(
            model_name=args.model,
            converted_path=converted_path,
            num_samples=args.samples
        )
    elif args.type == "whisper":
        results = validate_whisper_conversion(
            model_name=args.model,
            converted_path=converted_path,
            num_samples=args.samples
        )
    else:
        logger.error(f"Unsupported model type: {args.type}")
        sys.exit(1)
    
    # Print results
    print_results(results)
    
    # Exit with appropriate code
    if results.get("success", False) and results.get("passed", False):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

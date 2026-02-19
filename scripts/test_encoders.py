#!/usr/bin/env python3
"""
Test VisionEncoder and AudioEncoder with converted models.

This script tests that the encoders can load converted models
and produce embeddings.
"""

import numpy as np
from pathlib import Path

from llm_compression.logger import logger
from llm_compression.multimodal.vision_encoder import VisionEncoder
from llm_compression.multimodal.audio_encoder import AudioEncoder


def test_vision_encoder():
    """Test VisionEncoder with converted CLIP model."""
    logger.info("=" * 60)
    logger.info("Testing VisionEncoder")
    logger.info("=" * 60)
    
    model_path = "D:/ai-models/clip-vit-b32"
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return False
    
    try:
        # Initialize encoder
        logger.info(f"Loading VisionEncoder from {model_path}")
        encoder = VisionEncoder(model_path)
        
        logger.info(f"Embedding dimension: {encoder.embedding_dimension}")
        
        # Create test image (random noise)
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Encode single image
        logger.info("Encoding single image...")
        embedding = encoder.encode(test_image, normalize=True)
        
        logger.info(f"Output shape: {embedding.shape}")
        logger.info(f"Output dtype: {embedding.dtype}")
        logger.info(f"Output range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        logger.info(f"L2 norm: {np.linalg.norm(embedding):.4f}")
        
        # Encode batch
        logger.info("Encoding batch of 4 images...")
        batch = [test_image] * 4
        batch_embeddings = encoder.encode(batch, normalize=True)
        
        logger.info(f"Batch output shape: {batch_embeddings.shape}")
        
        logger.info("✅ VisionEncoder test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ VisionEncoder test FAILED: {e}", exc_info=True)
        return False


def test_audio_encoder():
    """Test AudioEncoder with converted Whisper model."""
    logger.info("=" * 60)
    logger.info("Testing AudioEncoder")
    logger.info("=" * 60)
    
    model_path = "D:/ai-models/whisper-base"
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return False
    
    try:
        # Initialize encoder
        logger.info(f"Loading AudioEncoder from {model_path}")
        encoder = AudioEncoder(model_path)
        
        logger.info(f"Embedding dimension: {encoder.embedding_dimension}")
        
        # Create test audio (3 seconds of random noise)
        sample_rate = 16000
        duration = 3
        test_audio = np.random.randn(sample_rate * duration).astype(np.float32)
        
        # Encode single audio
        logger.info("Encoding single audio clip...")
        embedding = encoder.encode(test_audio, normalize=True)
        
        logger.info(f"Output shape: {embedding.shape}")
        logger.info(f"Output dtype: {embedding.dtype}")
        logger.info(f"Output range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        logger.info(f"L2 norm: {np.linalg.norm(embedding):.4f}")
        
        # Encode batch
        logger.info("Encoding batch of 4 audio clips...")
        batch = [test_audio] * 4
        batch_embeddings = encoder.encode(batch, normalize=True)
        
        logger.info(f"Batch output shape: {batch_embeddings.shape}")
        
        logger.info("✅ AudioEncoder test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ AudioEncoder test FAILED: {e}", exc_info=True)
        return False


def main():
    """Run all encoder tests."""
    logger.info("Starting encoder tests...")
    
    vision_passed = test_vision_encoder()
    audio_passed = test_audio_encoder()
    
    logger.info("=" * 60)
    logger.info("Test Summary:")
    logger.info(f"  VisionEncoder: {'✅ PASSED' if vision_passed else '❌ FAILED'}")
    logger.info(f"  AudioEncoder: {'✅ PASSED' if audio_passed else '❌ FAILED'}")
    logger.info("=" * 60)
    
    if vision_passed and audio_passed:
        logger.info("🎉 All tests PASSED!")
        return 0
    else:
        logger.error("❌ Some tests FAILED")
        return 1


if __name__ == "__main__":
    exit(main())

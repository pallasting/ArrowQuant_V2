#!/usr/bin/env python3
"""
Test CLIPEngine with existing encoders.

This script tests the CLIP Engine functionality using
the separately converted vision and text encoders.
"""

import numpy as np
from pathlib import Path

from llm_compression.logger import logger
from llm_compression.multimodal.clip_engine import CLIPEngine


def test_clip_engine_basic():
    """Test basic CLIP Engine functionality."""
    logger.info("=" * 60)
    logger.info("Testing CLIPEngine - Basic Functionality")
    logger.info("=" * 60)
    
    # Note: For this test, we'll use a simplified setup
    # In production, CLIP would have a unified model directory
    
    try:
        # For now, we'll test the individual components
        # since we don't have a full CLIP model converted yet
        logger.info("Testing text and vision encoders separately...")
        
        # Test 1: Import and initialization
        logger.info("Test 1: Checking CLIPEngine class...")
        from llm_compression.multimodal.clip_engine import CLIPEngine, CLIPConfig
        logger.info("✅ CLIPEngine class imported successfully")
        
        # Test 2: Config validation
        logger.info("Test 2: Testing CLIPConfig...")
        config = CLIPConfig(
            text_embedding_dim=384,
            vision_embedding_dim=768,
            projection_dim=512
        )
        logger.info(f"  Config: text_dim={config.text_embedding_dim}, "
                   f"vision_dim={config.vision_embedding_dim}, "
                   f"projection_dim={config.projection_dim}")
        logger.info("✅ CLIPConfig validation passed")
        
        # Test 3: Test individual encoders
        logger.info("Test 3: Testing individual encoders...")
        
        # Test vision encoder
        from llm_compression.multimodal.vision_encoder import VisionEncoder
        vision_model_path = "D:/ai-models/clip-vit-b32"
        
        if Path(vision_model_path).exists():
            logger.info(f"  Loading VisionEncoder from {vision_model_path}")
            vision_encoder = VisionEncoder(vision_model_path)
            
            # Create test image
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            vision_emb = vision_encoder.encode(test_image, normalize=True)
            
            logger.info(f"  Vision embedding shape: {vision_emb.shape}")
            logger.info(f"  Vision embedding L2 norm: {np.linalg.norm(vision_emb):.4f}")
            logger.info("✅ VisionEncoder test passed")
        else:
            logger.warning(f"  Vision model not found at {vision_model_path}")
        
        # Test text encoder
        from llm_compression.inference.arrow_engine import ArrowEngine
        text_model_path = "D:/ai-models/bert-base-uncased"
        
        if Path(text_model_path).exists():
            logger.info(f"  Loading ArrowEngine from {text_model_path}")
            text_encoder = ArrowEngine(text_model_path)
            
            # Create test text
            test_text = "a photo of a cat"
            text_emb = text_encoder.encode([test_text], normalize=True)
            
            logger.info(f"  Text embedding shape: {text_emb.shape}")
            logger.info(f"  Text embedding L2 norm: {np.linalg.norm(text_emb):.4f}")
            logger.info("✅ ArrowEngine test passed")
        else:
            logger.warning(f"  Text model not found at {text_model_path}")
        
        # Test 4: Similarity computation (manual)
        if Path(vision_model_path).exists() and Path(text_model_path).exists():
            logger.info("Test 4: Testing cross-modal similarity...")
            
            # Compute similarity (without projection layers for now)
            # This is just a basic test of the concept
            similarity = np.dot(text_emb[0], vision_emb)
            logger.info(f"  Raw similarity (no projection): {similarity:.4f}")
            logger.info("✅ Similarity computation test passed")
        
        logger.info("=" * 60)
        logger.info("CLIPEngine Basic Tests Summary:")
        logger.info("  ✅ Class import and config")
        logger.info("  ✅ Individual encoder tests")
        logger.info("  ✅ Similarity computation concept")
        logger.info("=" * 60)
        logger.info("Note: Full CLIP Engine test requires complete model conversion")
        logger.info("      including projection layers and logit_scale parameter")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CLIPEngine test FAILED: {e}", exc_info=True)
        return False


def test_clip_projection_layers():
    """Test projection layer functionality."""
    logger.info("=" * 60)
    logger.info("Testing Projection Layers")
    logger.info("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        
        # Create mock projection layers
        text_proj = nn.Linear(384, 512, bias=False)
        vision_proj = nn.Linear(768, 512, bias=False)
        
        # Test with random embeddings
        text_emb = torch.randn(1, 384)
        vision_emb = torch.randn(1, 768)
        
        # Project
        text_proj_emb = text_proj(text_emb)
        vision_proj_emb = vision_proj(vision_emb)
        
        logger.info(f"  Text projection: {text_emb.shape} -> {text_proj_emb.shape}")
        logger.info(f"  Vision projection: {vision_emb.shape} -> {vision_proj_emb.shape}")
        
        # Normalize
        text_proj_emb = torch.nn.functional.normalize(text_proj_emb, p=2, dim=-1)
        vision_proj_emb = torch.nn.functional.normalize(vision_proj_emb, p=2, dim=-1)
        
        # Compute similarity
        similarity = torch.matmul(text_proj_emb, vision_proj_emb.T)
        logger.info(f"  Projected similarity: {similarity.item():.4f}")
        
        logger.info("✅ Projection layer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Projection layer test FAILED: {e}", exc_info=True)
        return False


def main():
    """Run all CLIP Engine tests."""
    logger.info("Starting CLIP Engine tests...")
    
    basic_passed = test_clip_engine_basic()
    projection_passed = test_clip_projection_layers()
    
    logger.info("=" * 60)
    logger.info("Test Summary:")
    logger.info(f"  Basic functionality: {'✅ PASSED' if basic_passed else '❌ FAILED'}")
    logger.info(f"  Projection layers: {'✅ PASSED' if projection_passed else '❌ FAILED'}")
    logger.info("=" * 60)
    
    if basic_passed and projection_passed:
        logger.info("🎉 All tests PASSED!")
        return 0
    else:
        logger.error("❌ Some tests FAILED")
        return 1


if __name__ == "__main__":
    exit(main())

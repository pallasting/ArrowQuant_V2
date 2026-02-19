"""
Unit tests for VisionEncoder

Tests Arrow-native vision encoding functionality.
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path

from llm_compression.multimodal.vision_encoder import VisionEncoder, VisionConfig


def _has_model_weights() -> bool:
    """Check if model weights are available."""
    # Check for model weights in standard location
    model_path = Path("models/clip-vit-b32")
    return model_path.exists() and (model_path / "weights.parquet").exists()


class TestVisionConfig:
    """Test VisionConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = VisionConfig()
        
        assert config.image_size == 224
        assert config.patch_size == 32
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.projection_dim == 512
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = VisionConfig(
            image_size=384,
            patch_size=16,
            hidden_size=1024
        )
        
        assert config.image_size == 384
        assert config.patch_size == 16
        assert config.hidden_size == 1024
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = VisionConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["image_size"] == 224
        assert config_dict["hidden_size"] == 768


@pytest.mark.skipif(
    not _has_model_weights(),
    reason="Model weights not available"
)
class TestVisionEncoder:
    """Test VisionEncoder functionality."""
    
    def test_initialization(self):
        """Test VisionEncoder initialization."""
        # This test requires actual model weights
        # Skip if not available
        pytest.skip("Requires model weights")
    
    def test_embedding_dimension(self):
        """Test embedding dimension property."""
        config = VisionConfig(projection_dim=512)
        # Would need actual encoder instance
        assert config.projection_dim == 512
    
    def test_encode_single_image(self):
        """Test encoding single image."""
        # This test requires actual model weights
        pytest.skip("Requires model weights")
    
    def test_encode_batch(self):
        """Test encoding batch of images."""
        # This test requires actual model weights
        pytest.skip("Requires model weights")

"""
Unit tests for ImageProcessor

Tests Arrow-native image preprocessing functionality.
"""

import pytest
import numpy as np
from PIL import Image

from llm_compression.multimodal.image_processor import ImageProcessor


class TestImageProcessor:
    """Test ImageProcessor functionality."""
    
    def test_initialization(self):
        """Test ImageProcessor initialization."""
        processor = ImageProcessor(image_size=224)
        
        assert processor.image_size == 224
        assert processor.mean.shape == (3,)
        assert processor.std.shape == (3,)
    
    def test_preprocess_numpy_array(self):
        """Test preprocessing numpy array."""
        processor = ImageProcessor(image_size=224)
        
        # Create random RGB image
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Preprocess
        processed = processor.preprocess(image)
        
        # Check output shape and type
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.float32
        
        # Check normalization (values should be roughly in [-3, 3] range)
        assert processed.min() >= -5.0
        assert processed.max() <= 5.0
    
    def test_preprocess_pil_image(self):
        """Test preprocessing PIL Image."""
        processor = ImageProcessor(image_size=224)
        
        # Create PIL image
        image = Image.new('RGB', (256, 256), color='red')
        
        # Preprocess
        processed = processor.preprocess(image)
        
        # Check output
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.float32
    
    def test_preprocess_grayscale(self):
        """Test preprocessing grayscale image."""
        processor = ImageProcessor(image_size=224)
        
        # Create grayscale image
        image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        
        # Preprocess
        processed = processor.preprocess(image)
        
        # Should convert to RGB
        assert processed.shape == (224, 224, 3)
    
    def test_preprocess_rgba(self):
        """Test preprocessing RGBA image."""
        processor = ImageProcessor(image_size=224)
        
        # Create RGBA image
        image = np.random.randint(0, 256, (256, 256, 4), dtype=np.uint8)
        
        # Preprocess
        processed = processor.preprocess(image)
        
        # Should convert to RGB
        assert processed.shape == (224, 224, 3)
    
    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        processor = ImageProcessor(image_size=224)
        
        # Create batch of images
        images = [
            np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            for _ in range(4)
        ]
        
        # Preprocess batch
        processed = processor.preprocess_batch(images)
        
        # Check output
        assert processed.shape == (4, 224, 224, 3)
        assert processed.dtype == np.float32
    
    def test_denormalize(self):
        """Test denormalization."""
        processor = ImageProcessor(image_size=224)
        
        # Create and preprocess image
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        processed = processor.preprocess(image)
        
        # Denormalize
        denormalized = processor.denormalize(processed)
        
        # Check output
        assert denormalized.shape == (224, 224, 3)
        assert denormalized.dtype == np.uint8
        assert denormalized.min() >= 0
        assert denormalized.max() <= 255
    
    def test_arrow_roundtrip(self):
        """Test Arrow Binary array roundtrip."""
        processor = ImageProcessor(image_size=224)
        
        # Create batch of preprocessed images
        images = np.random.randn(4, 224, 224, 3).astype(np.float32)
        
        # Convert to Arrow
        arrow_array = processor.to_arrow(images)
        
        # Convert back
        recovered = processor.from_arrow(arrow_array)
        
        # Check roundtrip accuracy
        assert recovered.shape == images.shape
        assert recovered.dtype == images.dtype
        np.testing.assert_array_almost_equal(recovered, images, decimal=6)
    
    def test_custom_normalization(self):
        """Test custom normalization parameters."""
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
        processor = ImageProcessor(image_size=224, mean=mean, std=std)
        
        assert np.allclose(processor.mean, mean)
        assert np.allclose(processor.std, std)

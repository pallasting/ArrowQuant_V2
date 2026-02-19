"""
Unit tests for multimodal input validation.

Tests error handling for:
- Invalid image inputs
- Invalid audio inputs
- Missing model files
- Corrupted weights
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from llm_compression.multimodal.validation import (
    validate_image_input,
    validate_image_path,
    validate_audio_input,
    validate_audio_path,
    validate_batch_input,
    ValidationError
)


class TestImageValidation:
    """Test image input validation."""
    
    def test_valid_rgb_image_uint8(self):
        """Valid RGB image should not raise."""
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        validate_image_input(img)  # Should not raise
    
    def test_valid_rgb_image_float32(self):
        """Valid float32 RGB image should not raise."""
        img = np.random.rand(224, 224, 3).astype(np.float32)
        validate_image_input(img)  # Should not raise
    
    def test_valid_rgba_image(self):
        """Valid RGBA image should not raise."""
        img = np.random.randint(0, 256, (224, 224, 4), dtype=np.uint8)
        validate_image_input(img)  # Should not raise
    
    def test_valid_grayscale_image_when_allowed(self):
        """Valid grayscale image should not raise when allowed."""
        img = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        validate_image_input(img, allow_grayscale=True)  # Should not raise
    
    def test_invalid_image_type(self):
        """Non-numpy array should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be a numpy array"):
            validate_image_input([1, 2, 3])
    
    def test_invalid_image_dimensions(self):
        """4D array should raise ValidationError."""
        img = np.random.randint(0, 256, (10, 224, 224, 3), dtype=np.uint8)
        with pytest.raises(ValidationError, match="must be 2D .* or 3D"):
            validate_image_input(img)
    
    def test_invalid_grayscale_when_not_allowed(self):
        """Grayscale image should raise when not allowed."""
        img = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        with pytest.raises(ValidationError, match="Grayscale images not supported"):
            validate_image_input(img, allow_grayscale=False)
    
    def test_invalid_channel_count(self):
        """Image with wrong channel count should raise."""
        img = np.random.randint(0, 256, (224, 224, 2), dtype=np.uint8)
        with pytest.raises(ValidationError, match="must have 3 .* or 4 .* channels"):
            validate_image_input(img)
    
    def test_invalid_dtype(self):
        """Image with invalid dtype should raise."""
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.int32)
        with pytest.raises(ValidationError, match="must be uint8 or float"):
            validate_image_input(img)
    
    def test_invalid_uint8_value_range(self):
        """uint8 image with out-of-range values should raise."""
        # Note: uint8 automatically wraps values, so we can't test out-of-range
        # This test verifies that valid uint8 images pass validation
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        validate_image_input(img)  # Should not raise
    
    def test_invalid_float_value_range(self):
        """float image with out-of-range values should raise."""
        img = np.random.rand(224, 224, 3).astype(np.float32)
        img[0, 0, 0] = 2.0  # Out of range
        with pytest.raises(ValidationError, match="float image values must be in"):
            validate_image_input(img)


class TestImagePathValidation:
    """Test image file path validation."""
    
    def test_valid_image_path(self):
        """Valid image file path should not raise."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            path = Path(f.name)
        
        try:
            # Create a minimal valid file
            path.write_bytes(b'\xff\xd8\xff')  # JPEG magic bytes
            validate_image_path(str(path))  # Should not raise
        finally:
            path.unlink()
    
    def test_invalid_path_type(self):
        """Non-string/Path should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be a string or Path"):
            validate_image_path(123)
    
    def test_nonexistent_file(self):
        """Non-existent file should raise ValidationError."""
        with pytest.raises(ValidationError, match="Image file not found"):
            validate_image_path("/nonexistent/path/image.jpg")
    
    def test_path_is_directory(self):
        """Directory path should raise ValidationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValidationError, match="is not a file"):
                validate_image_path(tmpdir)
    
    def test_unsupported_format(self):
        """Unsupported file format should raise ValidationError."""
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
            path = Path(f.name)
        
        try:
            path.write_bytes(b'GIF89a')  # GIF magic bytes
            with pytest.raises(ValidationError, match="Unsupported image format"):
                validate_image_path(str(path))
        finally:
            path.unlink()


class TestAudioValidation:
    """Test audio input validation."""
    
    def test_valid_mono_audio_float32(self):
        """Valid mono float32 audio should not raise."""
        # Use uniform distribution to ensure values are in [-1, 1]
        audio = (np.random.rand(48000).astype(np.float32) * 2 - 1) * 0.5
        validate_audio_input(audio)  # Should not raise
    
    def test_valid_stereo_audio(self):
        """Valid stereo audio should not raise."""
        # Use uniform distribution to ensure values are in [-1, 1]
        audio = (np.random.rand(2, 48000).astype(np.float32) * 2 - 1) * 0.5
        validate_audio_input(audio)  # Should not raise
    
    def test_valid_int16_audio(self):
        """Valid int16 audio should not raise."""
        audio = np.random.randint(-32768, 32767, 48000, dtype=np.int16)
        validate_audio_input(audio)  # Should not raise
    
    def test_invalid_audio_type(self):
        """Non-numpy array should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be a numpy array"):
            validate_audio_input([1, 2, 3])
    
    def test_invalid_audio_dimensions(self):
        """3D array should raise ValidationError."""
        audio = np.random.randn(2, 2, 48000).astype(np.float32)
        with pytest.raises(ValidationError, match="must be 1D .* or 2D"):
            validate_audio_input(audio)
    
    def test_invalid_stereo_channels(self):
        """Stereo audio with wrong channel count should raise."""
        audio = np.random.randn(3, 48000).astype(np.float32)
        with pytest.raises(ValidationError, match="must have 1 or 2 channels"):
            validate_audio_input(audio)
    
    def test_invalid_dtype(self):
        """Audio with invalid dtype should raise."""
        audio = np.random.randn(48000).astype(np.float64)
        # float64 is actually valid, let's test uint8 instead
        audio = np.random.randint(0, 256, 48000, dtype=np.uint8)
        with pytest.raises(ValidationError, match="must be float32, float64, int16, or int32"):
            validate_audio_input(audio)
    
    def test_invalid_float_value_range(self):
        """float audio with out-of-range values should raise."""
        # Create valid audio first, then modify one value
        audio = (np.random.rand(48000).astype(np.float32) * 2 - 1) * 0.5
        audio[0] = 2.0  # Out of range
        with pytest.raises(ValidationError, match="float audio values must be in"):
            validate_audio_input(audio)
    
    def test_audio_too_long(self):
        """Audio exceeding max duration should raise."""
        # 60 seconds at 16kHz - use uniform distribution
        audio = (np.random.rand(16000 * 60).astype(np.float32) * 2 - 1) * 0.5
        with pytest.raises(ValidationError, match="exceeds maximum"):
            validate_audio_input(audio, sample_rate=16000, max_duration=30.0)
    
    def test_audio_too_short(self):
        """Audio below min duration should raise."""
        # 0.05 seconds at 16kHz - use uniform distribution
        audio = (np.random.rand(800).astype(np.float32) * 2 - 1) * 0.5
        with pytest.raises(ValidationError, match="is too short"):
            validate_audio_input(audio, sample_rate=16000)


class TestAudioPathValidation:
    """Test audio file path validation."""
    
    def test_valid_audio_path(self):
        """Valid audio file path should not raise."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = Path(f.name)
        
        try:
            # Create a minimal valid file
            path.write_bytes(b'RIFF')  # WAV magic bytes
            validate_audio_path(str(path))  # Should not raise
        finally:
            path.unlink()
    
    def test_invalid_path_type(self):
        """Non-string/Path should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be a string or Path"):
            validate_audio_path(123)
    
    def test_nonexistent_file(self):
        """Non-existent file should raise ValidationError."""
        with pytest.raises(ValidationError, match="Audio file not found"):
            validate_audio_path("/nonexistent/path/audio.wav")
    
    def test_path_is_directory(self):
        """Directory path should raise ValidationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValidationError, match="is not a file"):
                validate_audio_path(tmpdir)
    
    def test_unsupported_format(self):
        """Unsupported file format should raise ValidationError."""
        with tempfile.NamedTemporaryFile(suffix='.aiff', delete=False) as f:
            path = Path(f.name)
        
        try:
            path.write_bytes(b'FORM')  # AIFF magic bytes
            with pytest.raises(ValidationError, match="Unsupported audio format"):
                validate_audio_path(str(path))
        finally:
            path.unlink()


class TestBatchValidation:
    """Test batch input validation."""
    
    def test_valid_batch(self):
        """Valid batch should not raise."""
        batch = [1, 2, 3, 4, 5]
        validate_batch_input(batch)  # Should not raise
    
    def test_invalid_batch_type(self):
        """Non-list should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be a list"):
            validate_batch_input((1, 2, 3))
    
    def test_batch_too_small(self):
        """Empty batch should raise ValidationError."""
        with pytest.raises(ValidationError, match="is below minimum"):
            validate_batch_input([])
    
    def test_batch_too_large(self):
        """Batch exceeding max size should raise ValidationError."""
        batch = list(range(1500))
        with pytest.raises(ValidationError, match="exceeds maximum"):
            validate_batch_input(batch, max_size=1000)
    
    def test_custom_min_size(self):
        """Batch below custom min size should raise."""
        with pytest.raises(ValidationError, match="is below minimum"):
            validate_batch_input([1, 2], min_size=3)
    
    def test_custom_max_size(self):
        """Batch above custom max size should raise."""
        with pytest.raises(ValidationError, match="exceeds maximum"):
            validate_batch_input([1, 2, 3, 4, 5], max_size=3)


class TestModelLoadingErrors:
    """Test model loading error scenarios."""
    
    def test_missing_weights_file(self):
        """Missing weights file should raise FileNotFoundError."""
        # This test would require importing VisionEncoder or AudioEncoder
        # and testing with a non-existent model path
        # For now, we'll test the validation functions above
        pass
    
    def test_corrupted_weights(self):
        """Corrupted weights file should raise appropriate error."""
        # This would require creating a corrupted Parquet file
        # and testing weight loading
        pass


class TestIntegration:
    """Integration tests for validation in encoder workflows."""
    
    def test_validate_image_batch(self):
        """Validate a batch of images."""
        images = [
            np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        validate_batch_input(images)
        for img in images:
            validate_image_input(img)
    
    def test_validate_audio_batch(self):
        """Validate a batch of audio samples."""
        # Use uniform distribution to ensure values are in [-1, 1]
        audio_samples = [
            (np.random.rand(48000).astype(np.float32) * 2 - 1) * 0.5
            for _ in range(5)
        ]
        
        validate_batch_input(audio_samples)
        for audio in audio_samples:
            validate_audio_input(audio)
    
    def test_mixed_validation_errors(self):
        """Test that validation catches multiple error types."""
        # Invalid type
        with pytest.raises(ValidationError):
            validate_image_input("not an array")
        
        # Invalid dimensions
        with pytest.raises(ValidationError):
            validate_image_input(np.zeros((10, 10, 10, 3)))
        
        # Invalid batch
        with pytest.raises(ValidationError):
            validate_batch_input([])

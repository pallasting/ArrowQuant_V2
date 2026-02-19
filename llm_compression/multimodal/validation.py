"""
Input Validation Utilities for Multimodal Encoders

Provides validation functions for image and audio inputs with
descriptive error messages.
"""

from typing import Union, List, Tuple
import numpy as np
from pathlib import Path

from llm_compression.errors import ConfigurationError


class ValidationError(ConfigurationError):
    """Raised when input validation fails."""
    pass


def validate_image_input(
    image: np.ndarray,
    expected_size: int = 224,
    allow_grayscale: bool = False
) -> None:
    """
    Validate image input format and dimensions.
    
    Args:
        image: Input image array
        expected_size: Expected image size (height and width)
        allow_grayscale: Whether to allow grayscale images
        
    Raises:
        ValidationError: If image format is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValidationError(
            f"Image must be a numpy array, got {type(image).__name__}"
        )
    
    # Check dimensions
    if image.ndim not in [2, 3]:
        raise ValidationError(
            f"Image must be 2D (grayscale) or 3D (RGB), got {image.ndim}D array"
        )
    
    # Check grayscale
    if image.ndim == 2 and not allow_grayscale:
        raise ValidationError(
            "Grayscale images not supported. Please convert to RGB format."
        )
    
    # Check RGB channels
    if image.ndim == 3:
        if image.shape[2] not in [3, 4]:
            raise ValidationError(
                f"Image must have 3 (RGB) or 4 (RGBA) channels, got {image.shape[2]}"
            )
    
    # Check data type
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        raise ValidationError(
            f"Image must be uint8 or float, got {image.dtype}"
        )
    
    # Check value range
    if image.dtype == np.uint8:
        if image.min() < 0 or image.max() > 255:
            raise ValidationError(
                f"uint8 image values must be in [0, 255], got [{image.min()}, {image.max()}]"
            )
    elif image.dtype in [np.float32, np.float64]:
        if image.min() < 0.0 or image.max() > 1.0:
            # Allow slightly out of range due to floating point errors
            if image.min() < -0.01 or image.max() > 1.01:
                raise ValidationError(
                    f"float image values must be in [0.0, 1.0], got [{image.min():.3f}, {image.max():.3f}]"
                )


def validate_image_path(path: str) -> None:
    """
    Validate image file path.
    
    Args:
        path: Path to image file
        
    Raises:
        ValidationError: If path is invalid
    """
    if not isinstance(path, (str, Path)):
        raise ValidationError(
            f"Image path must be a string or Path, got {type(path).__name__}"
        )
    
    path = Path(path)
    
    if not path.exists():
        raise ValidationError(
            f"Image file not found: {path}"
        )
    
    if not path.is_file():
        raise ValidationError(
            f"Image path is not a file: {path}"
        )
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if path.suffix.lower() not in valid_extensions:
        raise ValidationError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {', '.join(valid_extensions)}"
        )


def validate_audio_input(
    audio: np.ndarray,
    sample_rate: int = 16000,
    max_duration: float = 30.0
) -> None:
    """
    Validate audio input format and properties.
    
    Args:
        audio: Input audio waveform
        sample_rate: Expected sample rate
        max_duration: Maximum audio duration in seconds
        
    Raises:
        ValidationError: If audio format is invalid
    """
    if not isinstance(audio, np.ndarray):
        raise ValidationError(
            f"Audio must be a numpy array, got {type(audio).__name__}"
        )
    
    # Check dimensions
    if audio.ndim not in [1, 2]:
        raise ValidationError(
            f"Audio must be 1D (mono) or 2D (stereo), got {audio.ndim}D array"
        )
    
    # Check stereo
    if audio.ndim == 2:
        if audio.shape[0] not in [1, 2]:
            raise ValidationError(
                f"Stereo audio must have 1 or 2 channels, got {audio.shape[0]}"
            )
    
    # Check data type
    if audio.dtype not in [np.float32, np.float64, np.int16, np.int32]:
        raise ValidationError(
            f"Audio must be float32, float64, int16, or int32, got {audio.dtype}"
        )
    
    # Check value range
    if audio.dtype in [np.float32, np.float64]:
        if audio.min() < -1.0 or audio.max() > 1.0:
            # Allow slightly out of range
            if audio.min() < -1.1 or audio.max() > 1.1:
                raise ValidationError(
                    f"float audio values must be in [-1.0, 1.0], got [{audio.min():.3f}, {audio.max():.3f}]"
                )
    
    # Check duration
    num_samples = audio.shape[-1] if audio.ndim == 2 else audio.shape[0]
    duration = num_samples / sample_rate
    
    if duration > max_duration:
        raise ValidationError(
            f"Audio duration ({duration:.2f}s) exceeds maximum ({max_duration}s). "
            f"Please trim or split the audio."
        )
    
    if duration < 0.1:
        raise ValidationError(
            f"Audio duration ({duration:.2f}s) is too short. Minimum duration is 0.1s."
        )


def validate_audio_path(path: str) -> None:
    """
    Validate audio file path.
    
    Args:
        path: Path to audio file
        
    Raises:
        ValidationError: If path is invalid
    """
    if not isinstance(path, (str, Path)):
        raise ValidationError(
            f"Audio path must be a string or Path, got {type(path).__name__}"
        )
    
    path = Path(path)
    
    if not path.exists():
        raise ValidationError(
            f"Audio file not found: {path}"
        )
    
    if not path.is_file():
        raise ValidationError(
            f"Audio path is not a file: {path}"
        )
    
    # Check file extension
    valid_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    if path.suffix.lower() not in valid_extensions:
        raise ValidationError(
            f"Unsupported audio format: {path.suffix}. "
            f"Supported formats: {', '.join(valid_extensions)}"
        )


def validate_batch_input(
    batch: List,
    min_size: int = 1,
    max_size: int = 1000
) -> None:
    """
    Validate batch input.
    
    Args:
        batch: List of inputs
        min_size: Minimum batch size
        max_size: Maximum batch size
        
    Raises:
        ValidationError: If batch is invalid
    """
    if not isinstance(batch, list):
        raise ValidationError(
            f"Batch must be a list, got {type(batch).__name__}"
        )
    
    if len(batch) < min_size:
        raise ValidationError(
            f"Batch size ({len(batch)}) is below minimum ({min_size})"
        )
    
    if len(batch) > max_size:
        raise ValidationError(
            f"Batch size ({len(batch)}) exceeds maximum ({max_size}). "
            f"Please process in smaller batches."
        )

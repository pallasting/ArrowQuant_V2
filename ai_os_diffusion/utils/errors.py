"""
Error Types for AI-OS Diffusion System

Defines comprehensive error hierarchy for error handling and degradation strategies.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class DiffusionError(Exception):
    """
    Base class for all diffusion-related errors
    
    Attributes:
        message: Error message
        context: Additional context information
        timestamp: When the error occurred
        error_type: Type of error for logging/monitoring
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.now()
        self.error_type = self.__class__.__name__
        self.original_exception = original_exception
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging"""
        return {
            'error_type': self.error_type,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'original_exception': str(self.original_exception) if self.original_exception else None
        }
    
    def __str__(self) -> str:
        context_str = f", context={self.context}" if self.context else ""
        return f"{self.error_type}: {self.message}{context_str}"


class ModelLoadError(DiffusionError):
    """
    Model loading failed
    
    Raised when:
    - Model file not found
    - Model format invalid
    - Quantization failed
    - Weight loading failed
    """
    pass


class InferenceError(DiffusionError):
    """
    Inference execution failed
    
    Raised when:
    - Forward pass failed
    - Sampling failed
    - Device error (GPU OOM, etc.)
    """
    pass


class StorageError(DiffusionError):
    """
    Storage operation failed
    
    Raised when:
    - Cannot write to Arrow table
    - Cannot read from Arrow table
    - File system errors
    """
    pass


class EvolutionError(DiffusionError):
    """
    Evolution operation failed
    
    Raised when:
    - LoRA training failed
    - ControlNet loading failed
    - Uncertainty estimation failed
    """
    pass


class ConfigurationError(DiffusionError):
    """
    Configuration invalid or missing
    
    Raised when:
    - Required configuration missing
    - Configuration value invalid
    """
    pass


class GPUResourceError(DiffusionError):
    """
    GPU resource unavailable or insufficient
    
    Raised when:
    - GPU out of memory (OOM)
    - CUDA not available
    """
    pass

"""
Error Types for LLM Compression System

Defines comprehensive error hierarchy for error handling and degradation strategies.

Feature: llm-compression-integration
Requirements: 13.1, 13.2, 13.3, 13.4, 13.7
"""

from typing import Optional, Dict, Any
from datetime import datetime


class CompressionError(Exception):
    """
    Base class for all compression-related errors
    
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


class LLMAPIError(CompressionError):
    """
    LLM API call failed
    
    Raised when:
    - API returns non-200 status code
    - API response is malformed
    - Network connection fails
    - Authentication fails
    
    Context should include:
    - endpoint: API endpoint URL
    - status_code: HTTP status code (if available)
    - response_body: Error response body
    - request_id: Request ID for tracking
    """
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        request_id: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        context = {
            'endpoint': endpoint,
            'status_code': status_code,
            'response_body': response_body,
            'request_id': request_id
        }
        # Remove None values
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context, original_exception)


class LLMTimeoutError(CompressionError):
    """
    LLM request timed out
    
    Raised when:
    - Request exceeds timeout duration
    - Connection hangs
    
    Context should include:
    - endpoint: API endpoint URL
    - timeout_seconds: Configured timeout
    - elapsed_seconds: Actual elapsed time
    """
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
        original_exception: Optional[Exception] = None
    ):
        context = {
            'endpoint': endpoint,
            'timeout_seconds': timeout_seconds,
            'elapsed_seconds': elapsed_seconds
        }
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context, original_exception)


class ReconstructionError(CompressionError):
    """
    Memory reconstruction failed
    
    Raised when:
    - Cannot expand summary
    - Diff application fails
    - Decompression fails
    - Invalid compressed data format
    
    Context should include:
    - memory_id: ID of memory being reconstructed
    - summary_hash: Summary hash
    - stage: Which stage failed (lookup/expand/diff/decompress)
    - partial_result: Partial reconstruction if available
    """
    
    def __init__(
        self,
        message: str,
        memory_id: Optional[str] = None,
        summary_hash: Optional[str] = None,
        stage: Optional[str] = None,
        partial_result: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        context = {
            'memory_id': memory_id,
            'summary_hash': summary_hash,
            'stage': stage,
            'partial_result': partial_result[:100] if partial_result else None  # Truncate
        }
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context, original_exception)


class QualityError(CompressionError):
    """
    Compression/reconstruction quality below threshold
    
    Raised when:
    - Semantic similarity < threshold
    - Entity accuracy < threshold
    - Reconstruction confidence < threshold
    
    Context should include:
    - memory_id: ID of memory
    - quality_score: Actual quality score
    - threshold: Required threshold
    - metric_type: Which metric failed (similarity/entity/confidence)
    - details: Detailed quality metrics
    """
    
    def __init__(
        self,
        message: str,
        memory_id: Optional[str] = None,
        quality_score: Optional[float] = None,
        threshold: Optional[float] = None,
        metric_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        context = {
            'memory_id': memory_id,
            'quality_score': quality_score,
            'threshold': threshold,
            'metric_type': metric_type,
            'details': details
        }
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context, original_exception)


class StorageError(CompressionError):
    """
    Storage operation failed
    
    Raised when:
    - Cannot write to Arrow table
    - Cannot read from Arrow table
    - File system errors
    - Disk full
    - Permission denied
    
    Context should include:
    - operation: Operation type (read/write/delete)
    - path: File path
    - memory_id: Memory ID (if applicable)
    - category: Memory category
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        path: Optional[str] = None,
        memory_id: Optional[str] = None,
        category: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        context = {
            'operation': operation,
            'path': path,
            'memory_id': memory_id,
            'category': category
        }
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context, original_exception)


class GPUResourceError(CompressionError):
    """
    GPU resource unavailable or insufficient
    
    Raised when:
    - GPU out of memory (OOM)
    - CUDA not available
    - GPU device not found
    - GPU driver error
    
    Context should include:
    - gpu_id: GPU device ID
    - required_memory_mb: Required memory in MB
    - available_memory_mb: Available memory in MB
    - cuda_available: Whether CUDA is available
    - error_code: GPU error code
    """
    
    def __init__(
        self,
        message: str,
        gpu_id: Optional[int] = None,
        required_memory_mb: Optional[int] = None,
        available_memory_mb: Optional[int] = None,
        cuda_available: Optional[bool] = None,
        error_code: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        context = {
            'gpu_id': gpu_id,
            'required_memory_mb': required_memory_mb,
            'available_memory_mb': available_memory_mb,
            'cuda_available': cuda_available,
            'error_code': error_code
        }
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context, original_exception)


class ConfigurationError(CompressionError):
    """
    Configuration invalid or missing
    
    Raised when:
    - Required configuration missing
    - Configuration value invalid
    - Configuration file not found
    - Configuration parsing failed
    
    Context should include:
    - config_key: Configuration key
    - config_value: Invalid value
    - expected_type: Expected type
    - config_file: Configuration file path
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        config_file: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        context = {
            'config_key': config_key,
            'config_value': str(config_value) if config_value is not None else None,
            'expected_type': expected_type,
            'config_file': config_file
        }
        context = {k: v for k, v in context.items() if v is not None}
        super().__init__(message, context, original_exception)


# Convenience functions for error creation

def create_api_error(
    message: str,
    endpoint: str,
    status_code: Optional[int] = None,
    response_body: Optional[str] = None
) -> LLMAPIError:
    """Create LLM API error with common context"""
    return LLMAPIError(
        message=message,
        endpoint=endpoint,
        status_code=status_code,
        response_body=response_body
    )


def create_timeout_error(
    endpoint: str,
    timeout_seconds: float,
    elapsed_seconds: Optional[float] = None
) -> LLMTimeoutError:
    """Create timeout error with common context"""
    message = f"Request to {endpoint} timed out after {timeout_seconds}s"
    return LLMTimeoutError(
        message=message,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
        elapsed_seconds=elapsed_seconds
    )


def create_reconstruction_error(
    message: str,
    memory_id: str,
    stage: str,
    partial_result: Optional[str] = None
) -> ReconstructionError:
    """Create reconstruction error with common context"""
    return ReconstructionError(
        message=message,
        memory_id=memory_id,
        stage=stage,
        partial_result=partial_result
    )


def create_quality_error(
    message: str,
    memory_id: str,
    quality_score: float,
    threshold: float,
    metric_type: str
) -> QualityError:
    """Create quality error with common context"""
    return QualityError(
        message=message,
        memory_id=memory_id,
        quality_score=quality_score,
        threshold=threshold,
        metric_type=metric_type
    )


def create_storage_error(
    message: str,
    operation: str,
    path: str,
    original_exception: Optional[Exception] = None
) -> StorageError:
    """Create storage error with common context"""
    return StorageError(
        message=message,
        operation=operation,
        path=path,
        original_exception=original_exception
    )


def create_gpu_error(
    message: str,
    required_memory_mb: Optional[int] = None,
    available_memory_mb: Optional[int] = None
) -> GPUResourceError:
    """Create GPU resource error with common context"""
    return GPUResourceError(
        message=message,
        required_memory_mb=required_memory_mb,
        available_memory_mb=available_memory_mb
    )

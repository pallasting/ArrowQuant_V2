"""
GPU Resource Fallback Handler

Handles GPU out-of-memory errors and automatically falls back to:
1. CPU inference
2. Quantized models (INT8/INT4)
3. Cloud API

Feature: llm-compression-integration
Requirements: 13.5
Property 33: GPU Resource Fallback
"""

from typing import Optional, Dict, Any, Callable
from llm_compression.logger import logger
from llm_compression.errors import GPUResourceError


class GPUFallbackHandler:
    """
    Handles GPU resource errors and automatic fallback
    
    Fallback strategy:
    1. Detect GPU OOM error
    2. Try CPU inference
    3. Try quantized model (INT8/INT4)
    4. Fall back to cloud API
    
    Requirements: 13.5
    Property 33: GPU Resource Fallback
    """
    
    def __init__(
        self,
        enable_cpu_fallback: bool = True,
        enable_quantization_fallback: bool = True,
        enable_cloud_fallback: bool = True
    ):
        """
        Initialize GPU fallback handler
        
        Args:
            enable_cpu_fallback: Whether to try CPU inference
            enable_quantization_fallback: Whether to try quantized models
            enable_cloud_fallback: Whether to fall back to cloud API
        """
        self.enable_cpu_fallback = enable_cpu_fallback
        self.enable_quantization_fallback = enable_quantization_fallback
        self.enable_cloud_fallback = enable_cloud_fallback
        
        # Track fallback statistics
        self.fallback_stats = {
            'gpu_oom_count': 0,
            'cpu_fallback_success': 0,
            'cpu_fallback_failure': 0,
            'quantization_fallback_success': 0,
            'quantization_fallback_failure': 0,
            'cloud_fallback_success': 0,
            'cloud_fallback_failure': 0
        }
        
        # Check GPU availability
        self.cuda_available = self._check_cuda_availability()
        
        logger.info(
            f"GPUFallbackHandler initialized: "
            f"cuda_available={self.cuda_available}, "
            f"cpu_fallback={enable_cpu_fallback}, "
            f"quantization_fallback={enable_quantization_fallback}, "
            f"cloud_fallback={enable_cloud_fallback}"
        )
    
    def _check_cuda_availability(self) -> bool:
        """
        Check if CUDA is available
        
        Returns:
            bool: True if CUDA is available
        """
        try:
            import torch
            available = torch.cuda.is_available()
            if available:
                device_count = torch.cuda.device_count()
                logger.info(f"CUDA available: {device_count} device(s)")
            else:
                logger.info("CUDA not available")
            return available
        except ImportError:
            logger.warning("PyTorch not installed, CUDA check skipped")
            return False
        except Exception as e:
            logger.error(f"Error checking CUDA availability: {e}")
            return False
    
    def is_gpu_oom_error(self, exception: Exception) -> bool:
        """
        Check if exception is a GPU out-of-memory error
        
        Args:
            exception: Exception to check
            
        Returns:
            bool: True if GPU OOM error
        """
        # Check for PyTorch CUDA OOM
        try:
            import torch
            if isinstance(exception, torch.cuda.OutOfMemoryError):
                return True
        except ImportError:
            pass
        
        # Check for common OOM error messages (case-insensitive)
        error_msg = str(exception).lower()
        oom_indicators = [
            'out of memory',
            'outofmemory',
            'oom',
            'cuda',
            'gpu memory',
            'memory allocation failed'
        ]
        
        return any(indicator in error_msg for indicator in oom_indicators)
    
    async def handle_gpu_oom(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Handle GPU OOM error with automatic fallback
        
        Algorithm:
        1. Try operation on GPU
        2. If GPU OOM, try CPU
        3. If CPU fails, try quantized model
        4. If quantized fails, fall back to cloud API
        
        Args:
            operation: Async operation to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result from operation
            
        Raises:
            GPUResourceError: If all fallback attempts fail
            
        Requirements: 13.5
        Property 33: GPU Resource Fallback
        """
        # Try original operation (GPU)
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            if not self.is_gpu_oom_error(e):
                # Not a GPU OOM error, re-raise
                raise
            
            # GPU OOM detected
            self.fallback_stats['gpu_oom_count'] += 1
            logger.warning(f"GPU OOM detected: {e}")
            
            # Try CPU fallback
            if self.enable_cpu_fallback:
                try:
                    logger.info("Attempting CPU fallback")
                    result = await self._try_cpu_fallback(operation, *args, **kwargs)
                    self.fallback_stats['cpu_fallback_success'] += 1
                    logger.info("CPU fallback successful")
                    return result
                except Exception as cpu_error:
                    self.fallback_stats['cpu_fallback_failure'] += 1
                    logger.warning(f"CPU fallback failed: {cpu_error}")
            
            # Try quantization fallback
            if self.enable_quantization_fallback:
                try:
                    logger.info("Attempting quantization fallback")
                    result = await self._try_quantization_fallback(operation, *args, **kwargs)
                    self.fallback_stats['quantization_fallback_success'] += 1
                    logger.info("Quantization fallback successful")
                    return result
                except Exception as quant_error:
                    self.fallback_stats['quantization_fallback_failure'] += 1
                    logger.warning(f"Quantization fallback failed: {quant_error}")
            
            # Try cloud fallback
            if self.enable_cloud_fallback:
                try:
                    logger.info("Attempting cloud API fallback")
                    result = await self._try_cloud_fallback(operation, *args, **kwargs)
                    self.fallback_stats['cloud_fallback_success'] += 1
                    logger.info("Cloud API fallback successful")
                    return result
                except Exception as cloud_error:
                    self.fallback_stats['cloud_fallback_failure'] += 1
                    logger.error(f"Cloud API fallback failed: {cloud_error}")
            
            # All fallbacks failed
            raise GPUResourceError(
                message="GPU OOM and all fallback attempts failed",
                original_exception=e
            )
    
    async def _try_cpu_fallback(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Try operation on CPU
        
        Args:
            operation: Operation to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from operation
        """
        # Add device='cpu' to kwargs if not present
        if 'device' not in kwargs:
            kwargs['device'] = 'cpu'
        
        # Clear GPU cache if PyTorch is available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
        except ImportError:
            pass
        
        return await operation(*args, **kwargs)
    
    async def _try_quantization_fallback(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Try operation with quantized model
        
        Args:
            operation: Operation to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from operation
        """
        # Add quantization parameters to kwargs
        if 'quantization' not in kwargs:
            kwargs['quantization'] = 'int8'  # Try INT8 first
        
        # Clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared for quantization")
        except ImportError:
            pass
        
        return await operation(*args, **kwargs)
    
    async def _try_cloud_fallback(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Try operation with cloud API
        
        Args:
            operation: Operation to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from operation
        """
        # Add use_cloud=True to kwargs
        if 'use_cloud' not in kwargs:
            kwargs['use_cloud'] = True
        
        return await operation(*args, **kwargs)
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """
        Get GPU memory information
        
        Returns:
            Dict: GPU memory info including:
                - cuda_available: Whether CUDA is available
                - device_count: Number of GPU devices
                - memory_allocated: Allocated memory per device (MB)
                - memory_reserved: Reserved memory per device (MB)
                - memory_free: Free memory per device (MB)
        """
        info = {
            'cuda_available': self.cuda_available,
            'device_count': 0,
            'devices': []
        }
        
        if not self.cuda_available:
            return info
        
        try:
            import torch
            
            device_count = torch.cuda.device_count()
            info['device_count'] = device_count
            
            for i in range(device_count):
                device_info = {
                    'device_id': i,
                    'device_name': torch.cuda.get_device_name(i),
                    'memory_allocated_mb': torch.cuda.memory_allocated(i) / 1024 / 1024,
                    'memory_reserved_mb': torch.cuda.memory_reserved(i) / 1024 / 1024,
                    'memory_total_mb': torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
                }
                device_info['memory_free_mb'] = (
                    device_info['memory_total_mb'] - 
                    device_info['memory_reserved_mb']
                )
                info['devices'].append(device_info)
            
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
        
        return info
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """
        Get fallback statistics
        
        Returns:
            Dict: Statistics including OOM count and fallback success/failure rates
        """
        total_oom = self.fallback_stats['gpu_oom_count']
        
        stats = {
            **self.fallback_stats,
            'total_oom_events': total_oom
        }
        
        # Calculate success rates
        if total_oom > 0:
            stats['cpu_fallback_success_rate'] = (
                self.fallback_stats['cpu_fallback_success'] / total_oom
            )
            stats['quantization_fallback_success_rate'] = (
                self.fallback_stats['quantization_fallback_success'] / total_oom
            )
            stats['cloud_fallback_success_rate'] = (
                self.fallback_stats['cloud_fallback_success'] / total_oom
            )
        
        return stats
    
    def reset_stats(self):
        """Reset fallback statistics"""
        for key in self.fallback_stats:
            self.fallback_stats[key] = 0
        logger.info("GPU fallback statistics reset")

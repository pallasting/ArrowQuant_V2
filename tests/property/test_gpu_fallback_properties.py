"""
Property-Based Tests for GPU Fallback

Tests Property 33: GPU Resource Fallback

Feature: llm-compression-integration
Requirements: 13.5
Property 33: GPU Resource Fallback
"""

import pytest
from hypothesis import given, settings, strategies as st
from unittest.mock import Mock, AsyncMock, patch

from llm_compression.gpu_fallback import GPUFallbackHandler
from llm_compression.errors import GPUResourceError


# Test strategies
error_message_strategy = st.sampled_from([
    "CUDA out of memory",
    "GPU memory allocation failed",
    "torch.cuda.OutOfMemoryError",
    "OOM when allocating tensor",
    "out of memory"
])


class MockGPUOOMError(Exception):
    """Mock GPU OOM error"""
    pass


class TestGPUFallbackProperties:
    """
    Property-based tests for GPU fallback
    
    Feature: llm-compression-integration, Property 33: GPU Resource Fallback
    """
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=3000)
    @given(error_msg=error_message_strategy)
    async def test_property_33_gpu_oom_detection(
        self,
        error_msg
    ):
        """
        Feature: llm-compression-integration, Property 33: GPU Resource Fallback
        
        Test: *For any* GPU OOM error message, handler should correctly detect it
        
        Validates: Requirements 13.5
        """
        handler = GPUFallbackHandler()
        
        # Create exception with OOM message
        exception = MockGPUOOMError(error_msg)
        
        # Verify: Should detect as GPU OOM
        assert handler.is_gpu_oom_error(exception) is True
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=5000)
    @given(error_msg=error_message_strategy)
    async def test_property_33_cpu_fallback(
        self,
        error_msg
    ):
        """
        Feature: llm-compression-integration, Property 33: GPU Resource Fallback
        
        Test: *For any* GPU OOM error, handler should try CPU fallback
        
        Validates: Requirements 13.5
        """
        handler = GPUFallbackHandler(
            enable_cpu_fallback=True,
            enable_quantization_fallback=False,
            enable_cloud_fallback=False
        )
        
        # Mock operation that fails on GPU, succeeds on CPU
        call_count = {'count': 0}
        
        async def mock_operation(*args, **kwargs):
            call_count['count'] += 1
            if call_count['count'] == 1:
                # First call (GPU) fails
                raise MockGPUOOMError(error_msg)
            else:
                # Second call (CPU) succeeds - check device parameter
                if 'device' in kwargs:
                    assert kwargs['device'] == 'cpu'
                return "success"
        
        # Execute
        result = await handler.handle_gpu_oom(mock_operation)
        
        # Verify: Should succeed with CPU fallback
        assert result == "success"
        assert call_count['count'] == 2  # GPU attempt + CPU fallback
        
        # Verify stats
        stats = handler.get_fallback_stats()
        assert stats['gpu_oom_count'] == 1
        assert stats['cpu_fallback_success'] == 1
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=5000)
    @given(error_msg=error_message_strategy)
    async def test_property_33_quantization_fallback(
        self,
        error_msg
    ):
        """
        Feature: llm-compression-integration, Property 33: GPU Resource Fallback
        
        Test: *For any* GPU OOM error, if CPU fails, should try quantization
        
        Validates: Requirements 13.5
        """
        handler = GPUFallbackHandler(
            enable_cpu_fallback=True,
            enable_quantization_fallback=True,
            enable_cloud_fallback=False
        )
        
        # Mock operation that fails on GPU and CPU, succeeds with quantization
        call_count = {'count': 0}
        
        async def mock_operation(*args, **kwargs):
            call_count['count'] += 1
            if call_count['count'] == 1:
                # First call (GPU) fails
                raise MockGPUOOMError(error_msg)
            elif call_count['count'] == 2:
                # Second call (CPU) fails
                raise Exception("CPU also out of memory")
            else:
                # Third call (quantization) succeeds - check quantization parameter
                if 'quantization' in kwargs:
                    assert kwargs['quantization'] == 'int8'
                return "success_quantized"
        
        # Execute
        result = await handler.handle_gpu_oom(mock_operation)
        
        # Verify: Should succeed with quantization fallback
        assert result == "success_quantized"
        assert call_count['count'] == 3  # GPU + CPU + quantization
        
        # Verify stats
        stats = handler.get_fallback_stats()
        assert stats['gpu_oom_count'] == 1
        assert stats['cpu_fallback_failure'] == 1
        assert stats['quantization_fallback_success'] == 1
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=5000)
    @given(error_msg=error_message_strategy)
    async def test_property_33_cloud_fallback(
        self,
        error_msg
    ):
        """
        Feature: llm-compression-integration, Property 33: GPU Resource Fallback
        
        Test: *For any* GPU OOM error, if all local methods fail,
        should fall back to cloud API
        
        Validates: Requirements 13.5
        """
        handler = GPUFallbackHandler(
            enable_cpu_fallback=True,
            enable_quantization_fallback=True,
            enable_cloud_fallback=True
        )
        
        # Mock operation that fails locally, succeeds on cloud
        call_count = {'count': 0}
        
        async def mock_operation(*args, **kwargs):
            call_count['count'] += 1
            if call_count['count'] <= 3:
                # GPU, CPU, quantization all fail
                raise MockGPUOOMError(error_msg)
            else:
                # Cloud succeeds - check use_cloud parameter
                if 'use_cloud' in kwargs:
                    assert kwargs['use_cloud'] is True
                return "success_cloud"
        
        # Execute
        result = await handler.handle_gpu_oom(mock_operation)
        
        # Verify: Should succeed with cloud fallback
        assert result == "success_cloud"
        assert call_count['count'] == 4  # GPU + CPU + quantization + cloud
        
        # Verify stats
        stats = handler.get_fallback_stats()
        assert stats['gpu_oom_count'] == 1
        assert stats['cloud_fallback_success'] == 1
    
    @pytest.mark.asyncio
    @settings(max_examples=20, deadline=5000)
    @given(error_msg=error_message_strategy)
    async def test_property_33_all_fallbacks_fail(
        self,
        error_msg
    ):
        """
        Feature: llm-compression-integration, Property 33: GPU Resource Fallback
        
        Test: *For any* GPU OOM error, if all fallbacks fail,
        should raise GPUResourceError
        
        Validates: Requirements 13.5
        """
        handler = GPUFallbackHandler(
            enable_cpu_fallback=True,
            enable_quantization_fallback=True,
            enable_cloud_fallback=True
        )
        
        # Mock operation that always fails
        async def mock_operation(*args, **kwargs):
            raise MockGPUOOMError(error_msg)
        
        # Execute and verify: Should raise GPUResourceError
        with pytest.raises(GPUResourceError) as exc_info:
            await handler.handle_gpu_oom(mock_operation)
        
        # Verify error message
        assert "all fallback attempts failed" in str(exc_info.value).lower()
        
        # Verify stats
        stats = handler.get_fallback_stats()
        assert stats['gpu_oom_count'] == 1
        assert stats['cpu_fallback_failure'] == 1
        assert stats['quantization_fallback_failure'] == 1
        assert stats['cloud_fallback_failure'] == 1
    
    @pytest.mark.asyncio
    async def test_property_33_non_oom_error_passthrough(
        self
    ):
        """
        Feature: llm-compression-integration, Property 33: GPU Resource Fallback
        
        Test: *For any* non-OOM error, handler should pass it through
        without attempting fallback
        
        Validates: Requirements 13.5
        """
        handler = GPUFallbackHandler()
        
        # Mock operation that raises non-OOM error
        async def mock_operation(*args, **kwargs):
            raise ValueError("Invalid input parameter")
        
        # Execute and verify: Should raise original error
        with pytest.raises(ValueError) as exc_info:
            await handler.handle_gpu_oom(mock_operation)
        
        assert "Invalid input parameter" in str(exc_info.value)
        
        # Verify: No OOM recorded
        stats = handler.get_fallback_stats()
        assert stats['gpu_oom_count'] == 0


class TestGPUMemoryInfo:
    """Test GPU memory information retrieval"""
    
    def test_gpu_memory_info_structure(self):
        """
        Test: GPU memory info should have correct structure
        
        Validates: Requirements 13.5
        """
        handler = GPUFallbackHandler()
        
        # Get memory info
        info = handler.get_gpu_memory_info()
        
        # Verify structure
        assert 'cuda_available' in info
        assert 'device_count' in info
        assert 'devices' in info
        assert isinstance(info['cuda_available'], bool)
        assert isinstance(info['device_count'], int)
        assert isinstance(info['devices'], list)
    
    def test_fallback_stats_structure(self):
        """
        Test: Fallback stats should have correct structure
        
        Validates: Requirements 13.5
        """
        handler = GPUFallbackHandler()
        
        # Get stats
        stats = handler.get_fallback_stats()
        
        # Verify structure
        assert 'gpu_oom_count' in stats
        assert 'cpu_fallback_success' in stats
        assert 'cpu_fallback_failure' in stats
        assert 'quantization_fallback_success' in stats
        assert 'quantization_fallback_failure' in stats
        assert 'cloud_fallback_success' in stats
        assert 'cloud_fallback_failure' in stats
        assert 'total_oom_events' in stats

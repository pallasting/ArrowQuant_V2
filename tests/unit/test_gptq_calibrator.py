"""
Unit tests for GPTQCalibrator.

Tests the GPTQ calibration algorithm for optimal quantization parameters:
- Hessian computation and inversion
- Calibration dataset preparation
- Layer-wise calibration with error compensation
- Configuration validation
- Cache management
- Error handling

Requirements: 2.1, 2.8, 2.9, 9.3, 12.1, Task 16
"""

import pytest
import torch
import numpy as np

from llm_compression.inference.gptq_calibrator import (
    GPTQCalibrator,
    GPTQCalibrationConfig,
)
from llm_compression.errors import ConfigurationError


class TestGPTQCalibrationConfig:
    """Test GPTQCalibrationConfig validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GPTQCalibrationConfig()
        
        assert config.num_samples == 128
        assert config.block_size == 128
        assert config.dampening_factor == 0.01
        assert config.percdamp == 0.01
        assert config.use_cache is True
        assert config.device == 'cpu'
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = GPTQCalibrationConfig(
            num_samples=256,
            block_size=64,
            dampening_factor=0.02,
            percdamp=0.02,
            use_cache=False,
            device='cuda'
        )
        
        assert config.num_samples == 256
        assert config.block_size == 64
        assert config.dampening_factor == 0.02
        assert config.percdamp == 0.02
        assert config.use_cache is False
        assert config.device == 'cuda'
    
    def test_invalid_num_samples(self):
        """Test that invalid num_samples raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            GPTQCalibrationConfig(num_samples=0)
        
        assert 'num_samples must be >= 1' in str(exc_info.value)
    
    def test_invalid_block_size(self):
        """Test that invalid block_size raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            GPTQCalibrationConfig(block_size=-1)
        
        assert 'block_size must be >= 1' in str(exc_info.value)
    
    def test_invalid_dampening_factor(self):
        """Test that invalid dampening_factor raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            GPTQCalibrationConfig(dampening_factor=1.5)
        
        assert 'dampening_factor must be in (0, 1)' in str(exc_info.value)
    
    def test_zero_dampening_factor(self):
        """Test that zero dampening_factor raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            GPTQCalibrationConfig(dampening_factor=0.0)
        
        assert 'dampening_factor must be in (0, 1)' in str(exc_info.value)


class TestGPTQCalibratorBasic:
    """Test basic GPTQCalibrator functionality."""
    
    def test_initialization(self):
        """Test GPTQCalibrator initialization."""
        config = GPTQCalibrationConfig(num_samples=128)
        calibrator = GPTQCalibrator(config)
        
        assert calibrator.config == config
        assert calibrator.device == torch.device('cpu')
        assert calibrator.get_cache_size() == 0
    
    def test_initialization_with_cuda(self):
        """Test initialization with CUDA device."""
        config = GPTQCalibrationConfig(device='cuda')
        calibrator = GPTQCalibrator(config)
        
        # Should not fail even if CUDA not available
        assert calibrator.config.device == 'cuda'
    
    def test_cache_management(self):
        """Test cache management."""
        config = GPTQCalibrationConfig(use_cache=True)
        calibrator = GPTQCalibrator(config)
        
        assert calibrator.get_cache_size() == 0
        
        # Clear empty cache
        calibrator.clear_cache()
        assert calibrator.get_cache_size() == 0


class TestHessianComputation:
    """Test Hessian matrix computation."""
    
    def test_compute_hessian_2d(self):
        """Test Hessian computation from 2D data."""
        config = GPTQCalibrationConfig()
        calibrator = GPTQCalibrator(config)
        
        # Create calibration data [num_samples, in_features]
        torch.manual_seed(42)
        calibration_data = torch.randn(100, 64)
        
        H = calibrator.compute_hessian(calibration_data)
        
        # Check shape
        assert H.shape == (64, 64)
        
        # Check symmetry
        assert torch.allclose(H, H.t(), atol=1e-5)
        
        # Check positive diagonal (after dampening)
        assert torch.all(torch.diag(H) > 0)
    
    def test_compute_hessian_3d(self):
        """Test Hessian computation from 3D data."""
        config = GPTQCalibrationConfig()
        calibrator = GPTQCalibrator(config)
        
        # Create calibration data [batch, seq_len, in_features]
        torch.manual_seed(42)
        calibration_data = torch.randn(16, 32, 64)
        
        H = calibrator.compute_hessian(calibration_data)
        
        # Check shape
        assert H.shape == (64, 64)
        
        # Check symmetry
        assert torch.allclose(H, H.t(), atol=1e-5)
    
    def test_hessian_dampening(self):
        """Test that dampening is applied correctly."""
        config = GPTQCalibrationConfig(percdamp=0.1)
        calibrator = GPTQCalibrator(config)
        
        torch.manual_seed(42)
        calibration_data = torch.randn(100, 64)
        
        H = calibrator.compute_hessian(calibration_data)
        
        # Diagonal should be positive (dampening ensures this)
        assert torch.all(torch.diag(H) > 0)
    
    def test_hessian_caching(self):
        """Test Hessian caching."""
        config = GPTQCalibrationConfig(use_cache=True)
        calibrator = GPTQCalibrator(config)
        
        torch.manual_seed(42)
        calibration_data = torch.randn(100, 64)
        
        # First computation
        H1 = calibrator.compute_hessian(calibration_data, layer_name='layer1')
        assert calibrator.get_cache_size() == 1
        
        # Second computation (should use cache)
        H2 = calibrator.compute_hessian(calibration_data, layer_name='layer1')
        assert calibrator.get_cache_size() == 1
        
        # Should be identical (cached)
        assert torch.allclose(H1, H2)
    
    def test_hessian_no_caching(self):
        """Test Hessian without caching."""
        config = GPTQCalibrationConfig(use_cache=False)
        calibrator = GPTQCalibrator(config)
        
        torch.manual_seed(42)
        calibration_data = torch.randn(100, 64)
        
        # Compute multiple times
        H1 = calibrator.compute_hessian(calibration_data, layer_name='layer1')
        H2 = calibrator.compute_hessian(calibration_data, layer_name='layer1')
        
        # Cache should remain empty
        assert calibrator.get_cache_size() == 0
    
    def test_hessian_different_layers(self):
        """Test caching for different layers."""
        config = GPTQCalibrationConfig(use_cache=True)
        calibrator = GPTQCalibrator(config)
        
        torch.manual_seed(42)
        calibration_data1 = torch.randn(100, 64)
        calibration_data2 = torch.randn(100, 64)
        
        H1 = calibrator.compute_hessian(calibration_data1, layer_name='layer1')
        H2 = calibrator.compute_hessian(calibration_data2, layer_name='layer2')
        
        # Should cache both
        assert calibrator.get_cache_size() == 2
        
        # Should be different
        assert not torch.allclose(H1, H2)


class TestHessianInverse:
    """Test Hessian inverse computation."""
    
    def test_compute_hessian_inverse(self):
        """Test Hessian inverse computation."""
        config = GPTQCalibrationConfig()
        calibrator = GPTQCalibrator(config)
        
        # Create well-conditioned Hessian
        torch.manual_seed(42)
        calibration_data = torch.randn(100, 64)
        H = calibrator.compute_hessian(calibration_data)
        
        H_inv = calibrator.compute_hessian_inverse(H)
        
        assert H_inv is not None
        assert H_inv.shape == H.shape
        
        # Check that H * H_inv ≈ I
        I = torch.eye(64)
        product = H @ H_inv
        assert torch.allclose(product, I, atol=1e-3)
    
    def test_hessian_inverse_singular_matrix(self):
        """Test Hessian inverse with singular matrix."""
        config = GPTQCalibrationConfig()
        calibrator = GPTQCalibrator(config)
        
        # Create singular matrix (rank deficient)
        H = torch.zeros(64, 64)
        H[0, 0] = 1.0
        
        H_inv = calibrator.compute_hessian_inverse(H)
        
        # Should return None for singular matrix
        assert H_inv is None
    
    def test_hessian_inverse_ill_conditioned(self):
        """Test Hessian inverse with ill-conditioned matrix."""
        config = GPTQCalibrationConfig()
        calibrator = GPTQCalibrator(config)
        
        # Create ill-conditioned matrix
        H = torch.eye(64)
        H[0, 0] = 1e-10  # Very small eigenvalue
        
        # May fail or succeed depending on numerical precision
        H_inv = calibrator.compute_hessian_inverse(H)
        
        # If it succeeds, check shape
        if H_inv is not None:
            assert H_inv.shape == H.shape


class TestQuantizationParams:
    """Test quantization parameter computation."""
    
    def test_compute_params_int8_symmetric(self):
        """Test INT8 symmetric quantization parameters."""
        config = GPTQCalibrationConfig()
        calibrator = GPTQCalibrator(config)
        
        tensor = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
        scale, zero_point = calibrator._compute_quantization_params(
            tensor, qmin=-128, qmax=127, symmetric=True
        )
        
        # Symmetric: zero_point should be 0
        assert zero_point == 0
        # scale = max(|x|) / qmax = 10 / 127
        assert abs(scale - 10.0 / 127) < 1e-6
    
    def test_compute_params_int8_asymmetric(self):
        """Test INT8 asymmetric quantization parameters."""
        config = GPTQCalibrationConfig()
        calibrator = GPTQCalibrator(config)
        
        tensor = torch.tensor([0.0, 2.5, 5.0, 7.5, 10.0])
        scale, zero_point = calibrator._compute_quantization_params(
            tensor, qmin=-128, qmax=127, symmetric=False
        )
        
        # Asymmetric: zero_point should be non-zero
        assert zero_point == -128
        # scale = (max - min) / (qmax - qmin)
        assert abs(scale - 10.0 / 255) < 1e-6
    
    def test_compute_params_int2_symmetric(self):
        """Test INT2 symmetric quantization parameters."""
        config = GPTQCalibrationConfig()
        calibrator = GPTQCalibrator(config)
        
        tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        scale, zero_point = calibrator._compute_quantization_params(
            tensor, qmin=-2, qmax=1, symmetric=True
        )
        
        # Symmetric: zero_point should be 0
        assert zero_point == 0
        # scale = max(|x|) / qmax = 2 / 1 = 2.0
        assert abs(scale - 2.0) < 1e-6
    
    def test_compute_params_zero_tensor(self):
        """Test quantization params for zero tensor."""
        config = GPTQCalibrationConfig()
        calibrator = GPTQCalibrator(config)
        
        tensor = torch.zeros(10)
        scale, zero_point = calibrator._compute_quantization_params(
            tensor, qmin=-128, qmax=127, symmetric=True
        )
        
        # Should handle gracefully
        assert scale == 1.0
        assert zero_point == 0
    
    def test_compute_params_constant_tensor(self):
        """Test quantization params for constant tensor."""
        config = GPTQCalibrationConfig()
        calibrator = GPTQCalibrator(config)
        
        tensor = torch.full((10,), 5.0)
        scale, zero_point = calibrator._compute_quantization_params(
            tensor, qmin=-128, qmax=127, symmetric=False
        )
        
        # min == max, should handle gracefully
        assert scale == 1.0
        assert zero_point == 0


class TestLayerCalibration:
    """Test layer-wise calibration."""
    
    def test_calibrate_layer_int8(self):
        """Test INT8 layer calibration."""
        config = GPTQCalibrationConfig(num_samples=32)
        calibrator = GPTQCalibrator(config)
        
        torch.manual_seed(42)
        weight = torch.randn(64, 128)  # [out_features, in_features]
        calibration_data = torch.randn(32, 16, 128)  # [batch, seq, in_features]
        
        result = calibrator.calibrate_layer(
            weight=weight,
            calibration_data=calibration_data,
            quant_type='int8',
            symmetric=True,
            layer_name='test_layer'
        )
        
        # Check result structure
        assert 'quantized' in result
        assert 'scales' in result
        assert 'zero_points' in result
        assert 'error' in result
        
        # Check shapes
        assert result['quantized'].shape == weight.shape
        assert result['scales'].shape == (64,)  # per-channel
        assert result['zero_points'].shape == (64,)
        
        # Check quantized values in valid range
        assert torch.all(result['quantized'] >= -128)
        assert torch.all(result['quantized'] <= 127)
        
        # Check error is reasonable
        assert 0 <= result['error'] < 1.0
    
    def test_calibrate_layer_int2(self):
        """Test INT2 layer calibration."""
        config = GPTQCalibrationConfig(num_samples=32)
        calibrator = GPTQCalibrator(config)
        
        torch.manual_seed(42)
        weight = torch.randn(64, 128)
        calibration_data = torch.randn(32, 16, 128)
        
        result = calibrator.calibrate_layer(
            weight=weight,
            calibration_data=calibration_data,
            quant_type='int2',
            symmetric=True,
            layer_name='test_layer'
        )
        
        # Check quantized values in valid INT2 range
        assert torch.all(result['quantized'] >= -2)
        assert torch.all(result['quantized'] <= 1)
    
    def test_calibrate_layer_asymmetric(self):
        """Test asymmetric quantization."""
        config = GPTQCalibrationConfig(num_samples=32)
        calibrator = GPTQCalibrator(config)
        
        torch.manual_seed(42)
        weight = torch.randn(64, 128)
        calibration_data = torch.randn(32, 16, 128)
        
        result = calibrator.calibrate_layer(
            weight=weight,
            calibration_data=calibration_data,
            quant_type='int8',
            symmetric=False,
            layer_name='test_layer'
        )
        
        # Check that zero_points are non-zero (asymmetric)
        assert torch.any(result['zero_points'] != 0)
    
    def test_calibrate_layer_error_compensation(self):
        """Test that GPTQ error compensation reduces error."""
        config = GPTQCalibrationConfig(num_samples=64)
        calibrator = GPTQCalibrator(config)
        
        torch.manual_seed(42)
        weight = torch.randn(32, 64)
        calibration_data = torch.randn(64, 16, 64)
        
        result = calibrator.calibrate_layer(
            weight=weight,
            calibration_data=calibration_data,
            quant_type='int8',
            symmetric=True
        )
        
        # GPTQ should achieve low reconstruction error
        # Typically < 0.1 for INT8 with good calibration
        assert result['error'] < 0.2
    
    def test_calibrate_layer_invalid_shape(self):
        """Test that non-2D weight raises error."""
        config = GPTQCalibrationConfig()
        calibrator = GPTQCalibrator(config)
        
        # 1D weight
        weight = torch.randn(64)
        calibration_data = torch.randn(32, 16, 64)
        
        with pytest.raises(ValueError) as exc_info:
            calibrator.calibrate_layer(
                weight=weight,
                calibration_data=calibration_data,
                quant_type='int8'
            )
        
        assert 'Weight must be 2D' in str(exc_info.value)
    
    def test_calibrate_layer_invalid_quant_type(self):
        """Test that invalid quant_type raises error."""
        config = GPTQCalibrationConfig()
        calibrator = GPTQCalibrator(config)
        
        weight = torch.randn(64, 128)
        calibration_data = torch.randn(32, 16, 128)
        
        with pytest.raises(ValueError) as exc_info:
            calibrator.calibrate_layer(
                weight=weight,
                calibration_data=calibration_data,
                quant_type='int4'
            )
        
        assert 'Unsupported quant_type' in str(exc_info.value)
    
    def test_calibrate_layer_singular_hessian(self):
        """Test handling of singular Hessian."""
        config = GPTQCalibrationConfig()
        calibrator = GPTQCalibrator(config)
        
        # Create weight and calibration data that lead to singular Hessian
        weight = torch.randn(64, 128)
        # All zeros -> singular Hessian
        calibration_data = torch.zeros(32, 16, 128)
        
        result = calibrator.calibrate_layer(
            weight=weight,
            calibration_data=calibration_data,
            quant_type='int8'
        )
        
        # Should return original weights with infinite error
        assert result['error'] == float('inf')
        assert torch.allclose(result['quantized'], weight)


class TestCalibrationDatasetPreparation:
    """Test calibration dataset preparation."""
    
    def test_prepare_calibration_dataset(self):
        """Test calibration dataset preparation."""
        config = GPTQCalibrationConfig(num_samples=10)
        calibrator = GPTQCalibrator(config)
        
        # Mock tokenizer
        class MockTokenizer:
            def __call__(self, texts, max_length, padding, truncation, return_tensors):
                # Return mock encoded data
                batch_size = len(texts)
                return {
                    'input_ids': torch.randint(0, 1000, (batch_size, max_length))
                }
        
        tokenizer = MockTokenizer()
        texts = [f"Sample text {i}" for i in range(20)]
        
        input_ids = calibrator.prepare_calibration_dataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=128
        )
        
        # Should take first num_samples
        assert input_ids.shape == (10, 128)
    
    def test_prepare_calibration_dataset_fewer_samples(self):
        """Test with fewer samples than requested."""
        config = GPTQCalibrationConfig(num_samples=100)
        calibrator = GPTQCalibrator(config)
        
        class MockTokenizer:
            def __call__(self, texts, max_length, padding, truncation, return_tensors):
                batch_size = len(texts)
                return {
                    'input_ids': torch.randint(0, 1000, (batch_size, max_length))
                }
        
        tokenizer = MockTokenizer()
        texts = [f"Sample text {i}" for i in range(10)]
        
        input_ids = calibrator.prepare_calibration_dataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=128
        )
        
        # Should use all available samples
        assert input_ids.shape == (10, 128)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_small_weight(self):
        """Test calibration with very small weights."""
        config = GPTQCalibrationConfig(num_samples=32)
        calibrator = GPTQCalibrator(config)
        
        # Very small weights
        weight = torch.randn(32, 64) * 1e-6
        calibration_data = torch.randn(32, 16, 64)
        
        result = calibrator.calibrate_layer(
            weight=weight,
            calibration_data=calibration_data,
            quant_type='int8'
        )
        
        # Should handle gracefully
        assert result['quantized'] is not None
    
    def test_very_large_weight(self):
        """Test calibration with very large weights."""
        config = GPTQCalibrationConfig(num_samples=32)
        calibrator = GPTQCalibrator(config)
        
        # Very large weights
        weight = torch.randn(32, 64) * 1e6
        calibration_data = torch.randn(32, 16, 64)
        
        result = calibrator.calibrate_layer(
            weight=weight,
            calibration_data=calibration_data,
            quant_type='int8'
        )
        
        # Should handle gracefully
        assert result['quantized'] is not None
        # Values should be clipped to valid range
        assert torch.all(result['quantized'] >= -128)
        assert torch.all(result['quantized'] <= 127)
    
    def test_single_sample_calibration(self):
        """Test calibration with single sample."""
        config = GPTQCalibrationConfig(num_samples=1)
        calibrator = GPTQCalibrator(config)
        
        weight = torch.randn(32, 64)
        calibration_data = torch.randn(1, 16, 64)
        
        result = calibrator.calibrate_layer(
            weight=weight,
            calibration_data=calibration_data,
            quant_type='int8'
        )
        
        # Should work with single sample
        assert result['quantized'] is not None
    
    def test_large_layer(self):
        """Test calibration with large layer."""
        config = GPTQCalibrationConfig(num_samples=32)
        calibrator = GPTQCalibrator(config)
        
        # Large layer (e.g., 4096 x 4096)
        torch.manual_seed(42)
        weight = torch.randn(512, 512)  # Reduced for test speed
        calibration_data = torch.randn(32, 16, 512)
        
        result = calibrator.calibrate_layer(
            weight=weight,
            calibration_data=calibration_data,
            quant_type='int8'
        )
        
        # Should handle large layers
        assert result['quantized'].shape == weight.shape


class TestPerformance:
    """Test performance characteristics."""
    
    def test_calibration_time_reasonable(self):
        """Test that calibration completes in reasonable time."""
        import time
        
        config = GPTQCalibrationConfig(num_samples=64)
        calibrator = GPTQCalibrator(config)
        
        torch.manual_seed(42)
        weight = torch.randn(256, 256)
        calibration_data = torch.randn(64, 32, 256)
        
        start_time = time.time()
        result = calibrator.calibrate_layer(
            weight=weight,
            calibration_data=calibration_data,
            quant_type='int8'
        )
        elapsed = time.time() - start_time
        
        # Should complete in < 10 seconds for this size
        assert elapsed < 10.0
        assert result['quantized'] is not None
    
    def test_cache_improves_performance(self):
        """Test that caching improves performance."""
        import time
        
        config = GPTQCalibrationConfig(num_samples=64, use_cache=True)
        calibrator = GPTQCalibrator(config)
        
        torch.manual_seed(42)
        calibration_data = torch.randn(64, 32, 256)
        
        # First computation (no cache)
        start1 = time.time()
        H1 = calibrator.compute_hessian(calibration_data, layer_name='layer1')
        time1 = time.time() - start1
        
        # Second computation (cached)
        start2 = time.time()
        H2 = calibrator.compute_hessian(calibration_data, layer_name='layer1')
        time2 = time.time() - start2
        
        # Cached should be much faster
        assert time2 < time1 * 0.1  # At least 10x faster


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

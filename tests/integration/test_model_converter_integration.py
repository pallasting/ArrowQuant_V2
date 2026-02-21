"""
Integration tests for model converter with WeightLoader.

Tests end-to-end conversion workflow:
1. HuggingFace model → Parquet conversion
2. Loading converted weights with WeightLoader
3. Verifying weight correctness
"""

import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

from llm_compression.inference.model_converter import HuggingFaceToParquetConverter
from llm_compression.inference.weight_loader import WeightLoader


class TestModelConverterIntegration:
    """Integration tests for model converter with weight loader"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def mock_fp16_model(self, temp_dir):
        """Create mock FP16 model"""
        model_dir = temp_dir / "mock_fp16_model"
        model_dir.mkdir()
        
        # Create deterministic weights for verification
        torch.manual_seed(42)
        state_dict = {
            'layer.0.weight': torch.randn(128, 64, dtype=torch.float16),
            'layer.0.bias': torch.randn(128, dtype=torch.float16),
            'layer.1.weight': torch.randn(64, 32, dtype=torch.float16),
        }
        
        torch.save(state_dict, model_dir / "pytorch_model.bin")
        return model_dir, state_dict
    
    @pytest.fixture
    def mock_quantized_model(self, temp_dir):
        """Create mock quantized model"""
        model_dir = temp_dir / "mock_quantized_model"
        model_dir.mkdir()
        
        # Create deterministic quantized weights
        torch.manual_seed(42)
        state_dict = {
            'layer.0.weight': torch.randint(-128, 127, (128, 64), dtype=torch.int8),
            'layer.0.weight_scale': torch.rand(128, dtype=torch.float32) * 0.1,
            'layer.0.weight_zero_point': torch.zeros(128, dtype=torch.int8),
        }
        
        torch.save(state_dict, model_dir / "pytorch_model.bin")
        return model_dir, state_dict
    
    def test_convert_and_load_fp16_model(self, mock_fp16_model, temp_dir):
        """Test converting FP16 model and loading with WeightLoader"""
        model_dir, original_state_dict = mock_fp16_model
        output_path = temp_dir / "output.parquet"
        
        # 1. Convert HuggingFace → Parquet
        converter = HuggingFaceToParquetConverter()
        converter.convert(
            hf_model_path=str(model_dir),
            output_parquet=str(output_path)
        )
        
        # 2. Load with WeightLoader
        loader = WeightLoader(str(output_path))
        
        # 3. Verify weights match original
        for layer_name, original_tensor in original_state_dict.items():
            loaded_tensor = loader.get_layer(layer_name)
            
            # Convert to same dtype for comparison
            original_np = original_tensor.cpu().numpy()
            loaded_np = loaded_tensor.cpu().numpy()
            
            # Should be identical (or very close due to FP16 precision)
            np.testing.assert_allclose(
                original_np,
                loaded_np,
                rtol=1e-3,
                atol=1e-5,
                err_msg=f"Weight mismatch for {layer_name}"
            )
    
    def test_convert_and_load_quantized_model(self, mock_quantized_model, temp_dir):
        """Test converting quantized model and loading with WeightLoader"""
        model_dir, original_state_dict = mock_quantized_model
        output_path = temp_dir / "output.parquet"
        
        # 1. Convert HuggingFace → Parquet
        converter = HuggingFaceToParquetConverter()
        converter.convert(
            hf_model_path=str(model_dir),
            output_parquet=str(output_path)
        )
        
        # 2. Load with WeightLoader
        loader = WeightLoader(str(output_path))
        
        # 3. Verify quantized weights can be loaded
        loaded_tensor = loader.get_layer('layer.0.weight')
        
        # Should be dequantized to float
        assert loaded_tensor.dtype in [torch.float32, torch.float16]
        assert loaded_tensor.shape == (128, 64)
    
    def test_roundtrip_preserves_values(self, temp_dir):
        """Test that roundtrip conversion preserves values"""
        # Create simple model
        model_dir = temp_dir / "simple_model"
        model_dir.mkdir()
        
        # Use simple values for exact comparison
        state_dict = {
            'weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            'bias': torch.tensor([0.5, -0.5], dtype=torch.float32),
        }
        
        torch.save(state_dict, model_dir / "pytorch_model.bin")
        
        # Convert
        output_path = temp_dir / "output.parquet"
        converter = HuggingFaceToParquetConverter()
        converter.convert(
            hf_model_path=str(model_dir),
            output_parquet=str(output_path)
        )
        
        # Load and verify
        loader = WeightLoader(str(output_path))
        
        loaded_weight = loader.get_layer('weight')
        loaded_bias = loader.get_layer('bias')
        
        # Should be exactly equal
        torch.testing.assert_close(loaded_weight, state_dict['weight'])
        torch.testing.assert_close(loaded_bias, state_dict['bias'])
    
    def test_large_model_conversion(self, temp_dir):
        """Test converting larger model (stress test)"""
        model_dir = temp_dir / "large_model"
        model_dir.mkdir()
        
        # Create model with multiple layers
        torch.manual_seed(42)
        state_dict = {}
        for i in range(10):
            state_dict[f'layer.{i}.weight'] = torch.randn(512, 512, dtype=torch.float16)
            state_dict[f'layer.{i}.bias'] = torch.randn(512, dtype=torch.float16)
        
        torch.save(state_dict, model_dir / "pytorch_model.bin")
        
        # Convert
        output_path = temp_dir / "output.parquet"
        converter = HuggingFaceToParquetConverter()
        converter.convert(
            hf_model_path=str(model_dir),
            output_parquet=str(output_path)
        )
        
        # Verify all layers can be loaded
        loader = WeightLoader(str(output_path))
        
        for i in range(10):
            weight = loader.get_layer(f'layer.{i}.weight')
            bias = loader.get_layer(f'layer.{i}.bias')
            
            assert weight.shape == (512, 512)
            assert bias.shape == (512,)
    
    def test_converter_with_mixed_precision(self, temp_dir):
        """Test converter with mixed precision layers"""
        model_dir = temp_dir / "mixed_model"
        model_dir.mkdir()
        
        # Create model with different dtypes
        state_dict = {
            'fp32_layer.weight': torch.randn(64, 32, dtype=torch.float32),
            'fp16_layer.weight': torch.randn(32, 16, dtype=torch.float16),
            'int8_layer.weight': torch.randint(-128, 127, (16, 8), dtype=torch.int8),
            'int8_layer.weight_scale': torch.tensor([0.05], dtype=torch.float32),  # Per-tensor scale
            'int8_layer.weight_zero_point': torch.tensor([0], dtype=torch.int8),  # Per-tensor zero_point
        }
        
        torch.save(state_dict, model_dir / "pytorch_model.bin")
        
        # Convert
        output_path = temp_dir / "output.parquet"
        converter = HuggingFaceToParquetConverter()
        converter.convert(
            hf_model_path=str(model_dir),
            output_parquet=str(output_path)
        )
        
        # Load and verify all layers
        loader = WeightLoader(str(output_path))
        
        fp32_weight = loader.get_layer('fp32_layer.weight')
        fp16_weight = loader.get_layer('fp16_layer.weight')
        int8_weight = loader.get_layer('int8_layer.weight')
        
        assert fp32_weight.shape == (64, 32)
        assert fp16_weight.shape == (32, 16)
        assert int8_weight.shape == (16, 8)


class TestConverterErrorHandling:
    """Test error handling in integration scenarios"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_load_nonexistent_layer(self, temp_dir):
        """Test loading nonexistent layer raises error"""
        # Create simple model
        model_dir = temp_dir / "simple_model"
        model_dir.mkdir()
        
        state_dict = {
            'layer.0.weight': torch.randn(8, 4, dtype=torch.float32),
        }
        
        torch.save(state_dict, model_dir / "pytorch_model.bin")
        
        # Convert
        output_path = temp_dir / "output.parquet"
        converter = HuggingFaceToParquetConverter()
        converter.convert(
            hf_model_path=str(model_dir),
            output_parquet=str(output_path)
        )
        
        # Try to load nonexistent layer
        loader = WeightLoader(str(output_path))
        
        with pytest.raises(KeyError):
            loader.get_layer('nonexistent.layer')

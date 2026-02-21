"""
Unit tests for Per-Group quantization.

Tests the ArrowQuantizer per-group quantization functionality
as per ArrowQuant design document.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import pyarrow.parquet as pq

from llm_compression.inference.arrow_quantizer import ArrowQuantizer, QuantizationConfig
from llm_compression.inference.model_converter import HuggingFaceToParquetConverter
from llm_compression.inference.weight_loader import WeightLoader
from llm_compression.inference.quantization_schema import detect_schema_version


class TestPerGroupQuantization:
    """Test per-group quantization functionality."""
    
    def test_per_group_config_validation(self):
        """Test that group_size is validated correctly."""
        # Valid config
        config = QuantizationConfig(
            quant_type='int2',
            group_size=128
        )
        assert config.group_size == 128
        
        # Invalid group_size (negative)
        with pytest.raises(Exception):
            QuantizationConfig(
                quant_type='int2',
                group_size=-1
            )
    
    def test_auto_group_size_for_int2(self):
        """Test that group_size is auto-set to 128 for INT2."""
        config = QuantizationConfig(
            quant_type='int2',
            per_channel=False,
            group_size=0
        )
        
        # Should auto-set to 128
        assert config.group_size == 128
    
    def test_per_group_quantization_basic(self):
        """Test basic per-group quantization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test model
            model_dir = Path(tmpdir) / "test_model"
            model_dir.mkdir()
            
            # Create a simple weight matrix
            torch.manual_seed(42)
            weight = torch.randn(256, 256, dtype=torch.float16)  # 65536 params
            
            state_dict = {'layer.0.weight': weight}
            torch.save(state_dict, model_dir / "pytorch_model.bin")
            
            # Convert to Parquet V1
            converter = HuggingFaceToParquetConverter()
            original_path = str(Path(tmpdir) / "original.parquet")
            converter.convert(
                hf_model_path=str(model_dir),
                output_parquet=original_path
            )
            
            # Quantize with per-group (group_size=128)
            config = QuantizationConfig(
                quant_type='int2',
                per_channel=False,
                group_size=128
            )
            quantizer = ArrowQuantizer(config)
            
            quantized_path = str(Path(tmpdir) / "quantized.parquet")
            quantizer.quantize_model(
                input_parquet=original_path,
                output_parquet=quantized_path,
                show_progress=False
            )
            
            # Verify output
            table = pq.read_table(quantized_path)
            assert detect_schema_version(table) == 2
            
            # Check quantization metadata
            row = table.slice(0, 1).to_pydict()
            assert row['quant_type'][0] == 'int2'
            assert row['group_size'][0] == 128
            
            # Number of groups = ceil(65536 / 128) = 512
            scales_bytes = row['scales'][0]
            scales = np.frombuffer(scales_bytes, dtype=np.float32)
            assert len(scales) == 512
    
    def test_per_group_compression_ratio(self):
        """Test that per-group INT2 achieves good compression ratio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test model
            model_dir = Path(tmpdir) / "test_model"
            model_dir.mkdir()
            
            # Large weight matrix for better compression ratio
            torch.manual_seed(42)
            weight = torch.randn(1024, 1024, dtype=torch.float16)  # 1M params
            
            state_dict = {'layer.0.weight': weight}
            torch.save(state_dict, model_dir / "pytorch_model.bin")
            
            # Convert to Parquet V1
            converter = HuggingFaceToParquetConverter()
            original_path = str(Path(tmpdir) / "original.parquet")
            converter.convert(
                hf_model_path=str(model_dir),
                output_parquet=original_path
            )
            
            original_size = Path(original_path).stat().st_size
            
            # Quantize with per-group INT2
            config = QuantizationConfig(
                quant_type='int2',
                per_channel=False,
                group_size=128
            )
            quantizer = ArrowQuantizer(config)
            
            quantized_path = str(Path(tmpdir) / "quantized.parquet")
            quantizer.quantize_model(
                input_parquet=original_path,
                output_parquet=quantized_path,
                show_progress=False
            )
            
            quantized_size = Path(quantized_path).stat().st_size
            compression_ratio = original_size / quantized_size
            
            # Should achieve > 4x compression (target is ~6-8x for large models)
            assert compression_ratio > 4.0
            print(f"Compression ratio: {compression_ratio:.2f}x")
    
    def test_per_group_vs_per_tensor(self):
        """Test that per-group provides better accuracy than per-tensor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test model
            model_dir = Path(tmpdir) / "test_model"
            model_dir.mkdir()
            
            # Weight with varying ranges (benefits from per-group)
            torch.manual_seed(42)
            weight = torch.randn(512, 512, dtype=torch.float16)
            
            state_dict = {'layer.0.weight': weight}
            torch.save(state_dict, model_dir / "pytorch_model.bin")
            
            # Convert to Parquet V1
            converter = HuggingFaceToParquetConverter()
            original_path = str(Path(tmpdir) / "original.parquet")
            converter.convert(
                hf_model_path=str(model_dir),
                output_parquet=original_path
            )
            
            # Quantize with per-tensor
            config_tensor = QuantizationConfig(
                quant_type='int2',
                per_channel=False,
                group_size=0
            )
            quantizer_tensor = ArrowQuantizer(config_tensor)
            
            tensor_path = str(Path(tmpdir) / "quantized_tensor.parquet")
            quantizer_tensor.quantize_model(
                input_parquet=original_path,
                output_parquet=tensor_path,
                show_progress=False
            )
            
            # Quantize with per-group
            config_group = QuantizationConfig(
                quant_type='int2',
                per_channel=False,
                group_size=128
            )
            quantizer_group = ArrowQuantizer(config_group)
            
            group_path = str(Path(tmpdir) / "quantized_group.parquet")
            quantizer_group.quantize_model(
                input_parquet=original_path,
                output_parquet=group_path,
                show_progress=False
            )
            
            # Load and compare
            original_loader = WeightLoader(original_path)
            tensor_loader = WeightLoader(tensor_path)
            group_loader = WeightLoader(group_path)
            
            original_weights = original_loader.load_weights()
            tensor_weights = tensor_loader.load_weights()
            group_weights = group_loader.load_weights()
            
            layer_name = 'layer.0.weight'
            orig = original_weights[layer_name].cpu().numpy().flatten()
            tensor_quant = tensor_weights[layer_name].cpu().numpy().flatten()
            group_quant = group_weights[layer_name].cpu().numpy().flatten()
            
            # Compute cosine similarity
            def cosine_sim(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            tensor_sim = cosine_sim(orig, tensor_quant)
            group_sim = cosine_sim(orig, group_quant)
            
            print(f"Per-tensor similarity: {tensor_sim:.4f}")
            print(f"Per-group similarity: {group_sim:.4f}")
            
            # Per-group should have better or equal accuracy
            assert group_sim >= tensor_sim - 0.01  # Allow small margin
    
    def test_per_group_metadata_size(self):
        """Test that per-group metadata overhead is reasonable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test model
            model_dir = Path(tmpdir) / "test_model"
            model_dir.mkdir()
            
            torch.manual_seed(42)
            weight = torch.randn(1024, 1024, dtype=torch.float16)  # 1M params
            
            state_dict = {'layer.0.weight': weight}
            torch.save(state_dict, model_dir / "pytorch_model.bin")
            
            # Convert to Parquet V1
            converter = HuggingFaceToParquetConverter()
            original_path = str(Path(tmpdir) / "original.parquet")
            converter.convert(
                hf_model_path=str(model_dir),
                output_parquet=original_path
            )
            
            # Quantize with per-group
            config = QuantizationConfig(
                quant_type='int2',
                per_channel=False,
                group_size=128
            )
            quantizer = ArrowQuantizer(config)
            
            quantized_path = str(Path(tmpdir) / "quantized.parquet")
            quantizer.quantize_model(
                input_parquet=original_path,
                output_parquet=quantized_path,
                show_progress=False
            )
            
            # Check metadata size
            table = pq.read_table(quantized_path)
            row = table.slice(0, 1).to_pydict()
            
            # Data size (packed INT2)
            data_size = len(row['data'][0])
            
            # Metadata size (scales + zero_points)
            scales_size = len(row['scales'][0])
            zero_points_size = len(row['zero_points'][0])
            metadata_size = scales_size + zero_points_size
            
            # Metadata should be < 10% of data size for large models
            metadata_ratio = metadata_size / data_size
            print(f"Metadata overhead: {metadata_ratio * 100:.2f}%")
            
            # For 1M params with group_size=128:
            # - Data: 1M / 4 = 256KB (packed INT2)
            # - Metadata: (1M / 128) * 8 bytes = 64KB
            # - Ratio: 64KB / 256KB = 25%
            assert metadata_ratio < 0.30  # Should be < 30%


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

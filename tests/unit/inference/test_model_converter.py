"""
Unit tests for model_converter module.

Tests HuggingFaceToParquetConverter functionality including:
- FP model conversion (Schema V1)
- Quantized model conversion (Schema V2)
- Quantization detection
- Error handling
"""

import os
import tempfile
from pathlib import Path
from typing import Dict

import pytest
import numpy as np
import torch
import pyarrow.parquet as pq

from llm_compression.inference.model_converter import (
    HuggingFaceToParquetConverter,
    ParquetToHuggingFaceConverter,
    ModelConverterError,
    convert_hf_to_parquet,
)
from llm_compression.inference.quantization_schema import (
    detect_schema_version,
    validate_schema_v1,
    validate_schema_v2,
)


class TestHuggingFaceToParquetConverter:
    """Test HuggingFaceToParquetConverter"""
    
    @pytest.fixture
    def converter(self):
        """Create converter instance"""
        return HuggingFaceToParquetConverter()
    
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
        
        # Create mock state dict
        state_dict = {
            'layer.0.weight': torch.randn(128, 64, dtype=torch.float16),
            'layer.0.bias': torch.randn(128, dtype=torch.float16),
            'layer.1.weight': torch.randn(64, 32, dtype=torch.float16),
            'layer.1.bias': torch.randn(64, dtype=torch.float16),
        }
        
        # Save as pytorch_model.bin
        torch.save(state_dict, model_dir / "pytorch_model.bin")
        
        return model_dir
    
    @pytest.fixture
    def mock_quantized_model(self, temp_dir):
        """Create mock quantized model (INT8)"""
        model_dir = temp_dir / "mock_quantized_model"
        model_dir.mkdir()
        
        # Create mock quantized state dict
        state_dict = {
            'layer.0.weight': torch.randint(-128, 127, (128, 64), dtype=torch.int8),
            'layer.0.weight_scale': torch.randn(128, dtype=torch.float32),
            'layer.0.weight_zero_point': torch.zeros(128, dtype=torch.int8),
            'layer.1.weight': torch.randint(-128, 127, (64, 32), dtype=torch.int8),
            'layer.1.weight_scale': torch.randn(64, dtype=torch.float32),
        }
        
        # Save as pytorch_model.bin
        torch.save(state_dict, model_dir / "pytorch_model.bin")
        
        return model_dir
    
    def test_convert_fp16_model(self, converter, mock_fp16_model, temp_dir):
        """Test converting FP16 model to Parquet V1"""
        output_path = temp_dir / "output.parquet"
        
        # Convert
        converter.convert(
            hf_model_path=str(mock_fp16_model),
            output_parquet=str(output_path),
            auto_detect_quantization=True
        )
        
        # Verify output file exists
        assert output_path.exists()
        
        # Load and verify
        table = pq.read_table(output_path)
        
        # Should be Schema V1
        assert detect_schema_version(table) == 1
        assert validate_schema_v1(table)
        
        # Should have 4 layers (2 weights + 2 biases)
        assert len(table) == 4
        
        # Verify layer names
        layer_names = table['layer_name'].to_pylist()
        assert 'layer.0.weight' in layer_names
        assert 'layer.0.bias' in layer_names
        assert 'layer.1.weight' in layer_names
        assert 'layer.1.bias' in layer_names
    
    def test_convert_quantized_model(self, converter, mock_quantized_model, temp_dir):
        """Test converting quantized model to Parquet V2"""
        output_path = temp_dir / "output.parquet"
        
        # Convert
        converter.convert(
            hf_model_path=str(mock_quantized_model),
            output_parquet=str(output_path),
            auto_detect_quantization=True
        )
        
        # Verify output file exists
        assert output_path.exists()
        
        # Load and verify
        table = pq.read_table(output_path)
        
        # Should be Schema V2
        assert detect_schema_version(table) == 2
        assert validate_schema_v2(table)
        
        # Should have 2 weight layers (scales and zero_points are metadata)
        assert len(table) == 2
        
        # Verify quantization metadata
        for i in range(len(table)):
            row = table.slice(i, 1).to_pydict()
            assert row['quant_type'][0] == 'int8'
            assert len(row['scales'][0]) > 0
            assert len(row['zero_points'][0]) > 0
    
    def test_disable_auto_detect_quantization(self, converter, mock_quantized_model, temp_dir):
        """Test disabling auto-detection forces V1 schema"""
        output_path = temp_dir / "output.parquet"
        
        # Convert with auto_detect_quantization=False
        converter.convert(
            hf_model_path=str(mock_quantized_model),
            output_parquet=str(output_path),
            auto_detect_quantization=False
        )
        
        # Should use Schema V1 even for quantized model
        table = pq.read_table(output_path)
        assert detect_schema_version(table) == 1
    
    def test_model_not_found(self, converter, temp_dir):
        """Test error when model file not found"""
        nonexistent_path = temp_dir / "nonexistent_model"
        output_path = temp_dir / "output.parquet"
        
        with pytest.raises(ModelConverterError, match="No model file found"):
            converter.convert(
                hf_model_path=str(nonexistent_path),
                output_parquet=str(output_path)
            )
    
    def test_detect_quantization_fp_model(self, converter, mock_fp16_model):
        """Test quantization detection on FP model"""
        # Load state dict
        state_dict = torch.load(mock_fp16_model / "pytorch_model.bin")
        
        # Detect quantization
        is_quantized, quant_info = converter._detect_quantization(state_dict)
        
        assert is_quantized is False
        assert quant_info is None
    
    def test_detect_quantization_int8_model(self, converter, mock_quantized_model):
        """Test quantization detection on INT8 model"""
        # Load state dict
        state_dict = torch.load(mock_quantized_model / "pytorch_model.bin")
        
        # Detect quantization
        is_quantized, quant_info = converter._detect_quantization(state_dict)
        
        assert is_quantized is True
        assert quant_info is not None
        assert 'scales' in quant_info
        assert 'zero_points' in quant_info
        assert len(quant_info['scales']) > 0
    
    def test_extract_quantization_metadata(self, converter, mock_quantized_model):
        """Test extracting quantization metadata"""
        # Load state dict
        state_dict = torch.load(mock_quantized_model / "pytorch_model.bin")
        
        # Extract metadata
        metadata = converter._extract_quantization_metadata(state_dict)
        
        assert 'scales' in metadata
        assert 'zero_points' in metadata
        assert 'quant_type' in metadata
        
        # Should find scale tensors
        assert 'layer.0.weight' in metadata['scales']
        assert 'layer.1.weight' in metadata['scales']
    
    def test_is_weight_tensor(self, converter):
        """Test weight tensor detection"""
        # Weight tensors
        assert converter._is_weight_tensor('layer.0.weight') is True
        assert converter._is_weight_tensor('layer.0.bias') is True
        assert converter._is_weight_tensor('encoder.layer.0.weight') is True
        
        # Non-weight tensors
        assert converter._is_weight_tensor('num_batches_tracked') is False
        assert converter._is_weight_tensor('running_mean') is False
        assert converter._is_weight_tensor('running_var') is False
        assert converter._is_weight_tensor('position_ids') is False
    
    def test_convert_fp_model_rows(self, converter):
        """Test converting FP model to rows"""
        # Create mock state dict
        state_dict = {
            'layer.0.weight': torch.randn(4, 3, dtype=torch.float32),
            'layer.0.bias': torch.randn(4, dtype=torch.float32),
        }
        
        # Convert to rows
        rows = converter._convert_fp_model(state_dict)
        
        assert len(rows) == 2
        
        # Verify first row
        row0 = rows[0]
        assert row0['layer_name'] == 'layer.0.weight'
        assert row0['shape'] == [4, 3]
        assert row0['dtype'] == 'torch.float32'
        assert row0['num_params'] == 12
        assert isinstance(row0['data'], bytes)
    
    def test_convert_quantized_model_rows(self, converter):
        """Test converting quantized model to rows"""
        # Create mock quantized state dict
        state_dict = {
            'layer.0.weight': torch.randint(-128, 127, (4, 3), dtype=torch.int8),
        }
        
        # Create mock quant_info
        quant_info = {
            'scales': {
                'layer.0.weight': torch.tensor([0.1, 0.2, 0.15, 0.18], dtype=torch.float32)
            },
            'zero_points': {
                'layer.0.weight': torch.tensor([0, 0, 0, 0], dtype=torch.int8)
            },
            'quant_type': 'int8'
        }
        
        # Convert to rows
        rows = converter._convert_quantized_model(state_dict, quant_info)
        
        assert len(rows) == 1
        
        # Verify row
        row = rows[0]
        assert row['layer_name'] == 'layer.0.weight'
        assert row['shape'] == [4, 3]
        assert row['quant_type'] == 'int8'
        assert len(row['scales']) == 4
        assert len(row['zero_points']) == 4
        assert row['quant_axis'] == 0  # Per-channel
    
    def test_convert_quantized_model_per_tensor(self, converter):
        """Test converting quantized model with per-tensor quantization"""
        # Create mock quantized state dict
        state_dict = {
            'layer.0.weight': torch.randint(-128, 127, (4, 3), dtype=torch.int8),
        }
        
        # Create mock quant_info with per-tensor scale
        quant_info = {
            'scales': {
                'layer.0.weight': torch.tensor([0.1], dtype=torch.float32)  # Single scale
            },
            'zero_points': {
                'layer.0.weight': torch.tensor([0], dtype=torch.int8)  # Single zero_point
            },
            'quant_type': 'int8'
        }
        
        # Convert to rows
        rows = converter._convert_quantized_model(state_dict, quant_info)
        
        assert len(rows) == 1
        
        # Verify row
        row = rows[0]
        assert len(row['scales']) == 1
        assert len(row['zero_points']) == 1
        assert row['quant_axis'] == -1  # Per-tensor


class TestParquetToHuggingFaceConverter:
    """Test ParquetToHuggingFaceConverter"""
    
    def test_not_implemented(self):
        """Test that ParquetToHuggingFaceConverter raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            ParquetToHuggingFaceConverter()


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def mock_fp16_model(self, temp_dir):
        """Create mock FP16 model"""
        model_dir = temp_dir / "mock_model"
        model_dir.mkdir()
        
        state_dict = {
            'layer.0.weight': torch.randn(8, 4, dtype=torch.float16),
        }
        
        torch.save(state_dict, model_dir / "pytorch_model.bin")
        return model_dir
    
    def test_convert_hf_to_parquet(self, mock_fp16_model, temp_dir):
        """Test convert_hf_to_parquet convenience function"""
        output_path = temp_dir / "output.parquet"
        
        # Convert
        convert_hf_to_parquet(
            hf_model_path=str(mock_fp16_model),
            output_parquet=str(output_path)
        )
        
        # Verify
        assert output_path.exists()
        table = pq.read_table(output_path)
        assert len(table) > 0
    
    def test_convert_parquet_to_hf_not_implemented(self):
        """Test convert_parquet_to_hf raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            from llm_compression.inference.model_converter import convert_parquet_to_hf
            convert_parquet_to_hf("input.parquet", "output_dir")


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def converter(self):
        """Create converter instance"""
        return HuggingFaceToParquetConverter()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_empty_model(self, converter, temp_dir):
        """Test converting empty model"""
        model_dir = temp_dir / "empty_model"
        model_dir.mkdir()
        
        # Create empty state dict
        state_dict = {}
        torch.save(state_dict, model_dir / "pytorch_model.bin")
        
        output_path = temp_dir / "output.parquet"
        
        # Should not crash, but produce empty table
        converter.convert(
            hf_model_path=str(model_dir),
            output_parquet=str(output_path)
        )
        
        table = pq.read_table(output_path)
        assert len(table) == 0
    
    def test_model_with_only_metadata(self, converter, temp_dir):
        """Test model with only metadata tensors (no weights)"""
        model_dir = temp_dir / "metadata_only_model"
        model_dir.mkdir()
        
        # Create state dict with only metadata
        state_dict = {
            'num_batches_tracked': torch.tensor(100),
            'running_mean': torch.randn(64),
            'running_var': torch.randn(64),
        }
        torch.save(state_dict, model_dir / "pytorch_model.bin")
        
        output_path = temp_dir / "output.parquet"
        
        # Should skip all metadata tensors
        converter.convert(
            hf_model_path=str(model_dir),
            output_parquet=str(output_path)
        )
        
        table = pq.read_table(output_path)
        assert len(table) == 0
    
    def test_mixed_dtype_model(self, converter, temp_dir):
        """Test model with mixed dtypes"""
        model_dir = temp_dir / "mixed_dtype_model"
        model_dir.mkdir()
        
        # Create state dict with mixed dtypes
        state_dict = {
            'layer.0.weight': torch.randn(8, 4, dtype=torch.float32),
            'layer.1.weight': torch.randn(4, 2, dtype=torch.float16),
            'layer.2.weight': torch.randn(2, 1, dtype=torch.bfloat16),
        }
        torch.save(state_dict, model_dir / "pytorch_model.bin")
        
        output_path = temp_dir / "output.parquet"
        
        # Should handle mixed dtypes
        converter.convert(
            hf_model_path=str(model_dir),
            output_parquet=str(output_path)
        )
        
        table = pq.read_table(output_path)
        assert len(table) == 3
        
        # Verify dtypes are preserved
        dtypes = table['dtype'].to_pylist()
        assert 'torch.float32' in dtypes
        assert 'torch.float16' in dtypes
        assert 'torch.bfloat16' in dtypes

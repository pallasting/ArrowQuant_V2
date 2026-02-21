"""
End-to-end quantization validation tests.

Tests the complete quantization pipeline with real models:
- MiniLM model quantization (INT8/INT2)
- Compression ratio validation (>2x for INT8, >4x for INT2)
- Accuracy/precision validation (<15% loss for PTQ)
- V1/V2 schema compatibility
- Performance benchmarking

Requirements: 2.1, 2.8, 2.9, 9.3, 12.1
"""

import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pytest
import torch
import pyarrow.parquet as pq
from transformers import AutoModel, AutoTokenizer

from llm_compression.inference.arrow_quantizer import (
    ArrowQuantizer,
    QuantizationConfig,
)
from llm_compression.inference.model_converter import HuggingFaceToParquetConverter
from llm_compression.inference.weight_loader import WeightLoader
from llm_compression.inference.quantization_schema import detect_schema_version
from llm_compression.logger import logger


@pytest.fixture(scope="module")
def minilm_model():
    """Load MiniLM model for testing."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    try:
        logger.info(f"Loading model: {model_name}")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer, model_name
    except Exception as e:
        pytest.skip(f"Failed to load MiniLM model: {e}")


@pytest.fixture(scope="module")
def minilm_parquet(minilm_model, tmp_path_factory):
    """Convert MiniLM model to Parquet V1 format."""
    model, tokenizer, model_name = minilm_model
    tmpdir = tmp_path_factory.mktemp("minilm")
    
    # Save model to disk
    model_dir = tmpdir / "model"
    model_dir.mkdir()
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    
    # Convert to Parquet V1
    converter = HuggingFaceToParquetConverter()
    parquet_path = str(tmpdir / "minilm_v1.parquet")
    
    logger.info(f"Converting {model_name} to Parquet V1...")
    converter.convert(
        hf_model_path=str(model_dir),
        output_parquet=parquet_path
    )
    
    return parquet_path, model, tokenizer


class TestRealModelQuantization:
    """Test quantization with real models."""
    
    def test_minilm_int8_quantization(self, minilm_parquet, tmp_path):
        """Test INT8 quantization of MiniLM model."""
        input_parquet, model, tokenizer = minilm_parquet
        
        # Quantize to INT8
        config = QuantizationConfig(
            quant_type='int8',
            per_channel=True,
            symmetric=True
        )
        quantizer = ArrowQuantizer(config)
        
        output_parquet = str(tmp_path / "minilm_int8.parquet")
        
        logger.info("Quantizing MiniLM to INT8...")
        start_time = time.time()
        quantizer.quantize_model(
            input_parquet=input_parquet,
            output_parquet=output_parquet,
            show_progress=True
        )
        quantization_time = time.time() - start_time
        
        logger.info(f"Quantization completed in {quantization_time:.2f}s")
        
        # Verify output exists
        assert Path(output_parquet).exists()
        
        # Verify schema version
        table = pq.read_table(output_parquet)
        assert detect_schema_version(table) == 2
        
        # Verify all layers are quantized
        for i in range(len(table)):
            row = table.slice(i, 1).to_pydict()
            quant_type = row['quant_type'][0]
            assert quant_type in ['int8', 'fp16']  # fp16 for skipped layers
    
    def test_minilm_int2_quantization(self, minilm_parquet, tmp_path):
        """Test INT2 quantization of MiniLM model."""
        input_parquet, model, tokenizer = minilm_parquet
        
        # Quantize to INT2
        config = QuantizationConfig(
            quant_type='int2',
            per_channel=False,
            group_size=128,
            symmetric=True
        )
        quantizer = ArrowQuantizer(config)
        
        output_parquet = str(tmp_path / "minilm_int2.parquet")
        
        logger.info("Quantizing MiniLM to INT2...")
        start_time = time.time()
        quantizer.quantize_model(
            input_parquet=input_parquet,
            output_parquet=output_parquet,
            show_progress=True
        )
        quantization_time = time.time() - start_time
        
        logger.info(f"Quantization completed in {quantization_time:.2f}s")
        
        # Verify output exists
        assert Path(output_parquet).exists()
        
        # Verify schema version
        table = pq.read_table(output_parquet)
        assert detect_schema_version(table) == 2
        
        # Verify INT2 quantization
        for i in range(len(table)):
            row = table.slice(i, 1).to_pydict()
            quant_type = row['quant_type'][0]
            assert quant_type in ['int2', 'fp16']


class TestCompressionRatioValidation:
    """Validate compression ratio targets."""
    
    def test_int8_compression_ratio_target(self, minilm_parquet, tmp_path):
        """Validate INT8 compression ratio > 2x."""
        input_parquet, model, tokenizer = minilm_parquet
        
        # Get original size
        original_size = Path(input_parquet).stat().st_size
        logger.info(f"Original model size: {original_size / (1024**2):.2f} MB")
        
        # Quantize to INT8
        config = QuantizationConfig(
            quant_type='int8',
            per_channel=True,
            symmetric=True
        )
        quantizer = ArrowQuantizer(config)
        
        output_parquet = str(tmp_path / "minilm_int8.parquet")
        quantizer.quantize_model(
            input_parquet=input_parquet,
            output_parquet=output_parquet,
            show_progress=False
        )
        
        # Get quantized size
        quantized_size = Path(output_parquet).stat().st_size
        logger.info(f"Quantized model size (INT8): {quantized_size / (1024**2):.2f} MB")
        
        # Calculate compression ratio
        compression_ratio = original_size / quantized_size
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        
        # Validate target: > 2x
        assert compression_ratio > 2.0, f"INT8 compression ratio {compression_ratio:.2f}x < 2.0x target"
    
    def test_int2_compression_ratio_target(self, minilm_parquet, tmp_path):
        """Validate INT2 compression ratio > 4x."""
        input_parquet, model, tokenizer = minilm_parquet
        
        # Get original size
        original_size = Path(input_parquet).stat().st_size
        logger.info(f"Original model size: {original_size / (1024**2):.2f} MB")
        
        # Quantize to INT2
        config = QuantizationConfig(
            quant_type='int2',
            per_channel=False,
            group_size=128,
            symmetric=True
        )
        quantizer = ArrowQuantizer(config)
        
        output_parquet = str(tmp_path / "minilm_int2.parquet")
        quantizer.quantize_model(
            input_parquet=input_parquet,
            output_parquet=output_parquet,
            show_progress=False
        )
        
        # Get quantized size
        quantized_size = Path(output_parquet).stat().st_size
        logger.info(f"Quantized model size (INT2): {quantized_size / (1024**2):.2f} MB")
        
        # Calculate compression ratio
        compression_ratio = original_size / quantized_size
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        
        # Validate target: > 4x
        assert compression_ratio > 4.0, f"INT2 compression ratio {compression_ratio:.2f}x < 4.0x target"
    
    def test_compression_ratio_comparison(self, minilm_parquet, tmp_path):
        """Compare compression ratios across quantization modes."""
        input_parquet, model, tokenizer = minilm_parquet
        original_size = Path(input_parquet).stat().st_size
        
        results = {}
        
        # Test INT8
        config_int8 = QuantizationConfig(quant_type='int8', per_channel=True)
        quantizer_int8 = ArrowQuantizer(config_int8)
        output_int8 = str(tmp_path / "minilm_int8.parquet")
        quantizer_int8.quantize_model(input_parquet, output_int8, show_progress=False)
        
        int8_size = Path(output_int8).stat().st_size
        results['int8'] = {
            'size_mb': int8_size / (1024**2),
            'compression_ratio': original_size / int8_size,
            'memory_savings_pct': (1 - int8_size / original_size) * 100
        }
        
        # Test INT2
        config_int2 = QuantizationConfig(quant_type='int2', per_channel=False, group_size=128)
        quantizer_int2 = ArrowQuantizer(config_int2)
        output_int2 = str(tmp_path / "minilm_int2.parquet")
        quantizer_int2.quantize_model(input_parquet, output_int2, show_progress=False)
        
        int2_size = Path(output_int2).stat().st_size
        results['int2'] = {
            'size_mb': int2_size / (1024**2),
            'compression_ratio': original_size / int2_size,
            'memory_savings_pct': (1 - int2_size / original_size) * 100
        }
        
        # Log results
        logger.info("Compression Ratio Comparison:")
        logger.info(f"  Original: {original_size / (1024**2):.2f} MB")
        logger.info(f"  INT8: {results['int8']['size_mb']:.2f} MB "
                   f"({results['int8']['compression_ratio']:.2f}x, "
                   f"{results['int8']['memory_savings_pct']:.1f}% savings)")
        logger.info(f"  INT2: {results['int2']['size_mb']:.2f} MB "
                   f"({results['int2']['compression_ratio']:.2f}x, "
                   f"{results['int2']['memory_savings_pct']:.1f}% savings)")
        
        # Validate INT2 > INT8 compression
        assert results['int2']['compression_ratio'] > results['int8']['compression_ratio']


class TestAccuracyValidation:
    """Validate accuracy/precision targets."""
    
    def compute_cosine_similarity(
        self,
        original_weights: Dict[str, torch.Tensor],
        quantized_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute cosine similarity for each layer."""
        similarities = {}
        
        for layer_name in original_weights.keys():
            if layer_name not in quantized_weights:
                continue
            
            orig = original_weights[layer_name].cpu().numpy().flatten()
            quant = quantized_weights[layer_name].cpu().numpy().flatten()
            
            # Compute cosine similarity
            cosine_sim = np.dot(orig, quant) / (
                np.linalg.norm(orig) * np.linalg.norm(quant) + 1e-8
            )
            similarities[layer_name] = float(cosine_sim)
        
        return similarities
    
    def test_int8_accuracy_target(self, minilm_parquet, tmp_path):
        """Validate INT8 accuracy: cosine similarity > 0.85."""
        input_parquet, model, tokenizer = minilm_parquet
        
        # Quantize to INT8
        config = QuantizationConfig(quant_type='int8', per_channel=True)
        quantizer = ArrowQuantizer(config)
        
        output_parquet = str(tmp_path / "minilm_int8.parquet")
        quantizer.quantize_model(input_parquet, output_parquet, show_progress=False)
        
        # Load weights
        original_loader = WeightLoader(input_parquet)
        quantized_loader = WeightLoader(output_parquet)
        
        original_weights = original_loader.load_weights()
        quantized_weights = quantized_loader.load_weights()
        
        # Compute similarities
        similarities = self.compute_cosine_similarity(original_weights, quantized_weights)
        
        # Calculate average similarity
        avg_similarity = np.mean(list(similarities.values()))
        min_similarity = np.min(list(similarities.values()))
        
        logger.info(f"INT8 Accuracy Metrics:")
        logger.info(f"  Average cosine similarity: {avg_similarity:.4f}")
        logger.info(f"  Minimum cosine similarity: {min_similarity:.4f}")
        logger.info(f"  Layers with similarity < 0.85: "
                   f"{sum(1 for s in similarities.values() if s < 0.85)}/{len(similarities)}")
        
        # Validate target: average > 0.85
        assert avg_similarity > 0.85, f"INT8 average similarity {avg_similarity:.4f} < 0.85 target"
        
        # Calculate precision loss
        precision_loss_pct = (1 - avg_similarity) * 100
        logger.info(f"  Precision loss: {precision_loss_pct:.2f}%")
        
        # Validate PTQ baseline: < 15% loss
        assert precision_loss_pct < 15.0, f"INT8 precision loss {precision_loss_pct:.2f}% > 15% PTQ baseline"
    
    def test_int2_accuracy_target(self, minilm_parquet, tmp_path):
        """Validate INT2 accuracy: cosine similarity > 0.70 (relaxed for INT2)."""
        input_parquet, model, tokenizer = minilm_parquet
        
        # Quantize to INT2
        config = QuantizationConfig(quant_type='int2', per_channel=False, group_size=128)
        quantizer = ArrowQuantizer(config)
        
        output_parquet = str(tmp_path / "minilm_int2.parquet")
        quantizer.quantize_model(input_parquet, output_parquet, show_progress=False)
        
        # Load weights
        original_loader = WeightLoader(input_parquet)
        quantized_loader = WeightLoader(output_parquet)
        
        original_weights = original_loader.load_weights()
        quantized_weights = quantized_loader.load_weights()
        
        # Compute similarities
        similarities = self.compute_cosine_similarity(original_weights, quantized_weights)
        
        # Calculate average similarity
        avg_similarity = np.mean(list(similarities.values()))
        min_similarity = np.min(list(similarities.values()))
        
        logger.info(f"INT2 Accuracy Metrics:")
        logger.info(f"  Average cosine similarity: {avg_similarity:.4f}")
        logger.info(f"  Minimum cosine similarity: {min_similarity:.4f}")
        
        # Validate relaxed target for INT2: > 0.70
        assert avg_similarity > 0.70, f"INT2 average similarity {avg_similarity:.4f} < 0.70 target"
        
        # Calculate precision loss
        precision_loss_pct = (1 - avg_similarity) * 100
        logger.info(f"  Precision loss: {precision_loss_pct:.2f}%")
    
    def test_layer_wise_accuracy_distribution(self, minilm_parquet, tmp_path):
        """Analyze layer-wise accuracy distribution."""
        input_parquet, model, tokenizer = minilm_parquet
        
        # Quantize to INT8
        config = QuantizationConfig(quant_type='int8', per_channel=True)
        quantizer = ArrowQuantizer(config)
        
        output_parquet = str(tmp_path / "minilm_int8.parquet")
        quantizer.quantize_model(input_parquet, output_parquet, show_progress=False)
        
        # Load weights
        original_loader = WeightLoader(input_parquet)
        quantized_loader = WeightLoader(output_parquet)
        
        original_weights = original_loader.load_weights()
        quantized_weights = quantized_loader.load_weights()
        
        # Compute similarities
        similarities = self.compute_cosine_similarity(original_weights, quantized_weights)
        
        # Analyze distribution
        sim_values = list(similarities.values())
        logger.info("Layer-wise Accuracy Distribution:")
        logger.info(f"  Mean: {np.mean(sim_values):.4f}")
        logger.info(f"  Std: {np.std(sim_values):.4f}")
        logger.info(f"  Min: {np.min(sim_values):.4f}")
        logger.info(f"  Max: {np.max(sim_values):.4f}")
        logger.info(f"  Median: {np.median(sim_values):.4f}")
        
        # Find worst layers
        sorted_layers = sorted(similarities.items(), key=lambda x: x[1])
        logger.info("  Worst 5 layers:")
        for layer_name, sim in sorted_layers[:5]:
            logger.info(f"    {layer_name}: {sim:.4f}")


class TestSchemaCompatibility:
    """Test V1/V2 schema compatibility."""
    
    def test_v1_schema_read(self, minilm_parquet):
        """Test reading V1 schema."""
        input_parquet, model, tokenizer = minilm_parquet
        
        # Read V1 table
        table = pq.read_table(input_parquet)
        
        # Verify V1 schema
        assert detect_schema_version(table) == 1
        
        # Verify V1 columns
        expected_columns = {'layer_name', 'shape', 'dtype', 'data', 'num_params'}
        assert set(table.column_names) == expected_columns
    
    def test_v2_schema_write(self, minilm_parquet, tmp_path):
        """Test writing V2 schema."""
        input_parquet, model, tokenizer = minilm_parquet
        
        # Quantize to V2
        config = QuantizationConfig(quant_type='int8', per_channel=True)
        quantizer = ArrowQuantizer(config)
        
        output_parquet = str(tmp_path / "minilm_v2.parquet")
        quantizer.quantize_model(input_parquet, output_parquet, show_progress=False)
        
        # Read V2 table
        table = pq.read_table(output_parquet)
        
        # Verify V2 schema
        assert detect_schema_version(table) == 2
        
        # Verify V2 columns (superset of V1)
        expected_columns = {
            'layer_name', 'shape', 'dtype', 'data', 'num_params',
            'quant_type', 'scales', 'zero_points', 'quant_axis', 'group_size'
        }
        assert set(table.column_names) == expected_columns
    
    def test_v2_to_v1_compatibility(self, minilm_parquet, tmp_path):
        """Test that V2 can be read as V1 (backward compatibility)."""
        input_parquet, model, tokenizer = minilm_parquet
        
        # Quantize to V2
        config = QuantizationConfig(quant_type='int8', per_channel=True)
        quantizer = ArrowQuantizer(config)
        
        output_parquet = str(tmp_path / "minilm_v2.parquet")
        quantizer.quantize_model(input_parquet, output_parquet, show_progress=False)
        
        # Load with WeightLoader (should handle both V1 and V2)
        loader = WeightLoader(output_parquet)
        weights = loader.load_weights()
        
        # Verify weights loaded successfully
        assert len(weights) > 0
        
        # Verify all weights are tensors
        for layer_name, weight in weights.items():
            assert isinstance(weight, torch.Tensor)


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    def test_quantization_speed(self, minilm_parquet, tmp_path):
        """Benchmark quantization speed."""
        input_parquet, model, tokenizer = minilm_parquet
        
        results = {}
        
        # Benchmark INT8
        config_int8 = QuantizationConfig(quant_type='int8', per_channel=True)
        quantizer_int8 = ArrowQuantizer(config_int8)
        output_int8 = str(tmp_path / "minilm_int8.parquet")
        
        start_time = time.time()
        quantizer_int8.quantize_model(input_parquet, output_int8, show_progress=False)
        int8_time = time.time() - start_time
        results['int8'] = int8_time
        
        # Benchmark INT2
        config_int2 = QuantizationConfig(quant_type='int2', per_channel=False, group_size=128)
        quantizer_int2 = ArrowQuantizer(config_int2)
        output_int2 = str(tmp_path / "minilm_int2.parquet")
        
        start_time = time.time()
        quantizer_int2.quantize_model(input_parquet, output_int2, show_progress=False)
        int2_time = time.time() - start_time
        results['int2'] = int2_time
        
        # Log results
        logger.info("Quantization Speed Benchmarks:")
        logger.info(f"  INT8: {results['int8']:.2f}s")
        logger.info(f"  INT2: {results['int2']:.2f}s")
        
        # Validate reasonable performance (< 60s for MiniLM)
        assert results['int8'] < 60.0, f"INT8 quantization too slow: {results['int8']:.2f}s"
        assert results['int2'] < 60.0, f"INT2 quantization too slow: {results['int2']:.2f}s"
    
    def test_weight_loading_speed(self, minilm_parquet, tmp_path):
        """Benchmark weight loading speed."""
        input_parquet, model, tokenizer = minilm_parquet
        
        # Quantize first
        config = QuantizationConfig(quant_type='int8', per_channel=True)
        quantizer = ArrowQuantizer(config)
        output_parquet = str(tmp_path / "minilm_int8.parquet")
        quantizer.quantize_model(input_parquet, output_parquet, show_progress=False)
        
        # Benchmark loading
        loader = WeightLoader(output_parquet)
        
        start_time = time.time()
        weights = loader.load_weights()
        load_time = time.time() - start_time
        
        logger.info(f"Weight loading time: {load_time:.4f}s")
        logger.info(f"Loaded {len(weights)} layers")
        
        # Validate reasonable performance (< 5s for MiniLM)
        assert load_time < 5.0, f"Weight loading too slow: {load_time:.4f}s"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

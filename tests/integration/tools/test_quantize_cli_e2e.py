"""
End-to-End Integration Tests for Quantization CLI Tool

Tests the complete quantization workflow:
- CLI argument parsing
- Configuration loading from YAML
- Quantization pipeline execution
- Precision validation
- Report generation

These tests use real quantization on small test models to validate
the complete integration pipeline.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest
import yaml


class TestQuantizeCLIBasic:
    """Test basic CLI functionality"""
    
    def test_cli_help(self):
        """Test CLI help output"""
        result = subprocess.run(
            [sys.executable, "-m", "llm_compression.tools.quantize_cli", "--help"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "Quantize model weights" in result.stdout
        assert "--input" in result.stdout
        assert "--output" in result.stdout
        assert "--quant-type" in result.stdout
    
    def test_cli_quantize_help(self):
        """Test quantize subcommand help"""
        result = subprocess.run(
            [sys.executable, "-m", "llm_compression.tools.quantize_cli", "quantize", "--help"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "--validate" in result.stdout
        assert "--per-channel" in result.stdout
        assert "--input" in result.stdout
        assert "--output" in result.stdout


class TestQuantizeCLIExecution:
    """Test CLI execution with real quantization"""
    
    def test_quantize_int8_basic(self, sample_model_v1, tmp_path):
        """Test basic INT8 quantization via CLI"""
        output_path = tmp_path / "weights_int8.parquet"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", str(sample_model_v1),
                "--output", str(output_path),
                "--quant-type", "int8",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert output_path.exists()
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["success"] is True
        assert output["quantization"]["quant_type"] == "int8"
        assert output["compression"]["compression_ratio"] > 1.0
        assert output["compression"]["memory_savings_percent"] > 0
    
    def test_quantize_int8_with_validation(self, sample_model_v1, tmp_path):
        """Test INT8 quantization with precision validation"""
        output_path = tmp_path / "weights_int8.parquet"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", str(sample_model_v1),
                "--output", str(output_path),
                "--quant-type", "int8",
                "--validate",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["success"] is True
        assert output["validation"] is not None
        assert "cosine_similarity" in output["validation"]
        assert "passed" in output["validation"]
    
    def test_quantize_int2_basic(self, sample_model_v1, tmp_path):
        """Test INT2 quantization via CLI"""
        output_path = tmp_path / "weights_int2.parquet"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", str(sample_model_v1),
                "--output", str(output_path),
                "--quant-type", "int2",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert output_path.exists()
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["success"] is True
        assert output["quantization"]["quant_type"] == "int2"
        # INT2 should have higher compression than INT8
        assert output["compression"]["compression_ratio"] > 2.0
    
    def test_quantize_per_tensor(self, sample_model_v1, tmp_path):
        """Test per-tensor quantization"""
        output_path = tmp_path / "weights_per_tensor.parquet"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", str(sample_model_v1),
                "--output", str(output_path),
                "--quant-type", "int8",
                "--per-tensor",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["success"] is True
        assert output["quantization"]["per_channel"] is False
    
    def test_quantize_asymmetric(self, sample_model_v1, tmp_path):
        """Test asymmetric quantization"""
        output_path = tmp_path / "weights_asymmetric.parquet"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", str(sample_model_v1),
                "--output", str(output_path),
                "--quant-type", "int8",
                "--asymmetric",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["success"] is True
        assert output["quantization"]["symmetric"] is False
    
    def test_quantize_mixed_precision(self, sample_model_v1, tmp_path):
        """Test mixed precision quantization"""
        output_path = tmp_path / "weights_mixed.parquet"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", str(sample_model_v1),
                "--output", str(output_path),
                "--quant-type", "int8",
                "--mixed-precision", "embed", "lm_head",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["success"] is True


class TestQuantizeCLIConfiguration:
    """Test CLI configuration loading"""
    
    def test_load_config_from_yaml(self, sample_model_v1, tmp_path):
        """Test loading configuration from YAML file"""
        output_path = tmp_path / "weights_int8.parquet"
        config_path = tmp_path / "quantize_config.yaml"
        
        # Create config file
        config = {
            "quantization": {
                "input": str(sample_model_v1),
                "output": str(output_path),
                "quant_type": "int8",
                "calibration_method": "ptq",
                "per_channel": True,
                "symmetric": True,
            },
            "validation": {
                "enabled": False,
            },
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Run CLI with config file
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--config", str(config_path),
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert output_path.exists()
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["success"] is True
        assert output["quantization"]["quant_type"] == "int8"
    
    def test_cli_args_override_config(self, sample_model_v1, tmp_path):
        """Test that CLI arguments override config file"""
        output_path = tmp_path / "weights_int8.parquet"
        config_path = tmp_path / "quantize_config.yaml"
        
        # Create config file with INT8
        config = {
            "quantization": {
                "input": str(sample_model_v1),
                "output": str(output_path),
                "quant_type": "int8",
            },
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Run CLI with INT2 override
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--config", str(config_path),
                "--quant-type", "int2",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["success"] is True
        assert output["quantization"]["quant_type"] == "int2"


class TestQuantizeCLIValidation:
    """Test CLI validation functionality"""
    
    def test_validation_with_custom_threshold(self, sample_model_v1, tmp_path):
        """Test validation with custom cosine threshold"""
        output_path = tmp_path / "weights_int8.parquet"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", str(sample_model_v1),
                "--output", str(output_path),
                "--quant-type", "int8",
                "--validate",
                "--cosine-threshold", "0.90",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["validation"] is not None
    
    def test_validation_report_generation(self, sample_model_v1, tmp_path):
        """Test validation report generation"""
        output_path = tmp_path / "weights_int8.parquet"
        report_path = tmp_path / "validation_report.json"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", str(sample_model_v1),
                "--output", str(output_path),
                "--quant-type", "int8",
                "--validate",
                "--report", str(report_path),
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert report_path.exists()
        
        # Verify report content
        with open(report_path, "r") as f:
            report = json.load(f)
        
        assert "passed" in report
        assert "cosine_similarity" in report
        assert "num_samples" in report


class TestQuantizeCLIErrorHandling:
    """Test CLI error handling"""
    
    def test_missing_input_file(self, tmp_path):
        """Test error when input file is missing"""
        output_path = tmp_path / "weights_int8.parquet"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", "nonexistent.parquet",
                "--output", str(output_path),
                "--quant-type", "int8",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 1
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["success"] is False
        assert "error" in output
    
    def test_missing_required_args(self):
        """Test error when required arguments are missing"""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--quant-type", "int8",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 1
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()
    
    def test_invalid_quant_type(self, sample_model_v1, tmp_path):
        """Test error with invalid quantization type"""
        output_path = tmp_path / "weights_invalid.parquet"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", str(sample_model_v1),
                "--output", str(output_path),
                "--quant-type", "invalid",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode != 0


class TestQuantizeCLIOutput:
    """Test CLI output formatting"""
    
    def test_json_output_format(self, sample_model_v1, tmp_path):
        """Test JSON output format"""
        output_path = tmp_path / "weights_int8.parquet"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", str(sample_model_v1),
                "--output", str(output_path),
                "--quant-type", "int8",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        
        # Verify JSON is valid
        output = json.loads(result.stdout)
        
        # Check required fields
        assert "success" in output
        assert "input" in output
        assert "output" in output
        assert "quantization" in output
        assert "compression" in output
        assert "total_time_sec" in output
    
    def test_human_readable_output(self, sample_model_v1, tmp_path):
        """Test human-readable output format"""
        output_path = tmp_path / "weights_int8.parquet"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", str(sample_model_v1),
                "--output", str(output_path),
                "--quant-type", "int8",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        
        # Check for expected output sections
        assert "QUANTIZATION SUCCESSFUL" in result.stdout
        assert "Compression Metrics" in result.stdout
        assert "Performance" in result.stdout
    
    def test_verbose_output(self, sample_model_v1, tmp_path):
        """Test verbose logging output"""
        output_path = tmp_path / "weights_int8.parquet"
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", str(sample_model_v1),
                "--output", str(output_path),
                "--quant-type", "int8",
                "--verbose",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        # Verbose mode should produce more output
        assert len(result.stdout) > 0


@pytest.fixture
def sample_model_v1(tmp_path):
    """
    Create a small sample model in Parquet V1 format for testing.
    
    This creates a minimal model with a few layers to test quantization
    without requiring large model files.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    
    from llm_compression.inference.quantization_schema import WEIGHT_SCHEMA_V1
    
    # Create sample weights
    layers = [
        {
            "layer_name": "encoder.layer.0.weight",
            "shape": [128, 64],
            "dtype": "torch.float32",
            "data": torch.randn(128, 64).numpy().astype(np.float32).tobytes(),
            "num_params": 128 * 64,
        },
        {
            "layer_name": "encoder.layer.0.bias",
            "shape": [128],
            "dtype": "torch.float32",
            "data": torch.randn(128).numpy().astype(np.float32).tobytes(),
            "num_params": 128,
        },
        {
            "layer_name": "encoder.layer.1.weight",
            "shape": [64, 128],
            "dtype": "torch.float32",
            "data": torch.randn(64, 128).numpy().astype(np.float32).tobytes(),
            "num_params": 64 * 128,
        },
    ]
    
    # Write to Parquet
    model_path = tmp_path / "sample_model_v1.parquet"
    table = pa.Table.from_pylist(layers, schema=WEIGHT_SCHEMA_V1)
    pq.write_table(table, model_path)
    
    return model_path


@pytest.mark.integration
class TestQuantizeCLIEndToEnd:
    """End-to-end integration tests"""
    
    def test_complete_quantization_workflow(self, sample_model_v1, tmp_path):
        """Test complete quantization workflow with all features"""
        output_path = tmp_path / "weights_int8.parquet"
        report_path = tmp_path / "validation_report.json"
        
        # Run quantization with validation and report
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_compression.tools.quantize_cli",
                "quantize",
                "--input", str(sample_model_v1),
                "--output", str(output_path),
                "--quant-type", "int8",
                "--per-channel",
                "--symmetric",
                "--validate",
                "--report", str(report_path),
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        
        # Verify success
        assert result.returncode == 0
        assert output_path.exists()
        assert report_path.exists()
        
        # Verify output
        output = json.loads(result.stdout)
        assert output["success"] is True
        assert output["compression"]["compression_ratio"] > 1.0
        assert output["validation"]["passed"] is True
        
        # Verify report
        with open(report_path, "r") as f:
            report = json.load(f)
        assert report["passed"] is True
        assert report["cosine_similarity"] > 0.0

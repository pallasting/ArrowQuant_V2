"""
Unit tests for Whisper model conversion in ModelConverter.

Tests the Whisper-specific conversion functionality added to ModelConverter.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from llm_compression.tools.model_converter import (
    ModelConverter,
    ConversionConfig,
    ConversionResult,
)


class TestWhisperConversion:
    """Test Whisper model conversion functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Create mock Whisper model with encoder weights."""
        model = Mock()
        
        # Create mock state dict with encoder weights
        state_dict = {
            "encoder.conv1.weight": torch.randn(512, 80, 3),
            "encoder.conv1.bias": torch.randn(512),
            "encoder.conv2.weight": torch.randn(512, 512, 3),
            "encoder.conv2.bias": torch.randn(512),
            "encoder.embed_positions.weight": torch.randn(1500, 512),
            "encoder.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
            "encoder.layers.0.self_attn.k_proj.weight": torch.randn(512, 512),
            "encoder.layers.0.self_attn.v_proj.weight": torch.randn(512, 512),
            "encoder.layers.0.layer_norm1.weight": torch.randn(512),
            "encoder.layers.0.layer_norm1.bias": torch.randn(512),
            # Non-encoder weights (should be filtered out)
            "decoder.embed_tokens.weight": torch.randn(51865, 512),
            "decoder.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
        }
        model.state_dict.return_value = state_dict
        
        return model
    
    @pytest.fixture
    def mock_whisper_config(self):
        """Create mock Whisper config."""
        config = Mock()
        config.num_mel_bins = 80
        config.d_model = 512
        config.encoder_layers = 6
        config.encoder_attention_heads = 8
        config.encoder_ffn_dim = 2048
        config.max_source_positions = 1500
        return config
    
    def test_extract_whisper_weights(self, mock_whisper_model):
        """Test extraction of encoder weights from Whisper model."""
        converter = ModelConverter()
        weights = converter._extract_whisper_weights(mock_whisper_model)
        
        # Should extract only encoder weights
        assert len(weights) == 10  # 10 encoder weights (conv1 weight+bias, conv2 weight+bias, embed_positions, 5 layer weights)
        
        # Check that all extracted weights are encoder-related
        for key in weights.keys():
            assert key.startswith("encoder.")
        
        # Check that decoder weights are not included
        assert "decoder.embed_tokens.weight" not in weights
        assert "decoder.layers.0.self_attn.q_proj.weight" not in weights
    
    def test_map_whisper_keys(self):
        """Test Whisper key mapping (embed_positions → position_embedding)."""
        converter = ModelConverter()
        
        weights = {
            "encoder.conv1.weight": torch.randn(512, 80, 3),
            "encoder.embed_positions.weight": torch.randn(1500, 512),
            "encoder.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
        }
        
        mapped = converter._map_whisper_keys(weights)
        
        # Check that embed_positions was renamed
        assert "encoder.embed_positions.weight" not in mapped
        assert "encoder.position_embedding" in mapped
        
        # Check that other keys are unchanged
        assert "encoder.conv1.weight" in mapped
        assert "encoder.layers.0.self_attn.q_proj.weight" in mapped
        
        # Check tensor values are preserved
        assert torch.equal(
            mapped["encoder.position_embedding"],
            weights["encoder.embed_positions.weight"]
        )

    def test_generate_whisper_metadata(self, mock_whisper_config, temp_dir):
        """Test Whisper metadata generation."""
        converter = ModelConverter()
        
        weights = {
            "encoder.conv1.weight": torch.randn(512, 80, 3),
            "encoder.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
        }
        
        parquet_path = temp_dir / "weights.parquet"
        parquet_path.touch()  # Create empty file
        
        metadata = converter._generate_whisper_metadata(
            model_name="openai/whisper-base",
            config=mock_whisper_config,
            weights=weights,
            parquet_path=parquet_path,
        )
        
        # Check metadata structure
        assert metadata["model_name"] == "openai/whisper-base"
        assert metadata["model_type"] == "Whisper Audio Encoder"
        assert metadata["architecture"] == "Whisper"
        
        # Check config
        assert metadata["config"]["n_mels"] == 80
        assert metadata["config"]["hidden_size"] == 512
        assert metadata["config"]["num_layers"] == 6
        assert metadata["config"]["num_attention_heads"] == 8
        
        # Check parameters
        total_params = sum(t.numel() for t in weights.values())
        assert metadata["total_parameters"] == total_params
        assert metadata["num_weight_tensors"] == len(weights)
        
        # Check layer info
        assert "layer_info" in metadata
        assert len(metadata["layer_info"]) == len(weights)
    
    def test_load_whisper_model_import_error(self):
        """Test that _load_whisper_model raises ImportError when transformers not installed."""
        converter = ModelConverter()
        
        # Mock the import to fail
        with patch.dict('sys.modules', {'transformers': None}):
            with pytest.raises(ImportError, match="transformers not installed"):
                converter._load_whisper_model("openai/whisper-base")
    
    @patch.object(ModelConverter, "_load_whisper_model")
    @patch.object(ModelConverter, "_convert_to_arrow")
    def test_convert_whisper_success(
        self,
        mock_convert_to_arrow,
        mock_load_whisper_model,
        temp_dir,
        mock_whisper_model,
        mock_whisper_config
    ):
        """Test successful Whisper model conversion."""
        # Setup mocks
        mock_load_whisper_model.return_value = (mock_whisper_model, Mock(), mock_whisper_config)
        
        parquet_path = temp_dir / "weights.parquet"
        parquet_path.write_bytes(b"fake parquet data")
        mock_convert_to_arrow.return_value = parquet_path
        
        # Run conversion
        converter = ModelConverter(ConversionConfig(validate_output=False))
        result = converter._convert_whisper("openai/whisper-base", temp_dir)
        
        # Check result
        assert result.success
        assert result.model_name == "openai/whisper-base"
        assert result.output_dir == temp_dir
        assert result.parquet_path == parquet_path
        assert result.metadata_path == temp_dir / "metadata.json"
        assert result.total_parameters > 0
        assert result.conversion_time_sec > 0
        
        # Check metadata file was created
        assert result.metadata_path.exists()
        
        # Verify metadata content
        with open(result.metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata["model_type"] == "Whisper Audio Encoder"
        assert metadata["architecture"] == "Whisper"

    @patch.object(ModelConverter, "_load_whisper_model")
    def test_convert_whisper_failure(self, mock_load_whisper_model, temp_dir):
        """Test Whisper conversion failure handling."""
        # Setup mock to raise exception
        mock_load_whisper_model.side_effect = RuntimeError("Model loading failed")
        
        # Run conversion
        converter = ModelConverter()
        result = converter._convert_whisper("invalid/model", temp_dir)
        
        # Check result
        assert not result.success
        assert result.error_message == "Model loading failed"
        assert result.conversion_time_sec > 0
    
    @patch.object(ModelConverter, "_convert_whisper")
    def test_convert_routes_to_whisper(self, mock_convert_whisper, temp_dir):
        """Test that convert() routes to _convert_whisper for Whisper models."""
        mock_result = ConversionResult(
            success=True,
            model_name="openai/whisper-base",
            output_dir=temp_dir,
        )
        mock_convert_whisper.return_value = mock_result
        
        converter = ModelConverter()
        result = converter.convert(
            model_name_or_path="openai/whisper-base",
            output_dir=str(temp_dir),
            model_type="whisper"
        )
        
        # Check that _convert_whisper was called
        mock_convert_whisper.assert_called_once_with("openai/whisper-base", temp_dir)
        assert result == mock_result
    
    def test_whisper_weights_float16_conversion(self, mock_whisper_model):
        """Test float16 optimization for Whisper weights."""
        converter = ModelConverter(ConversionConfig(use_float16=True))
        
        # Extract weights
        weights = converter._extract_whisper_weights(mock_whisper_model)
        
        # Optimize to float16
        optimized = converter._optimize_weights(weights)
        
        # Check that float32 weights were converted to float16
        for name, tensor in optimized.items():
            if weights[name].dtype == torch.float32:
                assert tensor.dtype == torch.float16
    
    def test_whisper_conversion_parameter_count(self, mock_whisper_model):
        """Test that parameter count is calculated correctly."""
        converter = ModelConverter()
        weights = converter._extract_whisper_weights(mock_whisper_model)
        
        # Calculate expected parameter count
        expected_params = sum(t.numel() for t in weights.values())
        
        # Should match the sum of all encoder weights
        assert expected_params > 0
        
        # Verify specific weights are included
        assert "encoder.conv1.weight" in weights
        assert "encoder.conv2.weight" in weights
    
    def test_whisper_key_mapping_preserves_other_keys(self):
        """Test that key mapping only affects embed_positions."""
        converter = ModelConverter()
        
        weights = {
            "encoder.conv1.weight": torch.randn(512, 80, 3),
            "encoder.conv1.bias": torch.randn(512),
            "encoder.embed_positions.weight": torch.randn(1500, 512),
            "encoder.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
            "encoder.layers.0.layer_norm1.weight": torch.randn(512),
        }
        
        mapped = converter._map_whisper_keys(weights)
        
        # Check that only embed_positions was renamed
        assert len(mapped) == len(weights)
        assert "encoder.position_embedding" in mapped
        assert "encoder.embed_positions.weight" not in mapped
        
        # All other keys should be unchanged
        for key in weights.keys():
            if "embed_positions" not in key:
                assert key in mapped


def _has_transformers() -> bool:
    """Check if transformers is installed."""
    try:
        import transformers
        return True
    except ImportError:
        return False


class TestWhisperConversionIntegration:
    """Integration tests for Whisper conversion (requires transformers)."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        not _has_transformers(),
        reason="transformers not installed"
    )
    def test_whisper_conversion_real_model(self, temp_dir):
        """Test Whisper conversion with a real (small) model."""
        # This test would download a real model - skip in CI
        pytest.skip("Skipping real model download test")

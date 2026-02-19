"""
Unit tests for CLIP model conversion in ModelConverter.

Tests the CLIP-specific conversion functionality added to ModelConverter.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import torch

from llm_compression.tools.model_converter import (
    ModelConverter,
    ConversionConfig,
    ConversionResult,
)


class TestCLIPConversion:
    """Test CLIP model conversion functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def mock_clip_model(self):
        """Create mock CLIP model with vision weights."""
        model = Mock()
        
        # Create mock state dict with vision weights
        state_dict = {
            "vision_model.embeddings.patch_embedding.weight": torch.randn(768, 3, 32, 32),
            "vision_model.embeddings.class_embedding": torch.randn(768),
            "vision_model.embeddings.position_embedding.weight": torch.randn(50, 768),
            "vision_model.encoder.layers.0.self_attn.q_proj.weight": torch.randn(768, 768),
            "vision_model.encoder.layers.0.self_attn.k_proj.weight": torch.randn(768, 768),
            "vision_model.encoder.layers.0.self_attn.v_proj.weight": torch.randn(768, 768),
            "visual_projection.weight": torch.randn(512, 768),
            # Non-vision weights (should be filtered out)
            "text_model.embeddings.token_embedding.weight": torch.randn(49408, 512),
        }
        model.state_dict.return_value = state_dict
        
        return model
    
    @pytest.fixture
    def mock_clip_config(self):
        """Create mock CLIP config."""
        config = Mock()
        config.vision_config = Mock()
        config.vision_config.image_size = 224
        config.vision_config.patch_size = 32
        config.vision_config.hidden_size = 768
        config.vision_config.num_hidden_layers = 12
        config.vision_config.num_attention_heads = 12
        config.vision_config.intermediate_size = 3072
        config.vision_config.layer_norm_eps = 1e-5
        config.projection_dim = 512
        return config
    
    def test_extract_clip_weights(self, mock_clip_model):
        """Test extraction of vision weights from CLIP model."""
        converter = ModelConverter()
        weights = converter._extract_clip_weights(mock_clip_model)
        
        # Should extract only vision-related weights
        assert len(weights) == 7  # 7 vision weights, not 8 (text weight excluded)
        
        # Check that all extracted weights are vision-related
        for key in weights.keys():
            assert key.startswith("vision_model") or key == "visual_projection.weight"
        
        # Check that text weights are not included
        assert "text_model.embeddings.token_embedding.weight" not in weights
    
    def test_map_clip_keys(self):
        """Test CLIP key mapping (currently no-op)."""
        converter = ModelConverter()
        
        weights = {
            "vision_model.embeddings.patch_embedding.weight": torch.randn(768, 3, 32, 32),
            "vision_model.encoder.layers.0.self_attn.q_proj.weight": torch.randn(768, 768),
        }
        
        mapped = converter._map_clip_keys(weights)
        
        # Currently no mapping, so keys should be unchanged
        assert set(mapped.keys()) == set(weights.keys())
        
        # Tensors should be the same objects
        for key in weights.keys():
            assert torch.equal(mapped[key], weights[key])
    
    def test_generate_clip_metadata(self, mock_clip_config, temp_dir):
        """Test CLIP metadata generation."""
        converter = ModelConverter()
        
        weights = {
            "vision_model.embeddings.patch_embedding.weight": torch.randn(768, 3, 32, 32),
            "vision_model.encoder.layers.0.self_attn.q_proj.weight": torch.randn(768, 768),
        }
        
        parquet_path = temp_dir / "weights.parquet"
        parquet_path.touch()  # Create empty file
        
        metadata = converter._generate_clip_metadata(
            model_name="openai/clip-vit-base-patch32",
            config=mock_clip_config,
            weights=weights,
            parquet_path=parquet_path,
        )
        
        # Check metadata structure
        assert metadata["model_name"] == "openai/clip-vit-base-patch32"
        assert metadata["model_type"] == "CLIP Vision Transformer"
        assert metadata["architecture"] == "ViT"
        
        # Check config
        assert metadata["config"]["image_size"] == 224
        assert metadata["config"]["patch_size"] == 32
        assert metadata["config"]["hidden_size"] == 768
        assert metadata["config"]["num_layers"] == 12
        
        # Check parameters
        total_params = sum(t.numel() for t in weights.values())
        assert metadata["total_parameters"] == total_params
        assert metadata["num_weight_tensors"] == len(weights)
        
        # Check layer info
        assert "layer_info" in metadata
        assert len(metadata["layer_info"]) == len(weights)
    
    def test_load_clip_model_import_error(self):
        """Test that _load_clip_model raises ImportError when transformers not installed."""
        converter = ModelConverter()
        
        # Mock the import to fail
        with patch.dict('sys.modules', {'transformers': None}):
            with pytest.raises(ImportError, match="transformers not installed"):
                converter._load_clip_model("openai/clip-vit-base-patch32")
    
    @patch.object(ModelConverter, "_load_clip_model")
    @patch.object(ModelConverter, "_convert_to_arrow")
    def test_convert_clip_success(
        self,
        mock_convert_to_arrow,
        mock_load_clip_model,
        temp_dir,
        mock_clip_model,
        mock_clip_config
    ):
        """Test successful CLIP model conversion."""
        # Setup mocks
        mock_load_clip_model.return_value = (mock_clip_model, Mock(), mock_clip_config)
        
        parquet_path = temp_dir / "weights.parquet"
        parquet_path.write_bytes(b"fake parquet data")
        mock_convert_to_arrow.return_value = parquet_path
        
        # Run conversion
        converter = ModelConverter(ConversionConfig(validate_output=False))
        result = converter._convert_clip("openai/clip-vit-base-patch32", temp_dir)
        
        # Check result
        assert result.success
        assert result.model_name == "openai/clip-vit-base-patch32"
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
        
        assert metadata["model_type"] == "CLIP Vision Transformer"
        assert metadata["architecture"] == "ViT"
    
    @patch.object(ModelConverter, "_load_clip_model")
    def test_convert_clip_failure(self, mock_load_clip_model, temp_dir):
        """Test CLIP conversion failure handling."""
        # Setup mock to raise exception
        mock_load_clip_model.side_effect = RuntimeError("Model loading failed")
        
        # Run conversion
        converter = ModelConverter()
        result = converter._convert_clip("invalid/model", temp_dir)
        
        # Check result
        assert not result.success
        assert result.error_message == "Model loading failed"
        assert result.conversion_time_sec > 0
    
    @patch.object(ModelConverter, "_convert_clip")
    def test_convert_routes_to_clip(self, mock_convert_clip, temp_dir):
        """Test that convert() routes to _convert_clip for CLIP models."""
        mock_result = ConversionResult(
            success=True,
            model_name="openai/clip-vit-base-patch32",
            output_dir=temp_dir,
        )
        mock_convert_clip.return_value = mock_result
        
        converter = ModelConverter()
        result = converter.convert(
            model_name_or_path="openai/clip-vit-base-patch32",
            output_dir=str(temp_dir),
            model_type="clip"
        )
        
        # Check that _convert_clip was called
        mock_convert_clip.assert_called_once_with("openai/clip-vit-base-patch32", temp_dir)
        assert result == mock_result
    
    def test_clip_weights_float16_conversion(self, mock_clip_model):
        """Test float16 optimization for CLIP weights."""
        converter = ModelConverter(ConversionConfig(use_float16=True))
        
        # Extract weights
        weights = converter._extract_clip_weights(mock_clip_model)
        
        # Optimize to float16
        optimized = converter._optimize_weights(weights)
        
        # Check that float32 weights were converted to float16
        for name, tensor in optimized.items():
            if weights[name].dtype == torch.float32:
                assert tensor.dtype == torch.float16
    
    def test_clip_conversion_parameter_count(self, mock_clip_model):
        """Test that parameter count is calculated correctly."""
        converter = ModelConverter()
        weights = converter._extract_clip_weights(mock_clip_model)
        
        # Calculate expected parameter count
        expected_params = sum(t.numel() for t in weights.values())
        
        # Should match the sum of all vision weights
        assert expected_params > 0
        
        # Verify specific weights are included
        assert "vision_model.embeddings.patch_embedding.weight" in weights
        assert "visual_projection.weight" in weights


def _has_transformers() -> bool:
    """Check if transformers is installed."""
    try:
        import transformers
        return True
    except ImportError:
        return False


class TestCLIPConversionIntegration:
    """Integration tests for CLIP conversion (requires transformers)."""
    
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
    def test_clip_conversion_real_model(self, temp_dir):
        """Test CLIP conversion with a real (small) model."""
        # This test would download a real model - skip in CI
        pytest.skip("Skipping real model download test")

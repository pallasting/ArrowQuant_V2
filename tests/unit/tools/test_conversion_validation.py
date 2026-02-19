"""
Unit tests for model conversion validation.

Tests the validation script that compares embeddings between original
and converted models.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.validate_model_conversion import (
    compute_cosine_similarity,
    validate_clip_conversion,
    validate_whisper_conversion,
)


class TestCosineSimilarity:
    """Test cosine similarity computation."""
    
    def test_identical_vectors(self):
        """Test that identical vectors have similarity 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        similarity = compute_cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert abs(similarity) < 1e-6
    
    def test_opposite_vectors(self):
        """Test that opposite vectors have similarity -1.0."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6
    
    def test_similar_vectors(self):
        """Test that similar vectors have high similarity."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.1, 2.1, 2.9])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert similarity > 0.99


class TestCLIPValidation:
    """Test CLIP model validation."""
    
    @patch('scripts.validate_model_conversion.VisionEncoder')
    @patch('scripts.validate_model_conversion.CLIPProcessor')
    @patch('scripts.validate_model_conversion.CLIPModel')
    def test_validate_clip_basic(self, mock_model, mock_processor, mock_vision_encoder):
        """Test basic CLIP validation flow."""
        # Setup mocks
        mock_hf_model = Mock()
        mock_hf_processor = Mock()
        mock_arrow_encoder = Mock()
        
        mock_model.from_pretrained.return_value = mock_hf_model
        mock_processor.from_pretrained.return_value = mock_hf_processor
        mock_vision_encoder.from_pretrained.return_value = mock_arrow_encoder
        
        # Mock embeddings (very similar)
        base_embedding = np.random.randn(768).astype(np.float32)
        noise = np.random.randn(768).astype(np.float32) * 0.01  # Small noise
        
        mock_hf_output = Mock()
        mock_hf_output.cpu.return_value.numpy.return_value.flatten.return_value = base_embedding
        mock_hf_model.get_image_features.return_value = mock_hf_output
        
        mock_arrow_encoder.encode.return_value = base_embedding + noise
        
        # Mock vision_model for parameter counting
        mock_param = Mock()
        mock_param.numel.return_value = 1000000
        mock_hf_model.vision_model.parameters.return_value = [mock_param]
        
        # Mock file system
        mock_path = Mock()
        mock_parquet = Mock()
        mock_parquet.exists.return_value = True
        mock_parquet.stat.return_value.st_size = 10 * 1024 * 1024  # 10 MB
        mock_path.__truediv__ = Mock(return_value=mock_parquet)
        
        # Run validation
        results = validate_clip_conversion(
            model_name="openai/clip-vit-base-patch32",
            converted_path=mock_path,
            num_samples=5
        )
        
        # Verify results
        assert results["success"] is True
        assert results["model_type"] == "CLIP"
        assert results["num_samples"] == 5
        assert "avg_similarity" in results
        assert "compression_ratio" in results
    
    @patch('scripts.validate_model_conversion.VisionEncoder')
    @patch('scripts.validate_model_conversion.CLIPProcessor')
    @patch('scripts.validate_model_conversion.CLIPModel')
    def test_validate_clip_high_similarity(self, mock_model, mock_processor, mock_vision_encoder):
        """Test that high similarity passes validation."""
        # Setup mocks with identical embeddings
        mock_hf_model = Mock()
        mock_hf_processor = Mock()
        mock_arrow_encoder = Mock()
        
        mock_model.from_pretrained.return_value = mock_hf_model
        mock_processor.from_pretrained.return_value = mock_hf_processor
        mock_vision_encoder.from_pretrained.return_value = mock_arrow_encoder
        
        # Identical embeddings
        embedding = np.random.randn(768).astype(np.float32)
        
        mock_hf_output = Mock()
        mock_hf_output.cpu.return_value.numpy.return_value.flatten.return_value = embedding
        mock_hf_model.get_image_features.return_value = mock_hf_output
        
        mock_arrow_encoder.encode.return_value = embedding.copy()
        
        # Mock vision_model
        mock_param = Mock()
        mock_param.numel.return_value = 1000000
        mock_hf_model.vision_model.parameters.return_value = [mock_param]
        
        # Mock file system
        mock_path = Mock()
        mock_parquet = Mock()
        mock_parquet.exists.return_value = True
        mock_parquet.stat.return_value.st_size = 10 * 1024 * 1024
        mock_path.__truediv__ = Mock(return_value=mock_parquet)
        
        # Run validation
        results = validate_clip_conversion(
            model_name="openai/clip-vit-base-patch32",
            converted_path=mock_path,
            num_samples=3
        )
        
        # Should pass with similarity ~1.0
        assert results["success"] is True
        assert results["passed"] is True
        assert results["avg_similarity"] > 0.99


class TestWhisperValidation:
    """Test Whisper model validation."""
    
    @patch('scripts.validate_model_conversion.AudioEncoder')
    @patch('scripts.validate_model_conversion.WhisperProcessor')
    @patch('scripts.validate_model_conversion.WhisperModel')
    def test_validate_whisper_basic(self, mock_model, mock_processor, mock_audio_encoder):
        """Test basic Whisper validation flow."""
        # Setup mocks
        mock_hf_model = Mock()
        mock_hf_processor = Mock()
        mock_arrow_encoder = Mock()
        
        mock_model.from_pretrained.return_value = mock_hf_model
        mock_processor.from_pretrained.return_value = mock_hf_processor
        mock_audio_encoder.from_pretrained.return_value = mock_arrow_encoder
        
        # Mock embeddings
        base_embedding = np.random.randn(512).astype(np.float32)
        noise = np.random.randn(512).astype(np.float32) * 0.01
        
        # Mock HuggingFace output
        mock_encoder_output = Mock()
        mock_last_hidden_state = Mock()
        mock_last_hidden_state.mean.return_value.cpu.return_value.numpy.return_value.flatten.return_value = base_embedding
        mock_encoder_output.last_hidden_state = mock_last_hidden_state
        mock_hf_model.encoder.return_value = mock_encoder_output
        
        mock_arrow_encoder.encode.return_value = base_embedding + noise
        
        # Mock encoder for parameter counting
        mock_param = Mock()
        mock_param.numel.return_value = 500000
        mock_hf_model.encoder.parameters.return_value = [mock_param]
        
        # Mock file system
        mock_path = Mock()
        mock_parquet = Mock()
        mock_parquet.exists.return_value = True
        mock_parquet.stat.return_value.st_size = 5 * 1024 * 1024  # 5 MB
        mock_path.__truediv__ = Mock(return_value=mock_parquet)
        
        # Run validation
        results = validate_whisper_conversion(
            model_name="openai/whisper-base",
            converted_path=mock_path,
            num_samples=5
        )
        
        # Verify results
        assert results["success"] is True
        assert results["model_type"] == "Whisper"
        assert results["num_samples"] == 5
        assert "avg_similarity" in results
        assert "compression_ratio" in results
    
    @patch('scripts.validate_model_conversion.AudioEncoder')
    @patch('scripts.validate_model_conversion.WhisperProcessor')
    @patch('scripts.validate_model_conversion.WhisperModel')
    def test_validate_whisper_high_similarity(self, mock_model, mock_processor, mock_audio_encoder):
        """Test that high similarity passes validation."""
        # Setup mocks with identical embeddings
        mock_hf_model = Mock()
        mock_hf_processor = Mock()
        mock_arrow_encoder = Mock()
        
        mock_model.from_pretrained.return_value = mock_hf_model
        mock_processor.from_pretrained.return_value = mock_hf_processor
        mock_audio_encoder.from_pretrained.return_value = mock_arrow_encoder
        
        # Identical embeddings
        embedding = np.random.randn(512).astype(np.float32)
        
        # Mock HuggingFace output
        mock_encoder_output = Mock()
        mock_last_hidden_state = Mock()
        mock_last_hidden_state.mean.return_value.cpu.return_value.numpy.return_value.flatten.return_value = embedding
        mock_encoder_output.last_hidden_state = mock_last_hidden_state
        mock_hf_model.encoder.return_value = mock_encoder_output
        
        mock_arrow_encoder.encode.return_value = embedding.copy()
        
        # Mock encoder
        mock_param = Mock()
        mock_param.numel.return_value = 500000
        mock_hf_model.encoder.parameters.return_value = [mock_param]
        
        # Mock file system
        mock_path = Mock()
        mock_parquet = Mock()
        mock_parquet.exists.return_value = True
        mock_parquet.stat.return_value.st_size = 5 * 1024 * 1024
        mock_path.__truediv__ = Mock(return_value=mock_parquet)
        
        # Run validation
        results = validate_whisper_conversion(
            model_name="openai/whisper-base",
            converted_path=mock_path,
            num_samples=3
        )
        
        # Should pass with similarity ~1.0
        assert results["success"] is True
        assert results["passed"] is True
        assert results["avg_similarity"] > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

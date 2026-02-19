"""
Unit tests for model type auto-detection in ModelConverter

Tests the _detect_model_type method to ensure correct identification
of BERT, CLIP, and Whisper models from model names and configs.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from llm_compression.tools.model_converter import ModelConverter


class TestModelTypeDetection:
    """Test suite for model type auto-detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = ModelConverter()

    def test_detect_clip_from_name(self):
        """Test CLIP detection from model name."""
        # Test various CLIP model names
        clip_names = [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
            "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        ]
        
        for name in clip_names:
            result = self.converter._detect_model_type(name)
            assert result == "clip", f"Failed to detect CLIP from name: {name}"

    def test_detect_whisper_from_name(self):
        """Test Whisper detection from model name."""
        # Test various Whisper model names
        whisper_names = [
            "openai/whisper-base",
            "openai/whisper-small",
            "openai/whisper-large-v2",
        ]
        
        for name in whisper_names:
            result = self.converter._detect_model_type(name)
            assert result == "whisper", f"Failed to detect Whisper from name: {name}"

    def test_detect_bert_from_name(self):
        """Test BERT detection from model name."""
        # Test various BERT model names
        bert_names = [
            "bert-base-uncased",
            "sentence-transformers/all-MiniLM-L6-v2",
            "distilbert-base-uncased",
        ]
        
        for name in bert_names:
            result = self.converter._detect_model_type(name)
            assert result == "bert", f"Failed to detect BERT from name: {name}"

    def test_detect_clip_from_config(self):
        """Test CLIP detection from config when name is ambiguous."""
        # Mock AutoConfig to return a CLIP config
        mock_config = Mock()
        mock_config.model_type = "clip"
        
        with patch("transformers.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = mock_config
            
            result = self.converter._detect_model_type("some-ambiguous-model")
            assert result == "clip"

    def test_detect_whisper_from_config(self):
        """Test Whisper detection from config when name is ambiguous."""
        # Mock AutoConfig to return a Whisper config
        mock_config = Mock()
        mock_config.model_type = "whisper"
        
        with patch("transformers.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = mock_config
            
            result = self.converter._detect_model_type("some-ambiguous-model")
            assert result == "whisper"

    def test_detect_bert_from_config(self):
        """Test BERT detection from config when name is ambiguous."""
        # Test various BERT-like model types
        bert_types = ["bert", "roberta", "distilbert"]
        
        for model_type in bert_types:
            mock_config = Mock()
            mock_config.model_type = model_type
            
            with patch("transformers.AutoConfig") as mock_auto_config:
                mock_auto_config.from_pretrained.return_value = mock_config
                
                result = self.converter._detect_model_type("some-ambiguous-model")
                assert result == "bert", f"Failed to detect BERT from config.model_type: {model_type}"

    def test_detect_unknown_model(self):
        """Test unknown model detection when no patterns match."""
        # Mock AutoConfig to raise an exception
        with patch("transformers.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.side_effect = Exception("Model not found")
            
            result = self.converter._detect_model_type("unknown-model-xyz")
            assert result == "unknown"

    def test_detect_unknown_model_type_in_config(self):
        """Test unknown model when config has unsupported model_type."""
        # Mock AutoConfig to return an unsupported model type
        mock_config = Mock()
        mock_config.model_type = "gpt2"  # Unsupported type
        
        with patch("transformers.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = mock_config
            
            result = self.converter._detect_model_type("some-model")
            assert result == "unknown"

    def test_detect_case_insensitive(self):
        """Test that detection is case-insensitive."""
        # Test with various case combinations
        test_cases = [
            ("CLIP-vit-base", "clip"),
            ("Whisper-Base", "whisper"),
            ("BERT-base-uncased", "bert"),
        ]
        
        for name, expected_type in test_cases:
            result = self.converter._detect_model_type(name)
            assert result == expected_type, f"Failed case-insensitive detection for: {name}"

    def test_detect_with_config_no_model_type_attribute(self):
        """Test detection when config doesn't have model_type attribute."""
        # Mock AutoConfig to return a config without model_type
        mock_config = Mock(spec=[])  # Empty spec means no attributes
        
        with patch("transformers.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = mock_config
            
            result = self.converter._detect_model_type("some-model")
            assert result == "unknown"


class TestConvertWithAutoDetection:
    """Test suite for convert() method with auto-detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = ModelConverter()

    def test_convert_with_auto_clip(self):
        """Test conversion with auto-detection for CLIP model."""
        with patch.object(self.converter, '_detect_model_type', return_value='clip'):
            with patch.object(self.converter, '_convert_clip') as mock_convert_clip:
                mock_result = Mock()
                mock_convert_clip.return_value = mock_result
                
                result = self.converter.convert(
                    model_name_or_path="openai/clip-vit-base-patch32",
                    output_dir="test_output",
                    model_type="auto"
                )
                
                assert result == mock_result
                mock_convert_clip.assert_called_once()

    def test_convert_with_auto_whisper(self):
        """Test conversion with auto-detection for Whisper model."""
        with patch.object(self.converter, '_detect_model_type', return_value='whisper'):
            with patch.object(self.converter, '_convert_whisper') as mock_convert_whisper:
                mock_result = Mock()
                mock_convert_whisper.return_value = mock_result
                
                result = self.converter.convert(
                    model_name_or_path="openai/whisper-base",
                    output_dir="test_output",
                    model_type="auto"
                )
                
                assert result == mock_result
                mock_convert_whisper.assert_called_once()

    def test_convert_with_auto_unknown_raises_error(self):
        """Test that auto-detection of unknown model returns error in result."""
        with patch.object(self.converter, '_detect_model_type', return_value='unknown'):
            result = self.converter.convert(
                model_name_or_path="unknown-model",
                output_dir="test_output",
                model_type="auto"
            )
            
            # Should return a failed result, not raise exception
            assert result.success is False
            assert "Could not auto-detect model type" in result.error_message

    def test_convert_with_explicit_type_skips_detection(self):
        """Test that explicit model_type skips auto-detection."""
        with patch.object(self.converter, '_detect_model_type') as mock_detect:
            with patch.object(self.converter, '_convert_clip') as mock_convert_clip:
                mock_result = Mock()
                mock_convert_clip.return_value = mock_result
                
                result = self.converter.convert(
                    model_name_or_path="some-model",
                    output_dir="test_output",
                    model_type="clip"  # Explicit type
                )
                
                # Detection should not be called
                mock_detect.assert_not_called()
                # CLIP converter should be called
                mock_convert_clip.assert_called_once()

    def test_convert_with_unsupported_type_raises_error(self):
        """Test that unsupported model_type returns error in result."""
        result = self.converter.convert(
            model_name_or_path="some-model",
            output_dir="test_output",
            model_type="gpt2"  # Unsupported type
        )
        
        # Should return a failed result, not raise exception
        assert result.success is False
        assert "Unsupported model_type" in result.error_message

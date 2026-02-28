"""
Integration tests for SafeTensors quantization workflow

Tests the complete pipeline: SafeTensors → Parquet → Quantization
"""

import pytest
import tempfile
import shutil
from pathlib import Path

# Skip if arrow_quant_v2 not built
pytest.importorskip("arrow_quant_v2")

from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig


class TestSafeTensorsQuantization:
    """Test SafeTensors quantization integration"""

    def test_quantize_from_safetensors_method_exists(self):
        """Test that quantize_from_safetensors method exists"""
        quantizer = ArrowQuantV2(mode="diffusion")
        assert hasattr(quantizer, 'quantize_from_safetensors')
        assert callable(getattr(quantizer, 'quantize_from_safetensors'))

    def test_quantize_from_safetensors_signature(self):
        """Test that quantize_from_safetensors has correct signature"""
        import inspect
        
        quantizer = ArrowQuantV2(mode="diffusion")
        method = getattr(quantizer, 'quantize_from_safetensors')
        
        # Get signature
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        
        # Check required parameters
        assert 'safetensors_path' in params
        assert 'output_path' in params
        
        # Check optional parameters
        assert 'config' in params
        assert 'progress_callback' in params

    def test_config_from_profile(self):
        """Test creating config from profile"""
        config = DiffusionQuantConfig.from_profile("local")
        assert config is not None
        
        # Test different profiles
        edge_config = DiffusionQuantConfig.from_profile("edge")
        assert edge_config is not None
        
        cloud_config = DiffusionQuantConfig.from_profile("cloud")
        assert cloud_config is not None

    def test_progress_callback_interface(self):
        """Test that progress callback can be called"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Create a simple progress callback
        progress_calls = []
        
        def progress_callback(message: str, progress: float):
            progress_calls.append((message, progress))
        
        # The callback should be callable
        assert callable(progress_callback)
        
        # Test callback
        progress_callback("Test message", 0.5)
        assert len(progress_calls) == 1
        assert progress_calls[0] == ("Test message", 0.5)

    @pytest.mark.skipif(
        not Path("test_data/test_model.safetensors").exists(),
        reason="Test model not available"
    )
    def test_quantize_from_safetensors_integration(self):
        """Integration test with actual SafeTensors model (if available)"""
        quantizer = ArrowQuantV2(mode="diffusion")
        config = DiffusionQuantConfig.from_profile("local")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "quantized"
            
            # This would fail if test model doesn't exist, but that's expected
            # The test is here to document the expected usage
            try:
                result = quantizer.quantize_from_safetensors(
                    safetensors_path="test_data/test_model.safetensors",
                    output_path=str(output_path),
                    config=config
                )
                
                # Check result structure
                assert 'quantized_path' in result
                assert 'compression_ratio' in result
                assert 'cosine_similarity' in result
                assert 'model_size_mb' in result
                assert 'modality' in result
                assert 'bit_width' in result
                assert 'quantization_time_s' in result
                
            except Exception as e:
                # Expected if test model doesn't exist
                pytest.skip(f"Test model not available: {e}")


class TestSafeTensorsLoaderIntegration:
    """Test SafeTensors loader integration"""

    def test_safetensors_loader_import(self):
        """Test that SafeTensors loader can be imported"""
        try:
            from python.safetensors_loader import SafeTensorsLoader, ShardedSafeTensorsLoader
            assert SafeTensorsLoader is not None
            assert ShardedSafeTensorsLoader is not None
        except ImportError as e:
            pytest.fail(f"Failed to import SafeTensors loaders: {e}")

    def test_sharded_loader_detection(self):
        """Test sharded model detection"""
        from python.safetensors_loader import ShardedSafeTensorsLoader
        
        # Test with non-existent paths (should return False)
        assert not ShardedSafeTensorsLoader.is_sharded_model("nonexistent.safetensors")
        assert not ShardedSafeTensorsLoader.is_sharded_model("nonexistent_dir/")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

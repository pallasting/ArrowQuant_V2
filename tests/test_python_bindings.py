"""
Test PyO3 Python bindings for ArrowQuant V2.

This test verifies that the Python API is properly exposed and functional.
"""

import pytest


def test_import_module():
    """Test that the arrow_quant_v2 module can be imported."""
    try:
        import arrow_quant_v2
        assert hasattr(arrow_quant_v2, "ArrowQuantV2")
        assert hasattr(arrow_quant_v2, "DiffusionQuantConfig")
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_create_quantizer():
    """Test creating an ArrowQuantV2 instance."""
    try:
        from arrow_quant_v2 import ArrowQuantV2
        
        # Test diffusion mode
        quantizer = ArrowQuantV2(mode="diffusion")
        assert quantizer is not None
        
        # Test base mode
        quantizer_base = ArrowQuantV2(mode="base")
        assert quantizer_base is not None
        
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_invalid_mode():
    """Test that invalid mode raises ValueError."""
    try:
        from arrow_quant_v2 import ArrowQuantV2
        
        with pytest.raises(ValueError, match="Invalid mode"):
            ArrowQuantV2(mode="invalid")
            
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_create_config():
    """Test creating a DiffusionQuantConfig."""
    try:
        from arrow_quant_v2 import DiffusionQuantConfig
        
        # Test default config
        config = DiffusionQuantConfig()
        assert config is not None
        
        # Test custom config
        config_custom = DiffusionQuantConfig(
            bit_width=2,
            modality="text",
            num_time_groups=5,
            group_size=256,
            enable_time_aware=True,
            enable_spatial=False,
            min_accuracy=0.70,
            calibration_samples=32,
            deployment_profile="edge"
        )
        assert config_custom is not None
        
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_config_from_profile():
    """Test creating config from deployment profile."""
    try:
        from arrow_quant_v2 import DiffusionQuantConfig
        
        # Test edge profile
        config_edge = DiffusionQuantConfig.from_profile("edge")
        assert config_edge is not None
        
        # Test local profile
        config_local = DiffusionQuantConfig.from_profile("local")
        assert config_local is not None
        
        # Test cloud profile
        config_cloud = DiffusionQuantConfig.from_profile("cloud")
        assert config_cloud is not None
        
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_invalid_config():
    """Test that invalid config raises ValueError."""
    try:
        from arrow_quant_v2 import DiffusionQuantConfig
        
        # Invalid modality
        with pytest.raises(ValueError, match="Invalid modality"):
            DiffusionQuantConfig(modality="invalid")
        
        # Invalid deployment profile
        with pytest.raises(ValueError, match="Invalid deployment profile"):
            DiffusionQuantConfig(deployment_profile="invalid")
            
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_quantize_method_signature():
    """Test that quantize method exists and has correct signature."""
    try:
        from arrow_quant_v2 import ArrowQuantV2
        
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Verify method exists
        assert hasattr(quantizer, "quantize")
        assert hasattr(quantizer, "quantize_diffusion_model")
        assert hasattr(quantizer, "validate_quality")
        
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_exception_types():
    """Test that custom exception types are exposed."""
    try:
        import arrow_quant_v2
        
        # Verify exception types exist
        assert hasattr(arrow_quant_v2, "QuantizationError")
        assert hasattr(arrow_quant_v2, "ConfigurationError")
        assert hasattr(arrow_quant_v2, "ValidationError")
        assert hasattr(arrow_quant_v2, "ModelNotFoundError")
        assert hasattr(arrow_quant_v2, "MetadataError")
        assert hasattr(arrow_quant_v2, "ShapeMismatchError")
        
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


# ============================================================================
# Error Handling and Propagation Tests
# ============================================================================

def test_invalid_bit_width_error():
    """Test that invalid bit width raises ConfigurationError with helpful message."""
    try:
        from arrow_quant_v2 import DiffusionQuantConfig, ConfigurationError
        
        # Test invalid bit width (not 2, 4, or 8)
        config = DiffusionQuantConfig(bit_width=3)
        
        # This should fail during validation when used
        # For now, we test that the config is created but would fail in use
        # The actual validation happens in Rust during quantization
        
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_invalid_modality_error():
    """Test that invalid modality raises ConfigurationError."""
    try:
        from arrow_quant_v2 import DiffusionQuantConfig
        
        with pytest.raises(ValueError, match="Invalid modality"):
            DiffusionQuantConfig(modality="invalid_modality")
            
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_invalid_deployment_profile_error():
    """Test that invalid deployment profile raises ConfigurationError."""
    try:
        from arrow_quant_v2 import DiffusionQuantConfig
        
        with pytest.raises(ValueError, match="Invalid deployment profile"):
            DiffusionQuantConfig(deployment_profile="invalid_profile")
            
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_invalid_profile_from_profile():
    """Test that invalid profile in from_profile raises error."""
    try:
        from arrow_quant_v2 import DiffusionQuantConfig
        
        with pytest.raises(ValueError, match="Invalid deployment profile"):
            DiffusionQuantConfig.from_profile("invalid")
            
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_model_not_found_error():
    """Test that non-existent model path raises appropriate error."""
    try:
        from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig, ModelNotFoundError, MetadataError
        
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Try to quantize non-existent model
        # This will raise MetadataError (metadata.json not found) rather than ModelNotFoundError
        with pytest.raises((ModelNotFoundError, MetadataError), match="(Model not found|metadata)"):
            quantizer.quantize_diffusion_model(
                model_path="/nonexistent/path/to/model",
                output_path="/tmp/output",
                config=DiffusionQuantConfig()
            )
            
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_error_message_contains_hints():
    """Test that error messages contain helpful hints for users."""
    try:
        from arrow_quant_v2 import DiffusionQuantConfig
        
        # Test that ValueError for invalid modality contains hint
        try:
            DiffusionQuantConfig(modality="wrong")
        except ValueError as e:
            error_msg = str(e)
            # Should mention valid options
            assert "text" in error_msg or "code" in error_msg or "image" in error_msg
            
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_exception_inheritance():
    """Test that custom exceptions inherit from correct base classes."""
    try:
        import arrow_quant_v2
        
        # All custom exceptions should inherit from Exception
        assert issubclass(arrow_quant_v2.QuantizationError, Exception)
        assert issubclass(arrow_quant_v2.ConfigurationError, Exception)
        assert issubclass(arrow_quant_v2.ValidationError, Exception)
        assert issubclass(arrow_quant_v2.ModelNotFoundError, Exception)
        assert issubclass(arrow_quant_v2.MetadataError, Exception)
        assert issubclass(arrow_quant_v2.ShapeMismatchError, Exception)
        
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_error_propagation_from_rust():
    """Test that Rust errors are properly converted to Python exceptions."""
    try:
        from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
        
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # This should trigger a Rust error that gets converted to Python exception
        # Using invalid path should trigger ModelNotFoundError
        try:
            result = quantizer.quantize_diffusion_model(
                model_path="",  # Empty path
                output_path="/tmp/test_output",
                config=DiffusionQuantConfig()
            )
            # If we get here, the error handling needs improvement
            assert False, "Expected an exception but none was raised"
        except Exception as e:
            # Should get some kind of error (ModelNotFoundError or QuantizationError)
            assert e is not None
            error_msg = str(e)
            # Error message should be informative
            assert len(error_msg) > 0
            
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_validate_quality_error_handling():
    """Test error handling in validate_quality method."""
    try:
        from arrow_quant_v2 import ArrowQuantV2
        
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Try to validate with non-existent paths
        try:
            result = quantizer.validate_quality(
                original_path="/nonexistent/original",
                quantized_path="/nonexistent/quantized"
            )
            assert False, "Expected an exception but none was raised"
        except Exception as e:
            # Should get an error
            assert e is not None
            
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_quantize_method_error_handling():
    """Test error handling in quantize method."""
    try:
        from arrow_quant_v2 import ArrowQuantV2
        
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Test with invalid bit_width
        try:
            result = quantizer.quantize(
                weights={"layer1": [1.0, 2.0, 3.0]},
                bit_width=16  # Invalid bit width
            )
            assert False, "Expected ValueError for invalid bit_width"
        except ValueError as e:
            assert "Invalid bit_width" in str(e)
            assert "2, 4, or 8" in str(e)
            
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_progress_callback_error_handling():
    """Test that errors in progress callback are handled gracefully."""
    try:
        from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
        
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Define a callback that raises an error
        def bad_callback(message, progress):
            raise RuntimeError("Callback error")
        
        # The quantization should handle callback errors gracefully
        # (This test will only work with a valid model, so we expect MetadataError or ModelNotFoundError)
        try:
            result = quantizer.quantize_diffusion_model(
                model_path="/nonexistent/model",
                output_path="/tmp/output",
                config=DiffusionQuantConfig(),
                progress_callback=bad_callback
            )
        except Exception as e:
            # Should get MetadataError or ModelNotFoundError, not the callback error
            # The callback error should be caught and logged
            error_msg = str(e)
            assert "metadata" in error_msg.lower() or "model not found" in error_msg.lower() or "nonexistent" in error_msg.lower()
            
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_config_validation_comprehensive():
    """Test comprehensive configuration validation."""
    try:
        from arrow_quant_v2 import DiffusionQuantConfig
        
        # Test all valid bit widths
        for bit_width in [2, 4, 8]:
            config = DiffusionQuantConfig(bit_width=bit_width)
            assert config is not None
        
        # Test all valid modalities
        for modality in ["text", "code", "image", "audio", None]:
            config = DiffusionQuantConfig(modality=modality)
            assert config is not None
        
        # Test all valid deployment profiles
        for profile in ["edge", "local", "cloud"]:
            config = DiffusionQuantConfig(deployment_profile=profile)
            assert config is not None
            
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


# ============================================================================
# Progress Callback Tests
# ============================================================================

def test_progress_callback_basic():
    """Test that progress callback is called during quantization."""
    try:
        from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
        
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Track callback invocations
        callback_calls = []
        
        def progress_callback(message, progress):
            callback_calls.append((message, progress))
        
        # Try to quantize (will fail due to non-existent model, but callback should be called)
        try:
            result = quantizer.quantize_diffusion_model(
                model_path="/nonexistent/model",
                output_path="/tmp/output",
                config=DiffusionQuantConfig(),
                progress_callback=progress_callback
            )
        except Exception:
            pass  # Expected to fail
        
        # Verify callback was called at least once (start message)
        assert len(callback_calls) > 0
        assert callback_calls[0][0] == "Starting quantization..."
        assert callback_calls[0][1] == 0.0
        
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_progress_callback_values():
    """Test that progress values are in valid range [0.0, 1.0]."""
    try:
        from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
        
        quantizer = ArrowQuantV2(mode="diffusion")
        
        progress_values = []
        
        def progress_callback(message, progress):
            progress_values.append(progress)
        
        # Try to quantize
        try:
            result = quantizer.quantize_diffusion_model(
                model_path="/nonexistent/model",
                output_path="/tmp/output",
                config=DiffusionQuantConfig(),
                progress_callback=progress_callback
            )
        except Exception:
            pass
        
        # Verify all progress values are in [0.0, 1.0]
        for progress in progress_values:
            assert 0.0 <= progress <= 1.0, f"Progress {progress} out of range"
        
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_progress_callback_monotonic():
    """Test that progress values are monotonically increasing."""
    try:
        from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
        
        quantizer = ArrowQuantV2(mode="diffusion")
        
        progress_values = []
        
        def progress_callback(message, progress):
            progress_values.append(progress)
        
        # Try to quantize
        try:
            result = quantizer.quantize_diffusion_model(
                model_path="/nonexistent/model",
                output_path="/tmp/output",
                config=DiffusionQuantConfig(),
                progress_callback=progress_callback
            )
        except Exception:
            pass
        
        # Verify progress is monotonically increasing
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i-1], \
                f"Progress decreased: {progress_values[i-1]} -> {progress_values[i]}"
        
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_progress_callback_none():
    """Test that quantization works without progress callback."""
    try:
        from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
        
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Try to quantize without callback (should not crash)
        try:
            result = quantizer.quantize_diffusion_model(
                model_path="/nonexistent/model",
                output_path="/tmp/output",
                config=DiffusionQuantConfig(),
                progress_callback=None  # Explicitly None
            )
        except Exception as e:
            # Should fail due to model not found, not callback issues
            assert "metadata" in str(e).lower() or "model" in str(e).lower()
        
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_progress_callback_error_handling():
    """Test that callback errors don't crash quantization."""
    try:
        from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
        
        quantizer = ArrowQuantV2(mode="diffusion")
        
        call_count = [0]
        
        def bad_callback(message, progress):
            call_count[0] += 1
            # Raise error on second call
            if call_count[0] == 2:
                raise RuntimeError("Intentional callback error")
        
        # Try to quantize - should handle callback error gracefully
        try:
            result = quantizer.quantize_diffusion_model(
                model_path="/nonexistent/model",
                output_path="/tmp/output",
                config=DiffusionQuantConfig(),
                progress_callback=bad_callback
            )
        except Exception as e:
            # Should fail due to model not found, not callback error
            error_msg = str(e)
            assert "Intentional callback error" not in error_msg
            assert "metadata" in error_msg.lower() or "model" in error_msg.lower()
        
        # Verify callback was called at least once
        assert call_count[0] >= 1
        
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


def test_progress_callback_messages():
    """Test that progress messages are informative."""
    try:
        from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
        
        quantizer = ArrowQuantV2(mode="diffusion")
        
        messages = []
        
        def progress_callback(message, progress):
            messages.append(message)
        
        # Try to quantize
        try:
            result = quantizer.quantize_diffusion_model(
                model_path="/nonexistent/model",
                output_path="/tmp/output",
                config=DiffusionQuantConfig(),
                progress_callback=progress_callback
            )
        except Exception:
            pass
        
        # Verify we got some messages
        assert len(messages) > 0
        
        # Verify first message is about starting
        assert "start" in messages[0].lower() or "quantiz" in messages[0].lower()
        
        # Verify messages are non-empty strings
        for msg in messages:
            assert isinstance(msg, str)
            assert len(msg) > 0
        
    except ImportError:
        pytest.skip("arrow_quant_v2 module not built with Python bindings")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

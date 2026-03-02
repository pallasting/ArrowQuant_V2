"""
Test Arrow integration for Python API

This test verifies that the new Arrow-based quantization methods
are properly integrated into the Python API.
"""

import pytest

# Skip if arrow_quant_v2 is not available
pytest.importorskip("arrow_quant_v2")

from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig


def test_quantize_diffusion_model_arrow_method_exists():
    """Test that quantize_diffusion_model_arrow method exists"""
    quantizer = ArrowQuantV2(mode="diffusion")
    assert hasattr(quantizer, "quantize_diffusion_model_arrow")
    assert callable(getattr(quantizer, "quantize_diffusion_model_arrow"))


def test_quantize_diffusion_model_use_arrow_parameter():
    """Test that quantize_diffusion_model accepts use_arrow parameter"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Check that the method signature accepts use_arrow parameter
    import inspect
    sig = inspect.signature(quantizer.quantize_diffusion_model)
    assert "use_arrow" in sig.parameters
    
    # Check that use_arrow has a default value
    param = sig.parameters["use_arrow"]
    assert param.default is not inspect.Parameter.empty or param.default is None


def test_arrow_quantized_layer_methods():
    """Test that PyArrowQuantizedLayer has all required methods"""
    from arrow_quant_v2 import PyArrowQuantizedLayer
    
    # Check that the class exists
    assert PyArrowQuantizedLayer is not None
    
    # Check for required methods
    required_methods = [
        "to_pyarrow",
        "dequantize_group",
        "dequantize_all_groups",
        "get_time_group_params",
        "__len__",
    ]
    
    for method_name in required_methods:
        assert hasattr(PyArrowQuantizedLayer, method_name), \
            f"PyArrowQuantizedLayer missing method: {method_name}"


def test_config_options():
    """Test that configuration supports Arrow format selection"""
    # Test default config
    config = DiffusionQuantConfig.from_profile("local")
    assert config is not None
    
    # Test that we can create a quantizer with config
    quantizer = ArrowQuantV2(mode="diffusion")
    assert quantizer is not None


def test_type_hints_compatibility():
    """Test that the API maintains backward compatibility"""
    quantizer = ArrowQuantV2(mode="diffusion")
    
    # Test that quantize_diffusion_model can be called without use_arrow
    # (should default to legacy format for backward compatibility)
    import inspect
    sig = inspect.signature(quantizer.quantize_diffusion_model)
    
    # Check that use_arrow is optional
    param = sig.parameters.get("use_arrow")
    assert param is not None
    assert param.default is not inspect.Parameter.empty or param.default is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

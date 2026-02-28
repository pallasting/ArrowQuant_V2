"""
Unit tests for numpy_to_arrow_table() helper function.

Tests cover:
- 1D array conversion
- Multi-dimensional array conversion and shape preservation
- Invalid input handling (non-numpy arrays, wrong dtype)
- Empty dictionary and single layer dictionary
- Edge cases

Requirements tested: 5.1, 5.2, 5.4
"""

import pytest
import numpy as np
import pyarrow as pa

# Import the helper function
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from arrow_helpers import numpy_to_arrow_table


class TestNumpyToArrowTableBasic:
    """Test basic functionality of numpy_to_arrow_table()."""
    
    def test_single_layer_1d_array(self):
        """Test conversion of single layer with 1D array - Requirement 5.1, 5.2"""
        weights_dict = {
            "layer.0.weight": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        }
        
        table = numpy_to_arrow_table(weights_dict)
        
        # Verify table structure
        assert table.num_rows == 1
        assert table.num_columns == 3
        assert "layer_name" in table.column_names
        assert "weights" in table.column_names
        assert "shape" in table.column_names
        
        # Verify data
        assert table.column("layer_name")[0].as_py() == "layer.0.weight"
        assert table.column("weights")[0].as_py() == [1.0, 2.0, 3.0, 4.0]
        assert table.column("shape")[0].as_py() == [4]
    
    def test_multiple_layers_1d_arrays(self):
        """Test conversion of multiple layers with 1D arrays - Requirement 5.2"""
        weights_dict = {
            "layer.0.weight": np.array([1.0, 2.0], dtype=np.float32),
            "layer.1.weight": np.array([3.0, 4.0, 5.0], dtype=np.float32),
            "layer.2.weight": np.array([6.0], dtype=np.float32),
        }
        
        table = numpy_to_arrow_table(weights_dict)
        
        # Verify table structure
        assert table.num_rows == 3
        
        # Verify all layers present (order may vary due to dict)
        layer_names = [table.column("layer_name")[i].as_py() for i in range(3)]
        assert set(layer_names) == {"layer.0.weight", "layer.1.weight", "layer.2.weight"}
    
    def test_2d_array_flattening(self):
        """Test that 2D arrays are flattened and shape is preserved - Requirement 5.4"""
        weights_dict = {
            "layer.0.weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        }
        
        table = numpy_to_arrow_table(weights_dict)
        
        # Verify flattening
        weights = table.column("weights")[0].as_py()
        assert weights == [1.0, 2.0, 3.0, 4.0]
        
        # Verify shape preservation
        shape = table.column("shape")[0].as_py()
        assert shape == [2, 2]
    
    def test_3d_array_flattening(self):
        """Test that 3D arrays are flattened and shape is preserved - Requirement 5.4"""
        # Create 2x3x4 array
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights_dict = {"layer.0.weight": arr}
        
        table = numpy_to_arrow_table(weights_dict)
        
        # Verify flattening
        weights = table.column("weights")[0].as_py()
        assert len(weights) == 24
        assert weights == list(range(24))
        
        # Verify shape preservation
        shape = table.column("shape")[0].as_py()
        assert shape == [2, 3, 4]
    
    def test_empty_dictionary(self):
        """Test handling of empty dictionary - Requirement 5.2"""
        weights_dict = {}
        
        table = numpy_to_arrow_table(weights_dict)
        
        # Should create empty table with correct schema
        assert table.num_rows == 0
        assert table.num_columns == 3
        assert "layer_name" in table.column_names
        assert "weights" in table.column_names
        assert "shape" in table.column_names


class TestNumpyToArrowTableValidation:
    """Test input validation of numpy_to_arrow_table()."""
    
    def test_non_numpy_array_rejected(self):
        """Test that non-numpy arrays are rejected - Requirement 5.1"""
        weights_dict = {
            "layer.0.weight": [1.0, 2.0, 3.0]  # Python list, not numpy array
        }
        
        with pytest.raises(ValueError) as exc_info:
            numpy_to_arrow_table(weights_dict)
        
        assert "not a numpy array" in str(exc_info.value)
        assert "layer.0.weight" in str(exc_info.value)
    
    def test_wrong_dtype_rejected(self):
        """Test that non-float32 dtype is rejected - Requirement 5.1"""
        weights_dict = {
            "layer.0.weight": np.array([1.0, 2.0, 3.0], dtype=np.float64)
        }
        
        with pytest.raises(ValueError) as exc_info:
            numpy_to_arrow_table(weights_dict)
        
        assert "dtype" in str(exc_info.value).lower()
        assert "float32" in str(exc_info.value)
        assert "layer.0.weight" in str(exc_info.value)
    
    def test_int_dtype_rejected(self):
        """Test that integer dtype is rejected - Requirement 5.1"""
        weights_dict = {
            "layer.0.weight": np.array([1, 2, 3], dtype=np.int32)
        }
        
        with pytest.raises(ValueError) as exc_info:
            numpy_to_arrow_table(weights_dict)
        
        assert "dtype" in str(exc_info.value).lower()
        assert "float32" in str(exc_info.value)
    
    def test_empty_array_rejected(self):
        """Test that empty arrays are rejected - Requirement 5.1"""
        weights_dict = {
            "layer.0.weight": np.array([], dtype=np.float32)
        }
        
        with pytest.raises(ValueError) as exc_info:
            numpy_to_arrow_table(weights_dict)
        
        assert "empty" in str(exc_info.value).lower()
        assert "layer.0.weight" in str(exc_info.value)
    
    def test_non_dict_input_rejected(self):
        """Test that non-dictionary input is rejected - Requirement 5.1"""
        weights_list = [np.array([1.0, 2.0], dtype=np.float32)]
        
        with pytest.raises(ValueError) as exc_info:
            numpy_to_arrow_table(weights_list)
        
        assert "dictionary" in str(exc_info.value).lower()
    
    def test_non_string_layer_name_rejected(self):
        """Test that non-string layer names are rejected - Requirement 5.1"""
        weights_dict = {
            123: np.array([1.0, 2.0], dtype=np.float32)
        }
        
        with pytest.raises(ValueError) as exc_info:
            numpy_to_arrow_table(weights_dict)
        
        assert "string" in str(exc_info.value).lower()


class TestNumpyToArrowTableSchema:
    """Test that output schema matches quantize_batch_arrow() requirements."""
    
    def test_schema_has_correct_columns(self):
        """Test that schema has required columns - Requirement 5.5"""
        weights_dict = {
            "layer.0.weight": np.array([1.0, 2.0], dtype=np.float32)
        }
        
        table = numpy_to_arrow_table(weights_dict)
        schema = table.schema
        
        # Verify column names
        assert schema.names == ["layer_name", "weights", "shape"]
    
    def test_schema_has_correct_types(self):
        """Test that schema has correct types - Requirement 5.5"""
        weights_dict = {
            "layer.0.weight": np.array([1.0, 2.0], dtype=np.float32)
        }
        
        table = numpy_to_arrow_table(weights_dict)
        schema = table.schema
        
        # Verify types
        assert pa.types.is_string(schema.field("layer_name").type) or \
               pa.types.is_large_string(schema.field("layer_name").type)
        
        weights_type = schema.field("weights").type
        assert pa.types.is_list(weights_type)
        assert pa.types.is_float32(weights_type.value_type)
        
        shape_type = schema.field("shape").type
        assert pa.types.is_list(shape_type)
        assert pa.types.is_int64(shape_type.value_type)
    
    def test_compatible_with_quantize_batch_arrow(self):
        """Test that output is compatible with quantize_batch_arrow() - Requirement 5.5"""
        # This test verifies the schema is correct by attempting to use it
        # with quantize_batch_arrow() if available
        try:
            from arrow_quant_v2 import ArrowQuantV2
            
            weights_dict = {
                "layer.0.weight": np.random.randn(100).astype(np.float32),
                "layer.1.weight": np.random.randn(200).astype(np.float32),
            }
            
            table = numpy_to_arrow_table(weights_dict)
            quantizer = ArrowQuantV2(mode="diffusion")
            
            # Check if quantize_batch_arrow method exists
            if not hasattr(quantizer, 'quantize_batch_arrow'):
                pytest.skip("quantize_batch_arrow() not yet implemented")
            
            # Should not raise an error
            result = quantizer.quantize_batch_arrow(table, bit_width=4)
            
            # Verify result
            assert result is not None
            assert result.num_rows == 2
            
        except ImportError:
            pytest.skip("arrow_quant_v2 not available, skipping integration test")


class TestNumpyToArrowTableEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_large_array(self):
        """Test with large array - Requirement 5.2"""
        # 1M elements
        large_array = np.random.randn(1000000).astype(np.float32)
        weights_dict = {"large.layer": large_array}
        
        table = numpy_to_arrow_table(weights_dict)
        
        assert table.num_rows == 1
        assert len(table.column("weights")[0].as_py()) == 1000000
        assert table.column("shape")[0].as_py() == [1000000]
    
    def test_single_element_array(self):
        """Test with single element array - Requirement 5.2"""
        weights_dict = {
            "tiny.layer": np.array([42.0], dtype=np.float32)
        }
        
        table = numpy_to_arrow_table(weights_dict)
        
        assert table.num_rows == 1
        assert table.column("weights")[0].as_py() == [42.0]
        assert table.column("shape")[0].as_py() == [1]
    
    def test_special_layer_names(self):
        """Test with special characters in layer names - Requirement 5.2"""
        weights_dict = {
            "model.layers.0.self_attn.q_proj.weight": np.array([1.0], dtype=np.float32),
            "layer-with-dashes": np.array([2.0], dtype=np.float32),
            "layer_with_underscores": np.array([3.0], dtype=np.float32),
        }
        
        table = numpy_to_arrow_table(weights_dict)
        
        assert table.num_rows == 3
        layer_names = [table.column("layer_name")[i].as_py() for i in range(3)]
        assert "model.layers.0.self_attn.q_proj.weight" in layer_names
        assert "layer-with-dashes" in layer_names
        assert "layer_with_underscores" in layer_names
    
    def test_preserves_array_values(self):
        """Test that array values are preserved exactly - Requirement 5.3"""
        original_values = [1.5, -2.3, 0.0, 999.999, -0.001]
        weights_dict = {
            "layer.0": np.array(original_values, dtype=np.float32)
        }
        
        table = numpy_to_arrow_table(weights_dict)
        result_values = table.column("weights")[0].as_py()
        
        # Values should match (within float32 precision - 4 decimals is appropriate for float32)
        np.testing.assert_array_almost_equal(result_values, original_values, decimal=4)
    
    def test_multiple_layers_different_shapes(self):
        """Test multiple layers with different shapes - Requirement 5.4"""
        weights_dict = {
            "layer.0": np.random.randn(10).astype(np.float32),
            "layer.1": np.random.randn(5, 4).astype(np.float32),
            "layer.2": np.random.randn(2, 3, 4).astype(np.float32),
        }
        
        table = numpy_to_arrow_table(weights_dict)
        
        assert table.num_rows == 3
        
        # Find each layer and verify shape
        for i in range(3):
            layer_name = table.column("layer_name")[i].as_py()
            shape = table.column("shape")[i].as_py()
            weights = table.column("weights")[i].as_py()
            
            if layer_name == "layer.0":
                assert shape == [10]
                assert len(weights) == 10
            elif layer_name == "layer.1":
                assert shape == [5, 4]
                assert len(weights) == 20
            elif layer_name == "layer.2":
                assert shape == [2, 3, 4]
                assert len(weights) == 24


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Property-Based Test for Arrow Batch API Result Schema Completeness

**Feature: arrow-batch-api-unification, Property 4: Result Schema Completeness**

**Validates: Requirements 4.1, 4.5**

This property test verifies that for any successful quantization,
the result RecordBatch contains all required columns (layer_name,
quantized_data, scales, zero_points, shape, bit_width) with correct types.
"""

import pytest
import numpy as np
import pyarrow as pa
from hypothesis import given, strategies as st, settings
from arrow_quant_v2 import ArrowQuantV2


# Strategy for generating valid layer names
layer_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Ll', 'Nd'), whitelist_characters='._'),
    min_size=1,
    max_size=50
)

# Strategy for generating valid weights (finite float32 values)
weights_strategy = st.lists(
    st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
        width=32
    ),
    min_size=100,
    max_size=1000
)

# Strategy for generating valid shapes
shape_strategy = st.lists(
    st.integers(min_value=1, max_value=100),
    min_size=1,
    max_size=3
)

# Strategy for bit width
bit_width_strategy = st.sampled_from([2, 4, 8])


def create_arrow_table(layers_data):
    """
    Create an Arrow Table from layer data.
    
    Args:
        layers_data: List of tuples (layer_name, weights, shape)
    
    Returns:
        Arrow Table with required schema
    """
    layer_names = []
    weights_lists = []
    shapes_lists = []
    
    for layer_name, weights, shape in layers_data:
        layer_names.append(layer_name)
        weights_lists.append(weights)
        shapes_lists.append(shape)
    
    table = pa.Table.from_arrays(
        [
            pa.array(layer_names),
            pa.array(weights_lists, type=pa.list_(pa.float32())),
            pa.array(shapes_lists, type=pa.list_(pa.int64())),
        ],
        names=["layer_name", "weights", "shape"]
    )
    
    return table


def verify_result_schema(result_table):
    """
    Verify that the result table has all required columns with correct types.
    
    Args:
        result_table: Arrow Table to verify
    
    Raises:
        AssertionError: If schema is invalid
    """
    # Verify all required columns exist
    required_columns = {
        "layer_name",
        "quantized_data",
        "scales",
        "zero_points",
        "shape",
        "bit_width",
    }
    
    actual_columns = set(result_table.schema.names)
    assert required_columns == actual_columns, (
        f"Missing or extra columns. Expected: {required_columns}, Got: {actual_columns}"
    )
    
    # Verify column types
    schema = result_table.schema
    
    # layer_name should be string
    layer_name_field = schema.field("layer_name")
    assert pa.types.is_string(layer_name_field.type) or pa.types.is_large_string(layer_name_field.type), (
        f"layer_name should be string type, got {layer_name_field.type}"
    )
    
    # quantized_data should be binary
    quantized_data_field = schema.field("quantized_data")
    assert pa.types.is_binary(quantized_data_field.type) or pa.types.is_large_binary(quantized_data_field.type), (
        f"quantized_data should be binary type, got {quantized_data_field.type}"
    )
    
    # scales should be list<float32>
    scales_field = schema.field("scales")
    assert pa.types.is_list(scales_field.type) or pa.types.is_large_list(scales_field.type), (
        f"scales should be list type, got {scales_field.type}"
    )
    assert pa.types.is_float32(scales_field.type.value_type), (
        f"scales should contain float32, got {scales_field.type.value_type}"
    )
    
    # zero_points should be list<float32>
    zero_points_field = schema.field("zero_points")
    assert pa.types.is_list(zero_points_field.type) or pa.types.is_large_list(zero_points_field.type), (
        f"zero_points should be list type, got {zero_points_field.type}"
    )
    assert pa.types.is_float32(zero_points_field.type.value_type), (
        f"zero_points should contain float32, got {zero_points_field.type.value_type}"
    )
    
    # shape should be list<int64>
    shape_field = schema.field("shape")
    assert pa.types.is_list(shape_field.type) or pa.types.is_large_list(shape_field.type), (
        f"shape should be list type, got {shape_field.type}"
    )
    assert pa.types.is_int64(shape_field.type.value_type), (
        f"shape should contain int64, got {shape_field.type.value_type}"
    )
    
    # bit_width should be uint8
    bit_width_field = schema.field("bit_width")
    assert pa.types.is_uint8(bit_width_field.type), (
        f"bit_width should be uint8 type, got {bit_width_field.type}"
    )


class TestResultSchemaProperties:
    """Property-based tests for result schema completeness."""
    
    @given(
        num_layers=st.integers(min_value=1, max_value=10),
        bit_width=bit_width_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_prop_result_schema_complete(self, num_layers, bit_width):
        """
        **Validates: Requirements 4.1, 4.5**
        
        Property: Result Schema Completeness
        
        For any successful quantization, the result RecordBatch should contain
        all required columns (layer_name, quantized_data, scales, zero_points,
        shape, bit_width) with correct types.
        
        This property test verifies:
        1. All required columns are present in the result
        2. Column types match the expected schema
        3. layer_name is string type
        4. quantized_data is binary type
        5. scales is list<float32> type
        6. zero_points is list<float32> type
        7. shape is list<int64> type
        8. bit_width is uint8 type
        """
        # Generate layer data
        layers_data = []
        for i in range(num_layers):
            layer_name = f"layer_{i}"
            
            # Generate weights with reasonable size
            weights_len = 100 + (i * 100) % 900
            weights = [(j * 0.01) % 10.0 - 5.0 for j in range(weights_len)]
            
            # 1D shape for simplicity
            shape = [weights_len]
            
            layers_data.append((layer_name, weights, shape))
        
        # Create Arrow Table
        table = create_arrow_table(layers_data)
        
        # Create quantizer
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Quantize
        result_table = quantizer.quantize_batch_arrow(table, bit_width=bit_width)
        
        # Verify result schema
        verify_result_schema(result_table)
        
        # Verify number of rows matches input
        assert result_table.num_rows == num_layers, (
            f"Result should have {num_layers} rows, got {result_table.num_rows}"
        )
        
        # Verify bit_width column values
        bit_width_col = result_table.column("bit_width").to_pylist()
        for i, bw in enumerate(bit_width_col):
            assert bw == bit_width, (
                f"Row {i}: bit_width should be {bit_width}, got {bw}"
            )
    
    @given(
        num_layers=st.integers(min_value=1, max_value=5),
        bit_width=bit_width_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_prop_result_schema_complete_multidim(self, num_layers, bit_width):
        """
        **Validates: Requirements 4.1, 4.5**
        
        Property: Result Schema Completeness with Multi-dimensional Shapes
        
        Verifies that result schema is complete even when input has
        multi-dimensional shapes (2D, 3D, etc.).
        """
        # Generate layer data with multi-dimensional shapes
        layers_data = []
        for i in range(num_layers):
            layer_name = f"layer_{i}"
            
            # Generate 2D shape
            rows = 10 + (i * 5) % 20
            cols = 20 + (i * 10) % 30
            total_size = rows * cols
            
            # Generate weights
            weights = [(j * 0.01) % 10.0 - 5.0 for j in range(total_size)]
            
            # 2D shape
            shape = [rows, cols]
            
            layers_data.append((layer_name, weights, shape))
        
        # Create Arrow Table
        table = create_arrow_table(layers_data)
        
        # Create quantizer
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Quantize
        result_table = quantizer.quantize_batch_arrow(table, bit_width=bit_width)
        
        # Verify schema
        verify_result_schema(result_table)
        
        # Verify shapes are preserved
        shape_col = result_table.column("shape").to_pylist()
        for i, result_shape in enumerate(shape_col):
            expected_shape = layers_data[i][2]
            assert result_shape == expected_shape, (
                f"Row {i}: shape should be {expected_shape}, got {result_shape}"
            )
    
    @given(
        num_layers=st.integers(min_value=2, max_value=8),
        bit_width=bit_width_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_prop_result_schema_varying_sizes(self, num_layers, bit_width):
        """
        **Validates: Requirements 4.1, 4.5**
        
        Property: Result Schema Completeness with Varying Layer Sizes
        
        Verifies that result schema is complete even when layers have
        vastly different sizes (small to large).
        """
        # Generate layers with exponentially increasing sizes
        layers_data = []
        for i in range(num_layers):
            layer_name = f"layer_{i}"
            
            # Exponentially increasing size: 10, 100, 1000, etc.
            size = 10 ** ((i % 4) + 1)
            
            weights = [(j * 0.01) % 10.0 - 5.0 for j in range(size)]
            shape = [size]
            
            layers_data.append((layer_name, weights, shape))
        
        # Create Arrow Table
        table = create_arrow_table(layers_data)
        
        # Create quantizer
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Quantize
        result_table = quantizer.quantize_batch_arrow(table, bit_width=bit_width)
        
        # Verify schema
        verify_result_schema(result_table)
        
        # Verify all layers are present
        layer_names = result_table.column("layer_name").to_pylist()
        assert len(layer_names) == num_layers, (
            f"All {num_layers} layers should be present, got {len(layer_names)}"
        )
        
        # Verify quantized_data is not empty for any layer
        quantized_col = result_table.column("quantized_data")
        for i in range(result_table.num_rows):
            data = quantized_col[i].as_py()
            assert len(data) > 0, (
                f"Row {i}: quantized_data should not be empty"
            )
    
    @given(
        num_layers=st.integers(min_value=1, max_value=5),
        bit_width=bit_width_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_prop_result_data_validity(self, num_layers, bit_width):
        """
        **Validates: Requirements 4.1, 4.5**
        
        Property: Result Data Validity
        
        Verifies that all result data is valid:
        - scales and zero_points are finite
        - quantized_data is not empty
        - shapes match input
        """
        # Generate layer data
        layers_data = []
        for i in range(num_layers):
            layer_name = f"layer_{i}"
            weights_len = 100 + (i * 50)
            weights = [(j * 0.01) % 10.0 - 5.0 for j in range(weights_len)]
            shape = [weights_len]
            layers_data.append((layer_name, weights, shape))
        
        # Create Arrow Table
        table = create_arrow_table(layers_data)
        
        # Create quantizer
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Quantize
        result_table = quantizer.quantize_batch_arrow(table, bit_width=bit_width)
        
        # Verify schema first
        verify_result_schema(result_table)
        
        # Extract result data
        result_list = result_table.to_pylist()
        
        for i, row in enumerate(result_list):
            # Verify scales are finite and non-empty
            scales = row["scales"]
            assert len(scales) > 0, f"Row {i}: scales should not be empty"
            assert all(np.isfinite(s) for s in scales), (
                f"Row {i}: all scales should be finite"
            )
            
            # Verify zero_points are finite and non-empty
            zero_points = row["zero_points"]
            assert len(zero_points) > 0, f"Row {i}: zero_points should not be empty"
            assert all(np.isfinite(zp) for zp in zero_points), (
                f"Row {i}: all zero_points should be finite"
            )
            
            # Verify quantized_data is not empty
            quantized_data = row["quantized_data"]
            assert len(quantized_data) > 0, (
                f"Row {i}: quantized_data should not be empty"
            )
            
            # Verify bit_width matches
            assert row["bit_width"] == bit_width, (
                f"Row {i}: bit_width should be {bit_width}, got {row['bit_width']}"
            )
            
            # Verify shape matches input
            expected_shape = layers_data[i][2]
            assert row["shape"] == expected_shape, (
                f"Row {i}: shape should be {expected_shape}, got {row['shape']}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

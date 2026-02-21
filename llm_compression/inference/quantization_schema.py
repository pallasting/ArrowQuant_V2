"""
Parquet Schema definitions for quantized model weights.

This module defines Schema V1 (FP16/FP32) and Schema V2 (quantized) for
storing model weights in Arrow/Parquet format.

Schema V1 (Legacy - FP16/FP32):
- layer_name: string
- shape: list<int32>
- dtype: string
- data: binary (raw bytes)
- num_params: int64

Schema V2 (Quantized):
- layer_name: string
- shape: list<int32>
- dtype: string (original dtype before quantization)
- data: binary (packed int2/int4 or int8 data)
- num_params: int64
- quant_type: string ('int8', 'int2', 'fp16', 'fp32')
- scales: binary (FP32 scale array)
- zero_points: binary (FP32 zero_point array)
- quant_axis: int32 (-1 for per-tensor, 0+ for per-channel/per-group)
- group_size: int32 (group size for per-group quantization, 0 for per-tensor)

Version Detection:
- V1: No 'quant_type' column
- V2: Has 'quant_type' column
"""

from typing import Literal
import pyarrow as pa

from llm_compression.logger import logger


# Schema V1 (Legacy - FP16/FP32)
WEIGHT_SCHEMA_V1 = pa.schema([
    ('layer_name', pa.string()),
    ('shape', pa.list_(pa.int32())),
    ('dtype', pa.string()),
    ('data', pa.binary()),
    ('num_params', pa.int64()),
])


# Schema V2 (Quantized)
WEIGHT_SCHEMA_V2 = pa.schema([
    ('layer_name', pa.string()),
    ('shape', pa.list_(pa.int32())),
    ('dtype', pa.string()),  # Original dtype before quantization
    ('data', pa.binary()),  # Quantized data (packed for int2/int4, int8 for int8)
    ('num_params', pa.int64()),
    ('quant_type', pa.string()),  # 'int8', 'int2', 'fp16', 'fp32'
    ('scales', pa.binary()),  # FP32 scale array (binary for efficiency)
    ('zero_points', pa.binary()),  # FP32 zero_point array (binary for efficiency)
    ('quant_axis', pa.int32()),  # -1 for per-tensor, 0+ for per-channel/per-group
    ('group_size', pa.int32()),  # Group size for per-group quantization (0 for per-tensor)
])


QuantType = Literal['int8', 'int2', 'fp16', 'fp32']


def detect_schema_version(table: pa.Table) -> int:
    """
    Detect Parquet schema version.
    
    Args:
        table: PyArrow table
        
    Returns:
        Schema version (1 or 2)
        
    Example:
        >>> table = pq.read_table("weights.parquet")
        >>> version = detect_schema_version(table)
        >>> print(f"Schema version: {version}")
        Schema version: 2
    """
    # Check for V2-specific column
    if 'quant_type' in table.schema.names:
        logger.debug("Detected Schema V2 (quantized)")
        return 2
    else:
        logger.debug("Detected Schema V1 (FP16/FP32)")
        return 1


def validate_schema_v1(table: pa.Table) -> bool:
    """
    Validate that table conforms to Schema V1.
    
    Args:
        table: PyArrow table
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = {'layer_name', 'shape', 'dtype', 'data', 'num_params'}
    actual_columns = set(table.schema.names)
    
    if not required_columns.issubset(actual_columns):
        missing = required_columns - actual_columns
        logger.error(f"Schema V1 validation failed: missing columns {missing}")
        return False
    
    logger.debug("Schema V1 validation passed")
    return True


def validate_schema_v2(table: pa.Table) -> bool:
    """
    Validate that table conforms to Schema V2.
    
    Args:
        table: PyArrow table
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = {
        'layer_name', 'shape', 'dtype', 'data', 'num_params',
        'quant_type', 'scales', 'zero_points', 'quant_axis', 'group_size'
    }
    actual_columns = set(table.schema.names)
    
    if not required_columns.issubset(actual_columns):
        missing = required_columns - actual_columns
        logger.error(f"Schema V2 validation failed: missing columns {missing}")
        return False
    
    logger.debug("Schema V2 validation passed")
    return True


def get_schema_for_version(version: int) -> pa.Schema:
    """
    Get PyArrow schema for a specific version.
    
    Args:
        version: Schema version (1 or 2)
        
    Returns:
        PyArrow schema
        
    Raises:
        ValueError: If version is not 1 or 2
    """
    if version == 1:
        return WEIGHT_SCHEMA_V1
    elif version == 2:
        return WEIGHT_SCHEMA_V2
    else:
        raise ValueError(f"Invalid schema version: {version}. Must be 1 or 2.")


def is_quantized_layer(table: pa.Table, row_idx: int) -> bool:
    """
    Check if a specific layer is quantized (Schema V2 only).
    
    Args:
        table: PyArrow table (must be V2)
        row_idx: Row index
        
    Returns:
        True if layer is quantized (int8 or int2), False if FP16/FP32
    """
    if 'quant_type' not in table.schema.names:
        return False
    
    quant_type = table['quant_type'][row_idx].as_py()
    return quant_type in ['int8', 'int2']


def get_quantization_info(table: pa.Table, row_idx: int) -> dict:
    """
    Get quantization information for a layer (Schema V2 only).
    
    Args:
        table: PyArrow table (must be V2)
        row_idx: Row index
        
    Returns:
        Dictionary with quantization info:
        - quant_type: 'int8', 'int2', 'fp16', or 'fp32'
        - scales: Binary data (FP32 array)
        - zero_points: Binary data (FP32 array)
        - quant_axis: Quantization axis (-1 for per-tensor)
        - group_size: Group size (0 for per-tensor)
        
    Raises:
        ValueError: If table is not Schema V2
    """
    if 'quant_type' not in table.schema.names:
        raise ValueError("Table is not Schema V2 (no quant_type column)")
    
    return {
        'quant_type': table['quant_type'][row_idx].as_py(),
        'scales': table['scales'][row_idx].as_py(),
        'zero_points': table['zero_points'][row_idx].as_py(),
        'quant_axis': table['quant_axis'][row_idx].as_py(),
        'group_size': table['group_size'][row_idx].as_py() if 'group_size' in table.schema.names else 0,
    }


def create_v1_row(
    layer_name: str,
    shape: list,
    dtype: str,
    data: bytes,
    num_params: int
) -> dict:
    """
    Create a Schema V1 row dictionary.
    
    Args:
        layer_name: Layer name
        shape: Weight shape
        dtype: Data type string (e.g., 'torch.float32')
        data: Raw weight bytes
        num_params: Number of parameters
        
    Returns:
        Row dictionary compatible with Schema V1
    """
    return {
        'layer_name': layer_name,
        'shape': shape,
        'dtype': dtype,
        'data': data,
        'num_params': num_params,
    }


def create_v2_row(
    layer_name: str,
    shape: list,
    dtype: str,
    data: bytes,
    num_params: int,
    quant_type: QuantType,
    scales: bytes,
    zero_points: bytes,
    quant_axis: int,
    group_size: int = 0
) -> dict:
    """
    Create a Schema V2 row dictionary.
    
    Args:
        layer_name: Layer name
        shape: Weight shape
        dtype: Original data type before quantization
        data: Quantized weight bytes (packed for int2/int4)
        num_params: Number of parameters
        quant_type: Quantization type ('int8', 'int2', 'fp16', 'fp32')
        scales: Scaling factors as binary (FP32 array)
        zero_points: Zero points as binary (FP32 array)
        quant_axis: Quantization axis (-1 for per-tensor, 0+ for per-channel/per-group)
        group_size: Group size for per-group quantization (0 for per-tensor)
        
    Returns:
        Row dictionary compatible with Schema V2
    """
    return {
        'layer_name': layer_name,
        'shape': shape,
        'dtype': dtype,
        'data': data,
        'num_params': num_params,
        'quant_type': quant_type,
        'scales': scales,
        'zero_points': zero_points,
        'quant_axis': quant_axis,
        'group_size': group_size,
    }

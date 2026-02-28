"""
Arrow Helper Functions for ArrowQuant V2

This module provides helper functions for converting numpy arrays to Arrow Tables
for use with the zero-copy quantization API.
"""

import numpy as np
import pyarrow as pa
from typing import Dict


def numpy_to_arrow_table(weights_dict: Dict[str, np.ndarray]) -> pa.Table:
    """
    Convert numpy array dictionary to Arrow Table for quantize_batch_arrow().
    
    This function validates input arrays, extracts shapes, and creates an Arrow Table
    with the correct schema required by quantize_batch_arrow(). It leverages PyArrow's
    zero-copy numpy access capability for efficient data transfer.
    
    Args:
        weights_dict: Dictionary mapping layer names to numpy float32 arrays.
                     Arrays can be multi-dimensional and will be flattened.
    
    Returns:
        Arrow Table with schema:
            - layer_name: string
            - weights: list<float32>
            - shape: list<int64>
    
    Raises:
        ValueError: If input validation fails (non-numpy array, wrong dtype, etc.)
    
    Example:
        >>> import numpy as np
        >>> weights = {
        ...     "layer.0.weight": np.random.randn(4096, 4096).astype(np.float32),
        ...     "layer.1.weight": np.random.randn(4096, 11008).astype(np.float32),
        ... }
        >>> table = numpy_to_arrow_table(weights)
        >>> # Now use with quantize_batch_arrow()
        >>> quantizer = ArrowQuantV2(mode="diffusion")
        >>> result = quantizer.quantize_batch_arrow(table, bit_width=4)
    
    Requirements:
        - Validates: Requirement 5.1 (provides helper function)
        - Validates: Requirement 5.2 (converts numpy dict to Arrow Table)
        - Validates: Requirement 5.3 (leverages zero-copy numpy access)
        - Validates: Requirement 5.4 (handles multi-dimensional arrays)
        - Validates: Requirement 5.5 (returns correct schema)
    """
    # Validate input is a dictionary
    if not isinstance(weights_dict, dict):
        raise ValueError(
            f"Expected dictionary of numpy arrays, got {type(weights_dict).__name__}"
        )
    
    # Prepare lists for Arrow Table construction
    layer_names = []
    weights_lists = []
    shapes_lists = []
    
    for layer_name, weights_array in weights_dict.items():
        # Validate layer name is a string
        if not isinstance(layer_name, str):
            raise ValueError(
                f"Layer name must be a string, got {type(layer_name).__name__} for key {layer_name}"
            )
        
        # Validate input is a numpy array (Requirement 5.1)
        if not isinstance(weights_array, np.ndarray):
            raise ValueError(
                f"Layer '{layer_name}' is not a numpy array, got {type(weights_array).__name__}"
            )
        
        # Validate array dtype is float32 (Requirement 5.1)
        if weights_array.dtype != np.float32:
            raise ValueError(
                f"Layer '{layer_name}' has dtype {weights_array.dtype}, expected float32. "
                f"Please convert using: array.astype(np.float32)"
            )
        
        # Validate array is not empty
        if weights_array.size == 0:
            raise ValueError(
                f"Layer '{layer_name}' has empty array (size=0)"
            )
        
        # Extract shape before flattening (Requirement 5.4)
        original_shape = list(weights_array.shape)
        
        # Flatten multi-dimensional array (Requirement 5.4)
        flat_weights = weights_array.flatten()
        
        # Append to lists
        layer_names.append(layer_name)
        # PyArrow will use zero-copy access to numpy array (Requirement 5.3)
        weights_lists.append(flat_weights)
        shapes_lists.append(original_shape)
    
    # Build Arrow Table with correct schema (Requirement 5.5)
    # PyArrow's from_arrays uses zero-copy for numpy arrays internally
    table = pa.Table.from_arrays(
        [
            pa.array(layer_names),
            pa.array(weights_lists, type=pa.list_(pa.float32())),
            pa.array(shapes_lists, type=pa.list_(pa.int64())),
        ],
        names=["layer_name", "weights", "shape"]
    )
    
    return table

"""
Zero-copy weight loader for Arrow/Parquet model storage.

This module provides efficient weight loading from Parquet files using
memory-mapped Arrow tables for instant model loading without deserialization.

Key features:
- Zero-copy memory mapping (< 100ms load time)
- Direct Arrow Buffer to PyTorch Tensor conversion
- Lazy loading support (load layers on demand)
- Automatic dtype conversion (float16/float32)
- Weight caching for repeated access

Performance:
- Traditional PyTorch load: 2-5 seconds
- ArrowEngine WeightLoader: < 100ms (20-50x faster)
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from llm_compression.logger import logger
from llm_compression.inference.quantization_schema import detect_schema_version


class WeightLoader:
    """
    Zero-copy weight loader from Arrow/Parquet format.
    
    Loads model weights from Parquet files using memory-mapped Arrow tables
    for instant loading without full deserialization. Weights are lazily
    converted to PyTorch tensors only when accessed.
    
    Features:
    - Memory-mapped Parquet reading (zero-copy)
    - Direct Buffer to Tensor conversion (no intermediate copies)
    - Layer-wise lazy loading
    - Automatic dtype handling (float16/float32)
    - Weight caching
    
    Performance Target:
    - Load time: < 100ms for 100MB model
    - Memory overhead: < 10% of model size
    
    Example:
        >>> loader = WeightLoader("./models/minilm/weights.parquet")
        >>> weights = loader.load_weights()  # < 100ms
        >>> layer = loader.get_layer("encoder.layer.0.attention.self.query.weight")
    """
    
    def __init__(
        self,
        parquet_path: str,
        use_memory_map: bool = True,
        device: str = "cpu",
        cache_weights: bool = True,
        force_float32: bool = True,
    ):
        """
        Initialize WeightLoader.

        Args:
            parquet_path: Path to weights.parquet file
            use_memory_map: Use memory-mapped reading (recommended)
            device: Device to load weights to ("cpu", "cuda", "mps")
            cache_weights: Cache loaded weights in memory
            force_float32: Upcast float16 weights to float32 on load.
                           Prevents cumulative precision loss through deep
                           transformer layers when weights were saved as float16.
        """
        self.parquet_path = Path(parquet_path)
        self.use_memory_map = use_memory_map
        self.device = device
        self.cache_weights = cache_weights
        self.force_float32 = force_float32
        
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        # Cache for loaded weights
        self._weight_cache: Dict[str, torch.Tensor] = {}
        
        # Load Arrow table (memory-mapped, no deserialization)
        self._table: Optional[pa.Table] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._layer_name_map: Optional[Dict[str, int]] = None
        self._schema_version: Optional[int] = None
        
        logger.info(f"Initialized WeightLoader for {parquet_path}")
        logger.debug(f"Memory map: {use_memory_map}, Device: {device}, Cache: {cache_weights}")
    
    def _load_table(self) -> pa.Table:
        """
        Load Arrow table from Parquet with memory mapping.
        
        Returns:
            Memory-mapped Arrow table
        """
        if self._table is None:
            start_time = time.time()
            
            self._table = pq.read_table(
                self.parquet_path,
                memory_map=self.use_memory_map,
            )
            
            load_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Loaded Arrow table in {load_time_ms:.2f}ms (memory_map={self.use_memory_map})")
            
            # Detect schema version
            self._schema_version = detect_schema_version(self._table)
            logger.info(f"Detected Parquet Schema V{self._schema_version}")
            
            # Cache layer name to row index mapping for O(1) lookup
            self._layer_name_map = {
                self._table['layer_name'][i].as_py(): i 
                for i in range(len(self._table))
            }
            
            logger.debug(f"Table schema: {self._table.schema}")
            logger.debug(f"Table rows: {len(self._table)}")
        
        return self._table
    
    def load_weights(self) -> Dict[str, torch.Tensor]:
        """
        Load all weights from Parquet to PyTorch tensors.
        
        This method loads all model weights at once. For large models,
        consider using get_layer() for lazy loading.
        
        Returns:
            Dictionary mapping layer names to PyTorch tensors
            
        Example:
            >>> loader = WeightLoader("weights.parquet")
            >>> weights = loader.load_weights()
            >>> print(weights.keys())
            dict_keys(['layer1.weight', 'layer1.bias', ...])
        """
        start_time = time.time()
        
        # Load Arrow table (memory-mapped)
        table = self._load_table()
        
        weights = {}
        
        # Convert each row to a PyTorch tensor
        for i in range(len(table)):
            layer_name = table['layer_name'][i].as_py()
            
            # Check cache first
            if self.cache_weights and layer_name in self._weight_cache:
                weights[layer_name] = self._weight_cache[layer_name]
                continue
            
            # Convert Arrow data to PyTorch tensor
            tensor = self._row_to_tensor(table, i)
            
            # Cache if enabled
            if self.cache_weights:
                self._weight_cache[layer_name] = tensor
            
            weights[layer_name] = tensor
        
        load_time_ms = (time.time() - start_time) * 1000
        total_params = sum(t.numel() for t in weights.values())
        memory_mb = sum(t.numel() * t.element_size() for t in weights.values()) / (1024 * 1024)
        
        logger.info(f"Loaded {len(weights)} layers in {load_time_ms:.2f}ms")
        logger.info(f"Total parameters: {total_params:,}, Memory: {memory_mb:.2f}MB")
        
        return weights
    
    def get_layer(self, layer_name: str) -> torch.Tensor:
        """
        Get a specific layer's weights (lazy loading).
        
        Loads only the requested layer from Parquet, useful for
        large models or selective layer loading.
        
        Args:
            layer_name: Name of the layer to load
            
        Returns:
            PyTorch tensor for the layer
            
        Raises:
            KeyError: If layer_name not found in weights
            
        Example:
            >>> loader = WeightLoader("weights.parquet")
            >>> query_weight = loader.get_layer("encoder.layer.0.attention.self.query.weight")
        """
        # Check cache first
        if self.cache_weights and layer_name in self._weight_cache:
            logger.debug(f"Returning cached layer: {layer_name}")
            return self._weight_cache[layer_name]
        
        # Load table and find layer
        table = self._load_table()
        
        # Find row index for layer_name using cached map
        if layer_name not in self._layer_name_map:
            raise KeyError(f"Layer '{layer_name}' not found in weights.")
        
        row_idx = self._layer_name_map[layer_name]
        
        # Convert to tensor
        tensor = self._row_to_tensor(table, row_idx)
        
        # Cache if enabled
        if self.cache_weights:
            self._weight_cache[layer_name] = tensor
        
        logger.debug(f"Loaded layer: {layer_name}, shape: {tensor.shape}, dtype: {tensor.dtype}")
        
        return tensor
    
    def _row_to_tensor(self, table: pa.Table, row_idx: int) -> torch.Tensor:
        """
        Convert a single Arrow table row to PyTorch tensor.
        
        This performs zero-copy conversion when possible by directly
        wrapping the Arrow buffer in a NumPy array, then converting
        to PyTorch tensor. Automatically detects schema version and
        uses appropriate loading path.
        
        Args:
            table: Arrow table
            row_idx: Row index to convert
            
        Returns:
            PyTorch tensor
        """
        # Detect schema version if not already done
        if self._schema_version is None:
            self._schema_version = detect_schema_version(table)
        
        # Use appropriate loading path based on schema version
        if self._schema_version == 1:
            return self._load_v1_tensor(table, row_idx)
        else:
            return self._load_v2_tensor(table, row_idx)
    
    def _load_v1_tensor(self, table: pa.Table, row_idx: int) -> torch.Tensor:
        """
        Load tensor from Schema V1 (FP16/FP32 format).
        
        Schema V1 stores weights as raw bytes in the 'data' column.
        This method performs zero-copy conversion from Arrow buffer
        to PyTorch tensor.
        
        Args:
            table: Arrow table (Schema V1)
            row_idx: Row index to convert
            
        Returns:
            PyTorch tensor (FP16 or FP32)
        """
        # Extract row data
        shape = table['shape'][row_idx].as_py()
        dtype_str = table['dtype'][row_idx].as_py()
        # Direct buffer access for True Zero-Copy
        data_buffer = table['data'][row_idx].as_buffer()
        
        # Convert dtype string to NumPy dtype
        numpy_dtype = self._torch_dtype_to_numpy(dtype_str)
        
        # Zero-copy: Create NumPy array from buffer without copy()
        # Arrow's as_buffer() provides a memory-mapped view
        numpy_array = np.frombuffer(data_buffer, dtype=numpy_dtype)

        # Reshape to original shape
        numpy_array = numpy_array.reshape(shape)

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(numpy_array)

        # Upcast float16 → float32 to prevent cumulative precision loss through
        # deep transformer stacks (12+ layers). Without this, Vision/Audio encoders
        # suffer cosine similarity degradation (~0.48 vs target >0.95).
        if self.force_float32 and tensor.dtype == torch.float16:
            tensor = tensor.float()

        # Move to target device
        if self.device != "cpu":
            tensor = tensor.to(self.device)

        return tensor
    
    def _load_v2_tensor(self, table: pa.Table, row_idx: int) -> torch.Tensor:
        """
        Load tensor from Schema V2 (quantized format).
        
        Schema V2 stores quantized weights (INT8/INT2) with quantization
        metadata (scales, zero_points, quant_axis). This method loads
        the quantized data and performs dequantization.
        
        For FP16/FP32 weights in V2 format (mixed precision), this method
        loads them directly without dequantization.
        
        Args:
            table: Arrow table (Schema V2)
            row_idx: Row index to convert
            
        Returns:
            PyTorch tensor (dequantized to FP32)
        """
        # Extract row data
        shape = table['shape'][row_idx].as_py()
        dtype_str = table['dtype'][row_idx].as_py()
        quant_type = table['quant_type'][row_idx].as_py()
        data_buffer = table['data'][row_idx].as_buffer()
        
        # Check if this is a mixed precision layer (FP16/FP32 in V2 format)
        if quant_type in ['fp16', 'fp32']:
            # Load as floating point (no dequantization needed)
            numpy_dtype = self._torch_dtype_to_numpy(dtype_str)
            numpy_array = np.frombuffer(data_buffer, dtype=numpy_dtype)
            numpy_array = numpy_array.reshape(shape)
            tensor = torch.from_numpy(numpy_array)
            
            # Upcast to float32 if needed
            if self.force_float32 and tensor.dtype == torch.float16:
                tensor = tensor.float()
            
            # Move to target device
            if self.device != "cpu":
                tensor = tensor.to(self.device)
            
            return tensor
        
        # Quantized weights - need dequantization
        scales = table['scales'][row_idx].as_py()
        zero_points = table['zero_points'][row_idx].as_py()
        quant_axis = table['quant_axis'][row_idx].as_py()
        group_size = table['group_size'][row_idx].as_py() if 'group_size' in table.schema.names else 0
        
        # Load quantized data (unpack if int2)
        if quant_type == 'int2':
            packed_array = np.frombuffer(data_buffer, dtype=np.uint8)
            quantized_array = self._unpack_int2(packed_array, shape)
        else:
            quantized_array = np.frombuffer(data_buffer, dtype=np.int8)
            quantized_array = quantized_array.reshape(shape)
        
        # Dequantize
        dequantized = self._dequantize(
            quantized_array,
            scales,
            zero_points,
            quant_axis,
            group_size
        )
        
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(dequantized)
        
        # Move to target device
        if self.device != "cpu":
            tensor = tensor.to(self.device)
        
        return tensor
    
    def _dequantize(
        self,
        quantized: np.ndarray,
        scales: bytes,
        zero_points: bytes,
        quant_axis: int,
        group_size: int = 0
    ) -> np.ndarray:
        """
        Dequantize INT8/INT2 weights to FP32.
        
        Dequantization formula:
        - x = (q - zero_point) * scale
        
        Supports three quantization strategies:
        - Per-tensor: quant_axis = -1, group_size = 0, single scale/zero_point
        - Per-channel: quant_axis >= 0, group_size = 0, one scale/zero_point per channel
        - Per-group: group_size > 0, one scale/zero_point per group
        
        Args:
            quantized: Quantized weights (int8), shape (C, ...)
            scales: Scaling factors as binary (FP32 array)
            zero_points: Zero points as binary (FP32 array)
            quant_axis: Quantization axis (-1 for per-tensor, 0+ for per-channel)
            group_size: Group size for per-group quantization (0 for per-tensor/per-channel)
            
        Returns:
            Dequantized weights (float32)
        """
        # Convert binary format to numpy arrays
        scales_array = np.frombuffer(scales, dtype=np.float32)
        zero_points_array = np.frombuffer(zero_points, dtype=np.float32)
        
        # Per-group quantization
        if group_size > 0:
            # Flatten quantized array
            flat_quantized = quantized.flatten()
            flat_dequantized = np.zeros_like(flat_quantized, dtype=np.float32)
            
            # Dequantize each group
            num_groups = len(scales_array)
            for i in range(num_groups):
                start_idx = i * group_size
                end_idx = min(start_idx + group_size, len(flat_quantized))
                
                group_data = flat_quantized[start_idx:end_idx]
                flat_dequantized[start_idx:end_idx] = (
                    group_data.astype(np.float32) - zero_points_array[i]
                ) * scales_array[i]
            
            # Reshape back to original shape
            return flat_dequantized.reshape(quantized.shape)
        
        # Per-tensor quantization
        elif quant_axis == -1:
            dequantized = (quantized.astype(np.float32) - zero_points_array[0]) * scales_array[0]
        
        # Per-channel quantization
        else:
            dequantized = np.zeros_like(quantized, dtype=np.float32)
            
            if quant_axis == 0:
                # Quantization along axis 0 (most common for linear layers)
                for i in range(quantized.shape[0]):
                    dequantized[i] = (quantized[i].astype(np.float32) - zero_points_array[i]) * scales_array[i]
            
            elif quant_axis == 1:
                # Quantization along axis 1
                for i in range(quantized.shape[1]):
                    dequantized[:, i] = (quantized[:, i].astype(np.float32) - zero_points_array[i]) * scales_array[i]
            
            else:
                # Unsupported quantization axis
                logger.warning(
                    f"Unsupported quant_axis={quant_axis}, falling back to per-tensor dequantization"
                )
                dequantized = (quantized.astype(np.float32) - zero_points_array[0]) * scales_array[0]
        
        return dequantized
    
    def _torch_dtype_to_numpy(self, dtype_str: str):
        """
        Convert PyTorch dtype string to NumPy dtype.
        
        Args:
            dtype_str: PyTorch dtype string (e.g., "torch.float32")
            
        Returns:
            NumPy dtype
        """
        dtype_map = {
            "torch.float32": np.float32,
            "torch.float16": np.float16,
            "torch.int64": np.int64,
            "torch.int32": np.int32,
            "torch.int8": np.int8,
            "torch.uint8": np.uint8,
            "torch.bool": np.bool_,
        }
        return dtype_map.get(dtype_str, np.float32)

    def _unpack_int2(self, packed: np.ndarray, original_shape: List[int]) -> np.ndarray:
        """
        Unpack uint8 array back into int8 array of [-2, 1] values.
        
        Args:
            packed: Flat uint8 array of packed values
            original_shape: Original weight shape
            
        Returns:
            Dequantized int8 array with original shape
        """
        # Unpack bits
        # packed is uint8, each byte has 4 values
        # Bit layout: [v0: 2 bits] [v1: 2 bits] [v2: 2 bits] [v3: 2 bits]
        v0 = (packed >> 6) & 0x03
        v1 = (packed >> 4) & 0x03
        v2 = (packed >> 2) & 0x03
        v3 = packed & 0x03
        
        # Interleave into a single flat array
        unpacked = np.empty(len(packed) * 4, dtype=np.int8)
        unpacked[0::4] = v0
        unpacked[1::4] = v1
        unpacked[2::4] = v2
        unpacked[3::4] = v3
        
        # Map [0, 3] back to [-2, 1]
        # (v - 2)
        unpacked = unpacked.astype(np.int8) - 2
        
        # Trim to original size (it might have been padded to multiple of 4)
        num_params = np.prod(original_shape)
        unpacked = unpacked[:num_params]
        
        return unpacked.reshape(original_shape)
    
    def get_layer_names(self) -> List[str]:
        """
        Get list of all layer names in the model.
        
        Returns:
            List of layer names
        """
        table = self._load_table()
        return table['layer_name'].to_pylist()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the loaded weights.
        
        Returns:
            Metadata dictionary with:
            - num_layers: Number of layers
            - total_parameters: Total parameter count
            - dtypes: Set of dtypes in the model
            - layer_shapes: Dictionary of layer names to shapes
            - schema_version: Parquet schema version (1 or 2)
        """
        if self._metadata is None:
            table = self._load_table()
            
            layer_names = table['layer_name'].to_pylist()
            shapes = table['shape'].to_pylist()
            dtypes = table['dtype'].to_pylist()
            num_params = table['num_params'].to_pylist()
            
            self._metadata = {
                'num_layers': len(table),
                'total_parameters': sum(num_params),
                'dtypes': list(set(dtypes)),
                'layer_shapes': {
                    name: shape for name, shape in zip(layer_names, shapes)
                },
                'layer_dtypes': {
                    name: dtype for name, dtype in zip(layer_names, dtypes)
                },
                'schema_version': self._schema_version,
            }
        
        return self._metadata
    
    def get_schema_version(self) -> int:
        """
        Get the Parquet schema version.
        
        Returns:
            Schema version (1 for FP16/FP32, 2 for quantized)
            
        Example:
            >>> loader = WeightLoader("weights.parquet")
            >>> version = loader.get_schema_version()
            >>> print(f"Schema V{version}")
            Schema V2
        """
        if self._schema_version is None:
            self._load_table()
        return self._schema_version
    
    def clear_cache(self):
        """Clear the weight cache to free memory."""
        self._weight_cache.clear()
        logger.info("Weight cache cleared")
    
    def unload_layer(self, layer_name: str) -> bool:
        """
        Unload a specific layer from cache to free memory.
        
        This is useful for memory pressure scenarios where you want to
        selectively unload unused layers while keeping frequently accessed
        layers in cache.
        
        Args:
            layer_name: Name of the layer to unload
            
        Returns:
            True if layer was unloaded, False if not in cache
            
        Example:
            >>> loader = WeightLoader("weights.parquet", cache_weights=True)
            >>> loader.get_layer("encoder.layer.0.weight")
            >>> loader.unload_layer("encoder.layer.0.weight")
            True
        """
        if layer_name in self._weight_cache:
            del self._weight_cache[layer_name]
            logger.debug(f"Unloaded layer from cache: {layer_name}")
            return True
        return False
    
    def unload_layers(self, layer_names: List[str]) -> int:
        """
        Unload multiple layers from cache.
        
        Args:
            layer_names: List of layer names to unload
            
        Returns:
            Number of layers successfully unloaded
            
        Example:
            >>> loader = WeightLoader("weights.parquet", cache_weights=True)
            >>> layers_to_unload = ["encoder.layer.0.weight", "encoder.layer.1.weight"]
            >>> count = loader.unload_layers(layers_to_unload)
            >>> print(f"Unloaded {count} layers")
        """
        count = 0
        for layer_name in layer_names:
            if self.unload_layer(layer_name):
                count += 1
        
        if count > 0:
            logger.info(f"Unloaded {count} layers from cache")
        
        return count
    
    def get_cached_layers(self) -> List[str]:
        """
        Get list of currently cached layer names.
        
        Returns:
            List of cached layer names
            
        Example:
            >>> loader = WeightLoader("weights.parquet", cache_weights=True)
            >>> loader.get_layer("encoder.layer.0.weight")
            >>> cached = loader.get_cached_layers()
            >>> print(f"Cached layers: {cached}")
        """
        return list(self._weight_cache.keys())
    
    def get_cache_size_mb(self) -> float:
        """
        Get size of cached weights in MB.
        
        Returns:
            Cache size in megabytes
        """
        if not self._weight_cache:
            return 0.0
        
        total_bytes = sum(
            t.numel() * t.element_size() 
            for t in self._weight_cache.values()
        )
        return total_bytes / (1024 * 1024)
    
    def __repr__(self) -> str:
        """String representation."""
        metadata = self.get_metadata() if self._table else {}
        total_params = metadata.get('total_parameters', '?')
        params_str = f"{total_params:,}" if isinstance(total_params, int) else total_params
        return (
            f"WeightLoader("
            f"path={self.parquet_path.name}, "
            f"layers={metadata.get('num_layers', '?')}, "
            f"params={params_str}, "
            f"cache={self.get_cache_size_mb():.1f}MB)"
        )


class LazyWeightDict:
    """
    A proxy dictionary that loads weights from WeightLoader only when accessed.
    Compatible with InferenceCore's weight loading logic.
    """
    def __init__(self, loader: WeightLoader):
        self.loader = loader
        self._layer_names = loader.get_layer_names()
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = self.loader.get_layer(key)
        return self._cache[key]

    def __contains__(self, key):
        return key in self._layer_names

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self._layer_names

    def __len__(self):
        return len(self._layer_names)

    def __iter__(self):
        return iter(self._layer_names)

    def values(self):
        return (self.loader.get_layer(name) for name in self._layer_names)
        
    def items(self):
        return ((name, self.loader.get_layer(name)) for name in self._layer_names)

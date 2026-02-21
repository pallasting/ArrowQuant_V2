"""
ArrowQuantizer: Post-Training Quantization for Arrow/Parquet weights.

This module implements PTQ (Post-Training Quantization) for model weights stored
in Arrow/Parquet format, supporting INT8 and INT2 quantization with per-channel
and per-tensor modes.

Key features:
- INT8 and INT2 quantization
- Symmetric and asymmetric quantization
- Per-channel and per-tensor quantization
- Mixed precision (skip sensitive layers)
- Parquet Schema V2 output with quantization metadata

Performance targets:
- 75% memory reduction (INT8)
- >0.95 cosine similarity preservation
- <15% PPL increase for language models

Requirements: 2.1, 2.8, 2.9, 9.3, 12.1
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from llm_compression.logger import logger
from llm_compression.errors import ConfigurationError, StorageError
from llm_compression.inference.quantization_schema import (
    WEIGHT_SCHEMA_V1,
    WEIGHT_SCHEMA_V2,
    QuantType,
    detect_schema_version,
    create_v2_row,
)


@dataclass
class QuantizationConfig:
    """
    Quantization configuration.
    
    Attributes:
        quant_type: Quantization type ('int8', 'int2', 'fp16')
        calibration_method: Calibration method ('ptq', 'gptq')
        per_channel: Use per-channel quantization (True) or per-tensor (False)
        symmetric: Use symmetric quantization (True) or asymmetric (False)
        group_size: Group size for per-group quantization (0 for per-tensor, 128 default for int2)
        mixed_precision_layers: Layer name patterns to skip quantization (keep FP16)
        
    Example:
        >>> config = QuantizationConfig(
        ...     quant_type='int2',
        ...     calibration_method='ptq',
        ...     per_channel=False,
        ...     symmetric=True,
        ...     group_size=128,
        ...     mixed_precision_layers=['lm_head', 'embed']
        ... )
    """
    
    quant_type: Literal['int8', 'int2', 'fp16'] = 'int8'
    calibration_method: Literal['ptq', 'gptq'] = 'ptq'
    per_channel: bool = True
    symmetric: bool = True
    group_size: int = 0  # 0 = per-tensor, >0 = per-group (default 128 for int2)
    mixed_precision_layers: Optional[List[str]] = field(default_factory=lambda: None)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate quant_type
        if self.quant_type not in ['int8', 'int2', 'fp16']:
            raise ConfigurationError(
                message=f"Invalid quant_type: {self.quant_type}",
                config_key='quant_type',
                config_value=self.quant_type,
                expected_type="'int8', 'int2', or 'fp16'"
            )
        
        # Validate calibration_method
        if self.calibration_method not in ['ptq', 'gptq']:
            raise ConfigurationError(
                message=f"Invalid calibration_method: {self.calibration_method}",
                config_key='calibration_method',
                config_value=self.calibration_method,
                expected_type="'ptq' or 'gptq'"
            )
        
        # Validate boolean parameters
        if not isinstance(self.per_channel, bool):
            raise ConfigurationError(
                message=f"per_channel must be bool, got {type(self.per_channel)}",
                config_key='per_channel',
                config_value=self.per_channel,
                expected_type='bool'
            )
        
        if not isinstance(self.symmetric, bool):
            raise ConfigurationError(
                message=f"symmetric must be bool, got {type(self.symmetric)}",
                config_key='symmetric',
                config_value=self.symmetric,
                expected_type='bool'
            )
        
        # Validate group_size
        if not isinstance(self.group_size, int) or self.group_size < 0:
            raise ConfigurationError(
                message=f"group_size must be non-negative int, got {self.group_size}",
                config_key='group_size',
                config_value=self.group_size,
                expected_type='int >= 0'
            )
        
        # Auto-set group_size for int2 if not specified
        if self.quant_type == 'int2' and self.group_size == 0 and not self.per_channel:
            self.group_size = 128
            logger.info(f"Auto-set group_size=128 for INT2 quantization")
        
        # Validate mixed_precision_layers
        if self.mixed_precision_layers is not None:
            if not isinstance(self.mixed_precision_layers, list):
                raise ConfigurationError(
                    message=f"mixed_precision_layers must be list, got {type(self.mixed_precision_layers)}",
                    config_key='mixed_precision_layers',
                    config_value=self.mixed_precision_layers,
                    expected_type='list'
                )
            
            for pattern in self.mixed_precision_layers:
                if not isinstance(pattern, str):
                    raise ConfigurationError(
                        message=f"mixed_precision_layers must contain strings, got {type(pattern)}",
                        config_key='mixed_precision_layers',
                        config_value=pattern,
                        expected_type='str'
                    )
        
        logger.debug(f"QuantizationConfig validated: {self}")


class ArrowQuantizer:
    """
    Arrow weight quantizer with PTQ support.
    
    Implements Post-Training Quantization (PTQ) for model weights stored in
    Arrow/Parquet format. Supports INT8 and INT2 quantization with per-channel
    and per-tensor modes.
    
    Quantization formulas:
    - Symmetric: q = round(x / scale), where scale = max(|x|) / qmax
    - Asymmetric: q = round(x / scale) + zero_point
    - Dequantization: x = (q - zero_point) * scale
    
    Features:
    - INT8 quantization: 4x memory reduction
    - INT2 quantization: 16x memory reduction (experimental)
    - Per-channel quantization: Better accuracy for conv/linear layers
    - Per-tensor quantization: Simpler, faster
    - Mixed precision: Skip sensitive layers (embeddings, lm_head)
    
    Performance targets:
    - INT8: >0.95 cosine similarity, 75% memory reduction
    - INT2: >0.90 cosine similarity, 93% memory reduction
    
    Example:
        >>> config = QuantizationConfig(quant_type='int8', per_channel=True)
        >>> quantizer = ArrowQuantizer(config)
        >>> quantizer.quantize_model(
        ...     input_parquet='weights_fp16.parquet',
        ...     output_parquet='weights_int8.parquet'
        ... )
    """
    
    def __init__(self, config: QuantizationConfig):
        """
        Initialize ArrowQuantizer.
        
        Args:
            config: Quantization configuration
        """
        self.config = config
        logger.info(f"Initialized ArrowQuantizer with config: {config}")
    
    def quantize_model(
        self,
        input_parquet: str,
        output_parquet: str,
        calibration_data: Any = None,
        show_progress: bool = True
    ):
        """
        Quantize model weights from V1 to V2 Parquet format.
        
        Reads FP16/FP32 weights from input Parquet (Schema V1), applies
        quantization, and writes quantized weights to output Parquet (Schema V2).
        
        Args:
            input_parquet: Input Parquet file path (Schema V1)
            output_parquet: Output Parquet file path (Schema V2)
            calibration_data: Calibration data for GPTQ (optional)
            show_progress: Show progress bar during quantization (default: True)
            
        Raises:
            StorageError: If input file not found or output write fails
            ConfigurationError: If configuration is invalid
            
        Example:
            >>> quantizer.quantize_model(
            ...     'models/minilm/weights.parquet',
            ...     'models/minilm/weights_int8.parquet'
            ... )
        """
        import time
        
        input_path = Path(input_parquet)
        output_path = Path(output_parquet)
        
        # Validate input file exists
        if not input_path.exists():
            raise StorageError(
                message=f"Input Parquet file not found: {input_parquet}",
                operation='read',
                path=str(input_path)
            )
        
        logger.info(f"Quantizing model: {input_parquet} -> {output_parquet}")
        logger.info(f"Config: {self.config}")
        
        try:
            # Read input table (Schema V1)
            logger.info("Loading input Parquet file...")
            table = pq.read_table(input_parquet)
            schema_version = detect_schema_version(table)
            
            if schema_version != 1:
                logger.warning(
                    f"Input file is Schema V{schema_version}, expected V1. "
                    "Proceeding with quantization anyway."
                )
            
            logger.info(f"Loaded {len(table)} layers from {input_parquet}")
            
        except Exception as e:
            raise StorageError(
                message=f"Failed to read input Parquet: {e}",
                operation='read',
                path=str(input_path),
                original_exception=e
            )
        
        # Quantize each layer
        quantized_rows = []
        total_layers = len(table)
        skipped_layers = 0
        quantized_layers = 0
        total_params_original = 0
        total_params_quantized = 0
        
        start_time = time.time()
        
        logger.info(f"Quantizing {total_layers} layers...")
        
        for i in range(total_layers):
            layer_start_time = time.time()
            
            row = table.slice(i, 1).to_pydict()
            layer_name = row['layer_name'][0]
            shape = row['shape'][0]
            dtype_str = row['dtype'][0]
            data_buffer = row['data'][0]
            num_params = row['num_params'][0]
            
            # Convert buffer to numpy array
            numpy_dtype = self._torch_dtype_to_numpy(dtype_str)
            weight_data = np.frombuffer(data_buffer, dtype=numpy_dtype)
            weight_data = weight_data.reshape(shape).astype(np.float32)
            
            # Track original size
            original_size_bytes = weight_data.nbytes
            total_params_original += num_params
            
            # Check if should skip quantization (mixed precision)
            if self._should_skip_quantization(layer_name):
                logger.debug(f"Skipping quantization for {layer_name} (mixed precision)")
                
                # Keep as FP16
                fp16_data = weight_data.astype(np.float16)
                quantized_rows.append(create_v2_row(
                    layer_name=layer_name,
                    shape=shape,
                    dtype=dtype_str,
                    data=fp16_data.tobytes(),
                    num_params=num_params,
                    quant_type='fp16',
                    scales=b'',
                    zero_points=b'',
                    quant_axis=-1,
                    group_size=0
                ))
                skipped_layers += 1
                total_params_quantized += fp16_data.nbytes
                
                # Progress reporting
                if show_progress:
                    progress_pct = ((i + 1) / total_layers) * 100
                    logger.info(
                        f"[{i+1}/{total_layers}] ({progress_pct:.1f}%) "
                        f"Skipped {layer_name} (mixed precision)"
                    )
                continue
            
            # Quantize weight
            if self.config.calibration_method == 'ptq':
                quant_result = self._quantize_ptq(weight_data, shape)
            elif self.config.calibration_method == 'gptq':
                layer_calib_data = calibration_data
                if isinstance(calibration_data, dict):
                    layer_calib_data = calibration_data.get(layer_name)
                    if layer_calib_data is None:
                        base_name = layer_name.rsplit('.weight', 1)[0]
                        layer_calib_data = calibration_data.get(base_name)
                    if layer_calib_data is None:
                        logger.warning(f"No calibration data found for layer {layer_name}. Falling back to PTQ.")
                        
                quant_result = self._quantize_gptq(weight_data, shape, layer_calib_data)
            else:
                raise ConfigurationError(
                    message=f"Unknown calibration method: {self.config.calibration_method}",
                    config_key='calibration_method',
                    config_value=self.config.calibration_method
                )
            
            # Track quantized size
            quantized_size_bytes = quant_result['quantized'].nbytes
            total_params_quantized += quantized_size_bytes
            
            # Create V2 row
            quantized_rows.append(create_v2_row(
                layer_name=layer_name,
                shape=shape,
                dtype=dtype_str,
                data=quant_result['quantized'].tobytes(),
                num_params=num_params,
                quant_type=self.config.quant_type,
                scales=quant_result['scales'],
                zero_points=quant_result['zero_points'],
                quant_axis=quant_result['quant_axis'],
                group_size=quant_result['group_size']
            ))
            
            quantized_layers += 1
            
            # Progress reporting
            layer_time = time.time() - layer_start_time
            if show_progress:
                progress_pct = ((i + 1) / total_layers) * 100
                compression_ratio = original_size_bytes / quantized_size_bytes if quantized_size_bytes > 0 else 0
                logger.info(
                    f"[{i+1}/{total_layers}] ({progress_pct:.1f}%) "
                    f"Quantized {layer_name}: {shape} -> {self.config.quant_type} "
                    f"({compression_ratio:.2f}x compression, {layer_time:.2f}s)"
                )
            else:
                logger.debug(f"Quantized {layer_name}: shape={shape}, quant_type={self.config.quant_type}")
        
        quantization_time = time.time() - start_time
        
        # Calculate overall compression metrics
        overall_compression_ratio = total_params_original / total_params_quantized if total_params_quantized > 0 else 0
        memory_savings_pct = (1 - total_params_quantized / total_params_original) * 100 if total_params_original > 0 else 0
        
        logger.info(
            f"Quantization complete in {quantization_time:.2f}s: "
            f"{quantized_layers}/{total_layers} layers quantized "
            f"({skipped_layers} skipped for mixed precision)"
        )
        logger.info(
            f"Compression metrics: {overall_compression_ratio:.2f}x compression, "
            f"{memory_savings_pct:.1f}% memory savings"
        )
        
        # Write V2 Parquet
        try:
            logger.info("Writing quantized weights to output file...")
            quantized_table = pa.Table.from_pylist(quantized_rows, schema=WEIGHT_SCHEMA_V2)
            pq.write_table(quantized_table, output_parquet)
            
            output_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Output file written: {output_parquet} ({output_size_mb:.2f} MB)")
            
        except Exception as e:
            raise StorageError(
                message=f"Failed to write output Parquet: {e}",
                operation='write',
                path=str(output_path),
                original_exception=e
            )
    
    def _should_skip_quantization(self, layer_name: str) -> bool:
        """
        Check if layer should skip quantization (mixed precision).
        
        Args:
            layer_name: Layer name
            
        Returns:
            True if should skip quantization, False otherwise
        """
        if self.config.mixed_precision_layers is None:
            return False
        
        # Check if any pattern matches the layer name
        for pattern in self.config.mixed_precision_layers:
            if pattern in layer_name:
                return True
        
        return False
    
    def _quantize_ptq(
        self,
        weight: np.ndarray,
        shape: List[int]
    ) -> Dict[str, np.ndarray]:
        """
        PTQ (Post-Training Quantization) quantization.
        
        Implements min-max quantization with support for:
        - Per-tensor quantization (group_size=0)
        - Per-channel quantization (per_channel=True, group_size=0)
        - Per-group quantization (group_size>0)
        
        Args:
            weight: Weight array (float32)
            shape: Weight shape
            
        Returns:
            Dictionary with:
            - quantized: Quantized weights (packed for int2, int8 for int8)
            - scales: Scaling factors as binary (FP32 array)
            - zero_points: Zero points as binary (FP32 array)
            - quant_axis: Quantization axis (-1 for per-tensor, 0 for per-channel/per-group)
            - group_size: Group size (0 for per-tensor/per-channel)
        """
        # Determine quantization strategy
        if self.config.group_size > 0:
            # Per-group quantization (ArrowQuant design)
            return self._quantize_per_group(weight, shape)
        elif self.config.per_channel and len(shape) > 1:
            # Per-channel quantization (along axis 0)
            return self._quantize_per_channel(weight, shape)
        else:
            # Per-tensor quantization
            return self._quantize_per_tensor(weight, shape)
    
    def _quantize_per_tensor(
        self,
        weight: np.ndarray,
        shape: List[int]
    ) -> Dict[str, np.ndarray]:
        """Per-tensor quantization."""
        quant_axis = -1
        group_size = 0
        
        scale, zero_point = self._compute_quantization_params(weight)
        quantized = self._quantize_tensor(weight, scale, zero_point)
        
        scales = np.array([scale], dtype=np.float32)
        zero_points = np.array([zero_point], dtype=np.float32)
        
        quantized_flat = quantized.flatten()
        
        # Pack INT2 values
        if self.config.quant_type == 'int2':
            quantized_flat = self._pack_int2(quantized_flat)
        
        return {
            'quantized': quantized_flat,
            'scales': scales.tobytes(),
            'zero_points': zero_points.tobytes(),
            'quant_axis': quant_axis,
            'group_size': group_size,
        }
    
    def _quantize_per_channel(
        self,
        weight: np.ndarray,
        shape: List[int]
    ) -> Dict[str, np.ndarray]:
        """Per-channel quantization (along axis 0)."""
        quant_axis = 0
        group_size = 0
        num_channels = shape[0]
        
        scales = []
        zero_points = []
        quantized_channels = []
        
        for i in range(num_channels):
            channel = weight[i]
            scale, zero_point = self._compute_quantization_params(channel)
            scales.append(scale)
            zero_points.append(zero_point)
            
            # Quantize channel
            quantized = self._quantize_tensor(channel, scale, zero_point)
            quantized_channels.append(quantized)
        
        quantized = np.stack(quantized_channels, axis=0)
        scales = np.array(scales, dtype=np.float32)
        zero_points = np.array(zero_points, dtype=np.float32)
        
        quantized_flat = quantized.flatten()
        
        # Pack INT2 values
        if self.config.quant_type == 'int2':
            quantized_flat = self._pack_int2(quantized_flat)
        
        return {
            'quantized': quantized_flat,
            'scales': scales.tobytes(),
            'zero_points': zero_points.tobytes(),
            'quant_axis': quant_axis,
            'group_size': group_size,
        }
    
    def _quantize_per_group(
        self,
        weight: np.ndarray,
        shape: List[int]
    ) -> Dict[str, np.ndarray]:
        """
        Per-group quantization (ArrowQuant design).
        
        Divides weight into groups of size group_size and quantizes each group
        independently. This provides better accuracy than per-tensor while
        maintaining reasonable metadata overhead.
        """
        quant_axis = 0  # Groups are along flattened dimension
        group_size = self.config.group_size
        
        # Flatten weight
        weight_flat = weight.flatten()
        num_elements = len(weight_flat)
        num_groups = (num_elements + group_size - 1) // group_size
        
        scales = []
        zero_points = []
        quantized_groups = []
        
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min(start_idx + group_size, num_elements)
            group = weight_flat[start_idx:end_idx]
            
            # Compute scale and zero_point for this group
            scale, zero_point = self._compute_quantization_params(group)
            scales.append(scale)
            zero_points.append(zero_point)
            
            # Quantize group
            quantized = self._quantize_tensor(group, scale, zero_point)
            quantized_groups.append(quantized)
        
        # Concatenate all groups
        quantized = np.concatenate(quantized_groups)
        scales = np.array(scales, dtype=np.float32)
        zero_points = np.array(zero_points, dtype=np.float32)
        
        # Pack INT2 values
        if self.config.quant_type == 'int2':
            quantized = self._pack_int2(quantized)
        
        return {
            'quantized': quantized,
            'scales': scales.tobytes(),
            'zero_points': zero_points.tobytes(),
            'quant_axis': quant_axis,
            'group_size': group_size,
        }
    
    def _compute_quantization_params(
        self,
        tensor: np.ndarray
    ) -> tuple[float, int]:
        """
        Compute quantization parameters (scale and zero_point).
        
        Args:
            tensor: Input tensor (float32)
            
        Returns:
            Tuple of (scale, zero_point)
        """
        # Get quantization range
        if self.config.quant_type == 'int8':
            qmin, qmax = -128, 127
        elif self.config.quant_type == 'int2':
            qmin, qmax = -2, 1
        else:
            raise ConfigurationError(
                message=f"Unsupported quant_type for quantization: {self.config.quant_type}",
                config_key='quant_type',
                config_value=self.config.quant_type
            )
        
        if self.config.symmetric:
            # Symmetric quantization: zero_point = 0
            max_val = np.abs(tensor).max()
            
            # Avoid division by zero
            if max_val == 0:
                scale = 1.0
            else:
                scale = max_val / qmax
            
            zero_point = 0
        
        else:
            # Asymmetric quantization
            min_val = tensor.min()
            max_val = tensor.max()
            
            # Avoid division by zero
            if max_val == min_val:
                scale = 1.0
                zero_point = 0
            else:
                scale = (max_val - min_val) / (qmax - qmin)
                zero_point = int(qmin - min_val / scale)
                
                # Clip zero_point to valid range
                zero_point = np.clip(zero_point, qmin, qmax)
        
        return float(scale), int(zero_point)
    
    def _quantize_tensor(
        self,
        tensor: np.ndarray,
        scale: float,
        zero_point: int
    ) -> np.ndarray:
        """
        Quantize tensor to int8.
        
        Args:
            tensor: Input tensor (float32)
            scale: Scaling factor
            zero_point: Zero point
            
        Returns:
            Quantized tensor (int8)
        """
        # Get quantization range
        if self.config.quant_type == 'int8':
            qmin, qmax = -128, 127
        elif self.config.quant_type == 'int2':
            qmin, qmax = -2, 1
        else:
            raise ConfigurationError(
                message=f"Unsupported quant_type for quantization: {self.config.quant_type}",
                config_key='quant_type',
                config_value=self.config.quant_type
            )
        
        # Quantize: q = round(x / scale) + zero_point
        if scale == 0:
            quantized = np.zeros_like(tensor, dtype=np.int8)
        else:
            quantized = np.round(tensor / scale) + zero_point
        
        # Clip to valid range
        quantized = np.clip(quantized, qmin, qmax)
        
        return quantized.astype(np.int8)
    
    def _quantize_gptq(
        self,
        weight: np.ndarray,
        shape: List[int],
        calibration_data: Optional[torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        GPTQ quantization (Hessian-based calibration).
        
        Uses Optimal Brain Quantization (OBQ) principles applied row-by-row (or channel-by-channel).
        Given calibration input X, it computes the Hessian H = 2 X X^T.
        Weights are quantized one by one, and the quantization error is compensated
        across the remaining unquantized weights using H^{-1}.
        
        Args:
            weight: Weight array (float32), expected shape [out_features, in_features]
            shape: Weight shape
            calibration_data: Calibration input X to this layer. Shape: [num_samples, seq_len, in_features]
                              If None, falls back to PTQ.
            
        Returns:
            Dictionary with:
            - quantized: Quantized weights (int2/int8)
            - scales: Scaling factors (float32)
            - zero_points: Zero points (int8)
            - quant_axis: Quantization axis (-1 for per-tensor, usually 0 for per-channel)
        """
        if calibration_data is None:
            logger.warning("No calibration data provided for GPTQ. Falling back to simple PTQ.")
            return self._quantize_ptq(weight, shape)

        if len(shape) != 2:
            logger.warning(f"GPTQ currently supports 2D weight matrices. Got shape {shape}. Falling back to PTQ.")
            return self._quantize_ptq(weight, shape)

        out_features, in_features = shape
        
        # Ensure weight is a Parameter/Tensor for computations
        W = torch.from_numpy(weight).float()
        
        # Flatten calibration data [batch * seq_len, in_features]
        X = calibration_data.reshape(-1, in_features).float()
        num_samples = X.shape[0]
        
        if num_samples == 0:
            logger.warning("Calibration data is empty. Falling back to PTQ.")
            return self._quantize_ptq(weight, shape)

        # 1. Compute Hessian H = 2 * X^T * X / num_samples
        # Add a small dampening factor to ensure invertibility
        logger.debug(f"Computing Hessian for shape {shape} using {num_samples} samples.")
        H = 2 * (X.t() @ X) / num_samples
        damp = 0.01 * torch.mean(torch.diag(H))
        H[range(in_features), range(in_features)] += damp
        
        # Compute H^{-1}
        try:
            H_inv = torch.linalg.inv(H)
        except RuntimeError:
            logger.warning("Hessian matrix inversion failed. Falling back to PTQ.")
            return self._quantize_ptq(weight, shape)

        # 2. Setup Quantization Parameters
        # For GPTQ, it's typically per-output-channel (per-row)
        quant_axis = 0
        
        scales = []
        zero_points = []
        quantized_W = torch.zeros_like(W)
        
        # Get quantization range
        if self.config.quant_type == 'int8':
            qmin, qmax = -128, 127
        elif self.config.quant_type == 'int2':
            qmin, qmax = -2, 1
        else:
            raise ConfigurationError(
                message=f"Unsupported quant_type for quantization: {self.config.quant_type}",
                config_key='quant_type',
                config_value=self.config.quant_type
            )
            
        # 3. Block-wise or Channel-wise Iteration
        # To avoid massive memory/time overhead, we loop over output channels
        # and quantize input features block by block (or col by col).
        # We simplify here by doing column-by-column across all rows simultaneously (standard GPTQ).
        
        # Compute block size (e.g. 128) to speed up updates
        block_size = 128
        
        for i1 in range(0, in_features, block_size):
            i2 = min(i1 + block_size, in_features)
            count = i2 - i1
            
            # Sub-matrix of inverse Hessian
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            H_inv1 = H_inv[i1:i2, i1:i2]
            
            # Diagonal elements
            d = torch.diag(H_inv1)
            
            for i in range(count):
                w = W1[:, i]
                d_i = d[i]
                
                # Asymmetric min/max for the column / channel
                # Wait, scale needs to be per-row (out_features)!
                # But in standard GPTQ, scale is computed beforehand or dynamically.
                # Since we do per-row scale, we can compute it dynamically or statically. 
                # Let's compute static asymmetric scale per row.
                
                pass 
                
        # Actually, standard GPTQ computes scale and zero-point per row dynamically or statically.
        # Let's compute scales statically first using min/max of W, then run OBQ loop.
        
        logger.debug("Computing static scales and zero points per channel.")
        for i in range(out_features):
            row = W[i].numpy()
            s, z = self._compute_quantization_params(row)
            scales.append(s)
            zero_points.append(z)
            
        scales_t = torch.tensor(scales, dtype=torch.float32).unsqueeze(1)
        zps_t = torch.tensor(zero_points, dtype=torch.float32).unsqueeze(1)
        
        # GPTQ Main Loop
        logger.debug(f"Running iterative GPTQ compensation over {in_features} columns...")
        for i in range(in_features):
            w = W[:, i]
            d = H_inv[i, i]
            
            # Quantize
            # q = round(w / scale) + zp
            q = torch.round(w.unsqueeze(1) / scales_t + zps_t).squeeze(1)
            q = torch.clamp(q, qmin, qmax)
            
            # Dequantize
            w_q = (q - zps_t.squeeze(1)) * scales_t.squeeze(1)
            quantized_W[:, i] = q
            
            # Quantization error
            err = (w - w_q) / d
            
            # Update remaining weights
            W[:, i:] -= err.unsqueeze(1).matmul(H_inv[i, i:].unsqueeze(0))
            
        # Convert back to uint/int8 numpy representation
        quantized = quantized_W.numpy().astype(np.int8)
        scales_np = np.array(scales, dtype=np.float32)
        zps_np = np.array(zero_points, dtype=np.float32)
        
        quantized_flat = quantized.flatten()
        if self.config.quant_type == 'int2':
            quantized_flat = self._pack_int2(quantized_flat)
            
        return {
            'quantized': quantized_flat,
            'scales': scales_np.tobytes(),
            'zero_points': zps_np.tobytes(),
            'quant_axis': quant_axis,
            'group_size': 0,
        }
    
    @staticmethod
    def _pack_int2(data: np.ndarray) -> np.ndarray:
        """
        Pack int8 array of [-2, 1] values into uint8 array (4 values per byte).
        
        Implements bit packing as per ArrowQuant design:
        byte = (val_0) | (val_1 << 2) | (val_2 << 4) | (val_3 << 6)
        
        Args:
            data: Flat int8 array with values in [-2, 1]
            
        Returns:
            Flat uint8 array of packed values (length = ceil(len(data) / 4))
            
        Example:
            >>> data = np.array([0, 1, -1, -2], dtype=np.int8)
            >>> packed = ArrowQuantizer._pack_int2(data)
            >>> # [0, 1, -1, -2] -> [2, 3, 1, 0] -> 0b00_01_11_10 = 0x1E
        """
        # Ensure flat array and padding if necessary
        flat = data.flatten()
        pad_size = (4 - (len(flat) % 4)) % 4
        if pad_size > 0:
            flat = np.append(flat, np.zeros(pad_size, dtype=np.int8))
            
        # Map [-2, 1] to [0, 3] to ensure we only use 2 bits
        # -2 -> 0, -1 -> 1, 0 -> 2, 1 -> 3
        unsigned = (flat.astype(np.int8) + 2).astype(np.uint8) & 0x03
        
        # Pack 4 values into one byte
        packed = (unsigned[0::4]) | \
                 (unsigned[1::4] << 2) | \
                 (unsigned[2::4] << 4) | \
                 (unsigned[3::4] << 6)
                 
        return packed.astype(np.uint8)
    
    @staticmethod
    def _unpack_int2(packed: np.ndarray, num_elements: int) -> np.ndarray:
        """
        Unpack uint8 array into int8 array of [-2, 1] values (4 values per byte).
        
        Reverses the bit packing operation.
        
        Args:
            packed: Flat uint8 array of packed values
            num_elements: Number of elements to unpack
            
        Returns:
            Flat int8 array with values in [-2, 1]
            
        Example:
            >>> packed = np.array([0x1E], dtype=np.uint8)  # 0b00_01_11_10
            >>> unpacked = ArrowQuantizer._unpack_int2(packed, 4)
            >>> # [0x1E] -> [2, 3, 1, 0] -> [0, 1, -1, -2]
        """
        # Unpack 4 values from each byte
        val_0 = (packed) & 0x03
        val_1 = (packed >> 2) & 0x03
        val_2 = (packed >> 4) & 0x03
        val_3 = (packed >> 6) & 0x03
        
        # Interleave values
        unpacked = np.empty(len(packed) * 4, dtype=np.uint8)
        unpacked[0::4] = val_0
        unpacked[1::4] = val_1
        unpacked[2::4] = val_2
        unpacked[3::4] = val_3
        
        # Map [0, 3] back to [-2, 1]
        # 0 -> -2, 1 -> -1, 2 -> 0, 3 -> 1
        signed = unpacked.astype(np.int8) - 2
        
        # Trim to requested size
        return signed[:num_elements]

    def _torch_dtype_to_numpy(self, dtype_str: str):
        """
        Convert PyTorch dtype string to NumPy dtype.
        
        Args:
            dtype_str: PyTorch dtype string (e.g., 'torch.float32')
            
        Returns:
            NumPy dtype
        """
        dtype_map = {
            'torch.float32': np.float32,
            'torch.float16': np.float16,
            'torch.int64': np.int64,
            'torch.int32': np.int32,
            'torch.int8': np.int8,
            'torch.uint8': np.uint8,
            'torch.bool': np.bool_,
        }
        return dtype_map.get(dtype_str, np.float32)

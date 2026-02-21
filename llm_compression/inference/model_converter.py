"""
Model format converter between HuggingFace and Parquet formats.

This module provides converters to bridge HuggingFace models and AI-OS's
Parquet-based storage format, enabling integration with external quantization
tools like AngelSlim.

Supported conversions:
- HuggingFace → Parquet V1/V2
- Parquet V1/V2 → HuggingFace (future)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from llm_compression.logger import logger
from llm_compression.errors import CompressionError
from llm_compression.inference.quantization_schema import (
    WEIGHT_SCHEMA_V1,
    WEIGHT_SCHEMA_V2,
    create_v1_row,
    create_v2_row,
    QuantType,
)


class ModelConverterError(CompressionError):
    """Model conversion error"""
    pass


class HuggingFaceToParquetConverter:
    """
    Convert HuggingFace models to Parquet format.
    
    Supports:
    - FP16/BF16/FP32 models → Parquet V1
    - Quantized models (FP8/INT4/INT8) → Parquet V2
    
    Example:
        >>> converter = HuggingFaceToParquetConverter()
        >>> converter.convert(
        ...     hf_model_path="models/qwen3-0.6b",
        ...     output_parquet="models/qwen3-0.6b.parquet"
        ... )
    """
    
    def __init__(self):
        self.supported_dtypes = {
            torch.float32: 'torch.float32',
            torch.float16: 'torch.float16',
            torch.bfloat16: 'torch.bfloat16',
            torch.int8: 'torch.int8',
            torch.uint8: 'torch.uint8',
        }
    
    def convert(
        self,
        hf_model_path: str,
        output_parquet: str,
        auto_detect_quantization: bool = True
    ) -> None:
        """
        Convert HuggingFace model to Parquet format.
        
        Args:
            hf_model_path: Path to HuggingFace model directory
            output_parquet: Output Parquet file path
            auto_detect_quantization: Auto-detect if model is quantized
            
        Raises:
            ModelConverterError: If conversion fails
        """
        logger.info(f"Converting HuggingFace model: {hf_model_path}")
        logger.info(f"Output: {output_parquet}")
        
        try:
            # 1. Load model state dict
            state_dict = self._load_state_dict(hf_model_path)
            
            # 2. Detect quantization
            is_quantized = False
            quant_info = None
            if auto_detect_quantization:
                is_quantized, quant_info = self._detect_quantization(state_dict)
            
            # 3. Convert to rows
            if is_quantized:
                logger.info("Detected quantized model, using Schema V2")
                rows = self._convert_quantized_model(state_dict, quant_info)
                schema = WEIGHT_SCHEMA_V2
            else:
                logger.info("Detected FP model, using Schema V1")
                rows = self._convert_fp_model(state_dict)
                schema = WEIGHT_SCHEMA_V1
            
            # 4. Create table and save
            table = pa.Table.from_pylist(rows, schema=schema)
            pq.write_table(table, output_parquet)
            
            logger.info(f"Conversion complete: {len(rows)} layers saved")
            logger.info(f"Output file: {output_parquet}")
            
        except Exception as e:
            raise ModelConverterError(
                f"Failed to convert model: {e}",
                original_exception=e
            )
    
    def _load_state_dict(self, hf_model_path: str) -> Dict[str, torch.Tensor]:
        """
        Load model state dict from HuggingFace format.
        
        Args:
            hf_model_path: Path to HuggingFace model
            
        Returns:
            State dict
        """
        model_path = Path(hf_model_path)
        
        # Try different file names
        possible_files = [
            model_path / "pytorch_model.bin",
            model_path / "model.safetensors",
            model_path / "pytorch_model.bin.index.json",  # Sharded model
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                logger.debug(f"Loading from {file_path}")
                
                if file_path.suffix == ".bin":
                    return torch.load(file_path, map_location='cpu')
                elif file_path.suffix == ".safetensors":
                    # Try to load safetensors
                    try:
                        from safetensors.torch import load_file
                        return load_file(file_path)
                    except ImportError:
                        logger.warning("safetensors not installed, skipping")
                        continue
        
        raise ModelConverterError(
            f"No model file found in {hf_model_path}. "
            f"Expected pytorch_model.bin or model.safetensors"
        )
    
    def _detect_quantization(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Detect if model is quantized.
        
        Args:
            state_dict: Model state dict
            
        Returns:
            (is_quantized, quantization_info)
        """
        # Check for quantized tensors (int8, uint8)
        quantized_layers = []
        for name, tensor in state_dict.items():
            if tensor.dtype in [torch.int8, torch.uint8]:
                quantized_layers.append(name)
        
        if quantized_layers:
            logger.info(f"Found {len(quantized_layers)} quantized layers")
            
            # Try to find scale/zero_point tensors
            quant_info = self._extract_quantization_metadata(state_dict)
            
            return True, quant_info
        
        return False, None
    
    def _extract_quantization_metadata(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Extract quantization metadata from state dict.
        
        Args:
            state_dict: Model state dict
            
        Returns:
            Quantization metadata
        """
        metadata = {
            'scales': {},
            'zero_points': {},
            'quant_type': 'int8',  # Default
        }
        
        # Look for scale and zero_point tensors
        for name, tensor in state_dict.items():
            if 'scale' in name.lower():
                base_name = name.replace('_scale', '').replace('.scale', '')
                metadata['scales'][base_name] = tensor
            elif 'zero_point' in name.lower() or 'zp' in name.lower():
                base_name = name.replace('_zero_point', '').replace('.zero_point', '')
                metadata['zero_points'][base_name] = tensor
        
        logger.debug(f"Found {len(metadata['scales'])} scale tensors")
        logger.debug(f"Found {len(metadata['zero_points'])} zero_point tensors")
        
        return metadata
    
    def _convert_fp_model(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> List[Dict]:
        """
        Convert FP model to Parquet V1 rows.
        
        Args:
            state_dict: Model state dict
            
        Returns:
            List of row dictionaries
        """
        rows = []
        
        for layer_name, tensor in state_dict.items():
            # Skip non-weight tensors
            if not self._is_weight_tensor(layer_name):
                logger.debug(f"Skipping non-weight tensor: {layer_name}")
                continue
            
            # Convert tensor to bytes
            # Handle BFloat16 by converting to float32 first
            if tensor.dtype == torch.bfloat16:
                tensor_np = tensor.cpu().to(torch.float32).numpy()
            else:
                tensor_np = tensor.cpu().numpy()
            data_bytes = tensor_np.tobytes()
            
            # Get dtype string
            dtype_str = self.supported_dtypes.get(tensor.dtype, str(tensor.dtype))
            
            # Create row
            row = create_v1_row(
                layer_name=layer_name,
                shape=list(tensor.shape),
                dtype=dtype_str,
                data=data_bytes,
                num_params=int(tensor.numel())
            )
            
            rows.append(row)
            logger.debug(f"Converted {layer_name}: shape={tensor.shape}, dtype={dtype_str}")
        
        return rows
    
    def _convert_quantized_model(
        self,
        state_dict: Dict[str, torch.Tensor],
        quant_info: Dict[str, Any]
    ) -> List[Dict]:
        """
        Convert quantized model to Parquet V2 rows.
        
        Args:
            state_dict: Model state dict
            quant_info: Quantization metadata
            
        Returns:
            List of row dictionaries
        """
        rows = []
        
        for layer_name, tensor in state_dict.items():
            # Skip non-weight tensors and metadata tensors
            if not self._is_weight_tensor(layer_name):
                continue
            if 'scale' in layer_name.lower() or 'zero_point' in layer_name.lower():
                continue
            
            # Get quantization parameters
            scales = quant_info['scales'].get(layer_name, None)
            zero_points = quant_info['zero_points'].get(layer_name, None)
            
            # Determine quant_type
            if tensor.dtype == torch.int8:
                quant_type = 'int8'
            elif tensor.dtype == torch.uint8:
                quant_type = 'int8'  # Treat uint8 as int8
            else:
                # Not quantized, use FP16
                quant_type = 'fp16'
            
            # Convert tensor to bytes
            tensor_np = tensor.cpu().numpy()
            if quant_type in ['int8', 'int2']:
                data_bytes = tensor_np.astype(np.int8).tobytes()
            else:
                data_bytes = tensor_np.tobytes()
            
            # Get scales and zero_points as lists
            if scales is not None:
                scales_list = scales.cpu().numpy().tolist()
                if not isinstance(scales_list, list):
                    scales_list = [scales_list]
            else:
                # Default: per-tensor scale of 1.0
                scales_list = [1.0]
            
            if zero_points is not None:
                zp_list = zero_points.cpu().numpy().astype(np.int8).tolist()
                if not isinstance(zp_list, list):
                    zp_list = [zp_list]
            else:
                # Default: per-tensor zero_point of 0
                zp_list = [0]
            
            # Determine quant_axis
            if len(scales_list) > 1:
                quant_axis = 0  # Per-channel
            else:
                quant_axis = -1  # Per-tensor
            
            # Get original dtype
            dtype_str = self.supported_dtypes.get(tensor.dtype, str(tensor.dtype))
            
            # Create row
            row = create_v2_row(
                layer_name=layer_name,
                shape=list(tensor.shape),
                dtype=dtype_str,
                data=data_bytes,
                num_params=int(tensor.numel()),
                quant_type=quant_type,
                scales=scales_list,
                zero_points=zp_list,
                quant_axis=quant_axis
            )
            
            rows.append(row)
            logger.debug(
                f"Converted {layer_name}: shape={tensor.shape}, "
                f"quant_type={quant_type}, quant_axis={quant_axis}"
            )
        
        return rows
    
    def _is_weight_tensor(self, name: str) -> bool:
        """
        Check if tensor is a weight tensor (not metadata).
        
        Args:
            name: Tensor name
            
        Returns:
            True if weight tensor
        """
        # Skip common non-weight tensors
        skip_patterns = [
            'num_batches_tracked',
            'running_mean',
            'running_var',
            'position_ids',
            'token_type_ids',
        ]
        
        for pattern in skip_patterns:
            if pattern in name:
                return False
        
        return True


class ParquetToHuggingFaceConverter:
    """
    Convert Parquet models to HuggingFace format.
    
    This is the reverse operation of HuggingFaceToParquetConverter.
    Useful for:
    - Exporting AI-OS models to HuggingFace format
    - Enabling use of HuggingFace inference tools
    
    Note: This is a future feature, not yet implemented.
    """
    
    def __init__(self):
        raise NotImplementedError(
            "ParquetToHuggingFaceConverter is not yet implemented. "
            "This is a planned feature for future releases."
        )
    
    def convert(
        self,
        input_parquet: str,
        output_hf_path: str
    ) -> None:
        """
        Convert Parquet model to HuggingFace format.
        
        Args:
            input_parquet: Input Parquet file
            output_hf_path: Output HuggingFace model directory
        """
        raise NotImplementedError("Not yet implemented")


def convert_hf_to_parquet(
    hf_model_path: str,
    output_parquet: str,
    auto_detect_quantization: bool = True
) -> None:
    """
    Convenience function to convert HuggingFace model to Parquet.
    
    Args:
        hf_model_path: Path to HuggingFace model
        output_parquet: Output Parquet file path
        auto_detect_quantization: Auto-detect quantization
        
    Example:
        >>> convert_hf_to_parquet(
        ...     "models/qwen3-0.6b",
        ...     "models/qwen3-0.6b.parquet"
        ... )
    """
    converter = HuggingFaceToParquetConverter()
    converter.convert(hf_model_path, output_parquet, auto_detect_quantization)


def convert_parquet_to_hf(
    input_parquet: str,
    output_hf_path: str
) -> None:
    """
    Convenience function to convert Parquet model to HuggingFace.
    
    Args:
        input_parquet: Input Parquet file
        output_hf_path: Output HuggingFace model directory
        
    Note:
        This is a planned feature, not yet implemented.
    """
    raise NotImplementedError(
        "Parquet to HuggingFace conversion is not yet implemented"
    )

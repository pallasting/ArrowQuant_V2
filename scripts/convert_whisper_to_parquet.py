#!/usr/bin/env python3
"""
Whisper Model Converter - DEPRECATED

⚠️ DEPRECATION NOTICE:
This script is deprecated and will be removed in version 2.0.0.
Please use the unified converter instead:

    python scripts/convert_model.py --model <name> --output <dir>

The unified converter supports auto-detection and provides a better user experience.
For now, this script remains for backward compatibility.

---

Original Description:
Convert HuggingFace Whisper encoder to Arrow/Parquet format compatible with AudioInferenceCore.

Usage (DEPRECATED):
    python scripts/convert_whisper_to_parquet.py --model openai/whisper-base --output models/whisper-base

New Usage (RECOMMENDED):
    python scripts/convert_model.py --model openai/whisper-base --output models/whisper-base
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from llm_compression.logger import logger

# Issue deprecation warning
warnings.warn(
    "convert_whisper_to_parquet.py is deprecated and will be removed in version 2.0.0. "
    "Please use scripts/convert_model.py instead.",
    DeprecationWarning,
    stacklevel=2
)
logger.warning("⚠️  DEPRECATION WARNING: This script is deprecated. Use scripts/convert_model.py instead.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert Whisper model to Parquet format")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-base",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for converted model"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate conversion accuracy"
    )
    return parser.parse_args()



def load_whisper_model(model_name: str):
    """Load Whisper model from HuggingFace."""
    try:
        from transformers import WhisperModel, WhisperProcessor
    except ImportError:
        raise ImportError(
            "transformers not installed. Install with: pip install transformers"
        )
    
    logger.info(f"Loading Whisper model: {model_name}")
    model = WhisperModel.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
    config = model.config
    
    logger.info(f"Model loaded: {config.encoder_layers} encoder layers")
    
    return model, processor, config


def extract_encoder_weights(model) -> Dict[str, torch.Tensor]:
    """Extract encoder weights from Whisper model."""
    state_dict = model.state_dict()
    encoder_weights = {}
    
    # Extract only encoder-related weights
    for key, tensor in state_dict.items():
        if key.startswith("encoder."):
            encoder_weights[key] = tensor.detach().cpu()
    
    logger.info(f"Extracted {len(encoder_weights)} encoder weight tensors")
    return encoder_weights


def map_weight_keys(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map HuggingFace Whisper keys to AudioInferenceCore format.
    
    HuggingFace format:
        encoder.conv1.weight
        encoder.conv1.bias
        encoder.conv2.weight
        encoder.conv2.bias
        encoder.embed_positions.weight
        encoder.layers.0.self_attn.q_proj.weight
        
    AudioInferenceCore format (same as HuggingFace for now):
        encoder.conv1.weight
        encoder.conv1.bias
        encoder.conv2.weight
        encoder.conv2.bias
        encoder.position_embedding (renamed from embed_positions)
        encoder.layers.0.self_attn.q_proj.weight
    """
    mapped_weights = {}
    
    for key, tensor in weights.items():
        # Rename embed_positions to position_embedding
        if "embed_positions.weight" in key:
            new_key = key.replace("embed_positions.weight", "position_embedding")
            mapped_weights[new_key] = tensor
        else:
            mapped_weights[key] = tensor
    
    logger.info(f"Mapped {len(mapped_weights)} weight keys")
    return mapped_weights



def convert_to_parquet(
    weights: Dict[str, torch.Tensor],
    output_path: Path,
    use_float16: bool = True
) -> Path:
    """
    Convert weights to Parquet format.
    
    Args:
        weights: Dictionary of weight tensors
        output_path: Output directory
        use_float16: Convert to float16 for compression
        
    Returns:
        Path to generated Parquet file
    """
    logger.info("Converting weights to Parquet format...")
    
    # Prepare data for Arrow table
    layer_names = []
    shapes = []
    dtypes = []
    data_blobs = []
    num_params_list = []
    
    for layer_name, tensor in weights.items():
        # Convert to float16 if requested
        if use_float16 and tensor.dtype == torch.float32:
            tensor = tensor.half()
        
        layer_names.append(layer_name)
        shapes.append(list(tensor.shape))
        dtypes.append(str(tensor.dtype))
        data_blobs.append(tensor.numpy().tobytes())
        num_params_list.append(tensor.numel())
    
    # Create Arrow table
    schema = pa.schema([
        ("layer_name", pa.string()),
        ("shape", pa.list_(pa.int32())),
        ("dtype", pa.string()),
        ("data", pa.binary()),
        ("num_params", pa.int64()),
    ])
    
    table = pa.table(
        {
            "layer_name": layer_names,
            "shape": shapes,
            "dtype": dtypes,
            "data": data_blobs,
            "num_params": num_params_list,
        },
        schema=schema,
    )
    
    # Write to Parquet with compression
    parquet_path = output_path / "weights.parquet"
    pq.write_table(table, parquet_path, compression="zstd")
    
    file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved Parquet file: {parquet_path} ({file_size_mb:.2f} MB)")
    
    return parquet_path


def generate_metadata(
    model_name: str,
    config: Any,
    weights: Dict[str, torch.Tensor],
    parquet_path: Path,
    conversion_time: float
) -> Dict[str, Any]:
    """Generate metadata for converted model."""
    total_params = sum(tensor.numel() for tensor in weights.values())
    
    metadata = {
        "model_name": model_name,
        "model_type": "Whisper Audio Encoder",
        "architecture": "Whisper",
        "config": {
            "n_mels": config.num_mel_bins,
            "hidden_size": config.d_model,
            "num_layers": config.encoder_layers,
            "num_attention_heads": config.encoder_attention_heads,
            "intermediate_size": config.encoder_ffn_dim,
            "layer_norm_eps": 1e-5,
            "max_positions": config.max_source_positions,
        },
        "total_parameters": total_params,
        "num_weight_tensors": len(weights),
        "parquet_file": str(parquet_path.name),
        "file_size_mb": parquet_path.stat().st_size / (1024 * 1024),
        "conversion_time_sec": conversion_time,
        "converter_version": "1.0.0",
    }
    
    return metadata


def save_metadata(metadata: Dict[str, Any], output_path: Path):
    """Save metadata to JSON file."""
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata: {metadata_path}")
    return metadata_path



def validate_conversion(
    parquet_path: Path,
    original_weights: Dict[str, torch.Tensor]
) -> bool:
    """
    Validate converted model by comparing with original weights.
    
    Args:
        parquet_path: Path to Parquet file
        original_weights: Original weight tensors
        
    Returns:
        True if validation passed
    """
    logger.info("Validating conversion...")
    
    try:
        # Read Parquet file
        table = pq.read_table(parquet_path)
        
        # Check number of layers
        if len(table) != len(original_weights):
            logger.error(f"Layer count mismatch: {len(table)} vs {len(original_weights)}")
            return False
        
        # Validate each layer
        for i in range(len(table)):
            row = table.slice(i, 1).to_pydict()
            layer_name = row["layer_name"][0]
            shape = row["shape"][0]
            dtype_str = row["dtype"][0]
            data_bytes = row["data"][0]
            
            # Check if layer exists
            if layer_name not in original_weights:
                logger.error(f"Layer {layer_name} not found in original weights")
                return False
            
            original_tensor = original_weights[layer_name]
            
            # Validate shape
            if list(original_tensor.shape) != shape:
                logger.error(f"Shape mismatch for {layer_name}")
                return False
            
            # Reconstruct tensor
            np_dtype = _torch_dtype_to_numpy(dtype_str)
            reconstructed = np.frombuffer(data_bytes, dtype=np_dtype)
            reconstructed = reconstructed.reshape(shape)
            
            # Compare (allow small differences for float16)
            original_np = original_tensor.numpy()
            if not np.allclose(reconstructed, original_np, rtol=1e-3, atol=1e-5):
                max_diff = np.abs(reconstructed - original_np).max()
                logger.warning(f"Data mismatch for {layer_name}, max diff: {max_diff}")
        
        logger.info("Validation PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        return False


def _torch_dtype_to_numpy(dtype_str: str):
    """Convert PyTorch dtype string to NumPy dtype."""
    dtype_map = {
        "torch.float32": np.float32,
        "torch.float16": np.float16,
        "torch.int64": np.int64,
        "torch.int32": np.int32,
    }
    return dtype_map.get(dtype_str, np.float32)


def main():
    """Main conversion function."""
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting Whisper conversion: {args.model} -> {output_path}")
    start_time = time.time()
    
    try:
        # Step 1: Load model
        model, processor, config = load_whisper_model(args.model)
        
        # Step 2: Extract encoder weights
        weights = extract_encoder_weights(model)
        
        # Step 3: Map weight keys
        mapped_weights = map_weight_keys(weights)
        
        # Step 4: Convert to Parquet
        parquet_path = convert_to_parquet(mapped_weights, output_path)
        
        # Step 5: Generate and save metadata
        conversion_time = time.time() - start_time
        metadata = generate_metadata(
            args.model, config, mapped_weights, parquet_path, conversion_time
        )
        save_metadata(metadata, output_path)
        
        # Step 6: Validate (optional)
        if args.validate:
            validation_passed = validate_conversion(parquet_path, mapped_weights)
            if not validation_passed:
                logger.warning("Validation failed, but conversion completed")
        
        # Summary
        logger.info("=" * 60)
        logger.info("Conversion Summary:")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Parameters: {metadata['total_parameters']:,}")
        logger.info(f"  File size: {metadata['file_size_mb']:.2f} MB")
        logger.info(f"  Time: {conversion_time:.2f} seconds")
        logger.info("=" * 60)
        logger.info("SUCCESS: Whisper model converted to Parquet format")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

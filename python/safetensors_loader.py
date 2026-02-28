"""
SafeTensors Model Loader for ArrowQuant V2

This module provides Python utilities to load models from SafeTensors format
and prepare them for quantization with ArrowQuant V2.

Usage:
    from arrow_quant_v2.python.safetensors_loader import SafeTensorsLoader
    
    # Load model from SafeTensors
    loader = SafeTensorsLoader("model.safetensors")
    
    # Get model info
    print(f"Model has {len(loader.tensor_names())} layers")
    print(f"Detected modality: {loader.detect_modality()}")
    
    # Quantize with ArrowQuant V2
    from arrow_quant_v2 import ArrowQuantV2
    quantizer = ArrowQuantV2(mode="diffusion")
    
    result = quantizer.quantize_from_safetensors(
        safetensors_path="model.safetensors",
        output_path="model_quantized/",
        bit_width=2
    )
"""

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


class SafeTensorsLoader:
    """
    Load and parse SafeTensors format models.
    
    SafeTensors is a simple, safe format for storing tensors:
    - Header: JSON metadata (tensor names, shapes, dtypes, offsets)
    - Data: Raw tensor bytes in contiguous memory
    
    Attributes:
        path: Path to .safetensors file
        header: Parsed header with tensor metadata
        metadata: Model metadata (architecture, modality, etc.)
    """
    
    DTYPE_SIZES = {
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "I64": 8,
        "U8": 1,
    }
    
    def __init__(self, path: str):
        """
        Initialize loader with SafeTensors file.
        
        Args:
            path: Path to .safetensors file
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"SafeTensors file not found: {path}")
        
        self.header: Dict[str, Any] = {}
        self.metadata: Dict[str, str] = {}
        self._data_offset: int = 0
        
        self._load_header()
    
    def _load_header(self) -> None:
        """Load and parse SafeTensors header."""
        with open(self.path, "rb") as f:
            # Read header size (first 8 bytes, little-endian u64)
            header_size_bytes = f.read(8)
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            
            # Read header JSON
            header_bytes = f.read(header_size)
            header_json = json.loads(header_bytes.decode("utf-8"))
            
            # Store data offset
            self._data_offset = 8 + header_size
            
            # Parse header
            self.metadata = header_json.pop("__metadata__", {})
            self.header = header_json
    
    def tensor_names(self) -> List[str]:
        """Get list of all tensor names in the model."""
        return list(self.header.keys())
    
    def get_tensor_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific tensor.
        
        Args:
            name: Tensor name
            
        Returns:
            Dict with keys: dtype, shape, data_offsets
        """
        return self.header.get(name)
    
    def get_tensor(self, name: str) -> np.ndarray:
        """
        Load a single tensor as numpy array.
        
        Args:
            name: Tensor name
            
        Returns:
            Numpy array with tensor data (converted to float32)
        """
        info = self.get_tensor_info(name)
        if info is None:
            raise KeyError(f"Tensor not found: {name}")
        
        dtype = info["dtype"]
        shape = info["shape"]
        data_offsets = info["data_offsets"]
        
        # Read tensor data
        with open(self.path, "rb") as f:
            f.seek(self._data_offset + data_offsets[0])
            data_size = data_offsets[1] - data_offsets[0]
            data_bytes = f.read(data_size)
        
        # Convert to numpy array
        array = self._bytes_to_array(data_bytes, dtype, shape)
        
        return array
    
    def _bytes_to_array(
        self, data: bytes, dtype: str, shape: List[int]
    ) -> np.ndarray:
        """
        Convert raw bytes to numpy array.
        
        Args:
            data: Raw tensor bytes
            dtype: SafeTensors dtype string
            shape: Tensor shape
            
        Returns:
            Numpy array (float32)
        """
        # Map SafeTensors dtype to numpy dtype
        dtype_map = {
            "F32": np.float32,
            "F16": np.float16,
            "BF16": None,  # Special handling needed
            "I32": np.int32,
            "I64": np.int64,
            "U8": np.uint8,
        }
        
        if dtype == "BF16":
            # BF16 requires special handling
            array = self._load_bf16(data, shape)
        else:
            np_dtype = dtype_map.get(dtype)
            if np_dtype is None:
                raise ValueError(f"Unsupported dtype: {dtype}")
            
            array = np.frombuffer(data, dtype=np_dtype).reshape(shape)
        
        # Convert to float32 for quantization
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        
        return array
    
    def _load_bf16(self, data: bytes, shape: List[int]) -> np.ndarray:
        """
        Load BF16 (bfloat16) tensor.
        
        BF16 is float32 with lower 16 bits truncated.
        To convert: shift left by 16 bits.
        """
        # Read as uint16
        bf16_array = np.frombuffer(data, dtype=np.uint16)
        
        # Convert to float32 by shifting left 16 bits
        f32_bits = bf16_array.astype(np.uint32) << 16
        f32_array = f32_bits.view(np.float32)
        
        return f32_array.reshape(shape)
    
    def get_all_tensors(self) -> Dict[str, np.ndarray]:
        """
        Load all tensors as numpy arrays.
        
        Returns:
            Dict mapping tensor names to numpy arrays (float32)
        """
        tensors = {}
        for name in self.tensor_names():
            tensors[name] = self.get_tensor(name)
        return tensors
    
    def detect_modality(self) -> Optional[str]:
        """
        Detect model modality from metadata.
        
        Returns:
            Modality string: "text", "code", "image", "audio", or None
        """
        # Check explicit modality field
        if "modality" in self.metadata:
            return self.metadata["modality"]
        
        # Heuristics based on architecture
        if "architecture" in self.metadata:
            arch = self.metadata["architecture"].lower()
            
            if "text" in arch or "mdlm" in arch or "sedd" in arch:
                return "text"
            elif "code" in arch:
                return "code"
            elif "image" in arch or "dit" in arch or "vae" in arch:
                return "image"
            elif "audio" in arch or "wavegrad" in arch:
                return "audio"
        
        return None
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        return self.path.stat().st_size / (1024 * 1024)
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        total = 0
        for name in self.tensor_names():
            info = self.get_tensor_info(name)
            if info:
                shape = info["shape"]
                total += np.prod(shape)
        return int(total)
    
    def summary(self) -> str:
        """
        Get model summary.
        
        Returns:
            Human-readable summary string
        """
        lines = [
            f"SafeTensors Model: {self.path.name}",
            f"Size: {self.get_model_size_mb():.2f} MB",
            f"Parameters: {self.get_num_parameters():,}",
            f"Layers: {len(self.tensor_names())}",
        ]
        
        modality = self.detect_modality()
        if modality:
            lines.append(f"Modality: {modality}")
        
        if self.metadata:
            lines.append("\nMetadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


def convert_safetensors_to_parquet(
    safetensors_path: str,
    output_path: str,
    quantize: bool = False,
    bit_width: int = 8,
) -> None:
    """
    Convert SafeTensors model to Parquet format.
    
    Args:
        safetensors_path: Path to .safetensors file
        output_path: Output directory for Parquet files
        quantize: Whether to quantize during conversion
        bit_width: Target bit width if quantizing (2, 4, or 8)
    """
    from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
    
    loader = SafeTensorsLoader(safetensors_path)
    
    print(f"Loading model from {safetensors_path}")
    print(loader.summary())
    
    if quantize:
        print(f"\nQuantizing to INT{bit_width}...")
        
        # Create quantizer
        quantizer = ArrowQuantV2(mode="diffusion")
        
        # Detect modality
        modality = loader.detect_modality()
        
        # Create config
        config = DiffusionQuantConfig(
            bit_width=bit_width,
            modality=modality,
        )
        
        # Quantize
        result = quantizer.quantize_from_safetensors(
            safetensors_path=safetensors_path,
            output_path=output_path,
            config=config,
        )
        
        print(f"\nQuantization complete:")
        print(f"  Compression ratio: {result['compression_ratio']:.2f}x")
        print(f"  Cosine similarity: {result['cosine_similarity']:.4f}")
        print(f"  Output size: {result['model_size_mb']:.2f} MB")
    else:
        print(f"\nConverting to Parquet (no quantization)...")
        
        # Load all tensors
        tensors = loader.get_all_tensors()
        
        # Write to Parquet
        # TODO: Implement direct Parquet writing
        raise NotImplementedError("Direct conversion not yet implemented")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python safetensors_loader.py <model.safetensors>")
        sys.exit(1)
    
    path = sys.argv[1]
    loader = SafeTensorsLoader(path)
    print(loader.summary())



class ShardedSafeTensorsLoader:
    """
    Load and parse sharded SafeTensors models.
    
    Sharded models are split across multiple files for large models (>5GB).
    Format:
    - model.safetensors.index.json: Index file with shard mapping
    - model-00001-of-00005.safetensors: Shard files
    
    Attributes:
        base_dir: Directory containing shard files
        index: Parsed index with weight mapping
        metadata: Model metadata from index
    """
    
    def __init__(self, index_path: str):
        """
        Initialize loader with sharded model.
        
        Args:
            index_path: Path to .safetensors.index.json file or directory
        """
        self.index_path = Path(index_path)
        
        # Auto-detect if it's a directory or index file
        if self.index_path.is_dir():
            self.index_path = self._find_index_file(self.index_path)
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.base_dir = self.index_path.parent
        self.index: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self._shard_cache: Dict[str, SafeTensorsLoader] = {}
        
        self._load_index()
    
    @staticmethod
    def _find_index_file(directory: Path) -> Path:
        """Find index file in directory."""
        for file in directory.iterdir():
            if file.name.endswith(".safetensors.index.json"):
                return file
        raise FileNotFoundError(f"No index file found in {directory}")
    
    @staticmethod
    def is_sharded_model(path: str) -> bool:
        """
        Check if path is a sharded model.
        
        Args:
            path: Path to check (file or directory)
            
        Returns:
            True if sharded model, False otherwise
        """
        path_obj = Path(path)
        
        # Check if it's an index file
        if path_obj.is_file() and path_obj.name.endswith(".safetensors.index.json"):
            return True
        
        # Check if directory contains index file
        if path_obj.is_dir():
            for file in path_obj.iterdir():
                if file.name.endswith(".safetensors.index.json"):
                    return True
        
        return False
    
    def _load_index(self) -> None:
        """Load and parse index file."""
        with open(self.index_path, "r") as f:
            data = json.load(f)
        
        self.metadata = data.get("metadata", {})
        self.index = data.get("weight_map", {})
    
    def tensor_names(self) -> List[str]:
        """Get list of all tensor names across all shards."""
        return list(self.index.keys())
    
    def get_shard_for_tensor(self, tensor_name: str) -> Optional[str]:
        """
        Get shard file name for a specific tensor.
        
        Args:
            tensor_name: Name of the tensor
            
        Returns:
            Shard file name or None if not found
        """
        return self.index.get(tensor_name)
    
    def _load_shard(self, shard_name: str) -> SafeTensorsLoader:
        """Load a shard (with caching)."""
        if shard_name not in self._shard_cache:
            shard_path = self.base_dir / shard_name
            self._shard_cache[shard_name] = SafeTensorsLoader(str(shard_path))
        return self._shard_cache[shard_name]
    
    def get_tensor(self, name: str) -> np.ndarray:
        """
        Load a single tensor as numpy array.
        
        Args:
            name: Tensor name
            
        Returns:
            Numpy array with tensor data (float32)
        """
        shard_name = self.get_shard_for_tensor(name)
        if shard_name is None:
            raise KeyError(f"Tensor not found: {name}")
        
        shard = self._load_shard(shard_name)
        return shard.get_tensor(name)
    
    def get_all_tensors(self) -> Dict[str, np.ndarray]:
        """
        Load all tensors as numpy arrays.
        
        Loads shards on-demand to minimize memory usage.
        
        Returns:
            Dict mapping tensor names to numpy arrays (float32)
        """
        tensors = {}
        for name in self.tensor_names():
            tensors[name] = self.get_tensor(name)
        return tensors
    
    def detect_modality(self) -> Optional[str]:
        """
        Detect model modality from metadata.
        
        Returns:
            Modality string: "text", "code", "image", "audio", or None
        """
        # Check explicit modality field
        if "modality" in self.metadata:
            return self.metadata["modality"]
        
        # Heuristics based on architecture
        if "architecture" in self.metadata:
            arch = self.metadata["architecture"].lower()
            
            if "text" in arch or "mdlm" in arch:
                return "text"
            elif "code" in arch:
                return "code"
            elif "image" in arch or "dit" in arch:
                return "image"
            elif "audio" in arch:
                return "audio"
        
        return None
    
    def get_total_size(self) -> Optional[int]:
        """Get total model size in bytes from metadata."""
        return self.metadata.get("total_size")
    
    def num_shards(self) -> int:
        """Get number of shard files."""
        return len(set(self.index.values()))
    
    def shard_files(self) -> List[str]:
        """Get list of all shard file names."""
        return sorted(set(self.index.values()))
    
    def clear_cache(self) -> None:
        """Clear shard cache to free memory."""
        self._shard_cache.clear()
    
    def cache_memory_usage(self) -> int:
        """
        Get approximate memory usage of cached shards.
        
        Returns:
            Memory usage in bytes (approximate)
        """
        # Rough estimate: each cached shard uses ~100MB on average
        return len(self._shard_cache) * 100 * 1024 * 1024
    
    def get_model_size_mb(self) -> float:
        """Get total model size in MB."""
        total_size = self.get_total_size()
        if total_size:
            return total_size / (1024 * 1024)
        
        # Fallback: sum shard file sizes
        total = 0
        for shard_name in self.shard_files():
            shard_path = self.base_dir / shard_name
            if shard_path.exists():
                total += shard_path.stat().st_size
        return total / (1024 * 1024)
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        total = 0
        
        # Load each shard and count parameters
        for shard_name in self.shard_files():
            shard = self._load_shard(shard_name)
            for name in shard.tensor_names():
                info = shard.get_tensor_info(name)
                if info:
                    shape = info["shape"]
                    total += np.prod(shape)
        
        return int(total)
    
    def summary(self) -> str:
        """
        Get model summary.
        
        Returns:
            Human-readable summary string
        """
        lines = [
            f"Sharded SafeTensors Model: {self.index_path.parent.name}",
            f"Size: {self.get_model_size_mb():.2f} MB",
            f"Shards: {self.num_shards()}",
            f"Layers: {len(self.tensor_names())}",
        ]
        
        modality = self.detect_modality()
        if modality:
            lines.append(f"Modality: {modality}")
        
        if self.metadata:
            lines.append("\nMetadata:")
            for key, value in self.metadata.items():
                if key != "total_size":  # Already shown
                    lines.append(f"  {key}: {value}")
        
        lines.append("\nShard files:")
        for shard in self.shard_files():
            lines.append(f"  - {shard}")
        
        return "\n".join(lines)


def load_safetensors_model(path: str) -> Tuple[Dict[str, np.ndarray], Optional[str]]:
    """
    Load a SafeTensors model (single or sharded).
    
    Auto-detects whether the model is sharded or single-file.
    
    Args:
        path: Path to .safetensors file, .safetensors.index.json, or directory
        
    Returns:
        Tuple of (tensors dict, modality string)
    """
    path_obj = Path(path)
    
    # Check if sharded
    if ShardedSafeTensorsLoader.is_sharded_model(path):
        print(f"Detected sharded model at {path}")
        loader = ShardedSafeTensorsLoader(path)
        print(loader.summary())
        return loader.get_all_tensors(), loader.detect_modality()
    else:
        print(f"Detected single-file model at {path}")
        loader = SafeTensorsLoader(path)
        print(loader.summary())
        return loader.get_all_tensors(), loader.detect_modality()

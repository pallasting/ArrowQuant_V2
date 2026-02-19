"""
Vector Space Compressor for ArrowEngine.

Based on Design Spec: ARROWENGINE_NATIVE_ARCHITECTURE_VISION.md
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np
import torch

# Define CompressedMemory data structure
from .attention_extractor import AttentionBasedExtractor

@dataclass
class CompressedMemory:
    """Compressed memory representation."""
    sparse_vector: np.ndarray # Quantized sparse values (int8)
    sparse_indices: np.ndarray # Indices of kept dimensions (uint16)
    original_norm: float       # L2 norm for reconstruction scaling
    key_tokens: List[str] = field(default_factory=list)
    token_scores: List[float] = field(default_factory=list)
    relations: List[Tuple[str, str, float]] = field(default_factory=list)
    sparse_meta: Optional[dict] = None
    meta_info: Optional[dict] = None
    
    # Storage Compatibility Fields
    memory_id: str = ""
    summary_hash: str = ""
    entities: Dict[str, List[str]] = field(default_factory=dict)
    diff_data: bytes = b""
    embedding: List[float] = field(default_factory=list)
    compression_metadata: Optional[Any] = None
    original_fields: Dict[str, Any] = field(default_factory=dict)



class VectorSpaceCompressor:
    """
    Compresses text into vector space representations using ArrowEngine.
    """
    
    def __init__(self, arrow_engine):
        self.engine = arrow_engine
        self.attn_extractor = AttentionBasedExtractor(arrow_engine)
        self.device = arrow_engine.device
        
    def compress(
        self, 
        text: str, 
        compression_ratio: float = 0.3,
        dimension_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
        use_4bit: bool = False
    ) -> CompressedMemory:
        """
        Compress text into a sparse quantized vector (Single sequence).
        """
        results = self.compress_batch(
            [text], 
            compression_ratio=compression_ratio, 
            dimension_weights=dimension_weights,
            use_4bit=use_4bit
        )
        return results[0]

    def compress_batch(
        self,
        texts: List[str],
        compression_ratio: float = 0.3,
        dimension_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
        use_4bit: bool = False
    ) -> List[CompressedMemory]:
        """
        Compress multiple texts into sparse quantized vectors (GPU accelerated).
        
        Args:
            texts: List of input strings
            compression_ratio: Ratio of dimensions to keep
            dimension_weights: Optional weights (frequency-based)
            use_4bit: Whether to use 4-bit bit-packing
            
        Returns:
            List of CompressedMemory objects
        """
        # 1. Encode with attention (Batch)
        embeddings, attentions = self.engine.encode(
            texts, 
            output_attentions=True,
            normalize=True
        )
        
        # Ensure Torch tensors on device
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).to(self.device)
        
        # 2. Identify key dimensions (Vectorized across batch)
        # Shape: (batch, dim)
        abs_embedding = torch.abs(embeddings)
        
        if dimension_weights is not None:
            if isinstance(dimension_weights, np.ndarray):
                dimension_weights = torch.from_numpy(dimension_weights).to(self.device)
            # Apply weights: (batch, dim) * (dim,)
            abs_embedding = abs_embedding * dimension_weights
                
        # Top-k selection
        k = int(embeddings.shape[1] * compression_ratio)
        top_k_results = torch.topk(abs_embedding, k, dim=1)
        key_indices_batch = top_k_results.indices # (batch, k)
        
        # 3. Process each memory in batch
        results = []
        
        # Merge attention extraction with existing batch attentions to avoid double forward pass
        key_info_batch = self.attn_extractor.extract_key_information_batch(
            texts, 
            attentions=attentions
        )
        
        for i in range(len(texts)):
            embedding = embeddings[i]
            key_indices = key_indices_batch[i]
            key_info = key_info_batch[i]
            
            # Extract values
            original_norm = torch.norm(embedding).item()
            values = embedding[key_indices]
            
            # Quantize on device
            max_val = torch.max(torch.abs(values)).item() if values.numel() > 0 else 1.0
            if max_val == 0: max_val = 1.0
            
            if use_4bit:
                # 4-bit Quantization (16 levels: -8 to +7)
                # Map [-max, max] to [-8, 7]
                scale_factor = 7.0 / max_val
                
                # Round and clip to valid range
                int_values = torch.round(values * scale_factor).to(torch.int8)
                int_values = torch.clamp(int_values, -8, 7)
                
                # Move to CPU for bit-packing logic (numpy is easier for byte-level)
                int_values_np = int_values.cpu().numpy()
                count = len(int_values_np)
                if count % 2 != 0:
                    int_values_np = np.append(int_values_np, 0)
                
                # Fast packing with numpy
                # Mask to 4 bits (0x0F)
                val_masked = int_values_np.astype(np.uint8) & 0x0F
                # Pack two into one uint8: High nibble first
                packed = (val_masked[0::2] << 4) | val_masked[1::2]
                quantized = packed.astype(np.uint8).tobytes()
            else:
                # Standard 8-bit Quantization
                scale_factor = 127.0 / max_val
                int_values = torch.round(values * scale_factor).to(torch.int8)
                quantized = int_values.cpu().numpy().tobytes()
                
            results.append(CompressedMemory(
                sparse_vector=quantized,
                sparse_indices=key_indices.cpu().numpy().astype(np.uint16).tobytes(),
                original_norm=original_norm,
                key_tokens=key_info.key_tokens if key_info else [],
                token_scores=key_info.token_scores if key_info else [],
                relations=key_info.relations if key_info and hasattr(key_info, 'relations') else [],
                sparse_meta={
                    "scale_factor": float(scale_factor),
                    "full_dim": embeddings.shape[1],
                    "original_norm": float(original_norm),
                    "is_4bit": use_4bit,
                    "packed_length": len(quantized),
                    "original_count": len(values) # Important for reconstruction
                },
                meta_info={
                    "scale_factor": float(scale_factor),
                    "full_dim": embeddings.shape[1],
                    "is_4bit": use_4bit,
                    "packed_length": len(key_indices) if use_4bit else 0,
                    "key_tokens": key_info.key_tokens if key_info else [],
                    "token_scores": key_info.token_scores if key_info else []
                }
            ))
            
        return results

    def compress_vector(
        self,
        vector: np.ndarray,
        compression_ratio: float = 0.5,
        use_4bit: bool = True
    ) -> CompressedMemory:
        """
        Compress a raw vector (e.g. image embedding) without text processing.
        """
        # Wrap as batch of 1 tensor
        tensor = torch.from_numpy(vector).unsqueeze(0).to(self.device).float()
        
        # Call internal logic manually since compress_batch expects text
        # 1. Selection
        k = int(tensor.shape[1] * compression_ratio)
        top_k = torch.topk(torch.abs(tensor), k, dim=1)
        indices = top_k.indices[0]
        values = tensor[0][indices]
        original_norm = torch.norm(tensor).item()
        
        # 2. Quantization (4-bit)
        max_val = torch.max(torch.abs(values)).item() if values.numel() > 0 else 1.0
        if max_val == 0: max_val = 1.0
        
        scale_factor = 7.0 / max_val
        int_values = torch.round(values * scale_factor).to(torch.int8)
        int_values = torch.clamp(int_values, -8, 7)
        
        # Pack
        int_values_np = int_values.cpu().numpy()
        if len(int_values_np) % 2 != 0:
            int_values_np = np.append(int_values_np, 0)
            
        val_masked = int_values_np.astype(np.uint8) & 0x0F
        packed = (val_masked[0::2] << 4) | val_masked[1::2]
        quantized = packed.astype(np.uint8).tobytes()
        
        return CompressedMemory(
            sparse_vector=quantized,
            sparse_indices=indices.cpu().numpy().astype(np.uint16).tobytes(),
            original_norm=original_norm,
            sparse_meta={
                "scale_factor": float(scale_factor),
                "full_dim": tensor.shape[1],
                "original_norm": float(original_norm),
                "is_4bit": True,
                "packed_length": len(quantized),
                "original_count": len(values)
            },
            meta_info={}
        )
    
    def reconstruct(self, compressed: CompressedMemory) -> np.ndarray:
        """Reconstruct full embedding vector."""
        # Handle meta access
        sparse_meta = getattr(compressed, "sparse_meta", None)
        if sparse_meta is None:
            sparse_meta = getattr(compressed, "meta_info", {})
            
        full_dim = sparse_meta.get("full_dim", 384)
        reconstructed = np.zeros(full_dim, dtype=np.float32)
        
        # print(f"DEBUG Reconstruct: full_dim={full_dim}")
        
        scale_factor = sparse_meta.get("scale_factor", 1.0)
        is_4bit = sparse_meta.get("is_4bit", False)
        
        if is_4bit:
            # Unpack 4-bit values
            if sparse_meta is None: # Should be redundant now but keeping logic structure
                sparse_meta = getattr(compressed, "meta_info", {})
                
            sparse_vector = compressed.sparse_vector
            if isinstance(sparse_vector, bytes):
                sparse_vector = np.frombuffer(sparse_vector, dtype=np.uint8)
                
            packed = sparse_vector.view(np.uint8) 
            packing_count = sparse_meta.get("packed_length", len(packed))
            original_count = sparse_meta.get("original_count", packing_count * 2) 
            
            # Unpack high/low nibbles
            high_nibbles = (packed >> 4) 
            low_nibbles  = (packed & 0x0F)
            
            # Combine back into single array
            # Interleave: [h0, l0, h1, l1, ...]
            unpacked = np.empty(len(packed) * 2, dtype=np.int8)
            unpacked[0::2] = high_nibbles
            unpacked[1::2] = low_nibbles
            
            # 4-bit signed conversion
            mask_neg = unpacked >= 8
            unpacked[mask_neg] -= 16
            
            # Trim padding
            values = unpacked[:original_count].astype(np.float32)
            # print(f"DEBUG Unpacked range: {np.min(values)} to {np.max(values)}, scale: {scale_factor}")
        else:
            sparse_vector = compressed.sparse_vector
            if isinstance(sparse_vector, bytes):
                 sparse_vector = np.frombuffer(sparse_vector, dtype=np.float32)
            values = sparse_vector.astype(np.float32)
            
        # Apply scale_factor
        values = values / scale_factor
            
        # Handle different field names for indices
        key_indices = getattr(compressed, "key_indices", None)
        if key_indices is None:
            key_indices = getattr(compressed, "sparse_indices", None)
            
        if isinstance(key_indices, bytes):
            # We used uint16 in compress() to save space
            key_indices = np.frombuffer(key_indices, dtype=np.uint16)
            
        # Place values back
        valid_len = min(len(key_indices), len(values))
        reconstructed[key_indices[:valid_len]] = values[:valid_len]
        
        # Original Norm
        original_norm = getattr(compressed, "original_norm", None)
        if original_norm is None and sparse_meta:
            original_norm = sparse_meta.get("original_norm", 1.0)
        if original_norm is None:
            original_norm = 1.0
            
        # Renormalize
        current_norm = np.linalg.norm(reconstructed)
        if current_norm > 0:
            reconstructed = reconstructed * (original_norm / current_norm)
            
        return reconstructed

    def _summarize_attention(self, attentions) -> dict:
        """Helper to summarize attention for verification/stats."""
        # attentions is a list (batches) of tuples (layers) of tensors
        # We take the first batch
        if not attentions or not attentions[0]:
            return {}
            
        layer_attentions = attentions[0] # Tuple of tensors
        
        # Get stats from last layer
        last_layer = layer_attentions[-1] # (1, heads, seq, seq)
        avg_attention = float(torch.mean(last_layer).item())
        max_attention = float(torch.max(last_layer).item())
        
        return {
            "num_layers": len(layer_attentions),
            "last_layer_mean": avg_attention,
            "last_layer_max": max_attention
        }

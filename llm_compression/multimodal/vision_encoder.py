"""
VisionEncoder - High-level Vision Transformer Encoder

Wraps VisionInferenceCore with preprocessing, weight loading, and
a unified interface compatible with EmbeddingProvider protocol.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import torch

from llm_compression.logger import logger
from llm_compression.inference.vision_core import VisionInferenceCore
from llm_compression.inference.weight_loader import WeightLoader
from llm_compression.multimodal.image_processor import ImageProcessor


@dataclass
class VisionConfig:
    """Vision Transformer configuration."""
    image_size: int = 224
    patch_size: int = 32
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-5
    projection_dim: int = 512
    
    def to_dict(self):
        """Convert to dictionary for VisionInferenceCore."""
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "layer_norm_eps": self.layer_norm_eps,
            "projection_dim": self.projection_dim,
        }


class VisionEncoder:
    """
    CLIP Vision Transformer encoder.
    
    High-level interface that combines:
    - ImageProcessor for preprocessing
    - VisionInferenceCore for encoding
    - WeightLoader for loading from Parquet
    
    Features:
    - Fast loading (<500ms target)
    - Fast inference (<100ms target)
    - Arrow-native architecture
    - Reuses TransformerLayer from InferenceCore
    
    Example:
        encoder = VisionEncoder("models/clip-vit-b32")
        embedding = encoder.encode(image)  # (512,) float32
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        config: Optional[VisionConfig] = None
    ):
        """
        Initialize Vision Encoder.
        
        Args:
            model_path: Path to model directory with weights.parquet
            device: Device for inference ('cpu', 'cuda', or None for auto)
            config: Vision configuration (default: CLIP ViT-B/32)
        """
        self.model_path = Path(model_path)
        self.device = device or self._get_default_device()
        self.config = config or VisionConfig()
        
        logger.info(f"Initializing VisionEncoder from {self.model_path}")
        
        # Initialize image processor
        self.image_processor = ImageProcessor(
            image_size=self.config.image_size
        )
        
        # Load weights
        weights_path = self.model_path / "weights.parquet"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        self.weight_loader = WeightLoader(str(weights_path))
        weights = self.weight_loader.load_weights()
        
        # Initialize core
        self.core = VisionInferenceCore(
            weights=weights,
            config=self.config.to_dict(),
            device=self.device
        )
        
        logger.info(
            f"VisionEncoder initialized: "
            f"image_size={self.config.image_size}, "
            f"patch_size={self.config.patch_size}, "
            f"hidden_size={self.config.hidden_size}, "
            f"projection_dim={self.config.projection_dim}, "
            f"device={self.device}"
        )
    
    def _get_default_device(self) -> str:
        """Get default device (CPU or CUDA)."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.projection_dim
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension (compatibility method)."""
        return self.embedding_dimension
    
    def encode(
        self,
        images: Union[np.ndarray, List[np.ndarray], str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode images to embeddings.
        
        Args:
            images: Single image or batch of images
                   Can be: numpy array, file path, or list of either
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Embeddings: (batch, projection_dim) float32 numpy array
        """
        # Handle single image
        if isinstance(images, (str, np.ndarray)):
            if isinstance(images, str) or (isinstance(images, np.ndarray) and images.ndim == 3):
                images = [images]
        
        # Preprocess images
        preprocessed = self.image_processor.preprocess_batch(images)
        
        # Convert to tensor (batch, 224, 224, 3) -> (batch, 3, 224, 224)
        image_tensor = torch.from_numpy(preprocessed).permute(0, 3, 1, 2).to(self.device)
        
        # Encode
        with torch.no_grad():
            embeddings = self.core(image_tensor)
        
        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        
        # Return single embedding if single image
        if len(embeddings_np) == 1:
            return embeddings_np[0]
        
        return embeddings_np
    
    def encode_batch(
        self,
        images: List[Union[np.ndarray, str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode batch of images with optional batching.
        
        Args:
            images: List of images (numpy arrays or file paths)
            batch_size: Batch size for processing
            normalize: Whether to L2-normalize embeddings
            show_progress: Whether to show progress (not implemented yet)
            
        Returns:
            Embeddings: (n_images, projection_dim) float32 numpy array
        """
        if len(images) == 0:
            return np.zeros((0, self.embedding_dimension), dtype=np.float32)
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_embeddings = self.encode(batch, normalize=normalize)
            
            # Handle single vs batch output
            if batch_embeddings.ndim == 1:
                batch_embeddings = batch_embeddings[np.newaxis, :]
            
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def __repr__(self) -> str:
        return (
            f"VisionEncoder("
            f"model_path={self.model_path}, "
            f"device={self.device}, "
            f"embedding_dim={self.embedding_dimension})"
        )

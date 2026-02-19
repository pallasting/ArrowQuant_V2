"""
CLIPEngine - Dual-Encoder Multimodal Fusion

Combines text and vision encoders with projection layers for
cross-modal text-image understanding and retrieval.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from llm_compression.logger import logger
from llm_compression.inference.arrow_engine import ArrowEngine
from llm_compression.inference.weight_loader import WeightLoader
from llm_compression.multimodal.vision_encoder import VisionEncoder


@dataclass
class CLIPConfig:
    """CLIP engine configuration."""
    text_embedding_dim: int = 384  # BERT base output
    vision_embedding_dim: int = 768  # ViT-B/32 output
    projection_dim: int = 512  # Shared embedding space
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.text_embedding_dim > 0, "text_embedding_dim must be positive"
        assert self.vision_embedding_dim > 0, "vision_embedding_dim must be positive"
        assert self.projection_dim > 0, "projection_dim must be positive"


class CLIPEngine:
    """
    CLIP dual-encoder for text-image understanding.
    
    Combines:
    - Text encoder (ArrowEngine with BERT)
    - Vision encoder (VisionEncoder with ViT)
    - Projection layers to shared 512-dim space
    - Contrastive similarity computation with temperature scaling
    
    Features:
    - Fast loading (<1s for both encoders)
    - Fast inference (<200ms for text-image pairs)
    - Arrow-native architecture
    - Cross-modal retrieval
    
    Example:
        engine = CLIPEngine("models/clip")
        text_emb = engine.encode_text(["a cat", "a dog"])
        image_emb = engine.encode_image(images)
        similarity = engine.compute_similarity(text_emb, image_emb)
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        config: Optional[CLIPConfig] = None
    ):
        """
        Initialize CLIP Engine.
        
        Args:
            model_path: Path to model directory containing:
                       - text/ (ArrowEngine text encoder)
                       - vision/ (VisionEncoder)
                       - weights.parquet (projection layers + logit_scale)
            device: Device for inference ('cpu', 'cuda', or None for auto)
            config: CLIP configuration (default: CLIP ViT-B/32 + BERT base)
        """
        self.model_path = Path(model_path)
        self.device = device or self._get_default_device()
        self.config = config or CLIPConfig()
        
        logger.info(f"Initializing CLIPEngine from {self.model_path}")
        
        # Initialize text encoder (ArrowEngine)
        text_model_path = self.model_path / "text"
        if not text_model_path.exists():
            # Fallback: try loading from model_path directly
            text_model_path = self.model_path
        
        logger.info(f"Loading text encoder from {text_model_path}")
        self.text_encoder = ArrowEngine(
            model_path=str(text_model_path),
            device=self.device
        )
        
        # Initialize vision encoder
        vision_model_path = self.model_path / "vision"
        if not vision_model_path.exists():
            raise FileNotFoundError(
                f"Vision encoder not found at {vision_model_path}. "
                f"Expected directory structure: {self.model_path}/vision/"
            )
        
        logger.info(f"Loading vision encoder from {vision_model_path}")
        self.vision_encoder = VisionEncoder(
            model_path=str(vision_model_path),
            device=self.device
        )
        
        # Initialize projection layers
        self.text_projection = nn.Linear(
            self.config.text_embedding_dim,
            self.config.projection_dim,
            bias=False
        )
        self.vision_projection = nn.Linear(
            self.config.vision_embedding_dim,
            self.config.projection_dim,
            bias=False
        )
        
        # Temperature parameter for contrastive learning
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))
        
        # Load projection weights and logit_scale
        self._load_projection_weights()
        
        # Move to device
        self.text_projection.to(self.device)
        self.vision_projection.to(self.device)
        self.logit_scale.data = self.logit_scale.data.to(self.device)
        
        # Set to eval mode
        self.text_projection.eval()
        self.vision_projection.eval()
        
        logger.info(
            f"CLIPEngine initialized: "
            f"text_dim={self.config.text_embedding_dim}, "
            f"vision_dim={self.config.vision_embedding_dim}, "
            f"projection_dim={self.config.projection_dim}, "
            f"device={self.device}"
        )
    
    def _get_default_device(self) -> str:
        """Get default device (CPU or CUDA)."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _load_projection_weights(self):
        """Load projection layer weights and logit_scale from Parquet."""
        weights_path = self.model_path / "weights.parquet"
        
        if not weights_path.exists():
            logger.warning(
                f"Projection weights not found at {weights_path}. "
                f"Using random initialization."
            )
            return
        
        try:
            weight_loader = WeightLoader(str(weights_path))
            weights = weight_loader.load_weights()
            
            # Load text projection
            if "text_projection.weight" in weights:
                self.text_projection.weight.data = torch.from_numpy(
                    weights["text_projection.weight"]
                )
                logger.debug("Loaded text_projection weights")
            else:
                logger.warning("text_projection.weight not found in weights file")
            
            # Load vision projection
            if "visual_projection.weight" in weights:
                self.vision_projection.weight.data = torch.from_numpy(
                    weights["visual_projection.weight"]
                )
                logger.debug("Loaded visual_projection weights")
            else:
                logger.warning("visual_projection.weight not found in weights file")
            
            # Load logit_scale
            if "logit_scale" in weights:
                logit_scale_value = weights["logit_scale"]
                if isinstance(logit_scale_value, np.ndarray):
                    if logit_scale_value.ndim == 0:
                        logit_scale_value = logit_scale_value.item()
                    else:
                        logit_scale_value = logit_scale_value.flatten()[0]
                
                self.logit_scale.data = torch.tensor(float(logit_scale_value))
                logger.debug(f"Loaded logit_scale: {self.logit_scale.item():.4f}")
            else:
                logger.warning("logit_scale not found in weights file, using default")
        
        except Exception as e:
            logger.error(f"Failed to load projection weights: {e}")
            logger.warning("Using random initialization for projection layers")
    
    @property
    def embedding_dimension(self) -> int:
        """Get shared embedding dimension."""
        return self.config.projection_dim
    
    def get_embedding_dimension(self) -> int:
        """Get shared embedding dimension (compatibility method)."""
        return self.embedding_dimension
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to CLIP embedding space.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Embeddings: (batch, projection_dim) float32 numpy array
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Get text embeddings from ArrowEngine (384-dim)
        text_features = self.text_encoder.encode(texts, normalize=False)
        
        # Convert to tensor
        text_tensor = torch.from_numpy(text_features).to(self.device)
        
        # Project to shared space (384 → 512)
        with torch.no_grad():
            text_embeddings = self.text_projection(text_tensor)
        
        # Normalize if requested
        if normalize:
            text_embeddings = torch.nn.functional.normalize(
                text_embeddings, p=2, dim=-1
            )
        
        # Convert to numpy
        embeddings_np = text_embeddings.cpu().numpy().astype(np.float32)
        
        # Return single embedding if single text
        if len(embeddings_np) == 1 and isinstance(texts, list) and len(texts) == 1:
            return embeddings_np[0]
        
        return embeddings_np
    
    def encode_image(
        self,
        images: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode images to CLIP embedding space.
        
        Args:
            images: Single image or batch of images
                   Shape: (224, 224, 3) or (batch, 224, 224, 3)
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Embeddings: (batch, projection_dim) float32 numpy array
        """
        # Get vision embeddings (768-dim)
        vision_features = self.vision_encoder.encode(images, normalize=False)
        
        # Handle single image case
        if vision_features.ndim == 1:
            vision_features = vision_features[np.newaxis, :]
        
        # Convert to tensor
        vision_tensor = torch.from_numpy(vision_features).to(self.device)
        
        # Project to shared space (768 → 512)
        with torch.no_grad():
            image_embeddings = self.vision_projection(vision_tensor)
        
        # Normalize if requested
        if normalize:
            image_embeddings = torch.nn.functional.normalize(
                image_embeddings, p=2, dim=-1
            )
        
        # Convert to numpy
        embeddings_np = image_embeddings.cpu().numpy().astype(np.float32)
        
        # Return single embedding if single image
        if len(embeddings_np) == 1:
            return embeddings_np[0]
        
        return embeddings_np

    
    def compute_similarity(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
        apply_temperature: bool = True
    ) -> np.ndarray:
        """
        Compute text-image similarity matrix.
        
        Args:
            text_embeddings: (n_texts, projection_dim) text embeddings
            image_embeddings: (n_images, projection_dim) image embeddings
            apply_temperature: Whether to apply temperature scaling
            
        Returns:
            Similarity matrix: (n_texts, n_images) float32 array
        """
        # Handle single embeddings
        if text_embeddings.ndim == 1:
            text_embeddings = text_embeddings[np.newaxis, :]
        if image_embeddings.ndim == 1:
            image_embeddings = image_embeddings[np.newaxis, :]
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarity = text_embeddings @ image_embeddings.T
        
        # Apply temperature scaling
        if apply_temperature:
            logit_scale = self.logit_scale.cpu().item()
            similarity = similarity * np.exp(logit_scale)
        
        return similarity.astype(np.float32)
    
    def find_best_matches(
        self,
        texts: Union[str, List[str]],
        images: np.ndarray,
        top_k: int = 5
    ) -> List[List[int]]:
        """
        Find best matching images for each text query.
        
        Args:
            texts: Single text or list of text queries
            images: (n_images, 224, 224, 3) or (224, 224, 3) image array
            top_k: Number of top matches to return per query
            
        Returns:
            List of top-k image indices for each text query
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode both modalities
        text_emb = self.encode_text(texts, normalize=True)
        image_emb = self.encode_image(images, normalize=True)
        
        # Ensure 2D arrays
        if text_emb.ndim == 1:
            text_emb = text_emb[np.newaxis, :]
        if image_emb.ndim == 1:
            image_emb = image_emb[np.newaxis, :]
        
        # Compute similarity
        similarity = self.compute_similarity(text_emb, image_emb, apply_temperature=True)
        
        # Get top-k for each text
        matches = []
        for sim_row in similarity:
            # Get indices of top-k highest similarities
            top_indices = np.argsort(sim_row)[-top_k:][::-1]
            matches.append(top_indices.tolist())
        
        return matches
    
    def find_best_text_matches(
        self,
        images: np.ndarray,
        texts: List[str],
        top_k: int = 5
    ) -> List[List[int]]:
        """
        Find best matching texts for each image.
        
        Args:
            images: (n_images, 224, 224, 3) or (224, 224, 3) image array
            texts: List of text candidates
            top_k: Number of top matches to return per image
            
        Returns:
            List of top-k text indices for each image
        """
        # Encode both modalities
        image_emb = self.encode_image(images, normalize=True)
        text_emb = self.encode_text(texts, normalize=True)
        
        # Ensure 2D arrays
        if image_emb.ndim == 1:
            image_emb = image_emb[np.newaxis, :]
        if text_emb.ndim == 1:
            text_emb = text_emb[np.newaxis, :]
        
        # Compute similarity (transpose for image-to-text)
        similarity = self.compute_similarity(text_emb, image_emb, apply_temperature=True)
        similarity = similarity.T  # (n_images, n_texts)
        
        # Get top-k for each image
        matches = []
        for sim_row in similarity:
            # Get indices of top-k highest similarities
            top_indices = np.argsort(sim_row)[-top_k:][::-1]
            matches.append(top_indices.tolist())
        
        return matches
    
    def zero_shot_classification(
        self,
        images: np.ndarray,
        class_texts: List[str],
        return_probabilities: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform zero-shot image classification.
        
        Args:
            images: (n_images, 224, 224, 3) or (224, 224, 3) image array
            class_texts: List of class descriptions (e.g., ["a photo of a cat", "a photo of a dog"])
            return_probabilities: Whether to return softmax probabilities
            
        Returns:
            If return_probabilities=True:
                (predicted_classes, probabilities)
                - predicted_classes: (n_images,) int array of class indices
                - probabilities: (n_images, n_classes) float array of class probabilities
            If return_probabilities=False:
                predicted_classes: (n_images,) int array of class indices
        """
        # Encode both modalities
        image_emb = self.encode_image(images, normalize=True)
        text_emb = self.encode_text(class_texts, normalize=True)
        
        # Ensure 2D arrays
        if image_emb.ndim == 1:
            image_emb = image_emb[np.newaxis, :]
        
        # Compute similarity
        logits = self.compute_similarity(text_emb, image_emb, apply_temperature=True)
        logits = logits.T  # (n_images, n_classes)
        
        # Get predicted classes
        predicted_classes = np.argmax(logits, axis=1)
        
        if return_probabilities:
            # Apply softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return predicted_classes, probabilities
        
        return predicted_classes
    
    def __repr__(self) -> str:
        return (
            f"CLIPEngine("
            f"model_path={self.model_path}, "
            f"device={self.device}, "
            f"projection_dim={self.config.projection_dim})"
        )

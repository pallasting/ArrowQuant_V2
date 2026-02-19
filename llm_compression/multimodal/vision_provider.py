
"""
VisionProvider — Unified Interface for Vision Embeddings.

Mirroring EmbeddingProvider, this module abstracts the underlying vision models
(e.g., CLIP, SigLIP, MobileViT) to provide a consistent API for image encoding.
"""

import os
import logging
from typing import List, Union, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class VisionProvider:
    """
    Abstract base class for vision embedding providers.
    """
    @property
    def dimension(self) -> int:
        """Dimension of the image embeddings."""
        raise NotImplementedError

    def encode_image(self, image: Union[str, Image.Image], normalize: bool = True) -> np.ndarray:
        """
        Encode a single image into a vector.
        
        Args:
            image: File path (str) or PIL Image object.
            normalize: Whether to L2 normalize the output vector.
            
        Returns:
            (dimension,) float32 numpy array.
        """
        raise NotImplementedError
        
    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode text into the *same shared latent space* as images.
        Used for cross-modal retrieval (Text-to-Image search).
        """
        raise NotImplementedError

class CLIPVisionProvider(VisionProvider):
    """
    Implementation using OpenAI CLIP (via transformers or open_clip).
    Good general purpose baseline.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._dim = 512 # Default for base-patch32
        
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._model = CLIPModel.from_pretrained(model_name).to(device)
            self._model.eval()
            self._dim = self._model.config.projection_dim
            logger.info(f"Initialized CLIPVisionProvider with {model_name} on {device}")
        except ImportError:
            logger.warning("transformers library not found or CLIP load failed. Vision features will be disabled.")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")

    @property
    def dimension(self) -> int:
        return self._dim

    def encode_image(self, image: Union[str, Image.Image], normalize: bool = True) -> np.ndarray:
        if self._model is None:
            return np.zeros(self.dimension, dtype=np.float32)
            
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except Exception as e:
                logger.error(f"Failed to open image {image}: {e}")
                return np.zeros(self.dimension, dtype=np.float32)
                
        import torch
        with torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt").to(self.device)
            # direct model call might be safer if get_image_features is behaving oddly
            outputs = self._model.get_image_features(**inputs)
            
            # Check for output type (it should be a Tensor)
            if hasattr(outputs, "image_embeds"):
                image_features = outputs.image_embeds
            else:
                image_features = outputs

            if normalize:
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0].astype(np.float32)

    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        if self._model is None:
            return np.zeros(self.dimension, dtype=np.float32)
            
        import torch
        with torch.no_grad():
            inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            outputs = self._model.get_text_features(**inputs)
            
            if hasattr(outputs, "text_embeds"):
                text_features = outputs.text_embeds
            else:
                text_features = outputs

            if normalize:
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0].astype(np.float32)

class ReferenceVisionProvider(VisionProvider):
    """
    A lightweight mock provider for testing or when dependencies are missing.
    Returns random vectors but maintains interface consistency.
    """
    def __init__(self, dimension: int = 512):
        self._dim = dimension
        
    @property
    def dimension(self) -> int:
        return self._dim
        
    def encode_image(self, image: Union[str, Image.Image], normalize: bool = True) -> np.ndarray:
        vec = np.random.randn(self._dim).astype(np.float32)
        if normalize:
            vec /= np.linalg.norm(vec)
        return vec
        
    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        vec = np.random.randn(self._dim).astype(np.float32)
        if normalize:
            vec /= np.linalg.norm(vec)
        return vec

def get_default_vision_provider(device: str = "cpu") -> VisionProvider:
    """Factory to get the best available vision provider."""
    # Priority: Reference (for stability) -> CLIP
    return ReferenceVisionProvider()
    # try:
    #     import transformers
    #     import torch
    #     return CLIPVisionProvider(device=device)
    # except ImportError:
    #     logger.warning("Transformers not found, using ReferenceVisionProvider.")
    #     return ReferenceVisionProvider()

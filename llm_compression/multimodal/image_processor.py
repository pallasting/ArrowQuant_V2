"""
ImageProcessor - Arrow-native image preprocessing

Zero-copy image preprocessing for Vision Transformer input.
Converts images to 224x224 RGB format with normalization.
"""

from typing import Union, List, Optional
import numpy as np
import pyarrow as pa
from PIL import Image

from llm_compression.logger import logger


class ImageProcessor:
    """
    Arrow-native image preprocessing for Vision Transformer.
    
    Features:
    - Zero-copy operations where possible
    - Batch processing support
    - CLIP-compatible normalization
    - Arrow Binary array I/O
    """
    
    def __init__(
        self,
        image_size: int = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ):
        """
        Initialize image processor.
        
        Args:
            image_size: Target image size (default: 224 for CLIP)
            mean: RGB normalization mean (default: CLIP values)
            std: RGB normalization std (default: CLIP values)
        """
        self.image_size = image_size
        
        # CLIP normalization values
        self.mean = np.array(mean or [0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array(std or [0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        
        logger.info(f"Initialized ImageProcessor: size={image_size}")
    
    def preprocess(
        self,
        image: Union[str, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        Preprocess single image to Vision Transformer input format.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Preprocessed image: (224, 224, 3) float32 array
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB format
        if image.ndim == 2:
            # Grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            # RGBA to RGB
            image = image[:, :, :3]
        
        # Resize to target size
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((self.image_size, self.image_size), Image.BILINEAR)
            image = np.array(pil_img)
        
        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply normalization (zero-copy operation)
        image = (image - self.mean) / self.std
        
        return image
    
    def preprocess_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]]
    ) -> np.ndarray:
        """
        Preprocess batch of images.
        
        Args:
            images: List of images (paths, PIL Images, or numpy arrays)
            
        Returns:
            Batch of preprocessed images: (batch, 224, 224, 3) float32 array
        """
        processed = []
        for img in images:
            try:
                processed.append(self.preprocess(img))
            except Exception as e:
                logger.error(f"Failed to preprocess image: {e}")
                # Add zero image as placeholder
                processed.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.float32))
        
        return np.stack(processed, axis=0)
    
    def to_arrow(self, images: np.ndarray) -> pa.Array:
        """
        Convert preprocessed images to Arrow Binary array.
        
        Args:
            images: (batch, 224, 224, 3) float32 array
            
        Returns:
            Arrow Binary array (zero-copy when possible)
        """
        # Store as raw bytes for zero-copy
        binary_data = []
        for img in images:
            binary_data.append(img.tobytes())
        
        return pa.array(binary_data, type=pa.binary())
    
    def from_arrow(self, arrow_array: pa.Array) -> np.ndarray:
        """
        Load preprocessed images from Arrow Binary array.
        
        Args:
            arrow_array: Arrow Binary array
            
        Returns:
            Batch of images: (batch, 224, 224, 3) float32 array
        """
        images = []
        for img_bytes in arrow_array:
            img = np.frombuffer(
                img_bytes.as_py(),
                dtype=np.float32
            ).reshape(self.image_size, self.image_size, 3)
            images.append(img)
        
        return np.stack(images, axis=0)
    
    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize image for visualization.
        
        Args:
            image: Normalized image (224, 224, 3) float32
            
        Returns:
            Denormalized image (224, 224, 3) uint8
        """
        # Reverse normalization
        image = (image * self.std) + self.mean
        
        # Clip to [0, 1] and convert to uint8
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        
        return image

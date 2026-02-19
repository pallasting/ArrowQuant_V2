
"""
ImageManager — Vision Memory Storage & Retrieval.

Handles the lifecycle of visual memories: ingestion, thumbnail generation, vector storage, and cross-modal search.
"""

import os
import io
import time
import base64
import json
import logging
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from PIL import Image

# Import vision provider factory
from llm_compression.multimodal.vision_provider import VisionProvider, get_default_vision_provider

logger = logging.getLogger(__name__)

@dataclass
class VisualMemory:
    """A single visual memory unit."""
    id: str
    original_path: str
    embedding: np.ndarray # float32 vector
    caption: str = ""
    concepts: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    thumbnail_b64: str = "" # Small base64 thumbnail for quick preview

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "original_path": self.original_path,
            "caption": self.caption,
            "concepts": self.concepts,
            "timestamp": self.timestamp,
            "thumbnail_b64": self.thumbnail_b64
            # embedding is stored separately or reconstructing from path/provider
        }

class ImageManager:
    """
    Manages a collection of Visual Memories.
    """
    def __init__(self, storage_path: Union[str, Path], device: str = "cpu"):
        self.storage_path = Path(storage_path)
        self.images_path = self.storage_path / "images"
        self.images_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.images_path / "metadata.json"
        self.embeddings_file = self.images_path / "embeddings.npy"
        
        self.provider: Optional[VisionProvider] = None
        self.device = device
        
        self.memories: Dict[str, VisualMemory] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.ids: List[str] = []
        
        self._load()

    def _get_provider(self):
        if self.provider is None:
            self.provider = get_default_vision_provider(self.device)
        return self.provider

    def _load(self):
        """Load metadata and embeddings from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        # Reconstruct without embedding initially
                        vm = VisualMemory(
                            id=item["id"],
                            original_path=item["original_path"],
                            embedding=np.array([]), 
                            caption=item.get("caption", ""),
                            concepts=item.get("concepts", []),
                            timestamp=item.get("timestamp", 0.0),
                            thumbnail_b64=item.get("thumbnail_b64", "")
                        )
                        self.memories[vm.id] = vm
                        self.ids.append(vm.id)
                logger.info(f"Loaded {len(self.memories)} visual memories.")
            except Exception as e:
                logger.error(f"Failed to load image metadata: {e}")

        if self.embeddings_file.exists():
            try:
                self.embeddings = np.load(self.embeddings_file)
                # Link embeddings to memories
                if len(self.ids) == len(self.embeddings):
                    for i, mid in enumerate(self.ids):
                        self.memories[mid].embedding = self.embeddings[i]
                else:
                    logger.warning("Embedding count mismatch with metadata. Rebuilding index recommended.")
            except Exception as e:
                logger.error(f"Failed to load image embeddings: {e}")

    def save(self):
        """Persist metadata and embeddings."""
        try:
            # Save metadata
            data = [m.to_dict() for m in self.memories.values()]
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Save embeddings
            if self.embeddings is not None and len(self.embeddings) > 0:
                np.save(self.embeddings_file, self.embeddings)
                
            logger.info("Saved visual memory state.")
        except Exception as e:
            logger.error(f"Failed to save visual memory: {e}")

    def add_image(self, image_path: str, caption: str = "") -> Optional[str]:
        """
        Ingest a new image.
        
        Args:
            image_path: Path to the image file.
            caption: Optional text description.
            
        Returns:
            Image ID if successful, None otherwise.
        """
        path = Path(image_path)
        if not path.exists():
            logger.error(f"Image not found: {image_path}")
            return None
            
        try:
            pil_img = Image.open(path).convert("RGB")
            
            # 1. Generate Embedding
            provider = self._get_provider()
            embedding = provider.encode_image(pil_img)
            
            # 2. Generate Thumbnail (max 128x128)
            thumb = pil_img.copy()
            thumb.thumbnail((128, 128))
            buffered = io.BytesIO()
            thumb.save(buffered, format="JPEG", quality=70)
            thumb_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # 3. Create Memory
            import uuid
            img_id = str(uuid.uuid4())
            
            vm = VisualMemory(
                id=img_id,
                original_path=str(path.absolute()),
                embedding=embedding,
                caption=caption,
                timestamp=time.time(),
                thumbnail_b64=thumb_b64
            )
            
            # 4. Update Index
            self.memories[img_id] = vm
            self.ids.append(img_id)
            
            if self.embeddings is None:
                self.embeddings = np.expand_dims(embedding, axis=0)
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
                
            self.save()
            return img_id
            
        except Exception as e:
            logger.error(f"Failed to ingest image {image_path}: {e}")
            return None

    def search(self, query: Union[str, Image.Image], top_k: int = 5) -> List[Tuple[VisualMemory, float]]:
        """
        Search for visually similar images or text-to-image search.
        
        Args:
            query: Text string or PIL Image.
            top_k: Number of results.
            
        Returns:
            List of (VisualMemory, score).
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
            
        provider = self._get_provider()
        
        # Encode Query
        if isinstance(query, str):
            query_vec = provider.encode_text(query)
        else:
            query_vec = provider.encode_image(query)
            
        # Compute Cosine Similarity
        # (Assuming embeddings are normalized)
        scores = np.dot(self.embeddings, query_vec)
        
        # Get Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            mid = self.ids[idx]
            results.append((self.memories[mid], float(scores[idx])))
            
        return results

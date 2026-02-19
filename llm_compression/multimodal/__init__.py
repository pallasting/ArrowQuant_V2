"""
Multimodal Encoder System

Extends ArrowEngine with vision and audio encoding capabilities.
Implements the perception layer of the AI-OS complete loop architecture.
"""

from llm_compression.multimodal.image_processor import ImageProcessor
from llm_compression.multimodal.audio_processor import AudioProcessor, MelSpectrogramProcessor
from llm_compression.multimodal.vision_encoder import VisionEncoder, VisionConfig
from llm_compression.multimodal.audio_encoder import AudioEncoder, AudioConfig
from llm_compression.multimodal.clip_engine import CLIPEngine
from llm_compression.multimodal.multimodal_provider import (
    MultimodalEmbeddingProvider,
    get_multimodal_provider,
)
from llm_compression.multimodal.multimodal_storage import (
    MultimodalStorage,
    create_vision_embedding_schema,
    create_audio_embedding_schema,
    create_clip_embedding_schema,
)

__all__ = [
    "ImageProcessor",
    "AudioProcessor",
    "MelSpectrogramProcessor",
    "VisionEncoder",
    "VisionConfig",
    "AudioEncoder",
    "AudioConfig",
    "CLIPEngine",
    "MultimodalEmbeddingProvider",
    "get_multimodal_provider",
    "MultimodalStorage",
    "create_vision_embedding_schema",
    "create_audio_embedding_schema",
    "create_clip_embedding_schema",
]

"""
Arrow-optimized embedding tools package.

This package provides high-performance embedding generation using:
- Arrow/Parquet model storage (zero-copy)
- Rust tokenizers (10-20x faster)
- Optimized batch inference
"""

from llm_compression.tools.model_converter import (
    ModelConverter,
    ConversionConfig,
    ConversionResult,
)
from llm_compression.tools.embedding_tool import (
    EmbeddingTool,
    EmbeddingConfig,
    EmbeddingToolResult,
)

__version__ = "0.1.0"

__all__ = [
    "ModelConverter",
    "ConversionConfig",
    "ConversionResult",
    "EmbeddingTool",
    "EmbeddingConfig",
    "EmbeddingToolResult",
]

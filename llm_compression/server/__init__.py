"""
FastAPI server for ArrowEngine embedding service.

This package provides HTTP endpoints for text embedding generation,
similarity computation, and model information retrieval.
"""

from llm_compression.server.app import app, init_app, __version__
from llm_compression.server.models import (
    EmbedRequest,
    EmbedResponse,
    SimilarityRequest,
    SimilarityResponse,
    HealthResponse,
    InfoResponse,
)

__all__ = [
    "app",
    "init_app",
    "__version__",
    "EmbedRequest",
    "EmbedResponse",
    "SimilarityRequest",
    "SimilarityResponse",
    "HealthResponse",
    "InfoResponse",
]

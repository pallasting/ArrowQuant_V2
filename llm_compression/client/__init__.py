"""
Client library for ArrowEngine Embedding Service

Provides HTTP client and result objects for interacting with the service.
"""

from llm_compression.client.client import (
    ArrowEngineClient,
    ArrowEngineClientError,
    EmbedResult,
    SimilarityResult,
    HealthResult,
    InfoResult,
)

__all__ = [
    "ArrowEngineClient",
    "ArrowEngineClientError",
    "EmbedResult",
    "SimilarityResult",
    "HealthResult",
    "InfoResult",
]

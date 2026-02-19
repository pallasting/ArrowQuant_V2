"""
Pydantic models for FastAPI request/response schemas.

Defines data models for all API endpoints:
- /embed - Text embedding
- /similarity - Similarity computation
- /health - Health check
- /info - Model information
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class EmbedRequest(BaseModel):
    """Request model for /embed endpoint"""
    
    texts: List[str] = Field(
        ...,
        min_length=1,
        description="List of texts to embed (at least one required)"
    )
    normalize: bool = Field(
        default=False,
        description="Whether to normalize embeddings to unit vectors"
    )
    
    @field_validator('texts')
    @classmethod
    def validate_texts_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure texts list is not empty"""
        if not v:
            raise ValueError("texts list cannot be empty")
        return v


class EmbedResponse(BaseModel):
    """Response model for /embed endpoint"""
    
    embeddings: List[List[float]] = Field(
        ...,
        description="List of embedding vectors"
    )
    dimension: int = Field(
        ...,
        description="Embedding vector dimension"
    )
    count: int = Field(
        ...,
        description="Number of embeddings returned"
    )


class SimilarityRequest(BaseModel):
    """Request model for /similarity endpoint
    
    Supports two modes:
    1. Single pair: text1 and text2
    2. Multiple pairs: texts1 and texts2 (same length)
    """
    
    # Single pair mode
    text1: Optional[str] = Field(
        default=None,
        description="First text (for single pair mode)"
    )
    text2: Optional[str] = Field(
        default=None,
        description="Second text (for single pair mode)"
    )
    
    # Multiple pairs mode
    texts1: Optional[List[str]] = Field(
        default=None,
        description="First texts (for multiple pairs mode)"
    )
    texts2: Optional[List[str]] = Field(
        default=None,
        description="Second texts (for multiple pairs mode)"
    )
    
    @field_validator('texts1', 'texts2')
    @classmethod
    def validate_texts_length(cls, v: Optional[List[str]], info) -> Optional[List[str]]:
        """Ensure texts lists have same length if both provided"""
        if v is not None and len(v) == 0:
            raise ValueError("texts list cannot be empty")
        return v
    
    def model_post_init(self, __context) -> None:
        """Validate that either single pair or multiple pairs is provided"""
        single_mode = self.text1 is not None and self.text2 is not None
        multi_mode = self.texts1 is not None and self.texts2 is not None
        
        if not single_mode and not multi_mode:
            raise ValueError(
                "Must provide either (text1, text2) or (texts1, texts2)"
            )
        
        if single_mode and multi_mode:
            raise ValueError(
                "Cannot provide both single pair and multiple pairs"
            )
        
        if multi_mode:
            # Type guard: both are not None in multi_mode
            assert self.texts1 is not None and self.texts2 is not None
            if len(self.texts1) != len(self.texts2):
                raise ValueError(
                    f"texts1 and texts2 must have same length "
                    f"(got {len(self.texts1)} and {len(self.texts2)})"
                )


class SimilarityResponse(BaseModel):
    """Response model for /similarity endpoint
    
    Returns either 'similarity' (single value) or 'similarities' (list)
    """
    
    similarity: Optional[float] = Field(
        default=None,
        description="Similarity score for single pair (-1 to 1)"
    )
    similarities: Optional[List[float]] = Field(
        default=None,
        description="Similarity scores for multiple pairs"
    )


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    
    status: str = Field(
        ...,
        description="Health status (healthy/unhealthy)"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready"
    )
    device: Optional[str] = Field(
        default=None,
        description="Device the model is running on (cpu/cuda)"
    )


class InfoResponse(BaseModel):
    """Response model for /info endpoint"""
    
    model_name: str = Field(
        ...,
        description="Model identifier or path"
    )
    embedding_dimension: int = Field(
        ...,
        description="Dimension of embedding vectors"
    )
    max_seq_length: int = Field(
        ...,
        description="Maximum sequence length the model supports"
    )
    version: str = Field(
        ...,
        description="Server version"
    )
    device: Optional[str] = Field(
        default=None,
        description="Device the model is running on"
    )

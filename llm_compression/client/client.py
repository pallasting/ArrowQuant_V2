"""
HTTP Client for ArrowEngine Embedding Service

Provides a convenient Python interface to the ArrowEngine FastAPI service.

Usage:
    from llm_compression.client import ArrowEngineClient
    
    client = ArrowEngineClient(base_url="http://localhost:8000")
    
    # Generate embeddings
    embeddings = client.embed(["Hello, world!"])
    
    # Compute similarity
    similarity = client.similarity("AI", "Machine Learning")
    
    # Check health
    health = client.health()
"""

import requests
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class EmbedResult:
    """Result from embed operation"""
    embeddings: List[List[float]]
    dimension: int
    count: int


@dataclass
class SimilarityResult:
    """Result from similarity operation"""
    similarity: Optional[float] = None
    similarities: Optional[List[float]] = None


@dataclass
class HealthResult:
    """Result from health check"""
    status: str
    model_loaded: bool
    device: Optional[str] = None


@dataclass
class InfoResult:
    """Result from info query"""
    model_name: str
    embedding_dimension: int
    max_seq_length: int
    version: str
    device: Optional[str] = None


class ArrowEngineClientError(Exception):
    """Base exception for client errors"""
    pass


class ArrowEngineClient:
    """HTTP client for ArrowEngine embedding service"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize ArrowEngine client
        
        Args:
            base_url: Base URL of the ArrowEngine service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
    
    def embed(
        self,
        texts: List[str],
        normalize: bool = False
    ) -> EmbedResult:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings to unit vectors
            
        Returns:
            EmbedResult with embeddings and metadata
            
        Raises:
            ArrowEngineClientError: If request fails
        """
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        url = f"{self.base_url}/embed"
        payload = {"texts": texts, "normalize": normalize}
        
        try:
            response = self._post(url, payload)
            data = response.json()
            
            return EmbedResult(
                embeddings=data["embeddings"],
                dimension=data["dimension"],
                count=data["count"]
            )
        except Exception as e:
            raise ArrowEngineClientError(f"Embed request failed: {e}") from e
    
    def similarity(
        self,
        text1: Optional[str] = None,
        text2: Optional[str] = None,
        texts1: Optional[List[str]] = None,
        texts2: Optional[List[str]] = None
    ) -> SimilarityResult:
        """
        Compute similarity between text pairs
        
        Supports two modes:
        1. Single pair: text1 and text2
        2. Multiple pairs: texts1 and texts2
        
        Args:
            text1: First text (single pair mode)
            text2: Second text (single pair mode)
            texts1: First texts (multiple pairs mode)
            texts2: Second texts (multiple pairs mode)
            
        Returns:
            SimilarityResult with similarity score(s)
            
        Raises:
            ArrowEngineClientError: If request fails
        """
        url = f"{self.base_url}/similarity"
        
        payload: Dict[str, Any] = {}
        if text1 is not None and text2 is not None:
            payload = {"text1": text1, "text2": text2}
        elif texts1 is not None and texts2 is not None:
            payload = {"texts1": texts1, "texts2": texts2}
        else:
            raise ValueError(
                "Must provide either (text1, text2) or (texts1, texts2)"
            )
        
        try:
            response = self._post(url, payload)
            data = response.json()
            
            return SimilarityResult(
                similarity=data.get("similarity"),
                similarities=data.get("similarities")
            )
        except Exception as e:
            raise ArrowEngineClientError(f"Similarity request failed: {e}") from e
    
    def health(self) -> HealthResult:
        """
        Check service health
        
        Returns:
            HealthResult with service status
            
        Raises:
            ArrowEngineClientError: If request fails
        """
        url = f"{self.base_url}/health"
        
        try:
            response = self._get(url)
            data = response.json()
            
            return HealthResult(
                status=data["status"],
                model_loaded=data["model_loaded"],
                device=data.get("device")
            )
        except Exception as e:
            raise ArrowEngineClientError(f"Health check failed: {e}") from e
    
    def info(self) -> InfoResult:
        """
        Get model information
        
        Returns:
            InfoResult with model metadata
            
        Raises:
            ArrowEngineClientError: If request fails
        """
        url = f"{self.base_url}/info"
        
        try:
            response = self._get(url)
            data = response.json()
            
            return InfoResult(
                model_name=data["model_name"],
                embedding_dimension=data["embedding_dimension"],
                max_seq_length=data["max_seq_length"],
                version=data["version"],
                device=data.get("device")
            )
        except Exception as e:
            raise ArrowEngineClientError(f"Info request failed: {e}") from e
    
    def _get(self, url: str) -> requests.Response:
        """Execute GET request with retries"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                continue
        
        raise ArrowEngineClientError("Max retries exceeded")
    
    def _post(self, url: str, json_data: Dict[str, Any]) -> requests.Response:
        """Execute POST request with retries"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url,
                    json=json_data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                continue
        
        raise ArrowEngineClientError("Max retries exceeded")
    
    def close(self) -> None:
        """Close the client session"""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

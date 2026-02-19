"""
FastAPI Server Tests for ArrowEngine API

TDD approach: Write tests first, then implement API endpoints.

Test coverage:
- POST /embed - Text embedding endpoint
- POST /similarity - Text similarity endpoint
- GET /health - Health check endpoint
- GET /info - Model information endpoint
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


class TestEmbedEndpoint:
    """Test /embed endpoint"""
    
    def test_embed_single_text(self, client):
        """Test embedding single text"""
        response = client.post(
            "/embed",
            json={"texts": ["Hello, world!"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "embeddings" in data
        assert isinstance(data["embeddings"], list)
        assert len(data["embeddings"]) == 1
        assert isinstance(data["embeddings"][0], list)
        assert len(data["embeddings"][0]) > 0
    
    def test_embed_multiple_texts(self, client):
        """Test embedding multiple texts"""
        texts = [
            "Machine learning",
            "Deep learning",
            "Artificial intelligence"
        ]
        
        response = client.post(
            "/embed",
            json={"texts": texts}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "embeddings" in data
        assert len(data["embeddings"]) == len(texts)
    
    def test_embed_with_normalize(self, client):
        """Test embedding with normalization"""
        response = client.post(
            "/embed",
            json={
                "texts": ["Test text"],
                "normalize": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
    
    def test_embed_empty_text_fails(self, client):
        """Test that empty text list fails"""
        response = client.post(
            "/embed",
            json={"texts": []}
        )
        
        assert response.status_code == 422


class TestSimilarityEndpoint:
    """Test /similarity endpoint"""
    
    def test_similarity_single_pair(self, client):
        """Test similarity between two texts"""
        response = client.post(
            "/similarity",
            json={
                "text1": "Machine learning",
                "text2": "Deep learning"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "similarity" in data
        assert isinstance(data["similarity"], float)
        assert -1.0 <= data["similarity"] <= 1.0
    
    def test_similarity_multiple_pairs(self, client):
        """Test similarity for multiple text pairs"""
        response = client.post(
            "/similarity",
            json={
                "texts1": ["AI", "ML"],
                "texts2": ["Artificial Intelligence", "Machine Learning"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "similarities" in data
        assert len(data["similarities"]) == 2


class TestHealthEndpoint:
    """Test /health endpoint"""
    
    def test_health_check(self, client):
        """Test health check returns OK"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_includes_model_info(self, client):
        """Test health check includes model status"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


class TestInfoEndpoint:
    """Test /info endpoint"""
    
    def test_info_returns_model_details(self, client):
        """Test info endpoint returns model information"""
        response = client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_name" in data
        assert "embedding_dimension" in data
        assert "max_seq_length" in data
    
    def test_info_includes_server_version(self, client):
        """Test info includes server version"""
        response = client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "version" in data


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_endpoint_returns_404(self, client):
        """Test invalid endpoint returns 404"""
        response = client.get("/invalid")
        assert response.status_code == 404
    
    def test_invalid_method_returns_405(self, client):
        """Test invalid HTTP method returns 405"""
        response = client.get("/embed")
        assert response.status_code == 405


@pytest.fixture
def mock_engine():
    """Create mock ArrowEngine for testing"""
    engine = Mock()
    
    # Mock encode method to return fake embeddings
    def mock_encode(texts):
        # Return 384-dimensional embeddings (typical for MiniLM)
        return np.random.randn(len(texts), 384).astype(np.float32)
    
    engine.encode.side_effect = mock_encode
    
    # Mock similarity method
    def mock_similarity(text1, text2):
        return np.array(0.85)  # High similarity
    
    engine.similarity.side_effect = mock_similarity
    
    # Mock metadata methods
    engine.get_embedding_dimension.return_value = 384
    engine.get_max_seq_length.return_value = 512
    engine.device = "cpu"
    
    return engine


@pytest.fixture
def client(mock_engine):
    """Create test client with mocked ArrowEngine"""
    from llm_compression.server.app import app
    
    # Patch the get_engine function to return our mock
    with patch('llm_compression.server.app.get_engine', return_value=mock_engine):
        yield TestClient(app)


@pytest.fixture(scope="session")
def test_model_path():
    """Path to test model (mock for testing)"""
    return "./models/test-model"

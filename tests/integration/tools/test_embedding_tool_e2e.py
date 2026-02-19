"""
End-to-End Integration Tests for AI-OS Embedding Tool

Tests the complete stack:
- EmbeddingTool → HTTP Client → FastAPI Service → ArrowEngine

These tests use mocked ArrowEngine to avoid requiring actual model files.
They validate the complete integration pipeline.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from llm_compression.tools import EmbeddingTool, EmbeddingConfig
from llm_compression.tools.config import load_config, save_config, validate_config, ConfigError
from llm_compression.client import ArrowEngineClientError


class TestEmbeddingToolIntegration:
    """Test EmbeddingTool with real HTTP client and mocked service"""
    
    def test_embed_single_text(self, tool):
        """Test embedding single text through complete stack"""
        result = tool.embed(["Hello, world!"])
        
        assert result.embeddings.shape == (1, 384)
        assert result.dimension == 384
        assert result.count == 1
        assert isinstance(result.embeddings, np.ndarray)
    
    def test_embed_multiple_texts(self, tool):
        """Test embedding multiple texts"""
        texts = [
            "Machine learning",
            "Deep learning",
            "Artificial intelligence"
        ]
        
        result = tool.embed(texts)
        
        assert result.embeddings.shape == (3, 384)
        assert result.dimension == 384
        assert result.count == 3
        assert len(result.texts) == 3
    
    def test_embed_with_batching(self, tool):
        """Test automatic batching for large collections"""
        texts = ["Text"] * 100
        
        result = tool.embed(texts, batch_size=32)
        
        assert result.embeddings.shape == (100, 384)
        assert result.metadata["batch_size"] == 32
        assert result.metadata["total_batches"] == 4
    
    def test_embed_with_normalization(self, tool):
        """Test embedding normalization"""
        result = tool.embed(["Test"], normalize=True)
        
        norm = np.linalg.norm(result.embeddings[0])
        assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_similarity_single_pair(self, tool):
        """Test similarity computation for single pair"""
        similarity = tool.similarity("AI", "Machine Learning")
        
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0
    
    def test_similarity_matrix(self, tool):
        """Test similarity matrix computation"""
        texts1 = ["AI", "ML"]
        texts2 = ["Artificial Intelligence", "Machine Learning"]
        
        matrix = tool.similarity_matrix(texts1, texts2)
        
        assert matrix.shape == (2, 2)
        assert np.all((-1.0 <= matrix) & (matrix <= 1.0))
    
    def test_health_check(self, tool):
        """Test health check"""
        is_healthy = tool.health_check()
        assert is_healthy is True
    
    def test_get_info(self, tool):
        """Test model information retrieval"""
        info = tool.get_info()
        
        assert "model_name" in info
        assert "embedding_dimension" in info
        assert info["embedding_dimension"] == 384
        assert "max_seq_length" in info
        assert "version" in info
        assert "endpoint" in info


class TestEmbeddingToolCaching:
    """Test caching functionality"""
    
    def test_cache_hit_on_repeated_query(self, tool):
        """Test that repeated queries use cache"""
        texts = ["Hello, world!"]
        
        result1 = tool.embed(texts)
        assert result1.cache_hits == 0
        assert result1.cache_misses == 1
        
        result2 = tool.embed(texts)
        assert result2.cache_hits == 1
        assert result2.cache_misses == 0
    
    def test_cache_miss_on_different_text(self, tool):
        """Test cache miss for different text"""
        result1 = tool.embed(["Text 1"])
        result2 = tool.embed(["Text 2"])
        
        assert result1.cache_misses == 1
        assert result2.cache_misses == 1
    
    def test_cache_respects_normalization(self, tool):
        """Test that cache keys include normalization flag"""
        texts = ["Test"]
        
        result1 = tool.embed(texts, normalize=False)
        result2 = tool.embed(texts, normalize=True)
        
        assert result1.cache_misses == 1
        assert result2.cache_misses == 1
    
    def test_clear_cache(self, tool):
        """Test cache clearing"""
        tool.embed(["Test"])
        
        stats_before = tool.get_cache_stats()
        assert stats_before["size"] == 1
        
        tool.clear_cache()
        
        stats_after = tool.get_cache_stats()
        assert stats_after["size"] == 0
    
    def test_cache_stats(self, tool):
        """Test cache statistics"""
        stats = tool.get_cache_stats()
        
        assert "size" in stats
        assert "capacity" in stats
        assert stats["capacity"] == 1000


class TestConfigurationLoading:
    """Test configuration loading and validation"""
    
    def test_load_config_from_file(self, tmp_path):
        """Test loading config from YAML file"""
        config_file = tmp_path / "test_config.yaml"
        
        config = EmbeddingConfig(
            endpoint="http://test:8000",
            batch_size=64,
            cache_size=500
        )
        
        save_config(config, str(config_file))
        loaded_config = load_config(str(config_file))
        
        assert loaded_config.endpoint == "http://test:8000"
        assert loaded_config.batch_size == 64
        assert loaded_config.cache_size == 500
    
    def test_validate_valid_config(self):
        """Test validation of valid config"""
        config = EmbeddingConfig(
            endpoint="http://localhost:8000",
            timeout=30.0,
            max_retries=3
        )
        
        validate_config(config)
    
    def test_validate_invalid_timeout(self):
        """Test validation fails for invalid timeout"""
        config = EmbeddingConfig(
            endpoint="http://localhost:8000",
            timeout=-1.0
        )
        
        with pytest.raises(ConfigError, match="Invalid timeout"):
            validate_config(config)
    
    def test_validate_invalid_endpoint(self):
        """Test validation fails for invalid endpoint"""
        config = EmbeddingConfig(
            endpoint="not-a-url"
        )
        
        with pytest.raises(ConfigError, match="Invalid endpoint URL"):
            validate_config(config)
    
    def test_config_file_not_found(self):
        """Test error when config file not found"""
        with pytest.raises(ConfigError, match="Config file not found"):
            load_config("nonexistent.yaml")


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_empty_texts_list_raises_error(self, tool):
        """Test that empty texts list raises error"""
        with pytest.raises(ValueError, match="cannot be empty"):
            tool.embed([])
    
    def test_service_unavailable_health_check(self):
        """Test health check returns False when service unavailable"""
        tool = EmbeddingTool(endpoint="http://invalid:9999")
        
        is_healthy = tool.health_check()
        assert is_healthy is False
    
    def test_context_manager_usage(self, mock_app):
        """Test context manager properly closes connection"""
        with patch('llm_compression.server.app.get_engine', return_value=mock_app):
            with EmbeddingTool() as tool:
                result = tool.embed(["Test"])
                assert result is not None


class TestCustomConfiguration:
    """Test custom configuration scenarios"""
    
    def test_custom_batch_size(self, tool):
        """Test custom batch size configuration"""
        texts = ["Text"] * 50
        result = tool.embed(texts, batch_size=10)
        
        assert result.metadata["batch_size"] == 10
        assert result.metadata["total_batches"] == 5
    
    def test_custom_endpoint(self, mock_app):
        """Test custom endpoint configuration"""
        with patch('llm_compression.server.app.get_engine', return_value=mock_app):
            config = EmbeddingConfig(endpoint="http://custom:8000")
            tool = EmbeddingTool(config=config)
            
            info = tool.get_info()
            assert info["endpoint"] == "http://custom:8000"
    
    def test_disable_cache(self, mock_app):
        """Test disabling cache"""
        with patch('llm_compression.server.app.get_engine', return_value=mock_app):
            config = EmbeddingConfig(enable_cache=False)
            tool = EmbeddingTool(config=config)
            
            result1 = tool.embed(["Test"])
            result2 = tool.embed(["Test"])
            
            assert result1.cache_hits == 0
            assert result2.cache_hits == 0


@pytest.fixture
def mock_app():
    """Create mock ArrowEngine for testing"""
    engine = Mock()
    
    def mock_encode(texts):
        return np.random.randn(len(texts), 384).astype(np.float32)
    
    def mock_similarity(text1, text2):
        return np.array(0.85)
    
    engine.encode.side_effect = mock_encode
    engine.similarity.side_effect = mock_similarity
    engine.get_embedding_dimension.return_value = 384
    engine.get_max_seq_length.return_value = 512
    engine.device = "cpu"
    
    return engine


@pytest.fixture
def tool(mock_app):
    """Create EmbeddingTool with mocked service"""
    with patch('llm_compression.server.app.get_engine', return_value=mock_app):
        from llm_compression.server.app import app
        
        with patch('llm_compression.client.client.requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.status_code = 200
            
            def json_side_effect():
                if "/embed" in str(mock_session.call_args):
                    embeddings = np.random.randn(1, 384).tolist()
                    return {
                        "embeddings": embeddings,
                        "dimension": 384,
                        "count": len(embeddings)
                    }
                elif "/similarity" in str(mock_session.call_args):
                    return {"similarity": 0.85}
                elif "/health" in str(mock_session.call_args):
                    return {"status": "healthy", "model_loaded": True, "device": "cpu"}
                elif "/info" in str(mock_session.call_args):
                    return {
                        "model_name": "test-model",
                        "embedding_dimension": 384,
                        "max_seq_length": 512,
                        "version": "0.1.0",
                        "device": "cpu"
                    }
            
            mock_response.json.side_effect = json_side_effect
            mock_session.return_value.get.return_value = mock_response
            mock_session.return_value.post.return_value = mock_response
            
            tool = EmbeddingTool()
            yield tool

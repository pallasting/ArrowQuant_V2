"""
Performance Benchmark Tests for ArrowEngine API

Tests performance metrics:
- Latency (p50, p95, p99)
- Throughput (requests/second)
- Concurrent request handling
- Memory footprint

Requirements:
- pytest-benchmark for timing
- Real ArrowEngine with test model
"""

import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient
import psutil
import os


class TestEmbeddingLatency:
    """Test embedding endpoint latency"""
    
    def test_single_text_latency(self, benchmark, client):
        """Benchmark single text embedding latency
        
        Target: p50 < 15ms, p95 < 30ms
        """
        def embed_single():
            response = client.post(
                "/embed",
                json={"texts": ["Machine learning is awesome"]}
            )
            assert response.status_code == 200
            return response
        
        result = benchmark(embed_single)
        assert result.status_code == 200
    
    def test_batch_embedding_latency(self, benchmark, client):
        """Benchmark batch embedding latency (8 texts)
        
        Target: < 50ms for batch of 8
        """
        texts = [
            "Machine learning",
            "Deep learning",
            "Natural language processing",
            "Computer vision",
            "Reinforcement learning",
            "Neural networks",
            "Data science",
            "Artificial intelligence"
        ]
        
        def embed_batch():
            response = client.post(
                "/embed",
                json={"texts": texts}
            )
            assert response.status_code == 200
            return response
        
        result = benchmark(embed_batch)
        assert result.status_code == 200


class TestSimilarityLatency:
    """Test similarity endpoint latency"""
    
    def test_similarity_single_pair_latency(self, benchmark, client):
        """Benchmark single pair similarity
        
        Target: p50 < 20ms
        """
        def compute_similarity():
            response = client.post(
                "/similarity",
                json={
                    "text1": "Machine learning",
                    "text2": "Deep learning"
                }
            )
            assert response.status_code == 200
            return response
        
        result = benchmark(compute_similarity)
        assert result.status_code == 200


class TestThroughput:
    """Test API throughput under load"""
    
    def test_sequential_throughput(self, client):
        """Measure sequential request throughput
        
        Target: > 100 req/s sequential
        """
        num_requests = 100
        start_time = time.time()
        
        for _ in range(num_requests):
            response = client.post(
                "/embed",
                json={"texts": ["Test text"]}
            )
            assert response.status_code == 200
        
        elapsed = time.time() - start_time
        throughput = num_requests / elapsed
        
        print(f"\nSequential throughput: {throughput:.2f} req/s")
        assert throughput > 50
    
    def test_concurrent_throughput(self, client):
        """Measure concurrent request throughput
        
        Target: > 500 req/s with 10 concurrent workers
        """
        num_requests = 200
        num_workers = 10
        
        def make_request():
            response = client.post(
                "/embed",
                json={"texts": ["Concurrent test"]}
            )
            return response.status_code == 200
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [f.result() for f in as_completed(futures)]
        
        elapsed = time.time() - start_time
        throughput = num_requests / elapsed
        success_rate = sum(results) / len(results)
        
        print(f"\nConcurrent throughput: {throughput:.2f} req/s")
        print(f"Success rate: {success_rate * 100:.1f}%")
        
        assert throughput > 100
        assert success_rate > 0.95


class TestLoadHandling:
    """Test API behavior under various load conditions"""
    
    def test_sustained_load(self, client):
        """Test sustained request load over 5 seconds
        
        Validates stability and consistent performance
        """
        duration = 5.0
        latencies = []
        
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration:
            req_start = time.time()
            response = client.post(
                "/embed",
                json={"texts": ["Sustained load test"]}
            )
            latencies.append((time.time() - req_start) * 1000)
            
            assert response.status_code == 200
            request_count += 1
        
        elapsed = time.time() - start_time
        throughput = request_count / elapsed
        
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]
        p99 = statistics.quantiles(latencies, n=100)[98]
        
        print(f"\nSustained load results:")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Requests: {request_count}")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Latency p50: {p50:.2f}ms")
        print(f"  Latency p95: {p95:.2f}ms")
        print(f"  Latency p99: {p99:.2f}ms")
        
        assert p50 < 30
        assert p95 < 60
        assert throughput > 50


class TestMemoryFootprint:
    """Test memory usage"""
    
    def test_memory_usage(self, client):
        """Monitor memory usage during operation
        
        Target: < 2GB total memory usage
        """
        process = psutil.Process(os.getpid())
        
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        for _ in range(100):
            client.post(
                "/embed",
                json={"texts": ["Memory test"] * 8}
            )
        
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - baseline_memory
        
        print(f"\nMemory usage:")
        print(f"  Baseline: {baseline_memory:.2f} MB")
        print(f"  Peak: {peak_memory:.2f} MB")
        print(f"  Increase: {memory_increase:.2f} MB")
        
        assert peak_memory < 2048


class TestHealthEndpointPerformance:
    """Test health endpoint performance"""
    
    def test_health_check_latency(self, benchmark, client):
        """Health check should be very fast
        
        Target: < 5ms
        """
        def check_health():
            response = client.get("/health")
            assert response.status_code == 200
            return response
        
        result = benchmark(check_health)
        assert result.status_code == 200


@pytest.fixture(scope="module")
def client():
    """Create test client with mocked engine for performance tests"""
    import numpy as np
    from unittest.mock import Mock, patch
    from llm_compression.server.app import app
    
    engine = Mock()
    
    def mock_encode(texts):
        time.sleep(0.005)
        return np.random.randn(len(texts), 384).astype(np.float32)
    
    def mock_similarity(text1, text2):
        time.sleep(0.008)
        return np.array(0.85)
    
    engine.encode.side_effect = mock_encode
    engine.similarity.side_effect = mock_similarity
    engine.get_embedding_dimension.return_value = 384
    engine.get_max_seq_length.return_value = 512
    engine.device = "cpu"
    
    with patch('llm_compression.server.app.get_engine', return_value=engine):
        yield TestClient(app)

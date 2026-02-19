"""
Unit tests for WeightLoader module.

Tests cover:
- Memory-efficient weight loading (Property 11)
- Zero-copy memory mapping
- Lazy loading functionality
- Weight caching
- Parquet file reading
- Tensor conversion
- Error handling

**Property 11: Memory-Efficient Weight Loading**
For any model loaded via WeightLoader, the memory usage increase should be
approximately equal to the model size (no significant extra allocations), and
loading should complete in < 100ms for models < 200MB.
**Validates: Requirements 3.1, 3.2**
"""

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from llm_compression.inference.weight_loader import WeightLoader


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_parquet_file():
    """Create a temporary Parquet file with mock weights."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
        parquet_path = f.name
    
    # Create mock weight data
    layers = []
    
    # Embedding layers
    layers.append({
        'layer_name': 'embeddings.word_embeddings.weight',
        'shape': [1000, 64],
        'dtype': 'torch.float32',
        'data': np.random.randn(1000, 64).astype(np.float32).tobytes(),
        'num_params': 1000 * 64,
    })
    
    layers.append({
        'layer_name': 'embeddings.position_embeddings.weight',
        'shape': [512, 64],
        'dtype': 'torch.float32',
        'data': np.random.randn(512, 64).astype(np.float32).tobytes(),
        'num_params': 512 * 64,
    })
    
    # Attention layer
    layers.append({
        'layer_name': 'encoder.layer.0.attention.self.query.weight',
        'shape': [64, 64],
        'dtype': 'torch.float32',
        'data': np.random.randn(64, 64).astype(np.float32).tobytes(),
        'num_params': 64 * 64,
    })
    
    layers.append({
        'layer_name': 'encoder.layer.0.attention.self.query.bias',
        'shape': [64],
        'dtype': 'torch.float32',
        'data': np.random.randn(64).astype(np.float32).tobytes(),
        'num_params': 64,
    })
    
    # Create Arrow table
    table = pa.Table.from_pylist(layers)
    
    # Write to Parquet
    pq.write_table(table, parquet_path)
    
    yield parquet_path
    
    # Cleanup
    os.unlink(parquet_path)


@pytest.fixture
def small_parquet_file():
    """Create a small Parquet file for performance testing."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
        parquet_path = f.name
    
    # Create small model (< 10MB)
    layers = []
    
    # 10 layers with small dimensions
    for i in range(10):
        layers.append({
            'layer_name': f'layer.{i}.weight',
            'shape': [128, 128],
            'dtype': 'torch.float32',
            'data': np.random.randn(128, 128).astype(np.float32).tobytes(),
            'num_params': 128 * 128,
        })
    
    table = pa.Table.from_pylist(layers)
    pq.write_table(table, parquet_path)
    
    yield parquet_path
    
    os.unlink(parquet_path)


@pytest.fixture
def float16_parquet_file():
    """Create a Parquet file with float16 weights."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
        parquet_path = f.name
    
    layers = [{
        'layer_name': 'test.weight',
        'shape': [100, 64],
        'dtype': 'torch.float16',
        'data': np.random.randn(100, 64).astype(np.float16).tobytes(),
        'num_params': 100 * 64,
    }]
    
    table = pa.Table.from_pylist(layers)
    pq.write_table(table, parquet_path)
    
    yield parquet_path
    
    os.unlink(parquet_path)


# ============================================================================
# Test: Construction and Initialization
# ============================================================================


class TestWeightLoaderConstruction:
    """Test WeightLoader initialization."""
    
    def test_initialization_with_valid_path(self, temp_parquet_file):
        """Should initialize with valid Parquet file."""
        loader = WeightLoader(temp_parquet_file)
        
        assert loader.parquet_path == Path(temp_parquet_file)
        assert loader.use_memory_map is True
        assert loader.device == 'cpu'
        assert loader.cache_weights is True
    
    def test_initialization_with_custom_device(self, temp_parquet_file):
        """Should accept custom device parameter."""
        loader = WeightLoader(temp_parquet_file, device='cpu')
        assert loader.device == 'cpu'
    
    def test_initialization_with_memory_map_disabled(self, temp_parquet_file):
        """Should allow disabling memory mapping."""
        loader = WeightLoader(temp_parquet_file, use_memory_map=False)
        assert loader.use_memory_map is False
    
    def test_initialization_with_cache_disabled(self, temp_parquet_file):
        """Should allow disabling weight caching."""
        loader = WeightLoader(temp_parquet_file, cache_weights=False)
        assert loader.cache_weights is False
    
    def test_initialization_with_invalid_path(self):
        """Should raise FileNotFoundError for invalid path."""
        with pytest.raises(FileNotFoundError):
            WeightLoader('/nonexistent/path/weights.parquet')
    
    def test_repr_string(self, temp_parquet_file):
        """Should have informative string representation."""
        loader = WeightLoader(temp_parquet_file)
        repr_str = repr(loader)
        
        assert 'WeightLoader' in repr_str
        assert 'parquet' in repr_str.lower()


# ============================================================================
# Test: Property 11 - Memory-Efficient Weight Loading
# ============================================================================


class TestProperty11MemoryEfficientWeightLoading:
    """
    Validate Property 11: Memory-Efficient Weight Loading.
    
    For any model loaded via WeightLoader, the memory usage increase should be
    approximately equal to the model size (no significant extra allocations), and
    loading should complete in < 100ms for models < 200MB.
    
    **Validates: Requirements 3.1, 3.2**
    """
    
    def test_load_time_under_100ms(self, small_parquet_file):
        """Loading should complete in < 100ms for small models."""
        start_time = time.time()
        loader = WeightLoader(small_parquet_file)
        weights = loader.load_weights()
        load_time_ms = (time.time() - start_time) * 1000
        
        # Should load in < 100ms
        assert load_time_ms < 100, f"Load time {load_time_ms:.2f}ms exceeds 100ms"
        assert len(weights) > 0
    
    def test_memory_mapped_loading_is_fast(self, temp_parquet_file):
        """Memory-mapped loading should be faster than regular loading."""
        # Memory-mapped loading
        start_time = time.time()
        loader_mmap = WeightLoader(temp_parquet_file, use_memory_map=True)
        loader_mmap.load_weights()
        mmap_time = time.time() - start_time
        
        # Regular loading
        start_time = time.time()
        loader_regular = WeightLoader(temp_parquet_file, use_memory_map=False)
        loader_regular.load_weights()
        regular_time = time.time() - start_time
        
        # Memory-mapped should be at least as fast (usually faster)
        # Allow some variance due to system conditions
        assert mmap_time <= regular_time * 1.5
    
    def test_lazy_loading_defers_conversion(self, temp_parquet_file):
        """Lazy loading should not load all weights immediately."""
        loader = WeightLoader(temp_parquet_file, cache_weights=False)
        
        # Just initializing should be very fast
        start_time = time.time()
        loader._load_table()
        table_load_time = time.time() - start_time
        
        # Table loading should be nearly instant with memory mapping
        assert table_load_time < 0.05  # < 50ms
    
    def test_zero_copy_conversion(self, temp_parquet_file):
        """Weight conversion should use zero-copy when possible."""
        loader = WeightLoader(temp_parquet_file)
        
        # Load a single layer
        start_time = time.time()
        tensor = loader.get_layer('embeddings.word_embeddings.weight')
        conversion_time = time.time() - start_time
        
        # Conversion should be very fast (< 10ms)
        assert conversion_time < 0.01
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1000, 64)
    
    def test_memory_overhead_is_minimal(self, temp_parquet_file):
        """Memory overhead should be minimal (< 10% of model size)."""
        loader = WeightLoader(temp_parquet_file)
        weights = loader.load_weights()
        
        # Calculate actual model size
        model_size_bytes = sum(
            t.numel() * t.element_size() for t in weights.values()
        )
        
        # Cache size should be approximately equal to model size
        cache_size_bytes = loader.get_cache_size_mb() * 1024 * 1024
        
        # Allow up to 10% overhead
        assert cache_size_bytes <= model_size_bytes * 1.1
    
    def test_cached_access_is_instant(self, temp_parquet_file):
        """Cached weight access should be nearly instant."""
        loader = WeightLoader(temp_parquet_file, cache_weights=True)
        
        # First access (loads and caches)
        layer_name = 'embeddings.word_embeddings.weight'
        tensor1 = loader.get_layer(layer_name)
        
        # Second access (from cache)
        start_time = time.time()
        tensor2 = loader.get_layer(layer_name)
        cache_access_time = time.time() - start_time
        
        # Cache access should be < 1ms
        assert cache_access_time < 0.001
        assert torch.equal(tensor1, tensor2)


# ============================================================================
# Test: Weight Loading Functionality
# ============================================================================


class TestWeightLoading:
    """Test weight loading functionality."""
    
    def test_load_all_weights(self, temp_parquet_file):
        """Should load all weights from Parquet file."""
        loader = WeightLoader(temp_parquet_file)
        weights = loader.load_weights()
        
        assert len(weights) == 4
        assert 'embeddings.word_embeddings.weight' in weights
        assert 'embeddings.position_embeddings.weight' in weights
        assert 'encoder.layer.0.attention.self.query.weight' in weights
        assert 'encoder.layer.0.attention.self.query.bias' in weights
    
    def test_loaded_weights_are_tensors(self, temp_parquet_file):
        """Loaded weights should be PyTorch tensors."""
        loader = WeightLoader(temp_parquet_file)
        weights = loader.load_weights()
        
        for name, tensor in weights.items():
            assert isinstance(tensor, torch.Tensor)
    
    def test_loaded_weights_have_correct_shapes(self, temp_parquet_file):
        """Loaded weights should have correct shapes."""
        loader = WeightLoader(temp_parquet_file)
        weights = loader.load_weights()
        
        assert weights['embeddings.word_embeddings.weight'].shape == (1000, 64)
        assert weights['embeddings.position_embeddings.weight'].shape == (512, 64)
        assert weights['encoder.layer.0.attention.self.query.weight'].shape == (64, 64)
        assert weights['encoder.layer.0.attention.self.query.bias'].shape == (64,)
    
    def test_loaded_weights_have_correct_dtype(self, temp_parquet_file):
        """Loaded weights should have correct dtype."""
        loader = WeightLoader(temp_parquet_file)
        weights = loader.load_weights()
        
        for tensor in weights.values():
            assert tensor.dtype == torch.float32
    
    def test_load_float16_weights(self, float16_parquet_file):
        """Should correctly load float16 weights."""
        loader = WeightLoader(float16_parquet_file)
        weights = loader.load_weights()
        
        assert weights['test.weight'].dtype == torch.float16
        assert weights['test.weight'].shape == (100, 64)
    
    def test_loaded_weights_are_finite(self, temp_parquet_file):
        """Loaded weights should not contain NaN or Inf."""
        loader = WeightLoader(temp_parquet_file)
        weights = loader.load_weights()
        
        for tensor in weights.values():
            assert torch.isfinite(tensor).all()


# ============================================================================
# Test: Lazy Loading
# ============================================================================


class TestLazyLoading:
    """Test lazy loading functionality."""
    
    def test_get_single_layer(self, temp_parquet_file):
        """Should load a single layer on demand."""
        loader = WeightLoader(temp_parquet_file)
        
        tensor = loader.get_layer('embeddings.word_embeddings.weight')
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1000, 64)
    
    def test_get_layer_raises_on_invalid_name(self, temp_parquet_file):
        """Should raise KeyError for invalid layer name."""
        loader = WeightLoader(temp_parquet_file)
        
        with pytest.raises(KeyError):
            loader.get_layer('nonexistent.layer')
    
    def test_lazy_loading_multiple_layers(self, temp_parquet_file):
        """Should load multiple layers independently."""
        loader = WeightLoader(temp_parquet_file)
        
        layer1 = loader.get_layer('embeddings.word_embeddings.weight')
        layer2 = loader.get_layer('embeddings.position_embeddings.weight')
        
        assert layer1.shape == (1000, 64)
        assert layer2.shape == (512, 64)
        assert not torch.equal(layer1[:64], layer2[:64])  # Different data


# ============================================================================
# Test: Caching
# ============================================================================


class TestCaching:
    """Test weight caching functionality."""
    
    def test_cache_stores_loaded_weights(self, temp_parquet_file):
        """Cache should store loaded weights."""
        loader = WeightLoader(temp_parquet_file, cache_weights=True)
        
        # Load weights
        weights = loader.load_weights()
        
        # Check cache
        assert len(loader._weight_cache) == len(weights)
        for name in weights.keys():
            assert name in loader._weight_cache
    
    def test_cache_disabled_does_not_store(self, temp_parquet_file):
        """Disabled cache should not store weights."""
        loader = WeightLoader(temp_parquet_file, cache_weights=False)
        
        # Load weights
        loader.load_weights()
        
        # Cache should be empty
        assert len(loader._weight_cache) == 0
    
    def test_cached_layer_returns_same_tensor(self, temp_parquet_file):
        """Cached layer should return the same tensor object."""
        loader = WeightLoader(temp_parquet_file, cache_weights=True)
        
        layer_name = 'embeddings.word_embeddings.weight'
        tensor1 = loader.get_layer(layer_name)
        tensor2 = loader.get_layer(layer_name)
        
        # Should be the same object (cached)
        assert tensor1 is tensor2
    
    def test_clear_cache(self, temp_parquet_file):
        """Should clear the cache."""
        loader = WeightLoader(temp_parquet_file, cache_weights=True)
        
        # Load and cache
        loader.load_weights()
        assert len(loader._weight_cache) > 0
        
        # Clear cache
        loader.clear_cache()
        assert len(loader._weight_cache) == 0
    
    def test_get_cache_size(self, temp_parquet_file):
        """Should report cache size correctly."""
        loader = WeightLoader(temp_parquet_file, cache_weights=True)
        
        # Empty cache
        assert loader.get_cache_size_mb() == 0.0
        
        # Load weights
        weights = loader.load_weights()
        
        # Cache size should be > 0
        cache_size_mb = loader.get_cache_size_mb()
        assert cache_size_mb > 0
        
        # Calculate expected size
        expected_size_mb = sum(
            t.numel() * t.element_size() for t in weights.values()
        ) / (1024 * 1024)
        
        # Should match (within rounding)
        assert abs(cache_size_mb - expected_size_mb) < 0.1


# ============================================================================
# Test: Metadata
# ============================================================================


class TestMetadata:
    """Test metadata functionality."""
    
    def test_get_layer_names(self, temp_parquet_file):
        """Should return list of layer names."""
        loader = WeightLoader(temp_parquet_file)
        layer_names = loader.get_layer_names()
        
        assert len(layer_names) == 4
        assert 'embeddings.word_embeddings.weight' in layer_names
        assert 'embeddings.position_embeddings.weight' in layer_names
    
    def test_get_metadata(self, temp_parquet_file):
        """Should return metadata about weights."""
        loader = WeightLoader(temp_parquet_file)
        metadata = loader.get_metadata()
        
        assert 'num_layers' in metadata
        assert 'total_parameters' in metadata
        assert 'dtypes' in metadata
        assert 'layer_shapes' in metadata
        
        assert metadata['num_layers'] == 4
        assert metadata['total_parameters'] > 0
    
    def test_metadata_layer_shapes(self, temp_parquet_file):
        """Metadata should include layer shapes."""
        loader = WeightLoader(temp_parquet_file)
        metadata = loader.get_metadata()
        
        layer_shapes = metadata['layer_shapes']
        assert layer_shapes['embeddings.word_embeddings.weight'] == [1000, 64]
        assert layer_shapes['embeddings.position_embeddings.weight'] == [512, 64]
    
    def test_metadata_dtypes(self, temp_parquet_file):
        """Metadata should include dtypes."""
        loader = WeightLoader(temp_parquet_file)
        metadata = loader.get_metadata()
        
        assert 'torch.float32' in metadata['dtypes']


# ============================================================================
# Test: Device Handling
# ============================================================================


class TestDeviceHandling:
    """Test device handling for weights."""
    
    def test_weights_loaded_to_cpu(self, temp_parquet_file):
        """Weights should be loaded to CPU by default."""
        loader = WeightLoader(temp_parquet_file, device='cpu')
        weights = loader.load_weights()
        
        for tensor in weights.values():
            assert tensor.device.type == 'cpu'
    
    def test_device_parameter_respected(self, temp_parquet_file):
        """Device parameter should be respected."""
        loader = WeightLoader(temp_parquet_file, device='cpu')
        assert loader.device == 'cpu'


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_parquet_file(self):
        """Should handle invalid Parquet file gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as f:
            f.write("invalid parquet data")
            invalid_path = f.name
        
        try:
            loader = WeightLoader(invalid_path)
            with pytest.raises(Exception):  # Should raise some exception
                loader.load_weights()
        finally:
            os.unlink(invalid_path)
    
    def test_missing_required_columns(self):
        """Should handle missing required columns."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            parquet_path = f.name
        
        # Create table with missing columns
        table = pa.Table.from_pydict({
            'layer_name': ['test'],
            # Missing 'shape', 'dtype', 'data', 'num_params'
        })
        pq.write_table(table, parquet_path)
        
        try:
            loader = WeightLoader(parquet_path)
            with pytest.raises(Exception):
                loader.load_weights()
        finally:
            os.unlink(parquet_path)


# ============================================================================
# Test: Integration
# ============================================================================


class TestIntegration:
    """Test integration scenarios."""
    
    def test_load_and_use_weights_in_model(self, temp_parquet_file):
        """Should load weights that can be used in a model."""
        loader = WeightLoader(temp_parquet_file)
        weights = loader.load_weights()
        
        # Create a simple linear layer
        linear = torch.nn.Linear(64, 64, bias=True)
        
        # Load weights into layer
        query_weight = weights['encoder.layer.0.attention.self.query.weight']
        query_bias = weights['encoder.layer.0.attention.self.query.bias']
        
        with torch.no_grad():
            linear.weight.copy_(query_weight)
            linear.bias.copy_(query_bias)
        
        # Test forward pass
        input_tensor = torch.randn(2, 64)
        output = linear(input_tensor)
        
        assert output.shape == (2, 64)
        assert torch.isfinite(output).all()
    
    def test_multiple_loaders_same_file(self, temp_parquet_file):
        """Multiple loaders should be able to access the same file."""
        loader1 = WeightLoader(temp_parquet_file)
        loader2 = WeightLoader(temp_parquet_file)
        
        weights1 = loader1.load_weights()
        weights2 = loader2.load_weights()
        
        # Should load identical weights
        for name in weights1.keys():
            assert torch.equal(weights1[name], weights2[name])

"""
Unit tests for Arrow zero-copy utilities

Tests zero-copy views, memory-mapped loading, and vectorized operations.

Requirements: Task 12.1
"""

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile

from llm_compression.arrow_zero_copy import (
    ArrowMemoryView,
    ArrowBatchView,
    load_table_mmap,
    get_embeddings_buffer,
    prune_columns,
    filter_table_zero_copy,
    compute_similarity_zero_copy
)


@pytest.fixture
def sample_table():
    """Create sample Arrow table for testing"""
    schema = pa.schema([
        ('id', pa.string()),
        ('timestamp', pa.timestamp('us')),
        ('text', pa.string()),
        ('embedding', pa.list_(pa.float32())),
        ('score', pa.float32()),
    ])
    
    data = {
        'id': ['mem1', 'mem2', 'mem3'],
        'timestamp': [datetime.now() for _ in range(3)],
        'text': ['Hello world', 'Test data', 'Sample text'],
        'embedding': [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        'score': [0.9, 0.8, 0.7],
    }
    
    return pa.table(data, schema=schema)


@pytest.fixture
def temp_parquet_file(sample_table):
    """Create temporary Parquet file"""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        pq.write_table(sample_table, f.name)
        file_path = Path(f.name)
    
    yield file_path
    
    # Clean up with retry for Windows
    try:
        file_path.unlink()
    except PermissionError:
        import time
        time.sleep(0.1)  # Wait a bit for file handle to release
        try:
            file_path.unlink()
        except:
            pass  # Ignore if still fails


class TestArrowMemoryView:
    """Test ArrowMemoryView class"""
    
    def test_init(self, sample_table):
        """Test initialization"""
        view = ArrowMemoryView(sample_table, row_index=0)
        assert view is not None
        assert view._row_index == 0
    
    def test_getitem(self, sample_table):
        """Test field access"""
        view = ArrowMemoryView(sample_table, row_index=0)
        
        # Access field
        id_scalar = view['id']
        assert id_scalar.as_py() == 'mem1'
    
    def test_get_py(self, sample_table):
        """Test Python object materialization"""
        view = ArrowMemoryView(sample_table, row_index=1)
        
        # Get as Python object
        text = view.get_py('text')
        assert text == 'Test data'
    
    def test_get_numpy(self, sample_table):
        """Test NumPy array conversion"""
        view = ArrowMemoryView(sample_table, row_index=0)
        
        # Get embedding as NumPy array
        embedding = view.get_numpy('embedding')
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
    
    def test_keys(self, sample_table):
        """Test field name listing"""
        view = ArrowMemoryView(sample_table, row_index=0)
        
        keys = view.keys()
        assert 'id' in keys
        assert 'text' in keys
        assert 'embedding' in keys
    
    def test_contains(self, sample_table):
        """Test field existence check"""
        view = ArrowMemoryView(sample_table, row_index=0)
        
        assert 'id' in view
        assert 'nonexistent' not in view
    
    def test_schema(self, sample_table):
        """Test schema access"""
        view = ArrowMemoryView(sample_table, row_index=0)
        
        schema = view.schema
        assert 'id' in schema.names
        assert 'embedding' in schema.names


class TestArrowBatchView:
    """Test ArrowBatchView class"""
    
    def test_init(self, sample_table):
        """Test initialization"""
        batch = ArrowBatchView(sample_table)
        assert batch is not None
        assert len(batch) == 3
    
    def test_len(self, sample_table):
        """Test length"""
        batch = ArrowBatchView(sample_table)
        assert len(batch) == 3
    
    def test_getitem(self, sample_table):
        """Test row access"""
        batch = ArrowBatchView(sample_table)
        
        # Access row
        view = batch[0]
        assert isinstance(view, ArrowMemoryView)
        assert view.get_py('id') == 'mem1'
    
    def test_iteration(self, sample_table):
        """Test iteration"""
        batch = ArrowBatchView(sample_table)
        
        # Iterate over rows
        ids = []
        for view in batch:
            ids.append(view.get_py('id'))
        
        assert ids == ['mem1', 'mem2', 'mem3']
    
    def test_to_pandas(self, sample_table):
        """Test pandas conversion"""
        batch = ArrowBatchView(sample_table)
        
        # Convert to pandas
        df = batch.to_pandas(columns=['id', 'text'])
        assert len(df) == 3
        assert 'id' in df.columns
        assert 'text' in df.columns


class TestLoadTableMmap:
    """Test memory-mapped table loading"""
    
    def test_load_existing_file(self, temp_parquet_file):
        """Test loading existing file"""
        table = load_table_mmap(temp_parquet_file)
        assert table is not None
        assert len(table) == 3
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file"""
        with pytest.raises(FileNotFoundError):
            load_table_mmap('nonexistent.parquet')


class TestGetEmbeddingsBuffer:
    """Test embedding extraction"""
    
    def test_extract_embeddings(self, sample_table):
        """Test embedding extraction"""
        embeddings = get_embeddings_buffer(sample_table, 'embedding')
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 3)  # 3 rows, 3-dim embeddings
    
    def test_invalid_column(self, sample_table):
        """Test invalid column name"""
        with pytest.raises(ValueError):
            get_embeddings_buffer(sample_table, 'nonexistent')


class TestPruneColumns:
    """Test column pruning"""
    
    def test_prune_columns(self, sample_table):
        """Test column pruning"""
        pruned = prune_columns(sample_table, ['id', 'text'])
        
        assert len(pruned.schema) == 2
        assert 'id' in pruned.schema.names
        assert 'text' in pruned.schema.names
        assert 'embedding' not in pruned.schema.names
    
    def test_prune_invalid_columns(self, sample_table):
        """Test pruning with invalid columns"""
        with pytest.raises(ValueError):
            prune_columns(sample_table, ['id', 'nonexistent'])


class TestFilterTableZeroCopy:
    """Test zero-copy filtering"""
    
    def test_filter_table(self, sample_table):
        """Test table filtering"""
        # Create mask
        mask = pa.array([True, False, True])
        
        # Filter table
        filtered = filter_table_zero_copy(sample_table, mask)
        
        assert len(filtered) == 2
        assert filtered['id'][0].as_py() == 'mem1'
        assert filtered['id'][1].as_py() == 'mem3'
    
    def test_filter_invalid_mask(self, sample_table):
        """Test filtering with invalid mask"""
        # Wrong length mask
        mask = pa.array([True, False])
        
        with pytest.raises(ValueError):
            filter_table_zero_copy(sample_table, mask)


class TestComputeSimilarityZeroCopy:
    """Test vectorized similarity computation"""
    
    def test_compute_similarity(self):
        """Test similarity computation"""
        # Create embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        
        # Query embedding
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Compute similarities
        similarities = compute_similarity_zero_copy(embeddings, query)
        
        assert len(similarities) == 3
        assert similarities[0] > 0.99  # Should be ~1.0
        assert similarities[1] < 0.01  # Should be ~0.0
        assert similarities[2] < 0.01  # Should be ~0.0
    
    def test_compute_similarity_normalized(self):
        """Test similarity with non-normalized vectors"""
        embeddings = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ], dtype=np.float32)
        
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        similarities = compute_similarity_zero_copy(embeddings, query)
        
        # Should normalize automatically
        assert similarities[0] > 0.99
        assert similarities[1] < 0.01
    
    def test_compute_similarity_zero_norm(self):
        """Test similarity with zero-norm query"""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        
        query = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        similarities = compute_similarity_zero_copy(embeddings, query)
        
        # Should return zeros
        assert similarities[0] == 0.0


class TestZeroCopyPerformance:
    """Test zero-copy performance characteristics"""
    
    def test_no_copy_on_slice(self, sample_table):
        """Test that slicing doesn't copy data"""
        # Get original buffer address
        original_buffer = sample_table['id'].chunk(0).buffers()[1]
        original_address = original_buffer.address
        
        # Slice table
        sliced = sample_table.slice(0, 1)
        
        # Get sliced buffer address
        sliced_buffer = sliced['id'].chunk(0).buffers()[1]
        sliced_address = sliced_buffer.address
        
        # Addresses should be the same (zero-copy)
        assert sliced_address == original_address
    
    def test_no_copy_on_filter(self, sample_table):
        """Test that filtering doesn't copy data unnecessarily"""
        # Create mask
        mask = pa.array([True, True, True])
        
        # Filter table
        filtered = filter_table_zero_copy(sample_table, mask)
        
        # Should have same number of rows
        assert len(filtered) == len(sample_table)
    
    def test_column_pruning_reduces_memory(self, sample_table):
        """Test that column pruning reduces memory footprint"""
        # Prune to single column
        pruned = prune_columns(sample_table, ['id'])
        
        # Should have fewer columns
        assert len(pruned.schema) < len(sample_table.schema)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

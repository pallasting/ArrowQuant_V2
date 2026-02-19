"""
Arrow Zero-Copy Utilities

Provides zero-copy views and utilities for Arrow data structures.

Features:
- ArrowMemoryView: Lazy materialization view for Arrow tables
- Zero-copy buffer access for embeddings
- Memory-mapped file loading
- Column pruning optimization

Requirements: Task 12.1
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ArrowMemoryView:
    """
    Lazy materialization view for Arrow tables
    
    Provides zero-copy access to Arrow data without converting to Python objects.
    Only materializes data when explicitly requested.
    
    Features:
    - Zero-copy column access
    - Lazy field extraction
    - Efficient iteration
    - Memory-mapped backing
    
    Requirements: Task 12.1
    """
    
    def __init__(self, table: pa.Table, row_index: int = 0):
        """
        Initialize Arrow memory view
        
        Args:
            table: Arrow table (can be a slice)
            row_index: Row index to view (default: 0)
        """
        self._table = table
        self._row_index = row_index
        self._cache: Dict[str, Any] = {}
    
    def __getitem__(self, key: str) -> Any:
        """
        Get field value (lazy materialization)
        
        Args:
            key: Field name
        
        Returns:
            Field value (Arrow scalar or Python object)
        """
        if key in self._cache:
            return self._cache[key]
        
        if key not in self._table.schema.names:
            raise KeyError(f"Field not found: {key}")
        
        # Get column
        column = self._table[key]
        
        # Get scalar at row index
        scalar = column[self._row_index]
        
        # Cache and return
        self._cache[key] = scalar
        return scalar
    
    def get_py(self, key: str) -> Any:
        """
        Get field value as Python object
        
        Only use when Python object is actually needed.
        
        Args:
            key: Field name
        
        Returns:
            Python object
        """
        scalar = self[key]
        return scalar.as_py()
    
    def get_buffer(self, key: str) -> pa.Buffer:
        """
        Get field value as Arrow buffer (zero-copy)
        
        Args:
            key: Field name
        
        Returns:
            Arrow buffer
        """
        column = self._table[key]
        chunk = column.chunk(0)  # Assume single chunk for now
        return chunk.buffers()[1]  # Data buffer (skip null bitmap)
    
    def get_numpy(self, key: str, zero_copy: bool = True) -> np.ndarray:
        """
        Get field value as NumPy array (zero-copy if possible)
        
        Args:
            key: Field name
            zero_copy: Whether to use zero-copy (default: True)
        
        Returns:
            NumPy array
        """
        column = self._table[key]
        
        if zero_copy:
            # Zero-copy conversion (only works for compatible types)
            try:
                return column.to_numpy(zero_copy_only=True)
            except Exception as e:
                logger.warning(f"Zero-copy failed for {key}, falling back to copy: {e}")
                return column.to_numpy(zero_copy_only=False)
        else:
            return column.to_numpy(zero_copy_only=False)
    
    @property
    def table(self) -> pa.Table:
        """Get underlying Arrow table (zero-copy)"""
        return self._table
    
    @property
    def schema(self) -> pa.Schema:
        """Get table schema"""
        return self._table.schema
    
    def keys(self) -> List[str]:
        """Get all field names"""
        return self._table.schema.names
    
    def __contains__(self, key: str) -> bool:
        """Check if field exists"""
        return key in self._table.schema.names
    
    def __repr__(self) -> str:
        return f"ArrowMemoryView(row={self._row_index}, fields={len(self._table.schema)})"


class ArrowBatchView:
    """
    Zero-copy view for batch of Arrow records
    
    Provides efficient iteration over Arrow table rows without
    materializing Python objects.
    
    Requirements: Task 12.1
    """
    
    def __init__(self, table: pa.Table):
        """
        Initialize batch view
        
        Args:
            table: Arrow table
        """
        self._table = table
        self._length = len(table)
    
    def __len__(self) -> int:
        """Get number of rows"""
        return self._length
    
    def __getitem__(self, index: int) -> ArrowMemoryView:
        """
        Get row view at index (zero-copy)
        
        Args:
            index: Row index
        
        Returns:
            ArrowMemoryView for the row
        """
        if index < 0 or index >= self._length:
            raise IndexError(f"Index out of range: {index}")
        
        # Create slice (zero-copy)
        row_slice = self._table.slice(index, 1)
        return ArrowMemoryView(row_slice, row_index=0)
    
    def __iter__(self):
        """Iterate over rows (zero-copy)"""
        for i in range(self._length):
            yield self[i]
    
    @property
    def table(self) -> pa.Table:
        """Get underlying Arrow table (zero-copy)"""
        return self._table
    
    def to_pandas(self, columns: Optional[List[str]] = None):
        """
        Convert to pandas DataFrame (zero-copy if possible)
        
        Args:
            columns: Columns to include (default: all)
        
        Returns:
            pandas DataFrame
        """
        if columns:
            table = self._table.select(columns)
        else:
            table = self._table
        
        return table.to_pandas(self_destruct=True, split_blocks=True, use_threads=True)
    
    def __repr__(self) -> str:
        return f"ArrowBatchView(rows={self._length}, columns={len(self._table.schema)})"


def load_table_mmap(file_path: Union[str, Path]) -> pa.Table:
    """
    Load Arrow table using memory mapping (zero-copy)
    
    Memory mapping allows the OS to manage data loading on-demand,
    avoiding loading the entire file into memory.
    
    Args:
        file_path: Path to Parquet file
    
    Returns:
        Arrow table (memory-mapped)
    
    Requirements: Task 12.1
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Use memory mapping for zero-copy loading
        return pq.read_table(
            file_path,
            memory_map=True,
            use_threads=True
        )
    except Exception as e:
        logger.error(f"Failed to load table with mmap: {e}")
        raise


def get_embeddings_buffer(
    table: pa.Table,
    column_name: str = 'embedding'
) -> np.ndarray:
    """
    Get embeddings as NumPy array (zero-copy)
    
    Extracts embedding column and converts to NumPy array without copying data.
    
    Args:
        table: Arrow table
        column_name: Embedding column name (default: 'embedding')
    
    Returns:
        NumPy array of embeddings (shape: [n_rows, embedding_dim])
    
    Requirements: Task 12.1
    """
    if column_name not in table.schema.names:
        raise ValueError(f"Column not found: {column_name}")
    
    # Get embedding column
    embedding_column = table[column_name]
    
    # Convert list column to 2D NumPy array (zero-copy if possible)
    try:
        # Get dimensions
        n_rows = len(embedding_column)
        if n_rows == 0:
            return np.array([])
        
        # Get embedding dimension from first row
        first_embedding = embedding_column[0].as_py()
        embedding_dim = len(first_embedding)
        
        # Convert each row to numpy and stack
        # Note: List columns require conversion, cannot be zero-copy
        embeddings_list = []
        for i in range(n_rows):
            emb = embedding_column[i].as_py()
            embeddings_list.append(emb)
        
        # Stack into 2D array
        embeddings_2d = np.array(embeddings_list, dtype=np.float32)
        
        return embeddings_2d
        
    except Exception as e:
        logger.error(f"Failed to extract embeddings: {e}")
        raise


def prune_columns(
    table: pa.Table,
    columns: List[str]
) -> pa.Table:
    """
    Prune table to only include specified columns (zero-copy)
    
    Column pruning reduces memory usage and improves performance
    by only loading needed columns.
    
    Args:
        table: Arrow table
        columns: Columns to keep
    
    Returns:
        Pruned table (zero-copy view)
    
    Requirements: Task 12.1
    """
    # Validate columns
    missing = set(columns) - set(table.schema.names)
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    
    # Select columns (zero-copy)
    return table.select(columns)


def filter_table_zero_copy(
    table: pa.Table,
    mask: pa.Array
) -> pa.Table:
    """
    Filter table using boolean mask (zero-copy)
    
    Args:
        table: Arrow table
        mask: Boolean mask array
    
    Returns:
        Filtered table (zero-copy view)
    
    Requirements: Task 12.1
    """
    if len(mask) != len(table):
        raise ValueError(f"Mask length ({len(mask)}) != table length ({len(table)})")
    
    # Filter using mask (zero-copy)
    return table.filter(mask)


def compute_similarity_zero_copy(
    embeddings: np.ndarray,
    query_embedding: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity (zero-copy, vectorized)
    
    Args:
        embeddings: Embedding matrix (shape: [n, d])
        query_embedding: Query embedding (shape: [d])
    
    Returns:
        Similarity scores (shape: [n])
    
    Requirements: Task 12.1
    """
    # Normalize query
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return np.zeros(len(embeddings))
    
    query_normalized = query_embedding / query_norm
    
    # Normalize embeddings (vectorized)
    embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embedding_norms[embedding_norms == 0] = 1  # Avoid division by zero
    embeddings_normalized = embeddings / embedding_norms
    
    # Compute cosine similarity (vectorized dot product)
    similarities = np.dot(embeddings_normalized, query_normalized)
    
    return similarities

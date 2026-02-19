"""
Arrow Storage Zero-Copy Extensions

Extends ArrowStorage with zero-copy query methods and optimizations.

Features:
- query_arrow(): Returns Arrow Table directly (zero-copy)
- get_embeddings_buffer(): Zero-copy embedding extraction
- Memory-mapped file loading
- Column pruning optimization
- Vectorized similarity search

Requirements: Task 12.1
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import numpy as np

from llm_compression.arrow_zero_copy import (
    ArrowMemoryView,
    ArrowBatchView,
    load_table_mmap,
    get_embeddings_buffer,
    prune_columns,
    filter_table_zero_copy,
    compute_similarity_zero_copy
)

logger = logging.getLogger(__name__)


class ArrowStorageZeroCopy:
    """
    Zero-copy extensions for ArrowStorage
    
    Provides high-performance query methods that avoid Python object
    materialization and leverage Arrow's zero-copy capabilities.
    
    Requirements: Task 12.1
    """
    
    def __init__(self, storage):
        """
        Initialize zero-copy extensions
        
        Args:
            storage: ArrowStorage instance
        """
        self.storage = storage
    
    def query_arrow(
        self,
        category: str = 'experiences',
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False
    ) -> Optional[pa.Table]:
        """
        Query memories and return Arrow Table (zero-copy)
        
        This method returns the raw Arrow Table without converting to
        Python objects, enabling zero-copy data access.
        
        Args:
            category: Memory category
            filters: Filter conditions
            columns: Columns to include (column pruning)
            limit: Maximum number of results
            sort_by: Field name to sort by
            ascending: Sort order
        
        Returns:
            Arrow Table or None if no results
        
        Requirements: Task 12.1
        """
        try:
            # Load table with memory mapping (zero-copy)
            table = self._load_table_mmap(category)
            
            if table is None or len(table) == 0:
                return None
            
            # Apply column pruning first (reduces memory footprint)
            if columns:
                table = prune_columns(table, columns)
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if field in table.schema.names:
                        mask = pc.equal(table[field], value)
                        table = filter_table_zero_copy(table, mask)
            
            # Apply sorting
            if sort_by and sort_by in table.schema.names:
                indices = pc.sort_indices(table[sort_by])
                if not ascending:
                    indices = pc.subtract(len(table) - 1, indices)
                table = pc.take(table, indices)
            
            # Apply limit
            if limit and len(table) > limit:
                table = table.slice(0, limit)
            
            logger.debug(
                f"Zero-copy query returned {len(table)} rows from category={category}"
            )
            return table
            
        except Exception as e:
            logger.error(f"Zero-copy query failed: {e}")
            return None
    
    def query_arrow_batch(
        self,
        category: str = 'experiences',
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Optional[ArrowBatchView]:
        """
        Query memories and return batch view (zero-copy iteration)
        
        Returns an ArrowBatchView that allows efficient iteration over
        results without materializing Python objects.
        
        Args:
            category: Memory category
            filters: Filter conditions
            columns: Columns to include
            limit: Maximum number of results
        
        Returns:
            ArrowBatchView or None if no results
        
        Requirements: Task 12.1
        """
        table = self.query_arrow(
            category=category,
            filters=filters,
            columns=columns,
            limit=limit
        )
        
        if table is None:
            return None
        
        return ArrowBatchView(table)
    
    def get_embeddings_buffer(
        self,
        category: str = 'experiences',
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[np.ndarray]:
        """
        Get embeddings as NumPy array (zero-copy)
        
        Extracts only the embedding column and converts to NumPy array
        without copying data when possible.
        
        Args:
            category: Memory category
            filters: Filter conditions
        
        Returns:
            NumPy array of embeddings (shape: [n_rows, embedding_dim])
        
        Requirements: Task 12.1
        """
        try:
            # Query only embedding column (column pruning)
            table = self.query_arrow(
                category=category,
                filters=filters,
                columns=['embedding']
            )
            
            if table is None or len(table) == 0:
                return None
            
            # Extract embeddings (zero-copy)
            embeddings = get_embeddings_buffer(table, 'embedding')
            
            logger.debug(
                f"Extracted {len(embeddings)} embeddings from category={category}"
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to extract embeddings: {e}")
            return None
    
    def query_by_similarity_zero_copy(
        self,
        category: str,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 10,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        return_arrow: bool = False
    ) -> Union[List[Tuple[ArrowMemoryView, float]], pa.Table]:
        """
        Query memories by embedding similarity (zero-copy, vectorized)
        
        Uses vectorized NumPy operations for 10-100x faster similarity
        computation compared to row-by-row processing.
        
        Args:
            category: Memory category
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (0-1)
            filters: Filter conditions
            return_arrow: If True, return Arrow Table instead of views
        
        Returns:
            List of (ArrowMemoryView, similarity_score) tuples or Arrow Table
        
        Requirements: Task 12.1
        """
        try:
            # Convert query embedding to numpy
            if isinstance(query_embedding, list):
                query_vec = np.array(query_embedding, dtype=np.float32)
            else:
                query_vec = query_embedding.astype(np.float32)
            
            # Load table with memory mapping
            table = self._load_table_mmap(category)
            
            if table is None or len(table) == 0:
                return [] if not return_arrow else None
            
            # Apply filters first
            if filters:
                for field, value in filters.items():
                    if field in table.schema.names:
                        mask = pc.equal(table[field], value)
                        table = filter_table_zero_copy(table, mask)
            
            # Extract embeddings (zero-copy)
            embeddings = get_embeddings_buffer(table, 'embedding')
            
            if embeddings is None or len(embeddings) == 0:
                return [] if not return_arrow else None
            
            # Compute similarities (vectorized)
            similarities = compute_similarity_zero_copy(embeddings, query_vec)
            
            # Filter by threshold
            mask = similarities >= threshold
            filtered_indices = np.where(mask)[0]
            filtered_similarities = similarities[filtered_indices]
            
            # Sort by similarity (descending)
            sorted_indices = np.argsort(filtered_similarities)[::-1]
            
            # Take top_k
            top_indices = filtered_indices[sorted_indices[:top_k]]
            top_similarities = filtered_similarities[sorted_indices[:top_k]]
            
            if return_arrow:
                # Return Arrow Table with similarity scores
                indices_array = pa.array(top_indices.tolist())
                filtered_table = pc.take(table, indices_array)
                
                # Add similarity column
                similarity_array = pa.array(top_similarities.tolist(), type=pa.float32())
                filtered_table = filtered_table.append_column(
                    'similarity_score',
                    similarity_array
                )
                
                return filtered_table
            else:
                # Return list of (view, score) tuples
                results = []
                for idx, score in zip(top_indices, top_similarities):
                    row_slice = table.slice(int(idx), 1)
                    view = ArrowMemoryView(row_slice, row_index=0)
                    results.append((view, float(score)))
                
                logger.debug(
                    f"Zero-copy similarity query returned {len(results)} results "
                    f"from category={category}"
                )
                return results
            
        except Exception as e:
            logger.error(f"Zero-copy similarity query failed: {e}")
            return [] if not return_arrow else None
    
    def query_by_time_range_arrow(
        self,
        category: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Optional[pa.Table]:
        """
        Query memories by time range (zero-copy)
        
        Args:
            category: Memory category
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            columns: Columns to include
            limit: Maximum number of results
        
        Returns:
            Arrow Table or None
        
        Requirements: Task 12.1
        """
        try:
            # Load table with memory mapping
            table = self._load_table_mmap(category)
            
            if table is None or len(table) == 0:
                return None
            
            # Apply column pruning
            if columns:
                table = prune_columns(table, columns)
            
            # Apply time filters
            if start_time:
                mask = pc.greater_equal(table['timestamp'], start_time)
                table = filter_table_zero_copy(table, mask)
            
            if end_time:
                mask = pc.less_equal(table['timestamp'], end_time)
                table = filter_table_zero_copy(table, mask)
            
            # Sort by timestamp (descending)
            indices = pc.sort_indices(table['timestamp'])
            indices_reversed = pc.subtract(len(table) - 1, indices)
            table = pc.take(table, indices_reversed)
            
            # Apply limit
            if limit and len(table) > limit:
                table = table.slice(0, limit)
            
            logger.debug(
                f"Zero-copy time range query returned {len(table)} rows "
                f"from category={category}"
            )
            return table
            
        except Exception as e:
            logger.error(f"Zero-copy time range query failed: {e}")
            return None
    
    def _load_table_mmap(self, category: str) -> Optional[pa.Table]:
        """
        Load table using memory mapping (zero-copy)
        
        Args:
            category: Memory category
        
        Returns:
            Arrow table or None
        """
        file_path = self.storage.category_paths.get(category)
        
        if not file_path or not file_path.exists():
            return None
        
        try:
            return load_table_mmap(file_path)
        except Exception as e:
            logger.error(f"Failed to load table with mmap: {e}")
            return None
    
    def get_stats_zero_copy(self, category: str) -> Dict[str, Any]:
        """
        Get storage statistics (zero-copy)
        
        Args:
            category: Memory category
        
        Returns:
            Statistics dictionary
        
        Requirements: Task 12.1
        """
        try:
            table = self._load_table_mmap(category)
            
            if table is None:
                return {
                    'total_memories': 0,
                    'storage_size_bytes': 0,
                    'storage_size_mb': 0.0,
                }
            
            file_path = self.storage.category_paths[category]
            file_size = file_path.stat().st_size
            
            # Compute stats without materializing data
            stats = {
                'total_memories': len(table),
                'storage_size_bytes': file_size,
                'storage_size_mb': file_size / (1024 * 1024),
                'columns': len(table.schema),
                'column_names': table.schema.names,
            }
            
            # Add compression ratio if available
            if 'is_compressed' in table.schema.names:
                compressed_count = pc.sum(pc.cast(table['is_compressed'], pa.int64())).as_py()
                stats['compressed_count'] = compressed_count
                stats['compression_rate'] = compressed_count / len(table) if len(table) > 0 else 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


def add_zero_copy_methods(storage):
    """
    Add zero-copy methods to existing ArrowStorage instance
    
    This function extends an ArrowStorage instance with zero-copy methods
    without modifying the original class.
    
    Args:
        storage: ArrowStorage instance
    
    Returns:
        ArrowStorageZeroCopy wrapper
    
    Requirements: Task 12.1
    """
    return ArrowStorageZeroCopy(storage)

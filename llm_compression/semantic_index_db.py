"""
Parquet-based semantic index storage for AI-OS memory system.

This module provides persistent storage for semantic indexes using
Apache Arrow and Parquet format for efficient I/O.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from llm_compression.logger import logger


class SemanticIndexDB:
    """
    Parquet-based semantic index storage.
    
    Schema:
    - memory_id: string
    - category: string
    - embedding: list<float16>
    - timestamp: timestamp
    - indexed_at: timestamp
    
    Features:
    - Efficient Parquet storage
    - Category-based file organization
    - Vectorized similarity search
    - Incremental updates
    
    Example:
        >>> index_db = SemanticIndexDB("./data/indexes")
        >>> 
        >>> # Add single entry
        >>> index_db.add_entry(
        ...     memory_id="mem_123",
        ...     category="knowledge",
        ...     embedding=np.random.rand(384),
        ...     timestamp=datetime.now()
        ... )
        >>> 
        >>> # Query index
        >>> results = index_db.query(
        ...     category="knowledge",
        ...     query_embedding=np.random.rand(384),
        ...     top_k=5
        ... )
    """
    
    def __init__(self, index_path: str):
        """
        Initialize SemanticIndexDB.
        
        Args:
            index_path: Directory path for storing index files
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._schema = self._create_schema()
        logger.info(f"Initialized SemanticIndexDB at {self.index_path}")
    
    def _create_schema(self) -> pa.Schema:
        """
        Create Arrow schema for index storage.
        
        Returns:
            PyArrow schema
        """
        return pa.schema([
            ('memory_id', pa.string()),
            ('category', pa.string()),
            ('embedding', pa.list_(pa.float16())),
            ('timestamp', pa.timestamp('us')),
            ('indexed_at', pa.timestamp('us')),
        ])
    
    def add_entry(
        self,
        memory_id: str,
        category: str,
        embedding: np.ndarray,
        timestamp: Any
    ) -> None:
        """
        Add single entry to index.
        
        Args:
            memory_id: Unique memory identifier
            category: Memory category
            embedding: Embedding vector
            timestamp: Memory timestamp
            
        Example:
            >>> index_db.add_entry(
            ...     memory_id="mem_123",
            ...     category="knowledge",
            ...     embedding=np.random.rand(384),
            ...     timestamp=datetime.now()
            ... )
        """
        self.batch_add([{
            'memory_id': memory_id,
            'category': category,
            'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            'timestamp': self._normalize_timestamp(timestamp),
            'indexed_at': datetime.now()
        }])
    
    def batch_add(self, entries: List[Any]) -> None:
        """
        Batch add entries to index.
        
        More efficient than calling add_entry() multiple times.
        
        Args:
            entries: List of entry dicts or objects with required attributes
            
        Example:
            >>> entries = [
            ...     {
            ...         'memory_id': 'mem_1',
            ...         'category': 'knowledge',
            ...         'embedding': np.random.rand(384),
            ...         'timestamp': datetime.now()
            ...     },
            ...     # ... more entries
            ... ]
            >>> index_db.batch_add(entries)
        """
        if not entries:
            logger.warning("No entries to add")
            return
        
        logger.debug(f"Batch adding {len(entries)} entries to index")
        
        # Normalize entries to dicts
        normalized_entries = []
        for entry in entries:
            if isinstance(entry, dict):
                normalized = entry.copy()
            else:
                # Extract from object
                normalized = {
                    'memory_id': getattr(entry, 'memory_id', None),
                    'category': getattr(entry, 'category', None),
                    'embedding': getattr(entry, 'embedding', None),
                    'timestamp': getattr(entry, 'timestamp', None),
                }
            
            # Ensure embedding is list
            if isinstance(normalized['embedding'], np.ndarray):
                normalized['embedding'] = normalized['embedding'].tolist()
            
            # Normalize timestamp
            normalized['timestamp'] = self._normalize_timestamp(
                normalized['timestamp']
            )
            
            # Add indexed_at if not present
            if 'indexed_at' not in normalized:
                normalized['indexed_at'] = datetime.now()
            
            normalized_entries.append(normalized)
        
        # Group by category
        categories = {}
        for entry in normalized_entries:
            category = entry['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(entry)
        
        # Write to category-specific files
        for category, category_entries in categories.items():
            self._write_category_entries(category, category_entries)
        
        logger.debug(f"Batch added {len(entries)} entries")
    
    def _write_category_entries(
        self,
        category: str,
        entries: List[Dict]
    ) -> None:
        """
        Write entries to category-specific Parquet file.
        
        Args:
            category: Category name
            entries: List of entry dicts
        """
        category_file = self.index_path / f"{category}_index.parquet"
        
        try:
            # Convert to Arrow table
            table = pa.Table.from_pylist(entries, schema=self._schema)
            
            if category_file.exists():
                # Append to existing file
                existing = pq.read_table(category_file)
                combined = pa.concat_tables([existing, table])
                pq.write_table(combined, category_file, compression='snappy')
            else:
                # Create new file
                pq.write_table(table, category_file, compression='snappy')
            
            logger.debug(
                f"Wrote {len(entries)} entries to {category_file.name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to write entries to {category_file}: {e}")
            raise
    
    def query(
        self,
        category: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Query index for similar embeddings.
        
        Args:
            category: Category to search
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of result dicts with memory_id, similarity, timestamp
            
        Example:
            >>> results = index_db.query(
            ...     category="knowledge",
            ...     query_embedding=np.random.rand(384),
            ...     top_k=5,
            ...     threshold=0.7
            ... )
            >>> for r in results:
            ...     print(f"{r['memory_id']}: {r['similarity']:.3f}")
        """
        category_file = self.index_path / f"{category}_index.parquet"
        
        if not category_file.exists():
            logger.debug(f"No index file for category: {category}")
            return []
        
        try:
            # Load index
            table = pq.read_table(category_file)
            
            if len(table) == 0:
                return []
            
            # Convert embeddings to numpy array
            embeddings = np.array([
                e.as_py() for e in table['embedding']
            ], dtype=np.float32)
            
            # Ensure query_embedding is 1D
            if query_embedding.ndim == 2:
                query_embedding = query_embedding[0]
            
            # Vectorized cosine similarity computation
            # Normalize stored embeddings row-wise
            emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            emb_norms = np.where(emb_norms == 0, 1e-9, emb_norms)
            embeddings_norm = embeddings / emb_norms

            # Normalize query embedding
            q = query_embedding.astype(np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                q_norm = 1e-9
            q_normalized = q / q_norm

            similarities = embeddings_norm @ q_normalized
            
            # Apply threshold
            if threshold > 0:
                mask = similarities >= threshold
                valid_indices = np.where(mask)[0]
                if len(valid_indices) == 0:
                    return []
                similarities = similarities[valid_indices]
            else:
                valid_indices = np.arange(len(similarities))
            
            # Get top-k
            if len(similarities) > top_k:
                top_indices = np.argsort(similarities)[-top_k:][::-1]
            else:
                top_indices = np.argsort(similarities)[::-1]
            
            # Build results
            results = []
            for idx in top_indices:
                original_idx = valid_indices[idx]
                results.append({
                    'memory_id': table['memory_id'][original_idx].as_py(),
                    'similarity': float(similarities[idx]),
                    'timestamp': table['timestamp'][original_idx].as_py()
                })
            
            logger.debug(
                f"Query returned {len(results)} results for {category}"
            )
            return results
            
        except Exception as e:
            logger.error(f"Failed to query index for {category}: {e}")
            raise
    
    def clear_category(self, category: str) -> None:
        """
        Clear index for a category.
        
        Deletes the category-specific Parquet file.
        
        Args:
            category: Category to clear
            
        Example:
            >>> index_db.clear_category("knowledge")
        """
        category_file = self.index_path / f"{category}_index.parquet"
        
        if category_file.exists():
            category_file.unlink()
            logger.info(f"Cleared index for category: {category}")
        else:
            logger.debug(f"No index file to clear for category: {category}")
    
    def get_categories(self) -> List[str]:
        """
        Get list of indexed categories.
        
        Returns:
            List of category names
        """
        categories = []
        for file in self.index_path.glob("*_index.parquet"):
            category = file.stem.replace("_index", "")
            categories.append(category)
        return sorted(categories)
    
    def get_category_size(self, category: str) -> int:
        """
        Get number of entries in a category index.
        
        Args:
            category: Category name
            
        Returns:
            Number of entries
        """
        category_file = self.index_path / f"{category}_index.parquet"
        
        if not category_file.exists():
            return 0
        
        try:
            table = pq.read_table(category_file)
            return len(table)
        except Exception as e:
            logger.error(f"Failed to get size for {category}: {e}")
            return 0
    
    def _normalize_timestamp(self, timestamp: Any) -> datetime:
        """
        Normalize timestamp to datetime object.
        
        Args:
            timestamp: Timestamp in various formats
            
        Returns:
            datetime object
        """
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, str):
            # Try to parse ISO format
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                logger.warning(f"Failed to parse timestamp: {timestamp}")
                return datetime.now()
        else:
            logger.warning(f"Unknown timestamp type: {type(timestamp)}")
            return datetime.now()

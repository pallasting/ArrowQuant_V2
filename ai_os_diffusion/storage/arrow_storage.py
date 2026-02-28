"""
Arrow Storage Layer for AI-OS Diffusion

Simplified Phase 0 version for unified diffusion architecture.
Will integrate with Rust ArrowStorage backend in Phase 1.

Features (Phase 0):
- Basic Arrow/Parquet storage
- Vector similarity search
- Memory retrieval for conditioning

Future (Phase 1+):
- Rust backend integration via PyO3
- 10-50x speedup with SIMD
- Advanced indexing and compression

Architecture: 🦴 Rust Skeleton (Phase 1) + 🧠 Python Brain (Phase 0)
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ai_os_diffusion.utils.logger import logger
from ai_os_diffusion.utils.errors import DiffusionError


# ============================================================================
# Arrow Schema Definitions
# ============================================================================

def create_memory_schema() -> pa.Schema:
    """
    Create memory schema for diffusion conditioning
    
    Simplified schema for Phase 0:
    - memory_id: Unique identifier
    - timestamp: Creation time
    - content: Memory content (text)
    - embedding: Dense embedding vector (float32)
    - metadata: Additional metadata
    
    Phase 1+ will add:
    - Sparse vectors (Rust backend)
    - Quantized embeddings (INT8/INT4)
    - Advanced compression
    """
    return pa.schema([
        ('memory_id', pa.string()),
        ('timestamp', pa.timestamp('us')),
        ('content', pa.string()),
        ('embedding', pa.list_(pa.float32())),
        ('metadata', pa.struct([
            ('category', pa.string()),
            ('source', pa.string()),
            ('quality_score', pa.float32()),
        ])),
    ])


SCHEMA_REGISTRY = {
    'memories': create_memory_schema(),
}


# ============================================================================
# Arrow Storage Class
# ============================================================================

class ArrowStorage:
    """
    Arrow/Parquet-based storage for memory conditioning
    
    Phase 0: Python implementation with basic functionality
    Phase 1+: Rust backend integration via PyO3 for 10-50x speedup
    
    Features:
    - Store and retrieve memories
    - Vector similarity search
    - Memory conditioning for diffusion
    
    Architecture: 🦴 Rust Skeleton (future) + 🧠 Python Brain (current)
    """
    
    def __init__(
        self,
        storage_path: str = "~/.ai-os-diffusion/memory/",
        compression_level: int = 3
    ):
        """
        Initialize Arrow storage
        
        Args:
            storage_path: Base storage path
            compression_level: zstd compression level (default: 3)
        """
        self.storage_path = Path(storage_path).expanduser()
        self.compression_level = compression_level
        
        # Create storage directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Define storage file
        self.memory_file = self.storage_path / 'memories.parquet'
        
        logger.info(
            f"ArrowStorage initialized: path={self.storage_path}, "
            f"compression_level={compression_level}"
        )
    
    def save(
        self,
        memory_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save memory to storage
        
        Args:
            memory_id: Unique memory identifier
            content: Memory content (text)
            embedding: Dense embedding vector
            metadata: Optional metadata
        
        Raises:
            DiffusionError: If save fails
        """
        try:
            # Create record
            record = self._create_record(memory_id, content, embedding, metadata)
            
            # Append to table
            self._append_to_table(record)
            
            logger.debug(f"Saved memory: {memory_id}")
            
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            raise DiffusionError(f"Failed to save memory: {e}") from e
    
    def load(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Load memory by ID
        
        Args:
            memory_id: Memory ID to load
        
        Returns:
            Memory dict or None if not found
        """
        try:
            table = self._load_table()
            
            if table is None or len(table) == 0:
                return None
            
            # Filter by memory_id
            mask = pc.equal(table['memory_id'], memory_id)
            filtered = table.filter(mask)
            
            if len(filtered) == 0:
                return None
            
            # Convert to dict
            row = filtered.slice(0, 1)
            memory = self._record_to_dict(row)
            
            logger.debug(f"Loaded memory: {memory_id}")
            return memory
            
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return None
    
    def query_by_similarity(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Query memories by embedding similarity
        
        Phase 0: Python implementation with numpy
        Phase 1+: Rust backend with SIMD for 10-50x speedup
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (0-1)
        
        Returns:
            List of (memory_dict, similarity_score) tuples
        """
        try:
            import numpy as np
            
            table = self._load_table()
            
            if table is None or len(table) == 0:
                return []
            
            # Convert query embedding to numpy array
            query_vec = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_vec)
            
            if query_norm == 0:
                logger.warning("Query embedding has zero norm")
                return []
            
            # Compute similarities
            similarities = []
            for i in range(len(table)):
                row = table.slice(i, 1)
                embedding = np.array(row['embedding'][0].as_py(), dtype=np.float32)
                embedding_norm = np.linalg.norm(embedding)
                
                if embedding_norm == 0:
                    continue
                
                # Cosine similarity
                similarity = np.dot(query_vec, embedding) / (query_norm * embedding_norm)
                
                if similarity >= threshold:
                    memory = self._record_to_dict(row)
                    similarities.append((memory, float(similarity)))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k
            results = similarities[:top_k]
            
            logger.debug(f"Similarity query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Similarity query failed: {e}")
            return []
    
    def _create_record(
        self,
        memory_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> pa.RecordBatch:
        """
        Create Arrow record batch
        
        Args:
            memory_id: Memory ID
            content: Memory content
            embedding: Embedding vector
            metadata: Optional metadata
        
        Returns:
            Arrow record batch
        """
        schema = SCHEMA_REGISTRY['memories']
        
        # Default metadata
        if metadata is None:
            metadata = {}
        
        data = {
            'memory_id': [memory_id],
            'timestamp': [datetime.now()],
            'content': [content],
            'embedding': [embedding],
            'metadata': [{
                'category': metadata.get('category', 'general'),
                'source': metadata.get('source', 'unknown'),
                'quality_score': metadata.get('quality_score', 1.0),
            }],
        }
        
        arrays = [pa.array(data[field.name], type=field.type) for field in schema]
        return pa.RecordBatch.from_arrays(arrays, schema=schema)
    
    def _append_to_table(self, record: pa.RecordBatch) -> None:
        """
        Append record to table
        
        Args:
            record: Record batch to append
        """
        if self.memory_file.exists():
            # Append to existing file
            existing_table = pq.read_table(self.memory_file)
            combined_table = pa.concat_tables([existing_table, pa.Table.from_batches([record])])
            
            pq.write_table(
                combined_table,
                self.memory_file,
                compression='zstd',
                compression_level=self.compression_level
            )
        else:
            # Create new file
            table = pa.Table.from_batches([record])
            pq.write_table(
                table,
                self.memory_file,
                compression='zstd',
                compression_level=self.compression_level
            )
    
    def _load_table(self) -> Optional[pa.Table]:
        """
        Load table from storage
        
        Returns:
            Arrow table or None if file doesn't exist
        """
        if not self.memory_file.exists():
            return None
        
        try:
            return pq.read_table(self.memory_file)
        except Exception as e:
            logger.error(f"Failed to load table: {e}")
            return None
    
    def _record_to_dict(self, record: pa.Table) -> Dict[str, Any]:
        """
        Convert Arrow record to dict
        
        Args:
            record: Arrow table (single row)
        
        Returns:
            Memory dict
        """
        return {
            'memory_id': record['memory_id'][0].as_py(),
            'timestamp': record['timestamp'][0].as_py(),
            'content': record['content'][0].as_py(),
            'embedding': record['embedding'][0].as_py(),
            'metadata': record['metadata'][0].as_py(),
        }


class StorageError(DiffusionError):
    """Storage error"""
    pass

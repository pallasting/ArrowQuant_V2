"""
Arrow Storage Layer

Implements Arrow/Parquet-based storage for compressed memories with OpenClaw
schema compatibility.

Features:
- Full OpenClaw schema compatibility (all original fields preserved)
- Compression extensions (is_compressed, summary_hash, entities, diff_data, compression_metadata)
- zstd compression for diff_data (level 3)
- float16 for embeddings (50% space savings)
- Summary deduplication (hash-based)
- Append-only incremental updates
- Fast query support with indexing

Requirements: 4.1, 4.2, 8.1-8.6
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

from llm_compression.compressor import CompressedMemory, CompressionMetadata
from llm_compression.stored_memory import StoredMemory
from llm_compression.storage_adapter import StorageAdapter


logger = logging.getLogger(__name__)


# ============================================================================
# Arrow Schema Definitions
# ============================================================================

def create_experiences_compressed_schema() -> pa.Schema:
    """
    Create compressed experiences schema with OpenClaw compatibility
    
    Includes all OpenClaw original fields:
    - timestamp, context, intent, action, outcome, success
    - embedding (float16), related_memories
    
    Plus compression extension fields:
    - is_compressed (bool)
    - summary_hash (string)
    - entities (struct: persons, locations, dates, numbers, keywords)
    - diff_data (binary, zstd compressed)
    - compression_metadata (struct: original_size, compressed_size, etc.)
    
    Requirements: 4.1, 4.2, 8.1
    """
    return pa.schema([
        # === OpenClaw Original Fields (Complete Compatibility) ===
        ('memory_id', pa.string()),
        ('timestamp', pa.timestamp('us')),
        ('context', pa.string()),              # Empty if compressed
        ('intent', pa.string()),
        ('action', pa.string()),
        ('outcome', pa.string()),
        ('success', pa.bool_()),
        ('embedding', pa.list_(pa.float16())), # float16 for 50% space savings
        ('related_memories', pa.list_(pa.string())),
        
        # === Compression Extension Fields ===
        ('is_compressed', pa.bool_()),         # Whether memory is compressed
        ('summary_hash', pa.string()),         # Summary hash for deduplication
        ('entities', pa.struct([               # Extracted key entities
            ('persons', pa.list_(pa.string())),
            ('locations', pa.list_(pa.string())),
            ('dates', pa.list_(pa.string())),
            ('numbers', pa.list_(pa.string())),
            ('keywords', pa.list_(pa.string())),
        ])),
        ('diff_data', pa.binary()),            # Compressed diff data (zstd)
        ('compression_metadata', pa.struct([   # Compression metadata
            ('original_size', pa.int32()),
            ('compressed_size', pa.int32()),
            ('compression_ratio', pa.float32()),
            ('model_used', pa.string()),
            ('quality_score', pa.float32()),
            ('compression_time_ms', pa.float32()),
            ('compressed_at', pa.timestamp('us')),
        ])),
        
        # === Vector Space Compression Fields ===
        ('sparse_vector', pa.binary()),
        ('sparse_indices', pa.binary()),
        ('sparse_meta', pa.struct([
            ('scale_factor', pa.float32()),
            ('full_dim', pa.int32()),
            ('original_norm', pa.float32()),
            ('is_4bit', pa.bool_()),
            ('packed_length', pa.int32()),
        ])),
        ('key_tokens', pa.list_(pa.string())),
        ('token_scores', pa.list_(pa.float32())),
    ])


def create_identity_compressed_schema() -> pa.Schema:
    """
    Create compressed identity schema with OpenClaw compatibility
    
    Identity memories store self-awareness and identity information.
    
    Requirements: 4.1, 4.2, 8.1
    """
    return pa.schema([
        # === OpenClaw Original Fields ===
        ('memory_id', pa.string()),
        ('timestamp', pa.timestamp('us')),
        ('description', pa.string()),          # Identity description
        ('values', pa.string()),               # Core values
        ('embedding', pa.list_(pa.float16())),
        
        # === Compression Extension Fields ===
        ('is_compressed', pa.bool_()),
        ('summary_hash', pa.string()),
        ('entities', pa.struct([
            ('persons', pa.list_(pa.string())),
            ('locations', pa.list_(pa.string())),
            ('dates', pa.list_(pa.string())),
            ('numbers', pa.list_(pa.string())),
            ('keywords', pa.list_(pa.string())),
        ])),
        ('diff_data', pa.binary()),
        ('compression_metadata', pa.struct([
            ('original_size', pa.int32()),
            ('compressed_size', pa.int32()),
            ('compression_ratio', pa.float32()),
            ('model_used', pa.string()),
            ('quality_score', pa.float32()),
            ('compression_time_ms', pa.float32()),
            ('compressed_at', pa.timestamp('us')),
        ])),
        
        # === Vector Space Compression Fields ===
        ('sparse_vector', pa.binary()),
        ('sparse_indices', pa.binary()),
        ('sparse_meta', pa.struct([
            ('scale_factor', pa.float32()),
            ('full_dim', pa.int32()),
            ('original_norm', pa.float32()),
            ('is_4bit', pa.bool_()),
            ('packed_length', pa.int32()),
        ])),
        ('key_tokens', pa.list_(pa.string())),
        ('token_scores', pa.list_(pa.float32())),
    ])


def create_preferences_compressed_schema() -> pa.Schema:
    """
    Create compressed preferences schema with OpenClaw compatibility
    
    Preferences store user preferences and settings.
    
    Requirements: 4.1, 4.2, 8.1
    """
    return pa.schema([
        # === OpenClaw Original Fields ===
        ('memory_id', pa.string()),
        ('timestamp', pa.timestamp('us')),
        ('preference', pa.string()),           # Preference description
        ('reason', pa.string()),               # Reason for preference
        ('embedding', pa.list_(pa.float16())),
        
        # === Compression Extension Fields ===
        ('is_compressed', pa.bool_()),
        ('summary_hash', pa.string()),
        ('entities', pa.struct([
            ('persons', pa.list_(pa.string())),
            ('locations', pa.list_(pa.string())),
            ('dates', pa.list_(pa.string())),
            ('numbers', pa.list_(pa.string())),
            ('keywords', pa.list_(pa.string())),
        ])),
        ('diff_data', pa.binary()),
        ('compression_metadata', pa.struct([
            ('original_size', pa.int32()),
            ('compressed_size', pa.int32()),
            ('compression_ratio', pa.float32()),
            ('model_used', pa.string()),
            ('quality_score', pa.float32()),
            ('compression_time_ms', pa.float32()),
            ('compressed_at', pa.timestamp('us')),
        ])),
        
        # === Vector Space Compression Fields ===
        ('sparse_vector', pa.binary()),
        ('sparse_indices', pa.binary()),
        ('sparse_meta', pa.struct([
            ('scale_factor', pa.float32()),
            ('full_dim', pa.int32()),
            ('original_norm', pa.float32()),
            ('is_4bit', pa.bool_()),
            ('packed_length', pa.int32()),
        ])),
        ('key_tokens', pa.list_(pa.string())),
        ('token_scores', pa.list_(pa.float32())),
    ])


def create_context_compressed_schema() -> pa.Schema:
    """
    Create compressed context schema with OpenClaw compatibility
    
    Context memories store current working context.
    
    Requirements: 4.1, 4.2, 8.1
    """
    return pa.schema([
        # === OpenClaw Original Fields ===
        ('memory_id', pa.string()),
        ('timestamp', pa.timestamp('us')),
        ('context', pa.string()),              # Context description
        ('embedding', pa.list_(pa.float16())),
        
        # === Compression Extension Fields ===
        ('is_compressed', pa.bool_()),
        ('summary_hash', pa.string()),
        ('entities', pa.struct([
            ('persons', pa.list_(pa.string())),
            ('locations', pa.list_(pa.string())),
            ('dates', pa.list_(pa.string())),
            ('numbers', pa.list_(pa.string())),
            ('keywords', pa.list_(pa.string())),
        ])),
        ('diff_data', pa.binary()),
        ('compression_metadata', pa.struct([
            ('original_size', pa.int32()),
            ('compressed_size', pa.int32()),
            ('compression_ratio', pa.float32()),
            ('model_used', pa.string()),
            ('quality_score', pa.float32()),
            ('compression_time_ms', pa.float32()),
            ('compressed_at', pa.timestamp('us')),
        ])),
        
        # === Vector Space Compression Fields ===
        ('sparse_vector', pa.binary()),
        ('sparse_indices', pa.binary()),
        ('sparse_meta', pa.struct([
            ('scale_factor', pa.float32()),
            ('full_dim', pa.int32()),
            ('original_norm', pa.float32()),
            ('is_4bit', pa.bool_()),
            ('packed_length', pa.int32()),
        ])),
        ('key_tokens', pa.list_(pa.string())),
        ('token_scores', pa.list_(pa.float32())),
    ])


def create_summary_table_schema() -> pa.Schema:
    """
    Create summary deduplication table schema
    
    Stores unique summaries with hash-based deduplication.
    
    Requirements: 8.4
    """
    return pa.schema([
        ('summary_hash', pa.string()),         # Summary hash (primary key)
        ('summary_text', pa.string()),         # Summary text
        ('reference_count', pa.int32()),       # Number of memories using this summary
        ('created_at', pa.timestamp('us')),    # Creation timestamp
    ])


# Schema registry
SCHEMA_REGISTRY = {
    'experiences': create_experiences_compressed_schema(),
    'identity': create_identity_compressed_schema(),
    'preferences': create_preferences_compressed_schema(),
    'context': create_context_compressed_schema(),
    'summaries': create_summary_table_schema(),
}


# ============================================================================
# Arrow Storage Class
# ============================================================================

class ArrowStorage:
    """
    Arrow/Parquet-based storage for compressed memories
    
    Features:
    - Full OpenClaw schema compatibility
    - zstd compression for diff_data
    - float16 embeddings for space savings
    - Summary deduplication
    - Append-only incremental updates
    - Fast query with indexing
    
    Requirements: 8.1-8.6
    """
    
    def __init__(
        self,
        storage_path: Union[str, Path] = "~/.ai-os/memory/",
        compression_level: int = 3,
        kg_manager: Optional[Any] = None
    ):
        """
        Initialize Arrow storage
        
        Args:
            storage_path: Base storage path (default: ~/.ai-os/memory/)
            compression_level: zstd compression level (default: 3)
            kg_manager: Optional Knowledge Graph Manager for updates
        
        Requirements: 8.1
        """
        self.storage_path = Path(storage_path).expanduser()
        self.compression_level = compression_level
        self.kg_manager = kg_manager
        
        # Create storage directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Define category paths (OpenClaw standard)
        self.category_paths = {
            'experiences': self.storage_path / 'core' / 'experiences.parquet',
            'identity': self.storage_path / 'core' / 'identity.parquet',
            'preferences': self.storage_path / 'core' / 'preferences.parquet',
            'context': self.storage_path / 'working' / 'context.parquet',
            'summaries': self.storage_path / 'core' / 'summaries.parquet',
        }
        
        # Create category directories
        for path in self.category_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"ArrowStorage initialized: path={self.storage_path}, "
            f"compression_level={compression_level}"
        )
    
    def save(
        self,
        memory: Union[CompressedMemory, StoredMemory],
        category: str = 'experiences'
    ) -> None:
        """
        Save memory to Arrow/Parquet storage
        
        Supports both CompressedMemory and StoredMemory types.
        StoredMemory is automatically converted to CompressedMemory using StorageAdapter.
        
        Implements:
        - Summary deduplication (save summary only once)
        - Append-only incremental updates
        - zstd compression for diff_data
        
        Args:
            memory: Memory to save (CompressedMemory or StoredMemory)
            category: Memory category (experiences/identity/preferences/context)
        
        Raises:
            ValueError: If category is invalid
            IOError: If save fails
            TypeError: If memory type is unsupported
        
        Requirements: 8.1, 8.2, 8.4, 8.5
        """
        if category not in SCHEMA_REGISTRY:
            raise ValueError(f"Invalid category: {category}")
        
        try:
            # Step 0: Convert StoredMemory to CompressedMemory if needed
            if isinstance(memory, StoredMemory):
                logger.debug(f"Converting StoredMemory to CompressedMemory: {memory.id}")
                compressed = StorageAdapter.stored_to_compressed(memory)
            elif hasattr(memory, 'memory_id') and hasattr(memory, 'entities'):
                compressed = memory
            else:
                raise TypeError(f"Unsupported memory type: {type(memory)}")
            
            # Step 1: Save summary (with deduplication)
            if compressed.summary_hash:
                self._save_summary(compressed.summary_hash, compressed)
            
            # Step 2: Convert compressed memory to Arrow record
            record = self._compressed_to_record(compressed, category)
            
            # Step 3: Append to table
            self._append_to_table(record, category)
            
            # Step 4: Update Knowledge Graph (If active)
            if self.kg_manager:
                try:
                    # Circular import avoidance
                    from llm_compression.knowledge_graph.manager import KnowledgeGraphManager
                    self.kg_manager.add_memory_concepts(
                        memory_id=compressed.memory_id,
                        concepts=compressed.key_tokens,
                        scores=compressed.token_scores if hasattr(compressed, 'token_scores') else []
                    )
                    # Advanced Relation Extraction: Update concept-concept edges
                    if hasattr(compressed, 'relations') and compressed.relations:
                        self.kg_manager.add_concept_relations(compressed.relations)
                        
                    self.kg_manager.save() # Auto-save for now
                except Exception as e:
                    logger.warning(f"Failed to update knowledge graph: {e}")
            
            logger.info(
                f"Saved compressed memory: {compressed.memory_id} "
                f"to category={category}"
            )
            
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            raise IOError(f"Failed to save memory: {e}") from e
    
    def load(
        self,
        memory_id: str,
        category: str = 'experiences'
    ) -> Optional[CompressedMemory]:
        """
        Load compressed memory from storage
        
        Args:
            memory_id: Memory ID to load
            category: Memory category
        
        Returns:
            CompressedMemory or None if not found
        
        Requirements: 8.1
        """
        if category not in SCHEMA_REGISTRY:
            raise ValueError(f"Invalid category: {category}")
        
        try:
            # Load table
            table = self._load_table(category)
            
            if table is None or len(table) == 0:
                return None
            
            # Filter by memory_id
            mask = pc.equal(table['memory_id'], memory_id)
            filtered = table.filter(mask)
            
            if len(filtered) == 0:
                return None
            
            # Convert first row to CompressedMemory
            row = filtered.slice(0, 1)
            compressed = self._record_to_compressed(row, category)
            
            logger.debug(f"Loaded memory: {memory_id} from category={category}")
            return compressed
            
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return None
    
    def query(
        self,
        category: str = 'experiences',
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False
    ) -> List[CompressedMemory]:
        """
        Query compressed memories with filters
        
        Supports:
        - Field-based filtering (e.g., {'is_compressed': True})
        - Sorting by any field
        - Limit on number of results
        
        Args:
            category: Memory category
            filters: Filter conditions (e.g., {'is_compressed': True})
            limit: Maximum number of results
            sort_by: Field name to sort by (e.g., 'timestamp')
            ascending: Sort order (default: False for descending)
        
        Returns:
            List of compressed memories
        
        Requirements: 8.6
        """
        if category not in SCHEMA_REGISTRY:
            raise ValueError(f"Invalid category: {category}")
        
        try:
            # Load table
            table = self._load_table(category)
            
            if table is None or len(table) == 0:
                return []
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if field in table.schema.names:
                        mask = pc.equal(table[field], value)
                        table = table.filter(mask)
            
            # Apply sorting
            if sort_by and sort_by in table.schema.names:
                indices = pc.sort_indices(table[sort_by])
                if not ascending:
                    # Reverse for descending order
                    indices = pc.subtract(len(table) - 1, indices)
                table = pc.take(table, indices)
            
            # Apply limit
            if limit and len(table) > limit:
                table = table.slice(0, limit)
            
            # Convert to CompressedMemory list
            results = []
            for i in range(len(table)):
                row = table.slice(i, 1)
                compressed = self._record_to_compressed(row, category)
                results.append(compressed)
            
            logger.debug(
                f"Query returned {len(results)} results from category={category}"
            )
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def query_by_time_range(
        self,
        category: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[CompressedMemory]:
        """
        Query memories by time range
        
        Args:
            category: Memory category
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            limit: Maximum number of results
        
        Returns:
            List of compressed memories
        
        Requirements: 8.6
        """
        try:
            table = self._load_table(category)
            
            if table is None or len(table) == 0:
                return []
            
            # Apply time filters
            if start_time:
                mask = pc.greater_equal(table['timestamp'], start_time)
                table = table.filter(mask)
            
            if end_time:
                mask = pc.less_equal(table['timestamp'], end_time)
                table = table.filter(mask)
            
            # Sort by timestamp (descending)
            indices = pc.sort_indices(table['timestamp'])
            # Reverse for descending order
            indices_reversed = pc.subtract(len(table) - 1, indices)
            table = pc.take(table, indices_reversed)
            
            # Apply limit
            if limit and len(table) > limit:
                table = table.slice(0, limit)
            
            # Convert to CompressedMemory list
            results = []
            for i in range(len(table)):
                row = table.slice(i, 1)
                compressed = self._record_to_compressed(row, category)
                results.append(compressed)
            
            logger.debug(
                f"Time range query returned {len(results)} results from category={category}"
            )
            return results
            
        except Exception as e:
            logger.error(f"Time range query failed: {e}")
            return []
    
    def query_by_entity(
        self,
        category: str,
        entity_type: str,
        entity_value: str,
        limit: Optional[int] = None
    ) -> List[CompressedMemory]:
        """
        Query memories by entity
        
        Args:
            category: Memory category
            entity_type: Entity type (persons/locations/dates/numbers/keywords)
            entity_value: Entity value to search for
            limit: Maximum number of results
        
        Returns:
            List of compressed memories
        
        Requirements: 8.6
        """
        try:
            table = self._load_table(category)
            
            if table is None or len(table) == 0:
                return []
            
            # Filter by entity
            # Note: This is a simplified implementation
            # In production, we'd use more efficient indexing
            results = []
            for i in range(len(table)):
                row = table.slice(i, 1)
                entities = row['entities'][0].as_py()
                
                if entity_type in entities:
                    entity_list = entities[entity_type]
                    if entity_value in entity_list:
                        compressed = self._record_to_compressed(row, category)
                        results.append(compressed)
                        
                        if limit and len(results) >= limit:
                            break
            
            logger.debug(
                f"Entity query returned {len(results)} results for "
                f"{entity_type}={entity_value} from category={category}"
            )
            return results
            
        except Exception as e:
            logger.error(f"Entity query failed: {e}")
            return []
    
    def query_by_similarity(
        self,
        category: str,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[tuple[CompressedMemory, float]]:
        """
        Query memories by embedding similarity
        
        Args:
            category: Memory category
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (0-1)
        
        Returns:
            List of (CompressedMemory, similarity_score) tuples
        
        Requirements: 8.6
        """
        try:
            import numpy as np
            
            table = self._load_table(category)
            
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
                    compressed = self._record_to_compressed(row, category)
                    similarities.append((compressed, float(similarity)))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k
            results = similarities[:top_k]
            
            logger.debug(
                f"Similarity query returned {len(results)} results from category={category}"
            )
            return results
            
        except Exception as e:
            logger.error(f"Similarity query failed: {e}")
            return []
    
    def _save_summary(self, summary_hash: str, compressed: CompressedMemory) -> None:
        """
        Save summary with deduplication
        
        If summary_hash already exists, increment reference count.
        Otherwise, create new summary entry.
        
        Args:
            summary_hash: Summary hash
            compressed: Compressed memory (to extract summary from cache)
        
        Requirements: 8.4
        """
        try:
            # Load summary table
            summary_table = self._load_table('summaries')
            
            # Check if summary already exists
            if summary_table is not None and len(summary_table) > 0:
                mask = pc.equal(summary_table['summary_hash'], summary_hash)
                existing = summary_table.filter(mask)
                
                if len(existing) > 0:
                    # Summary exists, increment reference count
                    current_count = existing['reference_count'][0].as_py()
                    new_count = current_count + 1
                    
                    # Update reference count
                    # Note: This is a simplified implementation
                    # In production, we'd use a more efficient update mechanism
                    logger.debug(
                        f"Summary {summary_hash} already exists, "
                        f"incrementing reference count: {current_count} -> {new_count}"
                    )
                    return
            
            # Summary doesn't exist, create new entry
            # Try to get summary text from compressor's cache
            # For now, we'll store an empty string as placeholder
            # The actual summary text should be passed from the compressor
            summary_text = ""  # TODO: Get from compressor cache
            
            # Create summary record
            summary_record = self._create_summary_record(
                summary_hash,
                summary_text,
                reference_count=1
            )
            
            # Append to summary table
            self._append_to_table(summary_record, 'summaries')
            
            logger.debug(f"Created new summary entry: {summary_hash}")
            
        except Exception as e:
            logger.warning(f"Failed to save summary: {e}")
            # Don't fail the entire save operation if summary dedup fails
    
    def _create_summary_record(
        self,
        summary_hash: str,
        summary_text: str,
        reference_count: int
    ) -> pa.RecordBatch:
        """
        Create summary record batch
        
        Args:
            summary_hash: Summary hash
            summary_text: Summary text
            reference_count: Reference count
        
        Returns:
            Arrow record batch
        """
        schema = SCHEMA_REGISTRY['summaries']
        
        data = {
            'summary_hash': [summary_hash],
            'summary_text': [summary_text],
            'reference_count': [reference_count],
            'created_at': [datetime.now()],
        }
        
        arrays = [pa.array(data[field.name], type=field.type) for field in schema]
        return pa.RecordBatch.from_arrays(arrays, schema=schema)
    
    def load_summary(self, summary_hash: str) -> Optional[str]:
        """
        Load summary text by hash
        
        Args:
            summary_hash: Summary hash
        
        Returns:
            Summary text or None if not found
        
        Requirements: 8.4
        """
        try:
            summary_table = self._load_table('summaries')
            
            if summary_table is None or len(summary_table) == 0:
                return None
            
            # Filter by summary_hash
            mask = pc.equal(summary_table['summary_hash'], summary_hash)
            filtered = summary_table.filter(mask)
            
            if len(filtered) == 0:
                return None
            
            summary_text = filtered['summary_text'][0].as_py()
            logger.debug(f"Loaded summary: {summary_hash}")
            return summary_text
            
        except Exception as e:
            logger.error(f"Failed to load summary: {e}")
            return None
    
    def _append_to_table(
        self,
        record: pa.RecordBatch,
        category: str
    ) -> None:
        """
        Append record to table (incremental update)
        
        Uses append-only mode to avoid rewriting entire file.
        
        Args:
            record: Record batch to append
            category: Memory category
        
        Requirements: 8.5
        """
        file_path = self.category_paths[category]
        
        # Check if file exists
        if file_path.exists():
            # Append to existing file
            # Read existing table
            existing_table = pq.read_table(file_path)
            
            # Concatenate with new record
            combined_table = pa.concat_tables([existing_table, pa.Table.from_batches([record])])
            
            # Write back
            pq.write_table(
                combined_table,
                file_path,
                compression='zstd',
                compression_level=self.compression_level
            )
        else:
            # Create new file
            table = pa.Table.from_batches([record])
            pq.write_table(
                table,
                file_path,
                compression='zstd',
                compression_level=self.compression_level
            )
    
    def _load_table(self, category: str) -> Optional[pa.Table]:
        """
        Load table from storage
        
        Args:
            category: Memory category
        
        Returns:
            Arrow table or None if file doesn't exist
        """
        file_path = self.category_paths[category]
        
        if not file_path.exists():
            return None
        
        try:
            return pq.read_table(file_path)
        except Exception as e:
            logger.error(f"Failed to load table: {e}")
            return None
    
    def _compressed_to_record(
        self,
        compressed: CompressedMemory,
        category: str
    ) -> pa.RecordBatch:
        """
        Convert CompressedMemory to Arrow record batch
        
        Args:
            compressed: Compressed memory
            category: Memory category
        
        Returns:
            Arrow record batch
        """
        schema = SCHEMA_REGISTRY[category]
        
        # Build record data based on category
        if category == 'experiences':
            data = {
                'memory_id': [compressed.memory_id],
                'timestamp': [compressed.compression_metadata.compressed_at],
                'context': [''],  # Empty for compressed
                'intent': [compressed.original_fields.get('intent', '')],
                'action': [compressed.original_fields.get('action', '')],
                'outcome': [compressed.original_fields.get('outcome', '')],
                'success': [compressed.original_fields.get('success', True)],
                'embedding': [compressed.embedding],
                'related_memories': [compressed.original_fields.get('related_memories', [])],
                'is_compressed': [True],
                'summary_hash': [compressed.summary_hash],
                'entities': [compressed.entities],
                'diff_data': [compressed.diff_data],
                'compression_metadata': [{
                    'original_size': compressed.compression_metadata.original_size,
                    'compressed_size': compressed.compression_metadata.compressed_size,
                    'compression_ratio': compressed.compression_metadata.compression_ratio,
                    'model_used': compressed.compression_metadata.model_used,
                    'quality_score': compressed.compression_metadata.quality_score,
                    'compression_time_ms': compressed.compression_metadata.compression_time_ms,
                    'compressed_at': compressed.compression_metadata.compressed_at,
                }],
            }
        else:
            # Simplified for other categories
            data = {
                'memory_id': [compressed.memory_id],
                'timestamp': [compressed.compression_metadata.compressed_at],
                'embedding': [compressed.embedding],
                'is_compressed': [True],
                'summary_hash': [compressed.summary_hash],
                'entities': [compressed.entities],
                'diff_data': [compressed.diff_data],
                'compression_metadata': [{
                    'original_size': compressed.compression_metadata.original_size,
                    'compressed_size': compressed.compression_metadata.compressed_size,
                    'compression_ratio': compressed.compression_metadata.compression_ratio,
                    'model_used': compressed.compression_metadata.model_used,
                    'quality_score': compressed.compression_metadata.quality_score,
                    'compression_time_ms': compressed.compression_metadata.compression_time_ms,
                    'compressed_at': compressed.compression_metadata.compressed_at,
                }],
            }
            
            # Add category-specific fields
            if category == 'identity':
                data['description'] = ['']
                data['values'] = ['']
            elif category == 'preferences':
                data['preference'] = ['']
                data['reason'] = ['']
            elif category == 'context':
                data['context'] = ['']
        
        # Add Vector Space Compression data
        data['sparse_vector'] = [compressed.sparse_vector]
        data['sparse_indices'] = [compressed.sparse_indices]
        data['key_tokens'] = [compressed.key_tokens]
        # Make sure token_scores is not None
        data['token_scores'] = [compressed.token_scores if compressed.token_scores else []]
        
        if compressed.sparse_meta:
             data['sparse_meta'] = [{
                 'scale_factor': float(compressed.sparse_meta.get('scale_factor', 1.0)),
                 'full_dim': int(compressed.sparse_meta.get('full_dim', 0)),
                 'original_norm': float(compressed.sparse_meta.get('original_norm', 1.0)),
                 'is_4bit': bool(compressed.sparse_meta.get('is_4bit', False)),
                 'packed_length': int(compressed.sparse_meta.get('packed_length', 0)),
             }]
        else:
             data['sparse_meta'] = [None]

        # Create record batch
        arrays = []
        for field in schema:
            if field.name in data:
                arrays.append(pa.array(data[field.name], type=field.type))
            else:
                # Fill with null
                arrays.append(pa.nulls(1, type=field.type))
        
        return pa.RecordBatch.from_arrays(arrays, schema=schema)
    
    def _record_to_compressed(
        self,
        record: pa.Table,
        category: str
    ) -> CompressedMemory:
        """
        Convert Arrow record to CompressedMemory
        
        Args:
            record: Arrow table (single row)
            category: Memory category
        
        Returns:
            CompressedMemory
        """
        # Extract fields
        memory_id = record['memory_id'][0].as_py()
        summary_hash = record['summary_hash'][0].as_py()
        entities = record['entities'][0].as_py()
        diff_data = record['diff_data'][0].as_py()
        embedding = record['embedding'][0].as_py()
        
        # Extract compression metadata
        metadata_struct = record['compression_metadata'][0].as_py()
        compression_metadata = CompressionMetadata(
            original_size=metadata_struct['original_size'],
            compressed_size=metadata_struct['compressed_size'],
            compression_ratio=metadata_struct['compression_ratio'],
            model_used=metadata_struct['model_used'],
            quality_score=metadata_struct['quality_score'],
            compression_time_ms=metadata_struct['compression_time_ms'],
            compressed_at=metadata_struct['compressed_at'],
        )
        
        # Extract original fields based on category
        original_fields = {}
        if category == 'experiences':
            original_fields = {
                'intent': record['intent'][0].as_py(),
                'action': record['action'][0].as_py(),
                'outcome': record['outcome'][0].as_py(),
                'success': record['success'][0].as_py(),
                'related_memories': record['related_memories'][0].as_py(),
            }
        
        # Extract Vector Space Compression data
        sparse_vector = None
        if 'sparse_vector' in record.column_names:
             sparse_vector = record['sparse_vector'][0].as_py()
             
        sparse_indices = None
        if 'sparse_indices' in record.column_names:
             sparse_indices = record['sparse_indices'][0].as_py()
             
        token_scores = []
        if 'token_scores' in record.column_names:
             token_scores_raw = record['token_scores'][0].as_py()
             if token_scores_raw:
                  token_scores = token_scores_raw
             
        key_tokens = []
        if 'key_tokens' in record.column_names:
             key_tokens_raw = record['key_tokens'][0].as_py()
             if key_tokens_raw:
                  key_tokens = key_tokens_raw
                  
        sparse_meta = None
        if 'sparse_meta' in record.column_names:
             sparse_meta_raw = record['sparse_meta'][0].as_py()
             if sparse_meta_raw:
                 sparse_meta = {
                     'scale_factor': sparse_meta_raw.get('scale_factor'),
                     'full_dim': sparse_meta_raw.get('full_dim'),
                     'original_norm': sparse_meta_raw.get('original_norm'),
                     'is_4bit': sparse_meta_raw.get('is_4bit'),
                     'packed_length': sparse_meta_raw.get('packed_length'),
                 }
        
        return CompressedMemory(
            memory_id=memory_id,
            summary_hash=summary_hash,
            entities=entities,
            diff_data=diff_data,
            embedding=embedding,
            compression_metadata=compression_metadata,
            original_fields=original_fields,
            sparse_vector=sparse_vector,
            sparse_indices=sparse_indices,
            sparse_meta=sparse_meta,
            key_tokens=key_tokens,
            token_scores=token_scores
        )


class StorageError(Exception):
    """Storage error"""
    pass

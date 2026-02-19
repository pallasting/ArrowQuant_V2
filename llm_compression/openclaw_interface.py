"""
OpenClaw Memory Interface Adapter

Provides transparent compression and reconstruction through OpenClaw-compatible APIs.
Fully compatible with OpenClaw's Arrow schema and standard storage paths.

Features:
- Automatic compression decision (based on text length threshold)
- Transparent reconstruction (auto-detect compressed flag)
- Support for all standard paths (core/working/long-term/shared)
- Semantic search with embedding similarity
- Backward compatibility with uncompressed memories

Requirements: 4.1-4.7
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from llm_compression.compressor import LLMCompressor, CompressedMemory, MemoryType
from llm_compression.reconstructor import LLMReconstructor, ReconstructedMemory
from llm_compression.arrow_storage import ArrowStorage


logger = logging.getLogger(__name__)


class OpenClawMemoryInterface:
    """
    OpenClaw-compatible memory interface with transparent compression
    
    Provides standard OpenClaw APIs:
    - store_memory: Store memory with automatic compression
    - retrieve_memory: Retrieve memory with automatic reconstruction
    - search_memories: Semantic search using embeddings
    - get_related_memories: Find related memories by similarity
    
    Requirements: 4.1-4.7
    """
    
    def __init__(
        self,
        storage_path: str = "~/.ai-os/memory/",
        compressor: Optional[LLMCompressor] = None,
        reconstructor: Optional[LLMReconstructor] = None,
        storage: Optional[ArrowStorage] = None,
        auto_compress_threshold: int = 100
    ):
        """
        Initialize OpenClaw memory interface
        
        Args:
            storage_path: Base storage path (default: ~/.ai-os/memory/)
            compressor: LLM compressor instance
            reconstructor: LLM reconstructor instance
            storage: Arrow storage instance
            auto_compress_threshold: Minimum text length to compress (chars)
        
        Requirements: 4.3, 4.4
        """
        self.storage_path = Path(storage_path).expanduser()
        self.compressor = compressor
        self.reconstructor = reconstructor
        self.storage = storage or ArrowStorage(storage_path)
        self.auto_compress_threshold = auto_compress_threshold
        
        # Define OpenClaw standard storage paths
        self.category_paths = {
            'core': {
                'identity': 'identity',
                'experiences': 'experiences',
                'preferences': 'preferences',
            },
            'working': {
                'context': 'context',
            },
            'long_term': {
                'archived': 'archived',
            },
            'shared': {
                'team': 'team',
            }
        }
        
        logger.info(
            f"OpenClawMemoryInterface initialized: "
            f"storage_path={self.storage_path}, "
            f"auto_compress_threshold={auto_compress_threshold}"
        )
    
    async def store_memory(
        self,
        memory: Dict[str, Any],
        memory_category: str = "experiences"
    ) -> str:
        """
        Store memory with automatic compression decision
        
        Algorithm:
        1. Extract text fields based on category
        2. Calculate total text length
        3. If length >= threshold: compress
        4. Otherwise: store uncompressed
        5. Save to Arrow storage
        6. Return memory ID
        
        Args:
            memory: Memory object (OpenClaw schema compatible)
            memory_category: Memory category (identity/experiences/preferences/context)
        
        Returns:
            str: Memory ID
        
        Requirements: 4.4, 4.5
        """
        try:
            # Generate memory ID
            memory_id = self._generate_memory_id()
            
            # Extract text fields
            text_fields = self._extract_text_fields(memory, memory_category)
            combined_text = '\n'.join(
                f"{key}: {value}"
                for key, value in text_fields.items()
                if value
            )
            total_length = len(combined_text)
            
            logger.debug(
                f"Storing memory: category={memory_category}, "
                f"text_length={total_length}, "
                f"threshold={self.auto_compress_threshold}"
            )
            
            # Decide whether to compress
            if total_length >= self.auto_compress_threshold and self.compressor:
                # Compress memory
                logger.info(
                    f"Compressing memory {memory_id}: "
                    f"length={total_length} >= threshold={self.auto_compress_threshold}"
                )
                
                # Determine memory type
                memory_type = self._determine_memory_type(combined_text, memory_category)
                
                # Compress
                compressed = await self.compressor.compress(
                    combined_text,
                    memory_type,
                    metadata={
                        'memory_id': memory_id,
                        'category': memory_category,
                        **memory
                    }
                )
                
                # Update memory ID
                compressed.memory_id = memory_id
                
                # Save compressed memory
                self.storage.save(compressed, memory_category)
                
                logger.info(
                    f"Memory {memory_id} compressed and saved: "
                    f"{compressed.compression_metadata.original_size} -> "
                    f"{compressed.compression_metadata.compressed_size} bytes "
                    f"({compressed.compression_metadata.compression_ratio:.2f}x)"
                )
            else:
                # Store uncompressed
                logger.info(
                    f"Storing memory {memory_id} uncompressed: "
                    f"length={total_length} < threshold={self.auto_compress_threshold}"
                )
                
                # Create uncompressed memory (using compressor's helper)
                if self.compressor:
                    uncompressed = self.compressor._store_uncompressed(
                        combined_text,
                        metadata={
                            'memory_id': memory_id,
                            'category': memory_category,
                            **memory
                        }
                    )
                    uncompressed.memory_id = memory_id
                    self.storage.save(uncompressed, memory_category)
                else:
                    # Fallback: create minimal uncompressed memory without compressor
                    logger.warning("No compressor available, creating minimal uncompressed memory")
                    try:
                        import zstandard as zstd
                    except ImportError:
                        import zstd
                    import numpy as np
                    from llm_compression.compressor import CompressedMemory, CompressionMetadata
                    from datetime import datetime
                    
                    # Create minimal uncompressed memory
                    uncompressed = CompressedMemory(
                        memory_id=memory_id,
                        summary_hash='',
                        entities={},
                        diff_data=zstd.compress(combined_text.encode('utf-8'), level=3),
                        embedding=[0.0] * 384,  # Zero embedding as fallback
                        compression_metadata=CompressionMetadata(
                            original_size=len(combined_text),
                            compressed_size=len(combined_text),
                            compression_ratio=1.0,
                            model_used='uncompressed',
                            quality_score=1.0,
                            compression_time_ms=0.0,
                            compressed_at=datetime.now()
                        ),
                        original_fields={
                            'memory_id': memory_id,
                            'category': memory_category,
                            **memory
                        }
                    )
                    self.storage.save(uncompressed, memory_category)
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise StorageError(f"Failed to store memory: {e}") from e
    
    async def retrieve_memory(
        self,
        memory_id: str,
        memory_category: str = "experiences"
    ) -> Dict[str, Any]:
        """
        Retrieve memory with automatic reconstruction
        
        Algorithm:
        1. Load compressed memory from storage
        2. Check if compressed (is_compressed flag)
        3. If compressed: reconstruct using reconstructor
        4. Otherwise: return as-is
        5. Convert to OpenClaw memory format
        
        Args:
            memory_id: Memory ID
            memory_category: Memory category
        
        Returns:
            Dict: Full memory object (OpenClaw schema)
        
        Requirements: 4.4, 4.6
        """
        try:
            # Load from storage
            compressed = self.storage.load(memory_id, memory_category)
            
            if compressed is None:
                raise KeyError(f"Memory {memory_id} not found in category {memory_category}")
            
            # Check if compressed
            if compressed.summary_hash and self.reconstructor:
                # Reconstruct compressed memory
                logger.info(f"Reconstructing compressed memory: {memory_id}")
                
                reconstructed = await self.reconstructor.reconstruct(
                    compressed,
                    verify_quality=True
                )
                
                # Convert to OpenClaw format
                memory = self._reconstructed_to_memory(
                    reconstructed,
                    compressed,
                    memory_category
                )
                
                logger.info(
                    f"Memory {memory_id} reconstructed: "
                    f"confidence={reconstructed.confidence:.2f}"
                )
                
                if reconstructed.warnings:
                    logger.warning(
                        f"Reconstruction warnings for {memory_id}: "
                        f"{reconstructed.warnings}"
                    )
            else:
                # Uncompressed memory - diff_data contains raw text (not zstd compressed)
                logger.debug(f"Retrieving uncompressed memory: {memory_id}")
                
                # diff_data is already raw text for uncompressed memories
                original_text = compressed.diff_data.decode('utf-8')
                
                # Convert to OpenClaw format
                memory = self._uncompressed_to_memory(
                    original_text,
                    compressed,
                    memory_category
                )
            
            return memory
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            raise RetrievalError(f"Failed to retrieve memory: {e}") from e
    
    async def search_memories(
        self,
        query: str,
        memory_category: str = "experiences",
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using embedding similarity
        
        Algorithm:
        1. Compute query embedding
        2. Query storage by similarity
        3. Retrieve and reconstruct top-k memories
        4. Return as OpenClaw memory list
        
        Args:
            query: Search query text
            memory_category: Memory category
            top_k: Number of results to return
        
        Returns:
            List[Dict]: List of matching memories
        
        Requirements: 4.4
        """
        try:
            if not self.compressor:
                logger.error("No compressor available for embedding computation")
                return []
            
            # Compute query embedding
            query_embedding = self.compressor._compute_embedding(query)
            
            # Query storage by similarity
            results = self.storage.query_by_similarity(
                category=memory_category,
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=0.0
            )
            
            # Retrieve and reconstruct memories
            memories = []
            for compressed, similarity in results:
                try:
                    memory = await self.retrieve_memory(
                        compressed.memory_id,
                        memory_category
                    )
                    memory['_similarity'] = similarity
                    memories.append(memory)
                except Exception as e:
                    logger.warning(
                        f"Failed to retrieve memory {compressed.memory_id}: {e}"
                    )
            
            logger.info(
                f"Search returned {len(memories)} results for query: {query[:50]}..."
            )
            
            return memories
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def get_related_memories(
        self,
        memory_id: str,
        memory_category: str = "experiences",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get related memories by embedding similarity
        
        Algorithm:
        1. Load source memory
        2. Use its embedding to find similar memories
        3. Exclude source memory from results
        4. Return top-k related memories
        
        Args:
            memory_id: Source memory ID
            memory_category: Memory category
            top_k: Number of related memories to return
        
        Returns:
            List[Dict]: List of related memories
        
        Requirements: 4.4
        """
        try:
            # Load source memory
            compressed = self.storage.load(memory_id, memory_category)
            
            if compressed is None:
                logger.warning(f"Memory {memory_id} not found")
                return []
            
            # Query by similarity using source embedding
            results = self.storage.query_by_similarity(
                category=memory_category,
                query_embedding=compressed.embedding,
                top_k=top_k + 1,  # +1 to account for source memory
                threshold=0.0
            )
            
            # Retrieve and reconstruct related memories (exclude source)
            related_memories = []
            for related_compressed, similarity in results:
                if related_compressed.memory_id == memory_id:
                    continue  # Skip source memory
                
                try:
                    memory = await self.retrieve_memory(
                        related_compressed.memory_id,
                        memory_category
                    )
                    memory['_similarity'] = similarity
                    related_memories.append(memory)
                    
                    if len(related_memories) >= top_k:
                        break
                except Exception as e:
                    logger.warning(
                        f"Failed to retrieve related memory {related_compressed.memory_id}: {e}"
                    )
            
            logger.info(
                f"Found {len(related_memories)} related memories for {memory_id}"
            )
            
            return related_memories
            
        except Exception as e:
            logger.error(f"Failed to get related memories: {e}")
            return []
    
    def _extract_text_fields(
        self,
        memory: Dict[str, Any],
        category: str
    ) -> Dict[str, str]:
        """
        Extract text fields from memory based on category
        
        Args:
            memory: Memory object
            category: Memory category
        
        Returns:
            Dict mapping field names to text values
        
        Requirements: 4.5
        """
        if category == "experiences":
            return {
                'context': memory.get('context', ''),
                'action': memory.get('action', ''),
                'outcome': memory.get('outcome', ''),
            }
        elif category == "identity":
            return {
                'description': memory.get('description', ''),
                'values': memory.get('values', ''),
            }
        elif category == "preferences":
            return {
                'preference': memory.get('preference', ''),
                'reason': memory.get('reason', ''),
            }
        elif category == "context":
            return {
                'context': memory.get('context', ''),
            }
        else:
            # Unknown category, extract all string fields
            return {
                k: v for k, v in memory.items()
                if isinstance(v, str) and v
            }
    
    def _determine_memory_type(
        self,
        text: str,
        category: str
    ) -> MemoryType:
        """
        Determine memory type for compression
        
        Args:
            text: Combined text
            category: Memory category
        
        Returns:
            MemoryType enum value
        """
        # Check for code patterns
        code_indicators = ['def ', 'class ', 'import ', 'function ', 'const ', 'var ', 'let ']
        if any(indicator in text for indicator in code_indicators):
            return MemoryType.CODE
        
        # Check for long text
        if len(text) > 500:
            return MemoryType.LONG_TEXT
        
        # Default to text
        return MemoryType.TEXT
    
    def _reconstructed_to_memory(
        self,
        reconstructed: ReconstructedMemory,
        compressed: CompressedMemory,
        category: str
    ) -> Dict[str, Any]:
        """
        Convert reconstructed memory to OpenClaw format
        
        Args:
            reconstructed: Reconstructed memory
            compressed: Original compressed memory
            category: Memory category
        
        Returns:
            Dict: OpenClaw-compatible memory object
        """
        # Parse reconstructed text back into fields
        text_fields = self._parse_reconstructed_text(
            reconstructed.full_text,
            category
        )
        
        # Build memory object
        memory = {
            'memory_id': reconstructed.memory_id,
            'timestamp': compressed.compression_metadata.compressed_at,
            **text_fields,
            **compressed.original_fields,
            '_compressed': True,
            '_confidence': reconstructed.confidence,
        }
        
        if reconstructed.warnings:
            memory['_warnings'] = reconstructed.warnings
        
        return memory
    
    def _uncompressed_to_memory(
        self,
        original_text: str,
        compressed: CompressedMemory,
        category: str
    ) -> Dict[str, Any]:
        """
        Convert uncompressed memory to OpenClaw format
        
        Args:
            original_text: Original text
            compressed: Compressed memory object (with metadata)
            category: Memory category
        
        Returns:
            Dict: OpenClaw-compatible memory object
        """
        # Parse text back into fields
        text_fields = self._parse_reconstructed_text(original_text, category)
        
        # Build memory object
        memory = {
            'memory_id': compressed.memory_id,
            'timestamp': compressed.compression_metadata.compressed_at,
            **text_fields,
            **compressed.original_fields,
            '_compressed': False,
        }
        
        return memory
    
    def _parse_reconstructed_text(
        self,
        text: str,
        category: str
    ) -> Dict[str, str]:
        """
        Parse reconstructed text back into field structure
        
        Simple implementation: split by field labels
        
        Args:
            text: Reconstructed text
            category: Memory category
        
        Returns:
            Dict mapping field names to values
        """
        fields = {}
        
        if category == "experiences":
            # Try to parse "context: ...\naction: ...\noutcome: ..." format
            lines = text.split('\n')
            current_field = None
            current_value = []
            
            for line in lines:
                if line.startswith('context:'):
                    if current_field:
                        fields[current_field] = '\n'.join(current_value).strip()
                    current_field = 'context'
                    current_value = [line[8:].strip()]
                elif line.startswith('action:'):
                    if current_field:
                        fields[current_field] = '\n'.join(current_value).strip()
                    current_field = 'action'
                    current_value = [line[7:].strip()]
                elif line.startswith('outcome:'):
                    if current_field:
                        fields[current_field] = '\n'.join(current_value).strip()
                    current_field = 'outcome'
                    current_value = [line[8:].strip()]
                else:
                    if current_field:
                        current_value.append(line)
            
            if current_field:
                fields[current_field] = '\n'.join(current_value).strip()
            
            # Fallback: if parsing failed, put everything in context
            if not fields:
                fields['context'] = text
        else:
            # For other categories, use simple mapping
            if category == "identity":
                fields['description'] = text
            elif category == "preferences":
                fields['preference'] = text
            elif category == "context":
                fields['context'] = text
            else:
                fields['text'] = text
        
        return fields
    
    def _generate_memory_id(self) -> str:
        """
        Generate unique memory ID
        
        Format: timestamp_uuid
        
        Returns:
            str: Unique memory ID
        
        Requirements: 4.4
        """
        timestamp = int(time.time() * 1000)
        random_part = uuid.uuid4().hex[:8]
        return f"{timestamp}_{random_part}"


class StorageError(Exception):
    """Storage error"""
    pass


class RetrievalError(Exception):
    """Retrieval error"""
    pass

"""
Fallback Strategy for LLM Compression

Implements 4-level degradation strategy:
1. Cloud API (high quality)
2. Local Model (medium quality)
3. Simple Compression (Phase 0 - zstd only)
4. Direct Storage (uncompressed)

Feature: llm-compression-integration
Requirements: 13.1, 13.2, 13.3
Property 10: Model Fallback Strategy
"""

import time
try:
    import zstandard as zstd
except ImportError:
    import zstd
from typing import Optional, Dict, Any
from datetime import datetime

from llm_compression.logger import logger
from llm_compression.compressor import (
    LLMCompressor,
    CompressedMemory,
    CompressionMetadata,
    MemoryType
)
from llm_compression.model_selector import ModelSelector, QualityLevel
from llm_compression.errors import (
    LLMAPIError,
    LLMTimeoutError,
    CompressionError,
    GPUResourceError
)


class FallbackStrategy:
    """
    4-level fallback strategy for compression
    
    Degradation levels:
    1. Cloud API - Highest quality, uses remote LLM service
    2. Local Model - Medium quality, uses local LLM (if available)
    3. Simple Compression - Phase 0 algorithm (zstd only, no LLM)
    4. Direct Storage - No compression, stores original text
    
    Requirements: 13.1, 13.2, 13.3
    Property 10: Model Fallback Strategy
    """
    
    def __init__(
        self,
        compressor: LLMCompressor,
        model_selector: ModelSelector,
        enable_local_fallback: bool = True,
        enable_simple_compression: bool = True
    ):
        """
        Initialize fallback strategy
        
        Args:
            compressor: LLM compressor instance
            model_selector: Model selector for choosing models
            enable_local_fallback: Whether to try local models (Level 2)
            enable_simple_compression: Whether to try simple compression (Level 3)
        """
        self.compressor = compressor
        self.model_selector = model_selector
        self.enable_local_fallback = enable_local_fallback
        self.enable_simple_compression = enable_simple_compression
        
        # Track fallback statistics
        self.fallback_stats = {
            'cloud_api_success': 0,
            'cloud_api_failure': 0,
            'local_model_success': 0,
            'local_model_failure': 0,
            'simple_compression_success': 0,
            'simple_compression_failure': 0,
            'direct_storage': 0
        }
        
        logger.info(
            f"FallbackStrategy initialized: "
            f"local_fallback={enable_local_fallback}, "
            f"simple_compression={enable_simple_compression}"
        )
    
    async def compress_with_fallback(
        self,
        text: str,
        memory_type: MemoryType = MemoryType.TEXT,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CompressedMemory:
        """
        Compress with 4-level fallback strategy
        
        Algorithm:
        1. Try Cloud API (high quality)
        2. If fails, try Local Model (medium quality)
        3. If fails, try Simple Compression (Phase 0)
        4. If fails, Direct Storage (uncompressed)
        
        Args:
            text: Text to compress
            memory_type: Memory type
            metadata: Additional metadata
            
        Returns:
            CompressedMemory: Compressed memory (or uncompressed fallback)
            
        Requirements: 13.1, 13.2, 13.3
        Property 10: Model Fallback Strategy
        """
        # Level 1: Try Cloud API
        try:
            logger.debug("Level 1: Attempting Cloud API compression")
            model_config = self.model_selector.select_model(
                memory_type=memory_type,
                text_length=len(text),
                quality_requirement=QualityLevel.HIGH
            )
            
            if not model_config.is_local:
                compressed = await self.compressor.compress(
                    text,
                    memory_type,
                    metadata
                )
                self.fallback_stats['cloud_api_success'] += 1
                logger.info("Level 1: Cloud API compression successful")
                return compressed
            else:
                logger.debug("Level 1: No cloud API available, skipping")
                
        except (LLMAPIError, LLMTimeoutError) as e:
            self.fallback_stats['cloud_api_failure'] += 1
            logger.warning(f"Level 1: Cloud API failed: {e}, trying local model")
        except Exception as e:
            self.fallback_stats['cloud_api_failure'] += 1
            logger.warning(f"Level 1: Cloud API failed unexpectedly: {e}")
        
        # Level 2: Try Local Model
        if self.enable_local_fallback:
            try:
                logger.debug("Level 2: Attempting Local Model compression")
                model_config = self.model_selector.select_model(
                    memory_type=memory_type,
                    text_length=len(text),
                    quality_requirement=QualityLevel.STANDARD
                )
                
                if model_config.is_local:
                    compressed = await self.compressor.compress(
                        text,
                        memory_type,
                        metadata
                    )
                    self.fallback_stats['local_model_success'] += 1
                    logger.info("Level 2: Local Model compression successful")
                    return compressed
                else:
                    logger.debug("Level 2: No local model available, skipping")
                    
            except (LLMAPIError, LLMTimeoutError, GPUResourceError) as e:
                self.fallback_stats['local_model_failure'] += 1
                logger.warning(f"Level 2: Local Model failed: {e}, trying simple compression")
            except Exception as e:
                self.fallback_stats['local_model_failure'] += 1
                logger.warning(f"Level 2: Local Model failed unexpectedly: {e}")
        
        # Level 3: Try Simple Compression (Phase 0)
        if self.enable_simple_compression:
            try:
                logger.debug("Level 3: Attempting Simple Compression (Phase 0)")
                compressed = self._simple_compress(text, metadata)
                self.fallback_stats['simple_compression_success'] += 1
                logger.info("Level 3: Simple Compression successful")
                return compressed
            except Exception as e:
                self.fallback_stats['simple_compression_failure'] += 1
                logger.error(f"Level 3: Simple Compression failed: {e}, storing uncompressed")
        
        # Level 4: Direct Storage (uncompressed)
        logger.warning("Level 4: All compression methods failed, storing uncompressed")
        self.fallback_stats['direct_storage'] += 1
        return self._store_uncompressed(text, metadata)
    
    def _simple_compress(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> CompressedMemory:
        """
        Simple compression using zstd only (Phase 0 algorithm)
        
        No LLM involved, just byte-level compression.
        Achieves ~1.2-3x compression ratio.
        
        Args:
            text: Text to compress
            metadata: Additional metadata
            
        Returns:
            CompressedMemory: Compressed memory
            
        Requirements: 13.2
        """
        start_time = time.time()
        
        try:
            # Compress with zstd (level 9 for maximum compression)
            text_bytes = text.encode('utf-8')
            compressed_data = zstd.compress(text_bytes, level=9)
            
            # Calculate sizes and ratio
            original_size = len(text_bytes)
            compressed_size = len(compressed_data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            compression_time_ms = (time.time() - start_time) * 1000
            
            # Compute embedding (for search functionality)
            embedding = self.compressor._compute_embedding(text)
            
            compressed = CompressedMemory(
                memory_id=self.compressor._generate_id(),
                summary_hash="simple_compression",  # Special marker
                entities={},  # No entity extraction
                diff_data=compressed_data,
                embedding=embedding,
                compression_metadata=CompressionMetadata(
                    original_size=original_size,
                    compressed_size=compressed_size,
                    compression_ratio=compression_ratio,
                    model_used="zstd-level9",
                    quality_score=1.0,  # Lossless compression
                    compression_time_ms=compression_time_ms,
                    compressed_at=datetime.now()
                ),
                original_fields=metadata or {}
            )
            
            logger.debug(
                f"Simple compression: {original_size} -> {compressed_size} bytes "
                f"({compression_ratio:.2f}x ratio)"
            )
            
            return compressed
            
        except Exception as e:
            logger.error(f"Simple compression failed: {e}")
            raise CompressionError(f"Simple compression failed: {e}") from e
    
    def _store_uncompressed(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> CompressedMemory:
        """
        Store text without any compression (Level 4 fallback)
        
        Args:
            text: Original text
            metadata: Additional metadata
            
        Returns:
            CompressedMemory: Uncompressed memory
            
        Requirements: 13.3
        """
        original_size = len(text.encode('utf-8'))
        
        # Store original text as "diff"
        diff_data = text.encode('utf-8')
        
        # Compute embedding
        embedding = self.compressor._compute_embedding(text)
        
        compressed = CompressedMemory(
            memory_id=self.compressor._generate_id(),
            summary_hash="",  # Empty for uncompressed
            entities={},  # No entities extracted
            diff_data=diff_data,
            embedding=embedding,
            compression_metadata=CompressionMetadata(
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,  # No compression
                model_used="uncompressed",
                quality_score=1.0,  # Perfect quality (lossless)
                compression_time_ms=0.0,
                compressed_at=datetime.now()
            ),
            original_fields=metadata or {}
        )
        
        logger.debug(f"Stored uncompressed: {original_size} bytes")
        
        return compressed
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """
        Get fallback statistics
        
        Returns:
            Dict: Statistics including success/failure counts for each level
        """
        total_attempts = sum(self.fallback_stats.values())
        
        stats = {
            **self.fallback_stats,
            'total_attempts': total_attempts
        }
        
        # Calculate success rates
        if total_attempts > 0:
            stats['cloud_api_success_rate'] = (
                self.fallback_stats['cloud_api_success'] / total_attempts
            )
            stats['local_model_success_rate'] = (
                self.fallback_stats['local_model_success'] / total_attempts
            )
            stats['simple_compression_rate'] = (
                self.fallback_stats['simple_compression_success'] / total_attempts
            )
            stats['direct_storage_rate'] = (
                self.fallback_stats['direct_storage'] / total_attempts
            )
        
        return stats
    
    def reset_stats(self):
        """Reset fallback statistics"""
        for key in self.fallback_stats:
            self.fallback_stats[key] = 0
        logger.info("Fallback statistics reset")

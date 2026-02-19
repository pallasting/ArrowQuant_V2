"""
Unified memory search interface supporting multiple search modes.

This module provides a high-level search API that combines semantic,
entity-based, and time-based search capabilities.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Any

from llm_compression.vector_search import VectorSearch, SearchResult
from llm_compression.arrow_storage import ArrowStorage
from llm_compression.logger import logger


class SearchMode(Enum):
    """
    Search mode enumeration.
    
    Attributes:
        SEMANTIC: Vector similarity search
        ENTITY: Entity-based filtering
        TIME: Time range queries
        HYBRID: Combination of semantic + filters
    """
    
    SEMANTIC = "semantic"
    ENTITY = "entity"
    TIME = "time"
    HYBRID = "hybrid"


class MemorySearch:
    """
    Unified memory search interface.
    
    Supports multiple search modes:
    - Semantic: Vector similarity search
    - Entity: Entity-based filtering
    - Time: Time range queries
    - Hybrid: Combination of above
    
    Example:
        >>> from llm_compression.vector_search import VectorSearch
        >>> from llm_compression.arrow_storage import ArrowStorage
        >>> from llm_compression.embedding_provider import get_default_provider
        >>> 
        >>> provider = get_default_provider()
        >>> storage = ArrowStorage("./data/memories.parquet")
        >>> vector_search = VectorSearch(provider, storage)
        >>> search = MemorySearch(vector_search, storage)
        >>> 
        >>> # Semantic search
        >>> results = search.search(
        ...     query="machine learning",
        ...     category="knowledge",
        ...     mode=SearchMode.SEMANTIC,
        ...     top_k=5
        ... )
        >>> 
        >>> # Hybrid search (semantic + time filter)
        >>> results = search.search(
        ...     query="recent updates",
        ...     category="knowledge",
        ...     mode=SearchMode.HYBRID,
        ...     top_k=5,
        ...     start_time=datetime(2024, 1, 1)
        ... )
    """
    
    def __init__(
        self,
        vector_search: VectorSearch,
        storage: ArrowStorage
    ):
        """
        Initialize MemorySearch.
        
        Args:
            vector_search: VectorSearch instance for semantic search
            storage: ArrowStorage instance for direct queries
        """
        self.vector_search = vector_search
        self.storage = storage
        logger.info("Initialized MemorySearch")
    
    def search(
        self,
        query: str,
        category: str,
        mode: SearchMode = SearchMode.SEMANTIC,
        top_k: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """
        Unified search interface.
        
        Args:
            query: Search query text
            category: Memory category to search
            mode: Search mode (SEMANTIC, ENTITY, TIME, HYBRID)
            top_k: Number of results to return
            **kwargs: Mode-specific parameters:
                - entity_type: Entity type for ENTITY mode
                - entity_value: Entity value for ENTITY mode
                - start_time: Start time for TIME/HYBRID mode
                - end_time: End time for TIME/HYBRID mode
                - threshold: Similarity threshold for SEMANTIC/HYBRID mode
            
        Returns:
            List of search results
            
        Example:
            >>> # Semantic search
            >>> results = search.search(
            ...     query="Python programming",
            ...     category="code",
            ...     mode=SearchMode.SEMANTIC,
            ...     top_k=5,
            ...     threshold=0.7
            ... )
            >>> 
            >>> # Entity search
            >>> results = search.search(
            ...     query="",  # Not used in entity mode
            ...     category="code",
            ...     mode=SearchMode.ENTITY,
            ...     entity_type="language",
            ...     entity_value="Python"
            ... )
            >>> 
            >>> # Time search
            >>> results = search.search(
            ...     query="",  # Not used in time mode
            ...     category="code",
            ...     mode=SearchMode.TIME,
            ...     start_time=datetime(2024, 1, 1),
            ...     end_time=datetime(2024, 12, 31)
            ... )
            >>> 
            >>> # Hybrid search
            >>> results = search.search(
            ...     query="machine learning",
            ...     category="knowledge",
            ...     mode=SearchMode.HYBRID,
            ...     top_k=5,
            ...     start_time=datetime(2024, 1, 1)
            ... )
        """
        logger.debug(
            f"Search: query='{query[:50]}...', category={category}, "
            f"mode={mode.value}, top_k={top_k}"
        )
        
        if mode == SearchMode.SEMANTIC:
            threshold = kwargs.get('threshold', 0.0)
            return self.vector_search.search(
                query, category, top_k, threshold
            )
        
        elif mode == SearchMode.ENTITY:
            entity_type = kwargs.get('entity_type')
            entity_value = kwargs.get('entity_value')
            
            if not entity_type or not entity_value:
                logger.error("Entity search requires entity_type and entity_value")
                return []
            
            return self._search_by_entity(
                category, entity_type, entity_value, top_k
            )
        
        elif mode == SearchMode.TIME:
            start_time = kwargs.get('start_time')
            end_time = kwargs.get('end_time')
            
            if not start_time and not end_time:
                logger.error("Time search requires start_time or end_time")
                return []
            
            return self._search_by_time(
                category, start_time, end_time, top_k
            )
        
        elif mode == SearchMode.HYBRID:
            # Combine semantic + entity/time filtering
            threshold = kwargs.get('threshold', 0.0)
            semantic_results = self.vector_search.search(
                query, category, top_k * 2, threshold
            )
            
            # Apply additional filters
            filtered = self._apply_filters(semantic_results, **kwargs)
            return filtered[:top_k]
        
        else:
            logger.error(f"Unknown search mode: {mode}")
            return []
    
    def _search_by_entity(
        self,
        category: str,
        entity_type: str,
        entity_value: str,
        top_k: int
    ) -> List[SearchResult]:
        """
        Search by entity.
        
        Args:
            category: Memory category
            entity_type: Type of entity (e.g., 'person', 'location')
            entity_value: Entity value to match
            top_k: Number of results
            
        Returns:
            List of search results
        """
        logger.debug(
            f"Entity search: category={category}, "
            f"type={entity_type}, value={entity_value}"
        )
        
        try:
            # Query storage by entity
            results = self.storage.query_by_entity(
                category=category,
                entity_type=entity_type,
                entity_value=entity_value,
                limit=top_k
            )
            
            return [
                SearchResult(
                    memory_id=r['memory_id'],
                    similarity=1.0,  # Exact match
                    memory=r
                )
                for r in results
            ]
            
        except AttributeError:
            # Fallback if query_by_entity not implemented
            logger.warning(
                "query_by_entity not implemented in storage, "
                "using generic query"
            )
            results = self.storage.query(
                category=category,
                filters={entity_type: entity_value},
                limit=top_k
            )
            
            return [
                SearchResult(
                    memory_id=r.get('memory_id', str(i)),
                    similarity=1.0,
                    memory=r
                )
                for i, r in enumerate(results)
            ]
    
    def _search_by_time(
        self,
        category: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        top_k: int
    ) -> List[SearchResult]:
        """
        Search by time range.
        
        Args:
            category: Memory category
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            top_k: Number of results
            
        Returns:
            List of search results
        """
        logger.debug(
            f"Time search: category={category}, "
            f"start={start_time}, end={end_time}"
        )
        
        try:
            # Query storage by time range
            results = self.storage.query_by_time_range(
                category=category,
                start_time=start_time,
                end_time=end_time,
                limit=top_k
            )
            
            return [
                SearchResult(
                    memory_id=r['memory_id'],
                    similarity=1.0,
                    memory=r
                )
                for r in results
            ]
            
        except AttributeError:
            # Fallback if query_by_time_range not implemented
            logger.warning(
                "query_by_time_range not implemented in storage, "
                "using generic query with filters"
            )
            
            filters = {}
            if start_time:
                filters['timestamp_gte'] = start_time
            if end_time:
                filters['timestamp_lte'] = end_time
            
            results = self.storage.query(
                category=category,
                filters=filters,
                limit=top_k
            )
            
            return [
                SearchResult(
                    memory_id=r.get('memory_id', str(i)),
                    similarity=1.0,
                    memory=r
                )
                for i, r in enumerate(results)
            ]
    
    def _apply_filters(
        self,
        results: List[SearchResult],
        **kwargs
    ) -> List[SearchResult]:
        """
        Apply additional filters to search results.
        
        Args:
            results: Initial search results
            **kwargs: Filter parameters
            
        Returns:
            Filtered results
        """
        filtered = results
        
        # Time filter
        start_time = kwargs.get('start_time')
        end_time = kwargs.get('end_time')
        
        if start_time or end_time:
            filtered = [
                r for r in filtered
                if self._matches_time_filter(r, start_time, end_time)
            ]
        
        # Entity filter
        entity_type = kwargs.get('entity_type')
        entity_value = kwargs.get('entity_value')
        
        if entity_type and entity_value:
            filtered = [
                r for r in filtered
                if self._matches_entity_filter(r, entity_type, entity_value)
            ]
        
        logger.debug(
            f"Applied filters: {len(results)} -> {len(filtered)} results"
        )
        return filtered
    
    def _matches_time_filter(
        self,
        result: SearchResult,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> bool:
        """Check if result matches time filter."""
        if result.memory is None:
            return True
        
        if not hasattr(result.memory, 'timestamp') and 'timestamp' not in result.memory:
            return True
        
        timestamp = (
            result.memory.timestamp
            if hasattr(result.memory, 'timestamp')
            else result.memory.get('timestamp')
        )
        
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return True
        
        if start_time and timestamp < start_time:
            return False
        if end_time and timestamp > end_time:
            return False
        
        return True
    
    def _matches_entity_filter(
        self,
        result: SearchResult,
        entity_type: str,
        entity_value: str
    ) -> bool:
        """Check if result matches entity filter."""
        memory = result.memory
        
        if memory is None:
            return False
        
        # Check if entity_type exists in memory
        if hasattr(memory, entity_type):
            value = getattr(memory, entity_type)
        elif isinstance(memory, dict) and entity_type in memory:
            value = memory[entity_type]
        else:
            return False
        
        # Match entity value
        return str(value).lower() == str(entity_value).lower()

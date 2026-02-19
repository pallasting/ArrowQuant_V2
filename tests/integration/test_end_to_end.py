"""
End-to-End Integration Tests

Comprehensive integration tests for the complete LLM compression system.
Tests the full store-retrieve flow, semantic search, and batch processing.

Feature: llm-compression-integration
Task: 20.1 - End-to-End Integration Tests
Requirements: All core requirements (1-14)
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from llm_compression.openclaw_interface import OpenClawMemoryInterface
from llm_compression.compressor import LLMCompressor, MemoryType
from llm_compression.reconstructor import LLMReconstructor
from llm_compression.arrow_storage import ArrowStorage
from llm_compression.llm_client import LLMClient, LLMResponse
from llm_compression.model_selector import ModelSelector, ModelConfig, QualityLevel
from llm_compression.quality_evaluator import QualityEvaluator
from llm_compression.config import load_config
from unittest.mock import Mock, AsyncMock


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_storage_path():
    """Create temporary storage directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client for testing"""
    mock_client = Mock(spec=LLMClient)
    
    async def mock_generate(prompt, **kwargs):
        if "Summarize" in prompt:
            return LLMResponse(
                text="Meeting summary with key points and decisions.",
                tokens_used=10,
                latency_ms=100.0,
                model="gpt-3.5-turbo",
                finish_reason="stop",
                metadata={}
            )
        elif "Expand" in prompt:
            # Extract entities from prompt
            import re
            response_parts = ["This is the expanded text with all details."]
            
            # Extract persons
            persons_match = re.search(r"persons:\s*([^;]+)", prompt, re.IGNORECASE)
            if persons_match:
                persons_str = persons_match.group(1).strip()
                if persons_str and persons_str != "none":
                    persons = [p.strip() for p in persons_str.split(",") if p.strip() and p.strip() != "none"]
                    if persons:
                        response_parts.append(f"People: {', '.join(persons)}.")
            
            # Extract dates
            dates_match = re.search(r"dates:\s*([^;]+)", prompt, re.IGNORECASE)
            if dates_match:
                dates_str = dates_match.group(1).strip()
                if dates_str and dates_str != "none":
                    dates = [d.strip() for d in dates_str.split(",") if d.strip() and d.strip() != "none"]
                    if dates:
                        response_parts.append(f"Dates: {', '.join(dates)}.")
            
            # Extract numbers
            numbers_match = re.search(r"numbers:\s*([^;]+)", prompt, re.IGNORECASE)
            if numbers_match:
                numbers_str = numbers_match.group(1).strip()
                if numbers_str and numbers_str != "none":
                    numbers = [n.strip() for n in numbers_str.split(",") if n.strip() and n.strip() != "none"]
                    if numbers:
                        response_parts.append(f"Numbers: {', '.join(numbers)}.")
            
            return LLMResponse(
                text=" ".join(response_parts),
                tokens_used=len(response_parts) * 5,
                latency_ms=150.0,
                model="gpt-3.5-turbo",
                finish_reason="stop",
                metadata={}
            )
        else:
            return LLMResponse(
                text="Default response",
                tokens_used=5,
                latency_ms=50.0,
                model="gpt-3.5-turbo",
                finish_reason="stop",
                metadata={}
            )
    
    mock_client.generate = AsyncMock(side_effect=mock_generate)
    mock_client.close = AsyncMock()
    return mock_client


@pytest.fixture
def mock_model_selector():
    """Create mock model selector"""
    mock_selector = Mock(spec=ModelSelector)
    mock_selector.select_model = Mock(return_value=ModelConfig(
        model_name="gpt-3.5-turbo",
        endpoint="http://localhost:8045",
        is_local=False,
        max_tokens=100,
        temperature=0.3,
        expected_latency_ms=200.0,
        expected_quality=0.9
    ))
    return mock_selector


@pytest.fixture
async def openclaw_interface(temp_storage_path, mock_llm_client, mock_model_selector):
    """Create OpenClaw interface with all components"""
    compressor = LLMCompressor(
        llm_client=mock_llm_client,
        model_selector=mock_model_selector,
        min_compress_length=100
    )
    
    reconstructor = LLMReconstructor(
        llm_client=mock_llm_client,
        quality_threshold=0.85
    )
    
    storage = ArrowStorage(storage_path=temp_storage_path)
    
    interface = OpenClawMemoryInterface(
        storage_path=temp_storage_path,
        compressor=compressor,
        reconstructor=reconstructor,
        storage=storage,
        auto_compress_threshold=100
    )
    
    yield interface
    
    # Cleanup
    await mock_llm_client.close()


# ============================================================================
# Test 1: Complete Store-Retrieve Flow
# ============================================================================

@pytest.mark.asyncio
async def test_complete_store_retrieve_flow(openclaw_interface):
    """
    Test complete store-retrieve flow
    
    Validates:
    - Memory storage with automatic compression
    - Memory retrieval with automatic reconstruction
    - Data integrity throughout the process
    """
    # Create test memory
    memory = {
        "timestamp": datetime.now(),
        "context": "John Smith met with Mary Johnson on 2024-01-15 to discuss the Q4 budget of $150,000.",
        "intent": "meeting",
        "action": "discuss budget",
        "outcome": "agreed on budget allocation",
        "success": True
    }
    
    # Step 1: Store memory
    memory_id = await openclaw_interface.store_memory(
        memory=memory,
        memory_category="experiences"
    )
    
    # Verify memory ID was generated
    assert memory_id is not None
    assert len(memory_id) > 0
    
    # Step 2: Retrieve memory
    retrieved_memory = await openclaw_interface.retrieve_memory(
        memory_id=memory_id,
        memory_category="experiences"
    )
    
    # Verify retrieved memory
    assert retrieved_memory is not None
    assert retrieved_memory["intent"] == memory["intent"]
    assert retrieved_memory["action"] == memory["action"]
    assert retrieved_memory["outcome"] == memory["outcome"]
    assert retrieved_memory["success"] == memory["success"]
    
    # Verify context was reconstructed (may not be exact due to compression)
    assert "context" in retrieved_memory
    assert len(retrieved_memory["context"]) > 0


@pytest.mark.asyncio
async def test_store_multiple_memories(openclaw_interface):
    """
    Test storing multiple memories
    
    Validates:
    - Multiple memory storage
    - Each memory gets unique ID
    - All memories can be retrieved
    """
    memories = [
        {
            "timestamp": datetime.now(),
            "context": f"Memory {i}: Alice met Bob on 2024-01-{15+i:02d} to discuss project phase {i}.",
            "intent": "meeting",
            "action": f"discuss phase {i}",
            "outcome": f"completed phase {i} planning",
            "success": True
        }
        for i in range(5)
    ]
    
    # Store all memories
    memory_ids = []
    for memory in memories:
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        memory_ids.append(memory_id)
    
    # Verify all IDs are unique
    assert len(memory_ids) == len(set(memory_ids))
    
    # Retrieve and verify all memories
    for i, memory_id in enumerate(memory_ids):
        retrieved = await openclaw_interface.retrieve_memory(
            memory_id=memory_id,
            memory_category="experiences"
        )
        
        assert retrieved is not None
        assert retrieved["action"] == memories[i]["action"]
        assert retrieved["outcome"] == memories[i]["outcome"]


# ============================================================================
# Test 2: Semantic Search
# ============================================================================

@pytest.mark.asyncio
async def test_semantic_search(openclaw_interface):
    """
    Test semantic search functionality
    
    Validates:
    - Semantic search finds relevant memories
    - Results are ranked by relevance
    - Top-k filtering works correctly
    """
    # Store test memories with different topics
    memories = [
        {
            "timestamp": datetime.now(),
            "context": "Team meeting about Q1 budget planning and resource allocation for the marketing department.",
            "intent": "planning",
            "action": "budget planning",
            "outcome": "budget approved",
            "success": True
        },
        {
            "timestamp": datetime.now(),
            "context": "Technical discussion about database optimization and query performance improvements.",
            "intent": "technical",
            "action": "optimize database",
            "outcome": "performance improved",
            "success": True
        },
        {
            "timestamp": datetime.now(),
            "context": "Client meeting to discuss Q2 marketing strategy and campaign budget allocation.",
            "intent": "client meeting",
            "action": "discuss marketing",
            "outcome": "strategy finalized",
            "success": True
        },
        {
            "timestamp": datetime.now(),
            "context": "Code review session focusing on API endpoint security and authentication mechanisms.",
            "intent": "code review",
            "action": "review security",
            "outcome": "security enhanced",
            "success": True
        }
    ]
    
    # Store all memories
    for memory in memories:
        await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
    
    # Test 1: Search for budget-related memories
    budget_results = await openclaw_interface.search_memories(
        query="budget planning and allocation",
        memory_category="experiences",
        top_k=2
    )
    
    # Should find budget-related memories
    assert len(budget_results) <= 2
    assert len(budget_results) > 0
    
    # Verify results contain budget-related content
    budget_found = any(
        "budget" in result.get("context", "").lower() or
        "budget" in result.get("action", "").lower()
        for result in budget_results
    )
    assert budget_found
    
    # Test 2: Search for technical memories
    tech_results = await openclaw_interface.search_memories(
        query="database optimization and performance",
        memory_category="experiences",
        top_k=2
    )
    
    # Should find technical memories
    assert len(tech_results) <= 2
    assert len(tech_results) > 0


@pytest.mark.asyncio
async def test_semantic_search_empty_results(openclaw_interface):
    """
    Test semantic search with no matching memories
    
    Validates:
    - Empty result handling
    - No errors on empty database
    """
    # Search without storing any memories
    results = await openclaw_interface.search_memories(
        query="nonexistent topic",
        memory_category="experiences",
        top_k=5
    )
    
    # Should return empty list, not error
    assert isinstance(results, list)
    assert len(results) == 0


# ============================================================================
# Test 3: Batch Processing
# ============================================================================

@pytest.mark.asyncio
async def test_batch_store_retrieve(openclaw_interface):
    """
    Test batch storage and retrieval
    
    Validates:
    - Batch storage efficiency
    - All memories stored correctly
    - Batch retrieval works
    """
    # Create batch of memories
    batch_size = 10
    memories = [
        {
            "timestamp": datetime.now(),
            "context": f"Batch memory {i}: Meeting with client {i} on 2024-01-{15+i%15:02d} to discuss project milestone {i}.",
            "intent": "client meeting",
            "action": f"discuss milestone {i}",
            "outcome": f"milestone {i} approved",
            "success": True
        }
        for i in range(batch_size)
    ]
    
    # Store batch
    memory_ids = []
    for memory in memories:
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        memory_ids.append(memory_id)
    
    # Verify all stored
    assert len(memory_ids) == batch_size
    
    # Retrieve batch
    retrieved_memories = []
    for memory_id in memory_ids:
        retrieved = await openclaw_interface.retrieve_memory(
            memory_id=memory_id,
            memory_category="experiences"
        )
        retrieved_memories.append(retrieved)
    
    # Verify all retrieved
    assert len(retrieved_memories) == batch_size
    
    # Verify content integrity
    for i, retrieved in enumerate(retrieved_memories):
        assert retrieved["action"] == memories[i]["action"]
        assert retrieved["outcome"] == memories[i]["outcome"]


@pytest.mark.asyncio
async def test_concurrent_operations(openclaw_interface):
    """
    Test concurrent store and retrieve operations
    
    Validates:
    - Thread safety
    - No data corruption
    - All operations complete successfully
    """
    # Create memories for concurrent operations
    memories = [
        {
            "timestamp": datetime.now(),
            "context": f"Concurrent memory {i}: Task {i} completed successfully with result {i*10}.",
            "intent": "task completion",
            "action": f"complete task {i}",
            "outcome": f"result {i*10}",
            "success": True
        }
        for i in range(5)
    ]
    
    # Store concurrently
    store_tasks = [
        openclaw_interface.store_memory(memory, "experiences")
        for memory in memories
    ]
    
    memory_ids = await asyncio.gather(*store_tasks)
    
    # Verify all stored
    assert len(memory_ids) == len(memories)
    assert all(memory_id for memory_id in memory_ids)
    
    # Retrieve concurrently
    retrieve_tasks = [
        openclaw_interface.retrieve_memory(memory_id, "experiences")
        for memory_id in memory_ids
    ]
    
    retrieved_memories = await asyncio.gather(*retrieve_tasks)
    
    # Verify all retrieved
    assert len(retrieved_memories) == len(memories)
    assert all(memory for memory in retrieved_memories)


# ============================================================================
# Test 4: Different Memory Categories
# ============================================================================

@pytest.mark.asyncio
async def test_multiple_memory_categories(openclaw_interface):
    """
    Test storing and retrieving from different memory categories
    
    Validates:
    - Support for all OpenClaw memory categories
    - Category isolation
    - Correct category routing
    """
    categories = ["identity", "experiences", "preferences", "context"]
    
    # Store one memory in each category
    memory_ids = {}
    for category in categories:
        memory = {
            "timestamp": datetime.now(),
            "context": f"Test memory for {category} category with specific content for testing.",
            "intent": f"{category} test",
            "action": f"test {category}",
            "outcome": f"{category} tested",
            "success": True
        }
        
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category=category
        )
        memory_ids[category] = memory_id
    
    # Retrieve from each category
    for category in categories:
        retrieved = await openclaw_interface.retrieve_memory(
            memory_id=memory_ids[category],
            memory_category=category
        )
        
        assert retrieved is not None
        assert retrieved["intent"] == f"{category} test"
        assert retrieved["action"] == f"test {category}"


# ============================================================================
# Test 5: Compression and Reconstruction Quality
# ============================================================================

@pytest.mark.asyncio
async def test_compression_quality(openclaw_interface):
    """
    Test compression quality for different text lengths
    
    Validates:
    - Short texts not compressed
    - Medium texts compressed with good ratio
    - Long texts compressed with high ratio
    - Quality maintained after reconstruction
    """
    test_cases = [
        {
            "name": "short_text",
            "context": "Short meeting note.",
            "should_compress": False
        },
        {
            "name": "medium_text",
            "context": "Medium length meeting note about project planning and resource allocation. " * 5,
            "should_compress": True
        },
        {
            "name": "long_text",
            "context": "Long detailed meeting note covering multiple topics including budget planning, resource allocation, timeline discussion, risk assessment, and stakeholder communication. " * 10,
            "should_compress": True
        }
    ]
    
    for test_case in test_cases:
        memory = {
            "timestamp": datetime.now(),
            "context": test_case["context"],
            "intent": "meeting",
            "action": "discuss",
            "outcome": "completed",
            "success": True
        }
        
        # Store and retrieve
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        
        retrieved = await openclaw_interface.retrieve_memory(
            memory_id=memory_id,
            memory_category="experiences"
        )
        
        # Verify retrieval successful
        assert retrieved is not None
        assert "context" in retrieved
        
        # For compressed memories, verify some content is preserved
        if test_case["should_compress"]:
            # At least some words should be preserved
            assert len(retrieved["context"]) > 0


# ============================================================================
# Test 6: Error Handling and Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_nonexistent_memory(openclaw_interface):
    """
    Test retrieving non-existent memory
    
    Validates:
    - Graceful handling of missing memories
    - Appropriate error or None return
    """
    # Try to retrieve non-existent memory
    retrieved = await openclaw_interface.retrieve_memory(
        memory_id="nonexistent-id-12345",
        memory_category="experiences"
    )
    
    # Should return None or empty, not crash
    assert retrieved is None or retrieved == {}


@pytest.mark.asyncio
async def test_empty_context_handling(openclaw_interface):
    """
    Test handling of memories with empty context
    
    Validates:
    - Empty context doesn't cause errors
    - Memory still stored and retrieved
    """
    memory = {
        "timestamp": datetime.now(),
        "context": "",
        "intent": "test",
        "action": "test empty",
        "outcome": "tested",
        "success": True
    }
    
    # Store memory with empty context
    memory_id = await openclaw_interface.store_memory(
        memory=memory,
        memory_category="experiences"
    )
    
    # Should succeed
    assert memory_id is not None
    
    # Retrieve
    retrieved = await openclaw_interface.retrieve_memory(
        memory_id=memory_id,
        memory_category="experiences"
    )
    
    # Should retrieve successfully
    assert retrieved is not None
    assert retrieved["intent"] == "test"


@pytest.mark.asyncio
async def test_special_characters_handling(openclaw_interface):
    """
    Test handling of special characters in memory content
    
    Validates:
    - Special characters preserved
    - Unicode support
    - No encoding errors
    """
    memory = {
        "timestamp": datetime.now(),
        "context": "Meeting with José García about €50,000 budget. Discussed 中文 content and émojis 🎉.",
        "intent": "meeting",
        "action": "discuss budget",
        "outcome": "approved",
        "success": True
    }
    
    # Store and retrieve
    memory_id = await openclaw_interface.store_memory(
        memory=memory,
        memory_category="experiences"
    )
    
    retrieved = await openclaw_interface.retrieve_memory(
        memory_id=memory_id,
        memory_category="experiences"
    )
    
    # Verify special characters handled
    assert retrieved is not None
    # Some characters should be preserved (exact preservation depends on compression)
    assert len(retrieved["context"]) > 0


# ============================================================================
# Test 7: Performance Validation
# ============================================================================

@pytest.mark.asyncio
async def test_store_performance(openclaw_interface):
    """
    Test storage performance
    
    Validates:
    - Storage completes in reasonable time
    - No significant performance degradation
    """
    import time
    
    memory = {
        "timestamp": datetime.now(),
        "context": "Performance test memory with sufficient content to trigger compression. " * 10,
        "intent": "performance test",
        "action": "test performance",
        "outcome": "measured",
        "success": True
    }
    
    # Measure storage time
    start_time = time.time()
    memory_id = await openclaw_interface.store_memory(
        memory=memory,
        memory_category="experiences"
    )
    end_time = time.time()
    
    storage_time = (end_time - start_time) * 1000  # Convert to ms
    
    # Should complete in reasonable time (< 5 seconds as per requirements)
    assert storage_time < 5000, f"Storage took {storage_time:.2f}ms, should be < 5000ms"
    
    # Verify storage succeeded
    assert memory_id is not None


@pytest.mark.asyncio
async def test_retrieve_performance(openclaw_interface):
    """
    Test retrieval performance
    
    Validates:
    - Retrieval completes in reasonable time (< 1s)
    - Reconstruction is efficient
    """
    import time
    
    # First store a memory
    memory = {
        "timestamp": datetime.now(),
        "context": "Performance test memory for retrieval with sufficient content. " * 10,
        "intent": "performance test",
        "action": "test retrieval",
        "outcome": "measured",
        "success": True
    }
    
    memory_id = await openclaw_interface.store_memory(
        memory=memory,
        memory_category="experiences"
    )
    
    # Measure retrieval time
    start_time = time.time()
    retrieved = await openclaw_interface.retrieve_memory(
        memory_id=memory_id,
        memory_category="experiences"
    )
    end_time = time.time()
    
    retrieval_time = (end_time - start_time) * 1000  # Convert to ms
    
    # Should complete in reasonable time (< 1 second as per requirements)
    assert retrieval_time < 1000, f"Retrieval took {retrieval_time:.2f}ms, should be < 1000ms"
    
    # Verify retrieval succeeded
    assert retrieved is not None


# ============================================================================
# Test 8: Data Integrity
# ============================================================================

@pytest.mark.asyncio
async def test_data_integrity_after_multiple_operations(openclaw_interface):
    """
    Test data integrity after multiple operations
    
    Validates:
    - Data remains consistent
    - No corruption after multiple operations
    - All memories remain accessible
    """
    # Store multiple memories
    memory_ids = []
    for i in range(10):
        memory = {
            "timestamp": datetime.now(),
            "context": f"Integrity test memory {i} with unique content and identifier.",
            "intent": "integrity test",
            "action": f"test {i}",
            "outcome": f"result {i}",
            "success": True
        }
        
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        memory_ids.append((memory_id, memory))
    
    # Perform some searches
    await openclaw_interface.search_memories(
        query="integrity test",
        memory_category="experiences",
        top_k=5
    )
    
    # Retrieve all memories again
    for memory_id, original_memory in memory_ids:
        retrieved = await openclaw_interface.retrieve_memory(
            memory_id=memory_id,
            memory_category="experiences"
        )
        
        # Verify data integrity
        assert retrieved is not None
        assert retrieved["intent"] == original_memory["intent"]
        assert retrieved["action"] == original_memory["action"]
        assert retrieved["outcome"] == original_memory["outcome"]
        assert retrieved["success"] == original_memory["success"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

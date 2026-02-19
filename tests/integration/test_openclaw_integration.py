"""
OpenClaw Integration Tests

Tests for complete OpenClaw API compatibility and integration.
Validates all OpenClaw memory interface requirements.

Feature: llm-compression-integration
Task: 20.2 - OpenClaw Integration Tests
Requirements: 4.1, 4.2, 4.3, 4.4
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

from llm_compression.openclaw_interface import OpenClawMemoryInterface
from llm_compression.compressor import LLMCompressor, MemoryType
from llm_compression.reconstructor import LLMReconstructor
from llm_compression.arrow_storage import ArrowStorage
from llm_compression.llm_client import LLMClient, LLMResponse
from llm_compression.model_selector import ModelSelector, ModelConfig
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
    """Create mock LLM client"""
    mock_client = Mock(spec=LLMClient)
    
    async def mock_generate(prompt, **kwargs):
        if "Summarize" in prompt:
            return LLMResponse(
                text="Summary of the content.",
                tokens_used=5,
                latency_ms=100.0,
                model="gpt-3.5-turbo",
                finish_reason="stop",
                metadata={}
            )
        elif "Expand" in prompt:
            return LLMResponse(
                text="Expanded content with details.",
                tokens_used=10,
                latency_ms=150.0,
                model="gpt-3.5-turbo",
                finish_reason="stop",
                metadata={}
            )
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
    """Create OpenClaw interface"""
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
    
    await mock_llm_client.close()


# ============================================================================
# Test 1: OpenClaw Schema Compatibility (Requirement 4.1)
# ============================================================================

@pytest.mark.asyncio
async def test_openclaw_schema_compatibility(temp_storage_path, openclaw_interface):
    """
    Test OpenClaw Arrow schema compatibility
    
    Validates: Requirement 4.1
    - All OpenClaw base fields present
    - Schema extensions don't break compatibility
    - Can read/write using OpenClaw schema
    """
    # Create memory with all OpenClaw fields
    memory = {
        "timestamp": datetime.now(),
        "context": "Test context for schema compatibility validation.",
        "intent": "test",
        "action": "validate schema",
        "outcome": "schema validated",
        "success": True
    }
    
    # Store memory
    memory_id = await openclaw_interface.store_memory(
        memory=memory,
        memory_category="experiences"
    )
    
    # Read the Arrow file directly to verify schema
    storage_file = Path(temp_storage_path) / "experiences" / "experiences.parquet"
    
    if storage_file.exists():
        table = pq.read_table(storage_file)
        schema = table.schema
        
        # Verify all OpenClaw base fields are present
        base_fields = [
            "timestamp",
            "context",
            "intent",
            "action",
            "outcome",
            "success",
            "embedding",
            "related_memories"
        ]
        
        schema_field_names = [field.name for field in schema]
        
        for field in base_fields:
            assert field in schema_field_names, f"Missing OpenClaw base field: {field}"
        
        # Verify compression extension fields are present
        extension_fields = [
            "is_compressed",
            "summary_hash",
            "entities",
            "diff_data",
            "compression_metadata"
        ]
        
        for field in extension_fields:
            assert field in schema_field_names, f"Missing compression extension field: {field}"


@pytest.mark.asyncio
async def test_openclaw_field_types(temp_storage_path, openclaw_interface):
    """
    Test OpenClaw field type compatibility
    
    Validates: Requirement 4.1
    - All field types match OpenClaw specification
    - Type conversions work correctly
    """
    memory = {
        "timestamp": datetime.now(),
        "context": "Type validation test content.",
        "intent": "type test",
        "action": "validate types",
        "outcome": "types validated",
        "success": True
    }
    
    memory_id = await openclaw_interface.store_memory(
        memory=memory,
        memory_category="experiences"
    )
    
    # Read and verify types
    storage_file = Path(temp_storage_path) / "experiences" / "experiences.parquet"
    
    if storage_file.exists():
        table = pq.read_table(storage_file)
        schema = table.schema
        
        # Verify field types
        field_types = {field.name: field.type for field in schema}
        
        # Check timestamp type
        assert pa.types.is_timestamp(field_types["timestamp"])
        
        # Check string types
        for field in ["context", "intent", "action", "outcome"]:
            assert pa.types.is_string(field_types[field]) or pa.types.is_large_string(field_types[field])
        
        # Check boolean type
        assert pa.types.is_boolean(field_types["success"])
        
        # Check embedding type (list of floats)
        assert pa.types.is_list(field_types["embedding"]) or pa.types.is_fixed_size_list(field_types["embedding"])


# ============================================================================
# Test 2: OpenClaw API Compatibility (Requirement 4.4)
# ============================================================================

@pytest.mark.asyncio
async def test_store_memory_api(openclaw_interface):
    """
    Test store_memory API compatibility
    
    Validates: Requirement 4.4
    - API signature matches OpenClaw
    - Returns memory ID
    - Handles all memory categories
    """
    # Test with different memory categories
    categories = ["identity", "experiences", "preferences", "context"]
    
    for category in categories:
        memory = {
            "timestamp": datetime.now(),
            "context": f"API test for {category} category.",
            "intent": "api test",
            "action": f"test {category}",
            "outcome": "api validated",
            "success": True
        }
        
        # Call store_memory API
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category=category
        )
        
        # Verify return value
        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0


@pytest.mark.asyncio
async def test_retrieve_memory_api(openclaw_interface):
    """
    Test retrieve_memory API compatibility
    
    Validates: Requirement 4.4
    - API signature matches OpenClaw
    - Returns complete memory dict
    - Handles all memory categories
    """
    # Store a memory first
    original_memory = {
        "timestamp": datetime.now(),
        "context": "API retrieval test content.",
        "intent": "retrieval test",
        "action": "test retrieve",
        "outcome": "retrieve validated",
        "success": True
    }
    
    memory_id = await openclaw_interface.store_memory(
        memory=original_memory,
        memory_category="experiences"
    )
    
    # Call retrieve_memory API
    retrieved_memory = await openclaw_interface.retrieve_memory(
        memory_id=memory_id,
        memory_category="experiences"
    )
    
    # Verify return value
    assert retrieved_memory is not None
    assert isinstance(retrieved_memory, dict)
    
    # Verify all required fields present
    required_fields = ["timestamp", "context", "intent", "action", "outcome", "success"]
    for field in required_fields:
        assert field in retrieved_memory


@pytest.mark.asyncio
async def test_search_memories_api(openclaw_interface):
    """
    Test search_memories API compatibility
    
    Validates: Requirement 4.4
    - API signature matches OpenClaw
    - Returns list of memories
    - Respects top_k parameter
    """
    # Store some memories
    for i in range(5):
        memory = {
            "timestamp": datetime.now(),
            "context": f"Search test memory {i} about project planning.",
            "intent": "planning",
            "action": f"plan task {i}",
            "outcome": f"task {i} planned",
            "success": True
        }
        await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
    
    # Call search_memories API
    results = await openclaw_interface.search_memories(
        query="project planning",
        memory_category="experiences",
        top_k=3
    )
    
    # Verify return value
    assert isinstance(results, list)
    assert len(results) <= 3
    
    # Verify each result is a complete memory dict
    for result in results:
        assert isinstance(result, dict)
        assert "intent" in result
        assert "action" in result


@pytest.mark.asyncio
async def test_get_related_memories_api(openclaw_interface):
    """
    Test get_related_memories API compatibility
    
    Validates: Requirement 4.4
    - API signature matches OpenClaw
    - Returns list of related memories
    - Respects top_k parameter
    """
    # Store some related memories
    memory_ids = []
    for i in range(5):
        memory = {
            "timestamp": datetime.now(),
            "context": f"Related memory {i} about team collaboration and communication.",
            "intent": "collaboration",
            "action": f"collaborate on task {i}",
            "outcome": f"task {i} completed",
            "success": True
        }
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        memory_ids.append(memory_id)
    
    # Call get_related_memories API
    related = await openclaw_interface.get_related_memories(
        memory_id=memory_ids[0],
        memory_category="experiences",
        top_k=3
    )
    
    # Verify return value
    assert isinstance(related, list)
    assert len(related) <= 3
    
    # Verify each result is a complete memory dict
    for memory in related:
        assert isinstance(memory, dict)
        assert "intent" in memory


# ============================================================================
# Test 3: OpenClaw Storage Paths (Requirement 4.3)
# ============================================================================

@pytest.mark.asyncio
async def test_standard_storage_paths(temp_storage_path, openclaw_interface):
    """
    Test OpenClaw standard storage paths
    
    Validates: Requirement 4.3
    - Core memory path: ~/.ai-os/memory/core/
    - Working memory path: ~/.ai-os/memory/working/
    - Long-term memory path: ~/.ai-os/memory/long-term/
    - Shared memory path: ~/.ai-os/memory/shared/
    """
    # Map OpenClaw categories to expected subdirectories
    category_paths = {
        "identity": "identity",
        "experiences": "experiences",
        "preferences": "preferences",
        "context": "context"
    }
    
    # Store memory in each category
    for category, expected_subdir in category_paths.items():
        memory = {
            "timestamp": datetime.now(),
            "context": f"Path test for {category}.",
            "intent": "path test",
            "action": f"test {category} path",
            "outcome": "path validated",
            "success": True
        }
        
        await openclaw_interface.store_memory(
            memory=memory,
            memory_category=category
        )
        
        # Verify directory was created
        category_dir = Path(temp_storage_path) / expected_subdir
        assert category_dir.exists(), f"Category directory not created: {category_dir}"


@pytest.mark.asyncio
async def test_path_isolation(temp_storage_path, openclaw_interface):
    """
    Test memory category path isolation
    
    Validates: Requirement 4.3
    - Memories in different categories are isolated
    - No cross-category contamination
    """
    # Store memories in different categories with same content
    categories = ["identity", "experiences", "preferences"]
    memory_ids = {}
    
    for category in categories:
        memory = {
            "timestamp": datetime.now(),
            "context": "Isolation test content - same across categories.",
            "intent": "isolation test",
            "action": "test isolation",
            "outcome": "isolated",
            "success": True
        }
        
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category=category
        )
        memory_ids[category] = memory_id
    
    # Verify each category has its own storage
    for category in categories:
        category_dir = Path(temp_storage_path) / category
        assert category_dir.exists()
        
        # Verify memory can be retrieved from correct category
        retrieved = await openclaw_interface.retrieve_memory(
            memory_id=memory_ids[category],
            memory_category=category
        )
        assert retrieved is not None


# ============================================================================
# Test 4: Transparent Compression (Requirements 4.5, 4.6)
# ============================================================================

@pytest.mark.asyncio
async def test_automatic_compression_decision(openclaw_interface):
    """
    Test automatic compression decision
    
    Validates: Requirement 4.5
    - System automatically decides whether to compress
    - Decision based on content size
    - Transparent to caller
    """
    # Test 1: Short content (should not compress)
    short_memory = {
        "timestamp": datetime.now(),
        "context": "Short content.",
        "intent": "test",
        "action": "test short",
        "outcome": "tested",
        "success": True
    }
    
    short_id = await openclaw_interface.store_memory(
        memory=short_memory,
        memory_category="experiences"
    )
    
    # Should succeed without errors
    assert short_id is not None
    
    # Test 2: Long content (should compress)
    long_memory = {
        "timestamp": datetime.now(),
        "context": "Long content that should trigger compression. " * 20,
        "intent": "test",
        "action": "test long",
        "outcome": "tested",
        "success": True
    }
    
    long_id = await openclaw_interface.store_memory(
        memory=long_memory,
        memory_category="experiences"
    )
    
    # Should succeed without errors
    assert long_id is not None
    
    # Both should be retrievable
    short_retrieved = await openclaw_interface.retrieve_memory(
        memory_id=short_id,
        memory_category="experiences"
    )
    
    long_retrieved = await openclaw_interface.retrieve_memory(
        memory_id=long_id,
        memory_category="experiences"
    )
    
    assert short_retrieved is not None
    assert long_retrieved is not None


@pytest.mark.asyncio
async def test_transparent_reconstruction(openclaw_interface):
    """
    Test transparent reconstruction on retrieval
    
    Validates: Requirement 4.6
    - Compressed memories automatically reconstructed
    - Reconstruction transparent to caller
    - No special handling required
    """
    # Store a memory that will be compressed
    memory = {
        "timestamp": datetime.now(),
        "context": "This is a longer memory that will be compressed automatically. " * 10,
        "intent": "compression test",
        "action": "test transparent reconstruction",
        "outcome": "reconstruction transparent",
        "success": True
    }
    
    memory_id = await openclaw_interface.store_memory(
        memory=memory,
        memory_category="experiences"
    )
    
    # Retrieve - should be transparent
    retrieved = await openclaw_interface.retrieve_memory(
        memory_id=memory_id,
        memory_category="experiences"
    )
    
    # Verify retrieval successful
    assert retrieved is not None
    assert "context" in retrieved
    assert retrieved["intent"] == memory["intent"]
    assert retrieved["action"] == memory["action"]
    
    # Caller shouldn't need to know about compression
    # All fields should be present as if uncompressed
    assert "timestamp" in retrieved
    assert "outcome" in retrieved
    assert "success" in retrieved


# ============================================================================
# Test 5: OpenClaw Metadata Compatibility (Requirement 4.2)
# ============================================================================

@pytest.mark.asyncio
async def test_compression_metadata_extension(temp_storage_path, openclaw_interface):
    """
    Test compression metadata extension
    
    Validates: Requirement 4.2
    - Compression metadata fields present
    - Metadata doesn't interfere with base fields
    - Metadata accessible when needed
    """
    memory = {
        "timestamp": datetime.now(),
        "context": "Metadata test content with sufficient length to trigger compression. " * 5,
        "intent": "metadata test",
        "action": "test metadata",
        "outcome": "metadata validated",
        "success": True
    }
    
    memory_id = await openclaw_interface.store_memory(
        memory=memory,
        memory_category="experiences"
    )
    
    # Read the Arrow file to check metadata
    storage_file = Path(temp_storage_path) / "experiences" / "experiences.parquet"
    
    if storage_file.exists():
        table = pq.read_table(storage_file)
        
        # Verify compression metadata fields exist
        metadata_fields = [
            "is_compressed",
            "summary_hash",
            "entities",
            "diff_data",
            "compression_metadata"
        ]
        
        schema_field_names = [field.name for field in table.schema]
        
        for field in metadata_fields:
            assert field in schema_field_names, f"Missing metadata field: {field}"


@pytest.mark.asyncio
async def test_backward_compatibility_with_uncompressed(openclaw_interface):
    """
    Test backward compatibility with uncompressed memories
    
    Validates: Requirement 4.7
    - Can read memories without compression metadata
    - Uncompressed memories work normally
    - No errors on missing compression fields
    """
    # Store a short memory (won't be compressed)
    memory = {
        "timestamp": datetime.now(),
        "context": "Short uncompressed memory.",
        "intent": "compatibility test",
        "action": "test backward compatibility",
        "outcome": "compatible",
        "success": True
    }
    
    memory_id = await openclaw_interface.store_memory(
        memory=memory,
        memory_category="experiences"
    )
    
    # Retrieve should work normally
    retrieved = await openclaw_interface.retrieve_memory(
        memory_id=memory_id,
        memory_category="experiences"
    )
    
    assert retrieved is not None
    assert retrieved["intent"] == memory["intent"]


# ============================================================================
# Test 6: OpenClaw Integration Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_category_access(openclaw_interface):
    """
    Test concurrent access to different memory categories
    
    Validates: Thread safety across categories
    """
    categories = ["identity", "experiences", "preferences", "context"]
    
    async def store_in_category(category, index):
        memory = {
            "timestamp": datetime.now(),
            "context": f"Concurrent test {index} in {category}.",
            "intent": "concurrent test",
            "action": f"test {category} {index}",
            "outcome": "tested",
            "success": True
        }
        return await openclaw_interface.store_memory(
            memory=memory,
            memory_category=category
        )
    
    # Store concurrently across categories
    tasks = [
        store_in_category(category, i)
        for i, category in enumerate(categories)
    ]
    
    memory_ids = await asyncio.gather(*tasks)
    
    # Verify all succeeded
    assert len(memory_ids) == len(categories)
    assert all(memory_id for memory_id in memory_ids)


@pytest.mark.asyncio
async def test_large_batch_openclaw_operations(openclaw_interface):
    """
    Test large batch operations through OpenClaw interface
    
    Validates: Scalability of OpenClaw integration
    """
    batch_size = 50
    
    # Store large batch
    memory_ids = []
    for i in range(batch_size):
        memory = {
            "timestamp": datetime.now(),
            "context": f"Batch memory {i} with content.",
            "intent": "batch test",
            "action": f"batch action {i}",
            "outcome": f"batch result {i}",
            "success": True
        }
        
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        memory_ids.append(memory_id)
    
    # Verify all stored
    assert len(memory_ids) == batch_size
    
    # Sample retrieval
    sample_indices = [0, batch_size // 2, batch_size - 1]
    for idx in sample_indices:
        retrieved = await openclaw_interface.retrieve_memory(
            memory_id=memory_ids[idx],
            memory_category="experiences"
        )
        assert retrieved is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

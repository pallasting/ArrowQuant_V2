"""
Performance Requirement Tests

Tests to validate performance requirements are met:
- Compression latency < 5s
- Reconstruction latency < 1s  
- Throughput > 50/min

Feature: llm-compression-integration
Task: 20.3 - Performance Tests
Requirements: 6.5, 9.7
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List
import statistics

from llm_compression.openclaw_interface import OpenClawMemoryInterface
from llm_compression.compressor import LLMCompressor, MemoryType
from llm_compression.reconstructor import LLMReconstructor
from llm_compression.arrow_storage import ArrowStorage
from llm_compression.batch_processor import BatchProcessor
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
    """Create mock LLM client with realistic latencies"""
    mock_client = Mock(spec=LLMClient)
    
    async def mock_generate(prompt, **kwargs):
        # Simulate realistic latency
        await asyncio.sleep(0.05)  # 50ms latency
        
        if "Summarize" in prompt:
            return LLMResponse(
                text="Summary of the content with key points.",
                tokens_used=10,
                latency_ms=50.0,
                model="gpt-3.5-turbo",
                finish_reason="stop",
                metadata={}
            )
        elif "Expand" in prompt:
            # Extract entities from prompt
            import re
            response_parts = ["Expanded content with details."]
            
            persons_match = re.search(r"persons:\s*([^;]+)", prompt, re.IGNORECASE)
            if persons_match:
                persons_str = persons_match.group(1).strip()
                if persons_str and persons_str != "none":
                    persons = [p.strip() for p in persons_str.split(",") if p.strip() and p.strip() != "none"]
                    if persons:
                        response_parts.append(f"People: {', '.join(persons)}.")
            
            dates_match = re.search(r"dates:\s*([^;]+)", prompt, re.IGNORECASE)
            if dates_match:
                dates_str = dates_match.group(1).strip()
                if dates_str and dates_str != "none":
                    dates = [d.strip() for d in dates_str.split(",") if d.strip() and d.strip() != "none"]
                    if dates:
                        response_parts.append(f"Dates: {', '.join(dates)}.")
            
            return LLMResponse(
                text=" ".join(response_parts),
                tokens_used=len(response_parts) * 5,
                latency_ms=50.0,
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
# Test 1: Compression Latency < 5s (Requirement 6.5)
# ============================================================================

@pytest.mark.asyncio
async def test_compression_latency_single_memory(openclaw_interface):
    """
    Test compression latency for single memory
    
    Validates: Requirement 6.5
    - Compression completes in < 5s
    - Measured for various text lengths
    """
    test_cases = [
        {
            "name": "medium_text",
            "context": "Medium length text for compression testing. " * 10,
            "max_latency_ms": 5000
        },
        {
            "name": "long_text",
            "context": "Long text for compression testing with more content. " * 30,
            "max_latency_ms": 5000
        },
        {
            "name": "very_long_text",
            "context": "Very long text for compression testing with extensive content. " * 50,
            "max_latency_ms": 5000
        }
    ]
    
    for test_case in test_cases:
        memory = {
            "timestamp": datetime.now(),
            "context": test_case["context"],
            "intent": "performance test",
            "action": "test compression latency",
            "outcome": "latency measured",
            "success": True
        }
        
        # Measure compression time
        start_time = time.time()
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Verify latency requirement
        assert latency_ms < test_case["max_latency_ms"], \
            f"{test_case['name']}: Compression took {latency_ms:.2f}ms, should be < {test_case['max_latency_ms']}ms"
        
        # Verify storage succeeded
        assert memory_id is not None
        
        print(f"{test_case['name']}: Compression latency = {latency_ms:.2f}ms")


@pytest.mark.asyncio
async def test_compression_latency_statistics(openclaw_interface):
    """
    Test compression latency statistics
    
    Validates: Requirement 6.5
    - Average latency < 5s
    - P95 latency < 5s
    - Consistent performance
    """
    num_samples = 20
    latencies = []
    
    for i in range(num_samples):
        memory = {
            "timestamp": datetime.now(),
            "context": f"Performance test memory {i} with sufficient content for compression. " * 15,
            "intent": "performance test",
            "action": f"test {i}",
            "outcome": f"result {i}",
            "success": True
        }
        
        start_time = time.time()
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        assert memory_id is not None
    
    # Calculate statistics
    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    max_latency = max(latencies)
    
    print(f"\nCompression Latency Statistics ({num_samples} samples):")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  Median: {median_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")
    
    # Verify requirements
    assert avg_latency < 5000, f"Average latency {avg_latency:.2f}ms should be < 5000ms"
    assert p95_latency < 5000, f"P95 latency {p95_latency:.2f}ms should be < 5000ms"


# ============================================================================
# Test 2: Reconstruction Latency < 1s (Requirement 6.5)
# ============================================================================

@pytest.mark.asyncio
async def test_reconstruction_latency_single_memory(openclaw_interface):
    """
    Test reconstruction latency for single memory
    
    Validates: Requirement 6.5
    - Reconstruction completes in < 1s
    - Measured for various compressed sizes
    """
    test_cases = [
        {
            "name": "medium_compressed",
            "context": "Medium text for reconstruction testing. " * 10,
            "max_latency_ms": 1000
        },
        {
            "name": "long_compressed",
            "context": "Long text for reconstruction testing with more content. " * 30,
            "max_latency_ms": 1000
        },
        {
            "name": "very_long_compressed",
            "context": "Very long text for reconstruction testing with extensive content. " * 50,
            "max_latency_ms": 1000
        }
    ]
    
    for test_case in test_cases:
        # First store the memory
        memory = {
            "timestamp": datetime.now(),
            "context": test_case["context"],
            "intent": "reconstruction test",
            "action": "test reconstruction latency",
            "outcome": "latency measured",
            "success": True
        }
        
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        
        # Measure reconstruction time
        start_time = time.time()
        retrieved = await openclaw_interface.retrieve_memory(
            memory_id=memory_id,
            memory_category="experiences"
        )
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Verify latency requirement
        assert latency_ms < test_case["max_latency_ms"], \
            f"{test_case['name']}: Reconstruction took {latency_ms:.2f}ms, should be < {test_case['max_latency_ms']}ms"
        
        # Verify reconstruction succeeded
        assert retrieved is not None
        
        print(f"{test_case['name']}: Reconstruction latency = {latency_ms:.2f}ms")


@pytest.mark.asyncio
async def test_reconstruction_latency_statistics(openclaw_interface):
    """
    Test reconstruction latency statistics
    
    Validates: Requirement 6.5
    - Average latency < 1s
    - P95 latency < 1s
    - Consistent performance
    """
    num_samples = 20
    
    # First store memories
    memory_ids = []
    for i in range(num_samples):
        memory = {
            "timestamp": datetime.now(),
            "context": f"Reconstruction test memory {i} with sufficient content. " * 15,
            "intent": "reconstruction test",
            "action": f"test {i}",
            "outcome": f"result {i}",
            "success": True
        }
        
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        memory_ids.append(memory_id)
    
    # Measure reconstruction latencies
    latencies = []
    for memory_id in memory_ids:
        start_time = time.time()
        retrieved = await openclaw_interface.retrieve_memory(
            memory_id=memory_id,
            memory_category="experiences"
        )
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        assert retrieved is not None
    
    # Calculate statistics
    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    max_latency = max(latencies)
    
    print(f"\nReconstruction Latency Statistics ({num_samples} samples):")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  Median: {median_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")
    
    # Verify requirements
    assert avg_latency < 1000, f"Average latency {avg_latency:.2f}ms should be < 1000ms"
    assert p95_latency < 1000, f"P95 latency {p95_latency:.2f}ms should be < 1000ms"


# ============================================================================
# Test 3: Throughput > 50/min (Requirement 9.7)
# ============================================================================

@pytest.mark.asyncio
async def test_compression_throughput(openclaw_interface):
    """
    Test compression throughput
    
    Validates: Requirement 9.7
    - Can compress > 50 memories per minute
    - Sustained throughput
    """
    num_memories = 60  # Test with 60 to ensure > 50/min
    
    memories = [
        {
            "timestamp": datetime.now(),
            "context": f"Throughput test memory {i} with content for compression. " * 10,
            "intent": "throughput test",
            "action": f"test {i}",
            "outcome": f"result {i}",
            "success": True
        }
        for i in range(num_memories)
    ]
    
    # Measure throughput
    start_time = time.time()
    
    for memory in memories:
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        assert memory_id is not None
    
    end_time = time.time()
    
    # Calculate throughput
    total_time_seconds = end_time - start_time
    total_time_minutes = total_time_seconds / 60
    throughput_per_minute = num_memories / total_time_minutes
    
    print(f"\nCompression Throughput:")
    print(f"  Total memories: {num_memories}")
    print(f"  Total time: {total_time_seconds:.2f}s ({total_time_minutes:.2f}min)")
    print(f"  Throughput: {throughput_per_minute:.2f} memories/min")
    
    # Verify requirement
    assert throughput_per_minute > 50, \
        f"Throughput {throughput_per_minute:.2f}/min should be > 50/min"


@pytest.mark.asyncio
async def test_batch_processing_throughput(mock_llm_client, mock_model_selector, temp_storage_path):
    """
    Test batch processing throughput
    
    Validates: Requirement 9.7
    - Batch processing achieves higher throughput
    - Efficient parallel processing
    """
    compressor = LLMCompressor(
        llm_client=mock_llm_client,
        model_selector=mock_model_selector,
        min_compress_length=100
    )
    
    batch_processor = BatchProcessor(
        compressor=compressor,
        batch_size=16,
        max_concurrent=8
    )
    
    num_memories = 100
    texts = [
        f"Batch throughput test memory {i} with content for compression. " * 10
        for i in range(num_memories)
    ]
    
    # Measure batch throughput
    start_time = time.time()
    
    compressed_list = await batch_processor.compress_batch(
        texts=texts,
        memory_type=MemoryType.TEXT
    )
    
    end_time = time.time()
    
    # Calculate throughput
    total_time_seconds = end_time - start_time
    total_time_minutes = total_time_seconds / 60
    throughput_per_minute = num_memories / total_time_minutes
    
    print(f"\nBatch Processing Throughput:")
    print(f"  Total memories: {num_memories}")
    print(f"  Total time: {total_time_seconds:.2f}s ({total_time_minutes:.2f}min)")
    print(f"  Throughput: {throughput_per_minute:.2f} memories/min")
    
    # Verify all compressed
    assert len(compressed_list) == num_memories
    
    # Batch processing should achieve higher throughput
    # With mock LLM (50ms latency), we expect > 100/min with parallelization
    assert throughput_per_minute > 50, \
        f"Batch throughput {throughput_per_minute:.2f}/min should be > 50/min"


@pytest.mark.asyncio
async def test_end_to_end_throughput(openclaw_interface):
    """
    Test end-to-end throughput (store + retrieve)
    
    Validates: Requirement 9.7
    - Complete roundtrip throughput
    - System can handle sustained load
    """
    num_operations = 30  # 30 store + 30 retrieve = 60 operations
    
    # Store phase
    memory_ids = []
    store_start = time.time()
    
    for i in range(num_operations):
        memory = {
            "timestamp": datetime.now(),
            "context": f"E2E throughput test {i} with content. " * 10,
            "intent": "e2e test",
            "action": f"test {i}",
            "outcome": f"result {i}",
            "success": True
        }
        
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        memory_ids.append(memory_id)
    
    store_end = time.time()
    
    # Retrieve phase
    retrieve_start = time.time()
    
    for memory_id in memory_ids:
        retrieved = await openclaw_interface.retrieve_memory(
            memory_id=memory_id,
            memory_category="experiences"
        )
        assert retrieved is not None
    
    retrieve_end = time.time()
    
    # Calculate throughput
    store_time = store_end - store_start
    retrieve_time = retrieve_end - retrieve_start
    total_time = store_time + retrieve_time
    
    total_operations = num_operations * 2  # store + retrieve
    throughput_per_minute = (total_operations / total_time) * 60
    
    print(f"\nEnd-to-End Throughput:")
    print(f"  Store operations: {num_operations} in {store_time:.2f}s")
    print(f"  Retrieve operations: {num_operations} in {retrieve_time:.2f}s")
    print(f"  Total operations: {total_operations} in {total_time:.2f}s")
    print(f"  Throughput: {throughput_per_minute:.2f} operations/min")
    
    # Verify reasonable throughput
    # With mock LLM, we expect good performance
    assert throughput_per_minute > 30, \
        f"E2E throughput {throughput_per_minute:.2f}/min should be > 30/min"


# ============================================================================
# Test 4: Performance Under Load
# ============================================================================

@pytest.mark.asyncio
async def test_performance_under_concurrent_load(openclaw_interface):
    """
    Test performance under concurrent load
    
    Validates:
    - Performance maintained under concurrent operations
    - No significant degradation
    """
    num_concurrent = 10
    operations_per_task = 5
    
    async def concurrent_operations(task_id):
        latencies = []
        
        for i in range(operations_per_task):
            memory = {
                "timestamp": datetime.now(),
                "context": f"Concurrent load test {task_id}-{i} with content. " * 10,
                "intent": "load test",
                "action": f"test {task_id}-{i}",
                "outcome": f"result {task_id}-{i}",
                "success": True
            }
            
            start_time = time.time()
            memory_id = await openclaw_interface.store_memory(
                memory=memory,
                memory_category="experiences"
            )
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            assert memory_id is not None
        
        return latencies
    
    # Run concurrent tasks
    start_time = time.time()
    
    tasks = [concurrent_operations(i) for i in range(num_concurrent)]
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    # Flatten latencies
    all_latencies = [lat for task_latencies in results for lat in task_latencies]
    
    # Calculate statistics
    avg_latency = statistics.mean(all_latencies)
    max_latency = max(all_latencies)
    total_operations = num_concurrent * operations_per_task
    total_time = end_time - start_time
    throughput = (total_operations / total_time) * 60
    
    print(f"\nPerformance Under Concurrent Load:")
    print(f"  Concurrent tasks: {num_concurrent}")
    print(f"  Operations per task: {operations_per_task}")
    print(f"  Total operations: {total_operations}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average latency: {avg_latency:.2f}ms")
    print(f"  Max latency: {max_latency:.2f}ms")
    print(f"  Throughput: {throughput:.2f} operations/min")
    
    # Verify performance maintained
    assert avg_latency < 5000, f"Average latency under load {avg_latency:.2f}ms should be < 5000ms"
    assert throughput > 30, f"Throughput under load {throughput:.2f}/min should be > 30/min"


# ============================================================================
# Test 5: Performance Regression Detection
# ============================================================================

@pytest.mark.asyncio
async def test_performance_baseline(openclaw_interface):
    """
    Test performance baseline for regression detection
    
    Establishes baseline metrics for:
    - Compression latency
    - Reconstruction latency
    - Throughput
    """
    num_samples = 10
    
    # Compression baseline
    compression_latencies = []
    for i in range(num_samples):
        memory = {
            "timestamp": datetime.now(),
            "context": f"Baseline test {i} with standard content. " * 15,
            "intent": "baseline",
            "action": f"test {i}",
            "outcome": f"result {i}",
            "success": True
        }
        
        start_time = time.time()
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        end_time = time.time()
        
        compression_latencies.append((end_time - start_time) * 1000)
    
    # Reconstruction baseline
    reconstruction_latencies = []
    memory_ids = []
    
    for i in range(num_samples):
        memory = {
            "timestamp": datetime.now(),
            "context": f"Reconstruction baseline {i} with content. " * 15,
            "intent": "baseline",
            "action": f"test {i}",
            "outcome": f"result {i}",
            "success": True
        }
        
        memory_id = await openclaw_interface.store_memory(
            memory=memory,
            memory_category="experiences"
        )
        memory_ids.append(memory_id)
    
    for memory_id in memory_ids:
        start_time = time.time()
        retrieved = await openclaw_interface.retrieve_memory(
            memory_id=memory_id,
            memory_category="experiences"
        )
        end_time = time.time()
        
        reconstruction_latencies.append((end_time - start_time) * 1000)
    
    # Calculate baselines
    compression_baseline = {
        "avg": statistics.mean(compression_latencies),
        "median": statistics.median(compression_latencies),
        "p95": sorted(compression_latencies)[int(len(compression_latencies) * 0.95)],
        "max": max(compression_latencies)
    }
    
    reconstruction_baseline = {
        "avg": statistics.mean(reconstruction_latencies),
        "median": statistics.median(reconstruction_latencies),
        "p95": sorted(reconstruction_latencies)[int(len(reconstruction_latencies) * 0.95)],
        "max": max(reconstruction_latencies)
    }
    
    print(f"\nPerformance Baseline ({num_samples} samples):")
    print(f"  Compression:")
    print(f"    Average: {compression_baseline['avg']:.2f}ms")
    print(f"    Median: {compression_baseline['median']:.2f}ms")
    print(f"    P95: {compression_baseline['p95']:.2f}ms")
    print(f"    Max: {compression_baseline['max']:.2f}ms")
    print(f"  Reconstruction:")
    print(f"    Average: {reconstruction_baseline['avg']:.2f}ms")
    print(f"    Median: {reconstruction_baseline['median']:.2f}ms")
    print(f"    P95: {reconstruction_baseline['p95']:.2f}ms")
    print(f"    Max: {reconstruction_baseline['max']:.2f}ms")
    
    # Verify baselines meet requirements
    assert compression_baseline['avg'] < 5000
    assert reconstruction_baseline['avg'] < 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

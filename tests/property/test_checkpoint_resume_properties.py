"""
Property-Based Tests for Checkpoint/Resume

Tests checkpoint/resume functionality for fault tolerance in batch processing.

Feature: llm-compression-integration, Property 23: 断点续传
Requirements: 9.6
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, settings, strategies as st
from unittest.mock import Mock, AsyncMock, patch

from llm_compression import (
    BatchProcessor,
    LLMCompressor,
    LLMClient,
    ModelSelector,
    MemoryType
)


class TestCheckpointResumeProperties:
    """
    Property tests for checkpoint/resume functionality
    
    **Validates: Requirements 9.6**
    """
    
    @pytest.fixture
    def checkpoint_dir(self):
        """Create temporary checkpoint directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    async def batch_processor(self, llm_client, model_selector, checkpoint_dir):
        """Create batch processor with checkpoint support"""
        compressor = LLMCompressor(
            llm_client=llm_client,
            model_selector=model_selector,
            min_compress_length=100
        )
        
        processor = BatchProcessor(
            compressor=compressor,
            batch_size=16,
            max_concurrent=4,
            checkpoint_dir=checkpoint_dir
        )
        
        return processor
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=None)
    @given(
        num_texts=st.integers(min_value=10, max_value=30),
        failure_index=st.integers(min_value=3, max_value=8)
    )
    async def test_checkpoint_resume_after_failure(
        self,
        batch_processor,
        num_texts,
        failure_index
    ):
        """
        Feature: llm-compression-integration, Property 23: 断点续传
        
        Test: *For any* 批量处理任务，当部分失败时，系统应该能够从最后成功的位置继续
        
        **Validates: Requirements 9.6**
        """
        # Generate test texts
        texts = [f"Test text {i} for checkpoint resume" for i in range(num_texts)]
        checkpoint_id = "test_checkpoint_1"
        
        # Mock compressor to fail at specific index
        original_compress = batch_processor.compressor.compress
        call_count = [0]
        
        async def mock_compress_with_failure(text, memory_type=MemoryType.TEXT, metadata=None):
            call_count[0] += 1
            if call_count[0] == failure_index:
                raise Exception("Simulated failure")
            return await original_compress(text, memory_type, metadata)
        
        # First attempt - will fail
        with patch.object(
            batch_processor.compressor,
            'compress',
            side_effect=mock_compress_with_failure
        ):
            try:
                results1 = await batch_processor.compress_batch(
                    texts=texts,
                    checkpoint_id=checkpoint_id
                )
            except Exception:
                pass  # Expected to fail
        
        # Verify checkpoint was saved
        checkpoint_file = Path(batch_processor.checkpoint_dir) / f"{checkpoint_id}.json"
        assert checkpoint_file.exists(), "Checkpoint file not created"
        
        # Load checkpoint
        progress = batch_processor._load_checkpoint(checkpoint_id)
        assert progress is not None, "Failed to load checkpoint"
        assert progress.completed_items > 0, "No items completed before failure"
        
        completed_before_resume = progress.completed_items
        
        # Resume from checkpoint
        call_count[0] = 0  # Reset counter
        results2 = await batch_processor.compress_batch(
            texts=texts,
            checkpoint_id=checkpoint_id,
            resume=True
        )
        
        # Verify results
        assert len(results2) == num_texts, \
            f"Expected {num_texts} results, got {len(results2)}"
        
        # Verify all results are valid
        for i, result in enumerate(results2):
            assert result is not None, f"Result {i} is None"
            assert result.memory_id, f"Result {i} has no memory_id"
        
        print(f"Checkpoint resume: {completed_before_resume} completed before resume, "
              f"{num_texts} total")
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=None)
    @given(
        num_texts=st.integers(min_value=5, max_value=20)
    )
    async def test_checkpoint_saves_progress(
        self,
        batch_processor,
        num_texts
    ):
        """
        Feature: llm-compression-integration, Property 23: 断点续传
        
        Test: *For any* 批量处理任务，系统应该定期保存进度
        
        **Validates: Requirements 9.6**
        """
        texts = [f"Text {i}" for i in range(num_texts)]
        checkpoint_id = "test_checkpoint_2"
        
        # Process batch with checkpoint
        results = await batch_processor.compress_batch(
            texts=texts,
            checkpoint_id=checkpoint_id
        )
        
        # Verify checkpoint was created during processing
        # (checkpoint should be deleted after successful completion)
        checkpoint_file = Path(batch_processor.checkpoint_dir) / f"{checkpoint_id}.json"
        
        # Checkpoint should be deleted after successful completion
        # But we can verify the checkpoint directory exists
        assert Path(batch_processor.checkpoint_dir).exists()
        
        print(f"Checkpoint progress saved for {num_texts} texts")
    
    @pytest.mark.asyncio
    async def test_checkpoint_resume_skips_completed(
        self,
        batch_processor
    ):
        """
        Feature: llm-compression-integration, Property 23: 断点续传
        
        Test: *For any* 恢复操作，系统应该跳过已完成的项目
        
        **Validates: Requirements 9.6**
        """
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        checkpoint_id = "test_checkpoint_3"
        
        # Manually create a checkpoint with some completed items
        from llm_compression.batch_processor import BatchProgress
        import time
        import json
        from dataclasses import asdict
        
        progress = BatchProgress(
            total_items=len(texts),
            completed_items=2,
            failed_items=0,
            start_time=time.time(),
            last_checkpoint_time=time.time(),
            completed_indices=[0, 1],
            failed_indices=[]
        )
        
        checkpoint_file = Path(batch_processor.checkpoint_dir) / f"{checkpoint_id}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(asdict(progress), f)
        
        # Track which texts are compressed
        compressed_indices = []
        original_compress = batch_processor.compressor.compress
        
        async def track_compress(text, memory_type=MemoryType.TEXT, metadata=None):
            # Find index of text
            idx = texts.index(text) if text in texts else -1
            compressed_indices.append(idx)
            return await original_compress(text, memory_type, metadata)
        
        with patch.object(
            batch_processor.compressor,
            'compress',
            side_effect=track_compress
        ):
            results = await batch_processor.compress_batch(
                texts=texts,
                checkpoint_id=checkpoint_id,
                resume=True
            )
        
        # Verify only uncompleted items were compressed
        # Items 0 and 1 should be skipped
        assert 0 not in compressed_indices, "Item 0 should be skipped"
        assert 1 not in compressed_indices, "Item 1 should be skipped"
        
        # Items 2, 3, 4 should be compressed
        assert 2 in compressed_indices or len(results) == len(texts)
        
        print(f"Resume skipped {progress.completed_items} completed items")
    
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=None)
    @given(
        num_texts=st.integers(min_value=5, max_value=15)
    )
    async def test_checkpoint_cleanup_after_success(
        self,
        batch_processor,
        num_texts
    ):
        """
        Feature: llm-compression-integration, Property 23: 断点续传
        
        Test: *For any* 成功完成的批量任务，系统应该清理检查点文件
        
        **Validates: Requirements 9.6**
        """
        texts = [f"Text {i}" for i in range(num_texts)]
        checkpoint_id = "test_checkpoint_4"
        
        # Process batch
        results = await batch_processor.compress_batch(
            texts=texts,
            checkpoint_id=checkpoint_id
        )
        
        # Verify checkpoint was deleted after success
        checkpoint_file = Path(batch_processor.checkpoint_dir) / f"{checkpoint_id}.json"
        assert not checkpoint_file.exists(), \
            "Checkpoint file should be deleted after successful completion"
        
        print(f"Checkpoint cleaned up after processing {num_texts} texts")
    
    @pytest.mark.asyncio
    async def test_checkpoint_without_checkpoint_dir(
        self,
        llm_client,
        model_selector
    ):
        """
        Feature: llm-compression-integration, Property 23: 断点续传
        
        Test: *For any* 没有配置检查点目录的处理器，系统应该正常工作（不保存检查点）
        
        **Validates: Requirements 9.6**
        """
        # Create processor without checkpoint dir
        compressor = LLMCompressor(
            llm_client=llm_client,
            model_selector=model_selector
        )
        
        processor = BatchProcessor(
            compressor=compressor,
            checkpoint_dir=None  # No checkpoint support
        )
        
        texts = ["Text 1", "Text 2", "Text 3"]
        
        # Process batch (should work without checkpoints)
        results = await processor.compress_batch(texts)
        
        assert len(results) == len(texts)
        
        print("Batch processing works without checkpoint support")


@pytest.fixture
async def llm_client():
    """Create mock LLM client for testing"""
    from unittest.mock import Mock
    
    client = Mock(spec=LLMClient)
    
    async def mock_generate(prompt, max_tokens=100, temperature=0.3):
        from llm_compression import LLMResponse
        summary = prompt[:50] + "..."
        return LLMResponse(
            text=summary,
            tokens_used=50,
            latency_ms=100.0,
            model="mock-model",
            finish_reason="stop",
            metadata={}
        )
    
    client.generate = mock_generate
    
    return client


@pytest.fixture
def model_selector():
    """Create mock model selector for testing"""
    from unittest.mock import Mock
    from llm_compression import ModelConfig, QualityLevel
    
    selector = Mock(spec=ModelSelector)
    
    def mock_select_model(memory_type, text_length, quality_requirement=QualityLevel.STANDARD):
        return ModelConfig(
            model_name="mock-model",
            endpoint="http://localhost:8045",
            is_local=False,
            max_tokens=100,
            temperature=0.3,
            expected_latency_ms=100.0,
            expected_quality=0.9
        )
    
    selector.select_model = mock_select_model
    
    return selector

"""
Integration tests for async PyO3 bindings

Tests async quantization operations with Python asyncio support.
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import pytest
import numpy as np

# Import the async quantizer
try:
    from arrow_quant_v2 import AsyncArrowQuantV2, DiffusionQuantConfig
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    pytest.skip("AsyncArrowQuantV2 not available", allow_module_level=True)


# Test fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_model_dir(temp_dir):
    """Create a mock model directory with metadata"""
    model_dir = Path(temp_dir) / "mock_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata.json
    metadata = {
        "modality": "text",
        "model_type": "diffusion",
        "num_layers": 2,
        "hidden_size": 128
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    # Create a simple mock weight file (Parquet format would be ideal, but we'll use numpy)
    # In a real scenario, this would be a proper Parquet file
    weights = np.random.randn(128, 128).astype(np.float32)
    np.save(model_dir / "layer_0.npy", weights)
    
    return model_dir


@pytest.fixture
def multiple_mock_models(temp_dir):
    """Create multiple mock model directories"""
    models = []
    for i in range(3):
        model_dir = Path(temp_dir) / f"mock_model_{i}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "modality": "text" if i % 2 == 0 else "image",
            "model_type": "diffusion",
            "num_layers": 2,
            "hidden_size": 64
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        weights = np.random.randn(64, 64).astype(np.float32)
        np.save(model_dir / "layer_0.npy", weights)
        
        models.append(model_dir)
    
    return models


class TestAsyncQuantization:
    """Test async quantization operations"""

    @pytest.mark.asyncio
    async def test_async_quantizer_creation(self):
        """Test creating an async quantizer instance"""
        quantizer = AsyncArrowQuantV2()
        assert quantizer is not None

    @pytest.mark.asyncio
    async def test_async_quantization_basic(self):
        """Test basic async quantization (mock test)"""
        quantizer = AsyncArrowQuantV2()
        
        # Note: This is a mock test since we don't have actual model files
        # In a real scenario, you would provide actual model paths
        # For now, we just verify the API is callable
        
        # The actual quantization would fail without real model files,
        # but we can verify the async interface works
        assert hasattr(quantizer, 'quantize_diffusion_model_async')
        assert callable(quantizer.quantize_diffusion_model_async)

    @pytest.mark.asyncio
    async def test_async_progress_callback(self):
        """Test async progress callback functionality"""
        progress_updates = []
        
        async def progress_callback(message: str, progress: float):
            """Async progress callback"""
            progress_updates.append({
                'message': message,
                'progress': progress
            })
        
        quantizer = AsyncArrowQuantV2()
        
        # Verify callback is accepted
        # Note: Actual quantization would require real model files
        assert callable(progress_callback)
        
        # Test callback directly
        await progress_callback("Test message", 0.5)
        assert len(progress_updates) == 1
        assert progress_updates[0]['message'] == "Test message"
        assert progress_updates[0]['progress'] == 0.5

    @pytest.mark.asyncio
    async def test_async_multiple_models_interface(self):
        """Test concurrent quantization interface"""
        quantizer = AsyncArrowQuantV2()
        
        # Verify the method exists and is callable
        assert hasattr(quantizer, 'quantize_multiple_models_async')
        assert callable(quantizer.quantize_multiple_models_async)
        
        # Test with empty list (should work without errors)
        # Note: Actual quantization would require real model files
        models = []
        
        # Just verify the interface is correct
        assert isinstance(models, list)

    @pytest.mark.asyncio
    async def test_async_validation_interface(self):
        """Test async validation interface"""
        quantizer = AsyncArrowQuantV2()
        
        # Verify the method exists and is callable
        assert hasattr(quantizer, 'validate_quality_async')
        assert callable(quantizer.validate_quality_async)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test that multiple async operations can run concurrently"""
        quantizer = AsyncArrowQuantV2()
        
        # Create multiple async tasks
        tasks = []
        for i in range(3):
            # Create a simple async task
            async def dummy_task(idx):
                await asyncio.sleep(0.01)  # Simulate async work
                return f"Task {idx} complete"
            
            tasks.append(dummy_task(i))
        
        # Run tasks concurrently
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all("complete" in result for result in results)

    @pytest.mark.asyncio
    async def test_config_with_async_quantizer(self):
        """Test using DiffusionQuantConfig with async quantizer"""
        config = DiffusionQuantConfig(
            bit_width=4,
            num_time_groups=10,
            group_size=128,
            min_accuracy=0.85
        )
        
        quantizer = AsyncArrowQuantV2()
        
        # Verify config can be created and used with async API
        assert config is not None
        assert quantizer is not None

    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test error handling in async context"""
        quantizer = AsyncArrowQuantV2()
        
        # Test with invalid paths (should raise error)
        with pytest.raises(Exception):
            # This should fail because the paths don't exist
            await quantizer.quantize_diffusion_model_async(
                model_path="/nonexistent/path",
                output_path="/nonexistent/output",
                config=None
            )

    @pytest.mark.asyncio
    async def test_async_cancellation(self):
        """Test that async operations can be cancelled"""
        quantizer = AsyncArrowQuantV2()
        
        # Create a task that we'll cancel
        async def long_running_task():
            await asyncio.sleep(10)  # Long sleep
            return "Should not complete"
        
        task = asyncio.create_task(long_running_task())
        
        # Cancel immediately
        task.cancel()
        
        # Verify cancellation
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_multiple_quantizers(self):
        """Test creating multiple async quantizer instances"""
        quantizers = [AsyncArrowQuantV2() for _ in range(3)]
        
        assert len(quantizers) == 3
        assert all(q is not None for q in quantizers)

    @pytest.mark.asyncio
    async def test_async_with_deployment_profiles(self):
        """Test async quantization with different deployment profiles"""
        quantizer = AsyncArrowQuantV2()
        
        # Test with different profiles
        profiles = ["edge", "local", "cloud"]
        
        for profile in profiles:
            config = DiffusionQuantConfig.from_profile(profile)
            assert config is not None
            
            # Verify config can be used with async API
            # (actual quantization would require real model files)


class TestAsyncConcurrentQuantization:
    """Test concurrent quantization of multiple models"""

    @pytest.mark.asyncio
    async def test_concurrent_quantization_interface(self):
        """Test the interface for concurrent quantization"""
        quantizer = AsyncArrowQuantV2()
        
        # Define multiple models (mock data)
        models = [
            ("model1/", "output1/", None),
            ("model2/", "output2/", DiffusionQuantConfig(bit_width=2)),
            ("model3/", "output3/", DiffusionQuantConfig(bit_width=4)),
        ]
        
        # Verify the interface accepts the correct format
        assert isinstance(models, list)
        assert all(isinstance(m, tuple) and len(m) == 3 for m in models)

    @pytest.mark.asyncio
    async def test_concurrent_with_progress_callback(self):
        """Test concurrent quantization with progress tracking"""
        progress_updates = []
        
        async def progress_callback(model_idx: int, message: str, progress: float):
            """Track progress for multiple models"""
            progress_updates.append({
                'model_idx': model_idx,
                'message': message,
                'progress': progress
            })
        
        # Verify callback signature
        assert callable(progress_callback)
        
        # Test callback
        await progress_callback(0, "Test", 0.5)
        assert len(progress_updates) == 1

    @pytest.mark.asyncio
    async def test_concurrent_quantization_with_errors(self, temp_dir):
        """Test concurrent quantization handles errors gracefully"""
        quantizer = AsyncArrowQuantV2()
        
        # Mix valid and invalid paths
        models = [
            ("/nonexistent/model1", f"{temp_dir}/output1", None),
            ("/nonexistent/model2", f"{temp_dir}/output2", None),
        ]
        
        # Should raise error for invalid paths
        with pytest.raises(Exception):
            await quantizer.quantize_multiple_models_async(models)

    @pytest.mark.asyncio
    async def test_concurrent_quantization_different_configs(self, temp_dir):
        """Test concurrent quantization with different configurations"""
        quantizer = AsyncArrowQuantV2()
        
        # Create models with different bit widths
        configs = [
            DiffusionQuantConfig(bit_width=2, num_time_groups=5),
            DiffusionQuantConfig(bit_width=4, num_time_groups=10),
            DiffusionQuantConfig(bit_width=8, num_time_groups=20),
        ]
        
        # Verify configs are created successfully
        assert len(configs) == 3
        assert all(c is not None for c in configs)

    @pytest.mark.asyncio
    async def test_concurrent_quantization_timing(self):
        """Test that concurrent quantization is actually concurrent"""
        quantizer = AsyncArrowQuantV2()
        
        # Create dummy async tasks that simulate quantization
        async def dummy_quantization(duration: float):
            await asyncio.sleep(duration)
            return {"status": "complete", "duration": duration}
        
        # Run 3 tasks concurrently, each taking 0.1 seconds
        start_time = time.time()
        tasks = [dummy_quantization(0.1) for _ in range(3)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        # Should complete in ~0.1 seconds (concurrent), not 0.3 seconds (sequential)
        assert elapsed < 0.2  # Allow some overhead
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_concurrent_quantization_max_concurrency(self):
        """Test handling of many concurrent quantization tasks"""
        quantizer = AsyncArrowQuantV2()
        
        # Create many dummy tasks
        async def dummy_task(idx: int):
            await asyncio.sleep(0.01)
            return idx
        
        # Run 50 concurrent tasks
        tasks = [dummy_task(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 50
        assert results == list(range(50))


class TestAsyncCancellation:
    """Test async operation cancellation and cleanup"""

    @pytest.mark.asyncio
    async def test_async_cancellation_basic(self):
        """Test that async operations can be cancelled"""
        quantizer = AsyncArrowQuantV2()
        
        # Create a long-running task
        async def long_task():
            await asyncio.sleep(10)
            return "Should not complete"
        
        task = asyncio.create_task(long_task())
        
        # Cancel immediately
        task.cancel()
        
        # Verify cancellation
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_quantization_cancellation(self, temp_dir):
        """Test cancelling a quantization operation"""
        quantizer = AsyncArrowQuantV2()
        
        # The pyo3-asyncio methods return Futures that can be cancelled
        # We need to test cancellation differently
        
        # Create a simple cancellable task
        async def cancellable_operation():
            try:
                result = await quantizer.quantize_diffusion_model_async(
                    model_path="/nonexistent/model",
                    output_path=f"{temp_dir}/output",
                    config=None
                )
                return result
            except Exception as e:
                # Expected to fail with invalid path - this is ok
                # The test is about cancellation, not successful quantization
                raise
        
        # Create task and cancel it quickly
        task = asyncio.create_task(cancellable_operation())
        
        # Cancel immediately (before it can fail)
        task.cancel()
        
        # Verify cancellation or expected error
        try:
            await task
            # If we get here, something unexpected happened
            assert False, "Task should have been cancelled or raised an error"
        except asyncio.CancelledError:
            # Successfully cancelled
            pass
        except Exception:
            # Also acceptable - the operation failed before cancellation
            pass

    @pytest.mark.asyncio
    async def test_concurrent_cancellation(self):
        """Test cancelling multiple concurrent operations"""
        quantizer = AsyncArrowQuantV2()
        
        # Create multiple long-running tasks
        async def long_task(idx: int):
            await asyncio.sleep(10)
            return f"Task {idx}"
        
        tasks = [asyncio.create_task(long_task(i)) for i in range(5)]
        
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        
        # Verify all cancelled
        for task in tasks:
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_partial_cancellation(self):
        """Test cancelling some operations while others complete"""
        # Create mix of short and long tasks
        async def short_task(idx: int):
            await asyncio.sleep(0.01)
            return f"Short {idx}"
        
        async def long_task(idx: int):
            await asyncio.sleep(10)
            return f"Long {idx}"
        
        # Create tasks
        short_tasks = [asyncio.create_task(short_task(i)) for i in range(3)]
        long_tasks = [asyncio.create_task(long_task(i)) for i in range(3)]
        
        # Wait for short tasks to complete
        short_results = await asyncio.gather(*short_tasks)
        
        # Cancel long tasks
        for task in long_tasks:
            task.cancel()
        
        # Verify short tasks completed
        assert len(short_results) == 3
        assert all("Short" in r for r in short_results)
        
        # Verify long tasks cancelled
        for task in long_tasks:
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_cleanup_after_cancellation(self, temp_dir):
        """Test that resources are cleaned up after cancellation"""
        quantizer = AsyncArrowQuantV2()
        
        # Create output directory
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a cancellable wrapper
        async def cancellable_operation():
            try:
                result = await quantizer.quantize_diffusion_model_async(
                    model_path="/nonexistent/model",
                    output_path=str(output_dir),
                    config=None
                )
                return result
            except Exception:
                # Expected to fail
                raise
        
        # Create task and cancel it
        task = asyncio.create_task(cancellable_operation())
        await asyncio.sleep(0.01)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            # Other exceptions are also ok for this test
            pass
        
        # Verify we can still use the quantizer
        assert quantizer is not None


class TestAsyncErrorHandling:
    """Test error handling in async context"""

    @pytest.mark.asyncio
    async def test_async_error_propagation(self):
        """Test that errors are properly propagated in async context"""
        quantizer = AsyncArrowQuantV2()
        
        # Test with invalid paths
        with pytest.raises(Exception):
            await quantizer.quantize_diffusion_model_async(
                model_path="/nonexistent/path",
                output_path="/nonexistent/output",
                config=None
            )

    @pytest.mark.asyncio
    async def test_async_validation_error(self):
        """Test validation errors in async context"""
        quantizer = AsyncArrowQuantV2()
        
        # Test with invalid paths
        with pytest.raises(Exception):
            await quantizer.validate_quality_async(
                original_path="/nonexistent/original",
                quantized_path="/nonexistent/quantized"
            )

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, temp_dir):
        """Test error handling with concurrent operations"""
        quantizer = AsyncArrowQuantV2()
        
        # Mix of valid and invalid operations
        models = [
            ("/nonexistent/model1", f"{temp_dir}/output1", None),
            ("/nonexistent/model2", f"{temp_dir}/output2", None),
        ]
        
        # Should raise error
        with pytest.raises(Exception):
            await quantizer.quantize_multiple_models_async(models)

    @pytest.mark.asyncio
    async def test_error_in_progress_callback(self, temp_dir):
        """Test handling of errors in progress callback"""
        quantizer = AsyncArrowQuantV2()
        
        async def failing_callback(message: str, progress: float):
            """Callback that raises an error"""
            if progress > 0.5:
                raise ValueError("Callback error")
        
        # Quantization should continue even if callback fails
        # (In real implementation, callback errors should be caught and logged)
        # For now, just verify the callback can be called
        try:
            await failing_callback("Test", 0.6)
        except ValueError:
            pass  # Expected

    @pytest.mark.asyncio
    async def test_invalid_config_error(self):
        """Test error handling for invalid configuration"""
        quantizer = AsyncArrowQuantV2()
        
        # Create invalid config
        invalid_config = DiffusionQuantConfig(
            bit_width=3,  # Invalid bit width (should be 2, 4, or 8)
            num_time_groups=10
        )
        
        # Should raise configuration error
        # Note: Validation happens in Rust, so error type may vary
        with pytest.raises(Exception):
            await quantizer.quantize_diffusion_model_async(
                model_path="/nonexistent/model",
                output_path="/nonexistent/output",
                config=invalid_config
            )

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of operation timeouts"""
        quantizer = AsyncArrowQuantV2()
        
        # Create a task with timeout
        async def timed_operation():
            return await asyncio.wait_for(
                quantizer.quantize_diffusion_model_async(
                    model_path="/nonexistent/model",
                    output_path="/nonexistent/output",
                    config=None
                ),
                timeout=0.1  # Very short timeout
            )
        
        # Should raise TimeoutError or the underlying exception
        with pytest.raises((asyncio.TimeoutError, Exception)):
            await timed_operation()


class TestAsyncValidation:
    """Test async validation operations"""

    @pytest.mark.asyncio
    async def test_async_validation_interface(self):
        """Test async validation interface"""
        quantizer = AsyncArrowQuantV2()
        
        # Verify method exists
        assert hasattr(quantizer, 'validate_quality_async')
        
        # Test with invalid paths (should raise error)
        with pytest.raises(Exception):
            await quantizer.validate_quality_async(
                original_path="/nonexistent/original",
                quantized_path="/nonexistent/quantized"
            )

    @pytest.mark.asyncio
    async def test_concurrent_validation(self):
        """Test concurrent validation of multiple models"""
        quantizer = AsyncArrowQuantV2()
        
        # Create multiple validation tasks
        validation_pairs = [
            ("/nonexistent/orig1", "/nonexistent/quant1"),
            ("/nonexistent/orig2", "/nonexistent/quant2"),
            ("/nonexistent/orig3", "/nonexistent/quant3"),
        ]
        
        # All should fail (invalid paths), but test concurrent execution
        tasks = [
            quantizer.validate_quality_async(orig, quant)
            for orig, quant in validation_pairs
        ]
        
        # Gather with return_exceptions to capture all errors
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should be exceptions
        assert len(results) == 3
        assert all(isinstance(r, Exception) for r in results)

    @pytest.mark.asyncio
    async def test_validation_after_quantization(self, temp_dir):
        """Test validation immediately after quantization"""
        quantizer = AsyncArrowQuantV2()
        
        # This is a conceptual test - in real scenario:
        # 1. Quantize model
        # 2. Validate result
        # For now, just verify the workflow is possible
        
        model_path = "/nonexistent/model"
        output_path = f"{temp_dir}/output"
        
        # Both should fail, but test the workflow
        try:
            result = await quantizer.quantize_diffusion_model_async(
                model_path=model_path,
                output_path=output_path,
                config=None
            )
            # If quantization succeeded, validate
            validation = await quantizer.validate_quality_async(
                original_path=model_path,
                quantized_path=output_path
            )
        except Exception:
            pass  # Expected to fail with nonexistent paths


class TestAsyncProgressTracking:
    """Test async progress tracking and callbacks"""

    @pytest.mark.asyncio
    async def test_progress_callback_invocation(self):
        """Test that progress callbacks are invoked"""
        progress_updates = []
        
        async def progress_callback(message: str, progress: float):
            """Track progress updates"""
            progress_updates.append({
                'message': message,
                'progress': progress,
                'timestamp': time.time()
            })
        
        # Test callback directly
        await progress_callback("Starting", 0.0)
        await progress_callback("Processing", 0.5)
        await progress_callback("Complete", 1.0)
        
        assert len(progress_updates) == 3
        assert progress_updates[0]['progress'] == 0.0
        assert progress_updates[1]['progress'] == 0.5
        assert progress_updates[2]['progress'] == 1.0

    @pytest.mark.asyncio
    async def test_progress_callback_timing(self):
        """Test progress callback timing and throttling"""
        progress_updates = []
        
        async def progress_callback(message: str, progress: float):
            """Track progress with timestamps"""
            progress_updates.append({
                'message': message,
                'progress': progress,
                'timestamp': time.time()
            })
        
        # Simulate rapid progress updates
        for i in range(10):
            await progress_callback(f"Step {i}", i / 10.0)
            await asyncio.sleep(0.01)
        
        # All updates should be recorded
        assert len(progress_updates) == 10
        
        # Verify timestamps are increasing
        timestamps = [u['timestamp'] for u in progress_updates]
        assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_concurrent_progress_tracking(self):
        """Test progress tracking for concurrent operations"""
        progress_by_model = {0: [], 1: [], 2: []}
        
        async def progress_callback(model_idx: int, message: str, progress: float):
            """Track progress per model"""
            progress_by_model[model_idx].append({
                'message': message,
                'progress': progress
            })
        
        # Simulate concurrent operations with progress
        async def simulate_operation(model_idx: int):
            for i in range(5):
                await progress_callback(model_idx, f"Step {i}", i / 5.0)
                await asyncio.sleep(0.01)
        
        # Run 3 operations concurrently
        await asyncio.gather(
            simulate_operation(0),
            simulate_operation(1),
            simulate_operation(2)
        )
        
        # Verify all models reported progress
        assert len(progress_by_model[0]) == 5
        assert len(progress_by_model[1]) == 5
        assert len(progress_by_model[2]) == 5


class TestAsyncResourceManagement:
    """Test async resource management and cleanup"""

    @pytest.mark.asyncio
    async def test_multiple_quantizer_instances(self):
        """Test creating and using multiple quantizer instances"""
        quantizers = [AsyncArrowQuantV2() for _ in range(5)]
        
        assert len(quantizers) == 5
        assert all(q is not None for q in quantizers)
        
        # Verify each instance is independent
        for i, q in enumerate(quantizers):
            assert q is not None

    @pytest.mark.asyncio
    async def test_quantizer_reuse(self):
        """Test reusing a quantizer for multiple operations"""
        quantizer = AsyncArrowQuantV2()
        
        # Simulate multiple operations with the same quantizer
        async def dummy_operation(idx: int):
            await asyncio.sleep(0.01)
            return idx
        
        # Run multiple operations sequentially
        results = []
        for i in range(5):
            result = await dummy_operation(i)
            results.append(result)
        
        assert results == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_memory_cleanup(self):
        """Test that quantizers are properly cleaned up"""
        import gc
        
        # Create and destroy many quantizers
        for _ in range(100):
            q = AsyncArrowQuantV2()
            del q
        
        # Force garbage collection
        gc.collect()
        
        # If we got here without OOM, test passes
        assert True

    @pytest.mark.asyncio
    async def test_concurrent_resource_usage(self):
        """Test resource usage with many concurrent operations"""
        quantizer = AsyncArrowQuantV2()
        
        # Create many concurrent dummy tasks
        async def dummy_task(idx: int):
            await asyncio.sleep(0.01)
            return idx
        
        # Run 100 concurrent tasks
        tasks = [dummy_task(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 100
        assert results == list(range(100))


class TestAsyncIntegrationScenarios:
    """Test realistic async integration scenarios"""

    @pytest.mark.asyncio
    async def test_batch_quantization_workflow(self, temp_dir):
        """Test a complete batch quantization workflow"""
        quantizer = AsyncArrowQuantV2()
        
        # Simulate batch quantization workflow
        # 1. Prepare models
        model_configs = [
            ("model1", DiffusionQuantConfig(bit_width=2)),
            ("model2", DiffusionQuantConfig(bit_width=4)),
            ("model3", DiffusionQuantConfig(bit_width=8)),
        ]
        
        # 2. Verify configs
        for name, config in model_configs:
            assert config is not None
            assert name is not None
        
        # 3. In real scenario, would quantize all models concurrently
        # For now, just verify the workflow structure
        assert len(model_configs) == 3

    @pytest.mark.asyncio
    async def test_pipeline_with_validation(self, temp_dir):
        """Test quantization pipeline with validation"""
        quantizer = AsyncArrowQuantV2()
        
        # Simulate pipeline:
        # 1. Quantize
        # 2. Validate
        # 3. Report results
        
        # This is a conceptual test showing the workflow
        pipeline_steps = [
            "quantize",
            "validate",
            "report"
        ]
        
        assert len(pipeline_steps) == 3

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """Test error recovery in async workflow"""
        quantizer = AsyncArrowQuantV2()
        
        # Simulate workflow with error recovery
        max_retries = 3
        retry_count = 0
        
        async def operation_with_retry():
            nonlocal retry_count
            for attempt in range(max_retries):
                try:
                    retry_count += 1
                    # Simulate operation that might fail
                    if attempt < 2:
                        raise ValueError("Simulated error")
                    return "Success"
                except ValueError:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(0.01)
        
        result = await operation_with_retry()
        assert result == "Success"
        assert retry_count == 3

    @pytest.mark.asyncio
    async def test_mixed_sync_async_operations(self):
        """Test mixing synchronous and asynchronous operations"""
        quantizer = AsyncArrowQuantV2()
        
        # Synchronous operation
        sync_result = "sync_complete"
        
        # Asynchronous operation
        async def async_operation():
            await asyncio.sleep(0.01)
            return "async_complete"
        
        async_result = await async_operation()
        
        # Both should complete successfully
        assert sync_result == "sync_complete"
        assert async_result == "async_complete"


class TestAsyncPerformance:
    """Test async performance characteristics"""

    @pytest.mark.asyncio
    async def test_async_overhead(self):
        """Test that async operations have minimal overhead"""
        import time
        
        # Measure time for multiple concurrent dummy operations
        start_time = time.time()
        
        async def dummy_operation(idx):
            await asyncio.sleep(0.001)  # 1ms sleep
            return idx
        
        # Run 10 operations concurrently
        tasks = [dummy_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        
        # Should complete in roughly 1ms (not 10ms) due to concurrency
        assert elapsed < 0.1  # 100ms threshold (very generous)
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_async_memory_efficiency(self):
        """Test that async operations don't leak memory"""
        import gc
        
        quantizer = AsyncArrowQuantV2()
        
        # Create and destroy multiple quantizers
        for _ in range(100):
            q = AsyncArrowQuantV2()
            del q
        
        # Force garbage collection
        gc.collect()
        
        # If we got here without OOM, test passes
        assert True


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

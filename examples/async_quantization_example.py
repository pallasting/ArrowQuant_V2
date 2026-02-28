"""
Example: Async Quantization with ArrowQuant V2

This example demonstrates how to use the async PyO3 bindings for
non-blocking quantization operations with Python asyncio.

Requirements:
    - arrow_quant_v2 Python package installed
    - Python 3.10+
    - asyncio support

Usage:
    python examples/async_quantization_example.py
"""

import asyncio
from pathlib import Path
from typing import Dict, Any

# Import the async quantizer
# Note: This will only work after building the package with maturin
try:
    from arrow_quant_v2 import AsyncArrowQuantV2, DiffusionQuantConfig
    ASYNC_AVAILABLE = True
except ImportError:
    print("AsyncArrowQuantV2 not available. Build the package with:")
    print("  maturin build --features python --release")
    print("  pip install target/wheels/arrow_quant_v2-*.whl")
    ASYNC_AVAILABLE = False


async def example_basic_async_quantization():
    """Example 1: Basic async quantization"""
    print("\n=== Example 1: Basic Async Quantization ===\n")
    
    if not ASYNC_AVAILABLE:
        print("✓ Async quantization interface design:")
        print("  - AsyncArrowQuantV2() creates quantizer")
        print("  - quantize_diffusion_model_async() for single model")
        print("  - Returns awaitable coroutine")
        return
    
    quantizer = AsyncArrowQuantV2()
    
    # Configure quantization
    config = DiffusionQuantConfig(
        bit_width=4,
        num_time_groups=10,
        group_size=128,
        min_accuracy=0.85
    )
    
    print("Starting async quantization...")
    print("Note: This example uses placeholder paths")
    
    # In a real scenario, you would provide actual model paths:
    # result = await quantizer.quantize_diffusion_model_async(
    #     model_path="path/to/model",
    #     output_path="path/to/output",
    #     config=config
    # )
    # print(f"Compression ratio: {result['compression_ratio']}")
    # print(f"Cosine similarity: {result['cosine_similarity']}")
    
    print("✓ Async quantization interface ready")


async def example_with_progress_callback():
    """Example 2: Async quantization with progress tracking"""
    print("\n=== Example 2: Async Quantization with Progress ===\n")
    
    if not ASYNC_AVAILABLE:
        print("✓ Progress callback interface design:")
        print("  - async def progress_callback(message: str, progress: float)")
        print("  - Passed to quantize_diffusion_model_async()")
        print("  - Non-blocking progress updates")
        return
    
    # Define async progress callback
    async def progress_callback(message: str, progress: float):
        """Track quantization progress"""
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r[{bar}] {progress*100:.1f}% - {message}", end='', flush=True)
    
    quantizer = AsyncArrowQuantV2()
    
    print("Quantization with progress tracking:")
    
    # Simulate progress updates
    for i in range(11):
        await progress_callback(f"Processing step {i}/10", i / 10)
        await asyncio.sleep(0.1)
    
    print("\n✓ Progress tracking ready")


async def example_concurrent_quantization():
    """Example 3: Concurrent quantization of multiple models"""
    print("\n=== Example 3: Concurrent Multi-Model Quantization ===\n")
    
    if not ASYNC_AVAILABLE:
        print("✓ Concurrent quantization interface design:")
        print("  - quantize_multiple_models_async(models)")
        print("  - models = [(path, output, config), ...]")
        print("  - True parallelism with tokio tasks")
        return
    
    quantizer = AsyncArrowQuantV2()
    
    # Define multiple models to quantize
    models = [
        ("model1/", "model1-int2/", DiffusionQuantConfig(bit_width=2)),
        ("model2/", "model2-int4/", DiffusionQuantConfig(bit_width=4)),
        ("model3/", "model3-int8/", DiffusionQuantConfig(bit_width=8)),
    ]
    
    print(f"Preparing to quantize {len(models)} models concurrently...")
    
    # In a real scenario:
    # results = await quantizer.quantize_multiple_models_async(models)
    # for i, result in enumerate(results):
    #     print(f"Model {i+1}: {result['compression_ratio']}x compression")
    
    print("✓ Concurrent quantization interface ready")


async def example_async_validation():
    """Example 4: Async quality validation"""
    print("\n=== Example 4: Async Quality Validation ===\n")
    
    if not ASYNC_AVAILABLE:
        print("✓ Async validation interface design:")
        print("  - validate_quality_async(original_path, quantized_path)")
        print("  - Returns validation report dict")
        print("  - Non-blocking quality checks")
        return
    
    quantizer = AsyncArrowQuantV2()
    
    print("Validating quantization quality...")
    
    # In a real scenario:
    # report = await quantizer.validate_quality_async(
    #     original_path="path/to/original",
    #     quantized_path="path/to/quantized"
    # )
    # print(f"Validation passed: {report['passed']}")
    # print(f"Cosine similarity: {report['cosine_similarity']:.4f}")
    # print(f"Compression ratio: {report['compression_ratio']:.2f}x")
    
    print("✓ Async validation interface ready")


async def example_deployment_profiles():
    """Example 5: Using deployment profiles with async API"""
    print("\n=== Example 5: Deployment Profiles ===\n")
    
    if not ASYNC_AVAILABLE:
        print("✓ Deployment profiles work with async API:")
        print("  - Edge: INT2, <35MB, for edge devices")
        print("  - Local: INT4, <200MB, for local workstations")
        print("  - Cloud: INT8, 3B params, for cloud servers")
        return
    
    quantizer = AsyncArrowQuantV2()
    
    profiles = {
        "edge": "INT2, <35MB, for edge devices",
        "local": "INT4, <200MB, for local workstations",
        "cloud": "INT8, 3B params, for cloud servers"
    }
    
    for profile_name, description in profiles.items():
        config = DiffusionQuantConfig.from_profile(profile_name)
        print(f"✓ {profile_name.upper()} profile: {description}")
        print(f"  - Bit width: {config.bit_width}")
        print(f"  - Time groups: {config.num_time_groups}")
        print(f"  - Group size: {config.group_size}")
        print(f"  - Min accuracy: {config.min_accuracy}")
        print()


async def example_error_handling():
    """Example 6: Error handling in async context"""
    print("\n=== Example 6: Async Error Handling ===\n")
    
    if not ASYNC_AVAILABLE:
        print("✓ Error handling in async context:")
        print("  - Exceptions propagate correctly")
        print("  - try/except works with await")
        print("  - Detailed error messages provided")
        return
    
    quantizer = AsyncArrowQuantV2()
    
    print("Testing error handling with invalid paths...")
    
    try:
        # This should raise an error
        await quantizer.quantize_diffusion_model_async(
            model_path="/nonexistent/path",
            output_path="/nonexistent/output",
            config=None
        )
    except Exception as e:
        print(f"✓ Error caught correctly: {type(e).__name__}")
        print(f"  Message: {str(e)[:100]}...")


async def example_concurrent_operations():
    """Example 7: Running multiple async operations concurrently"""
    print("\n=== Example 7: Concurrent Async Operations ===\n")
    
    async def task(name: str, duration: float):
        """Simulate an async task"""
        print(f"Starting {name}...")
        await asyncio.sleep(duration)
        print(f"✓ {name} complete")
        return f"{name} result"
    
    print("Running 3 tasks concurrently...")
    
    # Run tasks concurrently
    results = await asyncio.gather(
        task("Task 1", 0.1),
        task("Task 2", 0.15),
        task("Task 3", 0.12)
    )
    
    print(f"\nAll tasks complete: {len(results)} results")


async def main():
    """Run all examples"""
    print("=" * 60)
    print("ArrowQuant V2 - Async Quantization Examples")
    print("=" * 60)
    
    if not ASYNC_AVAILABLE:
        print("\n⚠️  AsyncArrowQuantV2 not available")
        print("These examples demonstrate the API interface")
        print("Build the package to run actual quantization\n")
    
    # Run all examples
    await example_basic_async_quantization()
    await example_with_progress_callback()
    await example_concurrent_quantization()
    await example_async_validation()
    await example_deployment_profiles()
    await example_error_handling()
    await example_concurrent_operations()
    
    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

# Async API Documentation

## Overview

ArrowQuant V2 provides async PyO3 bindings for non-blocking quantization operations with Python asyncio support. This enables concurrent quantization of multiple models and provides a modern async API for Python users.

## Features

- **Non-blocking Operations**: All quantization operations return Python coroutines that can be awaited
- **Concurrent Quantization**: Quantize multiple models simultaneously with true parallelism
- **Async Progress Callbacks**: Track progress with async callback functions
- **Error Handling**: Proper async error propagation with detailed error messages
- **Backward Compatible**: Sync API remains unchanged, async API is additive

## Installation

Build the package with async support:

```bash
cd ai_os_diffusion/arrow_quant_v2
maturin build --features python --release
pip install target/wheels/arrow_quant_v2-*.whl
```

Or for development:

```bash
maturin develop --features python
```

## Quick Start

### Basic Async Quantization

```python
import asyncio
from arrow_quant_v2 import AsyncArrowQuantV2, DiffusionQuantConfig

async def main():
    quantizer = AsyncArrowQuantV2()
    
    result = await quantizer.quantize_diffusion_model_async(
        model_path="dream-7b/",
        output_path="dream-7b-int2/",
        config=DiffusionQuantConfig(bit_width=2)
    )
    
    print(f"Compression ratio: {result['compression_ratio']}")
    print(f"Cosine similarity: {result['cosine_similarity']}")

asyncio.run(main())
```

### With Progress Tracking

```python
import asyncio
from arrow_quant_v2 import AsyncArrowQuantV2, DiffusionQuantConfig

async def progress_callback(message: str, progress: float):
    """Track quantization progress"""
    print(f"[{progress*100:.1f}%] {message}")

async def main():
    quantizer = AsyncArrowQuantV2()
    
    result = await quantizer.quantize_diffusion_model_async(
        model_path="dream-7b/",
        output_path="dream-7b-int2/",
        config=DiffusionQuantConfig(bit_width=2),
        progress_callback=progress_callback
    )

asyncio.run(main())
```

### Concurrent Multi-Model Quantization

```python
import asyncio
from arrow_quant_v2 import AsyncArrowQuantV2, DiffusionQuantConfig

async def main():
    quantizer = AsyncArrowQuantV2()
    
    models = [
        ("model1/", "model1-int2/", DiffusionQuantConfig(bit_width=2)),
        ("model2/", "model2-int4/", DiffusionQuantConfig(bit_width=4)),
        ("model3/", "model3-int8/", None),  # Use default config
    ]
    
    results = await quantizer.quantize_multiple_models_async(models)
    
    for i, result in enumerate(results):
        print(f"Model {i}: {result['compression_ratio']}x compression")

asyncio.run(main())
```

## API Reference

### AsyncArrowQuantV2

The main async quantizer class.

#### `__init__()`

Create a new async quantizer instance.

```python
quantizer = AsyncArrowQuantV2()
```

#### `quantize_diffusion_model_async(model_path, output_path, config=None, progress_callback=None)`

Asynchronously quantize a diffusion model.

**Parameters:**
- `model_path` (str): Path to input model directory
- `output_path` (str): Path to output quantized model directory
- `config` (DiffusionQuantConfig, optional): Quantization configuration
- `progress_callback` (callable, optional): Async callback for progress updates
  - Signature: `async def callback(message: str, progress: float) -> None`

**Returns:**
- Coroutine that resolves to a dictionary containing:
  - `quantized_path`: Path to quantized model
  - `compression_ratio`: Compression ratio achieved
  - `cosine_similarity`: Average cosine similarity
  - `model_size_mb`: Size of quantized model in MB
  - `modality`: Detected modality (text, code, image, audio)
  - `bit_width`: Bit width used for quantization
  - `quantization_time_s`: Time taken for quantization in seconds

**Raises:**
- `QuantizationError`: If quantization fails
- `ConfigurationError`: If configuration is invalid

**Example:**

```python
result = await quantizer.quantize_diffusion_model_async(
    model_path="dream-7b/",
    output_path="dream-7b-int2/",
    config=DiffusionQuantConfig(bit_width=2)
)
```

#### `quantize_multiple_models_async(models, progress_callback=None)`

Asynchronously quantize multiple models concurrently.

**Parameters:**
- `models` (list): List of tuples `(model_path, output_path, config)`
  - `model_path` (str): Path to input model directory
  - `output_path` (str): Path to output quantized model directory
  - `config` (DiffusionQuantConfig or None): Configuration (None uses default)
- `progress_callback` (callable, optional): Async callback for progress updates
  - Signature: `async def callback(model_idx: int, message: str, progress: float) -> None`

**Returns:**
- Coroutine that resolves to a list of dictionaries (one per model)

**Raises:**
- `QuantizationError`: If any quantization fails

**Example:**

```python
models = [
    ("model1/", "output1/", DiffusionQuantConfig(bit_width=2)),
    ("model2/", "output2/", DiffusionQuantConfig(bit_width=4)),
    ("model3/", "output3/", None),
]

results = await quantizer.quantize_multiple_models_async(models)
```

#### `validate_quality_async(original_path, quantized_path)`

Asynchronously validate quantization quality.

**Parameters:**
- `original_path` (str): Path to original model directory
- `quantized_path` (str): Path to quantized model directory

**Returns:**
- Coroutine that resolves to a dictionary containing:
  - `cosine_similarity`: Average cosine similarity across layers
  - `compression_ratio`: Compression ratio achieved
  - `per_layer_accuracy`: Dictionary of per-layer cosine similarities
  - `passed`: Boolean indicating if validation passed

**Raises:**
- `ValidationError`: If validation fails

**Example:**

```python
report = await quantizer.validate_quality_async(
    original_path="dream-7b/",
    quantized_path="dream-7b-int2/"
)

print(f"Validation passed: {report['passed']}")
print(f"Cosine similarity: {report['cosine_similarity']}")
```

## Performance Characteristics

### Non-blocking I/O

All async operations release the GIL and run in background threads, allowing other async operations to proceed:

```python
# These operations run concurrently
results = await asyncio.gather(
    quantizer1.quantize_diffusion_model_async(...),
    quantizer2.quantize_diffusion_model_async(...),
    quantizer3.quantize_diffusion_model_async(...)
)
```

### Concurrent Quantization

The `quantize_multiple_models_async` method uses tokio tasks for true parallelism:

```python
# Quantize 3 models in parallel on a multi-core system
models = [
    ("model1/", "output1/", config1),
    ("model2/", "output2/", config2),
    ("model3/", "output3/", config3),
]

# All 3 models quantize simultaneously
results = await quantizer.quantize_multiple_models_async(models)
```

### Progress Throttling

Progress callbacks are throttled to 5-second intervals to avoid callback spam:

```python
async def progress_callback(message: str, progress: float):
    # Called at most once every 5 seconds
    print(f"[{progress*100:.1f}%] {message}")
```

## Error Handling

Async operations properly propagate errors:

```python
try:
    result = await quantizer.quantize_diffusion_model_async(
        model_path="/invalid/path",
        output_path="/invalid/output"
    )
except QuantizationError as e:
    print(f"Quantization failed: {e}")
except ConfigurationError as e:
    print(f"Invalid configuration: {e}")
```

## Integration with Async Frameworks

### FastAPI

```python
from fastapi import FastAPI
from arrow_quant_v2 import AsyncArrowQuantV2, DiffusionQuantConfig

app = FastAPI()
quantizer = AsyncArrowQuantV2()

@app.post("/quantize")
async def quantize_model(model_path: str, output_path: str):
    result = await quantizer.quantize_diffusion_model_async(
        model_path=model_path,
        output_path=output_path,
        config=DiffusionQuantConfig(bit_width=4)
    )
    return result
```

### aiohttp

```python
from aiohttp import web
from arrow_quant_v2 import AsyncArrowQuantV2

quantizer = AsyncArrowQuantV2()

async def handle_quantize(request):
    data = await request.json()
    result = await quantizer.quantize_diffusion_model_async(
        model_path=data['model_path'],
        output_path=data['output_path']
    )
    return web.json_response(result)

app = web.Application()
app.router.add_post('/quantize', handle_quantize)
```

## Deployment Profiles

All deployment profiles work with the async API:

```python
# Edge profile: INT2, <35MB
config = DiffusionQuantConfig.from_profile("edge")
result = await quantizer.quantize_diffusion_model_async(..., config=config)

# Local profile: INT4, <200MB
config = DiffusionQuantConfig.from_profile("local")
result = await quantizer.quantize_diffusion_model_async(..., config=config)

# Cloud profile: INT8, 3B params
config = DiffusionQuantConfig.from_profile("cloud")
result = await quantizer.quantize_diffusion_model_async(..., config=config)
```

## Comparison: Sync vs Async API

### Sync API (Blocking)

```python
from arrow_quant_v2 import ArrowQuantV2

quantizer = ArrowQuantV2(mode="diffusion")

# Blocks until complete
result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-int2/"
)
```

### Async API (Non-blocking)

```python
from arrow_quant_v2 import AsyncArrowQuantV2

quantizer = AsyncArrowQuantV2()

# Returns immediately, can await later
coro = quantizer.quantize_diffusion_model_async(
    model_path="dream-7b/",
    output_path="dream-7b-int2/"
)

# Do other work here...

# Await when ready
result = await coro
```

## Best Practices

### 1. Use Concurrent Quantization for Batch Processing

```python
# Good: Quantize multiple models concurrently
models = [...]
results = await quantizer.quantize_multiple_models_async(models)

# Less efficient: Quantize sequentially
for model_path, output_path, config in models:
    result = await quantizer.quantize_diffusion_model_async(
        model_path, output_path, config
    )
```

### 2. Implement Progress Tracking

```python
async def progress_callback(message: str, progress: float):
    # Log to file, update UI, send to monitoring system, etc.
    logger.info(f"[{progress*100:.1f}%] {message}")

result = await quantizer.quantize_diffusion_model_async(
    ...,
    progress_callback=progress_callback
)
```

### 3. Handle Errors Gracefully

```python
try:
    result = await quantizer.quantize_diffusion_model_async(...)
except QuantizationError as e:
    # Log error, retry with different config, notify user, etc.
    logger.error(f"Quantization failed: {e}")
    # Fallback to higher bit width
    result = await quantizer.quantize_diffusion_model_async(
        ...,
        config=DiffusionQuantConfig(bit_width=4)
    )
```

### 4. Use asyncio.gather for Multiple Operations

```python
# Run quantization and validation concurrently
quant_task = quantizer.quantize_diffusion_model_async(...)
val_task = quantizer.validate_quality_async(...)

quant_result, val_result = await asyncio.gather(quant_task, val_task)
```

## Troubleshooting

### Import Error

If you get `ImportError: cannot import name 'AsyncArrowQuantV2'`:

1. Ensure the package is built with Python features:
   ```bash
   maturin build --features python --release
   ```

2. Install the wheel:
   ```bash
   pip install target/wheels/arrow_quant_v2-*.whl
   ```

### Runtime Error

If you get `RuntimeError: no running event loop`:

```python
# Wrong: Calling async function without event loop
result = quantizer.quantize_diffusion_model_async(...)

# Correct: Use asyncio.run or await in async context
result = asyncio.run(quantizer.quantize_diffusion_model_async(...))

# Or in async function:
async def main():
    result = await quantizer.quantize_diffusion_model_async(...)
```

### Task Cancellation

To cancel a running quantization:

```python
task = asyncio.create_task(
    quantizer.quantize_diffusion_model_async(...)
)

# Cancel after 60 seconds
await asyncio.sleep(60)
task.cancel()

try:
    await task
except asyncio.CancelledError:
    print("Quantization cancelled")
```

## Examples

See `examples/async_quantization_example.py` for comprehensive examples demonstrating:

- Basic async quantization
- Progress tracking
- Concurrent multi-model quantization
- Async validation
- Deployment profiles
- Error handling
- Concurrent operations

Run the examples:

```bash
python examples/async_quantization_example.py
```

## See Also

- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Configuration Guide](CONFIGURATION_GUIDE.md) - Configuration options
- [Quickstart](QUICKSTART.md) - Getting started guide
- [Architecture](ARCHITECTURE.md) - System architecture overview

# ArrowQuant V2 User Guide

**Version**: 2.0  
**Last Updated**: 2026-02-26  
**Audience**: Data Scientists, ML Engineers, DevOps Engineers

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Configuration Guide](#configuration-guide)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)
9. [Use Cases](#use-cases)
10. [FAQ](#faq)

---

## Introduction

### What is ArrowQuant V2?

ArrowQuant V2 is a high-performance quantization library designed specifically for diffusion models. It reduces model size by 4-16x while maintaining quality, enabling deployment on resource-constrained devices and reducing inference costs.

### Key Features

- **Time-Aware Quantization**: Handles temporal variance in diffusion models
- **Spatial Quantization**: Optimizes activation maps with channel equalization
- **Zero-Copy Performance**: 29.5x faster than traditional approaches
- **Flexible Configuration**: Edge, local, and cloud deployment profiles
- **Production Ready**: Docker, Kubernetes, monitoring integration

### When to Use ArrowQuant V2

✅ **Use ArrowQuant V2 when you need to**:
- Deploy diffusion models on edge devices (mobile, IoT)
- Reduce cloud inference costs
- Speed up model loading and inference
- Maintain high quality with smaller models

❌ **Don't use ArrowQuant V2 for**:
- Non-diffusion models (use standard quantization)
- Models already quantized to INT4 or lower
- Scenarios requiring bit-exact reproduction

---

## Getting Started

### Installation

#### From PyPI (Recommended)

```bash
pip install arrow-quant-v2
```

#### From Source

```bash
git clone https://github.com/your-org/arrow-quant-v2.git
cd arrow-quant-v2
pip install -e .
```

#### System Requirements

**Minimum**:
- Python 3.10+
- 4GB RAM
- 2 CPU cores

**Recommended**:
- Python 3.11+
- 16GB RAM
- 8 CPU cores
- SSD storage

### Quick Start

```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

# Create quantizer
quantizer = ArrowQuantV2(mode="diffusion")

# Quantize a model
result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-int4/",
    config=DiffusionQuantConfig(bit_width=4)
)

print(f"✓ Compression: {result['compression_ratio']:.2f}x")
print(f"✓ Quality: {result['cosine_similarity']:.4f}")
print(f"✓ Size: {result['model_size_mb']:.2f} MB")
```

### Verify Installation

```python
import arrow_quant_v2
print(arrow_quant_v2.__version__)  # Should print: 2.0.0
```

---

## Basic Usage

### Quantizing Your First Model

#### Step 1: Prepare Your Model

Ensure your model is in one of these formats:
- PyTorch checkpoint (`.pt`, `.pth`)
- SafeTensors (`.safetensors`)
- HuggingFace model directory

```bash
# Example model structure
dream-7b/
├── config.json
├── model.safetensors
└── tokenizer.json
```

#### Step 2: Choose a Configuration

Select a deployment profile based on your target environment:

```python
from arrow_quant_v2 import DiffusionQuantConfig

# Edge deployment (mobile, IoT)
edge_config = DiffusionQuantConfig.from_profile("edge")
# → 2-bit, fast inference, 16x compression

# Local deployment (laptop, workstation)
local_config = DiffusionQuantConfig.from_profile("local")
# → 4-bit, balanced, 8x compression

# Cloud deployment (server, GPU)
cloud_config = DiffusionQuantConfig.from_profile("cloud")
# → 8-bit, high quality, 4x compression
```

#### Step 3: Quantize

```python
from arrow_quant_v2 import ArrowQuantV2

quantizer = ArrowQuantV2(mode="diffusion")

result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-int4/",
    config=local_config
)
```

#### Step 4: Validate Quality

```python
validation = quantizer.validate_quality(
    original_path="dream-7b/",
    quantized_path="dream-7b-int4/"
)

if validation['passed']:
    print(f"✓ Quality: {validation['cosine_similarity']:.4f}")
else:
    print("✗ Quality check failed")
    # Review per-layer accuracy
    for layer, quality in validation['per_layer_accuracy'].items():
        if quality < 0.80:
            print(f"  Low quality: {layer} ({quality:.4f})")
```

### Understanding Results

The quantization result contains:

```python
{
    'quantized_path': 'dream-7b-int4/',
    'compression_ratio': 8.2,           # 8.2x smaller
    'cosine_similarity': 0.87,          # 87% quality retained
    'model_size_mb': 850.5,             # Final size in MB
    'modality': 'text',                 # Detected modality
    'bit_width': 4,                     # Quantization bit width
    'quantization_time_s': 45.3         # Time taken
}
```

**Key Metrics**:
- **compression_ratio**: Higher is better (4-16x typical)
- **cosine_similarity**: Quality score (0.85+ is good, 0.90+ is excellent)
- **model_size_mb**: Final model size

---

## Advanced Features

### Progress Tracking

Monitor quantization progress with callbacks:

```python
def progress_callback(message, progress):
    """
    Args:
        message: Human-readable progress message
        progress: Float between 0.0 and 1.0
    """
    bar_length = 50
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r[{bar}] {progress*100:.1f}% - {message}", end='', flush=True)

result = quantizer.quantize_diffusion_model(
    model_path="model/",
    output_path="model-int4/",
    progress_callback=progress_callback
)
print()  # New line after completion
```

### Async Quantization

Quantize multiple models concurrently:

```python
import asyncio
from arrow_quant_v2 import AsyncArrowQuantV2, DiffusionQuantConfig

async def quantize_models():
    quantizer = AsyncArrowQuantV2()
    
    models = [
        ("model1/", "model1-int2/", DiffusionQuantConfig(bit_width=2)),
        ("model2/", "model2-int4/", DiffusionQuantConfig(bit_width=4)),
        ("model3/", "model3-int8/", DiffusionQuantConfig(bit_width=8)),
    ]
    
    results = await quantizer.quantize_multiple_models_async(models)
    
    for i, result in enumerate(results):
        print(f"Model {i+1}: {result['compression_ratio']:.2f}x")

asyncio.run(quantize_models())
```

### Batch Quantization with Arrow

Process multiple layers efficiently:

```python
import pyarrow as pa
import numpy as np

# Create input table
layers = []
for i in range(100):
    layers.append({
        'layer_name': f'layer_{i}',
        'weights': np.random.randn(1000).astype(np.float32).tolist(),
        'shape': [1000]
    })

table = pa.table({
    'layer_name': [l['layer_name'] for l in layers],
    'weights': [l['weights'] for l in layers],
    'shape': [l['shape'] for l in layers]
})

# Quantize in batch (37x faster than per-layer)
result = quantizer.quantize_batch_arrow(table)
print(f"Quantized {result.num_rows} layers in batch")
```

### SafeTensors Support

Quantize directly from SafeTensors:

```python
# Single file
result = quantizer.quantize_from_safetensors(
    safetensors_path="model.safetensors",
    output_path="model-int4/"
)

# Sharded files
result = quantizer.quantize_from_safetensors(
    safetensors_path="model-sharded/",  # Directory with shards
    output_path="model-int4/"
)
```

### Thermodynamic Optimization

Enable advanced quality optimization:

```python
from arrow_quant_v2 import (
    DiffusionQuantConfig,
    InterpolationMethod,
    BetaSchedule
)

config = DiffusionQuantConfig(
    bit_width=4,
    # Enable boundary smoothing (+2-3% accuracy)
    enable_boundary_smoothing=True,
    interpolation_method=InterpolationMethod.Cubic,
    
    # Enable transition optimization (+4-5% accuracy)
    enable_transition_optimization=True,
    markov_weight=0.1,
    entropy_weight=0.05,
    learning_rate=0.01,
    max_iterations=50,
    beta_schedule=BetaSchedule.Linear
)

result = quantizer.quantize_diffusion_model(
    model_path="model/",
    output_path="model-int4-optimized/",
    config=config
)
```

**Note**: Thermodynamic optimization adds 20-30% overhead but improves quality by 4-7%.

---

## Configuration Guide

### Deployment Profiles

#### Edge Profile

**Target**: Mobile devices, IoT, embedded systems

**Characteristics**:
- 2-bit quantization (16x compression)
- Minimal memory footprint (<35MB)
- Fast inference (<100ms)
- Quality: 0.65-0.75 cosine similarity

**Use when**:
- Memory < 4GB
- CPU-only inference
- Latency critical (<100ms)

```python
config = DiffusionQuantConfig.from_profile("edge")
```

#### Local Profile

**Target**: Laptops, workstations, local servers

**Characteristics**:
- 4-bit quantization (8x compression)
- Balanced memory/quality (<200MB)
- Good inference speed (100-500ms)
- Quality: 0.85-0.90 cosine similarity

**Use when**:
- Memory 8-16GB
- CPU or GPU inference
- Balanced requirements

```python
config = DiffusionQuantConfig.from_profile("local")
```

#### Cloud Profile

**Target**: Cloud servers, GPU clusters

**Characteristics**:
- 8-bit quantization (4x compression)
- High quality (<2GB)
- Slower inference (500ms-2s)
- Quality: 0.95+ cosine similarity

**Use when**:
- Memory > 32GB
- GPU inference
- Quality critical

```python
config = DiffusionQuantConfig.from_profile("cloud")
```

### Custom Configuration

Create custom configurations for specific needs:

```python
config = DiffusionQuantConfig(
    # Quantization settings
    bit_width=4,                    # 2, 4, or 8
    modality="text",                # text, code, image, audio
    
    # Time-aware settings
    num_time_groups=10,             # 5-20 typical
    enable_time_aware=True,
    
    # Spatial settings
    group_size=128,                 # 32, 64, 128, or 256
    enable_spatial=True,
    
    # Quality settings
    min_accuracy=0.85,              # 0.0-1.0
    calibration_samples=128,        # 32-512
    fail_fast=False,                # True to disable fallback
    
    # Deployment settings
    deployment_profile="local",
    num_threads=8,                  # CPU threads
    enable_streaming=False          # For large models
)
```

### Configuration from YAML

```yaml
# config.yaml
bit_width: 4
modality: text
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
```

```python
config = DiffusionQuantConfig.from_yaml("config.yaml")
```

### Environment Variables

Override configuration via environment variables:

```bash
export ARROW_QUANT_BIT_WIDTH=2
export ARROW_QUANT_MIN_ACCURACY=0.70
export ARROW_QUANT_NUM_THREADS=16
```

```python
# Automatically reads environment variables
config = DiffusionQuantConfig.from_profile("local")
# bit_width=2 from environment
```

---

## Performance Optimization

### Choosing the Right Bit Width

| Bit Width | Compression | Quality | Use Case |
|-----------|-------------|---------|----------|
| 2-bit | 16x | 0.65-0.75 | Edge devices, mobile |
| 4-bit | 8x | 0.85-0.90 | Local deployment, balanced |
| 8-bit | 4x | 0.95+ | Cloud, quality-critical |

**Rule of thumb**: Start with 4-bit, adjust based on quality requirements.

### Optimizing for Speed

```python
# Fast configuration
config = DiffusionQuantConfig(
    bit_width=2,                    # Faster quantization
    num_time_groups=5,              # Fewer groups
    group_size=256,                 # Larger groups
    calibration_samples=32,         # Fewer samples
    enable_transition_optimization=False,  # Skip optimization
    num_threads=16                  # Max parallelism
)
```

**Expected**: 2-3x faster quantization, 5-10% lower quality

### Optimizing for Quality

```python
# High-quality configuration
config = DiffusionQuantConfig(
    bit_width=8,                    # Higher precision
    num_time_groups=20,             # More groups
    group_size=64,                  # Smaller groups
    calibration_samples=512,        # More samples
    enable_transition_optimization=True,  # Enable optimization
    min_accuracy=0.95               # Strict threshold
)
```

**Expected**: 2-3x slower quantization, 5-10% higher quality

### Optimizing for Memory

```python
# Low-memory configuration
config = DiffusionQuantConfig(
    bit_width=2,                    # Minimal size
    enable_streaming=True,          # Stream large models
    calibration_samples=32,         # Fewer samples
    num_threads=4                   # Fewer threads
)
```

**Expected**: 50% lower peak memory usage

### Batch Processing

Use batch API for multiple layers:

```python
# Per-layer (slow)
for layer in layers:
    result = quantizer.quantize_layer(layer)  # N boundary crossings

# Batch (fast - 37x speedup)
result = quantizer.quantize_batch_arrow(table)  # 1 boundary crossing
```

### Parallel Quantization

Quantize multiple models concurrently:

```python
import asyncio

async def main():
    quantizer = AsyncArrowQuantV2()
    
    # Quantize 3 models in parallel
    models = [
        ("model1/", "model1-int4/", None),
        ("model2/", "model2-int4/", None),
        ("model3/", "model3-int4/", None),
    ]
    
    results = await quantizer.quantize_multiple_models_async(models)

asyncio.run(main())
```

---

## Troubleshooting

### Common Issues

#### Issue: "Quality check failed"

**Symptoms**: `cosine_similarity < min_accuracy`

**Solutions**:
1. Increase bit width (2 → 4 → 8)
2. Increase calibration samples
3. Enable thermodynamic optimization
4. Adjust `min_accuracy` threshold

```python
# Solution 1: Higher bit width
config = DiffusionQuantConfig(bit_width=8)

# Solution 2: More calibration
config = DiffusionQuantConfig(calibration_samples=512)

# Solution 3: Enable optimization
config = DiffusionQuantConfig(enable_transition_optimization=True)

# Solution 4: Lower threshold
config = DiffusionQuantConfig(min_accuracy=0.80)
```

#### Issue: "Out of memory"

**Symptoms**: `MemoryError` or system freeze

**Solutions**:
1. Enable streaming mode
2. Reduce calibration samples
3. Reduce number of threads
4. Use lower bit width

```python
config = DiffusionQuantConfig(
    enable_streaming=True,
    calibration_samples=32,
    num_threads=4,
    bit_width=2
)
```

#### Issue: "Quantization too slow"

**Symptoms**: Takes > 5 minutes for small models

**Solutions**:
1. Disable thermodynamic optimization
2. Reduce time groups
3. Increase group size
4. Use batch API

```python
config = DiffusionQuantConfig(
    enable_transition_optimization=False,
    num_time_groups=5,
    group_size=256
)
```

#### Issue: "Invalid schema error"

**Symptoms**: `ValueError: Invalid schema`

**Solution**: Ensure Arrow table has correct schema

```python
import pyarrow as pa

# Correct schema
schema = pa.schema([
    ('layer_name', pa.string()),
    ('weights', pa.list_(pa.float32())),
    ('shape', pa.list_(pa.int64()))  # Optional
])

table = pa.table({
    'layer_name': ['layer1'],
    'weights': [[1.0, 2.0, 3.0]],
    'shape': [[3]]
}, schema=schema)
```

### Debugging

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('arrow_quant_v2')
logger.setLevel(logging.DEBUG)

# Now quantization will print detailed logs
result = quantizer.quantize_diffusion_model(...)
```

### Getting Help

1. Check [API Reference](API_REFERENCE.md)
2. Review [Architecture](ARCHITECTURE.md)
3. Search [GitHub Issues](https://github.com/your-org/arrow-quant-v2/issues)
4. Ask on [Discord](https://discord.gg/arrow-quant)

---

## Best Practices

### 1. Always Validate Quality

```python
# Quantize
result = quantizer.quantize_diffusion_model(...)

# Validate
validation = quantizer.validate_quality(
    original_path="model/",
    quantized_path="model-int4/"
)

if not validation['passed']:
    print("Quality check failed - consider higher bit width")
```

### 2. Start with Profiles

```python
# Good: Use profiles as starting point
config = DiffusionQuantConfig.from_profile("local")

# Then customize if needed
config.min_accuracy = 0.90
config.calibration_samples = 256
```

### 3. Use Batch API for Multiple Layers

```python
# Good: Batch processing (37x faster)
result = quantizer.quantize_batch_arrow(table)

# Avoid: Per-layer processing
for layer in layers:
    result = quantizer.quantize_layer(layer)
```

### 4. Monitor Performance

```python
import time

start = time.time()
result = quantizer.quantize_diffusion_model(...)
elapsed = time.time() - start

print(f"Quantization took {elapsed:.2f}s")
print(f"Compression: {result['compression_ratio']:.2f}x")
print(f"Quality: {result['cosine_similarity']:.4f}")
```

### 5. Version Your Configurations

```yaml
# config-v1.yaml
version: "1.0"
bit_width: 4
min_accuracy: 0.85
deployment_profile: local
```

```python
config = DiffusionQuantConfig.from_yaml("config-v1.yaml")
```

### 6. Test on Representative Data

```python
# Test on a small subset first
test_result = quantizer.quantize_diffusion_model(
    model_path="model-subset/",
    output_path="model-subset-int4/"
)

if test_result['cosine_similarity'] >= 0.85:
    # Proceed with full model
    result = quantizer.quantize_diffusion_model(
        model_path="model-full/",
        output_path="model-full-int4/"
    )
```

### 7. Document Your Configurations

```python
# Good: Document why you chose these settings
config = DiffusionQuantConfig(
    bit_width=2,              # Edge deployment requirement
    min_accuracy=0.70,        # Acceptable for mobile
    calibration_samples=32,   # Memory constraint
    deployment_profile="edge"
)
```

---

## Use Cases

### Use Case 1: Mobile Deployment

**Scenario**: Deploy diffusion model on mobile app

**Requirements**:
- Model size < 50MB
- Inference < 100ms
- Acceptable quality (0.70+)

**Solution**:

```python
config = DiffusionQuantConfig.from_profile("edge")
config.bit_width = 2
config.min_accuracy = 0.70

result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-mobile/",
    config=config
)

# Expected: 16x compression, ~35MB, 0.70-0.75 quality
```

### Use Case 2: Cloud Cost Optimization

**Scenario**: Reduce cloud inference costs

**Requirements**:
- Maintain high quality (0.90+)
- Reduce model size by 4-8x
- Acceptable inference latency

**Solution**:

```python
config = DiffusionQuantConfig.from_profile("cloud")
config.bit_width = 4  # Balance size and quality
config.min_accuracy = 0.90
config.enable_transition_optimization = True

result = quantizer.quantize_diffusion_model(
    model_path="model/",
    output_path="model-int4/",
    config=config
)

# Expected: 8x compression, 0.90+ quality, 50% cost reduction
```

### Use Case 3: Batch Processing

**Scenario**: Quantize 100 models overnight

**Requirements**:
- Process multiple models
- Unattended operation
- Quality reports

**Solution**:

```python
import asyncio
from pathlib import Path

async def batch_quantize():
    quantizer = AsyncArrowQuantV2()
    
    # Find all models
    model_dirs = list(Path("models/").glob("*/"))
    
    models = [
        (str(model_dir), f"{model_dir}-int4/", None)
        for model_dir in model_dirs
    ]
    
    # Quantize all models
    results = await quantizer.quantize_multiple_models_async(models)
    
    # Generate report
    with open("quantization_report.txt", "w") as f:
        for model_dir, result in zip(model_dirs, results):
            f.write(f"{model_dir.name}:\n")
            f.write(f"  Compression: {result['compression_ratio']:.2f}x\n")
            f.write(f"  Quality: {result['cosine_similarity']:.4f}\n")
            f.write(f"  Size: {result['model_size_mb']:.2f} MB\n\n")

asyncio.run(batch_quantize())
```

### Use Case 4: A/B Testing

**Scenario**: Compare different quantization strategies

**Requirements**:
- Test multiple configurations
- Compare quality and size
- Select best configuration

**Solution**:

```python
configs = {
    "aggressive": DiffusionQuantConfig(bit_width=2, min_accuracy=0.65),
    "balanced": DiffusionQuantConfig(bit_width=4, min_accuracy=0.85),
    "conservative": DiffusionQuantConfig(bit_width=8, min_accuracy=0.95),
}

results = {}
for name, config in configs.items():
    result = quantizer.quantize_diffusion_model(
        model_path="model/",
        output_path=f"model-{name}/",
        config=config
    )
    results[name] = result

# Compare results
for name, result in results.items():
    print(f"{name}:")
    print(f"  Compression: {result['compression_ratio']:.2f}x")
    print(f"  Quality: {result['cosine_similarity']:.4f}")
    print(f"  Size: {result['model_size_mb']:.2f} MB")
```

---

## FAQ

### General Questions

**Q: What models are supported?**

A: ArrowQuant V2 supports diffusion models in PyTorch, SafeTensors, and HuggingFace formats. It's optimized for text, code, image, and audio diffusion models.

**Q: How much quality loss is acceptable?**

A: It depends on your use case:
- Edge: 0.65-0.75 (acceptable for mobile)
- Local: 0.85-0.90 (good for most applications)
- Cloud: 0.95+ (minimal quality loss)

**Q: Can I quantize non-diffusion models?**

A: ArrowQuant V2 is optimized for diffusion models. For other models, use standard quantization libraries like `torch.quantization`.

### Performance Questions

**Q: How long does quantization take?**

A: Typical times:
- Small model (1B params): 1-5 minutes
- Medium model (7B params): 5-15 minutes
- Large model (70B params): 30-60 minutes

**Q: Can I speed up quantization?**

A: Yes:
1. Use lower bit width (2-bit vs 8-bit)
2. Reduce calibration samples
3. Disable thermodynamic optimization
4. Increase CPU threads
5. Use batch API

**Q: Does quantization use GPU?**

A: No, quantization is CPU-only. However, inference with quantized models can use GPU.

### Quality Questions

**Q: Why is my quality score low?**

A: Common causes:
1. Bit width too low (try 4-bit or 8-bit)
2. Insufficient calibration samples
3. Model not suitable for quantization
4. Thermodynamic optimization disabled

**Q: Can I improve quality without increasing size?**

A: Yes:
1. Enable thermodynamic optimization (+4-7% quality)
2. Increase calibration samples
3. Adjust time groups and group size
4. Use custom configuration

**Q: What's the minimum acceptable quality?**

A: Depends on application:
- Critical applications: 0.95+
- Production applications: 0.85+
- Experimental/mobile: 0.65+

### Technical Questions

**Q: What is zero-copy quantization?**

A: Zero-copy means data is not copied between Python and Rust, using Arrow C Data Interface for direct memory access. This provides 29.5x speedup.

**Q: What is thermodynamic optimization?**

A: A technique that ensures smooth transitions between quantization groups, improving quality by 4-7% at the cost of 20-30% longer quantization time.

**Q: Can I use quantized models with PyTorch?**

A: Yes, quantized models can be loaded and used with PyTorch. See integration examples in the API Reference.

### Deployment Questions

**Q: How do I deploy quantized models?**

A: Quantized models are stored in Arrow/Parquet format. Load them using:

```python
import pyarrow.parquet as pq
table = pq.read_table("model-int4/quantized.parquet")
```

**Q: Can I use Docker?**

A: Yes, see [Deployment Architecture](ARCHITECTURE.md#deployment-architecture) for Docker and Kubernetes examples.

**Q: How do I monitor quantization in production?**

A: Use Prometheus metrics and Grafana dashboards. See [Monitoring](ARCHITECTURE.md#monitoring) section.

---

## Next Steps

### Learn More

- [API Reference](API_REFERENCE.md) - Detailed API documentation
- [Architecture](ARCHITECTURE.md) - System design and internals
- [Performance Benchmarks](PERFORMANCE_BENCHMARK_REPORT.md) - Performance analysis

### Get Involved

- [GitHub Repository](https://github.com/your-org/arrow-quant-v2)
- [Discord Community](https://discord.gg/arrow-quant)
- [Issue Tracker](https://github.com/your-org/arrow-quant-v2/issues)

### Support

- Email: support@arrow-quant.io
- Documentation: https://docs.arrow-quant.io
- Community Forum: https://forum.arrow-quant.io

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-26  
**Maintained By**: AI-OS Team  
**Review Status**: Approved

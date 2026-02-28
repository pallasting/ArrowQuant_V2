# ArrowQuant V2 Configuration Guide

This guide explains all configuration parameters for ArrowQuant V2 for Diffusion and provides practical recommendations for different deployment scenarios.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Parameters](#configuration-parameters)
3. [Deployment Profiles](#deployment-profiles)
4. [Environment Variables](#environment-variables)
5. [Tuning Recommendations](#tuning-recommendations)
6. [Validation Rules](#validation-rules)
7. [Examples](#examples)

## Quick Start

### Using Deployment Profiles

The easiest way to get started is using a deployment profile:

```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig, DeploymentProfile

# Edge device (2-4GB RAM)
config = DiffusionQuantConfig.from_profile(DeploymentProfile.Edge)

# Local machine (8+GB RAM)
config = DiffusionQuantConfig.from_profile(DeploymentProfile.Local)

# Cloud server (32+GB RAM)
config = DiffusionQuantConfig.from_profile(DeploymentProfile.Cloud)

quantizer = ArrowQuantV2(mode="diffusion")
result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-quantized/",
    config=config
)
```

### Using YAML Configuration

```bash
# Copy example config
cp config.example.yaml config.yaml

# Edit config.yaml with your settings
# Then load in Python:
```

```python
from arrow_quant_v2 import DiffusionQuantConfig

config = DiffusionQuantConfig.from_yaml("config.yaml")
config.apply_env_overrides()  # Apply environment variable overrides
config.validate()  # Validate configuration
```

## Configuration Parameters

### Core Parameters

#### `bit_width`
- **Type**: `int`
- **Valid values**: `2`, `4`, `8`
- **Default**: `4` (local profile)
- **Description**: Target bit width for quantization. Lower values provide better compression but may reduce accuracy.

**Recommendations**:
- `2`: Maximum compression for edge devices (10-16x compression)
- `4`: Balanced compression and accuracy for local deployment (4-8x compression)
- `8`: High accuracy for cloud deployment (2-4x compression)

**Example**:
```yaml
bit_width: 4
```

#### `modality`
- **Type**: `Optional[str]`
- **Valid values**: `"text"`, `"code"`, `"image"`, `"audio"`, `null`
- **Default**: `null` (auto-detect)
- **Description**: Modality of the diffusion model. When `null`, the system automatically detects modality from model metadata.

**Recommendations**:
- Leave as `null` for automatic detection
- Explicitly set if metadata is missing or incorrect
- Text/code models use discrete diffusion strategies
- Image/audio models use continuous diffusion strategies

**Example**:
```yaml
modality: text  # or null for auto-detection
```

### Time-Aware Quantization

#### `num_time_groups`
- **Type**: `int`
- **Valid range**: `1-100`
- **Default**: `10` (local profile)
- **Description**: Number of time groups for time-aware quantization. Groups similar timesteps together to handle temporal variance in diffusion models.

**Recommendations**:
- `5`: Edge devices (lower memory, faster processing)
- `10`: Local machines (balanced)
- `20`: Cloud servers (higher accuracy)
- Higher values improve accuracy but increase memory usage and processing time

**Example**:
```yaml
num_time_groups: 10
```

#### `enable_time_aware`
- **Type**: `bool`
- **Default**: `true`
- **Description**: Enable time-aware quantization for handling temporal variance in diffusion models.

**Recommendations**:
- `true`: For all diffusion models (recommended)
- `false`: Only for base mode or non-diffusion models

**Example**:
```yaml
enable_time_aware: true
```

### Spatial Quantization

#### `group_size`
- **Type**: `int`
- **Valid values**: `32`, `64`, `128`, `256`
- **Default**: `128` (local profile)
- **Description**: Group size for per-group quantization. Smaller groups provide finer granularity but increase metadata size.

**Recommendations**:
- `256`: Edge devices (coarser quantization, less metadata)
- `128`: Local machines (balanced)
- `64`: Cloud servers (finer quantization, better accuracy)
- `32`: Maximum accuracy (use only for critical layers)

**Example**:
```yaml
group_size: 128
```

#### `enable_spatial`
- **Type**: `bool`
- **Default**: `true` (local/cloud), `false` (edge)
- **Description**: Enable spatial quantization with channel equalization and activation smoothing.

**Recommendations**:
- `true`: For image/audio models and local/cloud deployments
- `false`: For edge devices to reduce computational overhead
- Provides significant accuracy improvements for continuous diffusion models

**Example**:
```yaml
enable_spatial: true
```

### Quality Control

#### `min_accuracy`
- **Type**: `float`
- **Valid range**: `0.0-1.0`
- **Default**: `0.85` (local profile)
- **Description**: Minimum cosine similarity threshold for quantization quality. If accuracy falls below this threshold, the system triggers fallback to higher bit width.

**Recommendations**:
- `0.65`: Edge devices (accept lower accuracy for size)
- `0.85`: Local machines (balanced)
- `0.95`: Cloud servers (high accuracy required)
- Set based on your application's tolerance for quality loss

**Example**:
```yaml
min_accuracy: 0.85
```

#### `calibration_samples`
- **Type**: `int`
- **Default**: `128` (local profile)
- **Description**: Number of calibration samples used to compute quantization parameters. More samples improve accuracy but increase processing time.

**Recommendations**:
- `32`: Edge devices (fast calibration)
- `128`: Local machines (balanced)
- `512`: Cloud servers (thorough calibration)
- Increase if quantization accuracy is below target

**Example**:
```yaml
calibration_samples: 128
```

### Deployment Configuration

#### `deployment_profile`
- **Type**: `str`
- **Valid values**: `"edge"`, `"local"`, `"cloud"`
- **Default**: `"local"`
- **Description**: Deployment profile that sets default values for all parameters.

**Profile Presets**:

| Parameter | Edge | Local | Cloud |
|-----------|------|-------|-------|
| `bit_width` | 2 | 4 | 8 |
| `num_time_groups` | 5 | 10 | 20 |
| `group_size` | 256 | 128 | 64 |
| `enable_time_aware` | true | true | true |
| `enable_spatial` | false | true | true |
| `min_accuracy` | 0.65 | 0.85 | 0.95 |
| `calibration_samples` | 32 | 128 | 512 |
| `enable_streaming` | true | false | false |

**Example**:
```yaml
deployment_profile: local
```

### Advanced Parameters

#### `fail_fast`
- **Type**: `bool`
- **Default**: `false`
- **Description**: When `true`, disables automatic fallback to higher bit widths on quantization failure. Useful for testing and debugging.

**Recommendations**:
- `false`: Production deployments (enable graceful degradation)
- `true`: Testing and validation (fail immediately to identify issues)

**Example**:
```yaml
fail_fast: false
```

#### `num_threads`
- **Type**: `int`
- **Default**: `0` (auto-detect)
- **Description**: Number of parallel threads for layer quantization. `0` uses all available CPU cores.

**Recommendations**:
- `0`: Auto-detect (recommended for most cases)
- `4-8`: Limit parallelism on shared systems
- `1`: Disable parallelism for debugging

**Example**:
```yaml
num_threads: 0  # Auto-detect
```

#### `enable_streaming`
- **Type**: `bool`
- **Default**: `true` (edge), `false` (local/cloud)
- **Description**: Enable streaming mode to load one layer at a time, minimizing memory usage at the cost of processing speed.

**Recommendations**:
- `true`: Edge devices or large models on limited RAM
- `false`: Local/cloud deployments for better performance
- Enable if you encounter out-of-memory errors

**Example**:
```yaml
enable_streaming: false
```

## Deployment Profiles

### Edge Profile

**Target**: Mobile devices, Raspberry Pi, edge servers (2-4GB RAM, ARM64)

**Characteristics**:
- Maximum compression (INT2)
- Minimal memory footprint
- Streaming mode enabled
- Reduced calibration samples
- Spatial quantization disabled

**Use Cases**:
- On-device text generation
- Embedded AI applications
- IoT devices
- Battery-powered devices

**Configuration**:
```yaml
deployment_profile: edge
bit_width: 2
num_time_groups: 5
group_size: 256
enable_time_aware: true
enable_spatial: false
min_accuracy: 0.65
calibration_samples: 32
enable_streaming: true
num_threads: 0
```

**Expected Results**:
- Dream 7B: ~35MB (16x compression)
- Cosine similarity: ≥0.65
- Quantization time: 2-5 minutes
- Memory usage: <2GB peak

### Local Profile

**Target**: Developer workstations, laptops (8+GB RAM, x86_64)

**Characteristics**:
- Balanced compression (INT4)
- Good accuracy
- Batch processing mode
- Full feature set enabled

**Use Cases**:
- Local development and testing
- Personal AI assistants
- Content creation tools
- Research and experimentation

**Configuration**:
```yaml
deployment_profile: local
bit_width: 4
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
enable_streaming: false
num_threads: 0
```

**Expected Results**:
- Dream 7B: ~70MB (8x compression)
- Cosine similarity: ≥0.85
- Quantization time: 5-10 minutes
- Memory usage: <4GB peak

### Cloud Profile

**Target**: Cloud servers, data centers (32+GB RAM, GPU)

**Characteristics**:
- High accuracy (INT8)
- Maximum calibration samples
- Fine-grained quantization
- All optimizations enabled

**Use Cases**:
- Production API services
- High-throughput inference
- Quality-critical applications
- Research and benchmarking

**Configuration**:
```yaml
deployment_profile: cloud
bit_width: 8
num_time_groups: 20
group_size: 64
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.95
calibration_samples: 512
enable_streaming: false
num_threads: 0
```

**Expected Results**:
- Dream 7B: ~140MB (4x compression)
- Cosine similarity: ≥0.95
- Quantization time: 10-20 minutes
- Memory usage: <8GB peak

## Environment Variables

You can override any configuration parameter using environment variables. This is useful for CI/CD pipelines, containerized deployments, and dynamic configuration.

### Supported Variables

| Variable | Type | Example |
|----------|------|---------|
| `ARROW_QUANT_BIT_WIDTH` | int | `2`, `4`, `8` |
| `ARROW_QUANT_NUM_TIME_GROUPS` | int | `5`, `10`, `20` |
| `ARROW_QUANT_GROUP_SIZE` | int | `32`, `64`, `128`, `256` |
| `ARROW_QUANT_MIN_ACCURACY` | float | `0.65`, `0.85`, `0.95` |
| `ARROW_QUANT_CALIBRATION_SAMPLES` | int | `32`, `128`, `512` |
| `ARROW_QUANT_FAIL_FAST` | bool | `true`, `false` |
| `ARROW_QUANT_NUM_THREADS` | int | `0`, `4`, `8` |
| `ARROW_QUANT_ENABLE_STREAMING` | bool | `true`, `false` |

### Usage Example

```bash
# Set environment variables
export ARROW_QUANT_BIT_WIDTH=2
export ARROW_QUANT_NUM_TIME_GROUPS=5
export ARROW_QUANT_MIN_ACCURACY=0.70
export ARROW_QUANT_ENABLE_STREAMING=true

# Run quantization (will use env overrides)
python scripts/quantize_diffusion.py \
    --model dream-7b/ \
    --output dream-7b-int2/ \
    --config config.yaml
```

```python
# In Python
from arrow_quant_v2 import DiffusionQuantConfig

# Load base config from YAML
config = DiffusionQuantConfig.from_yaml("config.yaml")

# Apply environment variable overrides
config.apply_env_overrides()

# Validate final configuration
config.validate()
```

### Docker Example

```dockerfile
FROM python:3.10-slim

# Set quantization configuration via environment
ENV ARROW_QUANT_BIT_WIDTH=2
ENV ARROW_QUANT_DEPLOYMENT_PROFILE=edge
ENV ARROW_QUANT_ENABLE_STREAMING=true

COPY . /app
WORKDIR /app

CMD ["python", "scripts/quantize_diffusion.py"]
```

## Tuning Recommendations

### Scenario 1: Maximum Compression

**Goal**: Smallest possible model size

**Configuration**:
```yaml
bit_width: 2
num_time_groups: 5
group_size: 256
enable_time_aware: true
enable_spatial: false
min_accuracy: 0.60  # Lower threshold
calibration_samples: 32
enable_streaming: true
```

**Trade-offs**:
- ✅ 10-16x compression
- ✅ Minimal memory usage
- ❌ Lower accuracy (0.60-0.70)
- ❌ May require fallback to INT4

### Scenario 2: Maximum Accuracy

**Goal**: Highest quality quantization

**Configuration**:
```yaml
bit_width: 8
num_time_groups: 20
group_size: 64
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.95
calibration_samples: 512
enable_streaming: false
```

**Trade-offs**:
- ✅ High accuracy (0.95+)
- ✅ Minimal quality loss
- ❌ Larger model size (4x compression)
- ❌ Longer quantization time

### Scenario 3: Fast Quantization

**Goal**: Minimize quantization time

**Configuration**:
```yaml
bit_width: 4
num_time_groups: 5
group_size: 128
enable_time_aware: true
enable_spatial: false
min_accuracy: 0.80
calibration_samples: 32
enable_streaming: false
num_threads: 0  # Use all cores
```

**Trade-offs**:
- ✅ Fast processing (2-5 minutes)
- ✅ Reasonable accuracy
- ❌ Slightly lower quality than full calibration

### Scenario 4: Memory-Constrained

**Goal**: Quantize large models on limited RAM

**Configuration**:
```yaml
bit_width: 4
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 64
enable_streaming: true  # Critical for memory
num_threads: 2  # Limit parallelism
```

**Trade-offs**:
- ✅ Low memory usage (<2GB peak)
- ✅ Can quantize large models
- ❌ Slower processing (streaming overhead)

### Scenario 5: Text/Code Models

**Goal**: Optimize for discrete diffusion (Dream 7B)

**Configuration**:
```yaml
modality: text  # or code
bit_width: 2
num_time_groups: 10
group_size: 128
enable_time_aware: true  # Critical for text
enable_spatial: false  # Less important for discrete
min_accuracy: 0.70
calibration_samples: 128
```

**Rationale**:
- Time-aware quantization handles temporal variance in discrete diffusion
- Spatial quantization less critical for text/code
- INT2 achieves good compression with acceptable accuracy

### Scenario 6: Image/Audio Models

**Goal**: Optimize for continuous diffusion

**Configuration**:
```yaml
modality: image  # or audio
bit_width: 4
num_time_groups: 10
group_size: 64
enable_time_aware: true
enable_spatial: true  # Critical for continuous
min_accuracy: 0.85
calibration_samples: 256
```

**Rationale**:
- Spatial quantization handles channel variance in continuous diffusion
- Finer group size (64) for better spatial resolution
- INT4 balances compression and perceptual quality

## Validation Rules

The configuration system validates all parameters to prevent invalid settings. Here are the validation rules and error messages:

### Bit Width Validation

**Rule**: `bit_width` must be 2, 4, or 8

**Error**: `InvalidBitWidth(value)`

```python
# Invalid
config.bit_width = 3  # Error: InvalidBitWidth(3)

# Valid
config.bit_width = 4  # OK
```

### Time Groups Validation

**Rule**: `num_time_groups` must be between 1 and 100

**Error**: `InvalidTimeGroups(value)`

```python
# Invalid
config.num_time_groups = 0    # Error: InvalidTimeGroups(0)
config.num_time_groups = 150  # Error: InvalidTimeGroups(150)

# Valid
config.num_time_groups = 10  # OK
```

### Group Size Validation

**Rule**: `group_size` must be 32, 64, 128, or 256

**Error**: `InvalidGroupSize(value)`

```python
# Invalid
config.group_size = 100  # Error: InvalidGroupSize(100)

# Valid
config.group_size = 128  # OK
```

### Accuracy Validation

**Rule**: `min_accuracy` must be between 0.0 and 1.0

**Error**: `InvalidAccuracy(value)`

```python
# Invalid
config.min_accuracy = 1.5   # Error: InvalidAccuracy(1.5)
config.min_accuracy = -0.1  # Error: InvalidAccuracy(-0.1)

# Valid
config.min_accuracy = 0.85  # OK
```

### Automatic Validation

Configuration is automatically validated when:
- Loading from YAML: `DiffusionQuantConfig.from_yaml()`
- Creating from profile: `DiffusionQuantConfig.from_profile()`
- Manual validation: `config.validate()`

```python
# Validation on load
try:
    config = DiffusionQuantConfig.from_yaml("config.yaml")
except QuantError as e:
    print(f"Configuration error: {e}")

# Manual validation
config = DiffusionQuantConfig.default()
config.bit_width = 3  # Invalid
try:
    config.validate()
except QuantError as e:
    print(f"Validation failed: {e}")
```

## Examples

### Example 1: Edge Deployment

Quantize Dream 7B for Raspberry Pi deployment:

```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig, DeploymentProfile

# Create edge configuration
config = DiffusionQuantConfig.from_profile(DeploymentProfile.Edge)

# Optional: Further customize
config.min_accuracy = 0.70  # Slightly higher threshold
config.calibration_samples = 64  # More samples

# Validate
config.validate()

# Quantize
quantizer = ArrowQuantV2(mode="diffusion")
result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-edge/",
    config=config
)

print(f"Model size: {result['model_size_mb']:.1f}MB")
print(f"Accuracy: {result['cosine_similarity']:.3f}")
print(f"Compression: {result['compression_ratio']:.1f}x")
```

### Example 2: Custom YAML Configuration

Create a custom configuration file:

```yaml
# custom_config.yaml
bit_width: 4
modality: text
num_time_groups: 15
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.80
calibration_samples: 256
deployment_profile: local
fail_fast: false
num_threads: 8
enable_streaming: false
```

Load and use:

```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

# Load custom config
config = DiffusionQuantConfig.from_yaml("custom_config.yaml")

# Apply environment overrides
config.apply_env_overrides()

# Quantize
quantizer = ArrowQuantV2(mode="diffusion")
result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-custom/",
    config=config
)
```

### Example 3: Dynamic Configuration

Adjust configuration based on available resources:

```python
import psutil
from arrow_quant_v2 import DiffusionQuantConfig, DeploymentProfile

# Detect available RAM
available_ram_gb = psutil.virtual_memory().available / (1024**3)

# Select profile based on RAM
if available_ram_gb < 4:
    profile = DeploymentProfile.Edge
    print("Using Edge profile (low RAM)")
elif available_ram_gb < 16:
    profile = DeploymentProfile.Local
    print("Using Local profile (medium RAM)")
else:
    profile = DeploymentProfile.Cloud
    print("Using Cloud profile (high RAM)")

config = DiffusionQuantConfig.from_profile(profile)

# Adjust streaming based on RAM
if available_ram_gb < 6:
    config.enable_streaming = True
    print("Enabling streaming mode")

# Quantize with dynamic config
quantizer = ArrowQuantV2(mode="diffusion")
result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-quantized/",
    config=config
)
```

### Example 4: Batch Quantization

Quantize multiple models with different configurations:

```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

quantizer = ArrowQuantV2(mode="diffusion")

models = [
    {
        "name": "dream-7b-edge",
        "input": "dream-7b/",
        "bit_width": 2,
        "profile": "edge"
    },
    {
        "name": "dream-7b-local",
        "input": "dream-7b/",
        "bit_width": 4,
        "profile": "local"
    },
    {
        "name": "dream-7b-cloud",
        "input": "dream-7b/",
        "bit_width": 8,
        "profile": "cloud"
    }
]

for model in models:
    print(f"\nQuantizing {model['name']}...")
    
    config = DiffusionQuantConfig.from_profile(model["profile"])
    config.bit_width = model["bit_width"]
    
    result = quantizer.quantize_diffusion_model(
        model_path=model["input"],
        output_path=f"models/{model['name']}/",
        config=config
    )
    
    print(f"  Size: {result['model_size_mb']:.1f}MB")
    print(f"  Accuracy: {result['cosine_similarity']:.3f}")
```

### Example 5: Fallback Testing

Test fallback behavior with strict accuracy requirements:

```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

quantizer = ArrowQuantV2(mode="diffusion")

# Start with aggressive INT2 settings
config = DiffusionQuantConfig(
    bit_width=2,
    num_time_groups=5,
    group_size=256,
    enable_time_aware=True,
    enable_spatial=False,
    min_accuracy=0.75,  # High threshold for INT2
    calibration_samples=32,
    fail_fast=False  # Enable fallback
)

result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-fallback-test/",
    config=config
)

# Check if fallback occurred
if result.get("fallback_occurred"):
    print(f"Fallback to INT{result['actual_bit_width']}")
    print(f"Reason: {result['fallback_reason']}")
else:
    print("INT2 quantization succeeded!")

print(f"Final accuracy: {result['cosine_similarity']:.3f}")
```

## Troubleshooting

### Issue: Out of Memory

**Symptoms**: Process killed, OOM errors

**Solutions**:
1. Enable streaming mode: `enable_streaming: true`
2. Reduce calibration samples: `calibration_samples: 32`
3. Limit parallelism: `num_threads: 2`
4. Use edge profile: `deployment_profile: edge`

### Issue: Low Accuracy

**Symptoms**: Cosine similarity below threshold

**Solutions**:
1. Increase bit width: `bit_width: 4` or `8`
2. Increase time groups: `num_time_groups: 15` or `20`
3. Decrease group size: `group_size: 64`
4. Enable spatial quantization: `enable_spatial: true`
5. Increase calibration samples: `calibration_samples: 256`

### Issue: Slow Quantization

**Symptoms**: Takes too long to complete

**Solutions**:
1. Disable streaming: `enable_streaming: false`
2. Reduce time groups: `num_time_groups: 5`
3. Reduce calibration samples: `calibration_samples: 32`
4. Disable spatial quantization: `enable_spatial: false`
5. Use all CPU cores: `num_threads: 0`

### Issue: Fallback Always Triggered

**Symptoms**: Always falls back to higher bit width

**Solutions**:
1. Lower accuracy threshold: `min_accuracy: 0.70`
2. Increase calibration samples: `calibration_samples: 256`
3. Enable all optimizations: `enable_time_aware: true`, `enable_spatial: true`
4. Check model compatibility (some models may not support INT2)

## Best Practices

1. **Start with a profile**: Use deployment profiles as a starting point, then customize
2. **Validate early**: Always call `config.validate()` before quantization
3. **Use environment variables**: For CI/CD and containerized deployments
4. **Monitor memory**: Enable streaming for large models or limited RAM
5. **Test fallback**: Disable `fail_fast` in production for graceful degradation
6. **Benchmark first**: Test different configurations on a small model before production
7. **Document custom configs**: Save and version control your YAML configurations
8. **Use auto-detection**: Let the system detect modality unless you have specific requirements

## Additional Resources

- [Quickstart Guide](QUICKSTART.md) - Get started quickly
- [API Reference](API_REFERENCE.md) - Detailed API documentation
- [Design Document](../.kiro/specs/arrowquant-v2-diffusion/design.md) - Architecture and implementation details
- [Example Config](../config.example.yaml) - Annotated configuration template

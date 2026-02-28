# ArrowQuant V2 API Reference

**Version**: 2.0  
**Last Updated**: 2026-02-26

## Table of Contents

1. [Overview](#overview)
2. [Python API](#python-api)
   - [Core Classes](#core-classes)
   - [Configuration](#configuration)
   - [Async API](#async-api)
3. [Rust API](#rust-api)
4. [Arrow FFI Integration](#arrow-ffi-integration)
5. [Examples](#examples)

---

## Overview

ArrowQuant V2 is a high-performance quantization library for diffusion models with:
- Time-aware quantization (temporal variance handling)
- Spatial quantization (channel equalization, activation smoothing)
- Extended Parquet V2 schema with diffusion metadata
- PyO3 Python bindings with zero-copy Arrow integration
- Async support for non-blocking operations

---

## Python API

### Core Classes

#### `ArrowQuantV2`

Main synchronous quantization interface.

```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

# Create instance
quantizer = ArrowQuantV2(mode="diffusion")  # or mode="base"
```

##### Methods

###### `quantize_diffusion_model()`

Quantize a diffusion model with diffusion-specific optimizations.

**Signature:**
```python
def quantize_diffusion_model(
    model_path: str,
    output_path: str,
    config: Optional[DiffusionQuantConfig] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Any]
```

**Parameters:**
- `model_path` (str): Path to input model directory
- `output_path` (str): Path to output quantized model directory
- `config` (Optional[DiffusionQuantConfig]): Quantization configuration (uses default if None)
- `progress_callback` (Optional[Callable]): Progress callback function
  - Signature: `fn(message: str, progress: float) -> None`
  - `message`: Human-readable progress message
  - `progress`: Float between 0.0 and 1.0 indicating completion

**Returns:**
Dictionary containing:
- `quantized_path` (str): Path to quantized model
- `compression_ratio` (float): Compression ratio achieved
- `cosine_similarity` (float): Average cosine similarity
- `model_size_mb` (float): Size of quantized model in MB
- `modality` (str): Detected modality (text, code, image, audio)
- `bit_width` (int): Bit width used for quantization
- `quantization_time_s` (float): Time taken for quantization in seconds

**Raises:**
- `QuantizationError`: If quantization fails
- `ConfigurationError`: If configuration is invalid

**Example:**
```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

quantizer = ArrowQuantV2(mode="diffusion")

def progress_handler(message, progress):
    print(f"[{progress*100:.1f}%] {message}")

result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-int4/",
    config=DiffusionQuantConfig(bit_width=4),
    progress_callback=progress_handler
)

print(f"Compression: {result['compression_ratio']:.2f}x")
print(f"Quality: {result['cosine_similarity']:.4f}")
```

---

###### `validate_quality()`

Validate quantization quality by comparing original and quantized models.

**Signature:**
```python
def validate_quality(
    original_path: str,
    quantized_path: str
) -> Dict[str, Any]
```

**Parameters:**
- `original_path` (str): Path to original model directory
- `quantized_path` (str): Path to quantized model directory

**Returns:**
Dictionary containing:
- `cosine_similarity` (float): Average cosine similarity across layers
- `compression_ratio` (float): Compression ratio achieved
- `model_size_mb` (float): Size of quantized model in MB
- `per_layer_accuracy` (Dict[str, float]): Per-layer cosine similarities
- `passed` (bool): Whether validation passed

**Raises:**
- `ValidationError`: If validation fails

**Example:**
```python
validation = quantizer.validate_quality(
    original_path="dream-7b/",
    quantized_path="dream-7b-int4/"
)

if validation['passed']:
    print(f"✓ Validation passed: {validation['cosine_similarity']:.4f}")
else:
    print(f"✗ Validation failed")
```

---

###### `quantize_from_safetensors()`

Quantize a model directly from SafeTensors format.

**Signature:**
```python
def quantize_from_safetensors(
    safetensors_path: str,
    output_path: str,
    config: Optional[DiffusionQuantConfig] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Any]
```

**Parameters:**
- `safetensors_path` (str): Path to .safetensors file or directory with sharded files
- `output_path` (str): Path to output quantized model directory
- `config` (Optional[DiffusionQuantConfig]): Quantization configuration
- `progress_callback` (Optional[Callable]): Progress callback function

**Returns:**
Same as `quantize_diffusion_model()`

**Example:**
```python
result = quantizer.quantize_from_safetensors(
    safetensors_path="model.safetensors",
    output_path="model-int4/"
)
```

---

###### `quantize_batch_arrow()`

Quantize multiple layers in batch using Arrow zero-copy interface.

**Signature:**
```python
def quantize_batch_arrow(
    table: pyarrow.Table,
    config: Optional[DiffusionQuantConfig] = None
) -> pyarrow.RecordBatch
```

**Parameters:**
- `table` (pyarrow.Table): Input table with schema:
  - `layer_name`: string (required)
  - `weights`: list<float32> (required)
  - `shape`: list<int64> (optional)
- `config` (Optional[DiffusionQuantConfig]): Quantization configuration

**Returns:**
`pyarrow.RecordBatch` with quantized results

**Raises:**
- `ValueError`: If schema is invalid
- `QuantizationError`: If quantization fails

**Example:**
```python
import pyarrow as pa
import numpy as np

# Create input table
table = pa.table({
    'layer_name': ['layer1', 'layer2'],
    'weights': [
        np.random.randn(1000).astype(np.float32).tolist(),
        np.random.randn(2000).astype(np.float32).tolist()
    ]
})

# Quantize in batch
result = quantizer.quantize_batch_arrow(table)
```

---

### Configuration

#### `DiffusionQuantConfig`

Configuration for diffusion model quantization.

**Constructor:**
```python
DiffusionQuantConfig(
    bit_width: int = 4,
    modality: Optional[str] = None,
    num_time_groups: int = 10,
    group_size: int = 128,
    enable_time_aware: bool = True,
    enable_spatial: bool = True,
    min_accuracy: float = 0.85,
    calibration_samples: int = 128,
    deployment_profile: str = "local",
    fail_fast: bool = False,
    enable_transition_optimization: bool = False,
    markov_weight: float = 0.1,
    entropy_weight: float = 0.05,
    learning_rate: float = 0.01,
    max_iterations: int = 50,
    convergence_threshold: float = 1e-4,
    beta_schedule: str = "linear"
)
```

**Parameters:**
- `bit_width` (int): Target bit width (2, 4, or 8), default is 4
- `modality` (Optional[str]): Modality ("text", "code", "image", "audio"), auto-detected if None
- `num_time_groups` (int): Number of time groups for time-aware quantization, default is 10
- `group_size` (int): Group size for per-group quantization (32, 64, 128, or 256), default is 128
- `enable_time_aware` (bool): Enable time-aware quantization, default is True
- `enable_spatial` (bool): Enable spatial quantization, default is True
- `min_accuracy` (float): Minimum cosine similarity threshold (0.0-1.0), default is 0.85
- `calibration_samples` (int): Number of calibration samples, default is 128
- `deployment_profile` (str): Deployment profile ("edge", "local", "cloud"), default is "local"
- `fail_fast` (bool): Disable fallback and fail immediately on quality threshold failure, default is False
- `enable_transition_optimization` (bool): Enable Phase 3 transition optimization, default is False
- `markov_weight` (float): Weight for Markov constraint loss (Phase 3), default is 0.1
- `entropy_weight` (float): Weight for entropy regularization (Phase 3), default is 0.05
- `learning_rate` (float): Learning rate for gradient descent (Phase 3), default is 0.01
- `max_iterations` (int): Maximum optimization iterations (Phase 3), default is 50
- `convergence_threshold` (float): Convergence threshold for early stopping (Phase 3), default is 1e-4
- `beta_schedule` (str): Beta schedule type ("linear" or "cosine") for Phase 3, default is "linear"

**Class Methods:**

###### `from_profile()`

Create configuration from deployment profile.

```python
@staticmethod
def from_profile(profile: str) -> DiffusionQuantConfig
```

**Parameters:**
- `profile` (str): Deployment profile ("edge", "local", or "cloud")

**Returns:**
`DiffusionQuantConfig` instance with profile-specific defaults

**Profile Defaults:**

| Parameter | Edge | Local | Cloud |
|-----------|------|-------|-------|
| bit_width | 2 | 4 | 8 |
| num_time_groups | 5 | 10 | 20 |
| group_size | 256 | 128 | 64 |
| min_accuracy | 0.65 | 0.85 | 0.95 |
| calibration_samples | 32 | 128 | 512 |
| enable_streaming | True | False | False |

**Example:**
```python
# Edge deployment (low memory, fast inference)
edge_config = DiffusionQuantConfig.from_profile("edge")

# Local deployment (balanced)
local_config = DiffusionQuantConfig.from_profile("local")

# Cloud deployment (high quality)
cloud_config = DiffusionQuantConfig.from_profile("cloud")
```

---

### Async API

#### `AsyncArrowQuantV2`

Asynchronous quantization interface for non-blocking operations.

```python
from arrow_quant_v2 import AsyncArrowQuantV2
import asyncio

# Create instance
quantizer = AsyncArrowQuantV2()
```

##### Methods

###### `quantize_diffusion_model_async()`

Asynchronously quantize a diffusion model.

**Signature:**
```python
async def quantize_diffusion_model_async(
    model_path: str,
    output_path: str,
    config: Optional[DiffusionQuantConfig] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Any]
```

**Parameters:**
Same as synchronous `quantize_diffusion_model()`

**Returns:**
Coroutine that resolves to the same dictionary as synchronous version

**Example:**
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

asyncio.run(main())
```

---

###### `quantize_multiple_models_async()`

Asynchronously quantize multiple models concurrently.

**Signature:**
```python
async def quantize_multiple_models_async(
    models: List[Tuple[str, str, Optional[DiffusionQuantConfig]]],
    progress_callback: Optional[Callable[[int, str, float], None]] = None
) -> List[Dict[str, Any]]
```

**Parameters:**
- `models` (List[Tuple]): List of tuples (model_path, output_path, config)
- `progress_callback` (Optional[Callable]): Progress callback
  - Signature: `fn(model_idx: int, message: str, progress: float) -> None`

**Returns:**
Coroutine that resolves to a list of result dictionaries

**Example:**
```python
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

---

###### `validate_quality_async()`

Asynchronously validate quantization quality.

**Signature:**
```python
async def validate_quality_async(
    original_path: str,
    quantized_path: str
) -> Dict[str, Any]
```

**Parameters:**
Same as synchronous `validate_quality()`

**Returns:**
Coroutine that resolves to validation results dictionary

**Example:**
```python
async def main():
    quantizer = AsyncArrowQuantV2()
    
    validation = await quantizer.validate_quality_async(
        original_path="dream-7b/",
        quantized_path="dream-7b-int4/"
    )
    
    print(f"Quality: {validation['cosine_similarity']:.4f}")

asyncio.run(main())
```

---

## Rust API

### Core Types

#### `DiffusionOrchestrator`

Main orchestrator for diffusion model quantization.

```rust
use arrow_quant_v2::{DiffusionOrchestrator, DiffusionQuantConfig};

let config = DiffusionQuantConfig::default();
let orchestrator = DiffusionOrchestrator::new(config)?;
```

##### Methods

###### `quantize_model()`

Quantize a diffusion model.

```rust
pub fn quantize_model(
    &self,
    model_path: &Path,
    output_path: &Path
) -> Result<QuantizationResult>
```

**Example:**
```rust
use std::path::PathBuf;
use arrow_quant_v2::{DiffusionOrchestrator, DiffusionQuantConfig};

let config = DiffusionQuantConfig::default();
let orchestrator = DiffusionOrchestrator::new(config)?;

let result = orchestrator.quantize_model(
    &PathBuf::from("dream-7b/"),
    &PathBuf::from("dream-7b-int4/")
)?;

println!("Compression: {:.2}x", result.compression_ratio);
```

---

#### `DiffusionQuantConfig`

Configuration struct for quantization.

```rust
pub struct DiffusionQuantConfig {
    pub bit_width: u8,
    pub modality: Option<Modality>,
    pub num_time_groups: usize,
    pub group_size: usize,
    pub enable_time_aware: bool,
    pub enable_spatial: bool,
    pub min_accuracy: f32,
    pub calibration_samples: usize,
    pub deployment_profile: DeploymentProfile,
    pub fail_fast: bool,
    pub num_threads: usize,
    pub enable_streaming: bool,
    pub thermodynamic: ThermodynamicConfig,
}
```

##### Methods

###### `from_profile()`

Create configuration from deployment profile.

```rust
pub fn from_profile(profile: DeploymentProfile) -> Self
```

**Example:**
```rust
use arrow_quant_v2::{DiffusionQuantConfig, DeploymentProfile};

let edge_config = DiffusionQuantConfig::from_profile(DeploymentProfile::Edge);
let local_config = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);
let cloud_config = DiffusionQuantConfig::from_profile(DeploymentProfile::Cloud);
```

---

###### `validate()`

Validate configuration parameters.

```rust
pub fn validate(&self) -> Result<()>
```

**Example:**
```rust
let config = DiffusionQuantConfig::default();
config.validate()?;
```

---

###### `from_yaml()`

Load configuration from YAML file.

```rust
pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self>
```

**Example:**
```rust
let config = DiffusionQuantConfig::from_yaml("config.yaml")?;
```

---

###### `to_yaml()`

Save configuration to YAML file.

```rust
pub fn to_yaml<P: AsRef<Path>>(&self, path: P) -> Result<()>
```

**Example:**
```rust
let config = DiffusionQuantConfig::default();
config.to_yaml("config.yaml")?;
```

---

### Enums

#### `Modality`

Modality of the diffusion model.

```rust
pub enum Modality {
    Text,
    Code,
    Image,
    Audio,
}
```

---

#### `DeploymentProfile`

Deployment profile presets.

```rust
pub enum DeploymentProfile {
    Edge,   // Low memory, fast inference
    Local,  // Balanced
    Cloud,  // High quality
}
```

---

#### `InterpolationMethod`

Interpolation method for boundary smoothing.

```rust
pub enum InterpolationMethod {
    Linear,   // Fast, C⁰ continuity
    Cubic,    // Smooth, C² continuity
    Sigmoid,  // Gradual transitions
}
```

---

#### `BetaSchedule`

Beta schedule type for diffusion process.

```rust
pub enum BetaSchedule {
    Linear,  // Linear schedule
    Cosine,  // Cosine schedule
}
```

---

### Configuration Structs

#### `ThermodynamicConfig`

Configuration for thermodynamic constraints.

```rust
pub struct ThermodynamicConfig {
    pub validation: ValidationConfig,
    pub boundary_smoothing: BoundarySmoothingConfig,
    pub transition_optimization: TransitionOptimizationConfig,
}
```

---

#### `ValidationConfig`

Configuration for Markov validation (Phase 1).

```rust
pub struct ValidationConfig {
    pub enabled: bool,
    pub smoothness_threshold: f32,
    pub log_violations: bool,
}
```

---

#### `BoundarySmoothingConfig`

Configuration for boundary smoothing (Phase 2).

```rust
pub struct BoundarySmoothingConfig {
    pub enabled: bool,
    pub window_size: usize,
    pub interpolation: InterpolationMethod,
}
```

---

#### `TransitionOptimizationConfig`

Configuration for transition optimization (Phase 3).

```rust
pub struct TransitionOptimizationConfig {
    pub enabled: bool,
    pub markov_weight: f32,
    pub entropy_weight: f32,
    pub learning_rate: f32,
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub beta_schedule: BetaSchedule,
}
```

---

## Arrow FFI Integration

### Zero-Copy PyArrow Integration

ArrowQuant V2 provides zero-copy integration with PyArrow using the Arrow C Data Interface.

#### Import PyArrow Array

```python
import pyarrow as pa
import numpy as np
from arrow_quant_v2 import ArrowQuantV2

# Create PyArrow array
arr = pa.array(np.random.randn(1000).astype(np.float32))

# Zero-copy import to Rust (no data copying)
quantizer = ArrowQuantV2()
# Internal: uses Arrow C Data Interface for zero-copy access
```

---

#### Export to PyArrow

```python
# Quantize and get result as PyArrow RecordBatch
result = quantizer.quantize_batch_arrow(table)

# Result is a PyArrow RecordBatch (zero-copy)
print(result.schema)
print(result.num_rows)
```

---

#### Schema Validation

The `quantize_batch_arrow()` method expects the following schema:

**Required Fields:**
- `layer_name`: string
- `weights`: list<float32>

**Optional Fields:**
- `shape`: list<int64>

**Example:**
```python
import pyarrow as pa

# Valid schema
schema = pa.schema([
    ('layer_name', pa.string()),
    ('weights', pa.list_(pa.float32())),
    ('shape', pa.list_(pa.int64()))
])

table = pa.table({
    'layer_name': ['layer1'],
    'weights': [[1.0, 2.0, 3.0]],
    'shape': [[3]]
}, schema=schema)
```

---

## Examples

### Basic Quantization

```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

# Create quantizer
quantizer = ArrowQuantV2(mode="diffusion")

# Quantize model
result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-int4/",
    config=DiffusionQuantConfig(bit_width=4)
)

print(f"✓ Quantization complete")
print(f"  Compression: {result['compression_ratio']:.2f}x")
print(f"  Quality: {result['cosine_similarity']:.4f}")
print(f"  Size: {result['model_size_mb']:.2f} MB")
```

---

### Custom Configuration

```python
from arrow_quant_v2 import DiffusionQuantConfig

# Create custom configuration
config = DiffusionQuantConfig(
    bit_width=2,
    modality="text",
    num_time_groups=20,
    group_size=64,
    enable_time_aware=True,
    enable_spatial=True,
    min_accuracy=0.90,
    calibration_samples=256,
    deployment_profile="cloud"
)

# Quantize with custom config
result = quantizer.quantize_diffusion_model(
    model_path="model/",
    output_path="model-int2/",
    config=config
)
```

---

### Progress Tracking

```python
def progress_callback(message, progress):
    bar_length = 40
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r[{bar}] {progress*100:.1f}% - {message}", end='', flush=True)

result = quantizer.quantize_diffusion_model(
    model_path="model/",
    output_path="model-int4/",
    progress_callback=progress_callback
)
print()  # New line after progress bar
```

---

### Async Batch Quantization

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
        print(f"Model {i+1}:")
        print(f"  Compression: {result['compression_ratio']:.2f}x")
        print(f"  Quality: {result['cosine_similarity']:.4f}")

asyncio.run(quantize_models())
```

---

### Arrow Batch Quantization

```python
import pyarrow as pa
import numpy as np
from arrow_quant_v2 import ArrowQuantV2

# Create input table
layers = []
for i in range(10):
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

# Quantize in batch (zero-copy)
quantizer = ArrowQuantV2()
result = quantizer.quantize_batch_arrow(table)

print(f"Quantized {result.num_rows} layers")
```

---

### Validation

```python
# Quantize model
result = quantizer.quantize_diffusion_model(
    model_path="model/",
    output_path="model-int4/"
)

# Validate quality
validation = quantizer.validate_quality(
    original_path="model/",
    quantized_path="model-int4/"
)

if validation['passed']:
    print(f"✓ Validation passed")
    print(f"  Overall quality: {validation['cosine_similarity']:.4f}")
    print(f"  Compression: {validation['compression_ratio']:.2f}x")
    
    # Check per-layer quality
    for layer, quality in validation['per_layer_accuracy'].items():
        if quality < 0.80:
            print(f"  ⚠ Low quality layer: {layer} ({quality:.4f})")
else:
    print(f"✗ Validation failed")
```

---

### SafeTensors Quantization

```python
# Quantize from SafeTensors file
result = quantizer.quantize_from_safetensors(
    safetensors_path="model.safetensors",
    output_path="model-int4/"
)

# Or from sharded SafeTensors
result = quantizer.quantize_from_safetensors(
    safetensors_path="model-sharded/",  # Directory with model-00001-of-00003.safetensors, etc.
    output_path="model-int4/"
)
```

---

### Thermodynamic Optimization

```python
from arrow_quant_v2 import (
    DiffusionQuantConfig,
    ValidationConfig,
    BoundarySmoothingConfig,
    TransitionOptimizationConfig,
    InterpolationMethod,
    BetaSchedule
)

# Enable all thermodynamic phases
config = DiffusionQuantConfig(
    bit_width=4,
    # Phase 1: Validation
    thermodynamic=ThermodynamicConfig(
        validation=ValidationConfig(
            enabled=True,
            smoothness_threshold=0.25,
            log_violations=True
        ),
        # Phase 2: Boundary smoothing (+2-3% accuracy)
        boundary_smoothing=BoundarySmoothingConfig(
            enabled=True,
            window_size=5,
            interpolation=InterpolationMethod.Linear
        ),
        # Phase 3: Transition optimization (+4-5% cumulative accuracy)
        transition_optimization=TransitionOptimizationConfig(
            enabled=True,
            markov_weight=0.1,
            entropy_weight=0.05,
            learning_rate=0.01,
            max_iterations=50,
            convergence_threshold=1e-4,
            beta_schedule=BetaSchedule.Linear
        )
    )
)

result = quantizer.quantize_diffusion_model(
    model_path="model/",
    output_path="model-int4-optimized/",
    config=config
)
```

---

## Error Handling

### Python Exceptions

```python
from arrow_quant_v2 import ArrowQuantV2

quantizer = ArrowQuantV2()

try:
    result = quantizer.quantize_diffusion_model(
        model_path="model/",
        output_path="model-int4/"
    )
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Quantization error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

### Rust Error Types

```rust
use arrow_quant_v2::{QuantError, Result};

match orchestrator.quantize_model(&model_path, &output_path) {
    Ok(result) => println!("Success: {:.2}x compression", result.compression_ratio),
    Err(QuantError::InvalidBitWidth(bw)) => eprintln!("Invalid bit width: {}", bw),
    Err(QuantError::ConfigurationError(msg)) => eprintln!("Config error: {}", msg),
    Err(e) => eprintln!("Error: {}", e),
}
```

---

## Performance Tips

1. **Use Arrow Batch API** for multiple layers to reduce Python-Rust boundary crossings (9,185x improvement)
2. **Enable streaming mode** for large models on memory-constrained devices
3. **Use async API** for concurrent quantization of multiple models
4. **Adjust calibration_samples** based on quality requirements (32 for edge, 512 for cloud)
5. **Enable thermodynamic optimization** only when quality is critical (adds 20-30% overhead)

---

## Version History

- **2.0.0** (2026-02-26): PyO3 zero-copy optimization, Arrow Batch API, thermodynamic constraints
- **1.1.0** (2026-02-15): Async API, SafeTensors support, mixed-precision quantization
- **1.0.0** (2026-02-01): Initial release with time-aware and spatial quantization

---

## See Also

- [Quick Start Guide](QUICK_START.md)
- [Arrow Migration Guide](ARROW_MIGRATION_GUIDE.md)
- [Performance Benchmark Report](PERFORMANCE_BENCHMARK_REPORT.md)
- [Phase 2.0 Optimization Report](PHASE_2.0_OPTIMIZATION_COMPLETION_REPORT.md)

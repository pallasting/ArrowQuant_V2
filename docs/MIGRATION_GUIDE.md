# Migration Guide: Base ArrowQuant → ArrowQuant V2 for Diffusion

This guide helps you migrate from the base ArrowQuant quantization system to ArrowQuant V2 for Diffusion, which adds diffusion model-specific optimizations while maintaining backward compatibility.

## Table of Contents

1. [Overview](#overview)
2. [What's New in V2](#whats-new-in-v2)
3. [Breaking Changes](#breaking-changes)
4. [Migration Checklist](#migration-checklist)
5. [Code Migration Examples](#code-migration-examples)
6. [Feature Adoption Guide](#feature-adoption-guide)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

---

## Overview

### Why Migrate?

ArrowQuant V2 for Diffusion provides significant improvements for diffusion models:

- **10-50% better accuracy** for diffusion models through time-aware and spatial quantization
- **Automatic modality detection** and strategy selection
- **Graceful degradation** with automatic fallback (INT2 → INT4 → INT8)
- **Deployment profiles** for edge, local, and cloud environments
- **Extended Parquet V2 schema** with diffusion-specific metadata
- **Backward compatible** with base ArrowQuant models

### Migration Effort

| Scenario | Effort | Time Estimate |
|----------|--------|---------------|
| **Simple quantization scripts** | Low | 15-30 minutes |
| **Custom configuration** | Medium | 1-2 hours |
| **Advanced workflows** | Medium-High | 2-4 hours |
| **Production deployment** | High | 1-2 days |

### Compatibility

✅ **Fully Compatible**:
- Existing Parquet V2 models can be loaded by V2
- Base quantization mode (`mode="base"`) works identically
- All existing APIs are preserved

⚠️ **Requires Updates**:
- Configuration format (YAML structure changed)
- Python import paths (new module name)
- Some parameter names (see Breaking Changes)

---

## What's New in V2

### 1. Diffusion-Specific Optimizations

**Time-Aware Quantization**:
- Groups similar timesteps together
- Adaptive quantization parameters per time group
- Handles temporal variance in diffusion models

**Spatial Quantization**:
- Channel equalization (DiTAS technique)
- Activation smoothing
- Per-group quantization for spatial variance

### 2. Automatic Modality Detection

V2 automatically detects model modality (text, code, image, audio) and selects optimal quantization strategies:

```python
# V2: Automatic detection
result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-int2/"
)
# Automatically detects "text" modality and uses R2Q + TimeAware

# V1: Manual strategy selection required
result = quantizer.quantize(
    model_path="dream-7b/",
    output_path="dream-7b-int2/",
    strategy="r2q"  # Had to specify manually
)
```

### 3. Deployment Profiles

Pre-configured profiles for different deployment scenarios:

```python
# V2: Use deployment profile
config = DiffusionQuantConfig.from_profile("edge")  # or "local", "cloud"

# V1: Manual configuration
config = {
    "bit_width": 2,
    "group_size": 256,
    "calibration_samples": 32,
    # ... many more parameters
}
```

### 4. Graceful Degradation

Automatic fallback when quantization fails:

```python
# V2: Automatic fallback
result = quantizer.quantize_diffusion_model(
    model_path="model/",
    output_path="quantized/",
    config=DiffusionQuantConfig(bit_width=2, min_accuracy=0.70)
)
# If INT2 fails, automatically tries INT4, then INT8

# V1: Manual retry logic required
try:
    result = quantizer.quantize(model_path, output_path, bit_width=2)
except QuantizationError:
    result = quantizer.quantize(model_path, output_path, bit_width=4)
```

### 5. Extended Parquet V2 Schema

New metadata fields for diffusion models:

```python
# V2: Rich metadata
{
    "is_diffusion_model": true,
    "modality": "text",
    "time_aware_quant": {
        "enabled": true,
        "num_time_groups": 10,
        "time_group_params": [...]
    },
    "spatial_quant": {
        "enabled": true,
        "channel_equalization": true,
        "equalization_scales": [...]
    }
}

# V1: Basic metadata only
{
    "quant_type": "int2",
    "scales": [...],
    "zero_points": [...]
}
```

### 6. Progress Callbacks

Monitor long-running quantization:

```python
# V2: Progress callback support
def progress(message, progress_pct):
    print(f"[{progress_pct:.0%}] {message}")

result = quantizer.quantize_diffusion_model(
    model_path="model/",
    output_path="quantized/",
    progress_callback=progress
)

# V1: No progress reporting
result = quantizer.quantize(model_path, output_path)  # Silent
```

---

## Breaking Changes

### 1. Module Name Change

**Before (V1)**:
```python
from arrow_quant import ArrowQuant
```

**After (V2)**:
```python
from arrow_quant_v2 import ArrowQuantV2
```

### 2. Constructor Parameter

**Before (V1)**:
```python
quantizer = ArrowQuant()  # No mode parameter
```

**After (V2)**:
```python
quantizer = ArrowQuantV2(mode="diffusion")  # or mode="base"
```

### 3. Configuration Structure

**Before (V1)**:
```python
config = {
    "bits": 4,  # Old name
    "group_size": 128,
    "samples": 128,  # Old name
}
```

**After (V2)**:
```python
config = DiffusionQuantConfig(
    bit_width=4,  # New name
    group_size=128,
    calibration_samples=128,  # New name
)
```

### 4. Method Names

**Before (V1)**:
```python
result = quantizer.quantize(model_path, output_path, config)
```

**After (V2)**:
```python
# For diffusion models
result = quantizer.quantize_diffusion_model(model_path, output_path, config)

# For base quantization (backward compatible)
result = quantizer.quantize(weights, bit_width=4)
```

### 5. Return Value Structure

**Before (V1)**:
```python
result = {
    "path": "quantized/model/",
    "compression": 16.0,
    "accuracy": 0.73,
    "size": 32.5
}
```

**After (V2)**:
```python
result = {
    "quantized_path": "quantized/model/",  # Renamed
    "compression_ratio": 16.0,  # Renamed
    "cosine_similarity": 0.73,  # Renamed
    "model_size_mb": 32.5,  # Renamed
    "modality": "text",  # New
    "bit_width": 2,  # New
    "quantization_time_s": 120.5  # New
}
```

### 6. Environment Variables

**Before (V1)**:
```bash
export ARROW_QUANT_BITS=4
export ARROW_QUANT_SAMPLES=128
```

**After (V2)**:
```bash
export ARROW_QUANT_BIT_WIDTH=4  # Renamed
export ARROW_QUANT_CALIBRATION_SAMPLES=128  # Renamed
```

---

## Migration Checklist

### Phase 1: Preparation (15 minutes)

- [ ] **Backup existing code and models**
  ```bash
  cp -r quantization_scripts/ quantization_scripts.backup/
  cp -r models/ models.backup/
  ```

- [ ] **Install ArrowQuant V2**
  ```bash
  pip install arrow-quant-v2
  # or
  pip install -e ai_os_diffusion/arrow_quant_v2/
  ```

- [ ] **Review breaking changes** (see section above)

- [ ] **Identify affected code**
  ```bash
  grep -r "from arrow_quant import" .
  grep -r "ArrowQuant(" .
  ```

### Phase 2: Code Updates (30-60 minutes)

- [ ] **Update imports**
  ```python
  # Old
  from arrow_quant import ArrowQuant
  
  # New
  from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
  ```

- [ ] **Update constructor calls**
  ```python
  # Old
  quantizer = ArrowQuant()
  
  # New
  quantizer = ArrowQuantV2(mode="diffusion")
  ```

- [ ] **Update method calls**
  ```python
  # Old
  result = quantizer.quantize(model_path, output_path, config)
  
  # New
  result = quantizer.quantize_diffusion_model(model_path, output_path, config)
  ```

- [ ] **Update configuration**
  ```python
  # Old
  config = {"bits": 4, "samples": 128}
  
  # New
  config = DiffusionQuantConfig(bit_width=4, calibration_samples=128)
  # or use profile
  config = DiffusionQuantConfig.from_profile("local")
  ```

- [ ] **Update result handling**
  ```python
  # Old
  print(f"Size: {result['size']}MB")
  
  # New
  print(f"Size: {result['model_size_mb']}MB")
  ```

### Phase 3: Testing (30-60 minutes)

- [ ] **Test basic quantization**
  ```python
  quantizer = ArrowQuantV2(mode="diffusion")
  result = quantizer.quantize_diffusion_model(
      model_path="test_model/",
      output_path="test_output/"
  )
  assert result["cosine_similarity"] >= 0.70
  ```

- [ ] **Test with custom configuration**
  ```python
  config = DiffusionQuantConfig(
      bit_width=4,
      num_time_groups=10,
      min_accuracy=0.85
  )
  result = quantizer.quantize_diffusion_model(
      model_path="test_model/",
      output_path="test_output/",
      config=config
  )
  ```

- [ ] **Test backward compatibility** (if using base mode)
  ```python
  quantizer = ArrowQuantV2(mode="base")
  result = quantizer.quantize(weights, bit_width=4)
  ```

- [ ] **Validate output models**
  ```python
  validation = quantizer.validate_quality(
      original_path="test_model/",
      quantized_path="test_output/"
  )
  assert validation["passed"]
  ```

### Phase 4: Deployment (varies)

- [ ] **Update CI/CD pipelines**
  - Update Docker images
  - Update environment variables
  - Update deployment scripts

- [ ] **Update documentation**
  - Update README files
  - Update API documentation
  - Update deployment guides

- [ ] **Monitor production**
  - Check quantization quality metrics
  - Monitor memory usage
  - Monitor quantization time

---

## Code Migration Examples

### Example 1: Simple Quantization Script

**Before (V1)**:
```python
from arrow_quant import ArrowQuant

quantizer = ArrowQuant()

result = quantizer.quantize(
    model_path="dream-7b/",
    output_path="dream-7b-int2/",
    config={"bits": 2, "samples": 128}
)

print(f"Compressed to {result['size']}MB")
print(f"Accuracy: {result['accuracy']:.3f}")
```

**After (V2)**:
```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

quantizer = ArrowQuantV2(mode="diffusion")

result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-int2/",
    config=DiffusionQuantConfig(bit_width=2, calibration_samples=128)
)

print(f"Compressed to {result['model_size_mb']}MB")
print(f"Accuracy: {result['cosine_similarity']:.3f}")
```

### Example 2: Batch Quantization

**Before (V1)**:
```python
from arrow_quant import ArrowQuant

quantizer = ArrowQuant()

models = ["model1", "model2", "model3"]
configs = [
    {"bits": 2, "samples": 32},
    {"bits": 4, "samples": 128},
    {"bits": 8, "samples": 512}
]

for model, config in zip(models, configs):
    result = quantizer.quantize(
        model_path=f"models/{model}/",
        output_path=f"quantized/{model}/",
        config=config
    )
    print(f"{model}: {result['compression']}x compression")
```

**After (V2)**:
```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

quantizer = ArrowQuantV2(mode="diffusion")

models = ["model1", "model2", "model3"]
profiles = ["edge", "local", "cloud"]

for model, profile in zip(models, profiles):
    config = DiffusionQuantConfig.from_profile(profile)
    
    result = quantizer.quantize_diffusion_model(
        model_path=f"models/{model}/",
        output_path=f"quantized/{model}/",
        config=config
    )
    print(f"{model}: {result['compression_ratio']}x compression")
```

### Example 3: Custom Configuration

**Before (V1)**:
```python
from arrow_quant import ArrowQuant

config = {
    "bits": 4,
    "group_size": 128,
    "samples": 256,
    "min_acc": 0.85,
    "threads": 8
}

quantizer = ArrowQuant()
result = quantizer.quantize(
    model_path="model/",
    output_path="quantized/",
    config=config
)
```

**After (V2)**:
```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

config = DiffusionQuantConfig(
    bit_width=4,
    group_size=128,
    calibration_samples=256,
    min_accuracy=0.85,
    num_threads=8,
    enable_time_aware=True,  # New feature
    enable_spatial=True  # New feature
)

quantizer = ArrowQuantV2(mode="diffusion")
result = quantizer.quantize_diffusion_model(
    model_path="model/",
    output_path="quantized/",
    config=config
)
```

### Example 4: Error Handling

**Before (V1)**:
```python
from arrow_quant import ArrowQuant, QuantizationError

quantizer = ArrowQuant()

try:
    result = quantizer.quantize(
        model_path="model/",
        output_path="quantized/",
        config={"bits": 2}
    )
except QuantizationError as e:
    print(f"Quantization failed: {e}")
    # Manual fallback
    result = quantizer.quantize(
        model_path="model/",
        output_path="quantized/",
        config={"bits": 4}
    )
```

**After (V2)**:
```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
from arrow_quant_v2 import QuantizationError, ConfigurationError

quantizer = ArrowQuantV2(mode="diffusion")

try:
    # Automatic fallback enabled by default
    result = quantizer.quantize_diffusion_model(
        model_path="model/",
        output_path="quantized/",
        config=DiffusionQuantConfig(bit_width=2, min_accuracy=0.70)
    )
    
    if result["bit_width"] > 2:
        print(f"Fallback occurred: using INT{result['bit_width']}")
    
except ConfigurationError as e:
    print(f"Invalid configuration: {e}")
except QuantizationError as e:
    print(f"Quantization failed: {e}")
```

### Example 5: YAML Configuration

**Before (V1)**:
```yaml
# config_v1.yaml
bits: 4
group_size: 128
samples: 128
min_acc: 0.85
```

```python
import yaml
from arrow_quant import ArrowQuant

with open("config_v1.yaml") as f:
    config = yaml.safe_load(f)

quantizer = ArrowQuant()
result = quantizer.quantize("model/", "quantized/", config)
```

**After (V2)**:
```yaml
# config_v2.yaml
bit_width: 4
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
```

```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

# Load config from YAML
config = DiffusionQuantConfig.from_yaml("config_v2.yaml")

quantizer = ArrowQuantV2(mode="diffusion")
result = quantizer.quantize_diffusion_model("model/", "quantized/", config)
```

### Example 6: Progress Monitoring

**Before (V1)**:
```python
from arrow_quant import ArrowQuant

quantizer = ArrowQuant()

# No progress reporting
result = quantizer.quantize("model/", "quantized/", {"bits": 4})
```

**After (V2)**:
```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

def progress_callback(message, progress):
    print(f"[{progress:.0%}] {message}")

quantizer = ArrowQuantV2(mode="diffusion")

result = quantizer.quantize_diffusion_model(
    model_path="model/",
    output_path="quantized/",
    config=DiffusionQuantConfig(bit_width=4),
    progress_callback=progress_callback
)
```

---

## Feature Adoption Guide

### 1. Adopting Time-Aware Quantization

**When to use**: Text and code diffusion models (discrete diffusion)

**Benefits**:
- 10-20% better accuracy for temporal models
- Handles timestep variance automatically
- Minimal configuration required

**Migration**:
```python
# V1: No time-aware support
result = quantizer.quantize(model_path, output_path, {"bits": 2})

# V2: Enable time-aware (enabled by default for text/code)
config = DiffusionQuantConfig(
    bit_width=2,
    enable_time_aware=True,  # Default: True
    num_time_groups=10  # Adjust based on model
)
result = quantizer.quantize_diffusion_model(model_path, output_path, config)
```

**Tuning**:
- `num_time_groups=5`: Edge devices (faster, less memory)
- `num_time_groups=10`: Local machines (balanced)
- `num_time_groups=20`: Cloud servers (best accuracy)

### 2. Adopting Spatial Quantization

**When to use**: Image and audio diffusion models (continuous diffusion)

**Benefits**:
- 10-15% better accuracy for spatial models
- Handles channel variance automatically
- Reduces quantization artifacts

**Migration**:
```python
# V1: No spatial support
result = quantizer.quantize(model_path, output_path, {"bits": 4})

# V2: Enable spatial (enabled by default for image/audio)
config = DiffusionQuantConfig(
    bit_width=4,
    enable_spatial=True,  # Default: True
    group_size=128  # Adjust based on model
)
result = quantizer.quantize_diffusion_model(model_path, output_path, config)
```

**Tuning**:
- `group_size=256`: Edge devices (coarser, faster)
- `group_size=128`: Local machines (balanced)
- `group_size=64`: Cloud servers (finer, better quality)

### 3. Adopting Deployment Profiles

**When to use**: Standardized deployment scenarios

**Benefits**:
- Pre-configured optimal settings
- Consistent across deployments
- Easy to switch between profiles

**Migration**:
```python
# V1: Manual configuration for each deployment
edge_config = {"bits": 2, "group_size": 256, "samples": 32}
local_config = {"bits": 4, "group_size": 128, "samples": 128}
cloud_config = {"bits": 8, "group_size": 64, "samples": 512}

# V2: Use deployment profiles
edge_config = DiffusionQuantConfig.from_profile("edge")
local_config = DiffusionQuantConfig.from_profile("local")
cloud_config = DiffusionQuantConfig.from_profile("cloud")

# Optional: Customize after loading profile
edge_config.min_accuracy = 0.70  # Adjust threshold
```

### 4. Adopting Automatic Fallback

**When to use**: Production deployments requiring reliability

**Benefits**:
- Graceful degradation (INT2 → INT4 → INT8)
- No manual retry logic needed
- Guaranteed quantization success

**Migration**:
```python
# V1: Manual fallback logic
try:
    result = quantizer.quantize(model_path, output_path, {"bits": 2})
except QuantizationError:
    try:
        result = quantizer.quantize(model_path, output_path, {"bits": 4})
    except QuantizationError:
        result = quantizer.quantize(model_path, output_path, {"bits": 8})

# V2: Automatic fallback (enabled by default)
config = DiffusionQuantConfig(
    bit_width=2,
    min_accuracy=0.70,
    fail_fast=False  # Enable fallback (default)
)
result = quantizer.quantize_diffusion_model(model_path, output_path, config)

# Check if fallback occurred
if result["bit_width"] > 2:
    print(f"Fallback to INT{result['bit_width']}")
```

### 5. Adopting Modality Detection

**When to use**: Multi-modal systems or unknown model types

**Benefits**:
- Automatic strategy selection
- No manual modality specification
- Optimal quantization per modality

**Migration**:
```python
# V1: Manual strategy selection
if model_type == "text":
    result = quantizer.quantize(model_path, output_path, {"strategy": "r2q"})
elif model_type == "image":
    result = quantizer.quantize(model_path, output_path, {"strategy": "gptq"})

# V2: Automatic modality detection
result = quantizer.quantize_diffusion_model(
    model_path=model_path,
    output_path=output_path
)
# Automatically detects modality and selects optimal strategy

# Optional: Override detection
config = DiffusionQuantConfig(modality="text")  # Force text modality
result = quantizer.quantize_diffusion_model(model_path, output_path, config)
```

### 6. Adopting Extended Metadata

**When to use**: Model versioning, debugging, or analysis

**Benefits**:
- Self-contained quantized models
- Reproducible quantization
- Better debugging information

**Migration**:
```python
# V1: Limited metadata
# (metadata stored separately or not at all)

# V2: Rich metadata automatically stored
result = quantizer.quantize_diffusion_model(model_path, output_path)

# Metadata is automatically saved in Parquet V2 Extended format
# Includes: modality, time_aware_quant, spatial_quant, activation_stats

# Read metadata later
import pyarrow.parquet as pq
table = pq.read_table("quantized/model/weights.parquet")
metadata = table.schema.metadata
print(f"Modality: {metadata[b'modality']}")
print(f"Time-aware: {metadata[b'time_aware_quant']}")
```

---

## Troubleshooting

### Issue 1: Import Error

**Error**:
```
ImportError: No module named 'arrow_quant_v2'
```

**Solution**:
```bash
# Install ArrowQuant V2
pip install arrow-quant-v2

# Or install from source
cd ai_os_diffusion/arrow_quant_v2
pip install -e .
```

### Issue 2: Configuration Error

**Error**:
```
ConfigurationError: Invalid bit_width: 3. Must be 2, 4, or 8.
```

**Solution**:
```python
# V1 used "bits", V2 uses "bit_width"
# Old
config = {"bits": 4}

# New
config = DiffusionQuantConfig(bit_width=4)
```

### Issue 3: Method Not Found

**Error**:
```
AttributeError: 'ArrowQuantV2' object has no attribute 'quantize'
```

**Solution**:
```python
# For diffusion models, use quantize_diffusion_model()
result = quantizer.quantize_diffusion_model(model_path, output_path)

# For base quantization (LoRA/ControlNet), use quantize()
result = quantizer.quantize(weights, bit_width=4)
```

### Issue 4: Result Key Error

**Error**:
```
KeyError: 'size'
```

**Solution**:
```python
# V1 result keys
print(result['size'])  # Old
print(result['accuracy'])  # Old
print(result['compression'])  # Old

# V2 result keys
print(result['model_size_mb'])  # New
print(result['cosine_similarity'])  # New
print(result['compression_ratio'])  # New
```

### Issue 5: Environment Variable Not Working

**Error**:
```
# Environment variable not applied
export ARROW_QUANT_BITS=4  # Old name
```

**Solution**:
```bash
# V2 uses different names
export ARROW_QUANT_BIT_WIDTH=4  # New name
export ARROW_QUANT_CALIBRATION_SAMPLES=128  # New name
```

### Issue 6: Backward Compatibility

**Error**:
```
# Need to load V1 quantized models in V2
```

**Solution**:
```python
# V2 can load V1 Parquet V2 models
from arrow_quant_v2 import ArrowQuantV2

quantizer = ArrowQuantV2(mode="base")

# Load V1 model (automatically detects schema version)
# No migration needed - V2 is backward compatible
```

---

## FAQ

### Q1: Do I need to re-quantize existing models?

**A**: No, ArrowQuant V2 can load models quantized with V1. However, re-quantizing with V2 will provide better accuracy for diffusion models due to time-aware and spatial optimizations.

### Q2: Can I use V2 for non-diffusion models?

**A**: Yes, use `mode="base"` for backward compatibility:
```python
quantizer = ArrowQuantV2(mode="base")
result = quantizer.quantize(weights, bit_width=4)
```

### Q3: What's the performance difference between V1 and V2?

**A**: V2 has similar performance to V1 for base quantization, but provides 10-50% better accuracy for diffusion models with minimal overhead (<10% slower due to time-aware/spatial processing).

### Q4: Can I disable new features?

**A**: Yes, disable time-aware and spatial quantization:
```python
config = DiffusionQuantConfig(
    bit_width=4,
    enable_time_aware=False,
    enable_spatial=False
)
```

### Q5: How do I migrate YAML configs?

**A**: Update field names:
```yaml
# V1
bits: 4
samples: 128
min_acc: 0.85

# V2
bit_width: 4
calibration_samples: 128
min_accuracy: 0.85
```

### Q6: What if my model doesn't have metadata?

**A**: Specify modality explicitly:
```python
config = DiffusionQuantConfig(modality="text")
result = quantizer.quantize_diffusion_model(model_path, output_path, config)
```

### Q7: Can I use both V1 and V2 in the same project?

**A**: Yes, but not recommended. They have different module names:
```python
from arrow_quant import ArrowQuant  # V1
from arrow_quant_v2 import ArrowQuantV2  # V2
```

### Q8: How do I test my migration?

**A**: Use the validation system:
```python
# Quantize with V2
result = quantizer.quantize_diffusion_model(model_path, output_path)

# Validate quality
validation = quantizer.validate_quality(model_path, output_path)
assert validation["passed"]
assert validation["cosine_similarity"] >= 0.70
```

### Q9: What's the recommended migration path?

**A**:
1. Start with deployment profiles (`from_profile()`)
2. Test on non-critical models first
3. Gradually adopt new features (time-aware, spatial)
4. Update production deployments last

### Q10: Where can I get help?

**A**: See additional resources:
- [Quickstart Guide](QUICKSTART.md)
- [API Reference](API_REFERENCE.md)
- [Configuration Guide](CONFIGURATION_GUIDE.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)

---

## Summary

### Key Takeaways

✅ **Backward Compatible**: V2 can load V1 models and supports base mode

✅ **Easy Migration**: Most changes are import paths and parameter names

✅ **Better Accuracy**: 10-50% improvement for diffusion models

✅ **New Features**: Time-aware, spatial, automatic fallback, deployment profiles

✅ **Production Ready**: Graceful degradation, progress monitoring, rich metadata

### Next Steps

1. **Review** this migration guide
2. **Update** your code using the examples
3. **Test** with non-critical models
4. **Validate** quantization quality
5. **Deploy** to production

### Migration Timeline

| Week | Activity |
|------|----------|
| Week 1 | Review guide, update development code |
| Week 2 | Test on staging environment |
| Week 3 | Validate quality metrics |
| Week 4 | Deploy to production |

---

**Need Help?** Check the [Troubleshooting Guide](TROUBLESHOOTING.md) or open an issue on GitHub.

**Ready to Start?** See the [Quickstart Guide](QUICKSTART.md) for hands-on examples.

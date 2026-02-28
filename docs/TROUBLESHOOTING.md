# ArrowQuant V2 Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using ArrowQuant V2 for Diffusion.

## Table of Contents

1. [Common Errors](#common-errors)
2. [Performance Issues](#performance-issues)
3. [Quality Problems](#quality-problems)
4. [Memory Issues](#memory-issues)
5. [Configuration Errors](#configuration-errors)
6. [Debugging Techniques](#debugging-techniques)
7. [FAQ](#faq)

## Common Errors

### ConfigurationError: Invalid bit width

**Error Message:**
```
ConfigurationError: Invalid bit width: 3. Must be 2, 4, or 8.
Hint: Use DiffusionQuantConfig(bit_width=2/4/8) or select a deployment profile.
```

**Cause:** The `bit_width` parameter must be exactly 2, 4, or 8. Other values are not supported.

**Solution:**
```python
# ❌ Wrong
config = DiffusionQuantConfig(bit_width=3)

# ✅ Correct
config = DiffusionQuantConfig(bit_width=4)
```

**Related:** See [Configuration Guide](CONFIGURATION_GUIDE.md#bit_width) for bit width recommendations.

---

### ConfigurationError: Invalid group size

**Error Message:**
```
ConfigurationError: Invalid group size: 100. Must be 32, 64, 128, or 256.
Hint: Smaller group sizes provide finer quantization but increase overhead.
```

**Cause:** The `group_size` parameter must be one of the supported values: 32, 64, 128, or 256.

**Solution:**
```python
# ❌ Wrong
config = DiffusionQuantConfig(group_size=100)

# ✅ Correct - Choose based on your needs
config = DiffusionQuantConfig(group_size=128)  # Balanced (recommended)
config = DiffusionQuantConfig(group_size=64)   # Finer quantization
config = DiffusionQuantConfig(group_size=256)  # Coarser quantization
```

**Trade-offs:**
- Smaller group sizes (32, 64): Better accuracy, more metadata overhead
- Larger group sizes (128, 256): Less overhead, slightly lower accuracy

---

### ConfigurationError: Invalid number of time groups

**Error Message:**
```
ConfigurationError: Invalid number of time groups: 150. Must be between 1 and 100.
Hint: More time groups provide better temporal adaptation but increase complexity.
```

**Cause:** The `num_time_groups` parameter must be between 1 and 100.

**Solution:**
```python
# ❌ Wrong
config = DiffusionQuantConfig(num_time_groups=150)

# ✅ Correct
config = DiffusionQuantConfig(num_time_groups=10)  # Balanced
config = DiffusionQuantConfig(num_time_groups=5)   # Edge devices
config = DiffusionQuantConfig(num_time_groups=20)  # Cloud servers
```

---

### ConfigurationError: Invalid accuracy threshold

**Error Message:**
```
ConfigurationError: Invalid accuracy threshold: 1.5. Must be between 0.0 and 1.0.
Hint: Typical thresholds are 0.70 (INT2), 0.85 (INT4), 0.95 (INT8).
```

**Cause:** The `min_accuracy` parameter must be a float between 0.0 and 1.0 (cosine similarity).

**Solution:**
```python
# ❌ Wrong
config = DiffusionQuantConfig(min_accuracy=1.5)

# ✅ Correct - Match threshold to bit width
config = DiffusionQuantConfig(bit_width=2, min_accuracy=0.70)  # INT2
config = DiffusionQuantConfig(bit_width=4, min_accuracy=0.85)  # INT4
config = DiffusionQuantConfig(bit_width=8, min_accuracy=0.95)  # INT8
```

**Recommended thresholds:**
- INT2: 0.65-0.70 (aggressive compression)
- INT4: 0.80-0.90 (balanced)
- INT8: 0.95+ (high accuracy)

---

### ValidationError: Quantization quality validation failed

**Error Message:**
```
ValidationError: Quantization quality validation failed: cosine similarity 0.6234 is below threshold 0.8500.
Suggestions: (1) Try higher bit width (INT4/INT8), (2) Enable spatial quantization,
(3) Increase calibration samples, (4) Use fallback mode for automatic degradation.
```

**Cause:** The quantized model's quality (measured by cosine similarity) is below the configured threshold.

**Solutions:**

**Option 1: Enable automatic fallback (recommended)**
```python
config = DiffusionQuantConfig(
    bit_width=2,
    min_accuracy=0.85,
    fail_fast=False  # Enable automatic fallback to INT4/INT8
)
```

**Option 2: Lower the accuracy threshold**
```python
config = DiffusionQuantConfig(
    bit_width=2,
    min_accuracy=0.65  # More lenient threshold for INT2
)
```

**Option 3: Use higher bit width**
```python
config = DiffusionQuantConfig(
    bit_width=4,  # Use INT4 instead of INT2
    min_accuracy=0.85
)
```

**Option 4: Improve quantization quality**
```python
config = DiffusionQuantConfig(
    bit_width=2,
    min_accuracy=0.70,
    enable_spatial=True,        # Enable spatial quantization
    calibration_samples=256,    # More calibration samples
    num_time_groups=15          # More time groups
)
```

**Diagnosis:** Check which layers are failing:
```python
result = quantizer.validate_quality(
    original_path="model/",
    quantized_path="model-quantized/"
)

# Check per-layer accuracy
for layer, accuracy in result['per_layer_accuracy'].items():
    if accuracy < 0.85:
        print(f"Layer {layer} failed: {accuracy:.3f}")
```

---

### ModelNotFoundError: Model not found

**Error Message:**
```
ModelNotFoundError: Model not found at path: 'dream-7b/'.
Hint: Ensure the model directory exists and contains valid Parquet files.
```

**Cause:** The specified model path doesn't exist or doesn't contain Parquet files.

**Solutions:**

**Check path exists:**
```bash
ls -la dream-7b/
# Should show .parquet files and metadata.json
```

**Verify Parquet files:**
```bash
find dream-7b/ -name "*.parquet" | head -5
# Should list layer files like layer_0.parquet, layer_1.parquet, etc.
```

**Check file permissions:**
```bash
ls -l dream-7b/
# Ensure files are readable (r-- permission)
```

**Use absolute path:**
```python
import os
model_path = os.path.abspath("dream-7b/")
quantizer.quantize_diffusion_model(model_path=model_path, ...)
```

---

### MetadataError: Failed to read model metadata

**Error Message:**
```
MetadataError: Unknown modality in model metadata.
Hint: metadata.json must contain 'modality' field with value 'text', 'code', 'image', or 'audio'.
You can also specify modality explicitly in DiffusionQuantConfig(modality='text').
```

**Cause:** The `metadata.json` file is missing, invalid, or doesn't contain a valid `modality` field.

**Solutions:**

**Option 1: Specify modality explicitly**
```python
config = DiffusionQuantConfig(modality="text")  # Bypass auto-detection
quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-quantized/",
    config=config
)
```

**Option 2: Create/fix metadata.json**
```bash
# Create metadata.json in model directory
cat > dream-7b/metadata.json << EOF
{
  "modality": "text",
  "model_type": "diffusion",
  "architecture": "dream-7b"
}
EOF
```

**Valid modality values:**
- `"text"` - Text generation models (discrete diffusion)
- `"code"` - Code generation models (discrete diffusion)
- `"image"` - Image generation models (continuous diffusion)
- `"audio"` - Audio generation models (continuous diffusion)

**Check existing metadata:**
```bash
cat dream-7b/metadata.json | python -m json.tool
# Should show valid JSON with "modality" field
```

---

### ShapeMismatchError: Shape mismatch

**Error Message:**
```
ShapeMismatchError: Shape mismatch: expected [1024, 4096], got [1024, 2048].
Hint: This usually indicates incompatible model architecture or corrupted weights.
```

**Cause:** The weight tensor shape doesn't match the expected dimensions.

**Solutions:**

**Verify model integrity:**
```python
import pyarrow.parquet as pq

# Check layer shapes
table = pq.read_table("dream-7b/layer_0.parquet")
print(f"Schema: {table.schema}")
print(f"Num rows: {table.num_rows}")
```

**Re-download model:**
```bash
# If model is corrupted, re-download
rm -rf dream-7b/
# Download again from source
```

**Check model compatibility:**
- Ensure model uses Parquet V2 or V2Extended schema
- Verify model was exported correctly
- Check for partial/incomplete downloads

---

### IOError: IO error during quantization

**Error Message:**
```
QuantizationError: IO error during quantization: Permission denied (os error 13).
Hint: Check file permissions, disk space, and path validity.
```

**Cause:** File system permission issues, insufficient disk space, or invalid paths.

**Solutions:**

**Check disk space:**
```bash
df -h .
# Ensure sufficient space (model size × 2 recommended)
```

**Check permissions:**
```bash
ls -ld dream-7b/ dream-7b-quantized/
# Ensure write permissions on output directory
```

**Create output directory:**
```bash
mkdir -p dream-7b-quantized/
chmod 755 dream-7b-quantized/
```

**Check for read-only filesystem:**
```bash
mount | grep $(df . | tail -1 | awk '{print $1}')
# Should not show "ro" (read-only)
```

---

## Performance Issues

### Slow Quantization Speed

**Symptoms:** Quantization takes much longer than expected (>30 minutes for Dream 7B).

**Diagnosis:**
```python
import time

start = time.time()
result = quantizer.quantize_diffusion_model(...)
elapsed = time.time() - start
print(f"Quantization took {elapsed:.1f} seconds")
```

**Common Causes & Solutions:**

**1. Streaming mode enabled (slower but uses less memory)**
```python
# Check if streaming is enabled
config = DiffusionQuantConfig.from_yaml("config.yaml")
print(f"Streaming: {config.enable_streaming}")

# Disable for faster processing (if you have enough RAM)
config.enable_streaming = False
```

**2. Too many calibration samples**
```python
# Reduce calibration samples for faster processing
config = DiffusionQuantConfig(
    calibration_samples=32,  # Down from 128 or 512
    bit_width=4
)
```

**3. Too many time groups**
```python
# Reduce time groups
config = DiffusionQuantConfig(
    num_time_groups=5,  # Down from 10 or 20
    bit_width=4
)
```

**4. Spatial quantization overhead**
```python
# Disable spatial quantization if not needed
config = DiffusionQuantConfig(
    enable_spatial=False,  # Faster, but lower accuracy for image/audio
    bit_width=4
)
```

**5. Limited CPU cores**
```python
# Use all available cores
config = DiffusionQuantConfig(
    num_threads=0,  # Auto-detect (default)
    enable_streaming=False
)
```

**Benchmark:** Expected times for Dream 7B on different hardware:
- Edge device (4 cores, 4GB RAM): 15-30 minutes (streaming mode)
- Local machine (8 cores, 16GB RAM): 5-10 minutes (parallel mode)
- Cloud server (32 cores, 64GB RAM): 2-5 minutes (parallel mode)

---

### High Memory Usage

**Symptoms:** Process uses excessive RAM or gets killed by OOM (Out of Memory).

**Diagnosis:**
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
```

**Solutions:**

**1. Enable streaming mode (most effective)**
```python
config = DiffusionQuantConfig(
    enable_streaming=True,  # Process one layer at a time
    bit_width=4
)
```

**2. Reduce calibration samples**
```python
config = DiffusionQuantConfig(
    calibration_samples=32,  # Down from 128
    enable_streaming=True
)
```

**3. Limit parallelism**
```python
config = DiffusionQuantConfig(
    num_threads=2,  # Limit concurrent layer processing
    enable_streaming=True
)
```

**4. Use edge profile**
```python
from arrow_quant_v2 import DeploymentProfile

config = DiffusionQuantConfig.from_profile(DeploymentProfile.Edge)
# Automatically sets: streaming=True, calibration_samples=32, etc.
```

**Memory requirements by mode:**
- Streaming mode: ~1-2GB peak (processes one layer at a time)
- Parallel mode: ~4-8GB peak (loads multiple layers)
- Cloud mode: ~8-16GB peak (maximum calibration samples)

---

## Quality Problems

### Low Cosine Similarity

**Symptoms:** Quantized model has low cosine similarity (<0.70 for INT2, <0.85 for INT4).

**Diagnosis:**
```python
result = quantizer.validate_quality(
    original_path="model/",
    quantized_path="model-quantized/"
)

print(f"Overall similarity: {result['cosine_similarity']:.3f}")
print(f"Passed: {result['passed']}")

# Check per-layer accuracy
for layer, accuracy in sorted(result['per_layer_accuracy'].items()):
    status = "✓" if accuracy >= 0.85 else "✗"
    print(f"{status} {layer}: {accuracy:.3f}")
```

**Solutions:**

**1. Use higher bit width**
```python
# INT2 → INT4 or INT8
config = DiffusionQuantConfig(
    bit_width=4,  # or 8 for maximum accuracy
    min_accuracy=0.85
)
```

**2. Enable all optimizations**
```python
config = DiffusionQuantConfig(
    bit_width=4,
    enable_time_aware=True,   # For text/code models
    enable_spatial=True,      # For image/audio models
    num_time_groups=15,       # More groups = better accuracy
    group_size=64,            # Finer quantization
    calibration_samples=256   # More samples = better calibration
)
```

**3. Use automatic fallback**
```python
config = DiffusionQuantConfig(
    bit_width=2,
    min_accuracy=0.85,
    fail_fast=False  # Automatically try INT4/INT8 if INT2 fails
)
```

**4. Adjust threshold for bit width**
```python
# Use realistic thresholds
config = DiffusionQuantConfig(
    bit_width=2,
    min_accuracy=0.70  # INT2 typically achieves 0.65-0.75
)
```

---

### Specific Layers Failing Validation

**Symptoms:** Most layers pass validation, but a few specific layers have low accuracy.

**Diagnosis:**
```python
result = quantizer.validate_quality(...)

# Find failing layers
failing_layers = {
    layer: acc 
    for layer, acc in result['per_layer_accuracy'].items() 
    if acc < 0.85
}

print(f"Failing layers: {len(failing_layers)}")
for layer, acc in sorted(failing_layers.items(), key=lambda x: x[1]):
    print(f"  {layer}: {acc:.3f}")
```

**Common patterns:**

**1. Attention layers failing**
- Attention layers are sensitive to quantization
- Solution: Use INT4 or INT8 for attention layers, INT2 for others (mixed precision)

**2. First/last layers failing**
- Input/output layers may need higher precision
- Solution: Skip quantization for first/last layers or use INT8

**3. Large variance layers**
- Layers with high activation variance are harder to quantize
- Solution: Increase `num_time_groups` or use spatial quantization

**Workaround (manual layer selection):**
```python
# This feature is planned but not yet implemented
# For now, use higher bit width globally or accept lower accuracy
```

---

## Memory Issues

### Out of Memory (OOM) Errors

**Error Message:**
```
Killed
# or
MemoryError: Unable to allocate array
```

**Immediate solution:**
```python
# Enable streaming mode
config = DiffusionQuantConfig.from_profile("edge")
config.enable_streaming = True
config.num_threads = 1

quantizer = ArrowQuantV2(mode="diffusion")
result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-quantized/",
    config=config
)
```

**Long-term solutions:**

**1. Monitor memory usage**
```python
import psutil

def monitor_memory():
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024**3
    print(f"Memory: {mem_gb:.2f} GB")
    return mem_gb

# Check before quantization
monitor_memory()
```

**2. Increase swap space (Linux)**
```bash
# Check current swap
free -h

# Add swap file (8GB)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**3. Use smaller model for testing**
```bash
# Test with a smaller model first
python scripts/quantize_diffusion.py \
    --model small-model/ \
    --output small-model-quantized/ \
    --config config.yaml
```

---

### Memory Leaks

**Symptoms:** Memory usage grows continuously during quantization.

**Diagnosis:**
```python
import tracemalloc

tracemalloc.start()

# Run quantization
result = quantizer.quantize_diffusion_model(...)

# Check memory
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024**3:.2f} GB")
print(f"Peak: {peak / 1024**3:.2f} GB")
tracemalloc.stop()
```

**Solutions:**

**1. Use streaming mode**
- Streaming mode automatically releases memory after each layer
- No manual intervention needed

**2. Process in batches**
```python
# For very large models, process in multiple runs
# (This feature is planned but not yet implemented)
```

---

## Configuration Errors

### YAML Parsing Errors

**Error Message:**
```
SerdeError: JSON serialization/deserialization error: expected value at line 5 column 1.
Hint: Check metadata.json format and ensure valid JSON syntax.
```

**Cause:** Invalid YAML syntax in config file.

**Solutions:**

**1. Validate YAML syntax**
```bash
# Use Python to check YAML
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

**2. Common YAML mistakes**
```yaml
# ❌ Wrong - inconsistent indentation
bit_width: 4
  num_time_groups: 10

# ✅ Correct
bit_width: 4
num_time_groups: 10

# ❌ Wrong - missing quotes for strings
modality: text with spaces

# ✅ Correct
modality: "text with spaces"
# or
modality: text

# ❌ Wrong - wrong boolean format
enable_streaming: yes

# ✅ Correct
enable_streaming: true
```

**3. Use example config as template**
```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your values
```

---

### Environment Variable Overrides Not Working

**Symptoms:** Environment variables don't override config file values.

**Diagnosis:**
```bash
# Check environment variables
env | grep ARROW_QUANT
```

**Solutions:**

**1. Use correct variable names**
```bash
# ✅ Correct format
export ARROW_QUANT_BIT_WIDTH=4
export ARROW_QUANT_NUM_TIME_GROUPS=10
export ARROW_QUANT_ENABLE_STREAMING=true

# ❌ Wrong format
export BIT_WIDTH=4  # Missing ARROW_QUANT_ prefix
export ARROW_QUANT_BITWIDTH=4  # Wrong name (should be BIT_WIDTH)
```

**2. Apply overrides in code**
```python
config = DiffusionQuantConfig.from_yaml("config.yaml")
config.apply_env_overrides()  # Must call this!
config.validate()
```

**3. Verify overrides**
```python
import os

os.environ['ARROW_QUANT_BIT_WIDTH'] = '2'

config = DiffusionQuantConfig.from_yaml("config.yaml")
print(f"Before override: {config.bit_width}")

config.apply_env_overrides()
print(f"After override: {config.bit_width}")  # Should be 2
```

---

## Debugging Techniques

### Enable Verbose Logging

**Python logging:**
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or configure specific logger
logging.getLogger('arrow_quant_v2').setLevel(logging.DEBUG)

# Run quantization with verbose output
result = quantizer.quantize_diffusion_model(...)
```

**Rust logging (for native code):**
```bash
# Set RUST_LOG environment variable
export RUST_LOG=debug

# Or for specific modules
export RUST_LOG=arrow_quant_v2::orchestrator=debug,arrow_quant_v2::validation=debug

python scripts/quantize_diffusion.py ...
```

---

### Progress Monitoring

**Use progress callback:**
```python
def progress_callback(message: str, progress: float):
    print(f"[{progress*100:.1f}%] {message}")

result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-quantized/",
    config=config,
    progress_callback=progress_callback
)
```

**Expected progress messages:**
```
[0.0%] Starting quantization...
[10.0%] Detecting model modality...
[15.0%] Detected text modality
[20.0%] Quantizing model layers...
[90.0%] Validating quantization quality...
[100.0%] Quantization complete
```

---

### Inspect Intermediate Results

**Check quantized layer files:**
```python
import pyarrow.parquet as pq

# Read quantized layer
table = pq.read_table("dream-7b-quantized/layer_0.parquet")
print(f"Schema: {table.schema}")
print(f"Columns: {table.column_names}")
print(f"Rows: {table.num_rows}")

# Check metadata
metadata = table.schema.metadata
if metadata:
    print(f"Metadata: {metadata}")
```

**Validate single layer:**
```python
from arrow_quant_v2 import ValidationSystem

validator = ValidationSystem.new(0.85)

# Load original and quantized weights
# (This requires manual weight extraction)
original_weights = [...]  # Load from original layer
quantized_weights = [...]  # Load from quantized layer

similarity = validator.cosine_similarity(original_weights, quantized_weights)
print(f"Layer similarity: {similarity:.3f}")
```

---

### Reproduce Issues

**Create minimal reproduction:**
```python
# minimal_repro.py
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

config = DiffusionQuantConfig(
    bit_width=2,
    min_accuracy=0.85,
    fail_fast=True  # Fail immediately for debugging
)

quantizer = ArrowQuantV2(mode="diffusion")

try:
    result = quantizer.quantize_diffusion_model(
        model_path="dream-7b/",
        output_path="dream-7b-test/",
        config=config
    )
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

**Run with debugging:**
```bash
python -m pdb minimal_repro.py
# Use 'n' to step, 'c' to continue, 'p variable' to print
```

---

### Performance Profiling

**Profile Python code:**
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run quantization
result = quantizer.quantize_diffusion_model(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

**Profile memory:**
```python
from memory_profiler import profile

@profile
def quantize_model():
    quantizer = ArrowQuantV2(mode="diffusion")
    result = quantizer.quantize_diffusion_model(...)
    return result

quantize_model()
```

---

## FAQ

### Q: What bit width should I use?

**A:** It depends on your deployment target and accuracy requirements:

- **INT2 (2-bit)**: Edge devices, maximum compression (10-16x), accuracy 0.65-0.75
- **INT4 (4-bit)**: Local machines, balanced (4-8x), accuracy 0.80-0.90
- **INT8 (8-bit)**: Cloud servers, high accuracy (2-4x), accuracy 0.95+

Start with INT4 for most use cases. Use INT2 only if size is critical and you can accept lower accuracy.

---

### Q: Why is my quantized model larger than expected?

**A:** Several factors affect model size:

1. **Metadata overhead**: Quantization parameters add metadata
2. **Group size**: Smaller groups (32, 64) add more metadata
3. **Time groups**: More time groups add more parameters
4. **Parquet compression**: Parquet files may not be optimally compressed

**Check actual compression:**
```bash
du -sh dream-7b/
du -sh dream-7b-quantized/

# Calculate ratio
python -c "print(f'{560/35:.1f}x')"  # Example: 560MB → 35MB = 16x
```

---

### Q: Can I quantize only specific layers?

**A:** Not yet. The current implementation quantizes all layers uniformly. Mixed-precision quantization (different bit widths per layer) is planned for a future release.

**Workaround:** Quantize the entire model, then manually replace specific layers from the original model.

---

### Q: How do I know if quantization succeeded?

**A:** Check these indicators:

1. **Validation passed**: `result['passed'] == True`
2. **Cosine similarity**: Meets or exceeds threshold
3. **Compression ratio**: Matches expected ratio for bit width
4. **No errors**: Quantization completed without exceptions

```python
result = quantizer.quantize_diffusion_model(...)

print(f"✓ Quantization completed")
print(f"✓ Cosine similarity: {result['cosine_similarity']:.3f}")
print(f"✓ Compression ratio: {result['compression_ratio']:.1f}x")
print(f"✓ Model size: {result['model_size_mb']:.1f} MB")
print(f"✓ Time: {result['quantization_time_s']:.1f}s")
```

---

### Q: What's the difference between time-aware and spatial quantization?

**A:**

**Time-aware quantization:**
- For discrete diffusion (text/code models)
- Groups timesteps with similar activation patterns
- Handles temporal variance in diffusion process
- Enabled by default for text/code

**Spatial quantization:**
- For continuous diffusion (image/audio models)
- Per-channel equalization and activation smoothing
- Handles spatial variance in feature maps
- Enabled by default for image/audio

**Both can be enabled simultaneously** for models that benefit from both strategies.

---

### Q: Can I use this with non-diffusion models?

**A:** Yes, use `mode="base"` for standard quantization:

```python
quantizer = ArrowQuantV2(mode="base")
result = quantizer.quantize(
    weights=model_weights,
    bit_width=4
)
```

However, diffusion-specific optimizations (time-aware, spatial) won't be applied.

---

### Q: How do I report a bug?

**A:** Include the following information:

1. **Error message**: Full error with stack trace
2. **Configuration**: Your config.yaml or Python config
3. **Environment**: OS, Python version, RAM, CPU cores
4. **Model**: Model name, size, modality
5. **Reproduction**: Minimal code to reproduce the issue

```python
# Example bug report template
"""
Error: ValidationError: cosine similarity 0.623 below threshold 0.850

Configuration:
- bit_width: 2
- num_time_groups: 10
- enable_streaming: False
- min_accuracy: 0.85

Environment:
- OS: Ubuntu 22.04
- Python: 3.10.12
- RAM: 16GB
- CPU: Intel i7-10700K (8 cores)

Model:
- Name: Dream 7B
- Size: 560MB
- Modality: text

Reproduction:
[paste minimal code here]
"""
```

---

### Q: Where can I find more help?

**A:** Check these resources:

1. **Documentation**:
   - [Quickstart Guide](QUICKSTART.md)
   - [Configuration Guide](CONFIGURATION_GUIDE.md)
   - [API Reference](API_REFERENCE.md)
   - [Architecture](ARCHITECTURE.md)

2. **Examples**:
   - `scripts/quantize_diffusion.py` - Basic quantization
   - `tests/test_python_bindings.py` - Python API examples
   - `config.example.yaml` - Configuration examples

3. **Design Documents**:
   - `.kiro/specs/arrowquant-v2-diffusion/design.md` - Technical design
   - `.kiro/specs/arrowquant-v2-diffusion/requirements.md` - Requirements

---

## Quick Reference

### Common Commands

```bash
# Quantize with default config
python scripts/quantize_diffusion.py \
    --model dream-7b/ \
    --output dream-7b-quantized/

# Quantize with custom config
python scripts/quantize_diffusion.py \
    --model dream-7b/ \
    --output dream-7b-int2/ \
    --config config.yaml

# Validate quantization
python scripts/validate_quality.py \
    --original dream-7b/ \
    --quantized dream-7b-quantized/

# Check model info
python scripts/model_info.py dream-7b/
```

### Environment Variables

```bash
# Override bit width
export ARROW_QUANT_BIT_WIDTH=2

# Enable streaming
export ARROW_QUANT_ENABLE_STREAMING=true

# Set accuracy threshold
export ARROW_QUANT_MIN_ACCURACY=0.70

# Limit threads
export ARROW_QUANT_NUM_THREADS=4
```

### Python Quick Start

```python
from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig

# Simple quantization
quantizer = ArrowQuantV2(mode="diffusion")
config = DiffusionQuantConfig.from_profile("local")

result = quantizer.quantize_diffusion_model(
    model_path="dream-7b/",
    output_path="dream-7b-quantized/",
    config=config
)

print(f"Done! Compression: {result['compression_ratio']:.1f}x")
```

---

## Summary

This troubleshooting guide covers:

- ✅ Common configuration errors and fixes
- ✅ Performance optimization techniques
- ✅ Quality validation and improvement
- ✅ Memory management strategies
- ✅ Debugging tools and techniques
- ✅ FAQ for common questions

For additional help, refer to the [Configuration Guide](CONFIGURATION_GUIDE.md) and [API Reference](API_REFERENCE.md).

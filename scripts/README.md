# ArrowQuant V2 Scripts

This directory contains production-ready scripts for offline quantization of diffusion models.

## Scripts

### `quantize_diffusion.py`

Main offline quantization script with comprehensive CLI support.

**Features:**
- Single model and batch quantization
- All configuration options via CLI
- Progress reporting and logging
- Error handling with graceful degradation
- Summary report generation
- Environment variable support

## Quick Start

### Single Model Quantization

```bash
# Edge deployment (INT2, <35MB)
python scripts/quantize_diffusion.py \
    --model dream-7b/ \
    --output dream-7b-edge/ \
    --profile edge

# Local deployment (INT4, balanced)
python scripts/quantize_diffusion.py \
    --model dream-7b/ \
    --output dream-7b-local/ \
    --profile local

# Cloud deployment (INT8, high accuracy)
python scripts/quantize_diffusion.py \
    --model dream-7b/ \
    --output dream-7b-cloud/ \
    --profile cloud
```

### Custom Configuration

```bash
# Custom parameters
python scripts/quantize_diffusion.py \
    --model dream-7b/ \
    --output dream-7b-custom/ \
    --bit-width 4 \
    --num-time-groups 15 \
    --group-size 64 \
    --min-accuracy 0.80 \
    --calibration-samples 256

# Using YAML config file
python scripts/quantize_diffusion.py \
    --model dream-7b/ \
    --output dream-7b-yaml/ \
    --config ../config.example.yaml

# Override YAML config with CLI args
python scripts/quantize_diffusion.py \
    --model dream-7b/ \
    --output dream-7b-override/ \
    --config ../config.example.yaml \
    --bit-width 2 \
    --min-accuracy 0.70
```

### Batch Quantization

```bash
# Quantize multiple models
python scripts/quantize_diffusion.py \
    --batch batch_example.json \
    --output-dir quantized_models/

# With custom report path
python scripts/quantize_diffusion.py \
    --batch batch_example.json \
    --output-dir quantized_models/ \
    --report batch_report.json
```

### Batch Configuration File

Create a JSON file with quantization jobs:

```json
{
  "jobs": [
    {
      "model_path": "models/dream-7b/",
      "output_path": "dream-7b-edge/",
      "profile": "edge",
      "bit_width": 2,
      "modality": "text"
    },
    {
      "model_path": "models/dream-7b/",
      "output_path": "dream-7b-local/",
      "profile": "local",
      "bit_width": 4
    }
  ]
}
```

See `batch_example.json` for a complete example.

## Command-Line Options

### Input/Output

- `--model PATH`: Path to input model directory (single model mode)
- `--batch PATH`: Path to JSON batch configuration file (batch mode)
- `--output PATH`: Path to output directory (required for `--model`)
- `--output-dir PATH`: Output directory for batch quantization (required for `--batch`)

### Configuration

- `--config PATH`: Path to YAML configuration file
- `--profile {edge,local,cloud}`: Deployment profile

### Quantization Parameters

- `--bit-width {2,4,8}`: Target bit width
- `--modality {text,code,image,audio}`: Model modality (auto-detected if not specified)
- `--num-time-groups N`: Number of time groups (1-100)
- `--group-size {32,64,128,256}`: Group size for quantization
- `--enable-time-aware`: Enable time-aware quantization
- `--disable-time-aware`: Disable time-aware quantization
- `--enable-spatial`: Enable spatial quantization
- `--disable-spatial`: Disable spatial quantization
- `--min-accuracy FLOAT`: Minimum cosine similarity threshold (0.0-1.0)
- `--calibration-samples N`: Number of calibration samples
- `--fail-fast`: Disable automatic fallback to higher bit widths

### Output Control

- `--verbose`: Enable verbose output
- `--quiet`: Suppress progress messages
- `--report PATH`: Path to save JSON summary report (default: `quantization_report.json`)

## Environment Variables

Override configuration parameters using environment variables:

```bash
export ARROW_QUANT_BIT_WIDTH=2
export ARROW_QUANT_NUM_TIME_GROUPS=5
export ARROW_QUANT_MIN_ACCURACY=0.70
export ARROW_QUANT_ENABLE_STREAMING=true

python scripts/quantize_diffusion.py \
    --model dream-7b/ \
    --output dream-7b-env/ \
    --profile edge
```

Supported variables:
- `ARROW_QUANT_BIT_WIDTH`
- `ARROW_QUANT_NUM_TIME_GROUPS`
- `ARROW_QUANT_GROUP_SIZE`
- `ARROW_QUANT_MIN_ACCURACY`
- `ARROW_QUANT_CALIBRATION_SAMPLES`
- `ARROW_QUANT_FAIL_FAST`
- `ARROW_QUANT_NUM_THREADS`
- `ARROW_QUANT_ENABLE_STREAMING`

## Examples

### Example 1: Edge Deployment

Quantize Dream 7B for Raspberry Pi:

```bash
python scripts/quantize_diffusion.py \
    --model models/dream-7b/ \
    --output models/dream-7b-edge/ \
    --profile edge \
    --verbose
```

Expected output:
- Model size: ~35MB
- Compression: ~16x
- Accuracy: ≥0.65

### Example 2: High-Accuracy Quantization

Quantize with maximum accuracy:

```bash
python scripts/quantize_diffusion.py \
    --model models/dream-7b/ \
    --output models/dream-7b-high-accuracy/ \
    --bit-width 8 \
    --num-time-groups 20 \
    --group-size 64 \
    --enable-time-aware \
    --enable-spatial \
    --min-accuracy 0.95 \
    --calibration-samples 512
```

### Example 3: Fast Quantization

Minimize quantization time:

```bash
python scripts/quantize_diffusion.py \
    --model models/dream-7b/ \
    --output models/dream-7b-fast/ \
    --bit-width 4 \
    --num-time-groups 5 \
    --disable-spatial \
    --calibration-samples 32
```

### Example 4: Image Model Quantization

Quantize image diffusion model with spatial optimization:

```bash
python scripts/quantize_diffusion.py \
    --model models/stable-diffusion/ \
    --output models/stable-diffusion-int4/ \
    --modality image \
    --bit-width 4 \
    --enable-spatial \
    --group-size 64 \
    --min-accuracy 0.85
```

### Example 5: Batch Quantization

Quantize multiple models with different configurations:

```bash
# Create batch config
cat > my_batch.json << EOF
{
  "jobs": [
    {
      "model_path": "models/dream-7b/",
      "output_path": "dream-7b-edge/",
      "profile": "edge"
    },
    {
      "model_path": "models/dream-7b/",
      "output_path": "dream-7b-local/",
      "profile": "local"
    },
    {
      "model_path": "models/stable-diffusion/",
      "output_path": "sd-int4/",
      "bit_width": 4,
      "modality": "image",
      "enable_spatial": true
    }
  ]
}
EOF

# Run batch quantization
python scripts/quantize_diffusion.py \
    --batch my_batch.json \
    --output-dir quantized/ \
    --report batch_report.json \
    --verbose
```

## Output

### Console Output

The script provides detailed progress information:

```
2024-01-15 10:30:00 - INFO - Quantization configuration:
2024-01-15 10:30:00 - INFO -   Model: dream-7b/
2024-01-15 10:30:00 - INFO -   Output: dream-7b-edge/
2024-01-15 10:30:00 - INFO -   Bit width: 2
2024-01-15 10:30:00 - INFO -   Profile: edge
2024-01-15 10:30:00 - INFO - Starting quantization of dream-7b/...
2024-01-15 10:30:05 - INFO - [dream-7b] Detecting model modality... (10.0% complete, 5.0s elapsed)
2024-01-15 10:30:10 - INFO - [dream-7b] Detected text modality (15.0% complete, 10.0s elapsed)
2024-01-15 10:30:15 - INFO - [dream-7b] Quantizing model layers... (20.0% complete, 15.0s elapsed)
...
2024-01-15 10:35:00 - INFO - Quantization completed successfully!
2024-01-15 10:35:00 - INFO -   Output: dream-7b-edge/
2024-01-15 10:35:00 - INFO -   Size: 34.50 MB
2024-01-15 10:35:00 - INFO -   Compression: 16.00x
2024-01-15 10:35:00 - INFO -   Accuracy: 0.7200
2024-01-15 10:35:00 - INFO -   Modality: text
2024-01-15 10:35:00 - INFO -   Bit width: 2
2024-01-15 10:35:00 - INFO -   Time: 300.00s
```

### Summary Report

The script generates a JSON summary report:

```json
{
  "summary": {
    "total_jobs": 3,
    "successful": 3,
    "failed": 0,
    "success_rate": 1.0,
    "total_size_mb": 245.5,
    "avg_compression_ratio": 10.5,
    "avg_cosine_similarity": 0.82,
    "total_time_s": 900.0
  },
  "results": [
    {
      "model_path": "models/dream-7b/",
      "output_path": "dream-7b-edge/",
      "success": true,
      "quantized_path": "dream-7b-edge/",
      "compression_ratio": 16.0,
      "cosine_similarity": 0.72,
      "model_size_mb": 34.5,
      "modality": "text",
      "bit_width": 2,
      "quantization_time_s": 300.0,
      "error_message": null
    }
  ]
}
```

### Log File

All output is also saved to `quantization.log` for debugging and auditing.

## Error Handling

The script handles errors gracefully:

1. **Configuration errors**: Validates all parameters before quantization
2. **Missing models**: Checks if model path exists
3. **Quantization failures**: Logs detailed error messages and continues with next job
4. **Fallback**: Automatically falls back to higher bit widths if accuracy threshold not met (unless `--fail-fast` is specified)

Example error output:

```
2024-01-15 10:30:00 - ERROR - Model path does not exist: models/nonexistent/
2024-01-15 10:30:00 - WARNING - Job 1 failed: Model path does not exist: models/nonexistent/
```

## Troubleshooting

### Issue: Import Error

**Error**: `Failed to import arrow_quant_v2`

**Solution**: Install the package:
```bash
cd ai_os_diffusion/arrow_quant_v2
pip install -e .
```

### Issue: Out of Memory

**Error**: Process killed or OOM errors

**Solution**: Enable streaming mode:
```bash
export ARROW_QUANT_ENABLE_STREAMING=true
python scripts/quantize_diffusion.py --model ... --output ...
```

### Issue: Low Accuracy

**Error**: Cosine similarity below threshold

**Solution**: Try higher bit width or more calibration samples:
```bash
python scripts/quantize_diffusion.py \
    --model ... \
    --output ... \
    --bit-width 4 \
    --calibration-samples 256
```

### Issue: Slow Quantization

**Solution**: Reduce calibration samples or disable spatial quantization:
```bash
python scripts/quantize_diffusion.py \
    --model ... \
    --output ... \
    --calibration-samples 32 \
    --disable-spatial
```

## Integration with CI/CD

### Docker Example

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install -e ai_os_diffusion/arrow_quant_v2/

# Set quantization configuration
ENV ARROW_QUANT_BIT_WIDTH=2
ENV ARROW_QUANT_DEPLOYMENT_PROFILE=edge

CMD ["python", "ai_os_diffusion/arrow_quant_v2/scripts/quantize_diffusion.py", \
     "--model", "/models/input", \
     "--output", "/models/output", \
     "--report", "/reports/quantization_report.json"]
```

### GitHub Actions Example

```yaml
name: Quantize Models

on:
  push:
    paths:
      - 'models/**'

jobs:
  quantize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e ai_os_diffusion/arrow_quant_v2/
      
      - name: Quantize models
        run: |
          python ai_os_diffusion/arrow_quant_v2/scripts/quantize_diffusion.py \
            --batch batch_config.json \
            --output-dir quantized_models/ \
            --report quantization_report.json
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: quantized-models
          path: quantized_models/
      
      - name: Upload report
        uses: actions/upload-artifact@v2
        with:
          name: quantization-report
          path: quantization_report.json
```

## Additional Resources

- [Quickstart Guide](../docs/QUICKSTART.md)
- [Configuration Guide](../docs/CONFIGURATION_GUIDE.md)
- [API Reference](../docs/API_REFERENCE.md)
- [Troubleshooting](../docs/TROUBLESHOOTING.md)

## Support

For issues or questions:
1. Check the [Troubleshooting Guide](../docs/TROUBLESHOOTING.md)
2. Review the [Configuration Guide](../docs/CONFIGURATION_GUIDE.md)
3. Check the log file: `quantization.log`
4. Open an issue on GitHub with the error message and configuration


### `release.sh`
Automated release build script for creating production releases.

**Usage:**
```bash
# Build everything (Rust + Python)
./scripts/release.sh --version 0.1.0

# Build Rust library only
./scripts/release.sh --version 0.1.0 --rust-only

# Build Python wheels only
./scripts/release.sh --version 0.1.0 --python-only

# Skip tests (faster)
./scripts/release.sh --version 0.1.0 --no-tests

# Dry run (no publishing)
./scripts/release.sh --version 0.1.0 --dry-run

# Build and publish to PyPI
./scripts/release.sh --version 0.1.0 --publish
```

**Features:**
- Automatic version management in Cargo.toml and pyproject.toml
- Rust library building for current platform
- Python wheel building with maturin
- Source distribution (sdist) generation
- Test execution (Rust and Python)
- PyPI publishing with twine
- Checksum generation (SHA256)
- Comprehensive error handling

**Requirements:**
- Rust 1.70+ with cargo
- Python 3.10+
- Maturin (for Python wheels): `pip install maturin`
- Twine (for PyPI publishing, optional): `pip install twine`

**Release Workflow:**
```bash
# 1. Build and test locally
./scripts/release.sh --version 0.1.0 --dry-run

# 2. Review artifacts
ls -lh target/release/
ls -lh dist/

# 3. Create git tag
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0

# 4. GitHub Actions will automatically create release
# Or manually publish to PyPI
./scripts/release.sh --version 0.1.0 --publish
```

For detailed release documentation, see [docs/RELEASE.md](../docs/RELEASE.md).

# Quantization Validation Guide

This guide explains how to run the end-to-end quantization validation tests and interpret the results.

---

## Overview

The quantization validation suite tests the complete ArrowQuantizer pipeline with real models to ensure:

1. **Compression Ratio**: INT8 >2x, INT2 >4x
2. **Accuracy**: INT8 >0.85 cosine similarity, <15% precision loss
3. **Schema Compatibility**: V1/V2 format compatibility
4. **Performance**: Quantization completes in <60s

---

## Prerequisites

### System Requirements

- **Python**: 3.10+
- **Memory**: 16GB+ RAM recommended
- **Storage**: 2GB free space for models and results
- **Internet**: Required for downloading models

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
pip install -e .
```

Additional test dependencies:

```bash
pip install pytest pytest-json-report transformers sentence-transformers
```

---

## Running Validation Tests

### Option 1: Run All Tests (Recommended)

Run the complete validation suite:

```bash
python scripts/run_quantization_validation.py
```

This will:
1. Download MiniLM model (~90MB)
2. Run all validation tests
3. Generate validation report
4. Display summary results

**Expected output**:
```
================================================================================
Quantization Validation Test Suite
================================================================================
Model: minilm
Output directory: validation_results

Running tests: tests/integration/test_quantization_e2e.py
Tests completed successfully!
Execution time: 127.34s

Extracting metrics...
Validating acceptance criteria...
Generating summary report...

================================================================================
Validation Summary
================================================================================
✅ ALL ACCEPTANCE CRITERIA PASSED

Full report: validation_results/validation_summary.md
Detailed report: docs/QUANTIZATION_VALIDATION_REPORT.md
================================================================================
```

### Option 2: Run Specific Test Categories

Run only compression ratio tests:

```bash
pytest tests/integration/test_quantization_e2e.py::TestCompressionRatioValidation -v
```

Run only accuracy tests:

```bash
pytest tests/integration/test_quantization_e2e.py::TestAccuracyValidation -v
```

Run only schema compatibility tests:

```bash
pytest tests/integration/test_quantization_e2e.py::TestSchemaCompatibility -v
```

### Option 3: Run Individual Tests

Run a single test:

```bash
pytest tests/integration/test_quantization_e2e.py::TestRealModelQuantization::test_minilm_int8_quantization -v -s
```

---

## Understanding Test Results

### Test Categories

#### 1. Real Model Quantization

**Tests**:
- `test_minilm_int8_quantization`: Quantize MiniLM to INT8
- `test_minilm_int2_quantization`: Quantize MiniLM to INT2

**What it validates**:
- Model conversion pipeline works end-to-end
- Quantized weights are stored correctly
- Schema version is V2

**Expected results**:
- ✅ Quantization completes without errors
- ✅ Output file exists and is valid Parquet
- ✅ All layers quantized (or skipped for mixed precision)

#### 2. Compression Ratio Validation

**Tests**:
- `test_int8_compression_ratio_target`: Validate INT8 >2x
- `test_int2_compression_ratio_target`: Validate INT2 >4x
- `test_compression_ratio_comparison`: Compare INT8 vs INT2

**What it validates**:
- File size reduction meets targets
- Compression efficiency is acceptable
- INT2 compresses more than INT8

**Expected results**:
- ✅ INT8: 2.1-2.3x compression
- ✅ INT2: 4.5-5.2x compression
- ✅ INT2 > INT8 compression ratio

#### 3. Accuracy Validation

**Tests**:
- `test_int8_accuracy_target`: Validate INT8 >0.85 similarity
- `test_int2_accuracy_target`: Validate INT2 >0.70 similarity
- `test_layer_wise_accuracy_distribution`: Analyze per-layer accuracy

**What it validates**:
- Quantized weights preserve accuracy
- Precision loss is within acceptable range
- No catastrophic accuracy degradation

**Expected results**:
- ✅ INT8: 0.87-0.92 cosine similarity, 8-13% loss
- ✅ INT2: 0.72-0.78 cosine similarity, 22-28% loss
- ✅ All layers maintain reasonable accuracy

#### 4. Schema Compatibility

**Tests**:
- `test_v1_schema_read`: Read V1 Parquet files
- `test_v2_schema_write`: Write V2 Parquet files
- `test_v2_to_v1_compatibility`: V2 backward compatibility

**What it validates**:
- V1 and V2 schemas are correctly implemented
- V2 files can be read by V1-compatible loaders
- No data corruption during conversion

**Expected results**:
- ✅ V1 files read correctly
- ✅ V2 files written correctly
- ✅ V2 files load as PyTorch tensors

#### 5. Performance Benchmarks

**Tests**:
- `test_quantization_speed`: Measure quantization time
- `test_weight_loading_speed`: Measure loading time

**What it validates**:
- Quantization completes in reasonable time
- Weight loading is fast enough for inference
- No performance regressions

**Expected results**:
- ✅ INT8 quantization: 15-25s
- ✅ INT2 quantization: 20-30s
- ✅ Weight loading: 0.3-0.8s

---

## Interpreting Validation Report

### Report Structure

The validation report (`docs/QUANTIZATION_VALIDATION_REPORT.md`) contains:

1. **Executive Summary**: High-level results and status
2. **Test Environment**: Hardware, software, and model details
3. **Validation Results**: Detailed metrics for each test category
4. **Acceptance Criteria**: Pass/fail status for each criterion
5. **Recommendations**: Production deployment guidance

### Key Metrics

#### Compression Ratio

**Formula**: `original_size / quantized_size`

**Interpretation**:
- **2.0x**: Theoretical maximum for INT8 (FP16 → INT8)
- **8.0x**: Theoretical maximum for INT2 (FP16 → INT2)
- **Actual < Theoretical**: Due to metadata overhead

**Example**:
```
Original: 90.2 MB (FP16)
INT8: 42.1 MB → 2.14x compression (107% efficiency)
INT2: 17.8 MB → 5.07x compression (63% efficiency)
```

#### Cosine Similarity

**Formula**: `dot(orig, quant) / (norm(orig) * norm(quant))`

**Interpretation**:
- **1.0**: Perfect similarity (no loss)
- **>0.95**: Excellent (minimal loss)
- **0.85-0.95**: Good (acceptable loss)
- **0.70-0.85**: Fair (noticeable loss)
- **<0.70**: Poor (significant loss)

**Example**:
```
INT8: 0.894 → Good accuracy
INT2: 0.748 → Fair accuracy (expected for INT2)
```

#### Precision Loss

**Formula**: `(1 - cosine_similarity) * 100`

**Interpretation**:
- **<5%**: Excellent (production-ready)
- **5-15%**: Good (PTQ baseline)
- **15-30%**: Fair (acceptable for INT2)
- **>30%**: Poor (not recommended)

**Example**:
```
INT8: 10.6% → Within PTQ baseline (<15%)
INT2: 25.2% → Acceptable for INT2
```

---

## Troubleshooting

### Common Issues

#### 1. Model Download Fails

**Error**:
```
Failed to load MiniLM model: Connection timeout
```

**Solution**:
- Check internet connection
- Try again (Hugging Face may be temporarily down)
- Use a different model: `--model qwen`

#### 2. Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solution**:
- Close other applications
- Use CPU-only mode (set `CUDA_VISIBLE_DEVICES=""`)
- Reduce batch size in tests

#### 3. Tests Timeout

**Error**:
```
pytest timeout after 300s
```

**Solution**:
- Increase timeout: `pytest --timeout=600`
- Run tests individually
- Check system resources (CPU/memory)

#### 4. Accuracy Below Target

**Error**:
```
AssertionError: INT8 average similarity 0.82 < 0.85 target
```

**Solution**:
- Check model quality (may be corrupted)
- Try different quantization config (per-channel vs per-tensor)
- Review layer-wise accuracy distribution

#### 5. Compression Ratio Below Target

**Error**:
```
AssertionError: INT8 compression ratio 1.8x < 2.0x target
```

**Solution**:
- Check metadata overhead (should be <5%)
- Verify quantization is actually applied (not skipped)
- Review mixed precision configuration

---

## Advanced Usage

### Custom Model Testing

Test with your own model:

```python
# Create custom test
import pytest
from transformers import AutoModel

@pytest.fixture
def custom_model():
    model = AutoModel.from_pretrained("your-model-name")
    return model

def test_custom_model_quantization(custom_model, tmp_path):
    # ... quantization code ...
    pass
```

### Custom Quantization Config

Test with custom configuration:

```python
config = QuantizationConfig(
    quant_type='int8',
    per_channel=True,
    symmetric=False,  # Try asymmetric
    group_size=64,    # Try different group size
    mixed_precision_layers=['embed', 'lm_head', 'layer.0']
)
```

### Batch Testing

Test multiple configurations:

```bash
# Test all quantization modes
for mode in int8 int2; do
    pytest tests/integration/test_quantization_e2e.py \
        -k "test_${mode}" \
        --output-dir="results_${mode}"
done
```

---

## Continuous Integration

### GitHub Actions

Add to `.github/workflows/quantization-validation.yml`:

```yaml
name: Quantization Validation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      - name: Run validation
        run: python scripts/run_quantization_validation.py
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: validation-results
          path: validation_results/
```

---

## References

- **ArrowQuantizer Implementation**: `llm_compression/inference/arrow_quantizer.py`
- **Unit Tests**: `tests/unit/test_arrow_quantizer.py`
- **Integration Tests**: `tests/integration/test_quantization_e2e.py`
- **Design Document**: `.kiro/specs/phase-2-quality-optimization/design.md`
- **Requirements**: `.kiro/specs/phase-2-quality-optimization/requirements.md`

---

## Support

For issues or questions:

1. Check this guide first
2. Review test output and logs
3. Check existing issues on GitHub
4. Create a new issue with:
   - Test output
   - System information
   - Steps to reproduce

---

**Last Updated**: 2024-02-17  
**Version**: Phase 2.0

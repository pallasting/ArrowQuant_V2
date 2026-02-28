# Baseline Thermodynamic Metrics

This document describes the baseline Markov smoothness metrics established for the thermodynamic enhancement feature (Phase 1).

## Overview

The baseline metrics establish a reference point for measuring the effectiveness of thermodynamic constraints in quantization. These metrics are collected during standard time-aware quantization **without** any smoothing or optimization applied.

## Key Metrics

### 1. Smoothness Score

**Definition**: A normalized score (0-1) measuring how smoothly quantization parameters transition between time groups.

**Formula**:
```
smoothness_score = 1 - (total_jump / max_possible_jump)

where:
  total_jump = Σ (scale_jump_i + zero_point_jump_i)
  max_possible_jump = 2 × (num_groups - 1)
```

**Expected Baseline (Dream 7B INT2)**:
- **Range**: 0.65 - 0.78
- **Typical**: ~0.70

**Interpretation**:
- **0.90+**: Excellent smoothness (Phase 3 target)
- **0.82+**: Good smoothness (Phase 2 target)
- **0.65-0.78**: Baseline (Phase 1, no enhancement)
- **<0.65**: Poor smoothness (needs enhancement)

### 2. Violation Count

**Definition**: Number of boundaries where parameter jumps exceed the smoothness threshold (default: 30%).

**Expected Baseline (Dream 7B INT2)**:
- **Typical**: 1-3 violations per model
- **Range**: 0-5 violations

**Severity Levels**:
- **Low**: <30% jump (warning)
- **Medium**: 30-50% jump (concerning)
- **High**: >50% jump (critical)

### 3. Boundary Scores

**Definition**: Per-boundary smoothness scores showing which transitions are problematic.

**Expected Baseline**:
- **Min**: 0.40 - 0.60 (worst boundary)
- **Max**: 0.95 - 1.00 (best boundary)
- **Mean**: 0.70 - 0.80
- **Std**: 0.10 - 0.20

## Common Violation Patterns

Based on empirical observations, common patterns include:

### 1. Early Boundary Violations
- **Pattern**: Violations cluster at boundaries 0-1
- **Cause**: Large parameter changes in early diffusion timesteps
- **Impact**: Affects initial denoising quality
- **Solution**: Phase 2 smoothing with larger window

### 2. Late Boundary Violations
- **Pattern**: Violations cluster at final boundaries
- **Cause**: Transition to near-zero noise levels
- **Impact**: Affects final image quality
- **Solution**: Phase 2 smoothing with sigmoid interpolation

### 3. Uniform Distribution
- **Pattern**: Violations spread across all boundaries
- **Cause**: Aggressive quantization (INT2) without calibration
- **Impact**: Overall quality degradation
- **Solution**: Phase 3 optimization with Markov constraints

### 4. High Severity Jumps
- **Pattern**: Few violations but >50% jumps
- **Cause**: Abrupt changes in weight distributions
- **Impact**: Error accumulation in diffusion process
- **Solution**: Phase 2 cubic smoothing + Phase 3 optimization

## Establishing Baseline

### Using the Script

```bash
# Run on Dream 7B (or any available model)
python scripts/establish_baseline_metrics.py \
    --model dream-7b/ \
    --output dream-7b-int2-baseline/ \
    --bit-width 2 \
    --report baseline_metrics.json \
    --html-report baseline_metrics.html

# Run with custom time groups
python scripts/establish_baseline_metrics.py \
    --model dream-7b/ \
    --output dream-7b-int2-baseline/ \
    --bit-width 2 \
    --num-time-groups 4 \
    --report baseline_metrics.json
```

### Output Files

1. **baseline_metrics.json**: Machine-readable metrics
2. **baseline_metrics.html**: Human-readable report with visualizations
3. **baseline_metrics.log**: Detailed execution log

### Example Output

```json
{
  "model_name": "dream-7b",
  "bit_width": 2,
  "num_time_groups": 4,
  "smoothness_score": 0.7234,
  "violation_count": 2,
  "violations": [
    {
      "boundary_idx": 0,
      "scale_jump": 0.58,
      "zero_point_jump": 0.12,
      "severity": "high"
    },
    {
      "boundary_idx": 2,
      "scale_jump": 0.35,
      "zero_point_jump": 0.08,
      "severity": "medium"
    }
  ],
  "boundary_scores": [0.45, 0.89, 0.72],
  "cosine_similarity": 0.7156,
  "compression_ratio": 16.0,
  "model_size_mb": 34.2,
  "min_boundary_score": 0.45,
  "max_boundary_score": 0.89,
  "mean_boundary_score": 0.6867,
  "std_boundary_score": 0.1823,
  "common_violation_patterns": [
    "High severity violations: 1 (50.0%)",
    "Medium severity violations: 1 (50.0%)",
    "Violations at boundaries: [0, 2]",
    "Scale jumps: max=58.0%, avg=46.5%"
  ]
}
```

## Interpreting Results

### Good Baseline (0.70-0.78)

**Characteristics**:
- Smoothness score in expected range
- 0-3 violations, mostly low/medium severity
- Boundary scores mostly >0.60
- Cosine similarity ≥0.70

**Action**: Proceed to Phase 2 for further improvement

### Poor Baseline (<0.65)

**Characteristics**:
- Smoothness score below expected range
- 4+ violations, including high severity
- Multiple boundary scores <0.50
- Cosine similarity <0.70

**Possible Causes**:
1. Aggressive quantization (INT2 without calibration)
2. Model architecture not suited for time-aware quantization
3. Insufficient time groups (try increasing to 8-15)
4. Poor weight distribution in model

**Actions**:
1. Try INT4 instead of INT2
2. Increase number of time groups
3. Enable calibration with representative samples
4. Proceed directly to Phase 3 (optimization)

### Excellent Baseline (>0.78)

**Characteristics**:
- Smoothness score above expected range
- 0-1 violations, low severity only
- All boundary scores >0.70
- Cosine similarity >0.75

**Action**: Model already has good smoothness, but Phase 2/3 can still improve accuracy

## Comparison with Enhanced Versions

| Metric | Baseline (Phase 1) | Phase 2 (Smoothing) | Phase 3 (Optimization) |
|--------|-------------------|---------------------|------------------------|
| Smoothness Score | 0.65-0.78 | 0.82+ | 0.90+ |
| Violation Count | 1-3 | 0-1 | 0 |
| INT2 Accuracy | Baseline | +2-3% | +6-8% |
| Overhead | <1% | <10% | <15% |

## Next Steps

After establishing baseline:

1. **Document Results**: Save baseline metrics for comparison
2. **Analyze Patterns**: Identify which violation patterns are present
3. **Plan Enhancement**: Decide which phases to implement based on needs
4. **Proceed to Phase 2**: Implement boundary smoothing if baseline is acceptable
5. **Skip to Phase 3**: Implement optimization if baseline is poor

## References

- [Thermodynamic Enhancement Design](.kiro/specs/thermodynamic-enhancement/design.md)
- [Thermodynamic Enhancement Requirements](.kiro/specs/thermodynamic-enhancement/requirements.md)
- [Thermodynamic Quantization Analysis](../../../THERMODYNAMIC_QUANTIZATION_ANALYSIS.md)

## Troubleshooting

### Script Fails to Import arrow_quant_v2

**Solution**: Install the package in editable mode:
```bash
cd ai_os_diffusion/arrow_quant_v2
pip install -e .
```

### Model Not Found

**Solution**: Ensure model path is correct and contains safetensors files:
```bash
ls -la dream-7b/
# Should show: model.safetensors or model-00001-of-00002.safetensors, etc.
```

### No Thermodynamic Metrics in Result

**Solution**: Ensure thermodynamic validation is enabled in config:
```yaml
quantization:
  thermodynamic:
    validation:
      enabled: true
```

### Unexpected Smoothness Score

**Possible Causes**:
1. Different model architecture than Dream 7B
2. Different number of time groups
3. Different bit width (INT4/INT8 typically have higher scores)
4. Model already pre-quantized or fine-tuned

**Solution**: Document actual baseline for your specific model and use it as reference.

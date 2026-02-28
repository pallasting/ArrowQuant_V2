# Baseline Metrics Script - Quick Start

This guide helps you establish baseline thermodynamic metrics for your quantized models.

## What Are Baseline Metrics?

Baseline metrics measure how smoothly quantization parameters transition between time groups **without** any thermodynamic enhancement. These metrics provide a reference point for measuring the effectiveness of:
- **Phase 2**: Boundary Smoothing (+2-3% accuracy)
- **Phase 3**: Transition Optimization (+6-8% accuracy)

## Quick Start

### 1. Basic Usage

```bash
# Establish baseline for Dream 7B (or any model)
python scripts/establish_baseline_metrics.py \
    --model dream-7b/ \
    --output dream-7b-int2-baseline/ \
    --bit-width 2 \
    --report baseline_metrics.json
```

### 2. With HTML Report

```bash
python scripts/establish_baseline_metrics.py \
    --model dream-7b/ \
    --output dream-7b-int2-baseline/ \
    --bit-width 2 \
    --report baseline_metrics.json \
    --html-report baseline_metrics.html
```

### 3. Custom Configuration

```bash
python scripts/establish_baseline_metrics.py \
    --model dream-7b/ \
    --output dream-7b-int2-baseline/ \
    --bit-width 2 \
    --num-time-groups 8 \
    --smoothness-threshold 0.25 \
    --report baseline_metrics.json
```

## Understanding the Output

### Console Output

```
================================================================================
BASELINE METRICS ESTABLISHED
================================================================================

Markov Smoothness Metrics:
  Overall smoothness score: 0.7234
  Violation count: 2
  Smoothness threshold: 0.3

Boundary Score Statistics:
  Min: 0.4523
  Max: 0.8912
  Mean: 0.6867
  Std: 0.1823
  Total boundaries: 3

Quantization Quality:
  Cosine similarity: 0.7156
  Compression ratio: 16.00x
  Model size: 34.20 MB
  Quantization time: 45.23s

Common Violation Patterns:
  - High severity violations: 1 (50.0%)
  - Medium severity violations: 1 (50.0%)
  - Violations at boundaries: [0, 2]
  - Scale jumps: max=58.0%, avg=46.5%
================================================================================
```

### JSON Report

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
    }
  ],
  "boundary_scores": [0.45, 0.89, 0.72],
  "cosine_similarity": 0.7156,
  "compression_ratio": 16.0,
  "model_size_mb": 34.2
}
```

### HTML Report

Open `baseline_metrics.html` in a browser to see:
- Color-coded smoothness score indicator
- Interactive metrics dashboard
- Detailed violation table
- Pattern analysis

## Interpreting Your Results

### Smoothness Score

| Score | Interpretation | Action |
|-------|---------------|--------|
| **0.90+** | Excellent | Already very smooth, Phase 2/3 optional |
| **0.82-0.89** | Good | Phase 2 target achieved |
| **0.65-0.78** | Baseline | Normal for INT2, proceed to Phase 2 |
| **<0.65** | Poor | Consider Phase 3 or increase time groups |

### Violation Count

| Count | Interpretation | Action |
|-------|---------------|--------|
| **0** | Perfect | No violations detected |
| **1-3** | Normal | Expected for INT2 baseline |
| **4-6** | Concerning | Consider increasing time groups |
| **7+** | Poor | Try INT4 or Phase 3 optimization |

## Common Scenarios

### Scenario 1: Good Baseline (Score 0.70-0.78)

**What it means**: Your model has acceptable smoothness for INT2 quantization.

**Next steps**:
1. Save the baseline metrics for comparison
2. Proceed to Phase 2 (Boundary Smoothing) for +2-3% accuracy
3. Optionally proceed to Phase 3 for +6-8% total accuracy

### Scenario 2: Poor Baseline (Score <0.65)

**What it means**: Quantization parameters have large jumps between time groups.

**Next steps**:
1. Try increasing `--num-time-groups` to 8 or 15
2. Try INT4 instead of INT2: `--bit-width 4`
3. Skip directly to Phase 3 (Transition Optimization)
4. Review violation patterns in HTML report

### Scenario 3: Excellent Baseline (Score >0.78)

**What it means**: Your model already has good smoothness properties.

**Next steps**:
1. Document the excellent baseline
2. Phase 2/3 can still improve accuracy by 2-8%
3. Consider if the overhead is worth the accuracy gain

## Troubleshooting

### Error: "Failed to import arrow_quant_v2"

**Solution**: Install the package in editable mode:
```bash
cd ai_os_diffusion/arrow_quant_v2
pip install -e .
```

### Error: "Model not found"

**Solution**: Ensure model path contains safetensors files:
```bash
ls -la dream-7b/
# Should show: model.safetensors or model-00001-of-00002.safetensors
```

### Error: "No thermodynamic metrics in result"

**Solution**: This means thermodynamic validation is not enabled. Check that:
1. You're using the latest version of arrow_quant_v2
2. The `enable_thermodynamic_validation` parameter is supported
3. The config has `thermodynamic.validation.enabled: true`

### Unexpected Smoothness Score

**Possible causes**:
- Different model architecture (not Dream 7B)
- Different number of time groups
- Different bit width (INT4/INT8 have higher scores)
- Model already pre-quantized

**Solution**: Document your actual baseline and use it as your reference point.

## Advanced Usage

### Custom Smoothness Threshold

```bash
# More strict threshold (20% instead of 30%)
python scripts/establish_baseline_metrics.py \
    --model dream-7b/ \
    --output dream-7b-int2-baseline/ \
    --bit-width 2 \
    --smoothness-threshold 0.2 \
    --report baseline_metrics.json
```

### More Time Groups

```bash
# Use 8 time groups for finer granularity
python scripts/establish_baseline_metrics.py \
    --model dream-7b/ \
    --output dream-7b-int2-baseline/ \
    --bit-width 2 \
    --num-time-groups 8 \
    --report baseline_metrics.json
```

### Verbose Output

```bash
# Enable verbose logging
python scripts/establish_baseline_metrics.py \
    --model dream-7b/ \
    --output dream-7b-int2-baseline/ \
    --bit-width 2 \
    --report baseline_metrics.json \
    --verbose
```

## Expected Baselines by Model Type

| Model | Bit Width | Expected Score | Typical Violations |
|-------|-----------|---------------|-------------------|
| Dream 7B | INT2 | 0.65-0.78 | 1-3 |
| Dream 7B | INT4 | 0.75-0.85 | 0-2 |
| Dream 7B | INT8 | 0.85-0.95 | 0-1 |
| Smaller Models (<3B) | INT2 | 0.70-0.80 | 0-2 |
| Larger Models (>10B) | INT2 | 0.60-0.75 | 2-4 |

## Next Steps

After establishing your baseline:

1. **Save the metrics**: Keep `baseline_metrics.json` for comparison
2. **Review the HTML report**: Understand your violation patterns
3. **Decide on enhancement**:
   - Good baseline (0.70-0.78): Proceed to Phase 2
   - Poor baseline (<0.65): Consider Phase 3 or adjust config
   - Excellent baseline (>0.78): Phase 2/3 optional

4. **Compare with enhanced versions**:
   - Run Phase 2 smoothing and compare metrics
   - Run Phase 3 optimization and compare metrics
   - Document accuracy improvements

## Documentation

For more details, see:
- [Baseline Metrics Documentation](../docs/BASELINE_METRICS.md)
- [Thermodynamic Enhancement Design](../.kiro/specs/thermodynamic-enhancement/design.md)
- [Thermodynamic Enhancement Requirements](../.kiro/specs/thermodynamic-enhancement/requirements.md)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the log file: `baseline_metrics.log`
3. Consult the full documentation: `docs/BASELINE_METRICS.md`
4. Open an issue with your baseline metrics JSON attached

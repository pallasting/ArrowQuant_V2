# Benchmark CI Documentation

This document describes the benchmark CI/CD pipeline for ArrowQuant V2.

## Overview

The benchmark CI pipeline automatically runs performance benchmarks and detects regressions. It runs on:
- **Releases**: When a new release is published
- **Main branch pushes**: To track performance trends
- **Pull requests**: To catch regressions before merge
- **Manual trigger**: Via workflow_dispatch

## Features

### 1. Performance Benchmarks
- Runs Criterion benchmarks for Rust code
- Measures execution time for critical operations
- Compares against baseline (main branch or previous release)

### 2. Regression Detection
- Compares current performance vs baseline
- Fails CI if performance regresses >10% (configurable)
- Provides detailed comparison reports

### 3. Memory Benchmarks
- Measures peak memory usage during quantization
- Validates memory efficiency targets (<50% vs Python)
- Non-blocking (warnings only)

### 4. Trend Analysis
- Stores benchmark results for historical tracking
- Enables performance trend visualization
- Retains results for 1 year (releases) or 90 days (commits)

## Workflow Files

### `.github/workflows/benchmark-ci.yml`
Main benchmark CI workflow with the following jobs:

#### Job: `benchmark`
- Runs Criterion benchmarks on current code
- Checks out baseline code (main branch)
- Runs baseline benchmarks
- Compares results using `benches/compare_results.py`
- Fails if regression >10% detected
- Uploads benchmark reports as artifacts
- Comments on PRs with results

#### Job: `benchmark-trend`
- Stores benchmark results for trend analysis
- Only runs on releases and main branch pushes
- Creates timestamped benchmark history
- Retains results for 365 days

#### Job: `memory-benchmark`
- Runs memory usage benchmarks
- Validates memory efficiency
- Non-blocking (warnings only)

#### Job: `all-benchmarks-passed`
- Final check that all benchmarks passed
- Fails if performance regressions detected

## Configuration

### Regression Threshold
The default regression threshold is 10%. To change it, modify the `REGRESSION_THRESHOLD` environment variable in `.github/workflows/benchmark-ci.yml`:

```yaml
env:
  REGRESSION_THRESHOLD: 10  # Change to desired percentage
```

### Baseline Reference
By default, benchmarks compare against the `main` branch. To use a different baseline:

```bash
# Via workflow_dispatch
gh workflow run benchmark-ci.yml -f baseline_ref=v1.0.0
```

## Running Benchmarks Locally

### Run all benchmarks
```bash
cd ai_os_diffusion/arrow_quant_v2
cargo bench
```

### Run specific benchmark
```bash
cargo bench --bench quantization_bench
```

### Compare with baseline
```bash
# Run baseline
git checkout main
cargo bench -- --save-baseline main

# Run current
git checkout your-branch
cargo bench -- --baseline main
```

### Compare results manually
```bash
python benches/compare_results.py \
  --current target/criterion/ \
  --baseline target/criterion/ \
  --threshold 10 \
  --output comparison.json
```

## Benchmark Results

### Artifacts
Benchmark results are uploaded as GitHub Actions artifacts:

- **benchmark-results-current**: Current benchmark results (90 days retention)
- **benchmark-report**: Comparison report (JSON + Markdown, 90 days retention)
- **benchmark-history-{sha}**: Historical results (365 days retention)
- **memory-benchmark-results**: Memory usage results (90 days retention)

### Accessing Results
1. Go to the Actions tab in GitHub
2. Click on a workflow run
3. Scroll to "Artifacts" section
4. Download the desired artifact

## Benchmark Comparison Script

The `benches/compare_results.py` script supports two modes:

### 1. Criterion Benchmarks
```bash
python benches/compare_results.py \
  --current target/criterion/ \
  --baseline target/criterion/ \
  --threshold 10 \
  --output report.json \
  --format json
```

### 2. Memory Benchmarks (Legacy)
```bash
python benches/compare_results.py baseline.json current.json
```

### Output Formats
- **JSON**: Machine-readable comparison results
- **Markdown**: Human-readable report for PRs

## Performance Targets

### Quantization Speed
- **Target**: 5-10x faster than Python
- **Measurement**: Time to quantize 100M, 600M, 7B parameter models
- **Threshold**: <10% regression

### Memory Usage
- **Target**: <50% memory vs Python
- **Measurement**: Peak memory during quantization
- **Threshold**: Warning only (non-blocking)

### Benchmark Coverage
Current benchmarks:
- `time_aware_grouping`: TimeAwareQuantizer timestep grouping
- `spatial_per_group_quantize`: SpatialQuantizer per-group quantization
- `channel_equalization`: Channel equalization performance

## Adding New Benchmarks

### 1. Add Rust benchmark
Edit `benches/quantization_bench.rs`:

```rust
fn bench_new_feature(c: &mut Criterion) {
    c.bench_function("new_feature", |b| {
        b.iter(|| {
            // Your benchmark code
        });
    });
}

criterion_group!(
    benches,
    bench_time_aware_grouping,
    bench_spatial_quantization,
    bench_new_feature  // Add here
);
```

### 2. Run locally
```bash
cargo bench --bench quantization_bench
```

### 3. Commit and push
The CI will automatically run the new benchmark.

## Troubleshooting

### Benchmark CI fails with "No Criterion results found"
- Ensure `cargo bench` runs successfully locally
- Check that `target/criterion/` directory is created
- Verify benchmark names match in `quantization_bench.rs`

### False positive regressions
- Check if the regression is consistent across multiple runs
- Consider increasing the threshold temporarily
- Investigate if the baseline is appropriate

### Memory benchmark warnings
- Memory benchmarks are non-blocking
- Warnings indicate potential memory issues
- Investigate if memory usage exceeds 500MB

### Artifacts not uploaded
- Check workflow logs for upload errors
- Verify artifact paths exist
- Ensure retention days are within limits

## Best Practices

### 1. Run benchmarks before submitting PR
```bash
cargo bench
```

### 2. Compare with main branch
```bash
git checkout main
cargo bench -- --save-baseline main
git checkout your-branch
cargo bench -- --baseline main
```

### 3. Document performance changes
If your PR intentionally changes performance:
- Document the reason in PR description
- Provide before/after benchmark results
- Explain the trade-off (e.g., memory vs speed)

### 4. Investigate regressions immediately
- Don't ignore benchmark CI failures
- Regressions compound over time
- Fix or document before merging

## Future Enhancements

### Planned Features
- [ ] Performance trend visualization dashboard
- [ ] Automatic baseline selection (last stable release)
- [ ] Per-benchmark regression thresholds
- [ ] Benchmark result database for long-term tracking
- [ ] Slack/Discord notifications for regressions
- [ ] Comparison with other quantization libraries

### Integration Opportunities
- [ ] Integrate with Codecov for coverage + performance
- [ ] Add benchmark results to release notes
- [ ] Create performance badges for README
- [ ] Generate performance comparison charts

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [GitHub Actions Artifacts](https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts)
- [ArrowQuant V2 Design Document](../../docs/ARCHITECTURE.md)
- [Performance Optimization Guide](../../docs/CONFIGURATION_GUIDE.md#performance-tuning)

## Support

For questions or issues with benchmark CI:
1. Check this documentation
2. Review workflow logs in GitHub Actions
3. Open an issue with the `ci` label
4. Contact the maintainers

---

**Last Updated**: 2025-02-23
**Maintainer**: ArrowQuant V2 Team

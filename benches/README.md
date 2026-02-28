# ArrowQuant V2 Benchmarks

This directory contains performance and memory benchmarks for ArrowQuant V2.

## Available Benchmarks

### 1. Performance Benchmarks (Rust)

**File**: `quantization_bench.rs`

Criterion-based benchmarks for core quantization operations:
- Time-aware grouping
- Spatial quantization
- Channel equalization

**Usage**:
```bash
cd ai_os_diffusion/arrow_quant_v2
cargo bench --bench quantization_bench
```

**Output**: Results are saved to `target/criterion/` with HTML reports.

### 2. SIMD Performance Benchmarks (Rust)

**File**: `simd_benchmark.rs`

Comprehensive SIMD performance benchmarks measuring speedup of SIMD-accelerated operations:
- Quantization (SIMD vs scalar)
- Dequantization (SIMD vs scalar)
- Cosine similarity (SIMD vs scalar)
- Roundtrip quantization (quantize + dequantize)
- Realistic layer quantization (768×768, 3072×768, 12288×3072)
- Expected speedup: 2-4x with SIMD (AVX2 on x86_64, NEON on ARM64)

**Usage**:

Using the Python runner (recommended):
```bash
cd ai_os_diffusion/arrow_quant_v2
python benches/run_simd_benchmark.py
```

Using Cargo directly:
```bash
cd ai_os_diffusion/arrow_quant_v2
cargo bench --bench simd_benchmark
```

Run specific benchmark group:
```bash
cargo bench --bench simd_benchmark -- quantize
cargo bench --bench simd_benchmark -- dequantize
cargo bench --bench simd_benchmark -- cosine_similarity
```

**Output**: 
- Console output with performance summary
- HTML reports in `target/criterion/report/index.html`
- JSON data in `target/criterion/*/base/estimates.json`

**Expected Results**:
- AVX2 (x86_64): 3-4x speedup (processes 8 floats per instruction)
- NEON (ARM64): 2-3x speedup (processes 4 floats per instruction)
- Scalar fallback: 1x baseline performance

### 3. Parallelization Benchmarks (Python)

**File**: `parallel_benchmark.py`

Comprehensive parallelization benchmarking with memory profiling:
- Parallel scaling analysis (1, 2, 4, 8, 16 threads)
- Model size performance (100M, 600M, 7B parameters)
- Streaming vs batch memory comparison
- Speedup and efficiency metrics
- Target: 4-8x speedup on 8 cores

**Requirements**:
```bash
pip install psutil numpy
cd ai_os_diffusion/arrow_quant_v2
maturin develop --release
```

**Usage**:

Run all parallelization benchmarks:
```bash
python benches/parallel_benchmark.py --all
```

Benchmark specific core counts:
```bash
python benches/parallel_benchmark.py --cores 1,2,4,8,16
```

Benchmark specific model size:
```bash
python benches/parallel_benchmark.py --model-size 600M --cores 1,4,8
```

Compare streaming vs batch:
```bash
python benches/parallel_benchmark.py --streaming-comparison
```

Custom output directory:
```bash
python benches/parallel_benchmark.py --all --output-dir ./my_benchmarks
```

**Output**:
- `.benchmarks/parallel/parallel_benchmark_report.txt` - Text report with speedup analysis
- `.benchmarks/parallel/parallel_benchmark_results.json` - JSON results
- Speedup and efficiency metrics for each configuration

### 4. Parallelization Benchmarks (Rust)

**File**: `parallel_benchmark.rs`

Criterion-based benchmarks for parallelization performance:
- Parallel vs sequential quantization
- Speedup on different core counts (1, 2, 4, 8, 16)
- Performance with different model sizes (100M, 600M, 7B)
- Streaming vs batch mode comparison

**Usage**:
```bash
cd ai_os_diffusion/arrow_quant_v2
cargo bench --bench parallel_benchmark
```

**Output**: Results are saved to `target/criterion/` with HTML reports.

### 5. Quantization Speed Benchmarks (Python)

**File**: `speed_benchmark.py`

Comprehensive quantization speed benchmarking comparing Rust vs Python implementations:
- Measures quantization time for different model sizes (100M, 600M, 7B)
- Calculates throughput (params/second)
- Validates performance targets (100M: <2min, 600M: <10min, 7B: <5min)
- Compares Rust vs Python speedup (target: 5-10x)
- Generates detailed reports and charts

**Requirements**:
```bash
pip install psutil numpy matplotlib pyarrow
cd ai_os_diffusion/arrow_quant_v2
maturin develop --release
```

**Usage**:

Run all speed benchmarks:
```bash
python benches/speed_benchmark.py --all --generate-charts
```

Benchmark specific model size:
```bash
python benches/speed_benchmark.py --model-size 100M
python benches/speed_benchmark.py --model-size 600M
python benches/speed_benchmark.py --model-size 7B
```

Benchmark with specific bit width:
```bash
python benches/speed_benchmark.py --all --bit-width 4
```

Skip Python implementation (Rust only):
```bash
python benches/speed_benchmark.py --all --skip-python
```

Custom number of runs for averaging:
```bash
python benches/speed_benchmark.py --all --num-runs 5
```

Custom output directory:
```bash
python benches/speed_benchmark.py --all --output-dir ./my_benchmarks
```

**Output**:
- `.benchmarks/speed/speed_benchmark_report.txt` - Text report with speedup analysis
- `.benchmarks/speed/speed_benchmark_results.json` - JSON results
- `.benchmarks/speed/speed_comparison.png` - Time comparison chart
- `.benchmarks/speed/speedup_comparison.png` - Speedup chart
- `.benchmarks/speed/throughput_comparison.png` - Throughput chart

**Expected Results**:
- 100M model: <120s (target: <2 minutes)
- 600M model: <600s (target: <10 minutes)
- 7B model: <300s (target: <5 minutes, Dream 7B)
- Rust vs Python speedup: 5-10x

### 6. Quantization Speed Benchmarks (Rust)

**File**: `speed_benchmark.rs`

Criterion-based benchmarks for quantization speed:
- Measures quantization time for different model sizes (100M, 600M, 7B)
- Compares different bit widths (INT2, INT4, INT8)
- Dream 7B specific target validation (<5 minutes)
- Optimization level comparison (baseline, time-aware, spatial, all)

**Usage**:
```bash
cd ai_os_diffusion/arrow_quant_v2
cargo bench --bench speed_benchmark
```

Run specific benchmark group:
```bash
cargo bench --bench speed_benchmark -- quantization_speed
cargo bench --bench speed_benchmark -- bit_width_comparison
cargo bench --bench speed_benchmark -- dream_7b_target
cargo bench --bench speed_benchmark -- optimization_levels
```

**Output**: 
- Console output with performance summary
- HTML reports in `target/criterion/report/index.html`
- JSON data in `target/criterion/*/base/estimates.json`

### 7. Speed Benchmark Runner (Python)

**File**: `run_speed_benchmark.py`

Convenient runner script for executing both Rust and Python speed benchmarks:

**Usage**:

Run all benchmarks (Rust + Python):
```bash
python benches/run_speed_benchmark.py --all --generate-charts
```

Run Rust benchmarks only:
```bash
python benches/run_speed_benchmark.py --rust-only
```

Run Python benchmarks only:
```bash
python benches/run_speed_benchmark.py --python-only --all
```

Run specific model size:
```bash
python benches/run_speed_benchmark.py --model-size 100M
```

**Output**:
- Rust results: `target/criterion/report/index.html`
- Python results: `.benchmarks/speed/speed_benchmark_report.txt`
- Charts: `.benchmarks/speed/*.png`

### 8. Memory Benchmarks (Rust)

**File**: `memory_benchmark.rs`

Criterion-based benchmarks for memory usage during quantization:
- Streaming vs batch mode comparison
- Different model sizes (100M, 600M, 7B)
- Different bit widths (INT2, INT4, INT8)
- Target: Streaming should use <50% memory vs batch

**Usage**:
```bash
cd ai_os_diffusion/arrow_quant_v2
cargo bench --bench memory_benchmark
```

Run specific benchmark group:
```bash
cargo bench --bench memory_benchmark -- streaming_vs_batch
cargo bench --bench memory_benchmark -- model_sizes
cargo bench --bench memory_benchmark -- bit_widths
```

**Output**: 
- Console output with memory measurements
- HTML reports in `target/criterion/report/index.html`
- JSON data in `target/criterion/*/base/estimates.json`

### 9. Memory Benchmarks (Python)

**File**: `memory_benchmark.py`

Comprehensive memory profiling comparing Rust vs Python implementations:
- Measures peak memory usage during quantization
- Tests with models of different sizes (100M, 600M, 7B parameters)
- Generates comparison reports and charts
- Target: Rust should use <50% memory vs Python

**Requirements**:
```bash
pip install psutil matplotlib numpy pyarrow
```

**Usage**:

Benchmark a single model size:
```bash
python benches/memory_benchmark.py --model-size 100M
```

Benchmark all model sizes:
```bash
python benches/memory_benchmark.py --all --generate-charts
```

Benchmark with specific bit width:
```bash
python benches/memory_benchmark.py --model-size 600M --bit-width 4
```

Skip Python implementation (Rust only):
```bash
python benches/memory_benchmark.py --all --skip-python
```

Custom output directory:
```bash
python benches/memory_benchmark.py --all --output-dir ./my_benchmarks
```

**Output**:
- `.benchmarks/memory/memory_benchmark_report.txt` - Text report with comparison
- `.benchmarks/memory/memory_benchmark_results.json` - JSON results
- `.benchmarks/memory/memory_comparison.png` - Memory usage chart
- `.benchmarks/memory/memory_ratio.png` - Memory efficiency chart

### 10. Memory Benchmark Runner (Python)

**File**: `run_memory_benchmark.py`

Unified runner for both Rust and Python memory benchmarks:

**Usage**:

Run all benchmarks (Rust + Python):
```bash
python benches/run_memory_benchmark.py --all --generate-charts
```

Run Rust benchmarks only:
```bash
python benches/run_memory_benchmark.py --rust-only
```

Run Python benchmarks only:
```bash
python benches/run_memory_benchmark.py --python-only --model-size 100M
```

Run streaming comparison:
```bash
python benches/run_memory_benchmark.py --streaming-comparison
```

**Output**:
- Rust results: `target/criterion/report/index.html`
- Python results: `.benchmarks/memory/memory_benchmark_report.txt`
- Charts: `.benchmarks/memory/*.png`

## Benchmark Results Interpretation

### Memory Metrics

- **Peak Memory**: Maximum memory usage during quantization
- **Baseline Memory**: Memory usage before quantization starts
- **Quantization Memory**: Additional memory used during quantization (Peak - Baseline)
- **Memory Ratio**: Rust memory / Python memory (target: <50%)

### Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Memory Usage | <50% vs Python | ✅ Achieved |
| Quantization Speed | 5-10x vs Python | ⏳ Testing |
| Parallel Speedup (8 cores) | 4-8x vs 1 core | ⏳ Testing |
| 100M Model | <2 minutes | ⏳ Testing |
| 600M Model | <10 minutes | ⏳ Testing |
| 7B Model (Dream 7B) | <5 minutes | ⏳ Testing |
| Streaming Memory | <50% vs Batch | ⏳ Testing |

## Example Output

### Speed Benchmark Output

```
================================================================================
ArrowQuant V2 Quantization Speed Benchmark Report
================================================================================

Summary Table:
--------------------------------------------------------------------------------
Model Size   Bit Width    Rust Time       Python Time     Speedup    Status    
--------------------------------------------------------------------------------
100M         INT2         18.45           127.32          6.90x      ✅ Pass   
600M         INT2         95.23           687.45          7.22x      ✅ Pass   
7B           INT2         245.67          1834.21         7.47x      ✅ Pass   

Model Size: 100M, Bit Width: INT2
--------------------------------------------------------------------------------
Rust Implementation:
  Quantization Time: 18.45s
  Throughput: 5.42M params/s
  Model Parameters: 100M

Python Implementation:
  Quantization Time: 127.32s
  Throughput: 0.79M params/s
  Model Parameters: 100M

Comparison:
  Speedup (Python/Rust): 6.90x
  Throughput Ratio (Rust/Python): 6.86x
  ✅ Target achieved: 6.90x >= 5.0x

================================================================================
Performance Targets Validation
================================================================================

100M Model:
  Target: <120s
  Actual: 18.45s
  Status: ✅ Pass

600M Model:
  Target: <600s
  Actual: 95.23s
  Status: ✅ Pass

7B Model:
  Target: <300s
  Actual: 245.67s
  Status: ✅ Pass
```

### Parallelization Benchmark Output

```
================================================================================
ArrowQuant V2 Parallelization Benchmark Report
================================================================================

Parallel Scaling Results:
--------------------------------------------------------------------------------
Threads    Time (s)     Memory (MB)     Speedup    Efficiency  
--------------------------------------------------------------------------------
1          120.45       245.32          N/A        N/A         
2          65.23        267.45          1.85x      92.3%       
4          35.67        298.12          3.38x      84.4%       
8          18.92        345.67          6.37x      79.6%       
16         12.45        412.34          9.68x      60.5%       

✅ Target achieved: 6.37x speedup on 8 cores (target: 4-8x)

Model Size Results:
--------------------------------------------------------------------------------
Model Size      Time (s)     Memory (MB)     Target          Status    
--------------------------------------------------------------------------------
100M            18.92        345.67          <120s           ✅ Pass   
600M            95.34        1234.56         <600s           ✅ Pass   

Streaming vs Batch Comparison:
--------------------------------------------------------------------------------
Mode            Time (s)     Memory (MB)    
--------------------------------------------------------------------------------
Batch           95.34        1234.56        
Streaming       102.45       567.89         

Memory Ratio (Streaming/Batch): 46.01%
✅ Target achieved: <50% memory usage
```

### Memory Benchmark Output

```
================================================================================
ArrowQuant V2 Memory Benchmark Report
================================================================================

Model Size: 100M, Bit Width: INT2
--------------------------------------------------------------------------------
Rust Implementation:
  Peak Memory: 245.32 MB
  Quantization Memory: 123.45 MB
  Quantization Time: 45.23s

Python Implementation:
  Peak Memory: 512.67 MB
  Quantization Memory: 289.34 MB
  Quantization Time: 234.56s

Comparison:
  Memory Ratio (Rust/Python): 42.67%
  Speedup (Python/Rust): 5.18x
  ✅ Target achieved: <50% memory usage
```

## Continuous Benchmarking

To track performance over time:

1. Run benchmarks before making changes:
   ```bash
   python benches/memory_benchmark.py --all --generate-charts
   cp .benchmarks/memory/memory_benchmark_results.json .benchmarks/memory/baseline.json
   ```

2. Make your changes

3. Run benchmarks again:
   ```bash
   python benches/memory_benchmark.py --all --generate-charts
   ```

4. Compare results:
   ```bash
   python benches/compare_results.py .benchmarks/memory/baseline.json .benchmarks/memory/memory_benchmark_results.json
   ```

## Troubleshooting

### Rust Implementation Not Available

If you see "Rust ArrowQuantV2 not available":
```bash
cd ai_os_diffusion/arrow_quant_v2
maturin develop --release
```

### Python Implementation Not Found

Ensure the LLM compression package is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Out of Memory Errors

For large models (7B), ensure you have sufficient RAM:
- Minimum: 16GB RAM
- Recommended: 32GB RAM
- Use `--skip-python` to reduce memory pressure

### Chart Generation Fails

Install matplotlib:
```bash
pip install matplotlib
```

## Contributing

When adding new benchmarks:
1. Follow the existing structure
2. Document usage in this README
3. Include example output
4. Update performance targets table

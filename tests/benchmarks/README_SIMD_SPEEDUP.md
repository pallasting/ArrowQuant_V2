# SIMD Speedup Benchmarks

## 概述

此基准测试套件验证 SIMD 加速量化相对于标量实现的性能提升。

**验证需求**: 3.5, 8.1  
**验证属性**: 属性 7 - SIMD 性能提升

## 测试策略

### 测试的数组大小

按照任务要求，测试以下数组大小：

- **1K** (1,000 元素): 小型数组
- **10K** (10,000 元素): 中型数组
- **100K** (100,000 元素): 大型数组
- **1M** (1,000,000 元素): 超大型数组

### 预期性能提升

| 数组大小 | 预期加速比 | 说明 |
|---------|-----------|------|
| 1K      | 2-3x      | SIMD 开销相对较大 |
| 10K     | 3-4x      | SIMD 优势开始显现 |
| 100K    | 4-5x      | SIMD 充分发挥优势 |
| 1M      | 5-6x      | 最大 SIMD 加速 |

### 测试内容

1. **基本加速比测试** (`bench_simd_speedup`)
   - 测试不同数组大小的 SIMD vs 标量性能
   - 测量吞吐量（元素/秒）
   - 计算加速比

2. **位宽测试** (`bench_simd_bit_widths`)
   - 测试 2-bit, 4-bit, 8-bit 量化
   - 验证 SIMD 在不同位宽下的性能

3. **数据模式测试** (`bench_simd_data_patterns`)
   - 随机数据
   - 排序数据
   - 重复数据

## 运行基准测试

### 快速运行

```bash
# 运行所有 SIMD 加速比基准测试
cargo bench --bench bench_simd_speedup

# 或使用便捷脚本
./tests/benchmarks/run_simd_speedup_benchmark.sh
```

### 运行特定测试

```bash
# 只运行基本加速比测试
cargo bench --bench bench_simd_speedup -- simd_speedup

# 只运行位宽测试
cargo bench --bench bench_simd_speedup -- simd_bit_widths

# 只运行数据模式测试
cargo bench --bench bench_simd_speedup -- simd_data_patterns
```

### 保存和比较基线

```bash
# 保存当前结果为基线
cargo bench --bench bench_simd_speedup -- --save-baseline my_baseline

# 与基线比较
cargo bench --bench bench_simd_speedup -- --baseline my_baseline
```

## 查看结果

### HTML 报告

基准测试完成后，Criterion 会生成 HTML 报告：

```bash
# 打开报告
open target/criterion/simd_speedup/report/index.html

# 或在 Linux 上
xdg-open target/criterion/simd_speedup/report/index.html
```

### 命令行输出

基准测试会在命令行输出：

```
simd_speedup/simd/1K    time:   [2.1234 µs 2.1456 µs 2.1678 µs]
simd_speedup/scalar/1K  time:   [6.3456 µs 6.4123 µs 6.4789 µs]
                        change: [-3.2% -2.8% -2.4%] (p = 0.00 < 0.05)
                        Performance has improved.
```

### 加速比计算

加速比 = 标量时间 / SIMD 时间

例如：
- 标量: 6.4123 µs
- SIMD: 2.1456 µs
- 加速比: 6.4123 / 2.1456 = 2.99x

## 验收标准

根据任务 9.5 的要求，基准测试应验证：

✓ SIMD 实现比标量实现快 3x-6x（对于 ≥ 1K 的数组）  
✓ 测试覆盖 1K, 10K, 100K, 1M 数组大小  
✓ 使用 Criterion 进行准确的性能测量  
✓ 生成详细的性能报告

## 性能优化建议

如果 SIMD 加速比低于预期：

1. **检查 SIMD 可用性**
   ```bash
   cargo run --release --example check_simd
   ```

2. **检查编译优化**
   ```bash
   # 确保使用 release 模式
   cargo bench --release
   ```

3. **检查 CPU 特性**
   ```bash
   # x86_64
   cat /proc/cpuinfo | grep flags | grep avx2
   
   # ARM64
   cat /proc/cpuinfo | grep Features | grep asimd
   ```

4. **检查内存对齐**
   - SIMD 代码应确保数据按 16/32 字节对齐
   - 使用 `#[repr(align(32))]` 或手动对齐

## 故障排除

### SIMD 不可用

如果 SIMD 在您的平台上不可用：

```
⚠ SIMD may not be available - results will show scalar performance only
```

解决方案：
- 确保 CPU 支持 AVX2 (x86_64) 或 NEON (ARM64)
- 检查编译器标志是否启用 SIMD
- 在不支持 SIMD 的平台上，代码会自动回退到标量实现

### 基准测试运行时间过长

如果基准测试运行时间过长，可以调整配置：

```rust
group.sample_size(10);  // 减少样本数量（默认 20）
group.measurement_time(std::time::Duration::from_secs(3));  // 减少测量时间
```

### 结果不稳定

如果结果波动较大：

1. 关闭其他应用程序
2. 禁用 CPU 频率缩放
3. 增加样本数量
4. 增加预热时间

## 相关文件

- `bench_simd_speedup.rs`: 基准测试实现
- `run_simd_speedup_benchmark.sh`: 运行脚本
- `README_SIMD_SPEEDUP.md`: 本文档
- `../../src/simd.rs`: SIMD 实现
- `../../src/time_aware.rs`: 时间感知量化实现

## 参考

- [Criterion.rs 文档](https://bheisler.github.io/criterion.rs/book/)
- [Rust SIMD 文档](https://doc.rust-lang.org/std/simd/)
- 任务 9.5: 编写 SIMD 性能基准测试
- 需求 3.5: SIMD 性能提升
- 需求 8.1: 量化速度提升 3x-6x

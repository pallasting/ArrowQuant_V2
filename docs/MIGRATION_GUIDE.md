# ArrowQuant V2 迁移指南 (Consolidated Migration Guide)

本指南旨在帮助用户从 Legacy 实现迁移到 ArrowQuant V2 的完整优化版本。

## 目录
1. [为什么要迁移](#1-为什么要迁移)
2. [核心变更概览](#2-核心变更概览)
3. [Python API 迁移](#3-python-api-迁移)
4. [Rust API 迁移](#4-rust-api-迁移)
5. [性能调优建议](#5-性能调优建议)
6. [常见问题 (FAQ)](#6-常见问题)

---

## 1. 为什么要迁移

ArrowQuant V2 引入了三大支柱性的优化：
1. **内存效率**：基于 Apache Arrow 的零拷贝架构，内存占用减少约 **44%-50%**。
2. **计算性能**：集成 SIMD (AVX2/AVX-512/NEON) 指令集，量化速度提升 **3x-6x**。
3. **算法优化**：使用 $O(n \log m)$ 二分查找算法进行时间组分配，由于 Legacy 的 $O(n \cdot m)$。

---

## 2. 核心变更概览

| 特性 | Legacy (V1) | Arrow V2 (基础) | Arrow V2 (性能版) |
| :--- | :--- | :--- | :--- |
| **内存模式** | 基础复制 | 零拷贝 (C-FFI) | 零拷贝 + Buffer 复用 |
| **接口类型** | 字典/列表 | PyArrow Table | PyArrow Table/RecordBatch |
| **计算引擎** | Python/标量 Rust | Rust 标量 | Rust SIMD 向量化 |
| **时间组分配** | $O(n)$ 线性 | $O(n)$ 线性 | $O(n \log m)$ 二分查找 |

---

## 3. Python API 迁移

### 3.1 构造函数变更
为了保持向下兼容，`ArrowQuantV2` 构造函数现在接受 `time_aware` 作为 `diffusion` 模式的别名。

```python
# 旧代码
quantizer = ArrowQuantV2(mode="time_aware")

# 新代码 (推荐)
quantizer = ArrowQuantV2(mode="diffusion")
# (旧版构造依然有效)
```

### 3.2 批量量化接口
推荐使用 `quantize_arrow` 代替旧的字典接口，因为它提供了真正的零拷贝集成。

```python
import pyarrow as pa
import numpy as np

# 新版优化的量化方式
table = pa.table({
    "layer_name": ["layer.0"],
    "weights": [np.random.randn(1000).astype(np.float32).tolist()]
})

# 自动启用 SIMD 和 零拷贝
result_table = quantizer.quantize_arrow(table, bit_width=4)

# 零拷贝转换为 Pandas
df = result_table.to_pandas(zero_copy_only=True)
```

### 3.3 启用 SIMD 自动检测
您可以使用 `quantize_layer_auto` 来利用运行时 SIMD 检测。

```python
# 显式使用自动优化接口
result = quantizer.quantize_layer_auto(
    weights=weights,
    params=params,
    enable_simd=True
)
```

---

## 4. Rust API 迁移

### 4.1 使用 TimeAwareQuantizer
在 Rust 端，直接使用 `TimeAwareQuantizer` 的 Arrow 接口。

```rust
use arrow_quant_v2::time_aware::{TimeAwareQuantizer, SimdQuantConfig};

// 创建量化器
let quantizer = TimeAwareQuantizer::new(10);

// Arrow 零拷贝量化
let quantized = quantizer.quantize_layer_arrow(&weights, &params)?;

// 零拷贝访问底层 Buffer
let data = quantized.quantized_data(); // &[u8]
```

---

## 5. 性能调优建议

1. **规模效应**：SIMD 在元素数量大于 128 (scalar_threshold) 时优势最显著。
2. **数据类型**：确保权重数据为 `float32`，非 `float32` 类型在 Python 端转换会增加额外复制开销。
3. **输出列名**：V2 输出 Schema 包含 `scales` 和 `zero_points` 列，代替了旧版的 `scale` 和 `zero_point` 单列存。

---

## 6. 常见问题 (FAQ)

**Q: 为什么生成的表格列名变了？**
A: 为了支持更细粒度的时间组和空间量化，V2 采用列式存储 `scales` 和 `zero_points` 列表，这使得每个时间组可以拥有独立的量化参数。

**Q: 如何验证我的系统是否支持 SIMD？**
A: V2 在初始化时会自动检测。如果系统支持，性能通常会有明显提升。您可以使用 `tests/performance_validation.rs` 进行基准测试。

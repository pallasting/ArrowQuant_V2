# Arrow 零拷贝时间感知量化 - 使用指南

## 概述

Arrow 零拷贝时间感知量化是 ArrowQuant V2 的核心特性，它通过 Apache Arrow 的统一内存架构实现了高效的时间感知量化，相比传统的数据复制方案可节省 40-90% 的内存。

## 核心优势

### 1. 零拷贝架构
- 使用 Arrow RecordBatch 存储量化数据
- 避免数据复制，直接通过指针访问
- 支持与 Python/Pandas/PyArrow 的零拷贝互操作

### 2. 内存效率
- **传统方案**：每个元素存储 scale + zero_point（9 bytes/element）
- **Arrow 方案**：使用 Dictionary 编码，只存储 time_group_id（5 bytes/element）
- **内存节省**：44-90%（取决于时间组数量）

### 3. 高性能
- 量化速度：<100ms for 1M elements（release 模式）
- 反量化速度：<50ms per group
- 并行反量化：支持 Rayon 并行处理

## 快速开始

### Rust API

```rust
use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};

// 1. 创建量化器
let num_time_groups = 10;
let quantizer = TimeAwareQuantizer::new(num_time_groups);

// 2. 准备权重数据
let weights: Vec<f32> = vec![/* your weights */];

// 3. 定义时间组参数
let params: Vec<TimeGroupParams> = (0..num_time_groups)
    .map(|i| TimeGroupParams {
        scale: 0.01,
        zero_point: 128.0,
        group_size: weights.len() / num_time_groups,
        time_range: (i * group_size, (i + 1) * group_size),
    })
    .collect();

// 4. 执行 Arrow 量化
let quantized = quantizer.quantize_layer_arrow(&weights, &params)?;

// 5. 访问量化数据（零拷贝）
let quantized_data = quantized.quantized_data(); // &[u8]
let time_group_ids = quantized.time_group_ids(); // &[u32]

// 6. 反量化特定时间组
let dequantized_group_0 = quantized.dequantize_group(0)?;

// 7. 并行反量化所有组
let all_groups = quantized.dequantize_all_groups_parallel()?;
```

### Python API

```python
import arrow_quant_v2 as aq
import pyarrow as pa

# 1. 创建量化器
quantizer = aq.ArrowQuantV2()

# 2. 量化模型（返回 Arrow 格式）
result = quantizer.quantize_diffusion_model_arrow(
    model_path="path/to/model",
    output_path="path/to/output",
    bit_width=8,
    num_time_groups=10,
    use_arrow=True  # 启用 Arrow 零拷贝
)

# 3. 访问量化层（零拷贝）
for layer_name, quantized_layer in result.items():
    # 导出为 PyArrow Table（零拷贝）
    arrow_table = quantized_layer.to_pyarrow()
    
    # 转换为 Pandas（零拷贝）
    df = arrow_table.to_pandas(zero_copy_only=True)
    
    # 反量化特定时间组
    group_data = quantized_layer.dequantize_group(0)
    
    # 获取时间组参数
    params = quantized_layer.get_time_group_params()
    print(f"Layer {layer_name}: {len(quantized_layer)} elements, {len(params)} groups")
```

## 数据结构

### Arrow Schema

Arrow 零拷贝实现使用以下 schema：

```
Schema {
    quantized_data: UInt8,           // 量化后的数据 [0, 255]
    time_group_id: UInt32,           // 时间组 ID
    scale: Dictionary<UInt32, Float32>,      // Scale（字典编码）
    zero_point: Dictionary<UInt32, Float32>, // Zero point（字典编码）
    original_index: UInt64 (nullable),       // 原始索引（可选）
}
```

### 内存布局

```
┌─────────────────┬──────────────┬───────┬────────────┬────────────────┐
│ quantized_data  │ time_group_id│ scale │ zero_point │ original_index │
│ (N × 1 byte)    │ (N × 4 bytes)│ (dict)│   (dict)   │   (nullable)   │
├─────────────────┼──────────────┼───────┼────────────┼────────────────┤
│ [127, 128, ...] │ [0, 0, 1,...]│ {0:   │ {0: 128.0, │ [null, ...]    │
│                 │              │  0.01,│  1: 127.5, │                │
│                 │              │  1:   │  ...}      │                │
│                 │              │  0.01,│            │                │
│                 │              │  ...} │            │                │
└─────────────────┴──────────────┴───────┴────────────┴────────────────┘
```

## 高级用法

### 1. 自定义时间组分配

```rust
// 使用自定义时间组参数
let custom_params = vec![
    TimeGroupParams {
        scale: 0.005,      // 更精细的量化
        zero_point: 128.0,
        group_size: 50000,
        time_range: (0, 50000),
    },
    TimeGroupParams {
        scale: 0.02,       // 更粗糙的量化
        zero_point: 128.0,
        group_size: 50000,
        time_range: (50000, 100000),
    },
];

let quantized = quantizer.quantize_layer_arrow(&weights, &custom_params)?;
```

### 2. 零拷贝导出到 Python

```rust
use pyo3::prelude::*;

#[pyfunction]
fn export_to_python(quantized: ArrowQuantizedLayer) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // 零拷贝导出为 PyArrow Table
        let py_table = quantized.to_pyarrow(py)?;
        Ok(py_table.into())
    })
}
```

### 3. 并行处理多个层

```rust
use rayon::prelude::*;

let layers: Vec<Vec<f32>> = vec![/* multiple layers */];
let params_list: Vec<Vec<TimeGroupParams>> = vec![/* params for each layer */];

let quantized_layers: Vec<_> = layers
    .par_iter()
    .zip(params_list.par_iter())
    .map(|(weights, params)| {
        quantizer.quantize_layer_arrow(weights, params)
    })
    .collect::<Result<Vec<_>>>()?;
```

### 4. 流式处理大模型

```rust
// 逐层量化，避免一次性加载整个模型
for layer_file in layer_files {
    let weights = load_layer(&layer_file)?;
    let params = compute_params(&weights, num_time_groups);
    
    let quantized = quantizer.quantize_layer_arrow(&weights, &params)?;
    
    // 立即保存，释放内存
    save_quantized_layer(&quantized, &output_path)?;
}
```

## 性能优化建议

### 1. 选择合适的时间组数量

```rust
// 时间组数量影响内存和精度的平衡
// - 更多时间组 = 更高精度 + 更多内存节省
// - 更少时间组 = 更低精度 + 更少内存节省

// 推荐配置：
// - 小模型（<100M 参数）：5-10 组
// - 中等模型（100M-1B 参数）：10-20 组
// - 大模型（>1B 参数）：20-50 组
```

### 2. 使用并行反量化

```rust
// 单线程反量化
let groups: Vec<Vec<f32>> = (0..num_groups)
    .map(|i| quantized.dequantize_group(i))
    .collect::<Result<Vec<_>>>()?;

// 并行反量化（推荐）
let groups = quantized.dequantize_all_groups_parallel()?;
```

### 3. 批量处理

```python
# 批量量化多个模型
models = ["model1", "model2", "model3"]
for model_path in models:
    quantizer.quantize_diffusion_model_arrow(
        model_path=model_path,
        output_path=f"{model_path}_quantized",
        bit_width=8,
        num_time_groups=10,
        use_arrow=True
    )
```

## 与 Legacy 实现的对比

| 特性 | Legacy 实现 | Arrow 实现 |
|------|------------|-----------|
| 内存使用 | 9 bytes/element | 5 bytes/element |
| 内存节省 | 0% | 44-90% |
| 量化速度 | ~100ms/1M | ~100ms/1M |
| 反量化速度 | ~50ms/group | ~50ms/group |
| 零拷贝导出 | ❌ | ✅ |
| Python 互操作 | 需要复制 | 零拷贝 |
| 并行反量化 | ❌ | ✅ |

## 迁移指南

### 从 Legacy 迁移到 Arrow

```rust
// Legacy 实现
let quantized = quantizer.quantize_layer(&weights, &params)?;
match quantized {
    QuantizedLayer::Legacy { data, scales, zero_points, .. } => {
        // 处理 Legacy 格式
    }
    _ => {}
}

// Arrow 实现
let quantized = quantizer.quantize_layer_arrow(&weights, &params)?;
// quantized 是 ArrowQuantizedLayer 类型
let data = quantized.quantized_data();
let group_ids = quantized.time_group_ids();
```

### 统一接口（推荐）

```rust
// 使用 QuantizedLayer 枚举实现向后兼容
let quantized = quantizer.quantize_layer(&weights, &params)?;

// 统一的反量化接口
let dequantized = quantized.dequantize_group(0)?;

// 转换为 Arrow（如果需要）
if let QuantizedLayer::Legacy { .. } = quantized {
    let arrow_quantized = quantized.to_arrow()?;
    // 现在可以使用 Arrow 特性
}
```

## 故障排查

### 问题 1：内存使用仍然很高

**原因**：可能使用了 Legacy 实现而不是 Arrow 实现。

**解决方案**：
```rust
// 确保使用 quantize_layer_arrow 而不是 quantize_layer
let quantized = quantizer.quantize_layer_arrow(&weights, &params)?;
```

### 问题 2：Python 导出失败

**原因**：PyArrow 未安装或版本不兼容。

**解决方案**：
```bash
pip install pyarrow>=14.0.0
```

### 问题 3：反量化精度不足

**原因**：时间组数量太少或 scale 设置不当。

**解决方案**：
```rust
// 增加时间组数量
let num_time_groups = 20; // 从 10 增加到 20

// 或调整 scale 参数
let params = compute_optimal_params(&weights, num_time_groups);
```

## 最佳实践

1. **始终使用 Arrow 实现**：除非有特殊原因，否则优先使用 `quantize_layer_arrow`
2. **合理选择时间组数量**：根据模型大小和精度要求选择
3. **使用并行反量化**：充分利用多核 CPU
4. **零拷贝导出**：在 Python 中使用 `to_pyarrow()` 避免数据复制
5. **流式处理**：对于大模型，逐层处理避免内存溢出

## 参考资料

- [Arrow 零拷贝设计文档](../ARROW_ZERO_COPY_DESIGN.md)
- [API 文档](./api_documentation.md)
- [性能基准测试](../tests/performance_validation.rs)
- [Python 集成示例](../tests/test_py_arrow_quantized_layer.py)

## 支持

如有问题或建议，请提交 Issue 到 GitHub 仓库。

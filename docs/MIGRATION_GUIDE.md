# Arrow 零拷贝迁移指南

从 Legacy 实现迁移到 Arrow 零拷贝实现的完整指南。

## 目录

- [为什么要迁移](#为什么要迁移)
- [迁移前准备](#迁移前准备)
- [Rust 代码迁移](#rust-代码迁移)
- [Python 代码迁移](#python-代码迁移)
- [性能对比](#性能对比)
- [常见问题](#常见问题)
- [迁移检查清单](#迁移检查清单)

---

## 为什么要迁移

### Arrow 实现的优势

1. **内存效率**：节省 44-90% 内存
2. **零拷贝**：与 Python/Pandas/PyArrow 无缝集成
3. **并行处理**：支持 Rayon 并行反量化
4. **标准化**：基于 Apache Arrow 标准格式
5. **未来兼容**：后续功能将优先支持 Arrow

### 性能对比

| 指标 | Legacy | Arrow | 改进 |
|------|--------|-------|------|
| 内存使用 | 9 MB/1M | 5 MB/1M | -44% |
| 量化速度 | 100ms | 100ms | 持平 |
| 反量化速度 | 50ms | 50ms | 持平 |
| Python 导出 | 需复制 | 零拷贝 | 10x+ |
| 并行支持 | ❌ | ✅ | 新增 |

---

## 迁移前准备

### 1. 检查依赖版本

```toml
# Cargo.toml
[dependencies]
arrow = "53.0"
pyo3 = "0.22"
```

```bash
# Python
pip install pyarrow>=14.0.0
```

### 2. 备份现有代码

```bash
# 创建备份分支
git checkout -b legacy-backup

# 或创建备份目录
cp -r src src_legacy_backup
```

### 3. 运行现有测试

```bash
# 确保所有测试通过
cargo test --lib
python -m pytest tests/
```

---

## Rust 代码迁移

### 步骤 1：更新量化调用

#### Before (Legacy)

```rust
use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};

let quantizer = TimeAwareQuantizer::new(10);
let weights = vec![1.0, 2.0, 3.0, 4.0];
let params = vec![/* ... */];

// Legacy 实现
let quantized = quantizer.quantize_layer(&weights, &params)?;

// 访问数据需要模式匹配
match quantized {
    QuantizedLayer::Legacy { data, scales, zero_points, .. } => {
        // 处理 Legacy 格式
        println!("Data: {:?}", data);
    }
    _ => {}
}
```

#### After (Arrow)

```rust
use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};

let quantizer = TimeAwareQuantizer::new(10);
let weights = vec![1.0, 2.0, 3.0, 4.0];
let params = vec![/* ... */];

// Arrow 实现
let quantized = quantizer.quantize_layer_arrow(&weights, &params)?;

// 零拷贝访问
let data = quantized.quantized_data();
let group_ids = quantized.time_group_ids();
println!("Data: {:?}", data);
```

### 步骤 2：更新反量化调用

#### Before (Legacy)

```rust
// Legacy 反量化
let dequantized = match &quantized {
    QuantizedLayer::Legacy { data, scales, zero_points, time_group_params } => {
        // 手动反量化逻辑
        let mut result = Vec::new();
        for (i, &q) in data.iter().enumerate() {
            let group_id = i / time_group_params[0].group_size;
            let scale = scales[group_id];
            let zero_point = zero_points[group_id];
            result.push((q as f32 - zero_point) * scale);
        }
        result
    }
    _ => vec![]
};
```

#### After (Arrow)

```rust
// Arrow 反量化（单组）
let dequantized = quantized.dequantize_group(0)?;

// 或并行反量化所有组
let all_groups = quantized.dequantize_all_groups_parallel()?;
```

### 步骤 3：更新数据访问

#### Before (Legacy)

```rust
// 需要模式匹配和复制
let (data, scales, zero_points) = match quantized {
    QuantizedLayer::Legacy { data, scales, zero_points, .. } => {
        (data.clone(), scales.clone(), zero_points.clone())
    }
    _ => (vec![], vec![], vec![])
};
```

#### After (Arrow)

```rust
// 零拷贝引用
let data = quantized.quantized_data(); // &[u8]
let group_ids = quantized.time_group_ids(); // &[u32]

// 无需复制，直接使用
for (i, &q) in data.iter().enumerate() {
    let group_id = group_ids[i];
    // 处理数据
}
```

### 步骤 4：更新并行处理

#### Before (Legacy)

```rust
// Legacy 不支持并行反量化
let mut all_groups = Vec::new();
for group_id in 0..num_groups {
    let group_data = dequantize_group_legacy(&quantized, group_id)?;
    all_groups.push(group_data);
}
```

#### After (Arrow)

```rust
// Arrow 支持并行反量化
let all_groups = quantized.dequantize_all_groups_parallel()?;
```

---

## Python 代码迁移

### 步骤 1：更新量化调用

#### Before (Legacy)

```python
import arrow_quant_v2 as aq

quantizer = aq.ArrowQuantV2()

# Legacy 实现
result = quantizer.quantize_diffusion_model(
    model_path="models/stable_diffusion",
    output_path="models/stable_diffusion_int8",
    bit_width=8,
    num_time_groups=10,
    use_arrow=False  # Legacy
)

# 结果是字典，需要手动处理
for layer_name, layer_data in result.items():
    # layer_data 是普通字典
    data = layer_data['data']
    scales = layer_data['scales']
    # ...
```

#### After (Arrow)

```python
import arrow_quant_v2 as aq

quantizer = aq.ArrowQuantV2()

# Arrow 实现
layers = quantizer.quantize_diffusion_model_arrow(
    model_path="models/stable_diffusion",
    output_path="models/stable_diffusion_int8",
    bit_width=8,
    num_time_groups=10
)

# 结果是 PyArrowQuantizedLayer 对象
for layer_name, quantized_layer in layers.items():
    # 零拷贝导出
    table = quantized_layer.to_pyarrow()
    df = table.to_pandas(zero_copy_only=True)
```

### 步骤 2：更新数据访问

#### Before (Legacy)

```python
# Legacy 需要复制数据
data = result['layer1']['data'].copy()
scales = result['layer1']['scales'].copy()
zero_points = result['layer1']['zero_points'].copy()

# 手动反量化
dequantized = []
for i, q in enumerate(data):
    group_id = i // group_size
    scale = scales[group_id]
    zero_point = zero_points[group_id]
    dequantized.append((q - zero_point) * scale)
```

#### After (Arrow)

```python
# Arrow 零拷贝访问
quantized_layer = layers['layer1']

# 直接反量化
dequantized = quantized_layer.dequantize_group(0)

# 或并行反量化所有组
all_groups = quantized_layer.dequantize_all_groups()
```

### 步骤 3：更新 Pandas 集成

#### Before (Legacy)

```python
import pandas as pd

# Legacy 需要手动构建 DataFrame
data = result['layer1']['data']
df = pd.DataFrame({
    'quantized_data': data,
    'scale': [scales[i // group_size] for i in range(len(data))],
    'zero_point': [zero_points[i // group_size] for i in range(len(data))]
})
```

#### After (Arrow)

```python
import pandas as pd
import pyarrow as pa

# Arrow 零拷贝转换
quantized_layer = layers['layer1']
table = quantized_layer.to_pyarrow()

# 零拷贝转换为 Pandas
df = table.to_pandas(zero_copy_only=True)

# 或保存为 Parquet
pa.parquet.write_table(table, "layer1.parquet")
```

---

## 渐进式迁移策略

### 策略 1：统一接口（推荐）

使用 `QuantizedLayer` 枚举保持向后兼容：

```rust
// 支持两种格式
fn process_layer(quantized: QuantizedLayer) -> Result<Vec<f32>> {
    // 统一的反量化接口
    quantized.dequantize_group(0)
}

// 可以传入 Legacy 或 Arrow 格式
let legacy = quantizer.quantize_layer(&weights, &params)?;
let arrow = quantizer.quantize_layer_arrow(&weights, &params)?;

process_layer(legacy)?;
process_layer(QuantizedLayer::Arrow(arrow))?;
```

### 策略 2：逐层迁移

```rust
// 先迁移关键层
let critical_layers = vec!["attention", "mlp"];

for layer_name in &all_layers {
    let quantized = if critical_layers.contains(&layer_name.as_str()) {
        // 使用 Arrow
        QuantizedLayer::Arrow(
            quantizer.quantize_layer_arrow(&weights, &params)?
        )
    } else {
        // 保持 Legacy
        quantizer.quantize_layer(&weights, &params)?
    };
    
    save_layer(layer_name, quantized)?;
}
```

### 策略 3：特性开关

```rust
// 使用配置控制
struct QuantConfig {
    use_arrow: bool,
    // ...
}

fn quantize_with_config(
    weights: &[f32],
    params: &[TimeGroupParams],
    config: &QuantConfig,
) -> Result<QuantizedLayer> {
    if config.use_arrow {
        Ok(QuantizedLayer::Arrow(
            quantizer.quantize_layer_arrow(weights, params)?
        ))
    } else {
        quantizer.quantize_layer(weights, params)
    }
}
```

---

## 性能对比测试

### 测试脚本

```rust
use std::time::Instant;

fn benchmark_legacy_vs_arrow() {
    let weights: Vec<f32> = (0..1_000_000).map(|i| i as f32 * 0.001).collect();
    let params = create_params(10, weights.len());
    
    // Legacy
    let start = Instant::now();
    let legacy = quantizer.quantize_layer(&weights, &params).unwrap();
    let legacy_time = start.elapsed();
    
    // Arrow
    let start = Instant::now();
    let arrow = quantizer.quantize_layer_arrow(&weights, &params).unwrap();
    let arrow_time = start.elapsed();
    
    println!("Legacy: {:?}", legacy_time);
    println!("Arrow: {:?}", arrow_time);
    println!("Speedup: {:.2}x", legacy_time.as_secs_f64() / arrow_time.as_secs_f64());
}
```

### 预期结果

```
Legacy: 102ms
Arrow: 98ms
Speedup: 1.04x

Memory (Legacy): 9 MB
Memory (Arrow): 5 MB
Savings: 44%
```

---

## 常见问题

### Q1: 迁移后性能反而下降？

**A**: 检查以下几点：
1. 确保使用 release 模式编译：`cargo build --release`
2. 检查是否启用了 Arrow 特性：`features = ["arrow"]`
3. 验证是否真的使用了 Arrow 实现而不是 Legacy

### Q2: Python 导出失败？

**A**: 确保 PyArrow 版本正确：
```bash
pip install --upgrade pyarrow>=14.0.0
```

### Q3: 内存使用没有减少？

**A**: 检查是否使用了 `quantize_layer_arrow` 而不是 `quantize_layer`：
```rust
// 错误：使用了 Legacy
let quantized = quantizer.quantize_layer(&weights, &params)?;

// 正确：使用 Arrow
let quantized = quantizer.quantize_layer_arrow(&weights, &params)?;
```

### Q4: 如何验证迁移成功？

**A**: 运行以下测试：
```bash
# Rust 测试
cargo test --lib test_arrow

# Python 测试
python -m pytest tests/test_arrow_integration.py

# 性能测试
cargo test --test performance_validation --release
```

---

## 迁移检查清单

### 代码迁移

- [ ] 更新所有 `quantize_layer` 调用为 `quantize_layer_arrow`
- [ ] 更新所有数据访问代码使用零拷贝引用
- [ ] 更新反量化代码使用新 API
- [ ] 移除不必要的数据复制
- [ ] 更新并行处理代码

### 测试验证

- [ ] 所有单元测试通过
- [ ] 所有集成测试通过
- [ ] 性能测试达标
- [ ] 内存使用减少 >40%
- [ ] Python 集成测试通过

### 文档更新

- [ ] 更新 API 文档
- [ ] 更新使用示例
- [ ] 更新性能指标
- [ ] 添加迁移说明

### 部署准备

- [ ] 更新依赖版本
- [ ] 更新 CI/CD 配置
- [ ] 准备回滚方案
- [ ] 通知团队成员

---

## 回滚方案

如果迁移后出现问题，可以快速回滚：

### 方案 1：使用统一接口

```rust
// 在配置中切换
config.use_arrow = false; // 回滚到 Legacy
```

### 方案 2：Git 回滚

```bash
# 回滚到迁移前的提交
git revert <migration-commit>

# 或切换到备份分支
git checkout legacy-backup
```

### 方案 3：条件编译

```rust
#[cfg(feature = "use-arrow")]
let quantized = quantizer.quantize_layer_arrow(&weights, &params)?;

#[cfg(not(feature = "use-arrow"))]
let quantized = quantizer.quantize_layer(&weights, &params)?;
```

---

## 获取帮助

如果在迁移过程中遇到问题：

1. 查看 [API 文档](./api_documentation.md)
2. 查看 [使用指南](./arrow_zero_copy_guide.md)
3. 查看 [示例代码](../tests/test_arrow_integration.py)
4. 提交 Issue 到 GitHub

---

## 总结

Arrow 零拷贝实现提供了显著的内存和性能优势，迁移过程相对简单。建议采用渐进式迁移策略，先迁移关键层，验证效果后再全面迁移。

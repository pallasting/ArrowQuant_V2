# API 选择指南

快速决策：选择最适合你场景的量化 API

---

## 快速决策树

```
开始
  │
  ├─ 需要最佳性能？
  │   └─ 是 → 使用 Arrow IPC API ⭐
  │
  ├─ 模型层数 > 50？
  │   └─ 是 → 使用 Arrow IPC API ⭐
  │
  ├─ 内存受限？
  │   └─ 是 → 使用 Arrow IPC API ⭐
  │
  ├─ 需要简单易用？
  │   └─ 是 → 使用 Batch API ✓
  │
  └─ 其他 → 使用 Batch API ✓
```

---

## API 对比表

| 特性 | Arrow IPC | Batch API | Legacy API |
|------|-----------|-----------|------------|
| **性能** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| **易用性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **内存效率** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **零拷贝** | ✅ 是 | ❌ 否 | ❌ 否 |
| **并行处理** | ✅ 是 | ✅ 是 | ❌ 否 |
| **推荐使用** | 生产环境 | 开发/原型 | 不推荐 |

---

## 详细对比

### 1. Arrow IPC API ⭐ (推荐生产环境)

**性能数据**:
- 4MB 层: ~5ms (29x 加速)
- 100 层: ~500ms (37x 加速)
- 内存开销: 1x (零拷贝)

**适用场景**:
- ✅ 大模型量化（>50 层）
- ✅ 内存受限环境
- ✅ 性能关键应用
- ✅ 批量处理大数据集
- ✅ 生产部署

**优势**:
- 🚀 最佳性能（30x 加速）
- 💾 最低内存开销（零拷贝）
- 🔄 支持流式处理
- 🌐 跨语言互操作性

**劣势**:
- 📚 需要了解 PyArrow
- 🔧 数据格式转换

**使用示例**:
```python
import numpy as np
import pyarrow as pa
from arrow_quant_v2 import ArrowQuantV2

# 1. 准备数据
weights_data = {
    "layer_name": ["layer.0.weight", "layer.1.weight"],
    "weights": [
        np.random.randn(1000000).astype(np.float32).tolist(),
        np.random.randn(1000000).astype(np.float32).tolist(),
    ],
    "shape": [[1000000], [1000000]],
}

# 2. 创建 Arrow Table
table = pa.Table.from_pydict(weights_data)

# 3. 零拷贝量化
quantizer = ArrowQuantV2(mode="diffusion")
result_table = quantizer.quantize_arrow(table, bit_width=4)

# 4. 访问结果
result_dict = result_table.to_pydict()
for i, layer_name in enumerate(result_dict["layer_name"]):
    quantized_data = result_dict["quantized_data"][i]
    scales = result_dict["scales"][i]
    print(f"{layer_name}: {len(quantized_data)} bytes")
```

---

### 2. Batch API ✓ (推荐开发/原型)

**性能数据**:
- 4MB 层: ~50ms (3x 加速)
- 100 层: ~2,000ms (9x 加速)
- 内存开销: 2x (数据复制)

**适用场景**:
- ✅ 小模型量化（<10 层）
- ✅ 快速原型开发
- ✅ 便捷性优先于性能
- ✅ 内存充足环境
- ✅ 开发和测试

**优势**:
- 😊 简单易用
- 🔄 并行处理
- 📉 减少边界跨越（100x）
- 🐍 原生 numpy 支持

**劣势**:
- 💾 数据复制（内存 2x）
- ⚡ 性能次于 Arrow IPC

**使用示例**:
```python
import numpy as np
from arrow_quant_v2 import ArrowQuantV2

# 1. 准备数据（直接使用 numpy）
weights = {
    "layer.0.weight": np.random.randn(1000, 1000).astype(np.float32),
    "layer.1.weight": np.random.randn(1000, 1000).astype(np.float32),
}

# 2. 批量量化
quantizer = ArrowQuantV2(mode="diffusion")
results = quantizer.quantize_batch(weights, bit_width=4)

# 3. 访问结果
for layer_name, result in results.items():
    quantized_data = result["quantized_data"]
    scales = result["scales"]
    print(f"{layer_name}: {len(quantized_data)} bytes")
```

**带进度回调**:
```python
def progress_callback(layer_name: str, progress: float):
    print(f"Processing {layer_name}: {progress*100:.1f}%")

results = quantizer.quantize_batch_with_progress(
    weights,
    bit_width=4,
    progress_callback=progress_callback
)
```

---

### 3. Legacy API ⚠️ (不推荐)

**性能数据**:
- 4MB 层: ~147ms (基准)
- 100 层: ~18,369ms (基准)
- 内存开销: 2x (数据转换)

**适用场景**:
- ⚠️ 向后兼容
- ⚠️ 单层量化
- ⚠️ 测试和调试

**问题**:
- ❌ 数据转换开销大（69%）
- ❌ 边界跨越频繁
- ❌ 性能最差

**使用示例**:
```python
import numpy as np
from arrow_quant_v2 import ArrowQuantV2

# 不推荐：需要转换为 list
weights = np.random.randn(1000, 1000).astype(np.float32)
weights_list = weights.tolist()  # 慢！

quantizer = ArrowQuantV2(mode="diffusion")
result = quantizer.quantize(
    {"layer.0.weight": weights_list},
    bit_width=4
)
```

---

## 性能对比图表

### 单层量化（4MB）

```
Legacy API    ████████████████████████████████ 147ms
Batch API     ██████████ 50ms (3x faster)
Arrow IPC     █ 5ms (29x faster) ⭐
```

### 多层量化（100 层，400MB）

```
Legacy API    ████████████████████████████████ 18,369ms
Batch API     ███ 2,000ms (9x faster)
Arrow IPC     █ 500ms (37x faster) ⭐
```

### 内存开销

```
Legacy API    ████████ 2x (转换)
Batch API     ████████ 2x (复制)
Arrow IPC     ████ 1x (零拷贝) ⭐
```

---

## 常见问题

### Q1: 我应该使用哪个 API？

**A**: 
- 生产环境 → Arrow IPC（最佳性能）
- 开发/原型 → Batch API（简单易用）
- 避免使用 Legacy API

### Q2: Arrow IPC 真的是零拷贝吗？

**A**: 是的！Arrow IPC 使用 Arrow C Data Interface 标准，通过指针传递数据，Rust 直接引用 Python 内存，无需复制。

### Q3: Batch API 为什么要复制数据？

**A**: 为了支持并行处理（rayon），必须释放 Python GIL。这要求拥有数据所有权，因此需要复制。这是性能权衡：复制开销（~50ms）换取并行加速和边界跨越减少（18s → 2ms）。

### Q4: 如何从 numpy 转换到 Arrow？

**A**: 使用 PyArrow 的 `from_pydict()`:
```python
import pyarrow as pa

table = pa.Table.from_pydict({
    "layer_name": ["layer.0"],
    "weights": [numpy_array.tolist()],
    "shape": [[numpy_array.shape]],
})
```

### Q5: Arrow IPC 支持哪些数据类型？

**A**: 
- 必需: `layer_name` (string), `weights` (list<float32>)
- 可选: `shape` (list<int64>)

### Q6: 可以混合使用不同 API 吗？

**A**: 可以，但不推荐。选择一个 API 并保持一致性。

---

## 迁移指南

### 从 Legacy API 迁移到 Batch API

**之前**:
```python
for layer_name, weights in weights_dict.items():
    result = quantizer.quantize(
        {layer_name: weights.tolist()},
        bit_width=4
    )
```

**之后**:
```python
# 单次调用处理所有层
results = quantizer.quantize_batch(weights_dict, bit_width=4)
```

**改进**: 9x 加速（100 层）

### 从 Batch API 迁移到 Arrow IPC

**之前**:
```python
weights = {
    "layer.0": np.array(...),
    "layer.1": np.array(...),
}
results = quantizer.quantize_batch(weights, bit_width=4)
```

**之后**:
```python
import pyarrow as pa

# 转换为 Arrow Table
table = pa.Table.from_pydict({
    "layer_name": list(weights.keys()),
    "weights": [w.tolist() for w in weights.values()],
})

# 零拷贝量化
result_table = quantizer.quantize_arrow(table, bit_width=4)
```

**改进**: 4x 加速（100 层），50% 内存节省

---

## 性能调优建议

### Arrow IPC 优化

1. **批量处理**: 一次处理多层
2. **预分配**: 使用 Arrow 的 builder 模式
3. **流式处理**: 对于超大模型使用 RecordBatch

### Batch API 优化

1. **控制并行度**: 设置 `RAYON_NUM_THREADS`
2. **内存预算**: 确保有足够内存（2x 数据量）
3. **错误处理**: 使用 `continue_on_error=True`

---

## 总结

| 场景 | 推荐 API | 原因 |
|------|---------|------|
| 生产部署 | Arrow IPC ⭐ | 最佳性能 + 最低内存 |
| 大模型（>50 层） | Arrow IPC ⭐ | 37x 加速 |
| 内存受限 | Arrow IPC ⭐ | 零拷贝 |
| 快速开发 | Batch API ✓ | 简单易用 |
| 小模型（<10 层） | Batch API ✓ | 足够快 |
| 向后兼容 | Legacy API ⚠️ | 仅用于兼容 |

**默认推荐**: 生产环境使用 **Arrow IPC**，开发环境使用 **Batch API**。

---

**文档版本**: 1.0  
**最后更新**: 2026-02-26  
**相关文档**: 
- `PERFORMANCE_BENCHMARK_REPORT.md` - 详细性能分析
- `ZERO_COPY_IMPLEMENTATION_REVIEW.md` - 代码审查
- `docs/ARROW_FFI_INTEGRATION.md` - Arrow FFI 技术细节

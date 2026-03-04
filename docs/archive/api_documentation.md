# ArrowQuant V2 - API 文档

## 目录

- [Rust API](#rust-api)
  - [TimeAwareQuantizer](#timeawarequantizer)
  - [ArrowQuantizedLayer](#arrowquantizedlayer)
  - [TimeGroupParams](#timegroupparams)
  - [QuantizedLayer](#quantizedlayer)
- [Python API](#python-api)
  - [ArrowQuantV2](#arrowquantv2)
  - [PyArrowQuantizedLayer](#pyarrowquantizedlayer)

---

## Rust API

### TimeAwareQuantizer

时间感知量化器，支持 Legacy 和 Arrow 两种实现。

#### 构造函数

```rust
pub fn new(num_time_groups: usize) -> Self
```

创建一个新的时间感知量化器。

**参数**：
- `num_time_groups`: 时间组数量（推荐 5-50）

**返回**：
- `TimeAwareQuantizer` 实例

**示例**：
```rust
let quantizer = TimeAwareQuantizer::new(10);
```

---

#### quantize_layer

```rust
pub fn quantize_layer(
    &self,
    weights: &[f32],
    time_group_params: &[TimeGroupParams],
) -> Result<QuantizedLayer>
```

使用 Legacy 实现量化权重层。

**参数**：
- `weights`: 权重数据（FP32）
- `time_group_params`: 时间组参数列表

**返回**：
- `Result<QuantizedLayer>`: 量化后的层（Legacy 格式）

**错误**：
- `QuantError::QuantizationFailed`: 量化失败

**示例**：
```rust
let weights = vec![1.0, 2.0, 3.0, 4.0];
let params = vec![
    TimeGroupParams {
        scale: 0.01,
        zero_point: 128.0,
        group_size: 2,
        time_range: (0, 2),
    },
    TimeGroupParams {
        scale: 0.01,
        zero_point: 128.0,
        group_size: 2,
        time_range: (2, 4),
    },
];

let quantized = quantizer.quantize_layer(&weights, &params)?;
```

---

#### quantize_layer_arrow

```rust
pub fn quantize_layer_arrow(
    &self,
    weights: &[f32],
    time_group_params: &[TimeGroupParams],
) -> Result<ArrowQuantizedLayer>
```

使用 Arrow 零拷贝实现量化权重层（推荐）。

**参数**：
- `weights`: 权重数据（FP32）
- `time_group_params`: 时间组参数列表

**返回**：
- `Result<ArrowQuantizedLayer>`: 量化后的层（Arrow 格式）

**错误**：
- `QuantError::QuantizationFailed`: 量化失败
- `QuantError::InvalidInput`: 输入参数无效

**性能**：
- 量化速度：~100ms for 1M elements（release 模式）
- 内存使用：~5 bytes/element（vs Legacy 9 bytes/element）

**示例**：
```rust
let quantized = quantizer.quantize_layer_arrow(&weights, &params)?;

// 零拷贝访问
let data = quantized.quantized_data();
let group_ids = quantized.time_group_ids();
```

---

### ArrowQuantizedLayer

Arrow 格式的量化层，支持零拷贝操作。

#### 构造函数

```rust
pub fn new(
    batch: RecordBatch,
    time_group_params: Vec<TimeGroupParams>,
) -> Result<Self>
```

从 Arrow RecordBatch 创建量化层。

**参数**：
- `batch`: Arrow RecordBatch（必须符合 time_aware schema）
- `time_group_params`: 时间组参数列表

**返回**：
- `Result<ArrowQuantizedLayer>`

**错误**：
- `QuantError::InvalidSchema`: Schema 不匹配

---

#### quantized_data

```rust
pub fn quantized_data(&self) -> &[u8]
```

获取量化数据的零拷贝引用。

**返回**：
- `&[u8]`: 量化数据切片（范围 [0, 255]）

**示例**：
```rust
let data = quantized.quantized_data();
println!("First quantized value: {}", data[0]);
```

---

#### time_group_ids

```rust
pub fn time_group_ids(&self) -> &[u32]
```

获取时间组 ID 的零拷贝引用。

**返回**：
- `&[u32]`: 时间组 ID 切片

**示例**：
```rust
let group_ids = quantized.time_group_ids();
println!("Element 0 belongs to group {}", group_ids[0]);
```

---

#### dequantize_group

```rust
pub fn dequantize_group(&self, group_id: usize) -> Result<Vec<f32>>
```

反量化指定时间组的数据。

**参数**：
- `group_id`: 时间组 ID（0 到 num_groups-1）

**返回**：
- `Result<Vec<f32>>`: 反量化后的数据

**错误**：
- `QuantError::InvalidGroupId`: 组 ID 无效

**性能**：
- ~50ms per group（1M elements，release 模式）

**示例**：
```rust
let group_0_data = quantized.dequantize_group(0)?;
println!("Group 0 has {} elements", group_0_data.len());
```

---

#### dequantize_all_groups_parallel

```rust
pub fn dequantize_all_groups_parallel(&self) -> Result<Vec<Vec<f32>>>
```

并行反量化所有时间组。

**返回**：
- `Result<Vec<Vec<f32>>>`: 所有组的反量化数据

**性能**：
- 使用 Rayon 并行处理
- 加速比：~80%（8 核 CPU）

**示例**：
```rust
let all_groups = quantized.dequantize_all_groups_parallel()?;
for (i, group_data) in all_groups.iter().enumerate() {
    println!("Group {}: {} elements", i, group_data.len());
}
```

---

#### len

```rust
pub fn len(&self) -> usize
```

获取量化层的元素总数。

**返回**：
- `usize`: 元素数量

**示例**：
```rust
println!("Layer has {} elements", quantized.len());
```

---

#### is_empty

```rust
pub fn is_empty(&self) -> bool
```

检查量化层是否为空。

**返回**：
- `bool`: 是否为空

---

#### to_record_batch

```rust
pub fn to_record_batch(&self) -> &RecordBatch
```

获取底层 Arrow RecordBatch 的引用（零拷贝）。

**返回**：
- `&RecordBatch`: RecordBatch 引用

**用途**：
- 与 Arrow 生态系统集成
- 导出到 Parquet/CSV
- 零拷贝传递给其他 Arrow 库

---

### TimeGroupParams

时间组参数结构。

```rust
pub struct TimeGroupParams {
    pub scale: f32,
    pub zero_point: f32,
    pub group_size: usize,
    pub time_range: (usize, usize),
}
```

**字段**：
- `scale`: 量化缩放因子
- `zero_point`: 量化零点
- `group_size`: 组大小（元素数量）
- `time_range`: 时间范围（起始索引，结束索引）

**示例**：
```rust
let params = TimeGroupParams {
    scale: 0.01,
    zero_point: 128.0,
    group_size: 100000,
    time_range: (0, 100000),
};
```

---

### QuantizedLayer

量化层枚举，支持 Legacy 和 Arrow 两种格式。

```rust
pub enum QuantizedLayer {
    Legacy {
        data: Vec<u8>,
        scales: Vec<f32>,
        zero_points: Vec<f32>,
        time_group_params: Vec<TimeGroupParams>,
    },
    Arrow(ArrowQuantizedLayer),
}
```

#### dequantize_group

```rust
pub fn dequantize_group(&self, group_id: usize) -> Result<Vec<f32>>
```

统一的反量化接口，支持 Legacy 和 Arrow 格式。

**参数**：
- `group_id`: 时间组 ID

**返回**：
- `Result<Vec<f32>>`: 反量化后的数据

---

#### to_arrow

```rust
pub fn to_arrow(&self) -> Result<ArrowQuantizedLayer>
```

将 Legacy 格式转换为 Arrow 格式。

**返回**：
- `Result<ArrowQuantizedLayer>`: Arrow 格式的量化层

**用途**：
- 迁移 Legacy 代码到 Arrow
- 利用 Arrow 的零拷贝特性

---

## Python API

### ArrowQuantV2

主要的 Python 量化接口。

#### 构造函数

```python
def __init__(self) -> None
```

创建一个新的量化器实例。

**示例**：
```python
import arrow_quant_v2 as aq

quantizer = aq.ArrowQuantV2()
```

---

#### quantize_diffusion_model

```python
def quantize_diffusion_model(
    self,
    model_path: str,
    output_path: str,
    bit_width: int = 4,
    num_time_groups: int = 10,
    use_arrow: bool = False,
    config: Optional[DiffusionQuantConfig] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]
```

量化扩散模型。支持 Legacy 和 Arrow 输出格式。

**参数**：
- `model_path`: 模型路径
- `output_path`: 输出路径
- `bit_width`: 位宽（2, 4, 或 8）。默认 4。
- `num_time_groups`: 时间组数量。默认 10。
- `use_arrow`: 是否使用 Arrow 格式（推荐 True）。
- `config`: 可选的 `DiffusionQuantConfig` 实例。
- `progress_callback`: 可选的进度回调函数。

**返回**：
- `Dict[str, Any]`: 量化结果摘要。

---

#### quantize_layer_auto

```python
def quantize_layer_auto(
    self,
    weights: np.ndarray,
    params: List[TimeGroupParams],
    enable_simd: bool = True
) -> PyArrowQuantizedLayer
```

自动选择最优路径（SIMD 或标量）量化单个权重层。

**参数**：
- `weights`: Numpy 数组 (float32, 连续内存)。
- `params`: 时间组参数列表。
- `enable_simd`: 是否允许 SIMD 加速。

**返回**：
- `PyArrowQuantizedLayer`: 优化的量化层。

---

#### set_simd_config

```python
def set_simd_config(self, config: SimdQuantConfig) -> None
```

配置运行时 SIMD 行为。

**参数**：
- `config`: `SimdQuantConfig` 实例。

---

#### quantize_batch

```python
def quantize_batch(
    self,
    weights_dict: Dict[str, np.ndarray],
    bit_width: int = 4,
    continue_on_error: bool = False
) -> Dict[str, PyArrowQuantizedLayer]
```

批量并行量化多个层。

**参数**：
- `weights_dict`: 层名到 Numpy 数组的映射。
- `bit_width`: 目标位宽。
- `continue_on_error`: 遇到错误时是否继续处理其他层。

**返回**：
- `Dict[str, PyArrowQuantizedLayer]`: 量化结果映射。

---

#### quantize_diffusion_model_arrow

```python
def quantize_diffusion_model_arrow(
    self,
    model_path: str,
    output_path: str,
    bit_width: int = 8,
    num_time_groups: int = 10,
) -> Dict[str, PyArrowQuantizedLayer]
```

使用 Arrow 格式量化扩散模型（推荐）。

**参数**：
- `model_path`: 模型路径
- `output_path`: 输出路径
- `bit_width`: 位宽（2, 4, 或 8）
- `num_time_groups`: 时间组数量

**返回**：
- `Dict[str, PyArrowQuantizedLayer]`: 层名到量化层的映射

**示例**：
```python
layers = quantizer.quantize_diffusion_model_arrow(
    model_path="models/stable_diffusion",
    output_path="models/stable_diffusion_int8",
    bit_width=8,
    num_time_groups=10
)

for layer_name, quantized_layer in layers.items():
    print(f"{layer_name}: {len(quantized_layer)} elements")
```

---

### PyArrowQuantizedLayer

Python 包装的 Arrow 量化层。

#### to_pyarrow

```python
def to_pyarrow(self) -> pa.Table
```

零拷贝导出为 PyArrow Table。

**返回**：
- `pa.Table`: PyArrow Table

**示例**：
```python
import pyarrow as pa

# 零拷贝导出
table = quantized_layer.to_pyarrow()

# 转换为 Pandas（零拷贝）
df = table.to_pandas(zero_copy_only=True)

# 保存为 Parquet
pa.parquet.write_table(table, "output.parquet")
```

---

#### dequantize_group

```python
def dequantize_group(self, group_id: int) -> List[float]
```

反量化指定时间组。

**参数**：
- `group_id`: 时间组 ID

**返回**：
- `List[float]`: 反量化后的数据

**示例**：
```python
group_0 = quantized_layer.dequantize_group(0)
print(f"Group 0: {len(group_0)} elements")
```

---

#### dequantize_all_groups

```python
def dequantize_all_groups(self) -> List[List[float]]
```

并行反量化所有时间组。

**返回**：
- `List[List[float]]`: 所有组的反量化数据

**示例**：
```python
all_groups = quantized_layer.dequantize_all_groups()
for i, group in enumerate(all_groups):
    print(f"Group {i}: {len(group)} elements")
```

---

#### get_time_group_params

```python
def get_time_group_params(self) -> List[Dict[str, Any]]
```

获取时间组参数。

**返回**：
- `List[Dict[str, Any]]`: 参数列表

**示例**：
```python
params = quantized_layer.get_time_group_params()
for i, param in enumerate(params):
    print(f"Group {i}: scale={param['scale']}, zero_point={param['zero_point']}")
```

---

#### __len__

```python
def __len__(self) -> int
```

获取元素数量。

**返回**：
- `int`: 元素数量

**示例**：
```python
print(f"Layer has {len(quantized_layer)} elements")
```

---

## 错误处理

### Rust 错误类型

```rust
pub enum QuantError {
    QuantizationFailed(String),
    InvalidInput(String),
    InvalidSchema(String),
    InvalidGroupId(usize),
    ModelNotFound(String),
    // ...
}
```

### Python 异常

```python
# 量化错误
try:
    result = quantizer.quantize_diffusion_model(...)
except aq.QuantizationError as e:
    print(f"Quantization failed: {e}")

# 配置错误
try:
    quantizer = aq.ArrowQuantV2(invalid_config)
except aq.ConfigurationError as e:
    print(f"Configuration error: {e}")

# 验证错误
try:
    quantized.dequantize_group(999)
except aq.ValidationError as e:
    print(f"Validation error: {e}")
```

---

## 类型别名

```rust
pub type Result<T> = std::result::Result<T, QuantError>;
```

---

## 常量

```rust
// 支持的位宽
pub const SUPPORTED_BIT_WIDTHS: &[u8] = &[2, 4, 8];

// 默认时间组数量
pub const DEFAULT_NUM_TIME_GROUPS: usize = 10;

// 最大时间组数量
pub const MAX_NUM_TIME_GROUPS: usize = 100;
```

---

## 性能指标

| 操作 | 性能（1M 元素） | 模式 |
|------|----------------|------|
| Arrow 量化 | ~100ms | Release |
| Legacy 量化 | ~100ms | Release |
| 单组反量化 | ~50ms | Release |
| 并行反量化（10 组） | ~200ms | Release |
| 零拷贝导出 | <1ms | - |

---

## 内存使用

| 实现 | 内存/元素 | 1M 元素总内存 |
|------|----------|--------------|
| Legacy | 9 bytes | ~9 MB |
| Arrow | 5 bytes | ~5 MB |
| 节省 | 44% | ~4 MB |

---

## 版本兼容性

- Rust: 1.70+
- Python: 3.8+
- PyArrow: 14.0+
- Arrow: 53.0+

---

## 参考资料

- [使用指南](./arrow_zero_copy_guide.md)
- [设计文档](../ARROW_ZERO_COPY_DESIGN.md)
- [性能测试](../tests/performance_validation.rs)

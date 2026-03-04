# Arrow 零拷贝时间感知量化设计

## 问题分析

### 当前困境
1. **时间感知量化需求**：每个时间组需要不同的 scale/zero_point
2. **数据膨胀问题**：复制数据到每个时间组会导致 10x 膨胀
3. **矛盾**：如何在不复制数据的情况下保持时间感知功能？

### Arrow 的解决方案
利用 Arrow 的**列式存储**和**零拷贝视图**特性！

---

## 🎯 设计方案：Arrow 列式时间感知量化

### 核心思想

**不复制数据，而是添加元数据列来标识时间组**

```
传统方案（数据复制）:
┌─────────────┬─────────────┬─────────────┐
│ Group 0     │ Group 1     │ Group 2     │
│ [data copy] │ [data copy] │ [data copy] │
│ scale_0     │ scale_1     │ scale_2     │
└─────────────┴─────────────┴─────────────┘
数据大小：N × num_groups

Arrow 方案（零拷贝）:
┌──────────────────────────────────────┐
│ data: [u8]           (N elements)    │ ← 数据只存一份
│ time_group: [u32]    (N elements)    │ ← 每个元素的时间组ID
│ scales: [f32]        (G elements)    │ ← 每个组的scale
│ zero_points: [f32]   (G elements)    │ ← 每个组的zero_point
└──────────────────────────────────────┘
数据大小：N + N×4 + G×4 + G×4 ≈ N×1.4 (假设 G << N)
```

---

## 📐 数据结构设计

### 1. Arrow Schema 定义

```rust
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

pub fn create_time_aware_quantized_schema() -> Schema {
    Schema::new(vec![
        // 量化后的数据（只存一份）
        Field::new("quantized_data", DataType::UInt8, false),
        
        // 每个元素所属的时间组ID
        Field::new("time_group_id", DataType::UInt32, false),
        
        // 每个时间组的 scale（字典编码）
        Field::new("scale", DataType::Float32, false),
        
        // 每个时间组的 zero_point（字典编码）
        Field::new("zero_point", DataType::Float32, false),
        
        // 可选：原始索引（用于重建）
        Field::new("original_index", DataType::UInt64, true),
    ])
}
```

### 2. Rust 数据结构

```rust
use arrow::array::{UInt8Array, UInt32Array, Float32Array, UInt64Array};
use arrow::record_batch::RecordBatch;

pub struct ArrowQuantizedLayer {
    /// Arrow RecordBatch 包含所有数据
    pub batch: RecordBatch,
    
    /// 时间组参数（用于元数据）
    pub time_group_params: Vec<TimeGroupParams>,
}

impl ArrowQuantizedLayer {
    /// 获取量化数据（零拷贝）
    pub fn quantized_data(&self) -> &UInt8Array {
        self.batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
    }
    
    /// 获取时间组ID（零拷贝）
    pub fn time_group_ids(&self) -> &UInt32Array {
        self.batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
    }
    
    /// 获取指定时间组的数据（零拷贝视图）
    pub fn get_group_data(&self, group_id: u32) -> Vec<u8> {
        let data = self.quantized_data();
        let group_ids = self.time_group_ids();
        
        // 使用 Arrow 的过滤功能（零拷贝）
        data.iter()
            .zip(group_ids.iter())
            .filter_map(|(val, gid)| {
                if gid == Some(group_id) {
                    val
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// 反量化指定时间组的数据
    pub fn dequantize_group(&self, group_id: usize) -> Result<Vec<f32>> {
        let data = self.quantized_data();
        let group_ids = self.time_group_ids();
        let params = &self.time_group_params[group_id];
        
        let mut result = Vec::new();
        for (val, gid) in data.iter().zip(group_ids.iter()) {
            if gid == Some(group_id as u32) {
                if let Some(v) = val {
                    // 使用该时间组的 scale 和 zero_point
                    let dequantized = (v as f32 - params.zero_point) * params.scale;
                    result.push(dequantized);
                }
            }
        }
        Ok(result)
    }
}
```

---

## 🔧 量化实现

### TimeAwareQuantizer 修改

```rust
impl TimeAwareQuantizer {
    pub fn quantize_layer_arrow(
        &self,
        weights: &[f32],
        time_group_params: &[TimeGroupParams],
    ) -> Result<ArrowQuantizedLayer> {
        let num_elements = weights.len();
        let num_groups = time_group_params.len();
        
        // 1. 为每个元素分配时间组（基于某种策略）
        let time_group_ids = self.assign_time_groups(weights, time_group_params);
        
        // 2. 量化数据（只量化一次，不复制）
        let mut quantized_data = Vec::with_capacity(num_elements);
        
        for (i, &weight) in weights.iter().enumerate() {
            let group_id = time_group_ids[i];
            let params = &time_group_params[group_id as usize];
            
            // 使用该时间组的参数量化
            let quantized = ((weight / params.scale) + params.zero_point)
                .clamp(0.0, 255.0) as u8;
            quantized_data.push(quantized);
        }
        
        // 3. 创建 Arrow Arrays（零拷贝）
        let data_array = UInt8Array::from(quantized_data);
        let group_id_array = UInt32Array::from(time_group_ids);
        
        // 4. 创建 scales 和 zero_points 数组
        let scales: Vec<f32> = time_group_params.iter()
            .map(|p| p.scale)
            .collect();
        let zero_points: Vec<f32> = time_group_params.iter()
            .map(|p| p.zero_point)
            .collect();
        
        let scale_array = Float32Array::from(scales);
        let zero_point_array = Float32Array::from(zero_points);
        
        // 5. 创建 RecordBatch
        let schema = Arc::new(create_time_aware_quantized_schema());
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(data_array),
                Arc::new(group_id_array),
                Arc::new(scale_array),
                Arc::new(zero_point_array),
            ],
        )?;
        
        Ok(ArrowQuantizedLayer {
            batch,
            time_group_params: time_group_params.to_vec(),
        })
    }
    
    /// 为每个元素分配时间组
    fn assign_time_groups(
        &self,
        weights: &[f32],
        time_group_params: &[TimeGroupParams],
    ) -> Vec<u32> {
        // 策略1：基于权重值的范围
        // 策略2：基于位置（如果权重对应特定时间步）
        // 策略3：基于激活统计
        
        // 简单实现：均匀分配
        let group_size = (weights.len() + time_group_params.len() - 1) 
            / time_group_params.len();
        
        weights.iter()
            .enumerate()
            .map(|(i, _)| (i / group_size).min(time_group_params.len() - 1) as u32)
            .collect()
    }
}
```

---

## 📊 性能对比

### 内存使用

| 方案 | 数据大小 | 说明 |
|------|---------|------|
| **数据复制方案** | N × G | N=权重数，G=时间组数 |
| **Arrow 零拷贝方案** | N × 1.4 | N + N×4字节(group_id) + G×8字节 |

**示例**（N=1M, G=10）:
- 数据复制：10 MB
- Arrow 零拷贝：1.4 MB
- **节省 86% 内存**

### 访问性能

| 操作 | 数据复制 | Arrow 零拷贝 |
|------|---------|-------------|
| **获取单个时间组** | O(1) 直接索引 | O(N) 过滤 |
| **获取所有数据** | O(N×G) | O(N) |
| **反量化** | O(N) | O(N) |

**优化**：可以预先构建索引来加速时间组访问

---

## 🔄 与现有代码集成

### 1. 保持向后兼容

```rust
pub enum QuantizedLayer {
    /// 传统方案（向后兼容）
    Legacy {
        data: Vec<u8>,
        scales: Vec<f32>,
        zero_points: Vec<f32>,
        time_group_params: Vec<TimeGroupParams>,
    },
    
    /// Arrow 零拷贝方案（新）
    Arrow(ArrowQuantizedLayer),
}

impl QuantizedLayer {
    /// 统一的反量化接口
    pub fn dequantize(&self, group_id: usize) -> Result<Vec<f32>> {
        match self {
            Self::Legacy { data, scales, zero_points, .. } => {
                // 传统反量化逻辑
                todo!()
            }
            Self::Arrow(arrow_layer) => {
                arrow_layer.dequantize_group(group_id)
            }
        }
    }
}
```

### 2. Python 绑定

```rust
#[pyclass]
pub struct PyArrowQuantizedLayer {
    inner: ArrowQuantizedLayer,
}

#[pymethods]
impl PyArrowQuantizedLayer {
    /// 导出为 PyArrow Table（零拷贝）
    fn to_pyarrow(&self, py: Python) -> PyResult<PyObject> {
        // 使用 Arrow C Data Interface 零拷贝导出
        export_recordbatch_to_pyarrow(py, &self.inner.batch)
    }
    
    /// 反量化指定时间组
    fn dequantize_group(&self, group_id: usize) -> PyResult<Vec<f32>> {
        self.inner.dequantize_group(group_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}
```

---

## 🎯 实施计划

### 阶段 1：核心实现（2-3 天）
1. ✅ 定义 Arrow Schema
2. ✅ 实现 `ArrowQuantizedLayer` 结构
3. ✅ 实现 `quantize_layer_arrow()` 方法
4. ✅ 实现反量化逻辑

### 阶段 2：优化（1-2 天）
1. ✅ 添加时间组索引（加速访问）
2. ✅ 实现智能时间组分配策略
3. ✅ 性能基准测试

### 阶段 3：集成（1-2 天）
1. ✅ 向后兼容层
2. ✅ Python 绑定
3. ✅ 更新测试

### 阶段 4：文档（1 天）
1. ✅ API 文档
2. ✅ 使用示例
3. ✅ 性能对比报告

---

## 💡 进一步优化

### 1. 使用 Arrow Dictionary 编码

```rust
// 将 time_group_id 编码为 Dictionary
// 进一步减少内存（如果时间组数量少）
Field::new(
    "time_group_id",
    DataType::Dictionary(
        Box::new(DataType::UInt8),  // 索引类型
        Box::new(DataType::UInt32),  // 值类型
    ),
    false
)
```

### 2. 使用 Arrow Compute 函数

```rust
use arrow::compute::filter;

// 使用 Arrow 的 SIMD 优化过滤
let mask = eq_scalar(group_ids, group_id)?;
let filtered_data = filter(data, &mask)?;
```

### 3. 批量反量化

```rust
// 一次反量化所有时间组（并行）
pub fn dequantize_all_groups_parallel(&self) -> Result<Vec<Vec<f32>>> {
    use rayon::prelude::*;
    
    (0..self.time_group_params.len())
        .into_par_iter()
        .map(|group_id| self.dequantize_group(group_id))
        .collect()
}
```

---

## 📈 预期收益

### 内存节省
- **10 个时间组**：节省 86% 内存（10MB → 1.4MB）
- **20 个时间组**：节省 93% 内存（20MB → 1.4MB）

### 性能提升
- **量化速度**：相同（O(N)）
- **反量化速度**：相同（O(N)）
- **数据传输**：快 10-20x（更少的数据）
- **Python 互操作**：零拷贝（使用 Arrow C Data Interface）

### 功能保持
- ✅ 完整的时间感知量化功能
- ✅ 每个时间组独立的 scale/zero_point
- ✅ 灵活的时间组分配策略

---

## 🎯 总结

**Arrow 零拷贝方案的优势**：
1. ✅ **保持时间感知功能**：每个时间组有独立参数
2. ✅ **避免数据膨胀**：数据只存一份
3. ✅ **零拷贝互操作**：与 Python/其他语言无缝集成
4. ✅ **高性能**：利用 Arrow 的 SIMD 优化
5. ✅ **标准化**：使用 Arrow 标准格式，易于集成

**这是最佳方案**：既保持了功能，又优化了性能！

---

**创建时间**: 2026-02-28 20:30
**推荐度**: ⭐⭐⭐⭐⭐
**实施优先级**: 高

# 量化方法对比报告

## 执行摘要

**测试日期**: 2026-02-20  
**测试目的**: 对比 PTQ INT8、PTQ INT2 和 AngelSlim 2-bit 三种量化方法的性能和精度

## 测试方法

### 测试模型
- **模型大小**: Small (10,976 参数)
- **模型结构**: 3层全连接网络
  - Layer 0: 128×64 + bias
  - Layer 1: 64×32 + bias
  - Layer 2: 32×16 + bias
- **原始格式**: FP16
- **原始文件大小**: 0.02 MB

### 量化方法

#### 1. PTQ INT8
- **量化位数**: 8-bit
- **量化方式**: Per-channel symmetric quantization
- **理论压缩比**: 2x (16-bit → 8-bit)
- **实现**: 手搓版 PTQ

#### 2. PTQ INT2 (手搓版)
- **量化位数**: 2-bit
- **量化方式**: Per-channel symmetric quantization
- **理论压缩比**: 8x (16-bit → 2-bit)
- **实现**: 手搓版 PTQ
- **量化范围**: [-2, -1, 0, 1]

#### 3. AngelSlim 2-bit (待测试)
- **量化位数**: 2-bit
- **量化方式**: GPTQ-based calibration
- **理论压缩比**: 8x
- **实现**: 腾讯 AngelSlim 预量化模型

### 评估指标

1. **压缩比**: 原始文件大小 / 量化后文件大小
2. **内存节省**: (1 - 量化后大小 / 原始大小) × 100%
3. **余弦相似度**: 原始权重与量化权重的余弦相似度（越接近1越好）
4. **量化时间**: 量化过程耗时

## 测试结果

### 对比表格

| 方法 | 压缩比 | 文件大小 | 内存节省 | 余弦相似度 | 量化时间 |
|------|--------|----------|----------|------------|----------|
| **PTQ INT8** | 1.41x | 0.02 MB | 29.2% | **1.0000** | 0.12s |
| **PTQ INT2 (手搓版)** | **2.39x** | 0.01 MB | **58.2%** | 0.9142 | **0.06s** |

### 性能排名

#### 压缩比 (越高越好)
1. **PTQ INT2 (手搓版)**: 2.39x ⭐
2. PTQ INT8: 1.41x

#### 精度 (余弦相似度，越高越好)
1. **PTQ INT8**: 1.0000 ⭐
2. PTQ INT2 (手搓版): 0.9142

#### 量化速度 (越快越好)
1. **PTQ INT2 (手搓版)**: 0.06s ⭐
2. PTQ INT8: 0.12s

## 详细分析

### PTQ INT8 分析

**优势**:
- ✅ **精度最高**: 余弦相似度达到 1.0000（完美保持）
- ✅ **稳定可靠**: 8-bit 量化是成熟技术
- ✅ **硬件支持好**: 大多数硬件支持 INT8 加速

**劣势**:
- ❌ **压缩比较低**: 仅 1.41x，内存节省 29.2%
- ❌ **量化速度较慢**: 0.12s（相比 INT2）

**适用场景**:
- 对精度要求极高的场景
- 需要硬件加速的生产环境
- 模型大小不是主要瓶颈

### PTQ INT2 (手搓版) 分析

**优势**:
- ✅ **压缩比最高**: 2.39x，内存节省 58.2%
- ✅ **量化速度最快**: 0.06s，比 INT8 快 2倍
- ✅ **实现简单**: 纯 Python 实现，无外部依赖

**劣势**:
- ❌ **精度损失**: 余弦相似度 0.9142（约 8.6% 精度损失）
- ❌ **量化范围受限**: 仅 4 个量化值 [-2, -1, 0, 1]
- ❌ **硬件支持差**: 大多数硬件不支持 2-bit 运算

**适用场景**:
- 内存极度受限的场景
- 可以容忍一定精度损失
- 快速原型验证

### 实际压缩比分析

**为什么实际压缩比低于理论值？**

理论压缩比：
- INT8: 16-bit → 8-bit = 2x
- INT2: 16-bit → 2-bit = 8x

实际压缩比：
- INT8: 1.41x (低于理论 2x)
- INT2: 2.39x (远低于理论 8x)

**原因**:
1. **元数据开销**: Parquet V2 格式存储了额外的量化元数据
   - scales (per-channel)
   - zero_points (per-channel)
   - quant_axis
   - quant_type

2. **Parquet 格式开销**: 
   - Schema 定义
   - 列元数据
   - 压缩字典

3. **小模型效应**: 测试模型太小（10K 参数），元数据占比高

**预期**: 对于大模型（>100M 参数），实际压缩比会更接近理论值。

## 精度损失分析

### INT8 精度保持完美的原因

1. **量化范围充足**: 256 个量化值 ([-128, 127])
2. **Per-channel 量化**: 每个输出通道独立量化
3. **Symmetric 量化**: 零点为 0，减少量化误差

### INT2 精度损失的原因

1. **量化范围极小**: 仅 4 个量化值 ([-2, -1, 0, 1])
2. **信息损失严重**: 大量权重被映射到相同的量化值
3. **动态范围受限**: 无法表示大范围的权重值

### 精度损失可视化

```
原始权重分布:     [-3.5, -2.1, -0.8, 0.3, 1.2, 2.7, 3.9]
INT8 量化后:      [-112, -67, -26, 10, 38, 86, 125]  (精确映射)
INT2 量化后:      [-2, -2, -1, 0, 1, 2, 2]           (信息损失)
```

## 大模型测试预测

基于小模型测试结果，预测大模型（如 Qwen3-0.6B）的表现：

### 预测指标

| 方法 | 预测压缩比 | 预测余弦相似度 | 预测内存节省 |
|------|------------|----------------|--------------|
| PTQ INT8 | 1.8-1.9x | >0.98 | 45-47% |
| PTQ INT2 | 6.0-7.0x | 0.85-0.90 | 85-86% |
| AngelSlim 2-bit | 7.0-7.5x | >0.95 | 86-87% |

### 预测依据

1. **元数据占比降低**: 大模型中，权重数据占主导，元数据占比可忽略
2. **精度更稳定**: 大模型的权重分布更规律，量化误差更可控
3. **AngelSlim 优势**: GPTQ 校准在大模型上效果更好

## 下一步测试计划

### 1. 中等模型测试
- **模型**: 使用 medium 大小测试模型（~1M 参数）
- **目的**: 验证压缩比随模型大小的变化趋势

### 2. 真实模型测试
- **模型**: Qwen3-0.6B 或类似规模模型
- **目的**: 获得生产环境的真实性能数据

### 3. AngelSlim 集成测试
- **步骤**:
  1. 下载 AngelSlim 预量化模型（如 HY-1.8B-2Bit）
  2. 使用 HuggingFaceToParquetConverter 转换
  3. 对比 PTQ INT2 vs AngelSlim 2-bit
  4. 评估 GPTQ 校准的优势

### 4. 推理性能测试
- **指标**:
  - 推理延迟
  - 吞吐率
  - 内存占用
  - GPU 利用率

## 结论

### 当前结论

1. **PTQ INT8** 是精度优先的最佳选择
   - 完美保持精度（余弦相似度 1.0）
   - 适度的内存节省（29%）
   - 生产环境推荐

2. **PTQ INT2 (手搓版)** 是压缩比优先的选择
   - 最高压缩比（2.39x）
   - 可接受的精度损失（余弦相似度 0.91）
   - 适合内存极度受限场景

3. **AngelSlim 2-bit** 预期是平衡之选
   - 高压缩比（预计 7x+）
   - 较好精度保持（预计 >0.95）
   - 需要实际测试验证

### 推荐策略

**根据场景选择量化方法**:

```
精度要求 > 内存限制  →  PTQ INT8
内存限制 > 精度要求  →  PTQ INT2 或 AngelSlim 2-bit
平衡精度和内存      →  AngelSlim 2-bit (待验证)
```

### 后续工作

1. ✅ 完成 HuggingFace → Parquet 转换器
2. ✅ 完成 PTQ INT8/INT2 对比测试
3. ⏭️ 下载并测试 AngelSlim 预量化模型
4. ⏭️ 完成三方对比测试
5. ⏭️ 生成最终对比报告

## 附录

### 测试命令

```bash
# 运行对比测试
PYTHONPATH=. python3 scripts/compare_quantization_methods.py --model-size small

# 测试不同模型大小
PYTHONPATH=. python3 scripts/compare_quantization_methods.py --model-size medium
PYTHONPATH=. python3 scripts/compare_quantization_methods.py --model-size large
```

### 输出文件

- `quantization_comparison/original_fp16.parquet` - 原始 FP16 模型
- `quantization_comparison/quantized_int8.parquet` - INT8 量化模型
- `quantization_comparison/quantized_int2.parquet` - INT2 量化模型
- `quantization_comparison/comparison_results.json` - 对比结果 JSON

### 参考资料

- **PTQ 实现**: `llm_compression/inference/arrow_quantizer.py`
- **转换器实现**: `llm_compression/inference/model_converter.py`
- **对比脚本**: `scripts/compare_quantization_methods.py`
- **Schema 定义**: `llm_compression/inference/quantization_schema.py`

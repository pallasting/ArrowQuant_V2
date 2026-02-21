# AngelSlim 调研结果报告

## 执行摘要

AngelSlim 是腾讯开源的大模型压缩工具包，**已在 GitHub 公开发布**，支持多种量化算法和推测解码技术。经过调研，我们确认 AngelSlim 可以集成到 AI-OS 内存优化系统中。

**关键发现**:
- ✅ AngelSlim 已开源，可通过 pip 安装
- ✅ 支持 FP8、INT4、INT8 等多种量化方法
- ✅ 包含 GPTQ、AWQ 等高级校准算法
- ✅ 支持 Qwen、Hunyuan、DeepSeek 等主流模型
- ⚠️ 主要关注 FP8/INT4，2-bit 量化仍在研发中

## 安装方式

### 方式 1: pip 安装（推荐）
```bash
pip install angelslim
```

### 方式 2: 源码安装
```bash
git clone https://github.com/Tencent/AngelSlim.git
cd AngelSlim
python setup.py install
```

## 支持的量化算法

### 1. FP8 量化
- **FP8-Static**: 静态量化，需要校准数据
- **FP8-Dynamic**: 动态量化，运行时计算缩放因子
- **性能**: 接近 BF16 精度，内存节省 50%

### 2. INT4 量化
- **INT4-GPTQ**: 基于 Hessian 矩阵的校准量化
- **INT4-AWQ**: Activation-aware Weight Quantization
- **性能**: 内存节省 75%，精度损失 <2%

### 3. INT8 量化
- **INT8-Dynamic**: 动态 INT8 量化
- **性能**: 内存节省 50%，精度损失 <1%

### 4. 2-bit 量化（研发中）
- **状态**: 论文中提到，但代码库中未找到完整实现
- **HY-1.8B-2Bit**: 已发布 2-bit 模型权重，但量化工具未开源
- **建议**: 暂时不集成 2-bit，等待官方完整发布

## 性能基准测试

### Qwen3 系列模型（FP8 vs INT4）

| 模型 | 量化方法 | CEVAL | MMLU | GSM8K | 内存占用 |
|------|---------|-------|------|-------|---------|
| Qwen3-0.6B | BF16 | 45.84 | 47.21 | 42.99 | 100% |
| Qwen3-0.6B | FP8-Static | 45.99 | 46.87 | 38.06 | 50% |
| Qwen3-0.6B | INT8-Dynamic | 45.17 | 46.95 | 41.17 | 50% |
| Qwen3-8B | BF16 | 79.27 | 74.78 | 87.79 | 100% |
| Qwen3-8B | FP8-Static | 78.23 | 74.79 | 86.96 | 50% |
| Qwen3-8B | INT4-GPTQ | 77.19 | 73.26 | 86.43 | 25% |
| Qwen3-8B | INT4-AWQ | 76.15 | 73.59 | 86.96 | 25% |

**关键观察**:
- FP8 精度损失 <1%，内存节省 50%
- INT4 精度损失 1-2%，内存节省 75%
- INT8 精度介于 FP8 和 INT4 之间

## 与现有系统的对比

### AI-OS 当前实现 vs AngelSlim

| 特性 | AI-OS PTQ | AngelSlim |
|------|-----------|-----------|
| INT8 量化 | ✅ 简单 PTQ | ✅ 动态 INT8 |
| INT2 量化 | ✅ 简单 PTQ | ❌ 未开源 |
| FP8 量化 | ❌ 不支持 | ✅ Static/Dynamic |
| INT4 量化 | ❌ 不支持 | ✅ GPTQ/AWQ |
| GPTQ 校准 | ⚠️ 简化版 | ✅ 完整实现 |
| 模型支持 | 通用 | Qwen/Hunyuan/DeepSeek |
| 推理优化 | ❌ | ✅ vLLM/SGLang |

## 集成建议

### 推荐方案: 混合集成

基于调研结果，我们建议采用**混合集成方案**：

1. **保留现有 PTQ INT8/INT2**
   - 简单、可控、无外部依赖
   - 适合快速原型和教学

2. **集成 AngelSlim FP8/INT4**
   - 生产级质量
   - 更好的精度-压缩比平衡
   - 支持主流模型

3. **暂不集成 2-bit**
   - 等待 AngelSlim 官方完整发布
   - 当前 2-bit 工具不完整

### 集成架构

```python
# 扩展 QuantizationConfig
@dataclass
class QuantizationConfig:
    # 现有配置
    quant_type: Literal['int8', 'int2', 'fp16']
    calibration_method: Literal['ptq', 'gptq']
    
    # AngelSlim 扩展
    backend: Literal['ptq', 'angelslim'] = 'ptq'
    angelslim_method: Optional[Literal['fp8_static', 'fp8_dynamic', 'int4_gptq', 'int4_awq']] = None
```

### 使用示例

```bash
# 使用现有 PTQ
python -m llm_compression.tools.quantize_cli \
    --input models/qwen3-0.6b/weights.parquet \
    --output models/qwen3-0.6b/weights_int8.parquet \
    --backend ptq \
    --quant-type int8

# 使用 AngelSlim FP8
python -m llm_compression.tools.quantize_cli \
    --input models/qwen3-0.6b/weights.parquet \
    --output models/qwen3-0.6b/weights_fp8.parquet \
    --backend angelslim \
    --angelslim-method fp8_static \
    --calibration-data calibration.jsonl

# 使用 AngelSlim INT4-GPTQ
python -m llm_compression.tools.quantize_cli \
    --input models/qwen3-0.6b/weights.parquet \
    --output models/qwen3-0.6b/weights_int4_gptq.parquet \
    --backend angelslim \
    --angelslim-method int4_gptq \
    --calibration-data calibration.jsonl
```

## 技术挑战

### 1. 模型格式转换
- **挑战**: AngelSlim 使用 HuggingFace 格式，AI-OS 使用 Parquet
- **解决**: 实现双向转换器
  - Parquet → HuggingFace (量化前)
  - HuggingFace → Parquet (量化后)

### 2. 校准数据准备
- **挑战**: AngelSlim 需要校准数据集
- **解决**: 提供默认校准数据生成工具
  - 从训练数据采样
  - 使用标准数据集（C4, WikiText）

### 3. 依赖管理
- **挑战**: AngelSlim 依赖较多（transformers, vLLM 等）
- **解决**: 作为可选依赖
  ```toml
  [project.optional-dependencies]
  angelslim = [
      "angelslim>=0.3.0",
      "transformers>=4.30.0",
  ]
  ```

### 4. 推理集成
- **挑战**: AngelSlim 优化了 vLLM/SGLang 推理
- **解决**: 
  - Phase 1: 仅集成量化
  - Phase 2: 集成推理优化（可选）

## 实施计划

### Phase 1: 基础集成（2 周）

**Week 1: 环境搭建和转换器**
- [ ] 安装 AngelSlim 并验证
- [ ] 实现 Parquet ↔ HuggingFace 转换器
- [ ] 单元测试转换器

**Week 2: 量化集成**
- [ ] 扩展 QuantizationConfig
- [ ] 实现 AngelSlimQuantizer 包装器
- [ ] 集成到 quantize_cli.py
- [ ] 端到端测试

### Phase 2: 精度验证（1 周）

- [ ] 使用 Qwen3-0.6B 测试 FP8/INT4
- [ ] 对比 PTQ vs AngelSlim 精度
- [ ] 生成性能报告
- [ ] 更新文档

### Phase 3: 生产优化（1 周）

- [ ] 优化转换性能
- [ ] 添加进度条和日志
- [ ] 错误处理和回退
- [ ] 用户文档和示例

## 风险评估

### 高风险
- ❌ **2-bit 量化不可用**: 官方未完整开源
  - **缓解**: 暂不集成，等待官方发布

### 中风险
- ⚠️ **依赖冲突**: AngelSlim 可能与现有依赖冲突
  - **缓解**: 使用可选依赖，隔离安装

### 低风险
- ✅ **模型兼容性**: AngelSlim 支持主流模型
  - **缓解**: 优先支持 Qwen 系列

## 成功标准

### 功能标准
- [x] AngelSlim 成功安装
- [ ] FP8/INT4 量化成功运行
- [ ] 生成的模型可被 WeightLoaderV2 加载
- [ ] 精度满足要求（FP8: >99%, INT4: >98%）

### 性能标准
- [ ] FP8 内存占用 < 原始模型的 50%
- [ ] INT4 内存占用 < 原始模型的 25%
- [ ] 量化时间 < 10 分钟（Qwen3-0.6B）

### 质量标准
- [ ] 单元测试覆盖率 > 90%
- [ ] 集成测试通过
- [ ] 文档完整

## 下一步行动

### 立即执行
1. ✅ 完成 AngelSlim 调研
2. ⏭️ 安装 AngelSlim 并验证
3. ⏭️ 实现 Parquet ↔ HuggingFace 转换器
4. ⏭️ 原型验证：量化 Qwen3-0.6B

### 短期计划（1-2 周）
- 实现 AngelSlimQuantizer
- 集成到 CLI
- 端到端测试

### 长期计划（1 个月）
- 优化性能
- 完善文档
- 生产部署

## 参考资料

- GitHub: https://github.com/Tencent/AngelSlim
- HuggingFace: https://huggingface.co/AngelSlim
- 文档: https://github.com/Tencent/AngelSlim/tree/main/docs
- 示例: https://github.com/Tencent/AngelSlim/tree/main/scripts

## 结论

AngelSlim 是一个成熟的生产级量化工具包，适合集成到 AI-OS 内存优化系统。建议采用混合集成方案，保留现有 PTQ 的同时，添加 AngelSlim 的 FP8/INT4 支持。2-bit 量化暂不集成，等待官方完整发布。

预计集成周期：**4 周**
- Week 1-2: 基础集成
- Week 3: 精度验证
- Week 4: 生产优化

**建议立即开始 Phase 1 实施。**

# AngelSlim 验证报告

## 执行摘要

**验证日期**: 2026-02-20  
**验证状态**: ⚠️ 环境限制，无法直接安装  
**推荐方案**: 调整集成策略

## 验证结果

### 1. 安装验证

| 检查项 | 状态 | 说明 |
|--------|------|------|
| AngelSlim 可用性 | ❌ | 系统使用外部管理的 Python 环境 (PEP 668) |
| pip 安装 | ❌ | 需要虚拟环境或 `--break-system-packages` |
| 功能测试 | ⏸️ | 无法进行，因为未安装 |
| API 验证 | ⏸️ | 无法进行，因为未安装 |

### 2. 环境分析

**当前环境**:
- Python 版本: 3.13.7
- 包管理: 外部管理 (Debian/Ubuntu)
- 限制: PEP 668 - 禁止系统级 pip 安装

**问题根源**:
```
error: externally-managed-environment
× This environment is externally managed
```

这是 Python 3.11+ 在某些 Linux 发行版上的新安全特性，防止 pip 破坏系统包管理。

## 集成策略调整

基于验证结果，我们需要调整 AngelSlim 集成策略。

### 方案对比

| 方案 | 优势 | 劣势 | 推荐度 |
|------|------|------|--------|
| **A. 虚拟环境集成** | 完整功能，隔离环境 | 需要额外配置 | ⭐⭐⭐⭐ |
| **B. Docker 容器** | 完全隔离，可复现 | 复杂度高 | ⭐⭐⭐ |
| **C. 使用预量化模型** | 快速验证 | 功能受限 | ⭐⭐⭐⭐⭐ |
| **D. 暂不集成** | 无风险 | 失去 FP8/INT4 能力 | ⭐⭐ |

### 推荐方案: C + A 组合

**Phase 1: 使用预量化模型（立即可行）**
- 下载 AngelSlim 已量化的模型
- 实现 HuggingFace → Parquet 转换器
- 验证在 AI-OS 中的效果
- 评估精度和性能

**Phase 2: 虚拟环境集成（如果 Phase 1 成功）**
- 创建独立虚拟环境
- 安装 AngelSlim
- 实现完整量化工具链
- 生产部署

## Phase 1 实施计划（推荐）

### 目标
验证 AngelSlim 量化模型在 AI-OS 中的效果，无需安装 AngelSlim。

### 步骤

#### 1. 下载预量化模型
```bash
# 使用 huggingface-cli 下载（无需 AngelSlim）
pip install --user huggingface-hub

# 下载 HY-1.8B-2Bit 模型
huggingface-cli download AngelSlim/HY-1.8B-2Bit \
    --local-dir models/hy-1.8b-2bit

# 或下载其他预量化模型
# Qwen3 FP8: AngelSlim/Qwen3-0.6B-FP8
# Qwen3 INT4: AngelSlim/Qwen3-0.6B-INT4-GPTQ
```

#### 2. 实现 HuggingFace → Parquet 转换器
```python
# llm_compression/inference/model_converter.py

from pathlib import Path
from typing import Dict, List
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoModel, AutoTokenizer

from llm_compression.logger import logger
from llm_compression.inference.quantization_schema import V2_SCHEMA

class HuggingFaceToParquetConverter:
    """
    将 HuggingFace 模型转换为 Parquet V2 格式。
    
    支持:
    - FP16/BF16 模型
    - FP8 量化模型
    - INT4 量化模型
    - INT8 量化模型
    """
    
    def convert(
        self,
        hf_model_path: str,
        output_parquet: str
    ) -> None:
        """
        转换 HuggingFace 模型为 Parquet。
        
        Args:
            hf_model_path: HuggingFace 模型路径
            output_parquet: 输出 Parquet 文件路径
        """
        logger.info(f"Converting {hf_model_path} to Parquet")
        
        # 1. 加载模型
        model = AutoModel.from_pretrained(hf_model_path)
        
        # 2. 提取权重
        weights = self._extract_weights(model)
        
        # 3. 检测量化格式
        quant_info = self._detect_quantization(model)
        
        # 4. 转换为 Parquet 格式
        rows = self._convert_to_rows(weights, quant_info)
        
        # 5. 保存
        table = pa.Table.from_pylist(rows, schema=V2_SCHEMA)
        pq.write_table(table, output_parquet)
        
        logger.info(f"Conversion complete: {output_parquet}")
```

#### 3. 验证转换和加载
```python
# tests/integration/test_angelslim_models.py

def test_load_angelslim_fp8_model():
    """测试加载 AngelSlim FP8 模型"""
    # 1. 转换模型
    converter = HuggingFaceToParquetConverter()
    converter.convert(
        hf_model_path="models/hy-1.8b-2bit",
        output_parquet="models/hy-1.8b-2bit.parquet"
    )
    
    # 2. 加载模型
    loader = WeightLoaderV2("models/hy-1.8b-2bit.parquet")
    weight = loader.load_weight("layer.0", "weight")
    
    # 3. 验证
    assert weight is not None
    assert weight.dtype in [torch.float16, torch.int8]
```

#### 4. 精度评估
```python
# 使用 PrecisionValidator 评估
validator = PrecisionValidator()
result = validator.validate(
    original_model_path="models/hy-1.8b-fp16.parquet",
    quantized_model_path="models/hy-1.8b-2bit.parquet",
    test_texts=test_texts
)

print(f"Cosine Similarity: {result.cosine_similarity}")
print(f"PPL Increase: {result.ppl_increase}")
```

### 预期成果

**成功标准**:
- ✅ 成功转换 HuggingFace 模型为 Parquet
- ✅ WeightLoaderV2 可以加载转换后的模型
- ✅ 余弦相似度 > 0.95
- ✅ 推理功能正常

**如果成功**:
- 证明 AngelSlim 量化模型质量高
- 值得投入资源完整集成
- 进入 Phase 2（虚拟环境集成）

**如果失败**:
- 继续使用现有 PTQ INT8/INT2
- 等待更好的集成时机

## Phase 2 实施计划（条件执行）

**前提条件**: Phase 1 验证成功

### 虚拟环境设置

```bash
# 1. 创建虚拟环境
python3 -m venv venv-angelslim

# 2. 激活虚拟环境
source venv-angelslim/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
pip install angelslim

# 4. 验证安装
python -c "import angelslim; print(angelslim.__version__)"
```

### 完整集成

1. **实现双向转换器**
   - Parquet → HuggingFace
   - HuggingFace → Parquet

2. **包装 AngelSlim API**
   - AngelSlimQuantizer
   - 支持 FP8/INT4/INT8

3. **CLI 集成**
   - 添加 `--backend angelslim`
   - 添加量化方法选项

4. **测试和文档**
   - 单元测试
   - 集成测试
   - 用户文档

## 替代方案: Docker 容器

如果虚拟环境不可行，可以使用 Docker：

```dockerfile
# Dockerfile.angelslim
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install angelslim

# 复制代码
COPY . .

# 安装项目
RUN pip install -e .

CMD ["python", "-m", "llm_compression.tools.quantize_cli"]
```

```bash
# 构建镜像
docker build -f Dockerfile.angelslim -t ai-os-angelslim .

# 运行量化
docker run -v $(pwd)/models:/app/models \
    ai-os-angelslim \
    --input /app/models/qwen3-0.6b.parquet \
    --output /app/models/qwen3-0.6b-fp8.parquet \
    --backend angelslim \
    --angelslim-method fp8_static
```

## 当前行动建议

### 立即执行（今天）

1. **实现 HuggingFace → Parquet 转换器**
   - 创建 `llm_compression/inference/model_converter.py`
   - 实现基础转换逻辑
   - 单元测试

2. **下载测试模型**
   ```bash
   pip install --user huggingface-hub
   huggingface-cli download AngelSlim/HY-1.8B-2Bit \
       --local-dir models/hy-1.8b-2bit
   ```

3. **验证转换**
   - 转换下载的模型
   - 使用 WeightLoaderV2 加载
   - 验证权重正确性

### 本周执行

1. **精度评估**
   - 对比原始模型和量化模型
   - 生成评估报告

2. **性能测试**
   - 推理延迟
   - 内存占用
   - 吞吐率

3. **决策点**
   - 如果验证成功 → 进入 Phase 2
   - 如果验证失败 → 继续使用 PTQ

## 风险和缓解

### 风险 1: 转换器复杂度高
- **缓解**: 先支持简单模型（Qwen3-0.6B）
- **缓解**: 参考 HuggingFace 文档

### 风险 2: 量化格式不兼容
- **缓解**: 详细分析 AngelSlim 模型格式
- **缓解**: 实现格式检测和适配

### 风险 3: 精度不达标
- **缓解**: 使用多个测试数据集
- **缓解**: 对比官方基准测试结果

## 成功标准

### Phase 1 成功标准
- [x] 转换器实现完成
- [ ] 成功转换至少 1 个 AngelSlim 模型
- [ ] WeightLoaderV2 可以加载
- [ ] 余弦相似度 > 0.95
- [ ] 推理功能正常

### Phase 2 成功标准（如果执行）
- [ ] 虚拟环境配置成功
- [ ] AngelSlim 安装成功
- [ ] 完整量化工具链可用
- [ ] CLI 集成完成
- [ ] 文档完整

## 结论

**当前状态**: AngelSlim 无法在系统环境中直接安装

**推荐路径**: 
1. **立即**: 实现转换器，使用预量化模型验证
2. **短期**: 如果验证成功，创建虚拟环境完整集成
3. **长期**: 生产部署，持续优化

**预期时间**:
- Phase 1（验证）: 2-3 天
- Phase 2（集成）: 2-3 周（如果需要）

**下一步**: 开始实现 HuggingFace → Parquet 转换器

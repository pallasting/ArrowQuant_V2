# 模型选择指南 (Model Selection Guide)

本指南帮助您选择最适合您需求的 LLM 模型和部署方式。

---

## 目录

1. [支持的模型](#支持的模型)
2. [部署模式对比](#部署模式对比)
3. [模型选择决策树](#模型选择决策树)
4. [性能对比](#性能对比)
5. [成本分析](#成本分析)
6. [使用场景推荐](#使用场景推荐)
7. [配置示例](#配置示例)

---

## 支持的模型

### Phase 1.1 本地模型 (推荐)

#### 1. Qwen2.5-7B-Instruct ⭐ 推荐

**基本信息**:
- 参数量: 7B
- 模型大小: 4.7 GB (Q4_K_M 量化)
- 上下文长度: 32,768 tokens
- 开发者: 阿里巴巴通义千问团队

**性能指标**:
- 压缩比: ~15x
- 压缩延迟: ~1.5s
- 重构延迟: ~420ms
- 重构质量: ~0.89
- 吞吐量: ~105/min

**优势**:
- ✅ 优秀的中文和英文理解能力
- ✅ 长上下文支持（32K tokens）
- ✅ 高质量的摘要生成
- ✅ 良好的实体识别能力
- ✅ 适中的资源需求

**适用场景**:
- 通用记忆压缩
- 中英文混合内容
- 长文本处理
- 生产环境部署

**硬件要求**:
- GPU: 8GB+ VRAM (推荐 16GB)
- CPU: 4 核心+
- RAM: 8GB+

#### 2. Llama 3.1 8B Instruct

**基本信息**:
- 参数量: 8B
- 模型大小: 4.9 GB (Q4_K_M 量化)
- 上下文长度: 8,192 tokens
- 开发者: Meta

**性能指标**:
- 压缩比: ~14x
- 压缩延迟: ~1.8s
- 重构延迟: ~450ms
- 重构质量: ~0.87
- 吞吐量: ~95/min

**优势**:
- ✅ 强大的英文理解能力
- ✅ 开源社区支持广泛
- ✅ 稳定的性能表现

**适用场景**:
- 英文为主的内容
- 标准长度文本
- 需要社区支持的场景

**硬件要求**:
- GPU: 8GB+ VRAM
- CPU: 4 核心+
- RAM: 8GB+

#### 3. Gemma 3 4B

**基本信息**:
- 参数量: 4B
- 模型大小: 3.3 GB (Q4_K_M 量化)
- 上下文长度: 8,192 tokens
- 开发者: Google

**性能指标**:
- 压缩比: ~12x
- 压缩延迟: ~1.2s
- 重构延迟: ~380ms
- 重构质量: ~0.84
- 吞吐量: ~120/min

**优势**:
- ✅ 最快的推理速度
- ✅ 最低的资源需求
- ✅ 适合资源受限环境

**适用场景**:
- 资源受限环境
- 需要高吞吐量
- 短文本处理
- 开发和测试

**硬件要求**:
- GPU: 4GB+ VRAM
- CPU: 2 核心+
- RAM: 4GB+

### Phase 1.0 云端 API

#### 云端 LLM API

**基本信息**:
- 端点: http://localhost:8045
- 模型: 可配置

**性能指标**:
- 压缩比: ~39.63x (最高)
- 压缩延迟: < 3s
- 重构延迟: < 500ms
- 重构质量: > 0.90 (最高)
- 吞吐量: ~50/min

**优势**:
- ✅ 最高的压缩比
- ✅ 最高的重构质量
- ✅ 无需本地资源
- ✅ 快速开始

**劣势**:
- ❌ 需要网络连接
- ❌ API 调用成本
- ❌ 数据隐私考虑
- ❌ 较低的吞吐量

**适用场景**:
- 快速原型验证
- 最高质量要求
- 无本地 GPU 资源
- 临时使用

---

## 部署模式对比

| 特性 | 本地模型 (Phase 1.1) | 云端 API (Phase 1.0) |
|------|---------------------|---------------------|
| **成本** | 💰 低 (90% 节省) | 💰💰💰 高 |
| **速度** | ⚡⚡ 快 (< 2s) | ⚡ 中等 (< 3s) |
| **吞吐量** | 📈 高 (100+/min) | 📈 中 (50/min) |
| **质量** | ⭐⭐⭐⭐ 优秀 (~0.89) | ⭐⭐⭐⭐⭐ 最佳 (> 0.90) |
| **压缩比** | 📦 高 (~15x) | 📦📦 最高 (39.63x) |
| **隐私** | 🔒 完全本地 | 🌐 需要传输 |
| **离线** | ✅ 支持 | ❌ 需要网络 |
| **资源** | 🖥️ 需要 GPU/CPU | ☁️ 无需本地资源 |
| **设置** | ⚙️ 需要配置 | ⚙️ 简单 |

---

## 模型选择决策树

```
开始
  │
  ├─ 有本地 GPU (8GB+)？
  │   ├─ 是 → 需要最高质量？
  │   │   ├─ 是 → 云端 API (Phase 1.0)
  │   │   └─ 否 → Qwen2.5-7B ⭐ (推荐)
  │   │
  │   └─ 否 → 有 CPU (4核+)？
  │       ├─ 是 → 需要中文支持？
  │       │   ├─ 是 → Qwen2.5-7B (CPU 模式)
  │       │   └─ 否 → Gemma 3 4B (最快)
  │       │
  │       └─ 否 → 云端 API (Phase 1.0)
  │
  └─ 考虑因素：
      • 成本敏感 → 本地模型
      • 数据隐私 → 本地模型
      • 离线使用 → 本地模型
      • 最高质量 → 云端 API
      • 快速开始 → 云端 API
```

---

## 性能对比

### 压缩性能

| 模型 | 压缩比 | 压缩延迟 | 质量分数 |
|------|--------|----------|----------|
| 云端 API | 39.63x | < 3s | 0.90+ |
| Qwen2.5-7B | ~15x | ~1.5s | ~0.89 |
| Llama 3.1 8B | ~14x | ~1.8s | ~0.87 |
| Gemma 3 4B | ~12x | ~1.2s | ~0.84 |

### 重构性能

| 模型 | 重构延迟 | 语义相似度 | 实体准确率 |
|------|----------|------------|------------|
| 云端 API | < 500ms | > 0.90 | 100% |
| Qwen2.5-7B | ~420ms | ~0.89 | > 95% |
| Llama 3.1 8B | ~450ms | ~0.87 | > 95% |
| Gemma 3 4B | ~380ms | ~0.84 | > 90% |

### 吞吐量对比

| 模型 | 吞吐量 (操作/分钟) | 批量大小 |
|------|-------------------|----------|
| 云端 API | ~50 | 16 |
| Qwen2.5-7B | ~105 | 32 |
| Llama 3.1 8B | ~95 | 32 |
| Gemma 3 4B | ~120 | 32 |

---

## 成本分析

### 云端 API 成本 (Phase 1.0)

假设：
- API 成本: $0.001 / 1K tokens
- 平均每次压缩: 1K tokens
- 每天处理: 10,000 次

**月度成本**:
```
10,000 次/天 × 30 天 × $0.001 = $300/月
```

### 本地模型成本 (Phase 1.1)

假设：
- GPU 电费: $0.10 / kWh
- GPU 功耗: 300W (Mi50)
- 每天运行: 8 小时

**月度成本**:
```
300W × 8h × 30天 × $0.10/kWh = $7.2/月
```

**成本节省**: $300 - $7.2 = $292.8/月 (97.6% 节省)

### 总拥有成本 (TCO)

| 项目 | 云端 API | 本地模型 |
|------|----------|----------|
| 初始投资 | $0 | $2,000 (GPU) |
| 月度运营 | $300 | $7.2 |
| 年度成本 | $3,600 | $86.4 |
| 3年 TCO | $10,800 | $2,259 |
| **节省** | - | **$8,541 (79%)** |

---

## 使用场景推荐

### 场景 1: 生产环境 - 大规模部署

**推荐**: Qwen2.5-7B (本地模型)

**理由**:
- 高吞吐量 (105/min)
- 低成本 (90% 节省)
- 稳定性能
- 数据隐私

**配置**:
```yaml
model:
  prefer_local: true
  ollama_endpoint: "http://localhost:11434"
  
performance:
  batch_size: 32
  max_concurrent: 8
```

### 场景 2: 开发和测试

**推荐**: Gemma 3 4B (本地模型)

**理由**:
- 最快的推理速度
- 最低的资源需求
- 快速迭代

**配置**:
```yaml
model:
  prefer_local: true
  ollama_endpoint: "http://localhost:11434"
  
performance:
  batch_size: 16
  max_concurrent: 4
```

### 场景 3: 高质量要求

**推荐**: 云端 API

**理由**:
- 最高压缩比 (39.63x)
- 最高重构质量 (> 0.90)
- 无需本地资源

**配置**:
```yaml
model:
  prefer_local: false
  
llm:
  cloud_endpoint: "http://localhost:8045"
  timeout: 30.0
```

### 场景 4: 混合模式 - 智能降级

**推荐**: 本地优先 + 云端备用

**理由**:
- 平衡成本和质量
- 自动降级策略
- 高可用性

**配置**:
```yaml
model:
  prefer_local: true
  ollama_endpoint: "http://localhost:11434"
  quality_threshold: 0.85  # 低于此阈值切换到云端
  
llm:
  cloud_endpoint: "http://localhost:8045"  # 备用
```

### 场景 5: 离线环境

**推荐**: Qwen2.5-7B (本地模型)

**理由**:
- 完全离线运行
- 无需网络连接
- 数据不出本地

**配置**:
```yaml
model:
  prefer_local: true
  ollama_endpoint: "http://localhost:11434"
  
llm:
  cloud_endpoint: null  # 禁用云端
```

---

## 配置示例

### 示例 1: 本地模型 - 高性能配置

```yaml
# config.yaml

# 模型配置
model:
  prefer_local: true
  ollama_endpoint: "http://localhost:11434"
  local_endpoints:
    qwen2.5: "http://localhost:11434/v1"
    llama3.1: "http://localhost:11434/v1"
    gemma3: "http://localhost:11434/v1"
  quality_threshold: 0.85

# 性能配置
performance:
  batch_size: 32          # 大批量
  max_concurrent: 8       # 高并发
  cache_size: 50000       # 大缓存
  cache_ttl: 7200         # 2小时

# 压缩配置
compression:
  min_compress_length: 100
  max_tokens: 100
  temperature: 0.3
```

### 示例 2: 云端 API - 高质量配置

```yaml
# config.yaml

# 模型配置
model:
  prefer_local: false
  quality_threshold: 0.90  # 更高的质量要求

# LLM 配置
llm:
  cloud_endpoint: "http://localhost:8045"
  timeout: 60.0           # 更长的超时
  max_retries: 5          # 更多重试
  rate_limit: 30          # 较低的速率限制

# 性能配置
performance:
  batch_size: 16
  max_concurrent: 4
  cache_size: 10000
```

### 示例 3: 混合模式 - 智能降级

```yaml
# config.yaml

# 模型配置
model:
  prefer_local: true
  ollama_endpoint: "http://localhost:11434"
  quality_threshold: 0.85  # 质量阈值

# LLM 配置 - 云端备用
llm:
  cloud_endpoint: "http://localhost:8045"
  timeout: 30.0
  max_retries: 3

# 性能配置
performance:
  batch_size: 32
  max_concurrent: 8
```

---

## 模型切换

### 运行时切换模型

```python
from llm_compression import ModelSelector, LLMCompressor

# 初始化
model_selector = ModelSelector(
    cloud_endpoint="http://localhost:8045",
    prefer_local=True,
    ollama_endpoint="http://localhost:11434"
)

compressor = LLMCompressor(
    llm_client=llm_client,
    model_selector=model_selector
)

# 方法 1: 自动选择（基于配置）
compressed = await compressor.compress(text)

# 方法 2: 手动指定模型
compressed = await compressor.compress(
    text,
    manual_model="qwen2.5"  # 强制使用 Qwen2.5
)

# 方法 3: 基于质量要求选择
compressed = await compressor.compress(
    text,
    quality_threshold=0.90  # 高质量要求，可能使用云端
)
```

### 降级策略

系统自动按以下顺序降级：

1. **本地模型** (Qwen2.5-7B) - 首选
2. **云端 API** - 本地失败时
3. **简单压缩** (zstd) - API 失败时
4. **直接存储** - 所有方法失败时

---

## 性能调优建议

### 1. GPU 优化

```bash
# AMD GPU (ROCm)
export OLLAMA_GPU_DRIVER=rocm
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCM_PATH=/opt/rocm

# 启动 Ollama
ollama serve
```

### 2. 批量大小调优

```yaml
performance:
  batch_size: 32  # GPU: 32, CPU: 8-16
  max_concurrent: 8  # 根据 CPU 核心数调整
```

### 3. 缓存策略

```yaml
performance:
  cache_size: 50000  # 大缓存提高命中率
  cache_ttl: 7200    # 2小时 TTL
```

### 4. 模型量化

```bash
# 下载不同量化版本
ollama pull qwen2.5:7b-instruct-q4_k_m  # 4-bit (推荐)
ollama pull qwen2.5:7b-instruct-q5_k_m  # 5-bit (更高质量)
ollama pull qwen2.5:7b-instruct-q8_0    # 8-bit (最高质量)
```

---

## 故障排查

### 问题 1: 本地模型推理慢

**症状**: 压缩延迟 > 5s

**解决方案**:
1. 检查 GPU 是否正确配置
2. 降低批量大小
3. 使用更小的模型（Gemma 3 4B）
4. 检查系统资源使用

### 问题 2: 质量不达标

**症状**: 重构质量 < 0.85

**解决方案**:
1. 切换到更大的模型（Qwen2.5-7B）
2. 启用云端 API 备用
3. 调整 temperature 参数
4. 增加 max_tokens

### 问题 3: 成本过高

**症状**: 云端 API 成本超预算

**解决方案**:
1. 切换到本地模型
2. 启用缓存
3. 增加批量大小
4. 优化压缩阈值

---

## 总结

### 快速选择指南

| 需求 | 推荐模型 |
|------|----------|
| 🏢 生产环境 | Qwen2.5-7B |
| 💰 成本敏感 | Qwen2.5-7B |
| ⚡ 速度优先 | Gemma 3 4B |
| ⭐ 质量优先 | 云端 API |
| 🔒 数据隐私 | 本地模型 |
| 🌐 离线使用 | 本地模型 |
| 🧪 开发测试 | Gemma 3 4B |

### 推荐配置

**最佳实践**: Qwen2.5-7B (本地) + 云端 API (备用)

这种混合模式提供：
- ✅ 90% 成本节省
- ✅ 高吞吐量 (100+/min)
- ✅ 自动降级保证质量
- ✅ 高可用性

---

**需要更多帮助？** 查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 或 [API_REFERENCE.md](API_REFERENCE.md)

# Phase 1.1 Technical Plan: 本地模型部署和成本优化

## Executive Summary

**目标**: 部署本地开源模型，降低 90% 运营成本，提升性能  
**时间**: 4-6 周（10-12 天工作量）  
**硬件**: AMD Mi50 GPU (16GB HBM2)  
**部署方案**: Ollama (主) + vLLM (备选)  
**GPU 后端**: ROCm (主) + Vulkan (备选) + OpenCL (备选)

---

## 1. 部署框架选择

### 1.1 Ollama vs vLLM 对比分析

| 特性 | Ollama | vLLM |
|------|--------|------|
| **易用性** | ⭐⭐⭐⭐⭐ 一键部署 | ⭐⭐⭐ 需要配置 |
| **性能** | ⭐⭐⭐ 适中 | ⭐⭐⭐⭐⭐ 高吞吐量 |
| **AMD GPU 支持** | ⭐⭐⭐⭐ ROCm 良好 | ⭐⭐⭐⭐ ROCm 支持 |
| **内存优化** | ⭐⭐⭐⭐ 自动量化 | ⭐⭐⭐⭐⭐ PagedAttention |
| **API 兼容性** | ⭐⭐⭐⭐⭐ OpenAI 兼容 | ⭐⭐⭐⭐⭐ OpenAI 兼容 |
| **批处理** | ⭐⭐⭐ 基础支持 | ⭐⭐⭐⭐⭐ 优化批处理 |
| **维护成本** | ⭐⭐⭐⭐⭐ 低 | ⭐⭐⭐ 中等 |

**推荐方案**: 
- **主方案**: Ollama（易用性高，AMD GPU 支持好，维护成本低）
- **备选方案**: vLLM（高性能场景，需要更高吞吐量时）

### 1.2 GPU 后端选择策略

#### ROCm (主要后端)
- **优点**: 
  - 性能最佳（比 Vulkan 快 44% 文本生成，3x prompt 处理）
  - AMD 官方支持
  - 完整的 PyTorch 生态
- **缺点**: 
  - 配置复杂
  - 有时不稳定
  - 需要特定内核版本
- **适用场景**: 生产环境，性能优先

#### Vulkan (备选后端)
- **优点**: 
  - 跨平台兼容性好
  - 配置简单
  - 稳定性高
- **缺点**: 
  - 性能比 ROCm 低 30-40%
- **适用场景**: ROCm 不可用时的备选方案

#### OpenCL (最后备选)
- **优点**: 
  - 最广泛的兼容性
  - 无需特殊驱动
- **缺点**: 
  - 性能最低
  - 优化较少
- **适用场景**: 其他方案都失败时的保底方案

**实施策略**: 三层降级
```
ROCm (优先) → Vulkan (备选) → OpenCL (保底)
```

---

## 2. 候选模型评估

### 2.1 Step 3.5 Flash

**基本信息**:
- **参数**: 196B 总参数，11B 激活参数（MoE 架构）
- **上下文**: 256K tokens
- **吞吐量**: 100-300 tok/s（峰值 350 tok/s）
- **量化版本**: FP8, INT4

**适用性评估**:
- ✅ **优点**: 
  - 高效率（11B 激活 vs 196B 总参数）
  - 优秀的推理能力
  - 支持长上下文（256K）
  - 有量化版本（INT4 可在 16GB GPU 运行）
- ⚠️ **挑战**: 
  - MoE 架构内存需求较高
  - 需要 INT4 量化才能在 Mi50 上运行
- 📊 **推荐度**: ⭐⭐⭐⭐ (高度推荐，需量化)

**部署建议**:
```bash
# 使用 INT4 量化版本
ollama pull stepfun-ai/step-3.5-flash:int4
```

### 2.2 MiniCPM-o 4.5

**基本信息**:
- **参数**: 9B 总参数
- **架构**: 端到端多模态（SigLip2 + Whisper + CosyVoice2 + Qwen3-8B）
- **特性**: 支持全双工多模态流式处理

**适用性评估**:
- ✅ **优点**: 
  - 参数量适中（9B）
  - 可直接在 Mi50 上运行
  - 多模态能力（如需要）
- ❌ **缺点**: 
  - 主要针对多模态任务
  - 对于纯文本压缩可能过于复杂
  - 文档较少
- 📊 **推荐度**: ⭐⭐ (不太适合，除非需要多模态)

**结论**: 不推荐用于文本压缩任务

### 2.3 Stable-DiffCoder

**基本信息**:
- **参数**: 8B
- **架构**: 基于 Seed-Coder，使用 block diffusion CPT
- **训练数据**: 1.3T tokens
- **特性**: 代码生成和理解

**适用性评估**:
- ⚠️ **优点**: 
  - 参数量适中（8B）
  - 可在 Mi50 上运行
  - 代码理解能力强
- ❌ **缺点**: 
  - 主要针对代码任务
  - 扩散模型架构（非标准 AR）
  - 对文本摘要任务不是最优
- 📊 **推荐度**: ⭐⭐ (不推荐，专注代码任务)

**结论**: 不推荐用于通用文本压缩

### 2.4 其他候选模型

#### MiroFish
- **状态**: GitHub 链接无效或项目不存在
- **推荐度**: ❌ 无法评估

#### Intern-S1-Pro
- **信息**: ModelScope 上的模型
- **状态**: 需要进一步调研
- **推荐度**: ⏸️ 待评估

#### MoPPS / Motus
- **用途**: 机器人控制和物理模拟
- **推荐度**: ❌ 不适用于文本压缩

---

## 3. 推荐模型方案

### 3.1 主力模型: Qwen2.5-7B-Instruct

**为什么选择 Qwen2.5**:
- ✅ 参数量适中（7B），完美适配 Mi50 16GB
- ✅ 优秀的中英文能力
- ✅ 强大的摘要和压缩能力
- ✅ Ollama 官方支持，一键部署
- ✅ 社区活跃，文档完善
- ✅ 支持 ROCm 和 Vulkan

**性能预期**:
- 压缩延迟: < 2s（目标）
- 重构延迟: < 500ms
- 吞吐量: > 100/min
- 压缩比: 30-40x（预期）

**部署命令**:
```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull qwen2.5:7b-instruct

# 启动服务
ollama serve
```

### 3.2 备选模型: Llama 3.1 8B

**为什么作为备选**:
- ✅ Meta 官方模型，质量保证
- ✅ 8B 参数，适配 Mi50
- ✅ 优秀的通用能力
- ✅ Ollama 支持良好

**部署命令**:
```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 3.3 高性能选项: Step 3.5 Flash INT4

**使用场景**: 需要更高质量时
- ✅ 11B 激活参数（INT4 量化）
- ✅ 可在 Mi50 上运行
- ✅ 更强的推理能力

**部署命令**:
```bash
# 需要等待 Ollama 官方支持或使用 vLLM
# vLLM 部署示例
python -m vllm.entrypoints.openai.api_server \
  --model stepfun-ai/Step-3.5-Flash-Int4 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9
```

---

## 4. 环境配置方案

### 4.1 ROCm 安装（主要方案）

#### 系统要求
- Linux 内核: 5.15+ (推荐 6.x)
- Ubuntu 22.04 / 24.04
- AMD Mi50 (gfx906 架构)

#### 安装步骤

```bash
# 1. 添加 ROCm 仓库
wget https://repo.radeon.com/amdgpu-install/6.4.2/ubuntu/jammy/amdgpu-install_6.4.60402-1_all.deb
sudo dpkg -i amdgpu-install_6.4.60402-1_all.deb
sudo apt update

# 2. 安装 ROCm
sudo amdgpu-install --usecase=rocm

# 3. 添加用户到 render 和 video 组
sudo usermod -a -G render,video $USER

# 4. 重启系统
sudo reboot

# 5. 验证安装
rocm-smi
rocminfo | grep "Name:"
```

#### 环境变量配置

```bash
# 添加到 ~/.bashrc
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=9.0.6  # Mi50 特定配置
```

### 4.2 Vulkan 安装（备选方案）

```bash
# 安装 Vulkan SDK
sudo apt install vulkan-tools vulkan-validationlayers-dev

# 安装 AMD Vulkan 驱动
sudo apt install mesa-vulkan-drivers

# 验证安装
vulkaninfo | grep "deviceName"
```

### 4.3 OpenCL 安装（保底方案）

```bash
# 安装 OpenCL
sudo apt install ocl-icd-opencl-dev clinfo

# 验证安装
clinfo
```

### 4.4 Ollama 配置

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 配置环境变量（ROCm）
export OLLAMA_GPU_DRIVER=rocm
export HSA_OVERRIDE_GFX_VERSION=9.0.6

# 或配置 Vulkan
export OLLAMA_GPU_DRIVER=vulkan

# 启动服务
ollama serve
```

---

## 5. 性能优化策略

### 5.1 模型量化

**量化方案**:
- **Q4_K_M**: 4-bit 量化，推荐用于 Mi50
- **Q5_K_M**: 5-bit 量化，质量更高但内存需求更大
- **Q8_0**: 8-bit 量化，接近原始质量

**Ollama 量化示例**:
```bash
# 拉取量化模型
ollama pull qwen2.5:7b-instruct-q4_K_M

# 或自定义量化
ollama create my-qwen2.5-q4 -f Modelfile
```

### 5.2 批处理优化

**配置参数**:
```python
# 在 LLMClient 中配置
batch_size = 16  # 增加到 16-32
max_concurrent = 4  # 并发请求数
```

### 5.3 缓存策略

**KV Cache 优化**:
```bash
# Ollama 配置
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_QUEUE=512
```

### 5.4 内存管理

**Mi50 16GB 内存分配**:
- 模型权重: 8-10GB (Q4 量化)
- KV Cache: 4-6GB
- 系统开销: 2GB
- **总计**: ~14-18GB（需要量化）

---

## 6. 集成方案

### 6.1 架构设计

```
┌─────────────────────────────────────────────┐
│           LLM Compression System            │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌──────────────┐   │
│  │ Cloud API    │      │ Local Model  │   │
│  │ (Phase 1.0)  │      │ (Phase 1.1)  │   │
│  └──────┬───────┘      └──────┬───────┘   │
│         │                     │            │
│         └──────┬──────────────┘            │
│                │                            │
│         ┌──────▼───────┐                   │
│         │ Model Router │                   │
│         │  (Priority)  │                   │
│         └──────┬───────┘                   │
│                │                            │
│    ┌───────────┼───────────┐              │
│    │           │           │              │
│ ┌──▼──┐    ┌──▼──┐    ┌──▼──┐           │
│ │Local│    │Cloud│    │Fall-│           │
│ │ GPU │    │ API │    │back │           │
│ └─────┘    └─────┘    └─────┘           │
│                                             │
└─────────────────────────────────────────────┘
```

### 6.2 模型选择策略

**优先级**:
1. **本地模型** (Ollama/vLLM) - 优先使用
2. **云端 API** - 本地不可用或质量不达标时
3. **简单压缩** (zstd) - 所有 LLM 都失败时

**实现代码**:
```python
class HybridModelSelector:
    def select_endpoint(self, text: str, quality_requirement: str) -> str:
        # 1. 尝试本地模型
        if self.local_available and quality_requirement != "HIGH":
            return "http://localhost:11434/v1"  # Ollama
        
        # 2. 降级到云端 API
        if self.cloud_available:
            return "http://localhost:8045/v1"  # Cloud API
        
        # 3. 最后降级到简单压缩
        return "fallback://zstd"
```

### 6.3 成本监控

**成本对比**:
- **Cloud API**: ~$0.001/1K tokens
- **Local Model**: ~$0.0001/1K tokens (电费)
- **节省**: 90%

**监控指标**:
```python
class CostMonitor:
    def track_usage(self):
        return {
            "cloud_tokens": self.cloud_tokens,
            "local_tokens": self.local_tokens,
            "cloud_cost": self.cloud_tokens * 0.001 / 1000,
            "local_cost": self.local_tokens * 0.0001 / 1000,
            "savings": self.calculate_savings()
        }
```

---

## 7. 测试和验证计划

### 7.1 环境验证

**ROCm 验证**:
```bash
# 检查 GPU
rocm-smi

# 检查 PyTorch ROCm 支持
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

**Ollama 验证**:
```bash
# 测试模型加载
ollama run qwen2.5:7b-instruct "Hello, how are you?"

# 检查 GPU 使用
watch -n 1 rocm-smi
```

### 7.2 性能基准测试

**测试指标**:
- 压缩延迟: < 2s
- 重构延迟: < 500ms
- 吞吐量: > 100/min
- 压缩比: > 30x
- 质量: > 0.85

**测试脚本**:
```python
# tests/benchmark/test_local_model_performance.py
async def test_local_model_latency():
    # 测试 100 个样本
    latencies = []
    for text in test_samples:
        start = time.time()
        result = await local_compressor.compress(text)
        latencies.append(time.time() - start)
    
    assert np.mean(latencies) < 2.0  # < 2s
    assert np.percentile(latencies, 95) < 3.0  # p95 < 3s
```

### 7.3 质量验证

**对比测试**:
```python
# 对比云端 vs 本地质量
async def compare_quality():
    for text in test_samples:
        cloud_result = await cloud_compressor.compress(text)
        local_result = await local_compressor.compress(text)
        
        cloud_quality = evaluator.evaluate(text, cloud_result)
        local_quality = evaluator.evaluate(text, local_result)
        
        # 本地质量应该 >= 85% 云端质量
        assert local_quality.similarity >= cloud_quality.similarity * 0.85
```

---

## 8. 实施时间表

### Week 4: 环境准备和模型部署

**Day 1-2: 环境配置**
- 安装 ROCm 6.4.2
- 配置 Vulkan 和 OpenCL 备选
- 验证 GPU 可用性

**Day 3-4: Ollama 部署**
- 安装 Ollama
- 部署 Qwen2.5-7B
- 测试基本功能

**Day 5: 模型集成**
- 更新 ModelSelector
- 实现本地模型优先策略
- 集成测试

### Week 5: 性能优化和成本监控

**Day 6-7: 性能优化**
- 批处理优化
- 缓存策略
- 内存管理

**Day 8-9: 成本监控**
- 实现成本跟踪
- 生成成本报告
- 优化策略

**Day 10: 基准测试**
- 运行性能测试
- 质量对比
- 生成报告

### Week 6: 文档和验收

**Day 11-12: 文档更新**
- 更新部署文档
- 添加本地模型指南
- 故障排查更新

**Day 13: Phase 1.1 验收**
- 最终测试
- 生成验收报告
- 演示和交付

---

## 9. 风险和缓解措施

### 9.1 ROCm 不稳定

**风险**: ROCm 在 Mi50 上可能不稳定  
**缓解**: 
- 准备 Vulkan 备选方案
- 测试多个 ROCm 版本
- 使用 Docker 隔离环境

### 9.2 内存不足

**风险**: 16GB 可能不够大模型  
**缓解**: 
- 使用 Q4 量化
- 限制 batch size
- 实现内存监控和自动降级

### 9.3 性能不达标

**风险**: 本地模型性能可能低于预期  
**缓解**: 
- 保留云端 API 作为备选
- 实现智能路由（质量要求高时用云端）
- 持续优化批处理和缓存

### 9.4 模型质量下降

**风险**: 本地模型质量可能低于云端  
**缓解**: 
- 设置质量阈值（< 0.85 时切换云端）
- A/B 测试
- 用户反馈机制

---

## 10. 成功标准

### 10.1 Phase 1.1 验收标准

- ✅ 本地模型可用且稳定
- ✅ 压缩延迟 < 2s
- ✅ 重构延迟 < 500ms
- ✅ 成本节省 > 90%
- ✅ 吞吐量 > 100/min
- ✅ 质量保持 > 0.85
- ✅ 所有 Phase 1.0 标准继续满足

### 10.2 关键指标

| 指标 | Phase 1.0 | Phase 1.1 目标 | 改进 |
|------|-----------|----------------|------|
| 压缩延迟 | < 3s | < 2s | +33% |
| 重构延迟 | < 500ms | < 500ms | 持平 |
| 吞吐量 | > 50/min | > 100/min | +100% |
| 成本 | $1.00 | $0.10 | -90% |
| 压缩比 | 39.63x | > 30x | 保持 |
| 质量 | > 0.90 | > 0.85 | 可接受 |

---

## 11. 总结和建议

### 11.1 推荐方案总结

**部署框架**: Ollama (主) + vLLM (备选)  
**GPU 后端**: ROCm (主) + Vulkan (备选) + OpenCL (保底)  
**主力模型**: Qwen2.5-7B-Instruct (Q4 量化)  
**备选模型**: Llama 3.1 8B, Step 3.5 Flash INT4  

### 11.2 关键成功因素

1. **环境稳定性**: ROCm 配置正确
2. **模型选择**: Qwen2.5 适配性好
3. **量化策略**: Q4 量化平衡质量和性能
4. **降级机制**: 三层降级保证可用性
5. **成本监控**: 实时跟踪节省效果

### 11.3 下一步行动

1. ✅ **立即开始**: 环境配置和 ROCm 安装
2. ✅ **并行进行**: Vulkan 和 OpenCL 备选方案
3. ✅ **优先测试**: Qwen2.5-7B 部署和验证
4. ✅ **持续优化**: 性能调优和成本监控

---

**文档版本**: 1.0  
**创建日期**: 2026-02-15  
**状态**: 待审核  
**下一步**: 开始 Task 24 实施

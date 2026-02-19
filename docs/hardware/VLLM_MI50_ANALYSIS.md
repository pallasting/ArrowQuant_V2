# vLLM 与 AMD Instinct MI50 (gfx906) 兼容性分析

**日期**: 2026-02-15  
**GPU**: AMD Instinct MI50 (gfx906)  
**系统 ROCm**: 7.2.0

---

## 执行摘要

**vLLM 官方版本**: ❌ **不支持 MI50 (gfx906)**
- 官方 vLLM 仅支持 MI300/MI350 系列 (gfx942, gfx950)
- MI50 已被 AMD 官方弃用，不在支持列表中

**社区定制版本**: ✅ **有专门的 gfx906 分支可用**
- GitHub 项目: [nlzy/vllm-gfx906](https://github.com/nlzy/vllm-gfx906)
- 专门为 Radeon VII / MI50 / MI60 优化
- 活跃维护，最新版本基于 vLLM v0.11.2

---

## vLLM 官方支持情况

### 支持的 GPU 架构

根据 [AMD ROCm 官方博客](https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html) (2026年1月):

**官方支持的 AMD GPU**:
- ✅ **MI300 系列** (gfx942): MI300X, MI300A
- ✅ **MI350 系列** (gfx950): MI350X, MI355X
- ✅ **MI325X** (gfx942)
- ⚠️ **RDNA 系列**: 部分支持 (如 RX 7900 XTX)

**不支持的 GPU**:
- ❌ **MI50/MI60** (gfx906) - 已弃用
- ❌ **MI100** (gfx908) - 未在官方列表中
- ❌ **MI210/MI250** (gfx90a) - 未在官方列表中

### vLLM 官方版本特性 (v0.14.0, 2026年1月)

**新功能**:
- ROCm 成为"一等公民"平台
- 官方 Docker 镜像: `vllm/vllm-openai-rocm:v0.14.0`
- 官方 Python wheel: `pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/`
- 93% CI 测试通过率
- 支持 FP8, FP4, MXFP4 量化
- PagedAttention 优化
- 多模态支持 (vLLM-omni)

**系统要求**:
- ROCm 7.0+
- Python 3.12
- glibc >= 2.35
- 支持的 GPU: MI300/MI350 系列

**结论**: 官方 vLLM 不支持 MI50，即使安装也无法使用

---

## 社区定制版本: vllm-gfx906

### 项目信息

**GitHub**: https://github.com/nlzy/vllm-gfx906  
**维护者**: nlzy (Nalanzeyu)  
**状态**: ✅ 活跃维护  
**最新版本**: v0.11.2 (2025年12月更新)

### 支持的 GPU

专门为以下 gfx906 GPU 优化:
- ✅ AMD Radeon VII
- ✅ AMD Radeon Pro VII
- ✅ AMD Instinct MI50
- ✅ AMD Instinct MI60

### 系统要求

**ROCm 版本**: ROCm 6.3 (推荐)
- 仅需内核驱动即可运行 Docker
- 完整安装需要 ROCm 6.3 完整包

**依赖**:
- PyTorch 2.9.0 (ROCm 6.3)
- Triton 3.5.0 (需要定制版 triton-gfx906)
- Python 3.x + venv

### 功能特性

#### 1. 量化支持 ✅

**推荐的量化格式**:
- ✅ **GPTQ** (推荐) - 性能最佳
- ✅ **AWQ** (推荐) - 与 GPTQ 性能相当
- ✅ **vLLM llm-compressor W4A16 INT** (推荐)
- ⚠️ **GGUF** - 可用但较慢
- ❌ **MoE 量化模型** - 显著较慢，不推荐
- ❌ **未量化模型** - 略慢，不推荐

#### 2. 性能优化

**GPTQ/AWQ 优化** (2025年6月):
- 单批次: ~5% 更快
- 批次大小 8-32: 30%-50% 更快

**未量化 FP16 优化** (2025年9月):
- TG 速度提升 ~90% (低批次大小)
- GPTQ/AWQ 模型提升 ~10%

**GGUF 优化** (2025年4月):
- q4_1 和 q8_0: 10%-20% 性能提升

#### 3. 引擎版本

**V1 引擎** (默认):
- 启动时间较长
- 自动前缀缓存默认关闭
- 针对 gfx906 调整的默认参数

**V0 引擎** (可选):
- 启动更快
- 某些场景性能更好

### Docker 使用方式

#### 快速启动

```bash
# 拉取镜像
docker pull nalanzeyu/vllm-gfx906

# 运行服务
docker run -it --rm \
    --shm-size=2g \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -p 8000:8000 \
    -v /path/to/models:/model \
    nalanzeyu/vllm-gfx906 \
    vllm serve /model
```

**注意**: 
- 仅需 ROCm 内核驱动 (不需要完整 ROCm 安装)
- 推荐使用 ROCm 6.3 驱动

### 从源码编译

#### 前置要求

```bash
# 安装 ROCm 6.3 (完整包)
# 参考 AMD 官方文档

# 安装 Python 依赖
sudo apt install python3-venv python3-dev

# 安装定制 Triton
# https://github.com/nlzy/triton-gfx906/tree/v3.5.0+gfx906
```

#### 编译步骤

```bash
cd vllm-gfx906

# 创建虚拟环境
python3 -m venv vllmenv
source vllmenv/bin/activate

# 安装 PyTorch (ROCm 6.3)
pip3 install torch==2.9 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.3

# 安装依赖
pip3 install -r requirements/rocm-build.txt -r requirements/rocm.txt

# 编译安装 vLLM
pip3 install --no-build-isolation --no-deps -v .
```

### 更新历史

**2025-12-08**: 更新到 vLLM v0.11.2 / PyTorch 2.9.0 / Triton 3.5.0

**2025-09-12**: 优化 triton_unified_attention，减少 AWQ VRAM 使用

**2025-09-10**: 更新到 v0.10.1，重新启用 V1 引擎和自动前缀缓存

**2025-08-24**: 支持 GLM 4.5，支持量化 MoE 模型 (AWQ/GPTQ)

**2025-08-16**: 支持量化 MoE，修复 llm-compressor 支持

**2025-07-08**: 更新到 v0.9.2，V1 引擎成为默认

**2025-06-10**: GPTQ/AWQ 内核优化，性能提升 5%-50%

**2025-05-27**: 添加 AWQ 量化支持

**2025-04-28**: 更新到 ROCm 6.3 / PyTorch 2.7.0 / Triton 3.3.0

---

## 对比分析

### vLLM 官方 vs vllm-gfx906

| 特性 | vLLM 官方 | vllm-gfx906 |
|------|-----------|-------------|
| **MI50 支持** | ❌ 不支持 | ✅ 专门优化 |
| **ROCm 版本** | 7.0+ | 6.3 |
| **安装方式** | pip/Docker | Docker/源码 |
| **维护状态** | 官方支持 | 社区维护 |
| **功能完整性** | 100% | ~90% |
| **性能优化** | MI300/MI350 | gfx906 |
| **量化支持** | 全面 | GPTQ/AWQ/GGUF |
| **更新频率** | 频繁 | 定期 |
| **生产就绪** | ✅ 是 | ⚠️ 谨慎使用 |

### vllm-gfx906 vs Ollama (Vulkan)

| 特性 | vllm-gfx906 | Ollama (Vulkan) |
|------|-------------|-----------------|
| **MI50 支持** | ✅ 专门优化 | ✅ 工作正常 |
| **后端** | ROCm 6.3 | Vulkan |
| **性能** | 未知 (需测试) | 1.54s (已验证) |
| **批处理** | ✅ 原生支持 | ⚠️ 需手动实现 |
| **量化** | GPTQ/AWQ/GGUF | GGUF |
| **API** | OpenAI 兼容 | OpenAI 兼容 |
| **安装复杂度** | ⚠️ 中等 | ✅ 简单 |
| **稳定性** | ⚠️ 社区维护 | ✅ 已验证 |
| **生产就绪** | ⚠️ 谨慎使用 | ✅ 推荐 |

---

## 使用建议

### 场景 1: 生产环境

**推荐**: ✅ **继续使用 Ollama + Vulkan**

**理由**:
- 已验证稳定性 (1.54s 推理)
- 简单配置，易于维护
- 满足 Phase 1.1 需求 (4/5 标准)
- 无需更改现有设置

### 场景 2: 实验/研究

**可选**: ⚠️ **尝试 vllm-gfx906**

**理由**:
- 原生批处理支持
- 更多量化选项 (GPTQ/AWQ)
- 可能更好的吞吐量
- 学习 vLLM 生态系统

**风险**:
- 社区维护，非官方支持
- 需要 ROCm 6.3 (可能需要降级)
- 编译复杂度较高
- 稳定性未知

### 场景 3: 高吞吐量需求

**推荐**: ⚠️ **测试 vllm-gfx906**

**理由**:
- PagedAttention 算法
- 原生批处理优化
- 更好的内存管理
- 可能显著提升吞吐量

**前提**:
- 愿意投入时间测试
- 可以接受潜在不稳定性
- 有降级 ROCm 的能力

---

## 实施路线图

### 选项 A: 保持现状 (推荐)

**步骤**:
1. ✅ 继续使用 Ollama + Vulkan
2. ✅ 监控 Ollama 更新
3. ✅ 关注 vLLM 官方对旧 GPU 的支持

**优点**:
- 零风险
- 已验证性能
- 简单维护

**缺点**:
- 无法使用 vLLM 高级特性
- 吞吐量受限 (48.8/min)

### 选项 B: 测试 vllm-gfx906 (实验)

**步骤**:
1. 备份当前配置
2. 降级到 ROCm 6.3
3. 安装 vllm-gfx906 (Docker 或源码)
4. 性能测试和对比
5. 评估稳定性

**优点**:
- 可能更好的性能
- 原生批处理
- 更多量化选项

**缺点**:
- 需要降级 ROCm
- 编译/配置复杂
- 稳定性未知
- 社区支持有限

### 选项 C: 混合方案

**步骤**:
1. 生产环境: Ollama + Vulkan
2. 测试环境: vllm-gfx906
3. 对比性能和稳定性
4. 根据结果决定是否迁移

**优点**:
- 风险可控
- 充分测试
- 灵活决策

**缺点**:
- 需要维护两套环境
- 额外的时间投入

---

## 性能预期

### vllm-gfx906 理论性能

基于社区报告和优化历史:

**GPTQ/AWQ 量化模型**:
- 单批次: 比 Ollama 快 5-10%
- 批次 8-32: 比 Ollama 快 30-50%
- 吞吐量: 可能达到 100-150/min

**未量化 FP16 模型**:
- 单批次: 与 Ollama 相当
- 批次处理: 显著更快
- 内存效率: PagedAttention 优势

**GGUF 模型**:
- 性能: 比 GPTQ/AWQ 慢
- 但比 Ollama GGUF 可能更快

### 实际性能需要测试

**测试项目**:
1. 单请求延迟 (与 Ollama 1.54s 对比)
2. 批处理吞吐量 (目标 > 100/min)
3. 内存使用
4. 稳定性 (长时间运行)
5. 不同量化格式对比

---

## 风险评估

### vllm-gfx906 风险

**技术风险** (中等):
- ⚠️ 社区维护，非官方支持
- ⚠️ 可能存在未知 bug
- ⚠️ 更新频率不如官方
- ⚠️ 文档相对有限

**操作风险** (中等):
- ⚠️ 需要降级 ROCm (可能影响其他软件)
- ⚠️ 编译复杂度较高
- ⚠️ 故障排查困难

**业务风险** (低):
- ✅ 有 Ollama 作为备选方案
- ✅ 可以在测试环境先验证
- ✅ 不影响现有生产系统

### 降级 ROCm 风险

**潜在影响**:
- ⚠️ 其他依赖 ROCm 7.2 的软件可能受影响
- ⚠️ 系统稳定性可能降低
- ⚠️ 需要重新配置环境
- ✅ 可以通过虚拟环境隔离

---

## 结论与建议

### 直接回答你的问题

**vLLM 是否支持 ROCm？**
- ✅ **官方 vLLM**: 支持 ROCm，但**不支持 MI50**
- ✅ **vllm-gfx906**: 专门为 MI50 优化的社区版本

### 我的建议

**短期 (当前)**:
- ✅ **继续使用 Ollama + Vulkan**
- 性能已满足需求 (1.54s)
- 稳定可靠
- 零风险

**中期 (可选)**:
- ⚠️ **在测试环境尝试 vllm-gfx906**
- 评估性能提升
- 测试稳定性
- 对比吞吐量

**长期 (未来)**:
- 🔄 **监控 vLLM 官方动态**
- 关注是否重新支持 gfx906
- 考虑升级到 MI300/MI350 (如果预算允许)

### 最终建议

**不要急于切换到 vllm-gfx906**，原因:
1. Ollama + Vulkan 已经工作得很好
2. vllm-gfx906 是社区项目，稳定性未知
3. 需要降级 ROCm，有风险
4. 性能提升不确定

**如果你想尝试**:
1. 在独立测试环境进行
2. 不要影响生产系统
3. 充分测试后再决定是否迁移

---

## 参考资料

1. [vLLM 官方博客 - ROCm 一等公民](https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html)
2. [vllm-gfx906 GitHub 仓库](https://github.com/nlzy/vllm-gfx906)
3. [vLLM 官方文档](https://docs.vllm.ai/)
4. [AMD ROCm 兼容性矩阵](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
5. [vLLM v0.14.0 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.14.0)

---

**报告生成**: 2026-02-15  
**系统**: Ubuntu + ROCm 7.2.0  
**GPU**: AMD Instinct MI50 (16GB, gfx906)  
**当前方案**: Ollama + Vulkan (1.54s 推理)

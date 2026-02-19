
# AI-OS Phase 9: 自进化智能体 (Self-Evolving Intelligence)

**日期**: 2026-02-19
**状态**: 设计完成，开始实现
**前置依赖**: Phase 7 (LoRA 基础设施 ✅), Phase 8 (蜂群联邦 ✅)

---

## 核心理念

AI-OS 的终极目标不是一个"有用的工具"，而是一个**自主学习、持续进化的智能体**。

当 AI-OS 遇到它无法解决的问题时，它不会停下来等待人类干预。
它会**自主地**从三个层次获取知识，并将新知识固化为本地 LoRA 技能卡片。

```
┌─────────────────────────────────────────────────────┐
│              AI-OS 自进化闭环                         │
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │  认知失调检测 (Cognitive Dissonance)          │   │
│  │  Router 置信度 < 阈值 → 触发学习循环           │   │
│  └──────────────────┬──────────────────────────┘   │
│                     │                                │
│           ┌─────────▼──────────┐                    │
│           │  三层知识获取策略     │                    │
│           └─────────┬──────────┘                    │
│                     │                                │
│  ┌──────────────────┼──────────────────────┐       │
│  │                  │                       │       │
│  ▼                  ▼                       ▼       │
│ Tier 1           Tier 2                  Tier 3     │
│ 蜂群查询          云端/互联网学习          权重蒸馏    │
│ (Federation)     (Cloud Distill)        (WeightMap)  │
│ [已实现 Phase 8]  [本阶段实现]           [本阶段实现]  │
│  │                  │                       │       │
│  └──────────────────┼──────────────────────┘       │
│                     │                                │
│           ┌─────────▼──────────┐                    │
│           │  知识内化与验证      │                    │
│           │  (LoRA Training)    │                    │
│           └─────────┬──────────┘                    │
│                     │                                │
│           ┌─────────▼──────────┐                    │
│           │  技能卡片生成        │                    │
│           │  .lora.arrow        │                    │
│           └─────────┬──────────┘                    │
│                     │                                │
│           ┌─────────▼──────────┐                    │
│           │  蜂群发布            │                    │
│           │  Federation Share   │                    │
│           └────────────────────┘                    │
└─────────────────────────────────────────────────────┘
```

---

## 三层异构节点架构

### 节点角色分类

| 角色 | 硬件特征 | 核心能力 | 蜂群角色 |
|------|---------|---------|---------|
| **叶节点 (Leaf)** | 低算力 (CPU/iGPU, <8GB RAM) | 云端蒸馏, 技能消费 | 消费者 + 学徒 |
| **枢纽节点 (Hub)** | 中算力 (GPU 8-16GB) | 本地微调, 技能生产 | 生产者 + 导师 |
| **超级节点 (Super)** | 高算力 (GPU 24GB+, 多卡) | 大模型权重蒸馏, 技能工厂 | 工厂 + 领袖 |

### Tier 1: 蜂群查询 (已实现 ✅)

```python
# Phase 8 已实现
engine.start_federation(port=9000)
engine.sync_remote_skills()
result = engine.encode_with_lora(text, intent_query="...")
# → 自动发现并热加载远程 LoRA
```

### Tier 2: 云端蒸馏 + 互联网自学 (Phase 9.1)

**适用场景**: 蜂群中没有相关技能，但节点有互联网连接。

**核心组件**: `CloudDistiller`

```python
class CloudDistiller:
    """
    从云端 API 和互联网知识库中学习，蒸馏为本地 LoRA。
    
    流程:
    1. 生成学习数据 (向云端 API 提问 + 互联网检索)
    2. 整理为训练对 (input, output)
    3. 后台微调 LoRA 适配器
    4. 验证并发布
    """
    
    def distill_from_cloud(self, query, api_provider):
        # 1. 向云端获取高质量回答
        response = api_provider.query(query)
        
        # 2. 生成训练对
        qa_pairs = self.format_training_pairs(query, response)
        
        # 3. 后台训练 LoRA
        card = self.trainer.train(qa_pairs, epochs=3)
        
        # 4. 验证
        if self.validate(card, qa_pairs):
            LoRAFormat.save(card, "new_skill.lora.arrow")
            self.federation.publish(card)
```

### Tier 3: 开源模型权重地图蒸馏 (Phase 9.2)

**适用场景**: 强算力节点，可加载大型开源模型。

**核心组件**: `WeightMapProbe` + `LoRAExtractor`

```python
class WeightMapProbe:
    """
    权重地图探测器。
    
    对大模型进行前向传播时，记录每一层的激活强度。
    识别出与特定任务相关的"热区"权重。
    """
    
    def probe(self, model_weights, test_inputs):
        # 1. 前向传播
        activations = self.forward_with_hooks(model_weights, test_inputs)
        
        # 2. 计算每层激活强度
        heat_map = {}
        for layer_name, activation in activations.items():
            # 计算 L2 范数作为激活强度
            heat_map[layer_name] = torch.norm(activation).item()
        
        # 3. 识别热区 (top-K 激活层)
        hot_zones = sorted(heat_map.items(), key=lambda x: x[1], reverse=True)
        return hot_zones[:self.top_k]


class LoRAExtractor:
    """
    从大模型的热区权重中提取 LoRA 适配器。
    
    原理: 对热区权重矩阵 W 进行 SVD 分解:
        W = U * Σ * V^T
    取前 r 个奇异值:
        W_approx ≈ (U[:, :r] * Σ[:r]) @ V[:r, :]^T
    这就是 rank-r 的 LoRA 近似:
        A = U[:, :r] * sqrt(Σ[:r])  # (d_in, r)
        B = sqrt(Σ[:r]) * V[:r, :] # (r, d_out)
    """
    
    def extract(self, weight_matrix, rank=8):
        U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)
        
        # 取前 rank 个分量
        A = U[:, :rank] * torch.sqrt(S[:rank])
        B = torch.sqrt(S[:rank]).unsqueeze(1) * Vt[:rank, :]
        
        # 计算保留的信息量
        explained_variance = S[:rank].sum() / S.sum()
        
        return A, B, explained_variance
```

### Tier 3.5: 大模型 LoRA 工厂 (Phase 9.3)

**适用场景**: 超级节点，日常运行大型开源模型。

**核心组件**: `SkillFactory`

```python
class SkillFactory:
    """
    技能工厂。在超级节点上运行，持续产出 LoRA 技能卡片。
    
    日常模式:
    - 用户使用大模型 (如 Qwen-72B) 处理日常任务
    - 后台记录所有 QA 对
    - 夜间批量训练 LoRA
    - 新技能自动发布到蜂群
    
    轮换模式:
    - 定期下载最新开源模型
    - 转换权重为 Arrow/Parquet
    - 通过 WeightMapProbe 提取技能
    - 与旧模型的技能做差异对比
    - 发布增量技能卡片
    """
    
    def nightly_distill(self):
        # 1. 收集今日 QA 对
        qa_log = self.collect_daily_qa()
        
        # 2. 按领域聚类
        clusters = self.cluster_by_domain(qa_log)
        
        # 3. 对每个领域训练 LoRA
        for domain, pairs in clusters.items():
            card = self.trainer.train(pairs, name=f"{domain}_v{self.version}")
            
            # 4. 验证
            if self.validate(card):
                self.publish(card)
    
    def model_rotation(self, new_model_path):
        # 1. 转换新模型权重
        converter.convert(new_model_path, output_format="parquet")
        
        # 2. 探测权重热区
        probe = WeightMapProbe(new_model_path)
        hot_zones = probe.probe(self.benchmark_queries)
        
        # 3. 提取新技能
        extractor = LoRAExtractor()
        for zone in hot_zones:
            card = extractor.extract_as_card(zone)
            self.publish(card)
```

---

## 实现路线图

### Phase 9.1: 核心学习循环 (本阶段)

| Task | 描述 | 状态 |
|------|------|------|
| 9.1.1 | `SkillDistiller` 基础框架 | 🔄 |
| 9.1.2 | `LoRATrainer` (权重微调核心) | 待定 |
| 9.1.3 | `CloudDistiller` (云端蒸馏接口) | 待定 |
| 9.1.4 | `QALogger` (问答对收集器) | 待定 |
| 9.1.5 | 端到端学习循环测试 | 待定 |

### Phase 9.2: 权重地图探测

| Task | 描述 | 状态 |
|------|------|------|
| 9.2.1 | `WeightMapProbe` (激活分析) | 待定 |
| 9.2.2 | `LoRAExtractor` (SVD 提取) | 待定 |
| 9.2.3 | 集成 ModelConverter (Arrow/Parquet 转换) | 待定 |
| 9.2.4 | 端到端权重蒸馏测试 | 待定 |

### Phase 9.3: 技能工厂

| Task | 描述 | 状态 |
|------|------|------|
| 9.3.1 | `SkillFactory` 编排器 | 待定 |
| 9.3.2 | 夜间批量训练调度器 | 待定 |
| 9.3.3 | 模型轮换工作流 | 待定 |

---

## 与现有架构的集成点

```
ArrowEngine (已有)
    ├── InferenceCore (推理核心 ✅)
    ├── LoRAManager (LoRA 管理 ✅)
    ├── LoRARouter (意图路由 ✅)
    ├── FederationManager (蜂群联邦 ✅)
    │
    └── SkillDistiller (NEW - Phase 9)
        ├── CognitiveTrigger (认知失调检测)
        ├── CloudDistiller (云端蒸馏)
        ├── WeightMapProbe (权重探测)
        ├── LoRAExtractor (LoRA 提取)
        ├── LoRATrainer (微调训练)
        ├── QALogger (数据收集)
        └── SkillFactory (技能工厂)
```

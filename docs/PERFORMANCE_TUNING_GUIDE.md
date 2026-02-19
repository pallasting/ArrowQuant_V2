# 性能调优指南 (Performance Tuning Guide)

本指南提供详细的性能优化策略，帮助您最大化 LLM 集成压缩系统的性能。

---

## 目录

1. [性能目标](#性能目标)
2. [本地模型优化](#本地模型优化)
3. [批量处理优化](#批量处理优化)
4. [缓存优化](#缓存优化)
5. [GPU 优化](#gpu-优化)
6. [网络优化](#网络优化)
7. [监控和诊断](#监控和诊断)
8. [常见性能问题](#常见性能问题)

---

## 性能目标

### Phase 1.0 (云端 API)

| 指标 | 目标 | 优化后 |
|------|------|--------|
| 压缩延迟 | < 5s | < 3s |
| 重构延迟 | < 1s | < 500ms |
| 吞吐量 | > 50/min | > 100/min |
| 压缩比 | > 10x | 39.63x |

### Phase 1.1 (本地模型)

| 指标 | 目标 | 优化后 |
|------|------|--------|
| 压缩延迟 | < 2s | ~1.5s |
| 重构延迟 | < 500ms | ~420ms |
| 吞吐量 | > 100/min | ~105/min |
| 成本节省 | > 80% | ~90% |

---

## 本地模型优化

### 1. GPU 后端配置

#### AMD GPU (ROCm) - 推荐

```bash
# 设置环境变量
export OLLAMA_GPU_DRIVER=rocm
export HSA_OVERRIDE_GFX_VERSION=9.0.6  # Mi50
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0

# 验证 ROCm
rocm-smi

# 启动 Ollama
ollama serve
```

**性能提升**: 3-5x 相比 CPU

#### Vulkan 后端 - 备选

```bash
# 设置环境变量
export OLLAMA_GPU_DRIVER=vulkan

# 验证 Vulkan
vulkaninfo --summary

# 启动 Ollama
ollama serve
```

**性能提升**: 2-3x 相比 CPU

#### CPU 模式 - 最后选择

```bash
# 设置环境变量
export OLLAMA_GPU_DRIVER=cpu
export OLLAMA_NUM_PARALLEL=4  # 根据 CPU 核心数

# 启动 Ollama
ollama serve
```

### 2. 模型量化选择

不同量化级别的性能对比：

| 量化类型 | 模型大小 | 推理速度 | 质量 | 推荐场景 |
|----------|----------|----------|------|----------|
| Q4_K_M | 4.7 GB | ⚡⚡⚡ 快 | ⭐⭐⭐⭐ 优秀 | 生产环境 ⭐ |
| Q5_K_M | 5.8 GB | ⚡⚡ 中等 | ⭐⭐⭐⭐⭐ 最佳 | 高质量要求 |
| Q8_0 | 7.7 GB | ⚡ 慢 | ⭐⭐⭐⭐⭐ 最佳 | 离线高质量 |
| INT4 | 3.5 GB | ⚡⚡⚡⚡ 最快 | ⭐⭐⭐ 良好 | 资源受限 |

**推荐**: Q4_K_M (最佳平衡)

```bash
# 下载推荐量化版本
ollama pull qwen2.5:7b-instruct-q4_k_m
```

### 3. Ollama 服务配置

```bash
# 高性能配置
export OLLAMA_MAX_LOADED_MODELS=2      # 同时加载模型数
export OLLAMA_NUM_PARALLEL=4           # 并行请求数
export OLLAMA_MAX_QUEUE=512            # 请求队列大小
export OLLAMA_KEEP_ALIVE=5m            # 模型保持时间

# 启动服务
ollama serve
```

**配置说明**:
- `MAX_LOADED_MODELS`: 增加可同时使用多个模型
- `NUM_PARALLEL`: 提高并发处理能力
- `MAX_QUEUE`: 处理突发请求
- `KEEP_ALIVE`: 避免频繁加载模型

---

## 批量处理优化

### 1. 批量大小调优

```yaml
# config.yaml
performance:
  batch_size: 32          # GPU: 32, CPU: 8-16
  max_concurrent: 8       # 并发任务数
```

**批量大小选择**:

| 硬件 | 推荐 batch_size | 预期吞吐量 |
|------|----------------|------------|
| AMD Mi50 (16GB) | 32 | ~105/min |
| AMD RX 6800 (16GB) | 32 | ~100/min |
| CPU (8核) | 16 | ~40/min |
| CPU (4核) | 8 | ~20/min |

### 2. 并发控制

```python
from llm_compression import BatchProcessor, PerformanceConfig

# 创建性能配置
perf_config = PerformanceConfig(
    batch_size=32,
    max_concurrent=8,
    cache_size=50000,
    cache_ttl=7200
)

# 初始化批量处理器
batch_processor = BatchProcessor(
    llm_client=llm_client,
    model_selector=model_selector,
    performance_config=perf_config
)

# 批量压缩
results = await batch_processor.compress_batch(
    texts=large_text_list,
    show_progress=True
)
```

### 3. 分组策略

系统自动将相似文本分组处理：

```python
# 自动分组（推荐）
results = await batch_processor.compress_batch(
    texts=texts,
    auto_group=True  # 自动分组相似文本
)

# 手动分组
from llm_compression import group_similar_texts

groups = group_similar_texts(texts, similarity_threshold=0.8)
for group in groups:
    results = await batch_processor.compress_batch(group)
```

**性能提升**: 20-30% 吞吐量提升

---

## 缓存优化

### 1. 缓存配置

```yaml
# config.yaml
performance:
  cache_size: 50000       # 缓存条目数（Phase 1.1）
  cache_ttl: 7200         # 2小时 TTL
```

**缓存大小建议**:

| 使用场景 | cache_size | 内存占用 |
|----------|------------|----------|
| 开发测试 | 10,000 | ~100 MB |
| 小规模生产 | 50,000 | ~500 MB |
| 大规模生产 | 100,000 | ~1 GB |

### 2. 缓存策略

系统使用 LRU (Least Recently Used) 缓存策略：

```python
from llm_compression import LLMCompressor

compressor = LLMCompressor(
    llm_client=llm_client,
    model_selector=model_selector
)

# 缓存自动工作
compressed1 = await compressor.compress(text)  # 未命中，调用 LLM
compressed2 = await compressor.compress(text)  # 命中缓存，快速返回
```

### 3. 缓存监控

```python
from llm_compression import PerformanceMonitor

monitor = PerformanceMonitor()

# 获取缓存统计
stats = monitor.get_cache_stats()
print(f"缓存命中率: {stats.hit_rate:.2%}")
print(f"缓存大小: {stats.size}/{stats.max_size}")
print(f"平均查找时间: {stats.avg_lookup_time_ms}ms")
```

**目标缓存命中率**: > 60%

### 4. 缓存预热

```python
# 预加载常用文本
common_texts = load_common_texts()

for text in common_texts:
    await compressor.compress(text)

print("缓存预热完成")
```

---

## GPU 优化

### 1. GPU 内存管理

```bash
# 监控 GPU 使用
watch -n 1 rocm-smi

# 或
watch -n 1 nvidia-smi  # NVIDIA GPU
```

**内存优化建议**:

| GPU 内存 | 推荐模型 | 批量大小 |
|----------|----------|----------|
| 4-8 GB | Gemma 3 4B | 16 |
| 8-16 GB | Qwen2.5-7B | 32 |
| 16+ GB | Llama 3.1 8B | 32-64 |

### 2. 多 GPU 支持

```bash
# 指定 GPU
export HIP_VISIBLE_DEVICES=0  # 使用第一个 GPU
export HIP_VISIBLE_DEVICES=0,1  # 使用两个 GPU

# 启动服务
ollama serve
```

### 3. GPU 降级策略

系统自动处理 GPU 内存不足：

```python
# 自动降级配置
from llm_compression import GPUFallbackStrategy

fallback = GPUFallbackStrategy(
    enable_cpu_fallback=True,      # GPU 失败时使用 CPU
    enable_quantization=True,      # 尝试更低量化
    enable_simple_compression=True # 最后使用简单压缩
)

compressor = LLMCompressor(
    llm_client=llm_client,
    model_selector=model_selector,
    fallback_strategy=fallback
)
```

---

## 网络优化

### 1. 连接池配置

```yaml
# config.yaml
llm:
  timeout: 30.0
  max_retries: 3
  rate_limit: 60  # 请求/分钟
```

### 2. 重试策略

```python
from llm_compression import RetryPolicy

# 自定义重试策略
retry_policy = RetryPolicy(
    max_retries=5,
    initial_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0
)

llm_client = LLMClient(
    endpoint=endpoint,
    retry_policy=retry_policy
)
```

### 3. 速率限制

```python
from llm_compression import RateLimiter

# 速率限制器
rate_limiter = RateLimiter(
    requests_per_minute=120,  # Phase 1.1: 更高限制
    burst_size=20
)

llm_client = LLMClient(
    endpoint=endpoint,
    rate_limiter=rate_limiter
)
```

---

## 监控和诊断

### 1. 性能监控

```python
from llm_compression import PerformanceMonitor

monitor = PerformanceMonitor()

# 记录操作
await monitor.record_compression(
    text_length=len(text),
    compression_time_ms=elapsed_ms,
    compression_ratio=ratio
)

# 获取统计
stats = monitor.get_statistics()
print(f"平均压缩延迟: {stats.avg_compression_latency_ms}ms")
print(f"P95 延迟: {stats.p95_compression_latency_ms}ms")
print(f"吞吐量: {stats.throughput_per_minute}/min")
```

### 2. 实时监控

```bash
# 启动 Prometheus 监控
python3 -m llm_compression.api --enable-prometheus

# 访问指标
curl http://localhost:9090/metrics
```

**关键指标**:
- `compression_latency_seconds`: 压缩延迟
- `reconstruction_latency_seconds`: 重构延迟
- `compression_ratio`: 压缩比
- `quality_score`: 质量分数
- `throughput_per_minute`: 吞吐量

### 3. 日志分析

```python
from llm_compression import logger

# 启用详细日志
logger.setLevel("DEBUG")

# 分析慢查询
slow_queries = logger.get_slow_queries(threshold_ms=2000)
for query in slow_queries:
    print(f"慢查询: {query.text_length} 字符, {query.elapsed_ms}ms")
```

---

## 常见性能问题

### 问题 1: 压缩延迟过高 (> 5s)

**症状**:
```
压缩延迟: 8.5s (目标 < 2s)
```

**诊断**:
```python
# 检查瓶颈
stats = monitor.get_bottleneck_analysis()
print(stats.slowest_component)  # 例如: "llm_inference"
```

**解决方案**:
1. **GPU 未启用**
   ```bash
   # 检查 GPU
   rocm-smi
   export OLLAMA_GPU_DRIVER=rocm
   ```

2. **批量大小过大**
   ```yaml
   performance:
     batch_size: 16  # 降低到 16
   ```

3. **模型过大**
   ```bash
   # 切换到更小的模型
   ollama pull gemma3:4b
   ```

### 问题 2: 吞吐量低 (< 50/min)

**症状**:
```
吞吐量: 35/min (目标 > 100/min)
```

**解决方案**:
1. **增加并发**
   ```yaml
   performance:
     max_concurrent: 8  # 增加到 8
   ```

2. **启用批量处理**
   ```python
   # 使用批量处理器
   results = await batch_processor.compress_batch(texts)
   ```

3. **优化缓存**
   ```yaml
   performance:
     cache_size: 50000  # 增加缓存
   ```

### 问题 3: 内存占用过高

**症状**:
```
内存使用: 12 GB (可用: 16 GB)
```

**解决方案**:
1. **降低缓存大小**
   ```yaml
   performance:
     cache_size: 10000  # 降低缓存
   ```

2. **使用更小的模型**
   ```bash
   ollama pull gemma3:4b  # 3.3 GB vs 4.7 GB
   ```

3. **启用模型卸载**
   ```bash
   export OLLAMA_KEEP_ALIVE=1m  # 1分钟后卸载
   ```

### 问题 4: GPU 内存不足

**症状**:
```
Error: GPU out of memory
```

**解决方案**:
1. **降低批量大小**
   ```yaml
   performance:
     batch_size: 8  # 降低到 8
   ```

2. **使用更低量化**
   ```bash
   ollama pull qwen2.5:7b-instruct-q4_k_m  # 使用 Q4
   ```

3. **启用 CPU 降级**
   ```python
   fallback = GPUFallbackStrategy(enable_cpu_fallback=True)
   ```

### 问题 5: 缓存命中率低 (< 30%)

**症状**:
```
缓存命中率: 25% (目标 > 60%)
```

**解决方案**:
1. **增加缓存大小**
   ```yaml
   performance:
     cache_size: 100000  # 增加到 100K
   ```

2. **延长 TTL**
   ```yaml
   performance:
     cache_ttl: 14400  # 4小时
   ```

3. **缓存预热**
   ```python
   # 预加载常用文本
   await preload_cache(common_texts)
   ```

---

## 性能基准测试

### 运行基准测试

```bash
# 完整基准测试
python3 scripts/benchmark_models.py

# 快速基准测试
python3 scripts/quick_benchmark.py
```

### 自定义基准测试

```python
from llm_compression import PerformanceBenchmark

benchmark = PerformanceBenchmark()

# 运行测试
results = await benchmark.run(
    models=["qwen2.5", "llama3.1", "gemma3"],
    test_sizes=[100, 500, 1000, 5000],
    iterations=10
)

# 生成报告
benchmark.generate_report(results, output="benchmark_report.md")
```

---

## 最佳实践总结

### 生产环境配置

```yaml
# config.yaml - 生产环境推荐配置

# 模型配置
model:
  prefer_local: true
  ollama_endpoint: "http://localhost:11434"
  quality_threshold: 0.85

# 性能配置
performance:
  batch_size: 32
  max_concurrent: 8
  cache_size: 50000
  cache_ttl: 7200

# 压缩配置
compression:
  min_compress_length: 100
  max_tokens: 100
  temperature: 0.3

# 监控配置
monitoring:
  enable_prometheus: true
  prometheus_port: 9090
  alert_quality_threshold: 0.85
```

### 环境变量

```bash
# GPU 配置
export OLLAMA_GPU_DRIVER=rocm
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCM_PATH=/opt/rocm

# Ollama 配置
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_QUEUE=512
export OLLAMA_KEEP_ALIVE=5m

# 性能配置
export BATCH_SIZE=32
export MAX_CONCURRENT=8
```

### 性能检查清单

- [ ] GPU 正确配置并启用
- [ ] 使用推荐的模型量化 (Q4_K_M)
- [ ] 批量大小根据硬件优化
- [ ] 缓存大小适当配置
- [ ] 启用性能监控
- [ ] 定期运行基准测试
- [ ] 监控缓存命中率
- [ ] 检查 GPU 内存使用
- [ ] 配置自动降级策略
- [ ] 启用 Prometheus 监控

---

## 获取帮助

- **性能问题**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **模型选择**: [MODEL_SELECTION_GUIDE.md](MODEL_SELECTION_GUIDE.md)
- **API 文档**: [API_REFERENCE.md](API_REFERENCE.md)

---

**性能优化是一个持续的过程。定期监控和调整配置以获得最佳性能。** 🚀

# 故障排查指南 (Troubleshooting Guide)

本指南帮助您诊断和解决 LLM 集成压缩系统的常见问题。

## 目录

- [快速诊断](#快速诊断)
- [本地模型问题 (Phase 1.1)](#本地模型问题-phase-11) ⭐ 新增
- [LLM 客户端问题](#llm-客户端问题)
- [压缩问题](#压缩问题)
- [重构问题](#重构问题)
- [存储问题](#存储问题)
- [性能问题](#性能问题)
- [配置问题](#配置问题)
- [OpenClaw 集成问题](#openclaw-集成问题)
- [日志分析](#日志分析)
- [性能调优](#性能调优)

---

## 快速诊断

### 运行健康检查

首先运行健康检查确定问题范围：

```bash
# 方法 1: 使用 Python
python3 -c "
from llm_compression.health import HealthChecker
from llm_compression.config import Config
import asyncio

async def check():
    config = Config.from_yaml('config.yaml')
    checker = HealthChecker(config=config)
    result = await checker.check_health()
    
    print(f'Overall Status: {result.overall_status}')
    print('\nComponents:')
    for name, comp in result.components.items():
        status_icon = '✅' if comp.status == 'healthy' else '⚠️' if comp.status == 'degraded' else '❌'
        print(f'  {status_icon} {name}: {comp.status} - {comp.message}')

asyncio.run(check())
"

# 方法 2: 使用 API
curl http://localhost:8000/health | python3 -m json.tool
```

### 检查日志

查看最近的错误日志：

```bash
# 查看最近 50 行日志
tail -n 50 compression.log

# 查看错误日志
grep "ERROR" compression.log | tail -n 20

# 实时监控日志
tail -f compression.log
```

### 验证配置

检查配置是否有效：

```python
from llm_compression.config import Config

try:
    config = Config.from_yaml("config.yaml")
    config.validate()
    print("✅ Configuration is valid")
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

---

## 本地模型问题 (Phase 1.1) ⭐

### 问题 1: Ollama 服务未运行

**症状**:
```
Error: Failed to connect to Ollama service
Connection refused at http://localhost:11434
```

**诊断**:
```bash
# 检查 Ollama 进程
pgrep -x ollama

# 检查 Ollama 服务状态
curl http://localhost:11434/api/tags
```

**解决方案**:
```bash
# 方法 1: 启动 Ollama 服务
ollama serve &

# 方法 2: 使用 systemd (如果已配置)
sudo systemctl start ollama

# 方法 3: 检查端口占用
lsof -i :11434
```

**验证**:
```bash
# 测试 Ollama
ollama list
ollama run qwen2.5:7b-instruct "Hello"
```

---

### 问题 2: 模型未下载

**症状**:
```
Error: Model 'qwen2.5:7b-instruct' not found
```

**诊断**:
```bash
# 列出已安装模型
ollama list
```

**解决方案**:
```bash
# 下载 Qwen2.5-7B 模型
ollama pull qwen2.5:7b-instruct

# 下载其他推荐模型
ollama pull llama3.1:8b-instruct-q4_k_m
ollama pull gemma3:4b

# 验证下载
ollama list
```

**预期输出**:
```
NAME                   ID              SIZE      MODIFIED    
qwen2.5:7b-instruct    845dbda0ea48    4.7 GB    5 hours ago
```

---

### 问题 3: GPU 未被识别

**症状**:
```
Warning: GPU not detected, using CPU
Compression latency: 8.5s (expected < 2s)
```

**诊断**:
```bash
# 检查 AMD GPU (ROCm)
rocm-smi

# 检查 Vulkan
vulkaninfo --summary

# 检查 OpenCL
clinfo

# 检查环境变量
echo $OLLAMA_GPU_DRIVER
echo $HSA_OVERRIDE_GFX_VERSION
```

**解决方案 - AMD GPU (ROCm)**:
```bash
# 1. 设置环境变量
export OLLAMA_GPU_DRIVER=rocm
export HSA_OVERRIDE_GFX_VERSION=9.0.6  # Mi50
export ROCM_PATH=/opt/rocm

# 2. 验证 ROCm
rocm-smi

# 3. 重启 Ollama
pkill ollama
ollama serve &

# 4. 测试推理
ollama run qwen2.5:7b-instruct "Test GPU"
```

**解决方案 - Vulkan 后备**:
```bash
# 使用 Vulkan 作为后备
export OLLAMA_GPU_DRIVER=vulkan

# 重启 Ollama
pkill ollama
ollama serve &
```

**验证 GPU 使用**:
```bash
# 监控 GPU 使用
watch -n 1 rocm-smi

# 运行推理时应该看到 GPU 使用率上升
```

---

### 问题 4: 推理速度慢

**症状**:
```
Compression latency: 5.2s (expected < 2s)
Using CPU instead of GPU
```

**诊断**:
```python
from llm_compression import ModelDeploymentSystem

deployment = ModelDeploymentSystem()

# 检查 GPU 状态
gpu_available = await deployment._check_gpu_available()
print(f"GPU Available: {gpu_available}")

# 检查模型信息
model_info = await deployment.get_model_info("qwen2.5:7b-instruct")
print(f"Model: {model_info.name}")
print(f"Size: {model_info.size_gb} GB")
print(f"Quantization: {model_info.quantization}")
```

**解决方案**:

1. **确保 GPU 启用**:
   ```bash
   export OLLAMA_GPU_DRIVER=rocm
   export HSA_OVERRIDE_GFX_VERSION=9.0.6
   pkill ollama && ollama serve &
   ```

2. **降低批量大小**:
   ```yaml
   # config.yaml
   performance:
     batch_size: 16  # 从 32 降低到 16
   ```

3. **使用更小的模型**:
   ```bash
   # 切换到 Gemma 3 4B (更快)
   ollama pull gemma3:4b
   ```

4. **使用更低量化**:
   ```bash
   # Q4 量化 (更快)
   ollama pull qwen2.5:7b-instruct-q4_k_m
   ```

---

### 问题 5: GPU 内存不足

**症状**:
```
Error: GPU out of memory
RuntimeError: HIP out of memory
```

**诊断**:
```bash
# 检查 GPU 内存使用
rocm-smi

# 查看详细信息
rocm-smi --showmeminfo vram
```

**解决方案**:

1. **降低批量大小**:
   ```yaml
   performance:
     batch_size: 8  # 降低到 8
     max_concurrent: 2  # 降低并发
   ```

2. **使用更小的模型**:
   ```bash
   # Gemma 3 4B (3.3 GB vs 4.7 GB)
   ollama pull gemma3:4b
   ```

3. **使用更低量化**:
   ```bash
   # INT4 量化 (最小内存)
   ollama pull qwen2.5:7b-instruct-int4
   ```

4. **启用 CPU 降级**:
   ```python
   from llm_compression import GPUFallbackStrategy
   
   fallback = GPUFallbackStrategy(
       enable_cpu_fallback=True,
       enable_quantization=True
   )
   
   compressor = LLMCompressor(
       llm_client=llm_client,
       model_selector=model_selector,
       fallback_strategy=fallback
   )
   ```

5. **清理 GPU 内存**:
   ```bash
   # 停止 Ollama
   pkill ollama
   
   # 等待几秒
   sleep 5
   
   # 重启 Ollama
   ollama serve &
   ```

---

### 问题 6: 模型加载失败

**症状**:
```
Error: Failed to load model 'qwen2.5:7b-instruct'
Model file corrupted or incomplete
```

**诊断**:
```bash
# 检查模型文件
ollama list

# 查看模型详情
ollama show qwen2.5:7b-instruct
```

**解决方案**:
```bash
# 1. 删除损坏的模型
ollama rm qwen2.5:7b-instruct

# 2. 重新下载
ollama pull qwen2.5:7b-instruct

# 3. 验证
ollama run qwen2.5:7b-instruct "Test"
```

---

### 问题 7: 本地模型质量不达标

**症状**:
```
Quality score: 0.78 (expected > 0.85)
Semantic similarity: 0.80 (expected > 0.85)
```

**诊断**:
```python
from llm_compression import QualityEvaluator

evaluator = QualityEvaluator()

# 评估质量
quality = await evaluator.evaluate(
    original_text=original,
    reconstructed_text=reconstructed,
    compressed_memory=compressed
)

print(f"Semantic Similarity: {quality.semantic_similarity}")
print(f"Entity Accuracy: {quality.entity_accuracy}")
print(f"BLEU Score: {quality.bleu_score}")
```

**解决方案**:

1. **切换到更大的模型**:
   ```bash
   # 从 Gemma 3 4B 切换到 Qwen2.5-7B
   ollama pull qwen2.5:7b-instruct
   ```

2. **使用更高量化**:
   ```bash
   # Q5 或 Q8 量化 (更高质量)
   ollama pull qwen2.5:7b-instruct-q5_k_m
   ```

3. **启用云端 API 备用**:
   ```yaml
   # config.yaml
   model:
     prefer_local: true
     quality_threshold: 0.85  # 低于此值切换到云端
   
   llm:
     cloud_endpoint: "http://localhost:8045"  # 备用
   ```

4. **调整温度参数**:
   ```yaml
   compression:
     temperature: 0.2  # 降低温度提高确定性
     max_tokens: 150  # 增加 tokens
   ```

---

### 问题 8: Ollama 配置问题

**症状**:
```
Warning: Ollama performance degraded
Low throughput: 30/min (expected > 100/min)
```

**诊断**:
```bash
# 检查 Ollama 配置
env | grep OLLAMA
```

**解决方案**:
```bash
# 优化 Ollama 配置
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_QUEUE=512
export OLLAMA_KEEP_ALIVE=5m

# 重启 Ollama
pkill ollama
ollama serve &
```

**验证**:
```bash
# 运行性能测试
python3 scripts/quick_benchmark.py
```

---

## LLM 客户端问题

### 问题 1: 连接失败

**症状**:
```
LLMAPIError: Connection refused
llm_client status: unhealthy
```

**诊断**:
```python
import aiohttp
import asyncio

async def test_connection():
    endpoint = "http://localhost:8045"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{endpoint}/health", timeout=5) as resp:
                print(f"✅ Connection successful: {resp.status}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

asyncio.run(test_connection())
```

**解决方案**:

1. **检查 LLM API 是否运行**:
```bash
# 检查端口是否监听
netstat -an | grep 8045
# 或
lsof -i :8045
```

2. **验证端点配置**:
```yaml
# config.yaml
llm:
  cloud_endpoint: "http://localhost:8045"  # 确认地址正确
```

3. **检查防火墙**:
```bash
# Linux
sudo ufw status
sudo ufw allow 8045

# macOS
sudo pfctl -sr | grep 8045
```

4. **测试网络连接**:
```bash
curl http://localhost:8045/health
```

### 问题 2: 请求超时

**症状**:
```
LLMTimeoutError: Request timeout after 30.0s
```

**诊断**:
```python
import time
import asyncio

async def measure_latency():
    from llm_compression import LLMClient
    
    client = LLMClient(endpoint="http://localhost:8045", timeout=60.0)
    
    start = time.time()
    try:
        response = await client.generate("Test prompt", max_tokens=10)
        latency = (time.time() - start) * 1000
        print(f"✅ Latency: {latency:.2f}ms")
    except Exception as e:
        print(f"❌ Error: {e}")

asyncio.run(measure_latency())
```

**解决方案**:

1. **增加超时时间**:
```yaml
llm:
  timeout: 60.0  # 增加到 60 秒
```

2. **检查 LLM API 负载**:
```bash
# 检查 CPU/内存使用
top -p $(pgrep -f llm-api)
```

3. **减少并发请求**:
```yaml
performance:
  max_concurrent: 2  # 降低并发数
```

4. **使用更快的模型**:
```python
from llm_compression import QualityLevel

model = selector.select_model(
    memory_type=MemoryType.TEXT,
    text_length=len(text),
    quality_requirement=QualityLevel.LOW  # 使用快速模型
)
```

### 问题 3: API 限流

**症状**:
```
LLMAPIError: Rate limit exceeded (429)
```

**诊断**:
```python
# 检查请求速率
metrics = llm_client.get_metrics()
print(f"Requests per minute: {metrics['requests_per_minute']}")
```

**解决方案**:

1. **降低速率限制**:
```yaml
llm:
  rate_limit: 30  # 降低到 30 请求/分钟
```

2. **启用请求队列**:
```python
from llm_compression import RateLimiter

rate_limiter = RateLimiter(requests_per_minute=30)
await rate_limiter.acquire()  # 等待许可
response = await client.generate(prompt)
```

3. **使用批量接口**:
```python
# 批量请求更高效
responses = await client.batch_generate(prompts)
```

### 问题 4: API 密钥无效

**症状**:
```
LLMAPIError: Invalid API key (401)
```

**解决方案**:

1. **设置 API 密钥**:
```bash
export LLM_CLOUD_API_KEY="your-api-key"
```

2. **或在配置文件中设置**:
```yaml
llm:
  cloud_api_key: "your-api-key"
```

3. **验证密钥**:
```python
import os
print(f"API Key: {os.getenv('LLM_CLOUD_API_KEY', 'Not set')}")
```

---

## 压缩问题

### 问题 1: 压缩比低于预期

**症状**:
```
Compression ratio: 3.5x (expected > 10x)
```

**诊断**:
```python
# 分析文本特征
text = "Your text here..."
print(f"Length: {len(text)} characters")
print(f"Unique words: {len(set(text.split()))}")
print(f"Repetition ratio: {len(text.split()) / len(set(text.split())):.2f}")
```

**解决方案**:

1. **检查文本长度**:
```python
if len(text) < 100:
    print("⚠️ Text too short for effective compression")
    # 建议: 增加 min_compress_length
```

2. **分析文本内容**:
```python
# 高度独特的文本（如代码、数据）压缩效果较差
if text.count('{') > 10 or text.count('[') > 10:
    print("⚠️ Text appears to be structured data")
    # 建议: 使用专门的代码压缩模型
```

3. **调整压缩参数**:
```yaml
compression:
  min_compress_length: 200  # 只压缩更长的文本
```

4. **使用更强的模型**:
```python
model = selector.select_model(
    memory_type=MemoryType.TEXT,
    text_length=len(text),
    quality_requirement=QualityLevel.HIGH
)
```

### 问题 2: 压缩失败

**症状**:
```
CompressionError: Failed to compress memory
```

**诊断**:
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

try:
    compressed = await compressor.compress(text)
except CompressionError as e:
    print(f"Error: {e}")
    print(f"Text length: {len(text)}")
    print(f"Text preview: {text[:100]}...")
```

**解决方案**:

1. **检查 LLM 客户端**:
```python
# 确保 LLM 客户端正常
result = await health_checker.check_health()
if result.components['llm_client'].status != 'healthy':
    print("❌ LLM client is not healthy")
```

2. **使用降级策略**:
```yaml
fallback:
  enable_simple_compression: true  # 启用简单压缩降级
```

3. **检查文本编码**:
```python
# 确保文本是有效的 UTF-8
try:
    text.encode('utf-8')
except UnicodeEncodeError:
    print("❌ Invalid UTF-8 encoding")
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
```

### 问题 3: 实体提取不准确

**症状**:
```
Entity accuracy: 0.75 (expected > 0.95)
```

**诊断**:
```python
# 检查提取的实体
entities = compressor._extract_entities(text)
print("Extracted entities:")
for entity_type, values in entities.items():
    print(f"  {entity_type}: {values}")
```

**解决方案**:

1. **使用更好的 NER 模型**:
```python
# 安装 spaCy 和模型
# pip install spacy
# python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
entities = [(ent.text, ent.label_) for ent in doc.ents]
```

2. **调整正则表达式**:
```python
# 自定义实体提取规则
import re

# 提取日期
dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
# 提取金额
amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
```

---

## 重构问题

### 问题 1: 重构质量不达标

**症状**:
```
Semantic similarity: 0.78 (expected > 0.85)
Quality score: 0.80 (below threshold)
```

**诊断**:
```python
# 比较原文和重构文本
from difflib import unified_diff

diff = unified_diff(
    original_text.split(),
    reconstructed_text.split(),
    lineterm=''
)
print("Differences:")
for line in diff:
    print(line)
```

**解决方案**:

1. **使用更高质量的模型**:
```python
model = selector.select_model(
    memory_type=MemoryType.TEXT,
    text_length=len(text),
    quality_requirement=QualityLevel.HIGH
)
```

2. **调整 LLM 参数**:
```python
response = await client.generate(
    prompt=prompt,
    max_tokens=200,  # 增加 token 数
    temperature=0.1  # 降低温度提高确定性
)
```

3. **检查 diff 数据**:
```python
# 确保 diff 数据完整
if len(compressed.diff_data) == 0:
    print("⚠️ No diff data, reconstruction may be inaccurate")
```

4. **降低质量阈值**（临时方案）:
```yaml
compression:
  quality_threshold: 0.80  # 降低阈值
```

### 问题 2: 重构失败

**症状**:
```
ReconstructionError: Failed to reconstruct memory
```

**诊断**:
```python
# 检查压缩数据完整性
print(f"Summary hash: {compressed.summary_hash}")
print(f"Entities: {compressed.entities}")
print(f"Diff data size: {len(compressed.diff_data)} bytes")
```

**解决方案**:

1. **启用部分重构**:
```yaml
fallback:
  enable_partial_reconstruction: true
```

2. **检查 LLM 可用性**:
```python
try:
    response = await client.generate("Test", max_tokens=10)
    print("✅ LLM is available")
except Exception as e:
    print(f"❌ LLM unavailable: {e}")
    # 使用降级策略
```

3. **验证压缩数据**:
```python
# 尝试解压 diff 数据
import zstd
try:
    diff_text = zstd.decompress(compressed.diff_data).decode('utf-8')
    print(f"✅ Diff data is valid: {len(diff_text)} bytes")
except Exception as e:
    print(f"❌ Diff data corrupted: {e}")
```

### 问题 3: 重构延迟过高

**症状**:
```
Reconstruction latency: 3500ms (expected < 1000ms)
```

**诊断**:
```python
import time

start = time.time()
reconstructed = await reconstructor.reconstruct(compressed)
latency = (time.time() - start) * 1000
print(f"Latency: {latency:.2f}ms")
```

**解决方案**:

1. **使用更快的模型**:
```python
model = selector.select_model(
    memory_type=MemoryType.TEXT,
    text_length=len(text),
    quality_requirement=QualityLevel.LOW  # 快速模型
)
```

2. **减少 max_tokens**:
```python
response = await client.generate(
    prompt=prompt,
    max_tokens=100  # 减少生成长度
)
```

3. **启用缓存**:
```yaml
performance:
  cache_size: 10000  # 启用缓存
```

4. **使用批量重构**:
```python
# 批量重构更高效
reconstructed_list = await reconstructor.reconstruct_batch(compressed_list)
```

---

## 存储问题

### 问题 1: 存储路径不存在

**症状**:
```
StorageError: Storage path does not exist: ~/.ai-os/memory/
```

**解决方案**:

```bash
# 创建存储目录
mkdir -p ~/.ai-os/memory/{core,working,long-term,shared}

# 设置权限
chmod 755 ~/.ai-os/memory
```

### 问题 2: 磁盘空间不足

**症状**:
```
StorageError: No space left on device
storage status: degraded (free space: 0.5GB)
```

**诊断**:
```bash
# 检查磁盘空间
df -h ~/.ai-os/memory/

# 检查存储使用
du -sh ~/.ai-os/memory/*
```

**解决方案**:

1. **清理旧数据**:
```bash
# 删除旧的归档数据
rm -rf ~/.ai-os/memory/long-term/archived_*.arrow
```

2. **压缩现有数据**:
```python
# 批量压缩未压缩的记忆
# 参考 OpenClaw 集成指南的迁移部分
```

3. **增加磁盘空间**:
```bash
# 扩展分区或挂载新磁盘
```

### 问题 3: Arrow 文件损坏

**症状**:
```
ArrowInvalid: Invalid Parquet file
```

**诊断**:
```python
import pyarrow.parquet as pq

try:
    table = pq.read_table("~/.ai-os/memory/core/experiences.arrow")
    print(f"✅ File is valid: {len(table)} rows")
except Exception as e:
    print(f"❌ File corrupted: {e}")
```

**解决方案**:

1. **从备份恢复**:
```bash
cp ~/.ai-os/memory/backup/experiences.arrow ~/.ai-os/memory/core/
```

2. **重建索引**:
```python
from llm_compression import ArrowStorage

storage = ArrowStorage(storage_path="~/.ai-os/memory/")
storage.rebuild_index(category="experiences")
```

3. **导出和重新导入**:
```python
# 导出为 JSON
table = pq.read_table("corrupted.arrow")
data = table.to_pydict()
import json
with open("backup.json", "w") as f:
    json.dump(data, f)

# 重新导入
# ... 重新创建 Arrow 表
```

---

## 性能问题

### 问题 1: 吞吐量低

**症状**:
```
Throughput: 20 memories/min (expected > 100/min)
```

**诊断**:
```python
# 测量吞吐量
import time

start = time.time()
count = 0
for text in texts:
    await compressor.compress(text)
    count += 1

elapsed = time.time() - start
throughput = (count / elapsed) * 60
print(f"Throughput: {throughput:.2f} memories/min")
```

**解决方案**:

1. **使用批量处理**:
```python
# 批量处理更高效
compressed_list = await batch_processor.compress_batch(texts)
```

2. **增加并发数**:
```yaml
performance:
  max_concurrent: 8  # 增加并发
  batch_size: 32     # 增加批量大小
```

3. **使用更快的模型**:
```python
model = selector.select_model(
    quality_requirement=QualityLevel.LOW
)
```

4. **启用缓存**:
```yaml
performance:
  cache_size: 50000  # 增大缓存
```

### 问题 2: 内存使用过高

**症状**:
```
Memory usage: 8GB (system has 8GB total)
```

**诊断**:
```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {memory_mb:.2f} MB")
```

**解决方案**:

1. **减少缓存大小**:
```yaml
performance:
  cache_size: 1000  # 减小缓存
```

2. **减少批量大小**:
```yaml
performance:
  batch_size: 8  # 减小批量
```

3. **使用 float16**:
```yaml
storage:
  use_float16: true  # 减少 embedding 内存
```

4. **定期清理缓存**:
```python
# 手动清理缓存
compressor.cache.clear()
```

### 问题 3: CPU 使用率高

**症状**:
```
CPU usage: 100% on all cores
```

**诊断**:
```bash
# 检查 CPU 使用
top -p $(pgrep -f python)
```

**解决方案**:

1. **减少并发数**:
```yaml
performance:
  max_concurrent: 2  # 降低并发
```

2. **使用异步 I/O**:
```python
# 确保使用 async/await
await compressor.compress(text)  # 正确
# compressor.compress(text)  # 错误（阻塞）
```

3. **优化正则表达式**:
```python
# 预编译正则表达式
import re
DATE_PATTERN = re.compile(r'\d{4}-\d{2}-\d{2}')
dates = DATE_PATTERN.findall(text)
```

---

## 配置问题

### 问题 1: 配置文件无效

**症状**:
```
ConfigValidationError: Invalid configuration
```

**诊断**:
```python
from llm_compression.config import Config

try:
    config = Config.from_yaml("config.yaml")
    config.validate()
except Exception as e:
    print(f"Configuration error: {e}")
```

**解决方案**:

1. **检查 YAML 语法**:
```bash
# 使用 yamllint 检查
pip install yamllint
yamllint config.yaml
```

2. **验证必需字段**:
```yaml
# 确保包含所有必需字段
llm:
  cloud_endpoint: "http://localhost:8045"  # 必需
storage:
  storage_path: "~/.ai-os/memory/"         # 必需
```

3. **检查数据类型**:
```yaml
# 确保类型正确
llm:
  timeout: 30.0      # float，不是 "30.0"
  max_retries: 3     # int，不是 "3"
```

### 问题 2: 环境变量未生效

**症状**:
```
Using default endpoint instead of environment variable
```

**诊断**:
```python
import os
print(f"LLM_CLOUD_ENDPOINT: {os.getenv('LLM_CLOUD_ENDPOINT', 'Not set')}")
```

**解决方案**:

1. **正确设置环境变量**:
```bash
# Linux/Mac
export LLM_CLOUD_ENDPOINT="http://localhost:8045"

# Windows
set LLM_CLOUD_ENDPOINT=http://localhost:8045
```

2. **在 Python 中设置**:
```python
import os
os.environ['LLM_CLOUD_ENDPOINT'] = "http://localhost:8045"
```

3. **使用 .env 文件**:
```bash
# 创建 .env 文件
echo "LLM_CLOUD_ENDPOINT=http://localhost:8045" > .env

# 加载 .env
pip install python-dotenv
```

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## OpenClaw 集成问题

### 问题 1: Schema 不兼容

**症状**:
```
ArrowInvalid: Schema mismatch
```

**解决方案**:

参考 [OpenClaw 集成指南](OPENCLAW_INTEGRATION.md) 的 Schema 兼容性部分。

### 问题 2: 检索记忆返回空

**症状**:
```
KeyError: 'context'
```

**诊断**:
```python
# 检查记忆是否存在
memory = await interface.retrieve_memory(memory_id)
print(f"Memory fields: {memory.keys()}")
```

**解决方案**:

1. **检查记忆 ID**:
```python
# 确认 ID 正确
print(f"Memory ID: {memory_id}")
```

2. **检查类别**:
```python
# 确认类别正确
memory = await interface.retrieve_memory(
    memory_id=memory_id,
    memory_category="experiences"  # 确认类别
)
```

---

## 日志分析

### 启用详细日志

```python
import logging

# 设置日志级别
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compression_debug.log'),
        logging.StreamHandler()
    ]
)
```

### 常见日志模式

**正常压缩**:
```
INFO - Compressing memory (length: 500 chars)
INFO - Selected model: gpt-4
INFO - Compression successful (ratio: 39.63x, quality: 0.92)
```

**压缩失败**:
```
ERROR - Compression failed: LLMAPIError
WARNING - Falling back to simple compression
INFO - Stored uncompressed memory
```

**重构警告**:
```
WARNING - Reconstruction quality below threshold: 0.82
WARNING - Entity accuracy: 0.90 (expected > 0.95)
```

---

## 性能调优

### 优化配置

```yaml
# 高性能配置
performance:
  batch_size: 32
  max_concurrent: 8
  cache_size: 50000

# 高质量配置
compression:
  quality_threshold: 0.90
  min_compress_length: 200

# 平衡配置
llm:
  timeout: 45.0
  max_retries: 3
  rate_limit: 100
```

### 监控指标

```python
# 获取性能统计
stats = monitor.get_statistics()

print(f"Compression:")
print(f"  Avg ratio: {stats['compression_ratio']['mean']:.2f}x")
print(f"  P95 latency: {stats['compression_latency']['p95']:.2f}ms")

print(f"Reconstruction:")
print(f"  Avg latency: {stats['reconstruction_latency']['mean']:.2f}ms")
print(f"  P99 latency: {stats['reconstruction_latency']['p99']:.2f}ms")
```

---

## 获取帮助

如果问题仍未解决：

1. **查看文档**:
   - [快速开始指南](QUICK_START.md)
   - [API 参考文档](API_REFERENCE.md)
   - [OpenClaw 集成指南](OPENCLAW_INTEGRATION.md)

2. **查看示例**:
   - [examples/](../examples/)
   - [notebooks/](../notebooks/)

3. **提交 Issue**:
   - [GitHub Issues](https://github.com/ai-os/llm-compression/issues)
   - 包含：错误信息、配置文件、日志片段

4. **社区讨论**:
   - [GitHub Discussions](https://github.com/ai-os/llm-compression/discussions)

---

**版本**: Phase 1.0  
**最后更新**: 2024  
**测试覆盖率**: 87.6%

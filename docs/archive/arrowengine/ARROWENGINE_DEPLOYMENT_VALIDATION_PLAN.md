# ArrowEngine 硬件环境部署验证计划

**目标**: 验证 ArrowEngine 在当前 Windows 环境下的实际性能和部署可行性

**预计时间**: 1-2 小时

---

## 验证步骤

### Step 1: 环境检查（5 分钟）

**检查项**:
1. Python 版本 (需要 3.10+)
2. PyTorch 安装状态
3. PyArrow 安装状态
4. 可用内存
5. CPU 信息

**执行命令**:
```bash
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pyarrow; print(f'PyArrow: {pyarrow.__version__}')"
wmic OS get FreePhysicalMemory
wmic cpu get name
```

---

### Step 2: 模型文件检查（5 分钟）

**检查项**:
1. 模型目录是否存在
2. 必需文件完整性
3. 模型大小

**执行命令**:
```bash
# 检查模型目录
dir models\minilm

# 应该包含:
# - metadata.json
# - weights.parquet
# - tokenizer/ 或 tokenizer.json
```

**预期结果**:
- metadata.json: ~1-5 KB
- weights.parquet: ~20-50 MB (float16 优化后)
- tokenizer/: ~500 KB

---

### Step 3: 基础功能测试（10 分钟）

**测试 1: 模型加载速度**

```python
# test_load_speed.py
import time
from llm_compression.inference.arrow_engine import ArrowEngine

# 测试加载时间
start = time.time()
engine = ArrowEngine("./models/minilm")
load_time_ms = (time.time() - start) * 1000

print(f"✅ 模型加载时间: {load_time_ms:.2f}ms")
print(f"   目标: < 100ms")
print(f"   状态: {'通过' if load_time_ms < 100 else '需优化'}")
print(f"   嵌入维度: {engine.get_embedding_dimension()}")
print(f"   设备: {engine.device}")
```

**测试 2: 单次推理延迟**

```python
# test_inference_latency.py
import time
import numpy as np
from llm_compression.inference.arrow_engine import ArrowEngine

engine = ArrowEngine("./models/minilm")

# 预热
for _ in range(5):
    engine.encode("warmup")

# 测试延迟
latencies = []
test_text = "This is a test sentence for latency measurement."

for _ in range(100):
    start = time.time()
    embedding = engine.encode(test_text)
    latencies.append((time.time() - start) * 1000)

median_latency = np.median(latencies)
p95_latency = np.percentile(latencies, 95)

print(f"✅ 推理延迟统计:")
print(f"   中位数: {median_latency:.2f}ms")
print(f"   P95: {p95_latency:.2f}ms")
print(f"   目标: < 5ms")
print(f"   状态: {'通过' if median_latency < 5 else '需优化'}")
```

**测试 3: 批量吞吐量**

```python
# test_batch_throughput.py
import time
from llm_compression.inference.arrow_engine import ArrowEngine

engine = ArrowEngine("./models/minilm")

# 测试批量处理
batch_size = 32
num_batches = 100
texts = ["Test sentence for throughput measurement."] * batch_size

start = time.time()
for _ in range(num_batches):
    embeddings = engine.encode(texts)
elapsed = time.time() - start

throughput = (batch_size * num_batches) / elapsed

print(f"✅ 批量吞吐量:")
print(f"   吞吐量: {throughput:.0f} 请求/秒")
print(f"   目标: > 2000 请求/秒")
print(f"   状态: {'通过' if throughput > 2000 else '需优化'}")
```

**测试 4: 内存占用**

```python
# test_memory_usage.py
import psutil
import os
from llm_compression.inference.arrow_engine import ArrowEngine

process = psutil.Process(os.getpid())

# 基线内存
baseline_mb = process.memory_info().rss / (1024 * 1024)

# 加载模型
engine = ArrowEngine("./models/minilm")

# 执行推理
for _ in range(10):
    engine.encode("Test sentence")

# 测量内存
current_mb = process.memory_info().rss / (1024 * 1024)
model_memory_mb = current_mb - baseline_mb

print(f"✅ 内存占用:")
print(f"   基线: {baseline_mb:.1f} MB")
print(f"   当前: {current_mb:.1f} MB")
print(f"   模型占用: {model_memory_mb:.1f} MB")
print(f"   目标: < 100 MB")
print(f"   状态: {'通过' if model_memory_mb < 100 else '需优化'}")
```

---

### Step 4: 精度验证（15 分钟）

**测试 5: 与 sentence-transformers 对比**

```python
# test_precision_validation.py
import numpy as np
from llm_compression.inference.arrow_engine import ArrowEngine
from sentence_transformers import SentenceTransformer

# 测试文本
test_texts = [
    "Artificial intelligence is transforming technology.",
    "Machine learning enables computers to learn from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand text.",
    "Computer vision allows machines to interpret images.",
]

# 加载两个引擎
arrow_engine = ArrowEngine("./models/minilm")
st_model = SentenceTransformer("all-MiniLM-L6-v2")

# 编码
arrow_embs = arrow_engine.encode(test_texts, normalize=True)
st_embs = st_model.encode(test_texts, normalize_embeddings=True)

# 计算相似度
similarities = []
for i in range(len(test_texts)):
    sim = np.dot(arrow_embs[i], st_embs[i])
    similarities.append(sim)
    print(f"   文本 {i+1}: {sim:.6f}")

avg_sim = np.mean(similarities)
min_sim = np.min(similarities)

print(f"\n✅ 精度验证:")
print(f"   平均相似度: {avg_sim:.6f}")
print(f"   最小相似度: {min_sim:.6f}")
print(f"   目标: ≥ 0.99")
print(f"   状态: {'通过' if min_sim >= 0.99 else '失败'}")
```

---

### Step 5: EmbeddingProvider 接口测试（10 分钟）

**测试 6: 统一接口验证**

```python
# test_embedding_provider.py
from llm_compression.embedding_provider import get_default_provider

# 获取默认提供者
provider = get_default_provider()

print(f"✅ EmbeddingProvider 接口测试:")
print(f"   提供者类型: {type(provider).__name__}")
print(f"   嵌入维度: {provider.dimension}")

# 测试单文本编码
text = "Hello, World!"
embedding = provider.encode(text)
print(f"   单文本编码: {embedding.shape}")

# 测试批量编码
texts = ["First sentence", "Second sentence", "Third sentence"]
embeddings = provider.encode_batch(texts)
print(f"   批量编码: {embeddings.shape}")

# 测试相似度计算
sim = provider.similarity(embedding, embedding)
print(f"   自相似度: {sim:.6f} (应该 ≈ 1.0)")

print(f"   状态: {'通过' if abs(sim - 1.0) < 0.01 else '失败'}")
```

---

### Step 6: 集成测试（15 分钟）

**测试 7: 与 ArrowStorage 集成**

```python
# test_arrow_storage_integration.py
import numpy as np
from datetime import datetime
from llm_compression.embedding_provider import get_default_provider
from llm_compression.arrow_storage import ArrowStorage
from llm_compression.stored_memory import CompressedMemory

# 初始化
provider = get_default_provider()
storage = ArrowStorage(storage_dir="./test_storage")

# 创建测试记忆
texts = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "Natural language processing handles text.",
]

memories = []
for i, text in enumerate(texts):
    embedding = provider.encode(text)
    memory = CompressedMemory(
        memory_id=f"test_{i}",
        timestamp=datetime.now(),
        context=text,
        summary=text,
        is_compressed=False,
        category="test",
        embedding=embedding,
    )
    memories.append(memory)

# 保存到存储
for memory in memories:
    storage.save(memory)

print(f"✅ ArrowStorage 集成测试:")
print(f"   保存记忆数: {len(memories)}")

# 测试相似度查询
query_text = "What is machine learning?"
query_embedding = provider.encode(query_text)

results = storage.query_by_similarity(
    category="test",
    query_embedding=query_embedding,
    top_k=3,
)

print(f"   查询结果数: {len(results)}")
for i, result in enumerate(results):
    print(f"   结果 {i+1}: 相似度 {result['similarity']:.4f}")

print(f"   状态: {'通过' if len(results) > 0 else '失败'}")
```

---

## 验收标准

### 必须通过（P0）
- [ ] 模型加载时间 < 500ms（Windows 环境可能比 Linux 慢）
- [ ] 推理延迟中位数 < 10ms
- [ ] 内存占用 < 150MB
- [ ] 精度最小相似度 ≥ 0.99
- [ ] EmbeddingProvider 接口正常工作
- [ ] ArrowStorage 集成正常

### 应该通过（P1）
- [ ] 模型加载时间 < 100ms
- [ ] 推理延迟中位数 < 5ms
- [ ] 批量吞吐量 > 2000 rps
- [ ] 内存占用 < 100MB
- [ ] 精度平均相似度 ≥ 0.995

---

## 故障排查

### 问题 1: 模型文件不存在

**症状**: `FileNotFoundError: Model path not found`

**解决方案**:
```bash
# 检查模型是否已转换
python -m llm_compression.tools.cli convert \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output ./models/minilm
```

### 问题 2: 加载时间过长

**可能原因**:
- 磁盘 I/O 慢（HDD vs SSD）
- 防病毒软件扫描
- 首次加载（缓存未建立）

**解决方案**:
- 使用 SSD 存储模型
- 添加模型目录到防病毒白名单
- 多次测试取平均值

### 问题 3: 推理延迟高

**可能原因**:
- CPU 性能不足
- 批量大小不合适
- 内存交换

**解决方案**:
- 检查 CPU 使用率
- 调整 batch_size
- 增加可用内存

### 问题 4: 精度不匹配

**可能原因**:
- 模型转换错误
- 权重损坏
- 配置不匹配

**解决方案**:
- 重新转换模型
- 验证权重完整性
- 检查 metadata.json

---

## 执行脚本

创建一个统一的验证脚本：

```python
# run_validation.py
import sys
import subprocess

tests = [
    ("环境检查", "test_environment.py"),
    ("模型加载", "test_load_speed.py"),
    ("推理延迟", "test_inference_latency.py"),
    ("批量吞吐", "test_batch_throughput.py"),
    ("内存占用", "test_memory_usage.py"),
    ("精度验证", "test_precision_validation.py"),
    ("接口测试", "test_embedding_provider.py"),
    ("集成测试", "test_arrow_storage_integration.py"),
]

print("=" * 60)
print("ArrowEngine 硬件环境部署验证")
print("=" * 60)

passed = 0
failed = 0

for name, script in tests:
    print(f"\n{'=' * 60}")
    print(f"测试: {name}")
    print(f"{'=' * 60}")
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print(f"✅ {name} - 通过")
            passed += 1
        else:
            print(f"❌ {name} - 失败")
            print(result.stderr)
            failed += 1
    
    except subprocess.TimeoutExpired:
        print(f"⏱️ {name} - 超时")
        failed += 1
    except Exception as e:
        print(f"❌ {name} - 错误: {e}")
        failed += 1

print(f"\n{'=' * 60}")
print(f"验证完成")
print(f"{'=' * 60}")
print(f"通过: {passed}/{len(tests)}")
print(f"失败: {failed}/{len(tests)}")
print(f"成功率: {passed/len(tests)*100:.1f}%")

if failed == 0:
    print("\n✅ 所有测试通过！ArrowEngine 已准备好集成到 AI-OS Memory 系统。")
    sys.exit(0)
else:
    print(f"\n❌ {failed} 个测试失败，请检查故障排查部分。")
    sys.exit(1)
```

---

## 下一步

### 如果验证通过 ✅
继续执行 **阶段 2: AI-OS Memory 系统集成**

### 如果验证失败 ❌
1. 查看故障排查部分
2. 根据具体错误调整配置
3. 重新运行验证
4. 如需帮助，提供详细错误日志

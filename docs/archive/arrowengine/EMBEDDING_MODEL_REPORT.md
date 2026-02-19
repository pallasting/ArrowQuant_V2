# 向量化模型配置报告

**生成时间**: 2026-02-17  
**查询**: 用户输入到向量化转换使用的模型及硬件要求

---

## 📊 当前配置

### 使用的模型

**模型名称**: `sentence-transformers/all-MiniLM-L6-v2`

**位置**:
- `llm_compression/compressor.py` (第 147 行)
- `llm_compression/quality_evaluator.py` (第 54 行)

**模型参数**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device='cpu'  # 强制使用 CPU（AMD ROCm 兼容性）
)
```

### 模型特性

| 特性 | 值 |
|------|-----|
| **向量维度** | 384 |
| **最大序列长度** | 256 tokens |
| **模型大小** | ~80 MB |
| **参数量** | ~22M |
| **语言支持** | 英文为主 |
| **推理速度** | 快（~1000 句/秒 on CPU） |

---

## 💻 硬件要求

### 最低配置

| 组件 | 要求 |
|------|------|
| **CPU** | 2 核心 |
| **内存** | 2 GB RAM |
| **存储** | 500 MB（模型 + 依赖） |
| **GPU** | 不需要（CPU 模式） |

### 推荐配置

| 组件 | 要求 |
|------|------|
| **CPU** | 4+ 核心 |
| **内存** | 4 GB RAM |
| **存储** | 1 GB |
| **GPU** | 可选（CUDA/ROCm） |

### 当前部署配置

**运行模式**: CPU only
- **原因**: AMD ROCm 兼容性问题
- **性能**: 足够快（embedding 计算 < 100ms）
- **首次加载**: ~8 秒（模型下载 + 初始化）

---

## 🔧 技术细节

### 1. 懒加载机制

```python
@property
def embedding_model(self):
    """Lazy load embedding model"""
    if self._embedding_model is None:
        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像
        
        from sentence_transformers import SentenceTransformer
        self._embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device='cpu'
        )
    return self._embedding_model
```

**优势**:
- 只在需要时加载（节省内存）
- 首次压缩时自动初始化
- 可选预热（`prewarm_embedding=True`）

### 2. 向量化流程

```
用户输入 (文本)
    ↓
Tokenization (分词)
    ↓
BERT Encoding (编码)
    ↓
Mean Pooling (池化)
    ↓
Normalization (归一化)
    ↓
384-dim Vector (向量)
```

### 3. 存储优化

**向量存储格式**: `float16`（半精度）
- 原始: 384 × 4 bytes = 1536 bytes
- 优化: 384 × 2 bytes = **768 bytes**（节省 50%）

```python
# arrow_storage.py
('embedding', pa.list_(pa.float16()))  # 使用 float16
```

---

## 🌍 多语言支持

### 当前模型限制

`all-MiniLM-L6-v2` 主要针对英文优化，中文支持有限。

### 推荐替代方案

#### 方案 1: 多语言模型（推荐）

```python
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

**特性**:
- 支持 50+ 语言（包括中文）
- 维度: 384（无需修改代码）
- 速度: 略慢（~800 句/秒）
- 大小: ~120 MB

#### 方案 2: 中文专用模型

```python
embedding_model = "BAAI/bge-small-zh-v1.5"
```

**特性**:
- 专为中文优化
- 维度: 512（需修改代码）
- 中文场景性能最佳
- 大小: ~100 MB

#### 方案 3: 高性能模型

```python
embedding_model = "BAAI/bge-large-zh-v1.5"
```

**特性**:
- 最高精度
- 维度: 1024
- 速度: 慢（~100 句/秒）
- 大小: ~1.3 GB
- **硬件要求**: 8 GB RAM

---

## ⚡ 性能基准

### CPU 模式（当前配置）

| 操作 | 延迟 |
|------|------|
| 模型加载（首次） | ~8 秒 |
| 单句向量化 | ~10-50 ms |
| 批量向量化（32 句） | ~200 ms |
| 相似度计算 | < 1 ms |

### GPU 模式（如果可用）

| 操作 | 延迟 |
|------|------|
| 模型加载 | ~2 秒 |
| 单句向量化 | ~5 ms |
| 批量向量化（32 句） | ~20 ms |
| 相似度计算 | < 0.1 ms |

**注**: 当前系统因 AMD ROCm 兼容性问题使用 CPU 模式。

---

## 🔄 如何更换模型

### 步骤 1: 修改配置

编辑 `llm_compression/quality_evaluator.py`:

```python
def __init__(
    self,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 改这里
    ...
):
```

编辑 `llm_compression/compressor.py`:

```python
self._embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 改这里
    device='cpu'
)
```

### 步骤 2: 清除缓存

```bash
rm -rf ~/.cache/huggingface/hub/models--sentence-transformers*
```

### 步骤 3: 重启系统

```bash
# 模型会在首次使用时自动下载
python your_script.py
```

---

## 📦 依赖项

### Python 包

```txt
sentence-transformers>=2.2.0
torch>=2.0.0
transformers>=4.30.0
```

### 安装命令

```bash
pip install sentence-transformers torch
```

### 国内镜像加速

系统已配置 HuggingFace 镜像：
```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

---

## 🎯 使用场景

### 1. 语义相似度计算

```python
# quality_evaluator.py
similarity = cosine_similarity(
    embedding_original,
    embedding_reconstructed
)
```

### 2. 记忆检索

```python
# openclaw_interface.py
def search_memories(query: str, top_k: int = 5):
    query_embedding = compute_embedding(query)
    # 使用余弦相似度检索最相关的记忆
```

### 3. 去重检测

```python
# batch_processor.py
def deduplicate(texts: List[str]):
    embeddings = [compute_embedding(t) for t in texts]
    # 基于 embedding 相似度去重
```

---

## 💡 优化建议

### 1. 预热模型（已实现）

```python
compressor = LLMCompressor(
    llm_client=client,
    model_selector=selector,
    prewarm_embedding=True  # 启动时加载模型
)
```

### 2. 批量处理

```python
# 批量计算 embedding（更快）
embeddings = model.encode(texts, batch_size=32)
```

### 3. 缓存 Embedding

```python
# 缓存常用文本的 embedding
embedding_cache = {}
if text in embedding_cache:
    return embedding_cache[text]
```

---

## 📊 总结

| 项目 | 值 |
|------|-----|
| **当前模型** | all-MiniLM-L6-v2 |
| **向量维度** | 384 |
| **运行模式** | CPU only |
| **内存需求** | 2-4 GB |
| **推理速度** | 10-50 ms/句 |
| **语言支持** | 英文为主 |
| **建议升级** | paraphrase-multilingual-MiniLM-L12-v2（多语言） |

**结论**: 当前配置适合英文场景，硬件要求低。如需中文支持，建议切换到多语言或中文专用模型。

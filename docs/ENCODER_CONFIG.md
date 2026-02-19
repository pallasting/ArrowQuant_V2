# 编码器配置指南

## 当前配置 ✅

**模型**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 维度: 384
- 语言: 50+ 语言（包括中文、英文等）
- 速度: 中等（~800 句/秒 on CPU）
- 大小: ~120 MB

**更新时间**: 2026-02-17

## 其他可选方案

### 1. 英文专用（更快）

```python
# llm_compression/quality_evaluator.py 第52行
embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
```

**优势**:
- 英文场景最快
- 维度: 384
- 大小: ~80 MB

**劣势**:
- 仅支持英文

### 2. 高精度中文

```python
embedding_model: str = "BAAI/bge-small-zh-v1.5"
```

**优势**:
- 专为中文优化
- 维度: 512
- 性能: 中文场景下最佳

**劣势**:
- 需要修改代码以支持512维

### 3. 最强性能（如果资源充足）

```python
embedding_model: str = "BAAI/bge-large-zh-v1.5"
```

**优势**:
- 最高精度
- 维度: 1024

**劣势**:
- 慢（~10x）
- 内存占用大

## 修改方法

### 方法 1: 修改默认值（全局）

```bash
# 编辑文件
vim llm_compression/quality_evaluator.py

# 第52行改为：
embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### 方法 2: 运行时指定（灵活）

```python
# examples/chat_agent.py
from llm_compression.quality_evaluator import QualityEvaluator

evaluator = QualityEvaluator(
    embedding_model="BAAI/bge-small-zh-v1.5"
)
```

## 性能对比

| 模型 | 维度 | 中文 | 英文 | 速度 | 内存 |
|------|------|------|------|------|------|
| all-MiniLM-L6-v2 | 384 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 快 | 90MB |
| paraphrase-multilingual | 384 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 | 120MB |
| bge-small-zh | 512 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 中 | 200MB |
| bge-large-zh | 1024 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 慢 | 1.3GB |

## 测试新模型

```bash
# 1. 安装（如果需要）
pip install sentence-transformers

# 2. 测试
python3 << 'EOF'
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embedding = model.encode("量子纠缠是什么？")
print(f"维度: {len(embedding)}")
print(f"前5个值: {embedding[:5]}")
EOF
```

## 我的建议

**对于你的中文使用场景**:

1. **立即**: 改用 `paraphrase-multilingual-MiniLM-L12-v2`
   - 零代码修改
   - 中文效果提升明显
   
2. **如果不满意**: 升级到 `bge-small-zh-v1.5`
   - 需要修改维度处理代码
   - 中文效果最佳

3. **保持当前**: 如果主要用英文或性能优先

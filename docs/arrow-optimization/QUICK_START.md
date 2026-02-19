# Arrow Optimization - Quick Start Guide

## Overview

This guide helps you get started with the Arrow-optimized embedding system.

## Prerequisites

```bash
# Python 3.10+
python --version

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-arrow.txt
```

## Quick Start

### 1. Convert a Model

```python
from llm_compression.tools.model_converter import convert_model

# Convert to Arrow format
result = convert_model(
    "sentence-transformers/all-MiniLM-L6-v2",
    output_dir="models/optimized"
)

print(f"✅ Converted: {result.file_size_mb:.2f} MB")
```

### 2. Run Inference

```python
from llm_compression.inference.arrow_engine import ArrowEmbeddingEngine

# Load model (zero-copy)
engine = ArrowEmbeddingEngine(
    model_path="models/optimized/all-MiniLM-L6-v2.parquet",
    tokenizer_path="models/optimized/tokenizer"
)

# Generate embedding
embedding = engine.encode("Hello, world!")
print(embedding.shape)  # (384,)
```

### 3. Start API Server

```bash
# Using Docker
docker-compose -f deployment/docker/docker-compose.yml up -d

# Or directly
uvicorn llm_compression.inference.server:app --port 8080
```

### 4. Use HTTP API

```bash
# Embed texts
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "Machine learning"],
    "normalize": true
  }'

# Calculate similarity
curl -X POST http://localhost:8080/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "cat",
    "text2": "kitten"
  }'
```

### 5. Integrate with AI-OS

```python
from llm_compression.tools.embedding_tool import EmbeddingTool

# Initialize tool
tool = EmbeddingTool(config={
    "model_path": "models/optimized/all-MiniLM-L6-v2.parquet"
})

# Execute action
result = tool.execute(
    action="search",
    params={
        "query": "machine learning",
        "corpus": documents,
        "top_k": 5
    }
)
```

## Performance Expectations

| Metric | Target | Typical |
|--------|--------|---------|
| Startup | < 100ms | ~50ms |
| Load Time | < 50ms | ~20ms |
| Latency (p50) | < 5ms | ~3ms |
| Latency (p99) | < 15ms | ~10ms |
| Throughput | > 2000 req/s | ~3000 req/s |
| Memory | < 200MB | ~150MB |

## Next Steps

1. **Read Documentation**:
   - [Architecture](ARCHITECTURE.md)
   - [Roadmap](ROADMAP.md)
   - [Tasks](TASKS.md)

2. **Explore Examples**:
   - `examples/arrow-optimization/quick_start.py`

3. **Run Benchmarks**:
   - `benchmarks/arrow_benchmarks.py`

4. **Follow Implementation**:
   - Start with Phase 1, Task T1.1
   - Follow the 6-week roadmap

## Troubleshooting

### Issue: "Module not found"
```bash
pip install -r requirements-arrow.txt
```

### Issue: "Model not found"
```bash
# Convert the model first
python -m llm_compression.tools.convert_model <model-name>
```

### Issue: "Port already in use"
```bash
# Change port
uvicorn llm_compression.inference.server:app --port 8081
```

## Support

- Documentation: `docs/arrow-optimization/`
- Examples: `examples/arrow-optimization/`
- Issues: GitHub Issues (when available)

---

**Status**: Implementation in progress
**Phase**: Phase 1 (Foundation)
**Next Milestone**: M1 (End of Week 2)

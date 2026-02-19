# ArrowEngine Embedding Tool - Quick Start Guide

## Overview

The ArrowEngine Embedding Tool provides a high-performance, production-ready embedding service for AI-OS memory compression and LLM workflows. It combines Arrow-optimized inference with intelligent caching and batching.

## Features

- **High Performance**: 20-50x faster model loading, 2-4x faster inference
- **Zero-Copy Architecture**: Arrow/Parquet memory-mapped model loading
- **Intelligent Caching**: LRU cache for frequently-used embeddings
- **Automatic Batching**: Efficient processing of large text collections
- **Production-Ready**: Docker deployment, health checks, monitoring

## Quick Start

### 1. Start the Service (Docker)

```bash
# One-command deployment
docker-compose up -d

# Verify service is running
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true,"device":"cpu"}
```

### 2. Install Python Client

```bash
pip install -e .
```

### 3. Basic Usage

```python
from llm_compression.tools import EmbeddingTool

# Initialize tool
tool = EmbeddingTool(endpoint="http://localhost:8000")

# Check service health
if not tool.health_check():
    raise RuntimeError("Service unavailable")

# Generate embeddings
result = tool.embed(["Hello, world!", "AI is amazing!"])
print(f"Embeddings shape: {result.embeddings.shape}")  # (2, 384)
print(f"Dimension: {result.dimension}")  # 384

# Compute similarity
similarity = tool.similarity("AI", "Machine Learning")
print(f"Similarity: {similarity:.4f}")  # 0.8234
```

## Common Use Cases

### 1. AI-OS Memory Compression

```python
from llm_compression.tools import EmbeddingTool
import numpy as np

tool = EmbeddingTool()

# Embed conversation history
conversations = [
    "User: What is AI?",
    "Assistant: AI is artificial intelligence...",
    "User: Tell me about deep learning",
    "Assistant: Deep learning uses neural networks..."
]

result = tool.embed(conversations)

# Find similar conversations
query = "What is machine learning?"
query_emb = tool.embed([query]).embeddings[0]

similarities = np.dot(result.embeddings, query_emb)
most_similar_idx = np.argmax(similarities)
print(f"Most similar: {conversations[most_similar_idx]}")
```

### 2. Semantic Search

```python
from llm_compression.tools import EmbeddingTool
import numpy as np

tool = EmbeddingTool()

# Embed document collection
documents = [
    "Python is a programming language",
    "Machine learning uses neural networks",
    "Natural language processing analyzes text",
    "Computer vision processes images"
]

doc_embeddings = tool.embed(documents).embeddings

# Search query
query = "How do I analyze text data?"
query_emb = tool.embed([query]).embeddings[0]

# Compute similarities
similarities = np.dot(doc_embeddings, query_emb)

# Get top 3 results
top_indices = np.argsort(similarities)[-3:][::-1]
for idx in top_indices:
    print(f"{similarities[idx]:.4f} - {documents[idx]}")
```

### 3. Batch Processing with Caching

```python
from llm_compression.tools import EmbeddingTool

tool = EmbeddingTool()

# Process large collection (automatic batching)
texts = ["Document text..."] * 1000

result = tool.embed(texts, batch_size=64)
print(f"Processed {result.metadata['total_batches']} batches")

# Repeated queries use cache (instant!)
result2 = tool.embed(texts[:10])
print(f"Cache hits: {result2.cache_hits}")  # 10
print(f"Cache misses: {result2.cache_misses}")  # 0
```

### 4. Similarity Matrix for Clustering

```python
from llm_compression.tools import EmbeddingTool
import numpy as np

tool = EmbeddingTool()

# Compute pairwise similarities
texts = [
    "AI and machine learning",
    "Deep learning neural networks",
    "Natural language processing",
    "Computer vision applications"
]

# Get similarity matrix (4x4)
matrix = tool.similarity_matrix(texts, texts)

# Use for clustering, deduplication, etc.
print(matrix)
# [[1.00, 0.85, 0.72, 0.63],
#  [0.85, 1.00, 0.68, 0.59],
#  [0.72, 0.68, 1.00, 0.55],
#  [0.63, 0.59, 0.55, 1.00]]
```

## Configuration

### Using YAML Configuration

```python
from llm_compression.tools.config import load_config
from llm_compression.tools import EmbeddingTool

# Load configuration
config = load_config("config/embedding_tool.yaml")

# Create tool with config
tool = EmbeddingTool(config=config)
```

### Environment-Specific Configurations

**Development** (`config/embedding_tool.development.yaml`):
```yaml
service:
  endpoint: "http://localhost:8000"
  timeout: 10.0

batching:
  batch_size: 8

logging:
  level: "DEBUG"
```

**Production** (`config/embedding_tool.production.yaml`):
```yaml
service:
  endpoint: "https://embeddings.production.com"
  timeout: 60.0
  max_retries: 5

batching:
  batch_size: 128
  normalize: true

cache:
  max_size: 10000
```

**Load by environment**:
```python
import os
from llm_compression.tools.config import load_config
from llm_compression.tools import EmbeddingTool

env = os.getenv("DEPLOY_ENV", "development")
config = load_config(f"config/embedding_tool.{env}.yaml")
tool = EmbeddingTool(config=config)
```

### Programmatic Configuration

```python
from llm_compression.tools import EmbeddingTool, EmbeddingConfig

# Create custom config
config = EmbeddingConfig(
    endpoint="http://localhost:8000",
    timeout=60.0,
    max_retries=5,
    batch_size=64,
    normalize=True,
    enable_cache=True,
    cache_size=2000
)

# Use config
tool = EmbeddingTool(config=config)
```

## Advanced Features

### Cache Management

```python
from llm_compression.tools import EmbeddingTool

tool = EmbeddingTool()

# Get cache statistics
stats = tool.get_cache_stats()
print(f"Cache: {stats['size']}/{stats['capacity']} entries")

# Clear cache
tool.clear_cache()
```

### Context Manager

```python
from llm_compression.tools import EmbeddingTool

# Automatic resource cleanup
with EmbeddingTool() as tool:
    result = tool.embed(["Text"])
# Connection automatically closed
```

### Model Information

```python
from llm_compression.tools import EmbeddingTool

tool = EmbeddingTool()

# Get model metadata
info = tool.get_info()
print(f"Model: {info['model_name']}")
print(f"Dimension: {info['embedding_dimension']}")
print(f"Max sequence length: {info['max_seq_length']}")
print(f"Version: {info['version']}")
print(f"Device: {info['device']}")
```

### Normalization

```python
from llm_compression.tools import EmbeddingTool
import numpy as np

tool = EmbeddingTool()

# L2 normalization for cosine similarity
result = tool.embed(["Text"], normalize=True)

# Verify unit vector
norm = np.linalg.norm(result.embeddings[0])
print(f"Norm: {norm:.6f}")  # 1.000000
```

## Deployment

### Docker Deployment

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Scale workers (future feature)
docker-compose up -d --scale arrowengine-api=3

# Stop service
docker-compose down
```

### Health Monitoring

```python
from llm_compression.tools import EmbeddingTool
import time

tool = EmbeddingTool()

# Health check loop
while True:
    if tool.health_check():
        print("Service healthy")
    else:
        print("Service unhealthy - alerting...")
    time.sleep(30)
```

### Production Checklist

- [ ] Configure production endpoint in YAML
- [ ] Set appropriate timeout values (60s recommended)
- [ ] Enable cache with appropriate size (10000+ for production)
- [ ] Configure max_retries (3-5 recommended)
- [ ] Set batch_size based on memory (64-128 recommended)
- [ ] Enable normalization if using cosine similarity
- [ ] Set up health check monitoring
- [ ] Configure resource limits in docker-compose.yml
- [ ] Set up logging and metrics collection

## Performance Tips

1. **Batch Processing**: Use `batch_size=64` or higher for better throughput
2. **Enable Caching**: Keep `enable_cache=True` for repeated queries
3. **Normalize Once**: Enable `normalize=True` in config rather than per-request
4. **Increase Cache Size**: Set `cache_size=10000+` for large workloads
5. **Use Context Manager**: Proper connection cleanup with `with EmbeddingTool() as tool:`

## Troubleshooting

### Service Not Responding

```python
from llm_compression.tools import EmbeddingTool

tool = EmbeddingTool(endpoint="http://localhost:8000")

if not tool.health_check():
    print("Service unavailable!")
    print("Check:")
    print("1. Is docker-compose running? (docker-compose ps)")
    print("2. Is port 8000 accessible? (curl http://localhost:8000/health)")
    print("3. Check logs: docker-compose logs arrowengine-api")
```

### Connection Timeout

```python
from llm_compression.tools import EmbeddingConfig, EmbeddingTool

# Increase timeout for slow networks
config = EmbeddingConfig(
    endpoint="http://localhost:8000",
    timeout=120.0,  # 2 minutes
    max_retries=5
)

tool = EmbeddingTool(config=config)
```

### Empty Results

```python
from llm_compression.tools import EmbeddingTool

tool = EmbeddingTool()

# Check model info
info = tool.get_info()
if info['embedding_dimension'] == 0:
    print("Model not loaded correctly")
else:
    print(f"Model OK - {info['embedding_dimension']}D embeddings")
```

## Next Steps

- Read the [API Reference](API_REFERENCE.md) for detailed documentation
- See [Configuration Guide](CONFIGURATION.md) for all config options
- Check [Deployment Guide](DEPLOYMENT.md) for production setup
- Review [Performance Tuning](PERFORMANCE_TUNING.md) for optimization tips

## Support

- GitHub Issues: https://github.com/ai-os/llm-compression/issues
- Documentation: https://github.com/ai-os/llm-compression/docs

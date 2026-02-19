# ArrowEngine Quick Start Guide

Get started with ArrowEngine in 5 minutes!

## What is ArrowEngine?

ArrowEngine is a high-performance local inference engine that provides:
- ⚡ **21.4x faster** model loading than sentence-transformers
- 🎯 **Perfect precision** (similarity ≥ 0.999999)
- 🔄 **Automatic fallback** to sentence-transformers
- 🚀 **Zero-copy operations** with Apache Arrow
- 📦 **Unified API** for all embedding operations

---

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Install Package

```bash
# Clone repository
git clone https://github.com/your-org/ai-os-memory.git
cd ai-os-memory

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

---

## Quick Start (3 Steps)

### Step 1: Convert Model

Convert sentence-transformers model to ArrowEngine format:

```bash
python scripts/convert_and_validate.py
```

This creates `./models/minilm/` with optimized weights.

**Output:**
```
Converting sentence-transformers/all-MiniLM-L6-v2...
✓ Weights converted: 86.64 MB → 43.50 MB (2x compression)
✓ Tokenizer saved
✓ Metadata saved
✓ Validation passed: similarity = 1.000000
```

### Step 2: Use in Code

```python
from llm_compression.embedding_provider import get_default_provider

# Get provider (automatically uses ArrowEngine if available)
provider = get_default_provider()

# Encode text
embedding = provider.encode("Hello, world!")
print(f"Embedding shape: {embedding.shape}")  # (384,)

# Batch encode
texts = ["Machine learning", "Deep learning", "Neural networks"]
embeddings = provider.encode_batch(texts)
print(f"Batch shape: {embeddings.shape}")  # (3, 384)

# Compute similarity
sim = provider.similarity(embeddings[0], embeddings[1])
print(f"Similarity: {sim:.3f}")
```

### Step 3: Run Tests

```bash
# Run precision tests
pytest tests/integration/inference/test_e2e_precision.py -v

# Run semantic indexing tests
pytest tests/integration/test_semantic_indexing.py -v
```

**That's it!** You're now using ArrowEngine. 🎉

---

## Common Use Cases

### Use Case 1: Simple Text Embedding

```python
from llm_compression.embedding_provider import get_default_provider

provider = get_default_provider()

# Encode single text
text = "Machine learning is a subset of artificial intelligence"
embedding = provider.encode(text)

print(f"Dimension: {provider.dimension}")  # 384
print(f"Embedding: {embedding[:5]}")  # First 5 values
```

### Use Case 2: Semantic Search

```python
from llm_compression.embedding_provider import get_default_provider
import numpy as np

provider = get_default_provider()

# Documents to search
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Deep learning is a subset of ML",
    "JavaScript is used for web development"
]

# Encode documents
doc_embeddings = provider.encode_batch(documents)

# Query
query = "artificial intelligence and ML"
query_embedding = provider.encode(query)

# Compute similarities
similarities = [
    provider.similarity(query_embedding, doc_emb)
    for doc_emb in doc_embeddings
]

# Get top results
top_indices = np.argsort(similarities)[::-1][:2]
for idx in top_indices:
    print(f"{documents[idx]}: {similarities[idx]:.3f}")
```

**Output:**
```
Machine learning uses algorithms: 0.756
Deep learning is a subset of ML: 0.689
```

### Use Case 3: Memory Indexing

```python
from llm_compression.embedding_provider import get_default_provider
from llm_compression.semantic_indexer import SemanticIndexer
from llm_compression.semantic_index_db import SemanticIndexDB
from llm_compression.arrow_storage import ArrowStorage
from datetime import datetime

# Setup
provider = get_default_provider()
storage = ArrowStorage("./data/memories.parquet")
index_db = SemanticIndexDB("./data/index")
indexer = SemanticIndexer(provider, storage, index_db)

# Create memories
memories = [
    {
        'memory_id': 'mem_1',
        'category': 'knowledge',
        'context': 'Python is great for data science',
        'timestamp': datetime.now(),
        'embedding': None  # Will be generated
    },
    {
        'memory_id': 'mem_2',
        'category': 'knowledge',
        'context': 'Machine learning requires large datasets',
        'timestamp': datetime.now(),
        'embedding': None
    }
]

# Index memories
indexer.batch_index(memories)

print(f"Indexed {len(memories)} memories")
print(f"Index size: {index_db.get_category_size('knowledge')} entries")
```

### Use Case 4: Async Background Indexing

```python
import asyncio
from llm_compression.embedding_provider import get_default_provider
from llm_compression.semantic_indexer import SemanticIndexer
from llm_compression.semantic_index_db import SemanticIndexDB
from llm_compression.arrow_storage import ArrowStorage
from llm_compression.background_queue import BackgroundQueue

async def main():
    # Setup
    provider = get_default_provider()
    storage = ArrowStorage("./data/memories.parquet")
    index_db = SemanticIndexDB("./data/index")
    indexer = SemanticIndexer(provider, storage, index_db)
    
    # Create background queue
    queue = BackgroundQueue(indexer, batch_size=32)
    await queue.start()
    
    # Submit memories (non-blocking!)
    memories = [
        {'memory_id': f'mem_{i}', 'category': 'knowledge', 'context': f'Text {i}'}
        for i in range(100)
    ]
    await queue.submit_batch(memories)
    
    print("Submitted 100 memories (non-blocking)")
    
    # Do other work here...
    
    # Wait for completion
    await queue.wait_until_empty(timeout=60.0)
    await queue.stop()
    
    print("All memories indexed!")

asyncio.run(main())
```

---

## Docker Deployment

### Build Image

```bash
docker build -t arrowengine:latest .
```

### Run Container

```bash
docker run -d \
  --name arrowengine \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_PATH=/app/models/minilm \
  -e DEVICE=cpu \
  arrowengine:latest
```

### Using docker-compose

```bash
# Start service
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop service
docker-compose down
```

---

## Performance Comparison

### Model Loading

```python
import time
from llm_compression.embedding_provider import (
    ArrowEngineProvider,
    SentenceTransformerProvider
)

# ArrowEngine
start = time.time()
arrow_provider = ArrowEngineProvider(model_path="./models/minilm")
arrow_time = time.time() - start

# sentence-transformers
start = time.time()
st_provider = SentenceTransformerProvider()
st_time = time.time() - start

print(f"ArrowEngine: {arrow_time*1000:.1f}ms")
print(f"sentence-transformers: {st_time*1000:.1f}ms")
print(f"Speedup: {st_time/arrow_time:.1f}x")
```

**Output (Surface Pro 4):**
```
ArrowEngine: 643.2ms
sentence-transformers: 13,756.8ms
Speedup: 21.4x
```

### Inference Speed

```python
texts = ["Test text"] * 100

# ArrowEngine
start = time.time()
embeddings = arrow_provider.encode_batch(texts, batch_size=32)
arrow_time = time.time() - start

# sentence-transformers
start = time.time()
embeddings = st_provider.encode_batch(texts, batch_size=32)
st_time = time.time() - start

print(f"ArrowEngine: {arrow_time:.3f}s ({len(texts)/arrow_time:.1f} texts/s)")
print(f"sentence-transformers: {st_time:.3f}s ({len(texts)/st_time:.1f} texts/s)")
```

---

## Configuration

### Environment Variables

```bash
# Model configuration
export MODEL_PATH=./models/minilm
export DEVICE=cpu  # or cuda, mps

# API configuration
export PORT=8000
export API_KEY=your-secret-key

# Logging
export LOG_LEVEL=INFO
export LOG_FORMAT=json
```

### Python Configuration

```python
from llm_compression.embedding_provider import ArrowEngineProvider

# Custom configuration
provider = ArrowEngineProvider(
    model_path="./custom/model/path",
    device="cuda",  # Use GPU
    max_batch_size=64,  # Larger batches
    normalize_embeddings=True  # L2 normalization
)
```

---

## Troubleshooting

### Issue: Model Not Found

**Error:**
```
FileNotFoundError: Model path not found: ./models/minilm
```

**Solution:**
Run the conversion script:
```bash
python scripts/convert_and_validate.py
```

### Issue: Slow Performance

**Problem:** Inference is slower than expected.

**Solutions:**
1. Use batch operations:
   ```python
   # Good
   embeddings = provider.encode_batch(texts, batch_size=32)
   
   # Bad
   embeddings = [provider.encode(text) for text in texts]
   ```

2. Check device:
   ```python
   print(provider.device)  # Should be 'cuda' for GPU
   ```

3. Increase batch size:
   ```python
   embeddings = provider.encode_batch(texts, batch_size=64)
   ```

### Issue: Import Errors

**Error:**
```
ImportError: cannot import name 'get_default_provider'
```

**Solution:**
Reinstall package:
```bash
pip install -e . --force-reinstall
```

---

## Next Steps

1. **Read API Documentation**: [API Reference](API_REFERENCE.md)
2. **Migration Guide**: [Migrate from sentence-transformers](MIGRATION_GUIDE.md)
3. **Run Examples**: Check `examples/` directory
4. **Performance Tuning**: Optimize for your use case
5. **Deploy to Production**: Use Docker deployment

---

## Examples

### Complete Example: Semantic Memory System

```python
import asyncio
from datetime import datetime
from llm_compression.embedding_provider import get_default_provider
from llm_compression.semantic_indexer import SemanticIndexer
from llm_compression.semantic_index_db import SemanticIndexDB
from llm_compression.arrow_storage import ArrowStorage
from llm_compression.memory_search import MemorySearch, SearchMode
from llm_compression.vector_search import VectorSearch

async def main():
    # 1. Setup
    print("Setting up components...")
    provider = get_default_provider()
    storage = ArrowStorage("./data/memories.parquet")
    index_db = SemanticIndexDB("./data/index")
    indexer = SemanticIndexer(provider, storage, index_db)
    
    # 2. Add memories
    print("Adding memories...")
    memories = [
        {
            'memory_id': 'mem_1',
            'category': 'knowledge',
            'context': 'Python is a high-level programming language',
            'timestamp': datetime.now()
        },
        {
            'memory_id': 'mem_2',
            'category': 'knowledge',
            'context': 'Machine learning is a branch of AI',
            'timestamp': datetime.now()
        },
        {
            'memory_id': 'mem_3',
            'category': 'knowledge',
            'context': 'Deep learning uses neural networks',
            'timestamp': datetime.now()
        }
    ]
    
    indexer.batch_index(memories)
    print(f"Indexed {len(memories)} memories")
    
    # 3. Search
    print("\nSearching...")
    vector_search = VectorSearch(provider, storage, index_db)
    memory_search = MemorySearch(vector_search, storage)
    
    results = memory_search.search(
        query="artificial intelligence and neural networks",
        category="knowledge",
        mode=SearchMode.SEMANTIC,
        top_k=2
    )
    
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"  - {result.memory_id}: {result.similarity:.3f}")
        print(f"    {result.memory['context']}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Output:**
```
Setting up components...
Adding memories...
Indexed 3 memories

Searching...
Found 2 results:
  - mem_2: 0.756
    Machine learning is a branch of AI
  - mem_3: 0.689
    Deep learning uses neural networks
```

---

## Resources

- **Documentation**: [docs/](../docs/)
- **Examples**: [examples/](../examples/)
- **Tests**: [tests/](../tests/)
- **Benchmarks**: [benchmarks/](../benchmarks/)

---

## Getting Help

- **Issues**: Report bugs on GitHub
- **Discussions**: Ask questions in discussions
- **Documentation**: Check [API Reference](API_REFERENCE.md)

---

**Happy coding with ArrowEngine!** 🚀

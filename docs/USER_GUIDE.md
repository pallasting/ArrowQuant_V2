# User Guide

**Version**: 2.0  
**Last Updated**: 2026-02-17

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)
8. [FAQ](#faq)

---

## Introduction

### What is LLM Compression System?

The LLM Compression System is a production-ready memory compression and retrieval system for AI applications. It uses Large Language Models to achieve 10-50x compression ratios while maintaining high semantic fidelity.

### Key Benefits

- **High Compression**: 10-50x compression ratios
- **Fast Retrieval**: <10ms semantic search
- **Quality Preservation**: >0.85 semantic similarity
- **Cost Efficient**: <$1/day API costs
- **Self-Learning**: Cognitive memory network

### Use Cases

1. **Conversational AI**: Compress chat history while maintaining context
2. **Knowledge Management**: Store and retrieve large document collections
3. **Memory Systems**: Long-term memory for AI agents
4. **Data Archival**: Compress logs and historical data

---

## Getting Started

### Prerequisites

**System Requirements**:
- Python 3.10 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB disk space
- Internet connection (for LLM services)

**Optional Services**:
- Ollama (for local LLM inference)
- Docker (for containerized deployment)

### Installation

#### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/ai-os-memory.git
cd ai-os-memory
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install in editable mode (for development)
pip install -e .
```

#### Step 4: Verify Installation

```bash
# Run tests
pytest tests/unit/ -v

# Check version
python -c "import llm_compression; print(llm_compression.__version__)"
```

### Quick Start

#### Option 1: Interactive Chat Agent

```bash
python examples/chat_agent_optimized.py
```

This launches an interactive chat session with memory capabilities.

#### Option 2: Python Script

```python
import asyncio
from llm_compression import (
    LLMClient,
    LLMCompressor,
    ModelSelector
)

async def main():
    # Initialize components
    client = LLMClient()
    selector = ModelSelector()
    compressor = LLMCompressor(client, selector)
    
    # Compress text
    text = "Python is a high-level programming language..."
    compressed = compressor.compress(text, mode="balanced")
    
    print(f"Original size: {len(text)} bytes")
    print(f"Compressed size: {compressed.compressed_size} bytes")
    print(f"Compression ratio: {compressed.compression_ratio}x")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Basic Usage

### 1. Text Compression

#### Simple Compression

```python
from llm_compression import LLMClient, LLMCompressor, ModelSelector

# Initialize
client = LLMClient()
selector = ModelSelector()
compressor = LLMCompressor(client, selector)

# Compress
text = "Your long text here..."
compressed = compressor.compress(text, mode="balanced")

# Access results
print(f"Summary: {compressed.summary}")
print(f"Entities: {compressed.entities}")
print(f"Ratio: {compressed.compression_ratio}x")
```

#### Compression Modes

```python
# Fast mode (< 3 seconds, lower quality)
compressed_fast = compressor.compress(text, mode="fast")

# Balanced mode (< 10 seconds, good quality)
compressed_balanced = compressor.compress(text, mode="balanced")

# High quality mode (< 20 seconds, best quality)
compressed_high = compressor.compress(text, mode="high")
```

### 2. Text Reconstruction

```python
from llm_compression import LLMReconstructor

# Initialize reconstructor
reconstructor = LLMReconstructor(client)

# Reconstruct
reconstructed = reconstructor.reconstruct(compressed)

# Access results
print(f"Reconstructed text: {reconstructed.text}")
print(f"Quality score: {reconstructed.quality_score}")
```

### 3. Memory Storage

#### Save Memories

```python
from llm_compression import ArrowStorage
import pyarrow as pa

# Create storage
storage = ArrowStorage()

# Create memory table
memory_table = pa.table({
    'memory_id': ['mem1', 'mem2'],
    'content': ['text1', 'text2'],
    'embedding': [embedding1, embedding2]
})

# Save to file
storage.save(memory_table, "memories.parquet")
```

#### Load Memories

```python
# Load from file
memory_table = storage.load("memories.parquet")

# Access data
memory_ids = memory_table['memory_id'].to_pylist()
contents = memory_table['content'].to_pylist()
```

### 4. Semantic Search

```python
from llm_compression.embedder import LocalEmbedder
from llm_compression.vector_search import VectorSearch

# Initialize
embedder = LocalEmbedder()
search = VectorSearch()

# Add memories
for memory_id, content in zip(memory_ids, contents):
    embedding = embedder.encode(content)
    search.add_memory(memory_id, embedding)

# Search
query = "What is Python?"
query_embedding = embedder.encode(query)
results = search.search(query_embedding, top_k=5)

# Access results
for result in results:
    print(f"Memory: {result.memory_id}, Score: {result.score}")
```

---

## Advanced Features

### 1. Conversational Agent

#### Basic Setup

```python
from llm_compression import (
    ConversationalAgent,
    CognitiveLoop,
    LLMClient,
    LLMCompressor,
    ModelSelector
)

# Initialize components
client = LLMClient()
selector = ModelSelector()
compressor = LLMCompressor(client, selector)

# Create cognitive loop
from llm_compression.expression_layer import MultiModalExpressor
from llm_compression.internal_feedback import InternalFeedbackSystem
from llm_compression.connection_learner import ConnectionLearner
from llm_compression.network_navigator import NetworkNavigator
from llm_compression.reconstructor import LLMReconstructor

reconstructor = LLMReconstructor(llm_client=client)
expressor = MultiModalExpressor(client, reconstructor)
feedback = InternalFeedbackSystem()
learner = ConnectionLearner()
navigator = NetworkNavigator()

cognitive_loop = CognitiveLoop(
    expressor=expressor,
    feedback=feedback,
    learner=learner,
    navigator=navigator
)

# Create agent
agent = ConversationalAgent(
    llm_client=client,
    compressor=compressor,
    cognitive_loop=cognitive_loop,
    user_id="user_001"
)

# Chat
response = await agent.chat("Hello!")
print(response.message)
```

#### With Personalization

```python
from llm_compression import PersonalizationEngine

# Create personalization engine
personalization = PersonalizationEngine(user_id="user_001")

# Create agent with personalization
agent = ConversationalAgent(
    llm_client=client,
    compressor=compressor,
    cognitive_loop=cognitive_loop,
    user_id="user_001",
    personalization_engine=personalization
)

# Chat (responses adapt to user preferences)
response = await agent.chat("Tell me about Python")
```

### 2. Batch Processing

```python
from llm_compression import BatchProcessor

# Initialize
processor = BatchProcessor(
    compressor=compressor,
    batch_size=100,
    max_workers=4
)

# Process batch
texts = ["text1", "text2", "text3", ...]
results = processor.process_batch(texts)

# Access results
for result in results:
    print(f"Compressed: {result.compressed_size} bytes")
```

### 3. Cost Monitoring

```python
from llm_compression.cost_monitor import CostMonitor

# Initialize monitor
monitor = CostMonitor(log_file="cost_log.jsonl")

# Record operations (automatic in production)
monitor.record_operation(
    operation_type="compression",
    model="gpt-4",
    tokens=1500,
    cost=0.045
)

# Get summary
summary = monitor.get_summary()
print(f"Total cost: ${summary['total_cost']:.2f}")
print(f"Operations: {summary['total_operations']}")

# Generate report
report = monitor.generate_report(output_file="cost_report.txt")
```

### 4. Performance Optimization

#### Enable Arrow Zero-Copy

```python
from llm_compression.cognitive_loop_arrow import CognitiveLoopArrow

# Create optimized cognitive loop
loop_arrow = CognitiveLoopArrow(
    enable_optimizations=True,
    adaptive_threshold=1000,
    batch_size=100,
    max_workers=4
)

# Process with zero-copy optimization
result = await loop_arrow.process_arrow(
    query="What is machine learning?",
    max_memories=10
)
```

#### Pre-load Models

```python
from llm_compression.embedder_cache import preload_default_model

# Pre-load embedding model at startup
preload_default_model()

# Now embeddings are instant
embedder = LocalEmbedder()
embedding = embedder.encode("text")  # Fast!
```

### 5. Model Quantization

```python
from llm_compression.inference.arrow_quantizer import ArrowQuantizer
import torch

# Load model
model = torch.load("model.pth")

# Create quantizer
quantizer = ArrowQuantizer(bits=8, strategy="per_channel")

# Quantize weights
quantized_table = quantizer.quantize_weights(model.state_dict())

# Save quantized model
import pyarrow.parquet as pq
pq.write_table(quantized_table, "model_quantized.parquet")

# Load and dequantize
quantized_table = pq.read_table("model_quantized.parquet")
weights = quantizer.dequantize_weights(quantized_table)
model.load_state_dict(weights)
```


---

## Configuration

### Configuration File

Create a `config.yaml` file:

```yaml
# LLM Configuration
llm:
  endpoint: "http://localhost:11434"
  timeout: 30.0
  max_retries: 3
  retry_delay: 1.0

# Model Selection
models:
  fast: "tinyllama"
  balanced: "gemma3"
  high: "qwen2.5-7b"

# Compression Settings
compression:
  default_mode: "balanced"
  quality_threshold: 0.85
  max_summary_tokens: 100

# Storage Settings
storage:
  data_dir: "./data"
  compression: "zstd"
  compression_level: 3

# Embedding Settings
embedding:
  model_name: "all-MiniLM-L6-v2"
  batch_size: 32
  cache_size: 1000

# Performance Settings
performance:
  enable_optimizations: true
  adaptive_threshold: 1000
  batch_size: 100
  max_workers: 4

# Cost Monitoring
cost:
  log_file: "cost_log.jsonl"
  daily_budget: 10.0  # USD
  alert_threshold: 0.8  # 80% of budget
```

### Load Configuration

```python
from llm_compression import Config

# Load from file
config = Config.from_yaml("config.yaml")

# Apply environment overrides
config.apply_env_overrides()

# Use in components
client = LLMClient(
    endpoint=config.llm.endpoint,
    timeout=config.llm.timeout
)
```

### Environment Variables

Override configuration with environment variables:

```bash
# LLM settings
export LLM_ENDPOINT="http://localhost:11434"
export LLM_TIMEOUT=30.0

# Model settings
export MODEL_FAST="tinyllama"
export MODEL_BALANCED="gemma3"

# Storage settings
export STORAGE_DATA_DIR="./data"
export STORAGE_COMPRESSION="zstd"
```

---

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed

**Symptom**: `LLMAPIError: Connection refused`

**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Pull required models
ollama pull gemma3
ollama pull qwen2.5-7b
```

#### 2. Slow First Run

**Symptom**: First embedding takes 30-60 seconds

**Cause**: Downloading embedding model (~500MB)

**Solution**:
```python
# Pre-load model at startup
from llm_compression.embedder_cache import preload_default_model
preload_default_model()
```

#### 3. High Memory Usage

**Symptom**: Process using >8GB RAM

**Solutions**:

**Option 1**: Reduce batch size
```python
loop_arrow = CognitiveLoopArrow(
    batch_size=50,  # Reduce from 100
    max_workers=2   # Reduce from 4
)
```

**Option 2**: Use memory mapping
```python
from llm_compression.arrow_storage_zero_copy import ArrowStorageZeroCopy

storage = ArrowStorageZeroCopy()
table = storage.load_table_mmap("large_file.parquet")  # Memory-mapped
```

**Option 3**: Enable adaptive optimization
```python
loop_arrow = CognitiveLoopArrow(
    enable_optimizations=True,
    adaptive_threshold=500  # Switch to Arrow earlier
)
```

#### 4. Low Compression Quality

**Symptom**: Quality score < 0.85

**Solutions**:

**Option 1**: Use higher quality mode
```python
compressed = compressor.compress(text, mode="high")
```

**Option 2**: Use better model
```python
selector = ModelSelector()
model = selector.select_model(
    memory_type=MemoryType.DOCUMENTATION,
    quality=QualityLevel.HIGH
)
```

**Option 3**: Increase summary tokens
```python
# In config.yaml
compression:
  max_summary_tokens: 150  # Increase from 100
```

#### 5. API Timeout

**Symptom**: `LLMTimeoutError: Request timed out`

**Solutions**:

**Option 1**: Increase timeout
```python
client = LLMClient(timeout=60.0)  # Increase from 30.0
```

**Option 2**: Use faster model
```python
compressed = compressor.compress(text, mode="fast")
```

**Option 3**: Enable retry
```python
client = LLMClient(
    max_retries=5,
    retry_delay=2.0
)
```

### Debug Mode

Enable detailed logging:

```python
from llm_compression.logger import setup_logger
import logging

# Enable debug logging
setup_logger(level=logging.DEBUG)

# Now all operations are logged
compressed = compressor.compress(text)
```

### Performance Profiling

Profile performance:

```python
from llm_compression import PerformanceMonitor

monitor = PerformanceMonitor()

# Record operations
with monitor.track("compression"):
    compressed = compressor.compress(text)

# Get statistics
stats = monitor.get_stats()
print(f"Average latency: {stats['compression']['avg_ms']:.1f}ms")
```

---

## Best Practices

### 1. Initialization

**Do**: Initialize components once at startup
```python
# Good: Initialize once
client = LLMClient()
compressor = LLMCompressor(client, selector)

# Use many times
for text in texts:
    compressed = compressor.compress(text)
```

**Don't**: Initialize in loops
```python
# Bad: Initialize repeatedly
for text in texts:
    client = LLMClient()  # Wasteful!
    compressor = LLMCompressor(client, selector)
    compressed = compressor.compress(text)
```

### 2. Batch Processing

**Do**: Process in batches
```python
# Good: Batch processing
embeddings = embedder.encode_batch(texts, batch_size=32)
```

**Don't**: Process one-by-one
```python
# Bad: Individual processing
embeddings = [embedder.encode(text) for text in texts]
```

### 3. Memory Management

**Do**: Use memory mapping for large files
```python
# Good: Memory-mapped loading
storage = ArrowStorageZeroCopy()
table = storage.load_table_mmap("large_file.parquet")
```

**Don't**: Load entire file into memory
```python
# Bad: Full load
table = pa.parquet.read_table("large_file.parquet")
```

### 4. Error Handling

**Do**: Handle errors gracefully
```python
# Good: Error handling
try:
    compressed = compressor.compress(text)
except CompressionError as e:
    logger.error(f"Compression failed: {e}")
    # Fallback strategy
    compressed = simple_compress(text)
```

**Don't**: Ignore errors
```python
# Bad: No error handling
compressed = compressor.compress(text)  # May crash!
```

### 5. Cost Management

**Do**: Monitor costs
```python
# Good: Cost monitoring
monitor = CostMonitor()
monitor.record_operation(...)

# Check daily
summary = monitor.get_daily_summary()
if summary['total_cost'] > 10.0:
    logger.warning("Daily budget exceeded!")
```

**Don't**: Ignore costs
```python
# Bad: No cost tracking
# Costs can spiral out of control
```

### 6. Quality Validation

**Do**: Validate quality
```python
# Good: Quality check
compressed = compressor.compress(text)
if compressed.quality_score < 0.85:
    logger.warning("Low quality, retrying with high mode")
    compressed = compressor.compress(text, mode="high")
```

**Don't**: Assume quality
```python
# Bad: No validation
compressed = compressor.compress(text)
# Quality might be poor!
```

### 7. Configuration Management

**Do**: Use configuration files
```yaml
# config.yaml
compression:
  default_mode: "balanced"
  quality_threshold: 0.85
```

```python
# Good: Load from config
config = Config.from_yaml("config.yaml")
compressor = LLMCompressor(client, selector, config=config)
```

**Don't**: Hardcode values
```python
# Bad: Hardcoded
compressor = LLMCompressor(
    client,
    selector,
    quality_threshold=0.85,  # Hardcoded!
    max_tokens=100
)
```

---

## FAQ

### General Questions

**Q: What compression ratio can I expect?**

A: Typical compression ratios:
- Fast mode: 10-20x
- Balanced mode: 20-30x
- High mode: 30-50x

Actual ratios depend on content type and redundancy.

**Q: How much does it cost to run?**

A: Cost breakdown:
- Local models (Ollama): Free (compute only)
- Cloud APIs: $0.50-$2.00 per day (1000 compressions)
- Storage: Minimal (<$0.01/GB/month)

**Q: Is my data secure?**

A: Security features:
- Local processing option (Ollama)
- Encryption at rest (optional)
- No data sent to external services (local mode)
- Audit logging available

**Q: Can I use custom models?**

A: Yes! Configure in `config.yaml`:
```yaml
models:
  custom: "my-model-name"
```

Then use:
```python
compressed = compressor.compress(text, model="custom")
```

### Technical Questions

**Q: What's the difference between compression modes?**

A:
- **Fast**: Uses TinyLlama, <3s, quality ~0.80
- **Balanced**: Uses Gemma3, <10s, quality ~0.85
- **High**: Uses Qwen2.5-7B, <20s, quality ~0.90

**Q: How does semantic search work?**

A: Two-stage process:
1. **Semantic Index**: FTS5 full-text search on summaries
2. **Vector Search**: Cosine similarity on embeddings
3. **Hybrid**: Combine both for best results

**Q: What's the maximum memory size?**

A: Limits:
- Single memory: No hard limit (tested up to 100KB)
- Total memories: 100K+ per instance
- Storage: 10GB+ files supported (memory-mapped)

**Q: Can I deploy to production?**

A: Yes! Deployment options:
- Docker container
- Kubernetes cluster
- Serverless (AWS Lambda, etc.)

See [Architecture Documentation](ARCHITECTURE.md) for details.

**Q: How do I migrate existing data?**

A: Migration steps:
1. Export existing data to JSON/CSV
2. Convert to Arrow format
3. Import using batch processor

See [Arrow Migration Guide](ARROW_MIGRATION_GUIDE.md) for details.

### Performance Questions

**Q: Why is the first run slow?**

A: First run downloads embedding model (~500MB). Subsequent runs use cached model.

**Q: How can I improve performance?**

A: Optimization checklist:
1. ✅ Pre-load models at startup
2. ✅ Enable Arrow zero-copy optimization
3. ✅ Use batch processing
4. ✅ Enable adaptive optimization
5. ✅ Adjust batch size and workers

**Q: What's the query latency?**

A: Typical latencies:
- Semantic search: <10ms
- Vector search: <50ms (1K memories)
- Full cognitive loop: <100ms

**Q: Can I use GPU acceleration?**

A: Yes! GPU acceleration available for:
- Embedding computation (sentence-transformers)
- Model inference (Ollama with GPU)
- Vector operations (CuPy, optional)

---

## Next Steps

### Learning Resources

- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Quick Start**: [QUICK_START.md](QUICK_START.md)
- **Migration Guide**: [ARROW_MIGRATION_GUIDE.md](ARROW_MIGRATION_GUIDE.md)

### Example Projects

- `examples/chat_agent_optimized.py`: Interactive chat agent
- `examples/batch_compression.py`: Batch processing example
- `examples/cost_monitor_integration.py`: Cost monitoring example

### Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Contributing**: See `CONTRIBUTING.md` (coming soon)

---

## Appendix

### Glossary

- **Compression Ratio**: Original size / Compressed size
- **Semantic Similarity**: Cosine similarity of embeddings (0-1)
- **Quality Score**: Overall reconstruction quality (0-1)
- **Embedding**: Dense vector representation of text
- **Zero-Copy**: Data access without copying memory
- **Memory Primitive**: Basic unit of memory in cognitive network

### Supported Models

**Embedding Models**:
- all-MiniLM-L6-v2 (default, 384 dimensions)
- all-mpnet-base-v2 (768 dimensions)
- Custom sentence-transformers models

**LLM Models** (Ollama):
- TinyLlama (fast, 1.1B parameters)
- Gemma3 (balanced, 3B parameters)
- Qwen2.5-7B (high quality, 7B parameters)
- Llama3.1 (highest quality, 8B parameters)

**LLM Models** (Cloud):
- OpenAI: gpt-4, gpt-3.5-turbo
- Anthropic: claude-opus-4, claude-sonnet-4
- Google: gemini-pro, gemini-flash

### File Formats

- **Parquet**: Columnar storage format (Arrow-compatible)
- **SQLite**: Full-text search index (FTS5)
- **JSONL**: Cost logs and audit trails
- **YAML**: Configuration files

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-17  
**Maintainer**: AI-OS Team

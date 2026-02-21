# API Reference

**Version**: 2.0  
**Last Updated**: 2026-02-17

---

## Table of Contents

1. [Core Components](#core-components)
2. [Storage Layer](#storage-layer)
3. [Compression & Reconstruction](#compression--reconstruction)
4. [Cognitive System](#cognitive-system)
5. [Quantization](#quantization)
6. [Monitoring & Optimization](#monitoring--optimization)
7. [Utilities](#utilities)

---

## Core Components

### LLMClient

**Purpose**: Interface for communicating with LLM services (Ollama, OpenAI, etc.)

**Import**:
```python
from llm_compression import LLMClient, LLMResponse
```

**Constructor**:
```python
LLMClient(
    endpoint: str = "http://localhost:11434",
    timeout: float = 30.0,
    max_retries: int = 3,
    retry_delay: float = 1.0
)
```

**Methods**:

#### `complete(prompt: str, model: str, **kwargs) -> LLMResponse`
Generate text completion from LLM.

**Parameters**:
- `prompt` (str): Input prompt text
- `model` (str): Model name (e.g., "gemma3", "qwen2.5")
- `max_tokens` (int, optional): Maximum tokens to generate
- `temperature` (float, optional): Sampling temperature (0.0-1.0)
- `stream` (bool, optional): Enable streaming response

**Returns**: `LLMResponse` object with `text`, `model`, `tokens_used`

**Example**:
```python
client = LLMClient()
response = client.complete(
    prompt="What is Python?",
    model="gemma3",
    max_tokens=100,
    temperature=0.7
)
print(response.text)
```

---

### ModelSelector

**Purpose**: Intelligent model selection based on task requirements

**Import**:
```python
from llm_compression import ModelSelector, MemoryType, QualityLevel
```

**Constructor**:
```python
ModelSelector(
    config: Optional[Config] = None
)
```

**Methods**:

#### `select_model(memory_type: MemoryType, quality: QualityLevel) -> str`
Select optimal model for compression task.

**Parameters**:
- `memory_type` (MemoryType): Type of memory (CODE, CONVERSATION, DOCUMENTATION)
- `quality` (QualityLevel): Quality requirement (FAST, BALANCED, HIGH)

**Returns**: Model name (str)

**Example**:
```python
selector = ModelSelector()
model = selector.select_model(
    memory_type=MemoryType.CODE,
    quality=QualityLevel.HIGH
)
# Returns: "qwen2.5-7b"
```

---

### ModelRouter

**Purpose**: Advanced model routing with cost estimation

**Import**:
```python
from llm_compression.model_router import ModelRouter
```

**Constructor**:
```python
ModelRouter(
    config: Optional[Config] = None
)
```

**Methods**:

#### `select_model(text: str, task_type: str, requirements: Dict) -> str`
Route to optimal model based on content and requirements.

**Parameters**:
- `text` (str): Input text to analyze
- `task_type` (str): Task type ("summary", "extraction", "generation")
- `requirements` (Dict): Requirements dict with keys:
  - `quality_threshold` (float): Minimum quality (0.0-1.0)
  - `latency_budget` (float): Maximum latency in seconds
  - `max_cost` (float, optional): Maximum cost per request

**Returns**: Model name (str)

**Example**:
```python
router = ModelRouter()
model = router.select_model(
    text="Long technical document...",
    task_type="summary",
    requirements={
        "quality_threshold": 0.85,
        "latency_budget": 10.0
    }
)
```

---

## Storage Layer

### ArrowStorage

**Purpose**: High-performance Arrow-based storage with compression

**Import**:
```python
from llm_compression import ArrowStorage
```

**Constructor**:
```python
ArrowStorage(
    compression: str = "zstd",
    compression_level: int = 3
)
```

**Methods**:

#### `save(table: pa.Table, path: str) -> None`
Save Arrow table to Parquet file.

**Parameters**:
- `table` (pa.Table): Arrow table to save
- `path` (str): Output file path

**Example**:
```python
storage = ArrowStorage()
storage.save(memory_table, "memories.parquet")
```

#### `load(path: str) -> pa.Table`
Load Arrow table from Parquet file.

**Parameters**:
- `path` (str): Input file path

**Returns**: Arrow Table

**Example**:
```python
table = storage.load("memories.parquet")
```

---

### ArrowStorageZeroCopy

**Purpose**: Zero-copy optimized storage with memory mapping

**Import**:
```python
from llm_compression.arrow_storage_zero_copy import ArrowStorageZeroCopy
```

**Constructor**:
```python
ArrowStorageZeroCopy(
    compression: str = "zstd",
    compression_level: int = 3
)
```

**Methods**:

#### `load_table_mmap(path: str) -> pa.Table`
Load table using memory mapping (zero-copy for large files).

**Parameters**:
- `path` (str): Parquet file path

**Returns**: Memory-mapped Arrow Table

**Example**:
```python
storage = ArrowStorageZeroCopy()
table = storage.load_table_mmap("large_memories.parquet")  # Supports 10GB+ files
```

#### `query_arrow(table: pa.Table, filter_expr: pc.Expression) -> pa.Table`
Query table with zero-copy filtering.

**Parameters**:
- `table` (pa.Table): Input table
- `filter_expr` (pc.Expression): PyArrow compute expression

**Returns**: Filtered table (zero-copy)

**Example**:
```python
import pyarrow.compute as pc

filtered = storage.query_arrow(
    table,
    pc.field("timestamp") > 1234567890
)
```

---

## Compression & Reconstruction

### LLMCompressor

**Purpose**: Compress text using LLM-based semantic compression

**Import**:
```python
from llm_compression import LLMCompressor, CompressedMemory
```

**Constructor**:
```python
LLMCompressor(
    llm_client: LLMClient,
    model_selector: ModelSelector,
    quality_evaluator: Optional[QualityEvaluator] = None
)
```

**Methods**:

#### `compress(text: str, mode: str = "balanced") -> CompressedMemory`
Compress text to semantic representation.

**Parameters**:
- `text` (str): Input text to compress
- `mode` (str): Compression mode ("fast", "balanced", "high")

**Returns**: `CompressedMemory` object

**Example**:
```python
compressor = LLMCompressor(llm_client, model_selector)
compressed = compressor.compress(
    text="Long conversation history...",
    mode="balanced"
)
print(f"Compression ratio: {compressed.compression_ratio}x")
```

---

### LLMReconstructor

**Purpose**: Reconstruct original text from compressed memory

**Import**:
```python
from llm_compression import LLMReconstructor, ReconstructedMemory
```

**Constructor**:
```python
LLMReconstructor(
    llm_client: LLMClient
)
```

**Methods**:

#### `reconstruct(compressed: CompressedMemory) -> ReconstructedMemory`
Reconstruct text from compressed representation.

**Parameters**:
- `compressed` (CompressedMemory): Compressed memory object

**Returns**: `ReconstructedMemory` with reconstructed text and quality score

**Example**:
```python
reconstructor = LLMReconstructor(llm_client)
reconstructed = reconstructor.reconstruct(compressed)
print(reconstructed.text)
print(f"Quality: {reconstructed.quality_score}")
```

---

## Cognitive System

### CognitiveLoop

**Purpose**: Core cognitive processing loop with memory, learning, and feedback

**Import**:
```python
from llm_compression import CognitiveLoop, CognitiveResult
```

**Constructor**:
```python
CognitiveLoop(
    expressor: MultiModalExpressor,
    feedback: InternalFeedbackSystem,
    learner: ConnectionLearner,
    navigator: NetworkNavigator,
    max_correction_iterations: int = 3
)
```

**Methods**:

#### `async process(query: str, query_embedding: np.ndarray, max_memories: int = 10) -> CognitiveResult`
Process query through complete cognitive loop.

**Parameters**:
- `query` (str): Query text
- `query_embedding` (np.ndarray): Query embedding vector
- `max_memories` (int): Maximum memories to retrieve

**Returns**: `CognitiveResult` with output, quality, and metadata

**Example**:
```python
loop = CognitiveLoop(expressor, feedback, learner, navigator)
result = await loop.process(
    query="What is machine learning?",
    query_embedding=embedder.encode("What is machine learning?"),
    max_memories=5
)
print(result.output)
```

#### `add_memory(memory: MemoryPrimitive) -> None`
Add memory to cognitive network.

**Parameters**:
- `memory` (MemoryPrimitive): Memory object to add

**Example**:
```python
memory = MemoryPrimitive(
    id="mem1",
    content="Python is a programming language",
    embedding=embedder.encode("Python is a programming language")
)
loop.add_memory(memory)
```

---

### CognitiveLoopArrow

**Purpose**: Zero-copy optimized cognitive loop with Arrow integration

**Import**:
```python
from llm_compression.cognitive_loop_arrow import CognitiveLoopArrow
```

**Constructor**:
```python
CognitiveLoopArrow(
    cognitive_loop: Optional[CognitiveLoop] = None,
    enable_optimizations: bool = True,
    adaptive_threshold: int = 1000,
    batch_size: int = 100,
    max_workers: int = 4
)
```

**Methods**:

#### `async process_arrow(query: str, max_memories: int = 10) -> CognitiveResult`
Process query with automatic embedding and zero-copy retrieval.

**Parameters**:
- `query` (str): Query text (embedding computed automatically)
- `max_memories` (int): Maximum memories to retrieve

**Returns**: `CognitiveResult`

**Example**:
```python
loop_arrow = CognitiveLoopArrow(enable_optimizations=True)
result = await loop_arrow.process_arrow(
    query="What is Python?",
    max_memories=5
)
```

#### `batch_add_memories_arrow(memory_ids: List[str], contents: List[str]) -> None`
Batch add memories with zero-copy optimization.

**Parameters**:
- `memory_ids` (List[str]): List of memory IDs
- `contents` (List[str]): List of memory contents

**Example**:
```python
loop_arrow.batch_add_memories_arrow(
    memory_ids=["mem1", "mem2", "mem3"],
    contents=["text1", "text2", "text3"]
)
```

---

### ConversationalAgent

**Purpose**: High-level conversational agent with memory and personalization

**Import**:
```python
from llm_compression import ConversationalAgent, AgentResponse
```

**Constructor**:
```python
ConversationalAgent(
    llm_client: LLMClient,
    compressor: LLMCompressor,
    cognitive_loop: CognitiveLoop,
    user_id: str,
    personalization_engine: Optional[PersonalizationEngine] = None
)
```

**Methods**:

#### `async chat(message: str) -> AgentResponse`
Process user message and generate response.

**Parameters**:
- `message` (str): User message

**Returns**: `AgentResponse` with message, quality, and metadata

**Example**:
```python
agent = ConversationalAgent(
    llm_client=llm_client,
    compressor=compressor,
    cognitive_loop=cognitive_loop,
    user_id="user_001"
)

response = await agent.chat("Hello!")
print(response.message)
print(f"Quality: {response.quality_score}")
```

---

## Quantization

### ArrowQuantizer

**Purpose**: Model weight quantization with Arrow storage

**Import**:
```python
from llm_compression.inference.arrow_quantizer import ArrowQuantizer
```

**Constructor**:
```python
ArrowQuantizer(
    bits: int = 8,
    strategy: str = "per_tensor"
)
```

**Methods**:

#### `quantize_weights(weights: Dict[str, torch.Tensor]) -> pa.Table`
Quantize model weights to lower precision.

**Parameters**:
- `weights` (Dict[str, torch.Tensor]): Model weight dictionary

**Returns**: Arrow Table with quantized weights

**Example**:
```python
quantizer = ArrowQuantizer(bits=8, strategy="per_channel")
quantized_table = quantizer.quantize_weights(model.state_dict())
```

#### `dequantize_weights(table: pa.Table) -> Dict[str, torch.Tensor]`
Dequantize weights back to full precision.

**Parameters**:
- `table` (pa.Table): Quantized weights table

**Returns**: Weight dictionary

**Example**:
```python
weights = quantizer.dequantize_weights(quantized_table)
model.load_state_dict(weights)
```

---

### GPTQCalibrator

**Purpose**: GPTQ-based quantization calibration for improved accuracy

**Import**:
```python
from llm_compression.inference.gptq_calibrator import GPTQCalibrator
```

**Constructor**:
```python
GPTQCalibrator(
    model: torch.nn.Module,
    bits: int = 4,
    group_size: int = 128
)
```

**Methods**:

#### `calibrate(calibration_data: List[torch.Tensor]) -> Dict[str, Any]`
Calibrate quantization parameters using GPTQ algorithm.

**Parameters**:
- `calibration_data` (List[torch.Tensor]): Calibration dataset

**Returns**: Calibration parameters dictionary

**Example**:
```python
calibrator = GPTQCalibrator(model, bits=4)
calib_params = calibrator.calibrate(calibration_dataset)
```

---

## Monitoring & Optimization

### CostMonitor

**Purpose**: Track and analyze API and compute costs

**Import**:
```python
from llm_compression.cost_monitor import CostMonitor
```

**Constructor**:
```python
CostMonitor(
    log_file: str = "cost_log.jsonl"
)
```

**Methods**:

#### `record_operation(operation_type: str, model: str, tokens: int, cost: float) -> None`
Record cost of an operation.

**Parameters**:
- `operation_type` (str): Operation type ("compression", "embedding", etc.)
- `model` (str): Model name
- `tokens` (int): Tokens used
- `cost` (float): Cost in USD

**Example**:
```python
monitor = CostMonitor()
monitor.record_operation(
    operation_type="compression",
    model="gpt-4",
    tokens=1500,
    cost=0.045
)
```

#### `get_summary() -> Dict[str, Any]`
Get cost summary statistics.

**Returns**: Dictionary with total cost, operation counts, etc.

**Example**:
```python
summary = monitor.get_summary()
print(f"Total cost: ${summary['total_cost']:.2f}")
```

---

### PerformanceMonitor

**Purpose**: Monitor system performance metrics

**Import**:
```python
from llm_compression import PerformanceMonitor, PerformanceMetrics
```

**Constructor**:
```python
PerformanceMonitor()
```

**Methods**:

#### `record_operation(operation: str, duration_ms: float, **metadata) -> None`
Record performance of an operation.

**Parameters**:
- `operation` (str): Operation name
- `duration_ms` (float): Duration in milliseconds
- `**metadata`: Additional metadata

**Example**:
```python
monitor = PerformanceMonitor()
monitor.record_operation(
    operation="compression",
    duration_ms=125.5,
    model="gemma3",
    text_length=1500
)
```

---

## Utilities

### LocalEmbedder

**Purpose**: Local text embedding using sentence-transformers

**Import**:
```python
from llm_compression.embedder import LocalEmbedder
```

**Constructor**:
```python
LocalEmbedder(
    model_name: str = "all-MiniLM-L6-v2"
)
```

**Methods**:

#### `encode(text: str) -> np.ndarray`
Encode single text to embedding vector.

**Parameters**:
- `text` (str): Input text

**Returns**: Embedding vector (np.ndarray)

**Example**:
```python
embedder = LocalEmbedder()
embedding = embedder.encode("Hello world")
```

#### `encode_batch(texts: List[str], batch_size: int = 32) -> np.ndarray`
Encode multiple texts in batch.

**Parameters**:
- `texts` (List[str]): List of texts
- `batch_size` (int): Batch size for processing

**Returns**: Embedding matrix (np.ndarray)

**Example**:
```python
embeddings = embedder.encode_batch(["text1", "text2", "text3"])
```

---

### QualityEvaluator

**Purpose**: Evaluate compression/reconstruction quality

**Import**:
```python
from llm_compression import QualityEvaluator, QualityMetrics
```

**Constructor**:
```python
QualityEvaluator(
    embedder: Optional[LocalEmbedder] = None
)
```

**Methods**:

#### `evaluate(original: str, reconstructed: str) -> QualityMetrics`
Evaluate reconstruction quality.

**Parameters**:
- `original` (str): Original text
- `reconstructed` (str): Reconstructed text

**Returns**: `QualityMetrics` with semantic_similarity, keyword_retention, etc.

**Example**:
```python
evaluator = QualityEvaluator()
metrics = evaluator.evaluate(original_text, reconstructed_text)
print(f"Semantic similarity: {metrics.semantic_similarity}")
print(f"Keyword retention: {metrics.keyword_retention}")
```

---

## Data Models

### CompressedMemory

**Attributes**:
- `memory_id` (str): Unique memory identifier
- `summary` (str): Semantic summary
- `entities` (List[Entity]): Extracted entities
- `diff_data` (bytes): Diff for reconstruction
- `compression_ratio` (float): Compression ratio achieved
- `quality_score` (float): Quality score (0.0-1.0)
- `model_used` (str): Model used for compression

### MemoryPrimitive

**Attributes**:
- `id` (str): Memory ID
- `content` (str): Memory content
- `embedding` (np.ndarray): Embedding vector
- `activation_level` (float): Current activation (0.0-1.0)
- `connections` (Dict[str, float]): Connections to other memories

### CognitiveResult

**Attributes**:
- `output` (str): Generated output
- `quality_score` (float): Output quality (0.0-1.0)
- `memories_used` (List[str]): Memory IDs used
- `corrections_applied` (int): Number of corrections
- `processing_time_ms` (float): Processing time

---

## Error Handling

### CompressionError

Base exception for compression-related errors.

```python
from llm_compression import CompressionError

try:
    compressed = compressor.compress(text)
except CompressionError as e:
    print(f"Compression failed: {e}")
```

### LLMAPIError

Exception for LLM API communication errors.

```python
from llm_compression import LLMAPIError

try:
    response = client.complete(prompt, model)
except LLMAPIError as e:
    print(f"API error: {e}")
```

---

## Configuration

### Config

**Purpose**: System configuration management

**Import**:
```python
from llm_compression import Config
```

**Methods**:

#### `from_yaml(path: str) -> Config`
Load configuration from YAML file.

**Example**:
```python
config = Config.from_yaml("config.yaml")
```

#### `apply_env_overrides() -> None`
Apply environment variable overrides.

**Example**:
```python
config.apply_env_overrides()
```

---

## See Also

- [Architecture Documentation](ARCHITECTURE.md)
- [User Guide](USER_GUIDE.md)
- [Quick Start Guide](QUICK_START.md)
- [Arrow Migration Guide](ARROW_MIGRATION_GUIDE.md)

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-17  
**Maintainer**: AI-OS Team

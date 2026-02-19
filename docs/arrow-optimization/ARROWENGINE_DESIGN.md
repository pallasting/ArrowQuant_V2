# ArrowEngine 架构设计

## 概述

ArrowEngine 是一个高性能的嵌入模型推理引擎，利用 Arrow/Parquet 零拷贝存储实现极低延迟的模型加载和推理。

## 设计目标

### 性能指标
- **模型加载时间**: < 100ms (相比传统方式的 2-5s)
- **单次推理延迟**: < 5ms (批大小=1)
- **批处理吞吐量**: > 2000 requests/s (批大小=32)
- **内存占用**: < 原始模型的 50% (通过 float16)

### 核心特性
1. **零拷贝权重加载**: 直接从 Parquet 映射到 PyTorch Tensor
2. **Fast Tokenization**: 集成 Rust tokenizers 库 (10-20x 加速)
3. **批处理优化**: 动态批处理和内存池
4. **缓存友好**: 权重预加载和 Tensor 复用

## 系统架构

### 组件层次

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  (FastAPI Service / AI-OS Tool / Python API)            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    ArrowEngine API                       │
│  - encode(texts) → embeddings                           │
│  - encode_batch(texts, batch_size) → embeddings         │
│  - get_embedding_dimension() → int                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   Core Components                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  WeightLoader│  │   Tokenizer  │  │InferenceCore │  │
│  │   (Arrow)    │  │    (Rust)    │  │  (PyTorch)   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   Storage Layer                          │
│  - weights.parquet (Arrow/Parquet format)               │
│  - tokenizer/ (Rust tokenizer files)                    │
│  - metadata.json (model configuration)                  │
└─────────────────────────────────────────────────────────┘
```

### 数据流

```
Input Text
    ↓
┌─────────────────┐
│ Fast Tokenizer  │  (Rust tokenizers, 10-20x faster)
│  - tokenize()   │
│  - encode()     │
└─────────────────┘
    ↓
Token IDs (List[int])
    ↓
┌─────────────────┐
│ Input Processor │  (Padding, attention masks)
│  - pad_sequence │
│  - create_masks │
└─────────────────┘
    ↓
Input Tensors (torch.Tensor)
    ↓
┌─────────────────┐
│ Inference Core  │  (Forward pass with cached weights)
│  - embedding()  │
│  - attention()  │
│  - pooling()    │
└─────────────────┘
    ↓
Output Embeddings (np.ndarray)
```

## 核心类设计

### 1. ArrowEngine (主类)

```python
class ArrowEngine:
    """
    High-performance embedding inference engine using Arrow/Parquet storage.
    
    Features:
    - Zero-copy weight loading from Parquet
    - Fast Rust tokenization (10-20x speedup)
    - Batch processing with dynamic batching
    - Memory-efficient float16 inference
    
    Performance Targets:
    - Model load time: < 100ms
    - Inference latency: < 5ms (batch_size=1)
    - Throughput: > 2000 req/s (batch_size=32)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_batch_size: int = 32,
        use_fast_tokenizer: bool = True,
    ):
        """
        Initialize ArrowEngine.
        
        Args:
            model_path: Path to converted model directory (containing weights.parquet)
            device: Device for inference ("cpu", "cuda", "mps")
            max_batch_size: Maximum batch size for inference
            use_fast_tokenizer: Use Rust tokenizer (recommended)
        """
        
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            Embeddings as numpy array, shape (n_texts, embedding_dim)
        """
        
    def encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a batch of texts (optimized for throughput).
        
        Args:
            texts: List of texts (up to max_batch_size)
            normalize: L2-normalize embeddings
            
        Returns:
            Normalized embeddings, shape (batch_size, embedding_dim)
        """
        
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        
    def get_max_seq_length(self) -> int:
        """Get maximum sequence length."""
```

### 2. WeightLoader (零拷贝加载)

```python
class WeightLoader:
    """
    Zero-copy weight loader from Arrow/Parquet format.
    
    Uses memory-mapped Arrow tables for instant loading.
    """
    
    def __init__(self, parquet_path: str):
        """Initialize weight loader."""
        
    def load_weights(self) -> Dict[str, torch.Tensor]:
        """
        Load weights from Parquet with zero-copy.
        
        Returns:
            Dictionary mapping layer names to PyTorch tensors
        """
        
    def get_layer(self, layer_name: str) -> torch.Tensor:
        """Get specific layer weights."""
```

### 3. FastTokenizer (Rust 集成)

```python
class FastTokenizer:
    """
    Fast tokenizer using Rust tokenizers library.
    
    Provides 10-20x speedup over Python tokenizers.
    """
    
    def __init__(self, tokenizer_path: str):
        """Load Rust tokenizer from directory."""
        
    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode texts to token IDs.
        
        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors
        """
        
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
```

### 4. InferenceCore (推理引擎)

```python
class InferenceCore:
    """
    Core inference engine for embedding generation.
    
    Implements optimized forward pass with caching.
    """
    
    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        """Initialize inference core with weights."""
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to generate embeddings.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            
        Returns:
            Embeddings, shape (batch_size, hidden_dim)
        """
        
    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pooling operation."""
```

## 优化策略

### 1. 零拷贝权重加载

**传统方式** (2-5秒):
```python
# PyTorch 默认加载
model = torch.load("model.pt")  # 完整反序列化
```

**ArrowEngine 方式** (<100ms):
```python
# 零拷贝内存映射
table = pq.read_table("weights.parquet", memory_map=True)
# 直接转换为 Tensor (无拷贝)
tensor = torch.from_numpy(table['data'][0].as_py())
```

### 2. Fast Tokenization

**Python Tokenizer**: ~500 tokens/s
```python
# 传统 Python tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model")
```

**Rust Tokenizer**: ~10,000 tokens/s (20x faster)
```python
# Rust tokenizers 库
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")
```

### 3. 批处理优化

```python
# 动态批处理策略
class BatchProcessor:
    def __init__(self, max_batch_size: int = 32):
        self.buffer = []
        self.max_batch_size = max_batch_size
        
    def add(self, text: str) -> Optional[List[np.ndarray]]:
        """
        添加文本到批处理缓冲区。
        当缓冲区满时自动触发推理。
        """
        self.buffer.append(text)
        if len(self.buffer) >= self.max_batch_size:
            return self.flush()
        return None
        
    def flush(self) -> List[np.ndarray]:
        """处理缓冲区中的所有文本"""
        results = self.engine.encode_batch(self.buffer)
        self.buffer.clear()
        return results
```

### 4. 内存优化

**Float16 存储**: 减少 50% 内存
```python
# 加载时自动转换为 float16
if config.use_float16:
    tensor = tensor.half()  # float32 → float16
```

**Tensor 复用**: 避免重复分配
```python
# 预分配缓冲区
self._embedding_buffer = torch.zeros(
    (max_batch_size, embedding_dim),
    dtype=torch.float16,
    device=device
)
```

## 性能基准

### 目标 vs 传统方式对比

| 指标 | ArrowEngine 目标 | 传统方式 (sentence-transformers) | 提升 |
|------|-----------------|----------------------------------|------|
| 模型加载 | < 100ms | 2-5s | **20-50x** |
| 单次推理 | < 5ms | 10-20ms | **2-4x** |
| 批处理吞吐量 | > 2000 req/s | 500-800 req/s | **2.5-4x** |
| 内存占用 | ~45MB (float16) | ~90MB (float32) | **2x** |

### 测试配置

- **模型**: sentence-transformers/all-MiniLM-L6-v2
- **硬件**: CPU (Intel i7), 16GB RAM
- **批大小**: 32
- **序列长度**: 128 tokens (平均)

## 实现计划

### Phase 1: 核心组件 (T2.2 - T2.4)

1. **WeightLoader** (T2.2, 8h)
   - Parquet 读取和零拷贝映射
   - 转换为 PyTorch Tensor
   - 权重缓存机制

2. **FastTokenizer** (T2.3, 4h)
   - Rust tokenizer 加载
   - 批量编码接口
   - Padding 和 attention mask 生成

3. **InferenceCore** (T2.4, 10h)
   - BERT-like 模型前向传播
   - Mean pooling 实现
   - 批处理支持

### Phase 2: 优化和集成 (T2.5 - T2.7)

4. **批处理优化** (T2.5, 6h)
   - 动态批处理
   - 内存池管理
   - Tensor 复用

5. **性能基准** (T2.6, 4h)
   - 加载时间测试
   - 推理延迟测试
   - 吞吐量测试
   - 内存占用分析

6. **集成测试** (T2.7, 4h)
   - 端到端测试
   - 与 ModelConverter 集成验证
   - 多模型测试

## API 使用示例

### 基本使用

```python
from llm_compression.inference import ArrowEngine

# 加载模型 (< 100ms)
engine = ArrowEngine("./models/minilm")

# 单文本编码
embedding = engine.encode("Hello, world!")
print(embedding.shape)  # (384,)

# 批量编码
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = engine.encode(texts, batch_size=32)
print(embeddings.shape)  # (3, 384)
```

### 高性能批处理

```python
# 大规模批处理
import numpy as np

texts = ["Document " + str(i) for i in range(10000)]

# 自动批处理，显示进度
embeddings = engine.encode(
    texts,
    batch_size=32,
    show_progress=True
)

# 性能指标
# - 总时间: ~5s
# - 吞吐量: ~2000 req/s
# - 内存峰值: ~200MB
```

### 与 ModelConverter 集成

```python
from llm_compression.tools import ModelConverter, ConversionConfig
from llm_compression.inference import ArrowEngine

# 步骤 1: 转换模型
config = ConversionConfig(use_float16=True, compression="lz4")
converter = ModelConverter(config)

result = converter.convert(
    model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    output_dir="./models/minilm",
    model_type="sentence-transformers"
)

# 步骤 2: 使用 ArrowEngine 推理
engine = ArrowEngine("./models/minilm")
embeddings = engine.encode("Fast inference with Arrow!")
```

## 依赖项

### Python 包
- `pyarrow >= 23.0.0` - Arrow/Parquet 读取
- `torch >= 2.0.0` - 推理引擎
- `tokenizers >= 0.21.0` - Rust tokenizer
- `numpy >= 1.24.0` - 数值计算

### 可选依赖
- `tqdm` - 进度条显示
- `psutil` - 内存监控

## 测试策略

### 单元测试
- `test_weight_loader.py` - 权重加载测试
- `test_tokenizer.py` - Tokenizer 测试
- `test_inference_core.py` - 推理引擎测试
- `test_arrow_engine.py` - ArrowEngine 集成测试

### 性能测试
- `benchmark_loading.py` - 加载时间基准
- `benchmark_inference.py` - 推理延迟基准
- `benchmark_throughput.py` - 吞吐量基准
- `benchmark_memory.py` - 内存占用分析

### 集成测试
- `test_e2e_conversion.py` - 转换 + 推理端到端测试
- `test_multiple_models.py` - 多模型兼容性测试
- `test_concurrent.py` - 并发推理测试

## 下一步

1. 创建 `llm_compression/inference/` 目录结构
2. 实现 `WeightLoader` 类 (T2.2)
3. 实现 `FastTokenizer` 类 (T2.3)
4. 实现 `InferenceCore` 类 (T2.4)
5. 整合为 `ArrowEngine` 主类
6. 编写单元测试
7. 运行性能基准测试
8. 优化和调优

---

**设计版本**: v1.0
**最后更新**: 2026-02-17
**作者**: ArrowEngine Team

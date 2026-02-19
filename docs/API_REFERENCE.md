# Multimodal Encoder System - API Reference

## Table of Contents

1. [MultimodalEmbeddingProvider](#multimodalembeddingprovider)
2. [MultimodalStorage](#multimodalstorage)
3. [VisionEncoder](#visionencoder)
4. [AudioEncoder](#audioencoder)
5. [CLIPEngine](#clipengine)

---

## MultimodalEmbeddingProvider

The main interface for multimodal encoding operations.

### Initialization

```python
from llm_compression.multimodal import get_multimodal_provider

provider = get_multimodal_provider(
    text_model_path: Optional[str] = None,
    vision_model_path: Optional[str] = None,
    audio_model_path: Optional[str] = None,
    clip_model_path: Optional[str] = None,
    device: Optional[str] = None
)
```

**Parameters:**
- `text_model_path`: Path to BERT text encoder model (optional)
- `vision_model_path`: Path to CLIP vision encoder model (optional)
- `audio_model_path`: Path to Whisper audio encoder model (optional)
- `clip_model_path`: Path to CLIP dual-encoder model (optional)
- `device`: Device to use ("cpu", "cuda", or None for auto-detection)

**Returns:** `MultimodalEmbeddingProvider` instance

### Text Encoding Methods

#### encode(text, normalize=True)

Encode a single text string to embedding.

**Parameters:**
- `text` (str): Text string to encode
- `normalize` (bool): Whether to L2-normalize the embedding (default: True)

**Returns:** `np.ndarray` of shape (384,)

**Example:**
```python
embedding = provider.encode("Hello, world!")
print(embedding.shape)  # (384,)
```

#### encode_batch(texts, normalize=True, batch_size=32)

Encode multiple texts in batches.

**Parameters:**
- `texts` (List[str]): List of text strings to encode
- `normalize` (bool): Whether to L2-normalize embeddings (default: True)
- `batch_size` (int): Batch size for processing (default: 32)

**Returns:** `np.ndarray` of shape (n_texts, 384)

**Example:**
```python
texts = ["text1", "text2", "text3"]
embeddings = provider.encode_batch(texts)
print(embeddings.shape)  # (3, 384)
```


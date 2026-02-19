# Multimodal Encoder System - Complete API Reference

## Overview

This document provides complete API documentation for the Multimodal Encoder System, including all public classes, methods, and their usage.

## Table of Contents

1. [MultimodalEmbeddingProvider](#multimodalembeddingprovider)
2. [MultimodalStorage](#multimodalstorage)
3. [Helper Functions](#helper-functions)

---

## MultimodalEmbeddingProvider

The main interface for multimodal encoding operations. Provides unified access to text, vision, and audio encoders.

### Initialization

```python
from llm_compression.multimodal import get_multimodal_provider

provider = get_multimodal_provider(
    text_model_path="D:/ai-models/bert-base-uncased",
    vision_model_path="D:/ai-models/clip-vit-b32",
    audio_model_path="D:/ai-models/whisper-base",
    device="cpu"
)
```

**Parameters:**
- `text_model_path` (Optional[str]): Path to BERT text encoder model
- `vision_model_path` (Optional[str]): Path to CLIP vision encoder model
- `audio_model_path` (Optional[str]): Path to Whisper audio encoder model
- `clip_model_path` (Optional[str]): Path to CLIP dual-encoder model
- `device` (Optional[str]): Device ("cpu", "cuda", or None for auto-detection)

**Returns:** `MultimodalEmbeddingProvider` instance

---

### Text Encoding Methods

#### encode(text, normalize=True)

Encode a single text string to embedding vector.

**Parameters:**
- `text` (str): Text string to encode
- `normalize` (bool): L2-normalize the embedding (default: True)

**Returns:** `np.ndarray` of shape (384,)

**Example:**
```python
embedding = provider.encode("Hello, world!")
# Shape: (384,)
```

#### encode_batch(texts, normalize=True, batch_size=32)

Encode multiple texts efficiently in batches.

**Parameters:**
- `texts` (List[str]): List of text strings
- `normalize` (bool): L2-normalize embeddings (default: True)
- `batch_size` (int): Batch size for processing (default: 32)

**Returns:** `np.ndarray` of shape (n_texts, 384)

**Example:**
```python
texts = ["cat", "dog", "bird"]
embeddings = provider.encode_batch(texts)
# Shape: (3, 384)
```

---

### Vision Encoding Methods

#### encode_image(images, normalize=True)

Encode images to embedding vectors using CLIP Vision Transformer.

**Parameters:**
- `images` (np.ndarray): Images array of shape (batch, 224, 224, 3), dtype uint8
- `normalize` (bool): L2-normalize embeddings (default: True)

**Returns:** `np.ndarray` of shape (batch, 512)

**Example:**
```python
import numpy as np
from PIL import Image

# Load image
img = Image.open("photo.jpg").resize((224, 224))
img_array = np.array(img)[np.newaxis, ...]

# Encode
embedding = provider.encode_image(img_array)
# Shape: (1, 512)
```

**Notes:**
- Images must be RGB format (3 channels)
- Images must be resized to 224x224 pixels
- Input dtype should be uint8 (0-255 range)

---

### Audio Encoding Methods

#### encode_audio(audio, normalize=True)

Encode audio waveforms to embedding vectors using Whisper encoder.

**Parameters:**
- `audio` (np.ndarray): Audio waveform(s) of shape (batch, n_samples) or (n_samples,)
- `normalize` (bool): L2-normalize embeddings (default: True)

**Returns:** `np.ndarray` of shape (batch, 512) or (512,)

**Example:**
```python
import librosa

# Load audio (16kHz required)
audio, sr = librosa.load("speech.wav", sr=16000)

# Encode
embedding = provider.encode_audio(audio)
# Shape: (512,)
```

**Notes:**
- Audio must be 16kHz sample rate
- Audio should be mono (single channel)
- Maximum audio length: 30 seconds

---

### Multimodal Methods

#### encode_multimodal(text=None, images=None, audio=None)

Encode multiple modalities in a single call.

**Parameters:**
- `text` (Optional[Union[str, List[str]]]): Text(s) to encode
- `images` (Optional[np.ndarray]): Images to encode
- `audio` (Optional[np.ndarray]): Audio to encode

**Returns:** Dict with keys "text", "vision", "audio" containing embeddings

**Example:**
```python
result = provider.encode_multimodal(
    text=["cat", "dog"],
    images=image_array,
    audio=audio_array
)

print(result["text"].shape)    # (2, 384)
print(result["vision"].shape)  # (n_images, 512)
print(result["audio"].shape)   # (n_audio, 512)
```

---

### Cross-Modal Methods (CLIP)

#### compute_cross_modal_similarity(texts, images)

Compute similarity matrix between texts and images using CLIP.

**Parameters:**
- `texts` (List[str]): List of text strings
- `images` (np.ndarray): Images array (batch, 224, 224, 3)

**Returns:** `np.ndarray` of shape (n_texts, n_images) with similarity scores

**Example:**
```python
texts = ["a cat", "a dog", "a bird"]
similarity = provider.compute_cross_modal_similarity(texts, images)
# Shape: (3, n_images)
```

#### find_best_image_matches(texts, images, top_k=5)

Find best matching images for each text query.

**Parameters:**
- `texts` (List[str]): Text queries
- `images` (np.ndarray): Images to search
- `top_k` (int): Number of top matches (default: 5)

**Returns:** List of lists containing top-k image indices for each text

**Example:**
```python
texts = ["a cat sitting"]
matches = provider.find_best_image_matches(texts, images, top_k=3)
# Returns: [[idx1, idx2, idx3]]
```

#### find_best_text_matches(images, texts, top_k=5)

Find best matching texts for each image.

**Parameters:**
- `images` (np.ndarray): Query images
- `texts` (List[str]): Texts to search
- `top_k` (int): Number of top matches (default: 5)

**Returns:** List of lists containing top-k text indices for each image

#### zero_shot_classification(images, class_labels)

Perform zero-shot image classification using text labels.

**Parameters:**
- `images` (np.ndarray): Images to classify
- `class_labels` (List[str]): List of class labels

**Returns:** Tuple of (predicted_indices, probabilities)

**Example:**
```python
labels = ["cat", "dog", "bird"]
predictions, probs = provider.zero_shot_classification(images, labels)
# predictions: array of class indices
# probs: array of probabilities (n_images, n_classes)
```

---

### Utility Methods

#### get_available_modalities()

Get list of available modalities based on loaded models.

**Returns:** List[str] containing "text", "vision", "audio", "clip"

**Example:**
```python
modalities = provider.get_available_modalities()
# Returns: ["text", "vision", "audio"]
```

#### dimension

Property that returns text embedding dimension.

**Returns:** int (384 for BERT)

#### vision_dimension

Property that returns vision embedding dimension.

**Returns:** int (512 for CLIP ViT)

#### audio_dimension

Property that returns audio embedding dimension.

**Returns:** int (512 for Whisper)

#### clip_dimension

Property that returns CLIP shared space dimension.

**Returns:** int (512)

---

## MultimodalStorage

Storage layer for multimodal embeddings with separate tables for each modality.

### Initialization

```python
from llm_compression.multimodal import MultimodalStorage

storage = MultimodalStorage(
    storage_path="./embeddings",
    compression_level=3
)
```

**Parameters:**
- `storage_path` (str): Directory path for storage
- `compression_level` (int): Zstandard compression level (default: 3)

---

### Storage Methods

#### store_vision_embedding(embedding_id, image_id, embedding, model, metadata=None)

Store a vision embedding.

**Parameters:**
- `embedding_id` (str): Unique embedding identifier
- `image_id` (str): Source image identifier
- `embedding` (np.ndarray): Embedding vector (512-dim)
- `model` (str): Model name (e.g., "clip-vit-b32")
- `metadata` (Optional[Dict]): Additional metadata

**Example:**
```python
storage.store_vision_embedding(
    embedding_id="emb_001",
    image_id="img_123",
    embedding=vision_emb,
    model="clip-vit-b32",
    metadata={"source": "camera"}
)
```

#### store_audio_embedding(embedding_id, audio_id, embedding, model, duration=None, metadata=None)

Store an audio embedding.

**Parameters:**
- `embedding_id` (str): Unique embedding identifier
- `audio_id` (str): Source audio identifier
- `embedding` (np.ndarray): Embedding vector (512-dim)
- `model` (str): Model name (e.g., "whisper-base")
- `duration` (Optional[float]): Audio duration in seconds
- `metadata` (Optional[Dict]): Additional metadata

#### store_clip_embedding(embedding_id, source_id, modality, embedding, model, metadata=None)

Store a CLIP embedding (text or image in shared space).

**Parameters:**
- `embedding_id` (str): Unique embedding identifier
- `source_id` (str): Source identifier
- `modality` (str): "text" or "image"
- `embedding` (np.ndarray): Embedding vector (512-dim)
- `model` (str): Model name
- `metadata` (Optional[Dict]): Additional metadata

---

### Query Methods

#### query_vision_embeddings(image_ids=None, limit=None)

Query vision embeddings.

**Parameters:**
- `image_ids` (Optional[List[str]]): Filter by image IDs
- `limit` (Optional[int]): Maximum number of results

**Returns:** pyarrow.Table with vision embeddings

#### query_audio_embeddings(audio_ids=None, limit=None)

Query audio embeddings.

**Parameters:**
- `audio_ids` (Optional[List[str]]): Filter by audio IDs
- `limit` (Optional[int]): Maximum number of results

**Returns:** pyarrow.Table with audio embeddings

#### query_clip_embeddings(modality=None, limit=None)

Query CLIP embeddings.

**Parameters:**
- `modality` (Optional[str]): Filter by modality ("text" or "image")
- `limit` (Optional[int]): Maximum number of results

**Returns:** pyarrow.Table with CLIP embeddings

---

### Utility Methods

#### get_storage_stats()

Get storage statistics for all modalities.

**Returns:** Dict with statistics for each modality

**Example:**
```python
stats = storage.get_storage_stats()
print(stats["vision"]["count"])  # Number of vision embeddings
print(stats["audio"]["size_mb"])  # Storage size in MB
```

---

## Helper Functions

### get_multimodal_provider()

Factory function to create MultimodalEmbeddingProvider instance.

See [MultimodalEmbeddingProvider Initialization](#initialization) for details.

---

## Error Handling

All methods may raise the following exceptions:

- `ValueError`: Invalid input parameters
- `RuntimeError`: Model not loaded or operation failed
- `FileNotFoundError`: Model files not found
- `MemoryError`: Insufficient memory

**Example:**
```python
try:
    embedding = provider.encode_image(invalid_image)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Encoding failed: {e}")
```

---

## Performance Tips

1. **Batch Processing**: Use `encode_batch()` for multiple texts
2. **Lazy Loading**: Models load only when first used
3. **Device Selection**: Use "cuda" for GPU acceleration
4. **Normalization**: Keep `normalize=True` for similarity computations
5. **Memory Management**: Process large batches in chunks

---

## Version Information

- API Version: 1.0
- Text Encoder: BERT base (384-dim)
- Vision Encoder: CLIP ViT-B/32 (512-dim)
- Audio Encoder: Whisper base (512-dim)
- CLIP Space: 512-dim shared


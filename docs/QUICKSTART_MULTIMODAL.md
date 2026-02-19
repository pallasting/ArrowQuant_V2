# Multimodal Encoder System - Complete Quickstart Guide

## Table of Contents

1. [Installation](#installation)
2. [Model Conversion](#model-conversion)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0+
- 8GB+ RAM (16GB recommended)
- Optional: CUDA-capable GPU for acceleration

### Step 1: Install Dependencies

```bash
# Clone repository (if not already done)
cd llm_compression

# Install all dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### Step 2: Verify Installation

```bash
# Run environment validation
python -c "import torch; import pyarrow; import numpy; print('Installation successful!')"
```

---

## Model Conversion

Before using the multimodal encoders, convert pretrained models from HuggingFace to Arrow/Parquet format.

### Unified Converter (Recommended)

The unified converter automatically detects model type and provides the best user experience:

```bash
# Convert CLIP vision model (auto-detect)
python scripts/convert_model.py \
    --model openai/clip-vit-base-patch32 \
    --output D:/ai-models/clip-vit-b32

# Convert Whisper audio model (auto-detect)
python scripts/convert_model.py \
    --model openai/whisper-base \
    --output D:/ai-models/whisper-base

# Convert BERT text model (auto-detect)
python scripts/convert_model.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output D:/ai-models/minilm
```

**Advanced Options:**

```bash
# Explicit model type
python scripts/convert_model.py \
    --model openai/clip-vit-base-patch32 \
    --output D:/ai-models/clip-vit-b32 \
    --type clip

# High compression (slower, smaller files)
python scripts/convert_model.py \
    --model openai/whisper-base \
    --output D:/ai-models/whisper-base \
    --compression-level 9

# Skip validation (faster)
python scripts/convert_model.py \
    --model openai/clip-vit-base-patch32 \
    --output D:/ai-models/clip-vit-b32 \
    --no-validate

# Keep float32 precision
python scripts/convert_model.py \
    --model openai/whisper-base \
    --output D:/ai-models/whisper-base \
    --no-float16
```

**Expected Output:**
```
======================================================================
  ArrowEngine Model Converter
  Convert HuggingFace models to optimized Arrow/Parquet format
======================================================================

Configuration:
  Model: openai/clip-vit-base-patch32
  Output: D:/ai-models/clip-vit-b32
  Type: auto
  Float16: True
  Compression: zstd (level 3)
  Validate: True

Starting conversion of openai/clip-vit-base-patch32...
Auto-detected model type: clip
Loading CLIP model...
Extracting vision encoder weights...
Converting to Arrow/Parquet format...

======================================================================
  Conversion Summary
======================================================================
  Model:           openai/clip-vit-base-patch32
  Output:          D:/ai-models/clip-vit-b32
  Parameters:      87,849,216
  File size:       167.56 MB
  Compression:     2.7x
  Time:            12.34 seconds
  Validation:      PASSED
======================================================================

✅ SUCCESS: Model converted successfully

Next steps:
  1. Load the model: ArrowEngine.from_pretrained('D:/ai-models/clip-vit-b32')
  2. See examples: examples/multimodal_complete_examples.py
  3. Read docs: docs/QUICKSTART_MULTIMODAL.md
```

### Legacy Scripts (Deprecated)

⚠️ **Note**: The standalone conversion scripts are deprecated and will be removed in version 2.0.0.
Please use `scripts/convert_model.py` instead.

For backward compatibility, the old scripts still work:

```bash
# DEPRECATED - Use convert_model.py instead
python scripts/convert_clip_to_parquet.py \
    --model_name openai/clip-vit-base-patch32 \
    --output_dir D:/ai-models/clip-vit-b32

# DEPRECATED - Use convert_model.py instead
python scripts/convert_whisper_to_parquet.py \
    --model_name openai/whisper-base \
    --output_dir D:/ai-models/whisper-base
```

### Model Specifications

**CLIP ViT-B/32:**
- Embedding dimension: 512
- Load time: ~1.7s
- Inference: ~45ms per image
- File size: ~168 MB (zstd level 3)

**Whisper Base:**
- Embedding dimension: 512
- Load time: ~0.2s
- Inference: ~200ms per 3s audio
- File size: ~39 MB (zstd level 3)

**BERT MiniLM-L6:**
- Embedding dimension: 384
- Load time: ~0.1s
- Inference: ~10ms per sentence
- File size: ~23 MB (zstd level 3)

### Text Model (Optional)

If you need text encoding, ensure BERT model is available:

```bash
# Convert text model using unified converter
# Check if it exists at: D:/ai-models/bert-base-uncased
```

---

## Basic Usage

### Example 1: Text Encoding

```python
from llm_compression.multimodal import get_multimodal_provider

# Initialize provider
provider = get_multimodal_provider(
    text_model_path="D:/ai-models/bert-base-uncased",
    device="cpu"
)

# Encode single text
embedding = provider.encode("Hello, world!")
print(f"Shape: {embedding.shape}")  # (384,)

# Encode batch
texts = ["cat", "dog", "bird"]
embeddings = provider.encode_batch(texts)
print(f"Batch shape: {embeddings.shape}")  # (3, 384)
```

### Example 2: Image Encoding

```python
import numpy as np
from PIL import Image
from llm_compression.multimodal import get_multimodal_provider

# Initialize provider
provider = get_multimodal_provider(
    vision_model_path="D:/ai-models/clip-vit-b32",
    device="cpu"
)

# Load and preprocess image
image = Image.open("photo.jpg").resize((224, 224))
image_array = np.array(image)[np.newaxis, ...]  # Add batch dim

# Encode
embedding = provider.encode_image(image_array, normalize=True)
print(f"Shape: {embedding.shape}")  # (1, 512)
```

### Example 3: Audio Encoding

```python
import librosa
from llm_compression.multimodal import get_multimodal_provider

# Initialize provider
provider = get_multimodal_provider(
    audio_model_path="D:/ai-models/whisper-base",
    device="cpu"
)

# Load audio (must be 16kHz)
audio, sr = librosa.load("speech.wav", sr=16000)

# Encode
embedding = provider.encode_audio(audio, normalize=True)
print(f"Shape: {embedding.shape}")  # (512,)
```

### Example 4: All Modalities

```python
from llm_compression.multimodal import get_multimodal_provider

# Initialize with all models
provider = get_multimodal_provider(
    text_model_path="D:/ai-models/bert-base-uncased",
    vision_model_path="D:/ai-models/clip-vit-b32",
    audio_model_path="D:/ai-models/whisper-base",
    device="cpu"
)

# Check available modalities
modalities = provider.get_available_modalities()
print(f"Available: {modalities}")  # ['text', 'vision', 'audio']

# Encode all at once
result = provider.encode_multimodal(
    text=["cat", "dog"],
    images=image_array,
    audio=audio_array
)

print(f"Text: {result['text'].shape}")
print(f"Vision: {result['vision'].shape}")
print(f"Audio: {result['audio'].shape}")
```

---

## Advanced Features

### Cross-Modal Similarity (CLIP)

Find images that match text descriptions:

```python
# Initialize with text and vision models
provider = get_multimodal_provider(
    text_model_path="D:/ai-models/bert-base-uncased",
    vision_model_path="D:/ai-models/clip-vit-b32"
)

# Prepare data
texts = ["a cat sitting", "a dog running", "a bird flying"]
images = load_images()  # Shape: (N, 224, 224, 3)

# Compute similarity matrix
similarity = provider.compute_cross_modal_similarity(texts, images)
print(f"Similarity shape: {similarity.shape}")  # (3, N)

# Find best matches
matches = provider.find_best_image_matches(texts, images, top_k=5)
for i, text in enumerate(texts):
    print(f"'{text}' matches: {matches[i]}")
```

### Zero-Shot Classification

Classify images without training:

```python
# Prepare images and labels
images = load_images()  # Shape: (N, 224, 224, 3)
labels = ["cat", "dog", "bird", "car", "tree"]

# Classify
predictions, probabilities = provider.zero_shot_classification(
    images, labels
)

for i, pred_idx in enumerate(predictions):
    print(f"Image {i}: {labels[pred_idx]} ({probabilities[i, pred_idx]:.2%})")
```

### Storage and Retrieval

Store embeddings for later use:

```python
from llm_compression.multimodal import MultimodalStorage

# Initialize storage
storage = MultimodalStorage(
    storage_path="./embeddings",
    compression_level=3
)

# Store vision embedding
storage.store_vision_embedding(
    embedding_id="img_001",
    image_id="photo_123",
    embedding=vision_emb,
    model="clip-vit-b32",
    metadata={"source": "camera", "timestamp": "2026-02-19"}
)

# Query embeddings
results = storage.query_vision_embeddings(
    image_ids=["photo_123"],
    limit=10
)

# Get statistics
stats = storage.get_storage_stats()
print(f"Vision embeddings: {stats['vision']['count']}")
print(f"Storage size: {stats['vision']['size_mb']:.2f} MB")
```

---

## Performance Optimization

### Use GPU Acceleration

```python
# Automatically detect and use GPU
provider = get_multimodal_provider(
    vision_model_path="D:/ai-models/clip-vit-b32",
    device="cuda"  # or None for auto-detection
)
```

### Batch Processing

Process multiple items efficiently:

```python
# Process images in batches
batch_size = 16
all_embeddings = []

for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    embeddings = provider.encode_image(batch)
    all_embeddings.append(embeddings)

all_embeddings = np.vstack(all_embeddings)
```

### Lazy Loading

Models load only when first used:

```python
# Initialize provider (no models loaded yet)
provider = get_multimodal_provider(
    text_model_path="D:/ai-models/bert-base-uncased",
    vision_model_path="D:/ai-models/clip-vit-b32",
    audio_model_path="D:/ai-models/whisper-base"
)

# Text encoder loads on first text encoding
text_emb = provider.encode("hello")  # Loads text model

# Vision encoder loads on first image encoding
image_emb = provider.encode_image(image)  # Loads vision model
```

### Memory Management

```python
# Process large datasets in chunks
def process_large_dataset(images, chunk_size=100):
    for i in range(0, len(images), chunk_size):
        chunk = images[i:i+chunk_size]
        embeddings = provider.encode_image(chunk)
        
        # Save to disk immediately
        save_embeddings(embeddings, f"chunk_{i}.npy")
        
        # Free memory
        del embeddings
```

---

## Troubleshooting

### Issue 1: Model Not Found

**Error:** `FileNotFoundError: Model files not found`

**Solution:**
```bash
# Verify model path
ls D:/ai-models/clip-vit-b32/

# Re-run conversion if needed
python scripts/convert_clip_to_parquet.py \
    --model_name openai/clip-vit-base-patch32 \
    --output_dir D:/ai-models/clip-vit-b32
```

### Issue 2: Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Reduce batch size
embeddings = provider.encode_image(images, batch_size=8)

# Or use CPU
provider = get_multimodal_provider(
    vision_model_path="D:/ai-models/clip-vit-b32",
    device="cpu"
)
```

### Issue 3: Invalid Image Dimensions

**Error:** `ValueError: Expected image shape (224, 224, 3)`

**Solution:**
```python
from PIL import Image
import numpy as np

# Resize image
image = Image.open("photo.jpg").resize((224, 224))

# Convert to RGB if needed
if image.mode != 'RGB':
    image = image.convert('RGB')

# Convert to numpy array
image_array = np.array(image)
```

### Issue 4: Audio Sample Rate

**Error:** `ValueError: Audio must be 16kHz`

**Solution:**
```python
import librosa

# Load with correct sample rate
audio, sr = librosa.load("audio.wav", sr=16000)

# Or resample existing audio
audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
```

### Issue 5: Import Errors

**Error:** `ModuleNotFoundError: No module named 'llm_compression'`

**Solution:**
```bash
# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/llm_compression"
```

---

## Performance Benchmarks

### Model Loading Times

| Model | Size | Load Time | Device |
|-------|------|-----------|--------|
| BERT Text | 43 MB | ~1.3s | CPU |
| CLIP Vision | 168 MB | ~1.7s | CPU |
| Whisper Audio | 39 MB | ~0.2s | CPU |

### Inference Speed

| Operation | Latency | Throughput | Device |
|-----------|---------|------------|--------|
| Text encoding | <5ms | 200+ texts/s | CPU |
| Image encoding | ~45ms | 20+ images/s | CPU |
| Audio encoding | ~200ms | 5+ clips/s | CPU |

### Memory Usage

| Configuration | Memory | Notes |
|---------------|--------|-------|
| Text only | ~50 MB | Minimal |
| Text + Vision | ~220 MB | Recommended |
| All modalities | ~250 MB | Full system |

---

## Next Steps

1. **Explore Examples**: Check `examples/multimodal_complete_examples.py`
2. **Read API Reference**: See `docs/API_REFERENCE_COMPLETE.md`
3. **Run Tests**: `pytest tests/unit/test_vision_encoder.py`
4. **Optimize Performance**: See performance tuning section above

---

## Additional Resources

- **Design Document**: `.kiro/specs/multimodal-encoder-system/design.md`
- **Requirements**: `.kiro/specs/multimodal-encoder-system/requirements.md`
- **Progress Report**: `MULTIMODAL_SYSTEM_PROGRESS.md`
- **Integration Guide**: `TASK_11_INTEGRATION_COMPLETE.md`

---

## Support

For issues or questions:
1. Check `TECHNICAL_DEBT.md` for known issues
2. Review `TROUBLESHOOTING.md` for common problems
3. Run validation: `python scripts/test_multimodal_integration.py`


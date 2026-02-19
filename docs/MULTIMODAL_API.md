# Multimodal Encoder System - API Documentation

## Overview

The Multimodal Encoder System extends ArrowEngine with vision and audio encoding capabilities, providing a unified interface for encoding text, images, and audio into dense vector representations.

## Core Components

### MultimodalEmbeddingProvider

The main entry point for multimodal encoding operations.

```python
from llm_compression.multimodal import get_multimodal_provider

# Initialize provider
provider = get_multimodal_provider(
    text_model_path="D:/ai-models/bert-base-uncased",
    vision_model_path="D:/ai-models/clip-vit-b32",
    audio_model_path="D:/ai-models/whisper-base",
    device="cpu"  # or "cuda"
)
```

#### Methods

##### encode(text: str | List[str]) -> np.ndarray

Encode text to embeddings (backward compatible with ArrowEngine).

**Parameters:**
- `text`: Single string or list of strings to encode
- Returns: (n_texts, 384) numpy array of text embeddings

**Example:**
```python
# Single text
embedding = provider.encode("Hello, world!")
# Shape: (384,)

# Batch of texts
embeddings = provider.encode(["text1", "text2", "text3"])
# Shape: (3, 384)
```

##### encode_image(images: np.ndarray, normalize: bool = True) -> np.ndarray

Encode images to embeddings using CLIP Vision Transformer.

**Parameters:**
- `images`: (batch, 224, 224, 3) numpy array of RGB images (uint8)
- `normalize`: Whether to L2-normalize embeddings (default: True)
- Returns: (batch, 512) numpy array of image embeddings

**Example:**
```python
import numpy as np
from PIL import Image

# Load and preprocess image
image = Image.open("photo.jpg").resize((224, 224))
image_array = np.array(image)[np.newaxis, ...]  # Add batch dimension

# Encode
embedding = provider.encode_image(image_array)
# Shape: (1, 512)
```


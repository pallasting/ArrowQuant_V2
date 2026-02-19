# Multimodal Encoder System - Quickstart Guide

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- PyArrow
- NumPy

### Install Dependencies

```bash
# Install from requirements
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

## Model Conversion

Before using the multimodal encoders, you need to convert pretrained models from HuggingFace format to Arrow/Parquet format.

### Convert CLIP Model

```bash
python scripts/convert_clip_to_parquet.py \
    --model_name openai/clip-vit-base-patch32 \
    --output_dir D:/ai-models/clip-vit-b32
```

**Output:**
- Model size: ~168 MB (compressed)
- Embedding dimension: 512
- Load time: ~1.7s

### Convert Whisper Model

```bash
python scripts/convert_whisper_to_parquet.py \
    --model_name openai/whisper-base \
    --output_dir D:/ai-models/whisper-base
```

**Output:**
- Model size: ~39 MB (compressed)
- Embedding dimension: 512
- Load time: ~0.2s

## Basic Usage

### Text Encoding (Backward Compatible)

```python
from llm_compression.multimodal import get_multimodal_provider

provider = get_multimodal_provider(
    text_model_path="D:/ai-models/bert-base-uncased"
)

# Encode text
embedding = provider.encode("Hello, world!")
print(embedding.shape)  # (384,)
```


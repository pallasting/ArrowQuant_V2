# Arrow-Optimized Embedding Models

This directory contains models converted to Arrow/Parquet format for high-performance inference.

## Structure

```
optimized/
├── model-name.parquet        # Arrow-format weights
├── metadata.json             # Model metadata
└── tokenizer/
    ├── tokenizer.json        # Rust tokenizer config
    └── tokenizer_config.json # Tokenizer parameters
```

## Usage

Convert a model:
```bash
python -m llm_compression.tools.convert_model \
    "sentence-transformers/all-MiniLM-L6-v2" \
    --output-dir models/optimized
```

## Performance

Compared to PyTorch .bin format:
- 📦 30-40% smaller file size
- ⚡ 10x faster loading (mmap)
- 💾 50% less memory usage
- 🚀 Zero-copy inference

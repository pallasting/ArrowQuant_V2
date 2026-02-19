# AGENTS.md - Agentic Coding Guidelines

This file provides guidance for AI agents working in this repository.

## Project Overview

This is an **LLM Compression System** for AI-OS memory. It uses Large Language Models to achieve 10-50x compression ratios on conversation and memory data while preserving semantic fidelity.

## Commands

### Installation
```bash
pip install -r requirements.txt
pip install -e .  # Editable mode
```

### Running Tests

**Run all tests:**
```bash
pytest
```

**Run specific test categories:**
```bash
pytest tests/unit/          # Unit tests
pytest tests/property/      # Property-based tests
pytest tests/integration/   # Integration tests
pytest tests/performance/   # Performance benchmarks
```

**Run a single test:**
```bash
pytest tests/unit/test_config.py::TestConfig::test_default_config
pytest -k "test_name_pattern"  # By pattern
```

**With coverage:**
```bash
pytest --cov=llm_compression tests/
```

### Code Quality

**Format code:**
```bash
black llm_compression/ tests/
```

**Lint:**
```bash
flake8 llm_compression/ tests/
```

**Type check:**
```bash
mypy llm_compression/
```

### Running Demos
```bash
python prototype/simple_demo.py
python prototype/generative.py
```

## Code Style Guidelines

### Imports

Organize imports in the following order (blank line between groups):
1. Standard library (`os`, `asyncio`, `typing`, `pathlib`, etc.)
2. Third-party packages (`yaml`, `pytest`, `pandas`, etc.)
3. Local packages (`llm_compression.*`)

```python
# Example
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
import pytest

from llm_compression.logger import logger
from llm_compression.errors import CompressionError
from llm_compression.config import Config
```

### Type Annotations

- Use type hints for all function parameters and return types
- Use `Optional[X]` instead of `X | None`
- Use explicit return type annotations (`-> None`, `-> Dict[str, Any]`)
- Use dataclasses for structured data containers

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List

@dataclass
class ModelConfig:
    name: str
    endpoint: str
    timeout: float = 30.0
    max_retries: int = 3

def process_data(config: ModelConfig) -> Dict[str, Any]:
    ...
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `LLMCompressor`, `Config`)
- **Functions/variables**: `snake_case` (e.g., `get_config`, `max_retries`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_BUFFER_SIZE`)
- **Private methods**: Prefix with underscore (e.g., `_internal_method`)

### Docstrings

- Use docstrings for all public classes and functions
- Follow Google-style or NumPy-style format
- Include Args, Returns, Raises sections for complex functions

```python
def compress_memory(data: str, config: Config) -> CompressedMemory:
    """
    Compress memory data using LLM.
    
    Args:
        data: Raw memory data to compress
        config: Configuration object
        
    Returns:
        CompressedMemory: Compressed representation
        
    Raises:
        CompressionError: If compression fails
    """
```

### Error Handling

- Use custom exception hierarchy from `llm_compression.errors`
- Base class: `CompressionError`
- Specific exceptions: `LLMAPIError`, `LLMTimeoutError`, `ReconstructionError`, `QualityError`, `StorageError`, `GPUResourceError`, `ConfigurationError`
- Include context in exceptions (not just message)

```python
from llm_compression.errors import LLMAPIError

raise LLMAPIError(
    message="API request failed",
    endpoint="http://localhost:8045",
    status_code=500,
    original_exception=e
)
```

### Dataclasses

Use `@dataclass` for data containers:
- Configuration classes
- Data transfer objects (DTOs)
- Return value structures

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class CompressionMetadata:
    original_size: int
    compressed_size: int
    compression_ratio: float
    model_used: str
    quality_score: float
    compression_time_ms: float
    compressed_at: datetime = field(default_factory=datetime.now)
```

### Testing

- Test files: `tests/unit/test_*.py`, `tests/integration/test_*.py`
- Test classes: `Test*` (e.g., `class TestConfig:`)
- Test functions: `test_*` (e.g., `def test_default_config(self):`)
- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`

```python
import pytest

class TestCompressor:
    def test_compress_basic(self):
        """Test basic compression"""
        compressor = LLMCompressor()
        result = compressor.compress("test data")
        assert result is not None
```

### Logging

Use the centralized logger from `llm_compression.logger`:

```python
from llm_compression.logger import logger

logger.info("Processing started")
logger.warning("Fallback to simple compression")
logger.error(f"Compression failed: {e}")
```

### Configuration

- Use YAML files for configuration
- Support environment variable overrides
- Provide sensible defaults
- Validate configuration on load

```python
# Load from YAML
config = Config.from_yaml("config.yaml")

# Apply env overrides
config.apply_env_overrides()

# Validate
config.validate()
```

## Project Structure

```
llm_compression/       # Main package
├── __init__.py       # Exports public API
├── config.py         # Configuration management
├── logger.py         # Logging setup
├── errors.py         # Exception hierarchy
├── compressor.py     # Core compression
├── reconstructor.py  # Memory reconstruction
├── llm_client.py     # LLM API client
├── model_selector.py # Model selection
├── quality_evaluator.py # Quality metrics
└── ...

tests/
├── unit/            # Unit tests
├── property/        # Property-based tests
├── integration/     # Integration tests
└── performance/     # Benchmarks

prototype/           # Experimental code
docs/               # Documentation
```

## Key Technologies

- **Python**: 3.10+
- **LLM**: OpenAI API, Ollama (local models)
- **Data**: PyArrow, Pandas, NumPy
- **Compression**: Zstandard
- **API**: FastAPI
- **Testing**: pytest, hypothesis

## Configuration File

Copy `config.example.yaml` to `config.yaml` and customize:
```bash
cp config.example.yaml config.yaml
```

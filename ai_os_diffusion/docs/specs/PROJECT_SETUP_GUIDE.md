# Project Setup Guide: AI-OS Unified Diffusion

## Quick Start

This guide will help you set up the new `ai-os-diffusion` project and begin the migration process.

## Prerequisites

- Python 3.10+
- Git
- 8GB+ RAM recommended
- Current `ai-os-memory` project as reference

## Step 1: Create New Project Directory

```bash
# Navigate to parent directory
cd M:\Documents\

# Create new project
mkdir ai-os-diffusion
cd ai-os-diffusion

# Initialize git
git init
```

## Step 2: Create Directory Structure

```bash
# Create main package directories
mkdir -p diffusion_engine/core
mkdir -p diffusion_engine/heads
mkdir -p diffusion_engine/conditioning
mkdir -p diffusion_engine/controlnet
mkdir -p inference
mkdir -p storage
mkdir -p evolution
mkdir -p config
mkdir -p utils
mkdir -p models/diffusion-base/heads
mkdir -p models/lora_cards
mkdir -p models/controlnets
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/property
mkdir -p scripts
mkdir -p docs

# Create __init__.py files
touch diffusion_engine/__init__.py
touch diffusion_engine/core/__init__.py
touch diffusion_engine/heads/__init__.py
touch diffusion_engine/conditioning/__init__.py
touch diffusion_engine/controlnet/__init__.py
touch inference/__init__.py
touch storage/__init__.py
touch evolution/__init__.py
touch config/__init__.py
touch utils/__init__.py
touch tests/__init__.py
```

## Step 3: Create Initial Files

### Create requirements.txt

```bash
cat > requirements.txt << 'EOF'
# Core dependencies
torch>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0
pandas>=2.0.0

# Transformers and tokenization
transformers>=4.30.0
tokenizers>=0.13.0
sentencepiece>=0.1.99

# Quantization
bitsandbytes>=0.41.0

# Configuration and utilities
pyyaml>=6.0
python-dotenv>=1.0.0
tqdm>=4.65.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
hypothesis>=6.82.0

# Code quality
black>=23.7.0
flake8>=6.1.0
mypy>=1.5.0

# Optional: Image/Audio processing
# Pillow>=10.0.0
# soundfile>=0.12.0
# librosa>=0.10.0
EOF
```

### Create setup.py

```bash
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="ai-os-diffusion",
    version="0.1.0",
    description="Unified Diffusion Architecture for AI-OS",
    author="AI-OS Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyarrow>=12.0.0",
        "pandas>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "hypothesis>=6.82.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "multimodal": [
            "Pillow>=10.0.0",
            "soundfile>=0.12.0",
            "librosa>=0.10.0",
        ],
    },
)
EOF
```

### Create README.md

```bash
cat > README.md << 'EOF'
# AI-OS Unified Diffusion Architecture

**Version**: 0.1.0 (In Development)
**Status**: Alpha

Unified diffusion architecture replacing autoregressive models with thermodynamic diffusion for multimodal generation (text, code, image, audio).

## Key Features

- **Unified Score Network**: Single Transformer backbone for all modalities
- **Memory-Guided Generation**: Personal memory drives content generation via ArrowStorage
- **5-Level Evolution**: Progressive self-evolution from score composition to full fine-tuning
- **Edge Deployment**: INT2 quantization for <35MB models on mobile devices
- **Zero-Copy Loading**: Parquet V2 format with ArrowQuant compression

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest

# Generate text (once implemented)
python scripts/generate_text.py "Write a poem about spring"
```

## Project Structure

```
ai-os-diffusion/
├── diffusion_engine/      # Core diffusion components
│   ├── core/             # Score network, diffusion core, samplers
│   ├── heads/            # Modality projection heads
│   ├── conditioning/     # Memory conditioning, uncertainty
│   └── controlnet/       # ControlNet implementations
├── inference/            # Inference engine (migrated from ai-os-memory)
├── storage/              # ArrowStorage (migrated)
├── evolution/            # LoRA, ControlNet training (migrated)
├── config/               # Configuration management
├── utils/                # Utilities (logger, errors)
├── models/               # Model weights
├── tests/                # Test suite
└── scripts/              # Utility scripts
```

## Documentation

- [Requirements](docs/UNIFIED_DIFFUSION_REQUIREMENTS.md)
- [Design](docs/UNIFIED_DIFFUSION_DESIGN.md)
- [Migration Guide](docs/MIGRATION_CHECKLIST.md)
- [API Reference](docs/API_REFERENCE.md) (Coming soon)

## Development Status

See [tasks.md](.kiro/specs/unified-diffusion-architecture/tasks.md) for implementation progress.

**Current Phase**: Phase 0 - Project Setup and Module Migration

## License

MIT License
EOF
```

### Create .gitignore

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.hypothesis/

# Models (large files)
models/**/*.parquet
models/**/*.bin
models/**/*.safetensors
!models/.gitkeep

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.bak
EOF
```

### Create pytest.ini

```bash
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    property: Property-based tests
    slow: Slow tests (deselect with '-m "not slow"')
EOF
```

## Step 4: Copy Migration Files

```bash
# Copy spec files from old project
cp ../ai-os-memory-optimization/.kiro/specs/unified-diffusion-architecture/*.md docs/

# Create .kiro directory for spec tracking
mkdir -p .kiro/specs/unified-diffusion-architecture
cp ../ai-os-memory-optimization/.kiro/specs/unified-diffusion-architecture/*.md .kiro/specs/unified-diffusion-architecture/
```

## Step 5: Begin Module Migration (Task 0.2-0.5)

### Migrate Inference Module

```bash
# Copy inference files
cp -r ../ai-os-memory-optimization/llm_compression/inference/*.py inference/

# Files to copy:
# - arrow_engine.py
# - weight_loader.py
# - fast_tokenizer.py
# - inference_core.py
# - __init__.py
```

### Migrate Storage Module

```bash
# Copy storage files
cp ../ai-os-memory-optimization/llm_compression/arrow_storage.py storage/
cp ../ai-os-memory-optimization/llm_compression/arrow_native_compressor.py storage/

# Create storage __init__.py
cat > storage/__init__.py << 'EOF'
"""
Storage module for ArrowStorage and memory management.
"""

from storage.arrow_storage import ArrowStorage, StorageError, SCHEMA_REGISTRY
from storage.arrow_native_compressor import ArrowNativeCompressor

__all__ = [
    "ArrowStorage",
    "StorageError",
    "SCHEMA_REGISTRY",
    "ArrowNativeCompressor",
]
EOF
```

### Migrate Evolution Module

```bash
# Copy evolution files
mkdir -p evolution
cp ../ai-os-memory-optimization/llm_compression/evolution/lora_trainer.py evolution/
cp ../ai-os-memory-optimization/llm_compression/evolution/skill_factory.py evolution/
cp ../ai-os-memory-optimization/llm_compression/evolution/weight_probe.py evolution/

# Create evolution __init__.py
cat > evolution/__init__.py << 'EOF'
"""
Evolution module for LoRA training and skill management.
"""

from evolution.lora_trainer import LoRATrainer
from evolution.skill_factory import SkillFactory
from evolution.weight_probe import WeightMapProbe

__all__ = [
    "LoRATrainer",
    "SkillFactory",
    "WeightMapProbe",
]
EOF
```

### Migrate Config and Utils

```bash
# Copy config and utils
cp ../ai-os-memory-optimization/llm_compression/config.py config/
cp ../ai-os-memory-optimization/llm_compression/logger.py utils/
cp ../ai-os-memory-optimization/llm_compression/errors.py utils/
cp ../ai-os-memory-optimization/llm_compression/embedding_provider.py utils/

# Create config __init__.py
cat > config/__init__.py << 'EOF'
"""
Configuration management.
"""

from config.config import Config

__all__ = ["Config"]
EOF

# Create utils __init__.py
cat > utils/__init__.py << 'EOF'
"""
Utility modules.
"""

from utils.logger import setup_logger, logger
from utils.errors import (
    DiffusionError,
    ModelLoadError,
    GenerationError,
    QuantizationError,
)
from utils.embedding_provider import (
    EmbeddingProvider,
    ArrowEngineProvider,
    get_default_provider,
)

__all__ = [
    "setup_logger",
    "logger",
    "DiffusionError",
    "ModelLoadError",
    "GenerationError",
    "QuantizationError",
    "EmbeddingProvider",
    "ArrowEngineProvider",
    "get_default_provider",
]
EOF
```

## Step 6: Fix Imports

After copying files, you'll need to update imports:

```python
# Old imports (ai-os-memory)
from llm_compression.config import Config
from llm_compression.logger import logger
from llm_compression.errors import CompressionError

# New imports (ai-os-diffusion)
from config.config import Config
from utils.logger import logger
from utils.errors import DiffusionError
```

Create a migration script:

```bash
cat > scripts/fix_imports.py << 'EOF'
#!/usr/bin/env python3
"""
Fix imports after migration from ai-os-memory to ai-os-diffusion.
"""

import os
import re
from pathlib import Path

# Import mapping
IMPORT_MAP = {
    "from llm_compression.config import": "from config.config import",
    "from llm_compression.logger import": "from utils.logger import",
    "from llm_compression.errors import": "from utils.errors import",
    "from llm_compression.embedding_provider import": "from utils.embedding_provider import",
    "from llm_compression.arrow_storage import": "from storage.arrow_storage import",
    "from llm_compression.arrow_native_compressor import": "from storage.arrow_native_compressor import",
    "from llm_compression.inference": "from inference",
    "from llm_compression.evolution": "from evolution",
}

def fix_imports_in_file(filepath):
    """Fix imports in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    for old_import, new_import in IMPORT_MAP.items():
        content = content.replace(old_import, new_import)
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed imports in: {filepath}")
        return True
    return False

def main():
    """Fix imports in all Python files."""
    root = Path(".")
    python_files = list(root.rglob("*.py"))
    
    fixed_count = 0
    for filepath in python_files:
        if ".git" in str(filepath) or "venv" in str(filepath):
            continue
        if fix_imports_in_file(filepath):
            fixed_count += 1
    
    print(f"\nFixed imports in {fixed_count} files.")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/fix_imports.py
```

## Step 7: Install and Test

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
# source venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"

# Fix imports
python scripts/fix_imports.py

# Run tests (will fail initially, that's expected)
pytest tests/ -v
```

## Step 8: Verify Migration

Create a verification script:

```bash
cat > scripts/verify_migration.py << 'EOF'
#!/usr/bin/env python3
"""
Verify that all essential modules were migrated correctly.
"""

import sys
from pathlib import Path

REQUIRED_FILES = [
    "inference/arrow_engine.py",
    "inference/weight_loader.py",
    "inference/fast_tokenizer.py",
    "inference/inference_core.py",
    "storage/arrow_storage.py",
    "storage/arrow_native_compressor.py",
    "evolution/lora_trainer.py",
    "evolution/skill_factory.py",
    "config/config.py",
    "utils/logger.py",
    "utils/errors.py",
    "utils/embedding_provider.py",
]

def verify_migration():
    """Verify all required files exist."""
    missing = []
    for filepath in REQUIRED_FILES:
        if not Path(filepath).exists():
            missing.append(filepath)
    
    if missing:
        print("❌ Migration incomplete. Missing files:")
        for filepath in missing:
            print(f"  - {filepath}")
        return False
    
    print("✅ All essential files migrated successfully!")
    
    # Try importing
    print("\nVerifying imports...")
    try:
        from config.config import Config
        print("  ✅ Config")
    except ImportError as e:
        print(f"  ❌ Config: {e}")
        return False
    
    try:
        from utils.logger import logger
        print("  ✅ Logger")
    except ImportError as e:
        print(f"  ❌ Logger: {e}")
        return False
    
    try:
        from utils.errors import DiffusionError
        print("  ✅ Errors")
    except ImportError as e:
        print(f"  ❌ Errors: {e}")
        return False
    
    print("\n✅ Migration verification complete!")
    return True

if __name__ == "__main__":
    success = verify_migration()
    sys.exit(0 if success else 1)
EOF

chmod +x scripts/verify_migration.py
python scripts/verify_migration.py
```

## Next Steps

After completing the setup:

1. ✅ **Verify migration**: Run `python scripts/verify_migration.py`
2. ✅ **Fix any import errors**: Run `python scripts/fix_imports.py`
3. ✅ **Update error classes**: Rename `CompressionError` → `DiffusionError`
4. ✅ **Write initial tests**: Create basic unit tests for migrated modules
5. ✅ **Begin Phase 1**: Start implementing core diffusion components (Task 1.1)

## Troubleshooting

### Import Errors

If you see import errors like `ModuleNotFoundError: No module named 'llm_compression'`:
- Run `python scripts/fix_imports.py`
- Manually check and fix any remaining imports

### Missing Dependencies

If you see missing dependency errors:
- Check `requirements.txt` includes all needed packages
- Run `pip install -r requirements.txt` again

### Test Failures

Initial test failures are expected:
- Migrated tests may reference old module names
- Some tests may need adaptation for new architecture
- Focus on fixing one module at a time

## Reference

- **Old Project**: `M:\Documents\ai-os-memory-optimization\`
- **New Project**: `M:\Documents\ai-os-diffusion\`
- **Spec Location**: `.kiro/specs/unified-diffusion-architecture/`
- **Migration Checklist**: `MIGRATION_CHECKLIST.md`
- **Task List**: `tasks.md`

---

*Last Updated: 2026-02-21*

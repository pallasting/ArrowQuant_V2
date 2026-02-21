# AI-OS Unified Diffusion Architecture

🦴 **Rust Skeleton** + 🧠 **Python Brain** = 🤖 **Self-Evolving AI-OS**

## Overview

This is the next-generation AI-OS built on unified diffusion architecture with a revolutionary dual-layer design:

- **🦴 Rust Skeleton**: Stable, high-performance infrastructure (10-50x speedup)
  - ArrowStorage, ArrowQuant, VectorSearch, FastTokenizer
  
- **🧠 Python Brain**: Flexible, evolution-friendly learning layer
  - DiffusionCore, EvolutionRouter, MemoryConditioner, UncertaintyEstimator

## Quick Start

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build Rust components (requires Rust toolchain)
cd rust
cargo build --release
cd ..

# Install in editable mode
pip install -e .
```

### Basic Usage

```python
from ai_os_diffusion import ArrowEngine

# Initialize engine
engine = ArrowEngine("config.yaml")

# Text generation (Diffusion mode)
text = engine.diffuse("写一首春天的诗", modality="text")

# Memory-guided generation
text = engine.diffuse(
    "回忆我去年的日本之旅",
    modality="text",
    memory_guided=True
)

# Multimodal generation
outputs = engine.diffuse(
    "帮我回忆日本之旅",
    modality=["text", "audio", "image"]
)
```

## Project Structure

```
ai-os-diffusion/
├── docs/specs/              # 📚 Complete specification documents
│   ├── README.md           # Spec overview
│   ├── requirements.md     # 18 functional requirements
│   ├── design.md           # System architecture
│   ├── tasks.md            # 56 implementation tasks
│   ├── ARROWENGINE_ARCHITECTURE.md  # ⭐ ArrowEngine详解
│   └── ...                 # Other architecture docs
│
├── rust/                   # 🦴 Rust Skeleton (Infrastructure)
│   ├── arrow_storage/      # Vector storage & retrieval
│   ├── arrow_quant/        # INT2/INT4 quantization
│   ├── vector_search/      # SIMD similarity search
│   └── fast_tokenizer/     # Parallel tokenization
│
├── diffusion_engine/       # 🧠 Python Brain (Core Logic)
│   ├── core/               # DiffusionCore, NoiseScheduler, Samplers
│   ├── conditioning/       # MemoryConditioner, UncertaintyEstimator
│   └── heads/              # Projection heads (text, image, audio)
│
├── inference/              # 🧠 ArrowEngine & routing
├── evolution/              # 🧠 EvolutionRouter, LoRA, ControlNet
├── storage/                # 🧠 Python wrapper for Rust ArrowStorage
├── config/                 # Configuration management
├── utils/                  # Utilities (logger, errors)
├── tests/                  # Test suites
├── scripts/                # Utility scripts
└── models/                 # Model weights storage
```

## Documentation

### 📋 Start Here

1. **[docs/specs/README.md](docs/specs/README.md)** - Spec overview and roadmap
2. **[docs/specs/ARROWENGINE_ARCHITECTURE.md](docs/specs/ARROWENGINE_ARCHITECTURE.md)** - ⭐ ArrowEngine 详细架构
3. **[docs/specs/ARCHITECTURE_PHILOSOPHY.md](docs/specs/ARCHITECTURE_PHILOSOPHY.md)** - Rust Skeleton + Python Brain 哲学

### 🏗️ Architecture

- **[design.md](docs/specs/design.md)** - Complete system design
- **[RUST_MIGRATION_STRATEGY.md](docs/specs/RUST_MIGRATION_STRATEGY.md)** - Rust components strategy
- **[BALANCED_EVOLUTION_STRATEGY.md](docs/specs/BALANCED_EVOLUTION_STRATEGY.md)** - Evolution approach

### 📝 Implementation

- **[tasks.md](docs/specs/tasks.md)** - 56 tasks across 5 phases
- **[requirements.md](docs/specs/requirements.md)** - 18 functional requirements
- **[MIGRATION_CHECKLIST.md](docs/specs/MIGRATION_CHECKLIST.md)** - Migration guide

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Property-based tests
pytest tests/property/
```

### Building Rust Components

```bash
cd rust
cargo build --release
cargo test
```

### Code Quality

```bash
# Format Python code
black ai_os_diffusion/ tests/

# Lint Python code
flake8 ai_os_diffusion/ tests/

# Type check
mypy ai_os_diffusion/

# Format Rust code
cd rust && cargo fmt
```

## Architecture Highlights

### Dual-Layer Design

```
用户代码
   ↓
ArrowEngine (🧠 Python - 统一入口)
   ↓
DiffusionCore, EvolutionRouter (🧠 Python - 业务逻辑)
   ↓
ArrowStorage, ArrowQuant, FastTokenizer (🦴 Rust - 基础设施)
```

### Key Features

- ✅ **Unified Generation**: One model for text, code, image, audio
- ✅ **Memory-Guided**: Personal memories guide generation
- ✅ **Self-Evolution**: 5-level progressive learning (L0-L4)
- ✅ **Edge Deployment**: <35MB with INT2 quantization
- ✅ **High Performance**: 10-50x speedup with Rust infrastructure

### Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Text generation (4-step, INT2, CPU) | <500ms | 🎯 Target |
| Image generation (512×512, CPU) | <30s | 🎯 Target |
| TTS latency | <2s | 🎯 Target |
| Edge model size (INT2) | <35MB | 🎯 Target |
| Rust ArrowStorage speedup | 10-50x | 🎯 Target |
| Rust ArrowQuant speedup | 5-10x | 🎯 Target |

## Roadmap

### Phase 0: Setup (Week 1-2) ⏳ Current
- [x] Create project structure
- [x] Migrate spec documents
- [ ] Set up Rust workspace
- [ ] Implement Rust infrastructure
- [ ] Migrate Python modules

### Phase 1: Text Generation (Week 3-4)
- [ ] Implement DiffusionCore
- [ ] Implement NoiseScheduler & Samplers
- [ ] Integrate ArrowEngine.diffuse()
- [ ] Validate text generation quality

### Phase 2: Memory-Guided (Week 5-6)
- [ ] Implement UnifiedScoreNetwork
- [ ] Implement MemoryConditioner
- [ ] Implement UncertaintyEstimator
- [ ] Implement EvolutionRouter (L0-L2)

### Phase 3: Multimodal (Week 7-9)
- [ ] Implement image generation
- [ ] Implement audio generation
- [ ] Multimodal parallel generation
- [ ] Edge deployment validation

### Phase 4: Advanced (Week 10-11)
- [ ] ControlNet training
- [ ] EBM constraints
- [ ] L3 selective fine-tuning
- [ ] Self-evolution closed loop

### Phase 5: Production (Week 12)
- [ ] Documentation
- [ ] Performance optimization
- [ ] Integration testing
- [ ] Production deployment

## Contributing

This is a research project. For implementation details, see:
- [docs/specs/tasks.md](docs/specs/tasks.md) - Task list
- [docs/specs/PROJECT_SETUP_GUIDE.md](docs/specs/PROJECT_SETUP_GUIDE.md) - Setup guide

## License

[Your License Here]

## Citation

If you use this work, please cite:

```bibtex
@software{aios_unified_diffusion,
  title={AI-OS Unified Diffusion Architecture},
  author={[Your Name]},
  year={2026},
  url={https://github.com/[your-repo]}
}
```

---

**Status**: 🚧 Phase 0 - Active Development  
**Version**: 0.1.0  
**Last Updated**: 2026-02-21

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup & Installation
- **Install dependencies**: `pip install -r requirements.txt`
- **Install package**: `pip install -e .` (Editable mode recommended for development)
- **Configuration**: Copy `config.example.yaml` to `config.yaml` to customize settings (e.g., LLM endpoints).

### Testing
- **Run all tests**: `pytest`
- **Run unit tests**: `pytest tests/unit`
- **Run property tests**: `pytest tests/property`
- **Run integration tests**: `pytest tests/integration`
- **Run performance tests**: `pytest tests/performance`
- **Check coverage**: `pytest --cov=llm_compression tests/`

### Code Quality
- **Format**: `black llm_compression/ tests/`
- **Lint**: `flake8 llm_compression/ tests/`
- **Type Check**: `mypy llm_compression/`

### Execution & Demos
- **Run Prototype Demo**: `python prototype/simple_demo.py`
- **Run Generative Prototype**: `python prototype/generative.py`

## Project State

> **Always read `STATUS.md` first** for the authoritative project status.
> Do NOT rely on `docs/vision/LEGACY_OVERVIEW.md` — it is severely out of date.

## Architecture & Structure

### Dual-Mainline Overview

This project has **two parallel development mainlines** sharing the `llm_compression/` package:

**Mainline A — ArrowEngine Capability Expansion**
```
Phase 0 (ArrowEngine Core) ✅ → Phase 7 (LoRA) ✅ → Phase 8 (Federation) ✅
→ Phase 9 (Self-Evolution) ✅ → Phase 10 (Dashboard) ✅
→ Phase 11 (Multimodal Sensors) 🔄 → Phase 12 (Knowledge Graph) 🔄
```

**Mainline B — LLM Memory Compression & Cognition**
```
Phase 1.0 (LLM Compression 10-50x) ✅ → Phase 1.1 (Quality Fix) ✅
→ Phase 2.0 (Self-Organizing Cognitive Architecture) ✅
→ Phase 2.0+ (Dialogue Agent MVP, Task 45) ✅
→ Phase 3 (Hybrid Model Architecture) ⏳ not started
```

### Codebase Structure
- **`llm_compression/`**: Core Python package — 116 files, ~19,500 LOC total.
  - `config.py`, `logger.py`: Configuration and logging.
  - `compressor.py`, `reconstructor.py`, `llm_client.py`: Mainline B compression core.
  - `memory_primitive.py`, `connection_learner.py`, `cognitive_loop.py`: Cognitive architecture (Phase 2.0).
  - `expression_layer.py`, `network_navigator.py`: Expression and navigation layers.
  - `inference/`: ArrowEngine core (LoRA, InferenceCore, EmbeddingProvider).
  - `federation/`: Phase 8 distributed federation (Arrow Flight).
  - `evolution/`: Phase 9 self-evolution (SkillDistiller, LoRATrainer).
  - `multimodal/`: Phase 10-11 multimodal (CLIP, audio, vision).
  - `sensors/`: Phase 11 sensor integration (vision, audio).
  - `knowledge_graph/`: Phase 12 knowledge graph navigation.
- **`prototype/`**: Experimental scripts used for validation before production.
- **`tests/`**: Four-layer hierarchical test suite:
  - `unit/`: Isolated component tests.
  - `property/`: Property-based robustness testing.
  - `integration/`: Cross-component and external system tests.
  - `performance/`: Benchmarks for compression and latency.
  - `performance/`: Benchmarks for compression ratios and speed.

### Key Technologies
- **LLM Integration**: Uses both cloud APIs and local models (via `ollama`, `transformers`) for semantic processing.
- **Data Handling**: Uses `pyarrow` for efficient, zero-copy data manipulation where possible.
- **Embeddings**: Utilizes `sentence-transformers` for semantic similarity and deduplication.
- **OpenClaw Compatibility**: Designed to interface with the OpenClaw memory system.

### Configuration
The system is driven by `config.yaml`, which controls:
- LLM backend selection (Cloud vs. Local).
- API endpoints and keys.
- Compression parameters and thresholds.

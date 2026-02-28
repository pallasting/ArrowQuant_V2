# Python Code Migration Status

## Overview

This document tracks the migration of essential Python modules from `llm_compression` to `ai-os-diffusion`.

**Migration Date**: 2026-02-21  
**Status**: In Progress

---

## Completed Migrations

### ✅ Configuration & Utilities (Phase 0.8)

| Source File | Destination | Status | Notes |
|------------|-------------|--------|-------|
| `llm_compression/config.py` | `ai-os-diffusion/config/config.py` | ✅ DONE | Updated for diffusion architecture |
| `llm_compression/logger.py` | `ai-os-diffusion/utils/logger.py` | ✅ DONE | No changes needed |
| `llm_compression/errors.py` | `ai-os-diffusion/utils/errors.py` | ✅ DONE | Simplified for diffusion use cases |

**Changes Made**:
- Updated `config.py` to include `DiffusionConfig` and `EvolutionConfig`
- Removed old LLM compression-specific configs
- Updated import paths from `llm_compression.*` to `ai_os_diffusion.*`
- Simplified error hierarchy to focus on diffusion-specific errors

---

## Pending Migrations

### ✅ Inference Module (Phase 0.6) - HIGH PRIORITY (Partial)

| Source File | Destination | Status | Notes |
|------------|-------------|--------|-------|
| `llm_compression/inference/arrow_engine.py` | `ai-os-diffusion/inference/arrow_engine.py` | ✅ DONE | Simplified Phase 0 version |
| `llm_compression/inference/weight_loader.py` | `ai-os-diffusion/inference/weight_loader.py` | ⏳ PENDING | Keep as-is (Rust backend) |
| `llm_compression/inference/fast_tokenizer.py` | `ai-os-diffusion/inference/fast_tokenizer.py` | ⏳ PENDING | Keep as-is (Rust backend) |
| `llm_compression/inference/inference_core.py` | `ai-os-diffusion/inference/inference_core.py` | ⏳ PENDING | Base for SharedTransformer |

**Completed Changes**:
- ✅ Created simplified ArrowEngine with `diffuse()` method stub
- ✅ Added mode routing architecture (AR vs Diffusion)
- ✅ Updated all imports to use `ai_os_diffusion.*`
- ✅ Removed LoRA router, federation, sensors (Phase 0 focus)
- ✅ Kept core structure for future expansion

**Pending**:
- ⏳ Migrate weight_loader.py, fast_tokenizer.py, inference_core.py
- ⏳ Implement encode() method (needs inference_core)
- ⏳ Implement diffuse() method (needs DiffusionCore from Phase 1)

### 🔄 Storage Module - HIGH PRIORITY

| Source File | Destination | Status | Notes |
|------------|-------------|--------|-------|
| `llm_compression/arrow_storage.py` | `ai-os-diffusion/storage/arrow_storage.py` | ⏳ PENDING | Will use Rust backend |
| `llm_compression/arrow_native_compressor.py` | `ai-os-diffusion/storage/arrow_native_compressor.py` | ⏳ PENDING | Optional (MEDIUM priority) |

### 🔄 Evolution Module (Phase 0.7) - MEDIUM PRIORITY

| Source File | Destination | Status | Notes |
|------------|-------------|--------|-------|
| `llm_compression/evolution/lora_trainer.py` | `ai-os-diffusion/evolution/lora_trainer.py` | ⏳ PENDING | L2 evolution level |
| `llm_compression/evolution/skill_factory.py` | `ai-os-diffusion/evolution/skill_factory.py` | ⏳ PENDING | Skill card management |
| `llm_compression/evolution/weight_probe.py` | `ai-os-diffusion/evolution/weight_probe.py` | ⏳ PENDING | Weight inspection |

### 🔄 Embedding Provider - HIGH PRIORITY

| Source File | Destination | Status | Notes |
|------------|-------------|--------|-------|
| `llm_compression/embedding_provider.py` | `ai-os-diffusion/utils/embedding_provider.py` | ⏳ PENDING | Needed for memory conditioning |

---

## Files NOT Being Migrated

The following files are **NOT** needed for the unified diffusion architecture:

### ❌ Old Compression Architecture
- `compressor.py` - LLM-based compression (replaced by diffusion)
- `reconstructor.py` - LLM-based reconstruction (not needed)
- `llm_client.py` - API calls (diffusion is local)
- `model_selector.py` - Model selection (unified model)
- `quality_evaluator.py` - Quality metrics (replaced by uncertainty estimation)

### ❌ Old Cognitive Loop
- `cognitive_loop.py` - Old AR-based architecture
- `cognitive_loop_arrow.py` - Old AR-based architecture
- `network_navigator.py` - Old Hopfield network
- `network_navigator_arrow.py` - Old Hopfield network

### ❌ Old Agent/Action Systems
- `conversational_agent.py` - Old agent architecture
- `conversation_memory.py` - Old conversation management
- `action/` directory - Phase 12 feature (not needed now)
- `agent/` directory - Phase 13 feature (not needed now)

### ❌ Expression Layer
- `expression_layer.py` - Old multimodal expression
- `expression/` directory - Entire expression module

### ❌ Federation & Cloud
- `federation/` directory - Phase 8 feature (not needed now)
- `evolution/cloud_distiller.py` - Cloud-based distillation (not needed)
- `evolution/skill_distiller.py` - Cloud-based distillation (not needed)

---

## Next Steps

1. **Migrate Inference Module** (arrow_engine.py, weight_loader.py, fast_tokenizer.py, inference_core.py)
   - Update imports to `ai_os_diffusion.*`
   - Add diffusion mode routing to ArrowEngine
   - Remove federation/evolution features from ArrowEngine (Phase 0 focus)

2. **Migrate Storage Module** (arrow_storage.py)
   - Update imports
   - Prepare for Rust backend integration

3. **Migrate Evolution Module** (lora_trainer.py, skill_factory.py, weight_probe.py)
   - Update imports
   - Simplify for Phase 0 needs

4. **Create __init__.py files** for each package
   - Export public APIs
   - Enable clean imports

5. **Update requirements.txt**
   - Remove old dependencies
   - Keep only 5 core packages: torch, numpy, pyarrow, transformers, pyyaml

6. **Test imports**
   - Verify all imports work
   - Run basic smoke tests

---

## Architecture Notes

**Rust Skeleton + Python Brain Philosophy**:
- 🦴 **Rust components** (ArrowStorage, ArrowQuant, FastTokenizer): High-performance infrastructure
- 🧠 **Python components** (DiffusionCore, EvolutionRouter, training): Flexible learning layer

**Phase 0 Focus**:
- Essential infrastructure only
- No federation, no cloud features
- Minimal dependencies
- Fast iteration

---

*Last Updated: 2026-02-21*

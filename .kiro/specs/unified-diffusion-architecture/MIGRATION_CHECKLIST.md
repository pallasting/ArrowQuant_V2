# Module Migration Checklist

## Overview

This document provides a detailed checklist for migrating modules from the current `ai-os-memory` project to the new `ai-os-diffusion` project. Each module is categorized as:
- ✅ **MIGRATE**: Essential for unified diffusion architecture
- ⚠️ **OPTIONAL**: May be useful but not critical
- ❌ **SKIP**: Not needed for diffusion architecture

---

## Core Inference Modules

### ✅ MIGRATE: llm_compression/inference/

| File | Status | Priority | Notes |
|------|--------|----------|-------|
| `arrow_engine.py` | ✅ MIGRATE | HIGH | Extend to support diffusion mode routing |
| `weight_loader.py` | ✅ MIGRATE | HIGH | Already supports Parquet V2 + ArrowQuant |
| `fast_tokenizer.py` | ✅ MIGRATE | HIGH | Needed for text/code tokenization |
| `inference_core.py` | ✅ MIGRATE | HIGH | Base for SharedTransformer |
| `__init__.py` | ✅ MIGRATE | HIGH | Package initialization |

**Migration Notes**:
- ArrowEngine needs new `diffuse()` method
- Add mode routing: `mode="ar"` vs `mode="diffusion"`
- Keep existing `encode()` and `generate()` for backward compatibility

---

## Storage Modules

### ✅ MIGRATE: Storage Components

| File | Status | Priority | Notes |
|------|--------|----------|-------|
| `arrow_storage.py` | ✅ MIGRATE | HIGH | Vector retrieval for memory-guided generation |
| `arrow_native_compressor.py` | ✅ MIGRATE | MEDIUM | Memory compression |
| `arrow_storage_zero_copy.py` | ⚠️ OPTIONAL | LOW | May be redundant with arrow_storage.py |

**Migration Notes**:
- ArrowStorage is critical for MemoryConditioner
- SCHEMA_REGISTRY needed for memory schemas
- Zero-copy optimizations already in arrow_storage.py

---

## Evolution Modules

### ✅ MIGRATE: llm_compression/evolution/

| File | Status | Priority | Notes |
|------|--------|----------|-------|
| `lora_trainer.py` | ✅ MIGRATE | HIGH | L2 evolution level |
| `skill_factory.py` | ✅ MIGRATE | HIGH | Skill card management |
| `weight_probe.py` | ✅ MIGRATE | MEDIUM | Weight inspection utilities |
| `lora_extractor.py` | ⚠️ OPTIONAL | LOW | May not be needed initially |
| `skill_distiller.py` | ⚠️ OPTIONAL | LOW | Cloud-based distillation |
| `cloud_distiller.py` | ❌ SKIP | - | Not needed for initial implementation |

**Migration Notes**:
- LoRATrainer is essential for L2 evolution
- SkillFactory manages LoRA cards
- WeightMapProbe useful for debugging

---

## Configuration and Utilities

### ✅ MIGRATE: Core Utilities

| File | Status | Priority | Notes |
|------|--------|----------|-------|
| `config.py` | ✅ MIGRATE | HIGH | Configuration management |
| `logger.py` | ✅ MIGRATE | HIGH | Logging system |
| `errors.py` | ✅ MIGRATE | HIGH | Exception hierarchy |
| `embedding_provider.py` | ✅ MIGRATE | HIGH | Embedding generation |

**New Files to Create**:
- `diffusion_config.py` - Diffusion-specific configuration

---

## Modules to SKIP

### ❌ SKIP: LLM Compression (Old Architecture)

| File | Status | Reason |
|------|--------|--------|
| `compressor.py` | ❌ SKIP | LLM-based compression not used in diffusion |
| `reconstructor.py` | ❌ SKIP | LLM-based reconstruction not used |
| `llm_client.py` | ❌ SKIP | Diffusion models don't need API calls |
| `model_selector.py` | ❌ SKIP | Unified model, no selection needed |
| `quality_evaluator.py` | ❌ SKIP | Replaced by uncertainty estimation |

### ❌ SKIP: Cognitive Loop (Old Architecture)

| File | Status | Reason |
|------|--------|--------|
| `cognitive_loop.py` | ❌ SKIP | Old AR-based cognitive architecture |
| `cognitive_loop_arrow.py` | ❌ SKIP | Old AR-based cognitive architecture |
| `network_navigator.py` | ❌ SKIP | Old Hopfield network navigation |
| `network_navigator_arrow.py` | ❌ SKIP | Old Hopfield network navigation |

### ❌ SKIP: Conversation and Agent (Old Architecture)

| File | Status | Reason |
|------|--------|--------|
| `conversation_memory.py` | ❌ SKIP | Old conversation management |
| `conversational_agent.py` | ❌ SKIP | Old agent architecture |
| `personalization.py` | ❌ SKIP | Old personalization system |
| `memory_primitive.py` | ❌ SKIP | Old memory primitive system |
| `connection_learner.py` | ❌ SKIP | Old connection learning |
| `internal_feedback.py` | ❌ SKIP | Old feedback system |

### ❌ SKIP: Expression Layer (Old Architecture)

| File | Status | Reason |
|------|--------|--------|
| `expression_layer.py` | ❌ SKIP | Old multimodal expression |
| `llm_compression/expression/` | ❌ SKIP | Entire expression module |

### ❌ SKIP: Batch Processing (Old Architecture)

| File | Status | Reason |
|------|--------|--------|
| `batch_processor.py` | ❌ SKIP | Diffusion has own batching |
| `batch_processor_arrow.py` | ❌ SKIP | Diffusion has own batching |
| `batch_optimizer.py` | ❌ SKIP | Needs redesign for diffusion |

### ❌ SKIP: Performance Monitoring (Needs Redesign)

| File | Status | Reason |
|------|--------|--------|
| `performance_monitor.py` | ❌ SKIP | Needs redesign for diffusion metrics |
| `performance_optimizer.py` | ❌ SKIP | Needs redesign for diffusion |
| `performance_config.py` | ❌ SKIP | Needs redesign for diffusion |

### ❌ SKIP: Miscellaneous

| File | Status | Reason |
|------|--------|--------|
| `visualizer.py` | ⚠️ OPTIONAL | May be useful for debugging |
| `openclaw_interface.py` | ❌ SKIP | OpenClaw integration not needed |
| `api.py` | ❌ SKIP | Old API structure |
| `fallback.py` | ❌ SKIP | Old fallback logic |
| `gpu_fallback.py` | ❌ SKIP | Old GPU fallback |
| `health.py` | ❌ SKIP | Old health check |
| `monitoring.py` | ❌ SKIP | Old monitoring |
| `cost_monitor.py` | ❌ SKIP | Not relevant for diffusion |

### ❌ SKIP: Subdirectories

| Directory | Status | Reason |
|-----------|--------|--------|
| `llm_compression/agent/` | ❌ SKIP | Phase 13, not needed initially |
| `llm_compression/action/` | ❌ SKIP | Phase 12, not needed initially |
| `llm_compression/federation/` | ❌ SKIP | Phase 8, not needed initially |
| `llm_compression/multimodal/` | ⚠️ OPTIONAL | May be useful for input processing |
| `llm_compression/sensors/` | ⚠️ OPTIONAL | May be useful for embodied AI |
| `llm_compression/client/` | ❌ SKIP | Old client architecture |
| `llm_compression/compression/` | ❌ SKIP | Old compression architecture |
| `llm_compression/core/` | ❌ SKIP | Old core architecture |
| `llm_compression/deployment/` | ❌ SKIP | Old deployment scripts |
| `llm_compression/knowledge_graph/` | ❌ SKIP | Not needed for diffusion |
| `llm_compression/learning/` | ❌ SKIP | Old learning architecture |
| `llm_compression/models/` | ❌ SKIP | Old model definitions |
| `llm_compression/server/` | ❌ SKIP | Old server architecture |
| `llm_compression/tools/` | ❌ SKIP | Old tool definitions |

---

## Migration Priority Summary

### Phase 0: Essential Modules (Week 1-2)

**HIGH Priority** (Must migrate first):
1. ✅ `inference/arrow_engine.py` - Core engine
2. ✅ `inference/weight_loader.py` - Weight loading
3. ✅ `inference/fast_tokenizer.py` - Tokenization
4. ✅ `inference/inference_core.py` - Transformer base
5. ✅ `arrow_storage.py` - Memory storage
6. ✅ `config.py` - Configuration
7. ✅ `logger.py` - Logging
8. ✅ `errors.py` - Exceptions
9. ✅ `embedding_provider.py` - Embeddings

**MEDIUM Priority** (Migrate in Phase 0):
10. ✅ `evolution/lora_trainer.py` - LoRA training
11. ✅ `evolution/skill_factory.py` - Skill management
12. ✅ `evolution/weight_probe.py` - Weight inspection
13. ✅ `arrow_native_compressor.py` - Compression

### Phase 1+: Optional Modules

**LOW Priority** (Migrate if needed):
- ⚠️ `visualizer.py` - Debugging visualization
- ⚠️ `multimodal/` - Input processing (if needed)
- ⚠️ `sensors/` - Embodied AI (if needed)

---

## File Count Summary

| Category | Count | Percentage |
|----------|-------|------------|
| ✅ MIGRATE (Essential) | 13 files | ~20% |
| ⚠️ OPTIONAL | 5 files | ~8% |
| ❌ SKIP | 45+ files | ~72% |

**Result**: The new project will be **~70% smaller** than the current one, focusing only on essential diffusion components.

---

## Next Steps

1. ✅ Create new project directory structure (Task 0.1)
2. ✅ Migrate HIGH priority modules (Tasks 0.2-0.5)
3. ✅ Implement new diffusion components (Phase 1-3)
4. ⚠️ Evaluate optional modules as needed
5. ❌ Archive old project as reference

---

## Migration Validation Checklist

After migration, verify:
- [ ] All migrated modules compile without errors
- [ ] Unit tests pass for migrated components
- [ ] ArrowEngine can load models via WeightLoader
- [ ] ArrowStorage can perform vector search
- [ ] LoRATrainer can create LoRA adaptations
- [ ] Configuration loads from YAML files
- [ ] Logging outputs correctly
- [ ] No circular import dependencies

---

*Last Updated: 2026-02-21*

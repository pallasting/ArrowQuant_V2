# Unified Diffusion Architecture - Spec Overview

## Status

**Version**: 2.0  
**Date**: 2026-02-21  
**Status**: ✅ Ready for Implementation  
**Current Phase**: Phase 0 - Rust Skeleton + Python Brain Setup

---

## What is This?

This spec defines the migration from the current AI-OS memory system to a **Unified Diffusion Architecture** with a revolutionary **dual-layer design**:

```
🦴 Rust Skeleton (Arrow基础设施) - 稳定、高效、10-50x性能
        +
🧠 Python Brain (推理进化层) - 灵活、学习、用户个性化
        =
🤖 Self-Evolving AI-OS
```

### Key Innovation

```
Architecture: Rust骨骼 + Python大脑
Old: N modalities = N separate AR models (2+ GB on mobile)
New: 1 Score Network + N projection heads = All modalities (<35 MB on mobile)
Performance: 10-50x faster (Rust infrastructure)
Evolution: Continuous learning (Python brain)
```

---

## Spec Documents

### 📋 Core Documents

1. **[requirements.md](requirements.md)** - 18 functional requirements with acceptance criteria
2. **[design.md](design.md)** - System architecture and module design
3. **[tasks.md](tasks.md)** - 56 implementation tasks across 5 phases (including Rust setup)

### 🏗️ Architecture Documents

4. **[ARROWENGINE_ARCHITECTURE.md](ARROWENGINE_ARCHITECTURE.md)** - ⭐ ArrowEngine 详细架构说明 (统一入口点)
5. **[ARCHITECTURE_PHILOSOPHY.md](ARCHITECTURE_PHILOSOPHY.md)** - Rust Skeleton + Python Brain 设计哲学
6. **[RUST_MIGRATION_STRATEGY.md](RUST_MIGRATION_STRATEGY.md)** - Rust 组件迁移策略
7. **[BALANCED_EVOLUTION_STRATEGY.md](BALANCED_EVOLUTION_STRATEGY.md)** - 平衡进化策略 (Baby Approach)

### 📚 Supporting Documents

8. **[MIGRATION_CHECKLIST.md](MIGRATION_CHECKLIST.md)** - Module-by-module migration guide
9. **[PROJECT_SETUP_GUIDE.md](PROJECT_SETUP_GUIDE.md)** - Step-by-step setup instructions
10. **[FRAMEWORK_DECISIONS.md](FRAMEWORK_DECISIONS.md)** - Framework and technology decisions
11. **[EVOLUTION_FIRST_ANALYSIS.md](EVOLUTION_FIRST_ANALYSIS.md)** - Evolution-first analysis

---

## Quick Summary

### Problem

Current AI-OS uses separate autoregressive models for each modality:
- ❌ High memory usage (N models = N × weights)
- ❌ Cascading delays (text → audio → lip sync is sequential)
- ❌ Context fragmentation (models don't share state)
- ❌ Poor uncertainty estimation (can't detect "I don't know")

### Solution

Unified diffusion architecture with:
- ✅ Single shared Transformer backbone (90% of parameters)
- ✅ Modality-specific projection heads (10% of parameters)
- ✅ Memory-guided generation via ArrowStorage
- ✅ Uncertainty-aware self-evolution (5 levels)
- ✅ Edge deployment with INT2 quantization (<35MB)

---

## Architecture Overview: Dual-Layer Design

```
┌─────────────────────────────────────────────────────────┐
│          🧠 Python Brain (Evolution Layer)              │
│                                                          │
│  Flexible, Learning, User-Specific                      │
│  ┌────────────────────────────────────────────────┐    │
│  │ ArrowEngine (Router & Orchestration)            │    │
│  │ DiffusionCore (Inference Logic)                 │    │
│  │ EvolutionRouter (L0-L4 Learning Strategies)     │    │
│  │ MemoryConditioner (Retrieval Strategies)        │    │
│  │ UncertaintyEstimator (Self-Awareness)           │    │
│  │ ControlNet Bank (Behavior Patterns)             │    │
│  │ LoRA Adaptations (Learned Skills)               │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  Features:                                               │
│  ✅ Frequent updates (every learning cycle)             │
│  ✅ User-specific (personalized brain)                  │
│  ✅ Experimental (new strategies, new knowledge)        │
│  ✅ Hot-reload friendly (Python flexibility)            │
└─────────────────────────────────────────────────────────┘
                          ↕️ PyO3 Interface
┌─────────────────────────────────────────────────────────┐
│          🦴 Rust Skeleton (Infrastructure Layer)        │
│                                                          │
│  Stable, Efficient, Shared                              │
│  ┌────────────────────────────────────────────────┐    │
│  │ ArrowStorage (Memory Engine) - 10-50x faster    │    │
│  │ ArrowQuant (Quantization Engine) - 5-10x faster │    │
│  │ VectorSearch (Retrieval Engine) - 10-50x faster│    │
│  │ FastTokenizer (Text Engine) - 2-5x faster       │    │
│  │ WeightLoader (Loading Engine) - Zero-copy       │    │
│  │ SIMD Kernels (Compute Primitives)               │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  Features:                                               │
│  ✅ Long-term stable (infrastructure rarely changes)    │
│  ✅ Performance critical (10-50x speedup)               │
│  ✅ Universal (shared across all users)                 │
│  ✅ Memory safe (Rust guarantees)                       │
└─────────────────────────────────────────────────────────┘
```

### Why This Architecture?

**Biological Analogy**: Skeleton + Brain
- **Rust Skeleton**: Stable bones that support the body (don't change daily)
- **Python Brain**: Plastic neurons that learn constantly (change every day)

**Engineering Benefits**:
1. **Performance**: Rust provides 10-50x speedup on critical paths
2. **Flexibility**: Python enables rapid iteration and experimentation
3. **Evolution-Friendly**: Brain can learn while skeleton stays stable
4. **Edge Deployment**: Rust compiles to efficient binaries (<35MB)
5. **User Personalization**: Each user's Python brain is unique

---

## Implementation Phases

### Phase 0: Rust Skeleton + Python Brain Setup (Week 1-2) ⏳ Current

**Goal**: Build the stable foundation (Rust) and flexible brain (Python)

**Rust Skeleton Tasks** (🦴 Infrastructure):
- Create Rust workspace structure
- Implement ArrowStorage (arrow-rs) - 10-50x faster retrieval
- Implement ArrowQuant (ndarray + rayon) - 5-10x faster quantization
- Implement VectorSearch (simsimd) - 10-50x faster SIMD search
- Implement FastTokenizer (tokenizers) - 2-5x faster encoding
- Build PyO3 bindings for Python integration

**Python Brain Tasks** (🧠 Evolution Layer):
- Migrate Python inference module (ArrowEngine, InferenceCore)
- Migrate Python storage wrapper (ArrowStorage Python API)
- Migrate Python evolution module (LoRA, SkillFactory)
- Migrate config and utils (logger, errors, embedding_provider)
- Verify Rust-Python integration

**Deliverable**: 
- Rust skeleton providing 10-50x performance boost
- Python brain ready for rapid iteration
- Seamless PyO3 integration

---

### Phase 1: Text Generation (Week 3-4) 📅 Next

**Goal**: Python brain learns text generation using Rust skeleton

**Python Brain Tasks** (🧠 Learning):
- Implement DiffusionCore (denoising logic)
- Implement NoiseScheduler (discrete + continuous)
- Implement DiscreteSampler (text/code sampling)
- Implement TextProjectionHead (language output)
- Integrate ArrowEngine.diffuse() API
- Train/adapt text generation quality

**Rust Skeleton Support** (🦴 Performance):
- Fast tokenization (FastTokenizer)
- Efficient weight loading (WeightLoader)
- Quick memory retrieval (ArrowStorage)

**Deliverable**: 
- Working text generation via diffusion
- PPL within 20% of AR baseline
- 4-step latency < 500ms (Rust acceleration)

---

### Phase 2: Memory-Guided Generation (Week 5-6)

**Goal**: Python brain learns to use memories effectively

**Python Brain Tasks** (🧠 Learning):
- Implement UnifiedScoreNetwork (shared backbone)
- Implement MemoryConditioner (retrieval strategies)
- Implement UncertaintyEstimator (self-awareness)
- Implement EvolutionRouter (L0-L2 learning)
- Learn memory relevance scoring
- Learn evolution strategy selection

**Rust Skeleton Support** (🦴 Performance):
- Ultra-fast vector search (VectorSearch)
- Efficient memory storage (ArrowStorage)
- Quick quantization (ArrowQuant)

**Deliverable**: 
- Memory-guided generation working
- >80% memory relevance rate
- Evolution triggering functional

---

### Phase 3: Multimodal Generation (Week 7-9)

**Goal**: Python brain learns image and audio generation

**Python Brain Tasks** (🧠 Learning):
- Implement ContinuousSampler (image/audio)
- Implement ImageProjectionHead (vision output)
- Implement AudioProjectionHead (audio output)
- Learn multimodal coordination
- Learn efficiency optimizations (step compression)
- Integrate VAE for high-quality images

**Rust Skeleton Support** (🦴 Performance):
- Fast image processing
- Efficient audio I/O
- Parallel multimodal generation

**Deliverable**: 
- Image generation < 30s (512×512)
- Audio generation < 2s (TTS)
- Multimodal parallel generation
- Edge deployment validated (<35MB)

---

### Phase 4: Advanced Evolution (Week 10-12)

**Goal**: Python brain develops meta-learning capabilities

**Python Brain Tasks** (🧠 Meta-Learning):
- Implement ControlNet training (L1)
- Implement EBM constraints
- Implement L3 selective fine-tuning
- Learn learning strategies (meta-learning)
- Learn knowledge transfer
- Learn self-reflection

**Rust Skeleton Support** (🦴 Performance):
- Efficient training kernels
- Fast gradient computation
- Quick checkpoint saving

**Deliverable**: 
- Complete L0-L4 evolution system
- Meta-learning functional
- Self-reflection working
- Production-ready system

---

## Key Metrics & Targets

| Metric | Target | Requirement |
|--------|--------|-------------|
| Text PPL vs AR | < 20% difference | REQ-1 AC5 |
| Text latency (4-step, INT2, CPU) | < 500ms | REQ-1 AC7 |
| Image generation (512×512, CPU) | < 30s | REQ-2 AC5 |
| TTS latency | < 2s | REQ-3 AC3 |
| Shared backbone params | > 90% | REQ-4 AC2 |
| Memory guidance relevance | > 80% | REQ-5 AC5 |
| Uncertainty correlation | > 0.7 | REQ-6 AC5 |
| Edge model size (INT2) | < 35MB | REQ-14 AC1 |

---

## Module Migration Summary

### 🦴 Rust Skeleton (New - Build from scratch)

**Immediate Priority** (Phase 0):
- `ArrowStorage` (arrow-rs) - Vector storage and retrieval
- `ArrowQuant` (ndarray + rayon) - INT2/INT4 quantization
- `VectorSearch` (simsimd) - SIMD-accelerated similarity search
- `FastTokenizer` (tokenizers) - Text tokenization

**Medium Priority** (Phase 2-3):
- `Transformer` (candle) - Inference engine
- `Sampler` (candle) - Diffusion sampling
- `WeightLoader` (memmap2) - Zero-copy weight loading

**Performance Gains**: 10-50x speedup on critical paths

---

### 🧠 Python Brain (Migrate + Extend)

**Migrate from ai-os-memory** (~13 files, ~20%):
- `inference/arrow_engine.py` - Routing and orchestration
- `inference/inference_core.py` - Transformer base (for SharedTransformer)
- `storage/arrow_storage.py` - Python wrapper for Rust ArrowStorage
- `evolution/lora_trainer.py` - L2 LoRA training
- `evolution/skill_factory.py` - Skill management
- `config/config.py` - Configuration
- `utils/logger.py`, `utils/errors.py` - Utilities

**New Python Components** (Build in Python):
- `diffusion_engine/core/diffusion_core.py` - Denoising logic
- `diffusion_engine/core/noise_scheduler.py` - Scheduling
- `diffusion_engine/core/samplers.py` - Sampling strategies
- `diffusion_engine/conditioning/memory_conditioner.py` - Memory retrieval
- `diffusion_engine/conditioning/uncertainty_estimator.py` - Self-awareness
- `diffusion_engine/core/evolution_router.py` - L0-L4 routing
- `diffusion_engine/heads/` - Projection heads (text, image, audio)

---

### ❌ Skip (Not needed - ~45+ files, ~72%)

**Old LLM Compression**:
- `compressor.py`, `reconstructor.py`, `llm_client.py`, `model_selector.py`

**Old Cognitive Loop**:
- `cognitive_loop.py`, `network_navigator.py`, `conversation_memory.py`

**Old Agent/Action**:
- `agent/`, `action/`, `federation/`

**Result**: New project is ~70% smaller, focused on diffusion + evolution

---

## Getting Started

### 1. Read the Spec

- Start with [requirements.md](requirements.md) to understand what we're building
- Review [design.md](design.md) to understand how it works
- Check [tasks.md](tasks.md) to see the implementation plan

### 2. Set Up New Project

Follow [PROJECT_SETUP_GUIDE.md](PROJECT_SETUP_GUIDE.md):

```bash
# Create new project
mkdir ai-os-diffusion
cd ai-os-diffusion

# Follow setup guide to create structure
# Copy essential modules from ai-os-memory
# Fix imports and verify migration
```

### 3. Start Implementation

Begin with Phase 0 tasks:

```bash
# Task 0.1: Create directory structure
# Task 0.2: Migrate inference module
# Task 0.3: Migrate storage module
# Task 0.4: Migrate evolution module
# Task 0.5: Migrate config and utils
# Task 0.6: Verify migration
```

### 4. Track Progress

Update task status in [tasks.md](tasks.md):

```markdown
- [x] 0.1 Create New Project Structure
- [ ] 0.2 Migrate Core Inference Module
- [ ] 0.3 Migrate Storage Module
...
```

---

## Key Design Decisions

### 1. Why Diffusion over Autoregressive?

- **Bidirectional context**: Can do infilling, not just left-to-right
- **Uncertainty quantification**: Noise residuals measure confidence
- **Unified architecture**: One backbone for all modalities
- **Parallel generation**: All modalities from single forward pass

### 2. Why Shared Transformer?

- **Memory efficiency**: 90% parameter sharing across modalities
- **Knowledge transfer**: Cross-modal learning
- **Scalability**: Add new modality = train 10M param head only

### 3. Why 5-Level Evolution?

- **Efficiency**: Start with zero-training (L0), escalate only if needed
- **Flexibility**: Different uncertainty levels → different evolution strategies
- **Practicality**: L0-L2 work on edge, L3-L4 need cloud

### 4. Why ArrowStorage Integration?

- **Personalization**: User memories guide generation
- **Differentiation**: Not just generic generation, but memory-aware
- **Closed loop**: Generation → uncertainty → evolution → storage

---

## Dependencies

### From Existing Project

- ✅ ArrowEngine (inference routing)
- ✅ WeightLoader V2 (Parquet V2 + ArrowQuant)
- ✅ ArrowStorage (vector retrieval)
- ✅ LoRA infrastructure (L2 evolution)

### New Components

- 🆕 DiffusionCore (denoising loop)
- 🆕 UnifiedScoreNetwork (shared backbone)
- 🆕 NoiseScheduler (discrete + continuous)
- 🆕 MemoryConditioner (memory → conditions)
- 🆕 UncertaintyEstimator (evolution trigger)
- 🆕 EvolutionRouter (L0-L4 hub)

---

## Success Criteria

### Phase 0 Complete When:
- [ ] New project structure created
- [ ] Essential modules migrated (13 files)
- [ ] Imports fixed and working
- [ ] Basic tests pass
- [ ] Migration verified

### Phase 1 Complete When:
- [ ] Text generation works via diffusion
- [ ] PPL within 20% of AR baseline
- [ ] 4-step latency < 500ms on CPU
- [ ] Infilling quality validated

### Phase 2 Complete When:
- [ ] Memory-guided generation works
- [ ] Uncertainty estimation functional
- [ ] Evolution triggering works
- [ ] Cross-modal tests pass

### Phase 3 Complete When:
- [ ] Image generation < 30s (512×512)
- [ ] Audio generation < 2s (TTS)
- [ ] Multimodal parallel generation works
- [ ] Edge deployment validated (<35MB)

### Final Success When:
- [ ] All 18 requirements met
- [ ] All 54 tasks complete
- [ ] Documentation complete
- [ ] Production deployed

---

## Questions?

- **Architecture questions**: See [design.md](design.md)
- **Implementation questions**: See [tasks.md](tasks.md)
- **Migration questions**: See [MIGRATION_CHECKLIST.md](MIGRATION_CHECKLIST.md)
- **Setup questions**: See [PROJECT_SETUP_GUIDE.md](PROJECT_SETUP_GUIDE.md)

---

## Timeline

**Total Duration**: 10-12 weeks

- **Week 1-2**: Phase 0 (Setup & Migration)
- **Week 3-4**: Phase 1 (Text Diffusion)
- **Week 5-6**: Phase 2 (Unified Network)
- **Week 7-9**: Phase 3 (Multimodal)
- **Week 10-11**: Phase 4 (Advanced)
- **Week 12**: Phase 5 (Production)

---

*Last Updated: 2026-02-21*
*Spec Version: 1.0*
*Status: Ready for Implementation*

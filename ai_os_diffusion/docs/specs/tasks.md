# Implementation Tasks: Unified Diffusion Architecture

## Task Overview

This document outlines the implementation tasks for migrating from the current AI-OS memory system to the Unified Diffusion Architecture. The implementation follows the **Rust Skeleton + Python Brain** dual-layer philosophy:

- 🦴 **Rust Skeleton**: Stable, high-performance infrastructure (ArrowStorage, ArrowQuant, VectorSearch, FastTokenizer)
- 🧠 **Python Brain**: Flexible, evolution-friendly learning layer (DiffusionCore, EvolutionRouter, training logic)

The implementation follows a phased approach: Project Setup → Module Migration → Core Implementation → Integration & Testing.

---

## Phase 0: Project Setup and Module Migration

### Task 0.1: 🦴 Create Rust Workspace Structure

**Description**: Set up Rust workspace for core infrastructure components using Cargo.

**Subtasks**:
- [ ] 0.1.1 Initialize Rust workspace with `cargo new --lib ai-os-rust`
- [ ] 0.1.2 Create Cargo workspace members: arrow_storage, arrow_quant, vector_search, fast_tokenizer
- [ ] 0.1.3 Configure PyO3 dependencies in Cargo.toml for Python bindings
- [ ] 0.1.4 Configure maturin for building Python wheels
- [ ] 0.1.5 Set up Rust project structure (src/, tests/, benches/)
- [ ] 0.1.6 Create .cargo/config.toml with optimization flags

**Acceptance Criteria**:
- Rust workspace compiles without errors
- PyO3 bindings are configured correctly
- maturin can build Python wheels
- Cargo workspace structure is clean and organized

---

### Task 0.2: 🧠 Create Python Project Structure

**Description**: Set up Python project directory for brain/evolution layer.

**Subtasks**:
- [x] 0.2.1 Create root directory `ai-os-diffusion/`
- [x] 0.2.2 Create core module directories (diffusion_engine/, inference/, storage/, evolution/)
- [~] 0.2.3 Create support directories (config/, utils/, tests/, scripts/, models/)
- [~] 0.2.4 Initialize Python package structure with __init__.py files
- [~] 0.2.5 Create requirements.txt with essential dependencies (torch, numpy, transformers, pyyaml)
- [~] 0.2.6 Create README.md with dual-layer architecture overview

**Acceptance Criteria**:
- Directory structure matches design specification
- All __init__.py files are in place
- requirements.txt includes only 5 core packages: torch, numpy, pyarrow, transformers, pyyaml
- README explains Rust Skeleton + Python Brain philosophy

---

### Task 0.3: 🦴 Implement ArrowStorage in Rust

**Description**: Build high-performance ArrowStorage with vector retrieval in Rust using arrow-rs.

**Subtasks**:
- [ ] 0.3.1 Implement ArrowStorage struct with arrow-rs RecordBatch
- [ ] 0.3.2 Implement vector search using simsimd for SIMD-accelerated similarity
- [ ] 0.3.3 Implement schema definitions (SCHEMA_REGISTRY)
- [ ] 0.3.4 Create PyO3 Python bindings for ArrowStorage
- [ ] 0.3.5 Write Rust unit tests for storage operations
- [ ] 0.3.6 Write Python integration tests calling Rust bindings

**Acceptance Criteria**:
- ArrowStorage can perform vector search with 10-50x speedup vs Python
- Schema registry is accessible from both Rust and Python
- PyO3 bindings work seamlessly from Python
- Integration tests pass for store/retrieve operations

---

### Task 0.4: 🦴 Implement ArrowQuant in Rust

**Description**: Build quantization engine in Rust using ndarray and rayon for parallel processing.

**Subtasks**:
- [ ] 0.4.1 Implement INT2/INT4 quantization algorithms in Rust
- [ ] 0.4.2 Implement dequantization with SIMD optimizations
- [ ] 0.4.3 Implement zero-copy weight loading from Parquet
- [ ] 0.4.4 Create PyO3 Python bindings for ArrowQuant
- [ ] 0.4.5 Write Rust benchmarks for quantization performance
- [ ] 0.4.6 Write Python integration tests

**Acceptance Criteria**:
- Quantization achieves 5-10x speedup vs Python implementation
- Zero-copy loading works correctly
- PyO3 bindings expose quantization functions to Python
- Benchmarks show expected performance gains

---

### Task 0.5: 🦴 Implement FastTokenizer in Rust

**Description**: Build fast tokenizer using Hugging Face tokenizers crate.

**Subtasks**:
- [ ] 0.5.1 Wrap tokenizers crate with custom interface
- [ ] 0.5.2 Implement batch tokenization with parallel processing
- [ ] 0.5.3 Support BPE, WordPiece, and Unigram tokenizers
- [ ] 0.5.4 Create PyO3 Python bindings for FastTokenizer
- [ ] 0.5.5 Write Rust unit tests for tokenization
- [ ] 0.5.6 Benchmark tokenization speed vs Python

**Acceptance Criteria**:
- Tokenization achieves 10-100x speedup vs Python
- Supports all major tokenizer types
- PyO3 bindings work seamlessly from Python
- Unit tests verify correctness

---

### Task 0.6: 🧠 Migrate Python Inference Module

**Description**: Migrate essential inference components from llm_compression/inference/ to new project.

**Subtasks**:
- [~] 0.6.1 Copy and adapt ArrowEngine to support diffusion mode routing
- [~] 0.6.2 Copy WeightLoader V2 with Parquet V2 support (will use Rust ArrowQuant)
- [~] 0.6.3 Integrate Rust FastTokenizer bindings for text/code tokenization
- [~] 0.6.4 Copy InferenceCore as base for SharedTransformer
- [~] 0.6.5 Update imports and dependencies
- [~] 0.6.6 Write unit tests for migrated components

**Acceptance Criteria**:
- All inference components compile without errors
- Unit tests pass for WeightLoader and FastTokenizer integration
- ArrowEngine can route between "ar" and "diffusion" modes
- Rust components are properly integrated via PyO3

---

### Task 0.7: 🧠 Migrate Python Evolution Module

**Description**: Migrate LoRA and skill management components for L2 evolution level.

**Subtasks**:
- [~] 0.7.1 Copy LoRATrainer for L2 fine-tuning
- [~] 0.7.2 Copy SkillFactory for skill card management
- [~] 0.7.3 Copy WeightMapProbe for weight inspection
- [~] 0.7.4 Update imports and dependencies
- [~] 0.7.5 Write unit tests for LoRA operations

**Acceptance Criteria**:
- LoRATrainer can create and apply LoRA adaptations
- SkillFactory can load/save skill cards
- Unit tests pass for evolution components

---

### Task 0.8: 🧠 Migrate Configuration and Utilities

**Description**: Migrate configuration management and utility modules.

**Subtasks**:
- [~] 0.8.1 Copy config.py with configuration management
- [~] 0.8.2 Copy logger.py for logging setup
- [~] 0.8.3 Copy errors.py with exception hierarchy
- [~] 0.8.4 Copy embedding_provider.py for embedding generation
- [~] 0.8.5 Create diffusion_config.py for diffusion-specific settings
- [~] 0.8.6 Write configuration validation tests

**Acceptance Criteria**:
- Config can load from YAML files
- Logger outputs to console and file
- All custom exceptions are defined
- Configuration validation works correctly

---

## Phase 1: Core Diffusion Components (Phase 3a)

### Task 1.1: 🧠 Implement NoiseScheduler

**Description**: Implement unified noise scheduler supporting both discrete and continuous modalities in Python.

**Subtasks**:
- [~] 1.1.1 Implement base NoiseScheduler class
- [~] 1.1.2 Implement cosine schedule for continuous modalities
- [~] 1.1.3 Implement mask schedule for discrete modalities
- [~] 1.1.4 Implement timestep sampling (uniform and non-uniform)
- [~] 1.1.5 Implement add_noise() for forward process
- [~] 1.1.6 Write unit tests for all schedule types

**Acceptance Criteria**:
- NoiseScheduler supports both discrete and continuous modes
- Cosine schedule produces correct σ(t) values
- Mask schedule produces correct β(t) values
- Unit tests achieve >90% coverage

---

### Task 1.2: 🧠 Implement DiscreteSampler

**Description**: Implement discrete sampler for text/code generation using mask-based denoising in Python.

**Subtasks**:
- [~] 1.2.1 Implement DiscreteSampler class
- [~] 1.2.2 Implement step() method for unmask operations
- [~] 1.2.3 Implement confidence-based token selection
- [~] 1.2.4 Implement infilling support (preserve prefix/suffix)
- [~] 1.2.5 Write unit tests for sampling operations
- [~] 1.2.6 Write property-based tests for sampling invariants

**Acceptance Criteria**:
- DiscreteSampler can perform iterative unmasking
- Infilling preserves provided context tokens
- Property tests verify sampling correctness
- Unit tests achieve >85% coverage

---

### Task 1.3: 🧠 Implement TextProjectionHead

**Description**: Implement text projection head for discrete text generation in Python.

**Subtasks**:
- [~] 1.3.1 Implement TextProjectionHead class
- [~] 1.3.2 Implement encode() method (tokens → hidden space)
- [~] 1.3.3 Implement decode() method (hidden space → logits)
- [~] 1.3.4 Implement embedding layer and output projection
- [~] 1.3.5 Write unit tests for forward/backward pass
- [~] 1.3.6 Validate parameter count <10M

**Acceptance Criteria**:
- TextProjectionHead has <10M parameters
- encode() and decode() are inverse operations
- Unit tests verify correct tensor shapes
- Gradient flow works correctly

---

### Task 1.4: 🧠 Implement DiffusionCore (Text-Only)

**Description**: Implement core diffusion inference engine for text generation in Python.

**Subtasks**:
- [~] 1.4.1 Implement DiffusionCore class
- [~] 1.4.2 Implement generate() method for text modality
- [~] 1.4.3 Implement noise initialization for masked sequences
- [~] 1.4.4 Implement denoising loop with DiscreteSampler
- [~] 1.4.5 Integrate with NoiseScheduler
- [~] 1.4.6 Write integration tests for end-to-end generation

**Acceptance Criteria**:
- DiffusionCore can generate coherent text
- Denoising loop completes in expected steps
- Integration tests pass for various prompts
- Memory usage is within acceptable limits

---

### Task 1.5: 🧠 Convert MDLM Weights to Parquet V2

**Description**: Convert open-source MDLM model weights to Parquet V2 format with ArrowQuant (uses Rust ArrowQuant backend).

**Subtasks**:
- [~] 1.5.1 Download MDLM checkpoint from HuggingFace
- [~] 1.5.2 Implement weight conversion script in Python
- [~] 1.5.3 Apply Rust ArrowQuant INT2 quantization via PyO3 bindings
- [~] 1.5.4 Save to Parquet V2 format
- [~] 1.5.5 Create metadata.json with model config
- [~] 1.5.6 Validate converted weights load correctly

**Acceptance Criteria**:
- Converted model size <35MB (INT2)
- Weights load via WeightLoader without errors
- metadata.json contains all required fields
- Validation script confirms weight integrity
- Rust ArrowQuant provides 5-10x speedup

---

### Task 1.6: 🧠 Integrate ArrowEngine.diffuse() for Text

**Description**: Extend ArrowEngine with diffuse() method for text generation in Python.

**Subtasks**:
- [~] 1.6.1 Add diffuse() method to ArrowEngine
- [~] 1.6.2 Implement mode routing (ar vs diffusion)
- [~] 1.6.3 Integrate DiffusionCore for text generation
- [~] 1.6.4 Add parameter validation and error handling
- [~] 1.6.5 Write API integration tests
- [~] 1.6.6 Update API documentation

**Acceptance Criteria**:
- ArrowEngine.diffuse(modality="text") works correctly
- Mode routing switches between AR and diffusion
- API tests cover all parameter combinations
- Documentation includes usage examples

---

### Task 1.7: 🧠 Text Generation Quality Validation

**Description**: Validate text generation quality against AR baseline in Python.

**Subtasks**:
- [~] 1.7.1 Implement perplexity evaluation script
- [~] 1.7.2 Generate test samples from diffusion model
- [~] 1.7.3 Generate baseline samples from AR model
- [~] 1.7.4 Compute perplexity difference
- [~] 1.7.5 Test infilling quality with various contexts
- [~] 1.7.6 Document quality metrics in validation report

**Acceptance Criteria**:
- Perplexity within 20% of AR baseline (REQ-1 AC5)
- Infilling produces coherent middle content
- Validation report documents all metrics
- Quality meets acceptance criteria

---

### Task 1.8: 🧠 Text Generation Latency Benchmark

**Description**: Benchmark text generation latency with consistency distillation in Python.

**Subtasks**:
- [~] 1.8.1 Implement latency measurement script
- [~] 1.8.2 Measure 4-step generation latency on CPU
- [~] 1.8.3 Test with INT2 quantized 350M model (using Rust ArrowQuant)
- [~] 1.8.4 Profile bottlenecks if latency exceeds target
- [~] 1.8.5 Optimize critical paths
- [~] 1.8.6 Document latency results

**Acceptance Criteria**:
- 4-step generation completes in <500ms on CPU (REQ-1 AC7)
- Latency measurements are reproducible
- Bottlenecks are identified and documented
- Optimization opportunities are noted
- Rust components provide expected speedup

---

## Phase 2: Unified Score Network (Phase 3b)

### Task 2.1: 🧠 Implement SharedTransformer

**Description**: Implement shared Transformer backbone for all modalities in Python (PyTorch).

**Subtasks**:
- [~] 2.1.1 Adapt InferenceCore as SharedTransformer base
- [~] 2.1.2 Implement modality embedding layer
- [~] 2.1.3 Implement sinusoidal time embedding
- [~] 2.1.4 Implement cross-attention for conditioning
- [~] 2.1.5 Write unit tests for transformer layers
- [~] 2.1.6 Validate parameter distribution (>90% in backbone)

**Acceptance Criteria**:
- SharedTransformer accounts for >90% of parameters (REQ-4 AC2)
- Modality and time embeddings work correctly
- Cross-attention accepts condition vectors
- Unit tests achieve >85% coverage

---

### Task 2.2: 🧠 Implement UnifiedScoreNetwork

**Description**: Implement unified score network combining SharedTransformer with projection heads in Python.

**Subtasks**:
- [~] 2.2.1 Implement UnifiedScoreNetwork class
- [~] 2.2.2 Integrate SharedTransformer as backbone
- [~] 2.2.3 Integrate TextProjectionHead
- [~] 2.2.4 Implement forward() method with modality routing
- [~] 2.2.5 Implement condition injection via cross-attention
- [~] 2.2.6 Write integration tests for score network

**Acceptance Criteria**:
- UnifiedScoreNetwork processes all modalities
- Single forward pass produces hidden states
- Condition injection works via cross-attention
- Integration tests verify correct behavior

---

### Task 2.3: 🧠 Implement MemoryConditioner

**Description**: Implement memory conditioner converting ArrowStorage results to diffusion conditions in Python (uses Rust ArrowStorage backend).

**Subtasks**:
- [~] 2.3.1 Implement MemoryConditioner class
- [~] 2.3.2 Integrate Rust ArrowStorage via PyO3 bindings for vector retrieval
- [~] 2.3.3 Implement embedding extraction from top-K results
- [~] 2.3.4 Implement projection to condition dimension
- [~] 2.3.5 Write unit tests for conditioning pipeline
- [~] 2.3.6 Measure conditioning latency

**Acceptance Criteria**:
- MemoryConditioner retrieves top-K memories
- Projection to condition space works correctly
- Conditioning latency <10ms (REQ-5 AC6) with Rust backend
- Unit tests verify correct tensor shapes

---

### Task 2.4: 🧠 Implement UncertaintyEstimator

**Description**: Implement uncertainty estimator for self-evolution triggering in Python.

**Subtasks**:
- [~] 2.4.1 Implement UncertaintyEstimator class
- [~] 2.4.2 Implement residual norm computation
- [~] 2.4.3 Implement normalization by expected noise level
- [~] 2.4.4 Implement should_evolve() threshold check
- [~] 2.4.5 Write unit tests for uncertainty computation
- [~] 2.4.6 Validate against human-annotated uncertain scenarios

**Acceptance Criteria**:
- Uncertainty metric correlates >0.7 with human annotations (REQ-6 AC5)
- False trigger rate <5% (REQ-6 AC6)
- should_evolve() correctly identifies high uncertainty
- Unit tests cover edge cases

---

### Task 2.5: 🧠 Integrate Memory-Guided Generation

**Description**: Integrate MemoryConditioner with DiffusionCore for memory-guided generation in Python.

**Subtasks**:
- [~] 2.5.1 Update DiffusionCore to accept condition vectors
- [~] 2.5.2 Update ArrowEngine.diffuse() to support memory_guided parameter
- [~] 2.5.3 Implement automatic Rust ArrowStorage querying via PyO3
- [~] 2.5.4 Write integration tests for memory-guided generation
- [~] 2.5.5 Validate semantic relevance of generated outputs
- [~] 2.5.6 Document memory-guided generation workflow

**Acceptance Criteria**:
- memory_guided=True triggers Rust ArrowStorage query
- Generated outputs are semantically relevant >80% (REQ-5 AC5)
- Integration tests verify end-to-end workflow
- Documentation includes usage examples

---

### Task 2.6: 🧠 Implement EvolutionRouter (L0-L2)

**Description**: Implement evolution router supporting L0 score composition, L1 ControlNet, and L2 LoRA in Python.

**Subtasks**:
- [~] 2.6.1 Implement EvolutionRouter class
- [~] 2.6.2 Implement L0 score composition via weighted sum
- [~] 2.6.3 Implement L1 ControlNet bank and injection
- [~] 2.6.4 Integrate L2 LoRA router from migrated module
- [~] 2.6.5 Implement get_fused_score() method
- [~] 2.6.6 Write unit tests for each evolution level

**Acceptance Criteria**:
- L0 score composition works with multiple scores
- L1 ControlNet injection adds <10% parameters (REQ-8 AC3)
- L2 LoRA fine-tuning adds ~1% parameters (REQ-9 AC3)
- Unit tests verify correct score fusion

---

### Task 2.7: 🧠 Cross-Modal End-to-End Testing

**Description**: Validate unified score network with cross-modal generation scenarios in Python.

**Subtasks**:
- [~] 2.7.1 Implement cross-modal test suite
- [~] 2.7.2 Test text generation with memory conditioning
- [~] 2.7.3 Test uncertainty estimation during generation
- [~] 2.7.4 Test evolution triggering on high uncertainty
- [~] 2.7.5 Validate parameter sharing across modalities
- [~] 2.7.6 Document cross-modal test results

**Acceptance Criteria**:
- All cross-modal tests pass
- Parameter sharing verified (>90% in backbone)
- Uncertainty triggering works correctly
- Test results documented in validation report

---

## Phase 3: Multimodal Generation (Phase 3c)

### Task 3.1: 🧠 Implement ContinuousSampler

**Description**: Implement continuous sampler for image/audio generation using Gaussian noise denoising in Python.

**Subtasks**:
- [~] 3.1.1 Implement ContinuousSampler class
- [~] 3.1.2 Implement step() method for Langevin dynamics
- [~] 3.1.3 Implement DDPM/DDIM sampling strategies
- [~] 3.1.4 Write unit tests for sampling operations
- [~] 3.1.5 Write property-based tests for sampling invariants
- [~] 3.1.6 Benchmark sampling performance

**Acceptance Criteria**:
- ContinuousSampler performs Gaussian denoising
- DDPM and DDIM strategies both work
- Property tests verify sampling correctness
- Unit tests achieve >85% coverage

---

### Task 3.2: 🧠 Implement ImageProjectionHead + VAE

**Description**: Implement image projection head and integrate VAE for latent space generation in Python (PyTorch).

**Subtasks**:
- [~] 3.2.1 Implement ImageProjectionHead class
- [~] 3.2.2 Implement encode() for patch embeddings
- [~] 3.2.3 Implement decode() for latent predictions
- [~] 3.2.4 Integrate VAE encoder/decoder
- [~] 3.2.5 Write unit tests for image head
- [~] 3.2.6 Validate parameter count <10M

**Acceptance Criteria**:
- ImageProjectionHead has <10M parameters
- VAE encoder/decoder work correctly
- Latent space has correct dimensionality
- Unit tests verify tensor shapes

---

### Task 3.3: 🧠 Implement AudioProjectionHead

**Description**: Implement audio projection head for waveform/latent audio generation in Python (PyTorch).

**Subtasks**:
- [~] 3.3.1 Implement AudioProjectionHead class
- [~] 3.3.2 Implement encode() for audio features
- [~] 3.3.3 Implement decode() for waveform generation
- [~] 3.3.4 Support 16kHz+ sampling rate
- [~] 3.3.5 Write unit tests for audio head
- [~] 3.3.6 Validate parameter count <10M

**Acceptance Criteria**:
- AudioProjectionHead has <10M parameters
- Generated audio has ≥16kHz sampling rate (REQ-3 AC2)
- Waveform generation works correctly
- Unit tests verify audio quality

---

### Task 3.4: 🧠 Extend DiffusionCore for Image/Audio

**Description**: Extend DiffusionCore to support image and audio generation in Python.

**Subtasks**:
- [~] 3.4.1 Update generate() to support image modality
- [~] 3.4.2 Update generate() to support audio modality
- [~] 3.4.3 Implement latent space noise initialization
- [~] 3.4.4 Integrate ContinuousSampler for continuous modalities
- [~] 3.4.5 Write integration tests for image/audio generation
- [~] 3.4.6 Validate generation quality

**Acceptance Criteria**:
- DiffusionCore generates 512x512 images in <30s (REQ-2 AC5)
- DiffusionCore generates audio in <2s (REQ-3 AC3)
- Integration tests pass for all modalities
- Generated outputs meet quality standards

---

### Task 3.5: 🧠 Implement Consistency Distillation Training

**Description**: Implement consistency distillation to compress 50-step diffusion to 4 steps in Python.

**Subtasks**:
- [~] 3.5.1 Implement consistency distillation training script
- [~] 3.5.2 Implement consistency loss function
- [~] 3.5.3 Train 4-step distilled model from 50-step teacher
- [~] 3.5.4 Validate quality retention (≥90% of teacher)
- [~] 3.5.5 Measure latency improvement
- [~] 3.5.6 Document distillation process and results

**Acceptance Criteria**:
- 4-step model maintains ≥90% quality of 50-step (REQ-7 AC2)
- Latency meets targets for all modalities
- Training script is reproducible
- Documentation includes hyperparameters

---

### Task 3.6: 🧠 Implement Multimodal Parallel Generation

**Description**: Implement simultaneous generation of multiple modalities from single forward pass in Python.

**Subtasks**:
- [~] 3.6.1 Update UnifiedScoreNetwork for parallel head execution
- [~] 3.6.2 Implement temporal synchronization for outputs
- [~] 3.6.3 Update ArrowEngine.diffuse() to accept multiple modalities
- [~] 3.6.4 Write integration tests for parallel generation
- [~] 3.6.5 Validate output synchronization
- [~] 3.6.6 Benchmark parallel vs sequential generation

**Acceptance Criteria**:
- Single forward pass produces all modality outputs (REQ-16 AC1)
- Outputs are temporally synchronized (REQ-16 AC3)
- Parallel generation is faster than sequential
- Integration tests verify correct behavior

---

### Task 3.7: 🧠 Edge Deployment Validation (ARM + INT2)

**Description**: Validate deployment on edge devices with ARM CPU and INT2 quantization (uses Rust ArrowQuant).

**Subtasks**:
- [~] 3.7.1 Set up ARM test environment (Raspberry Pi / mobile)
- [~] 3.7.2 Deploy INT2 quantized model (<35MB) using Rust ArrowQuant
- [~] 3.7.3 Test text + audio generation on edge device
- [~] 3.7.4 Measure latency and memory usage
- [~] 3.7.5 Validate quality on edge deployment
- [~] 3.7.6 Document edge deployment guide

**Acceptance Criteria**:
- Model size <35MB with INT2 (REQ-14 AC1)
- Text + audio modalities work on edge (REQ-14 AC4)
- Latency meets targets on edge hardware
- Deployment guide is complete
- Rust components provide expected performance

---

## Phase 4: Advanced Features

### Task 4.1: 🧠 Implement ControlNet Training

**Description**: Implement training pipeline for ControlNet structural constraints in Python.

**Subtasks**:
- [~] 4.1.1 Implement ControlNet architecture
- [~] 4.1.2 Implement zero-initialized residual connections
- [~] 4.1.3 Implement ControlNet training script
- [~] 4.1.4 Train CoT-ControlNet for reasoning structure
- [~] 4.1.5 Train ToolSchema-ControlNet for JSON compliance
- [~] 4.1.6 Validate ControlNet effectiveness

**Acceptance Criteria**:
- ControlNet has <10% of base model parameters (REQ-8 AC3)
- CoT-ControlNet enforces reasoning structure (REQ-8 AC4)
- ToolSchema-ControlNet enforces JSON schema (REQ-8 AC5)
- Training script is reproducible

---

### Task 4.2: 🧠 Implement Energy-Based Model Constraints

**Description**: Implement EBM constraint fusion for hard constraint enforcement in Python.

**Subtasks**:
- [~] 4.2.1 Implement EBM interface
- [~] 4.2.2 Implement energy gradient computation
- [~] 4.2.3 Integrate EBM with EvolutionRouter
- [~] 4.2.4 Implement constraint validation
- [~] 4.2.5 Write unit tests for EBM fusion
- [~] 4.2.6 Validate constraint enforcement

**Acceptance Criteria**:
- EBM gradients are computed correctly
- Score fusion follows formula: final = diffusion - η*energy (REQ-10 AC2)
- Multiple EBMs can be active simultaneously (REQ-10 AC3)
- Constraint validation works correctly

---

### Task 4.3: 🧠 Implement Self-Evolution Closed Loop

**Description**: Implement complete feedback loop from memory to generation to evolution in Python (uses Rust ArrowStorage).

**Subtasks**:
- [~] 4.3.1 Integrate UncertaintyEstimator with EvolutionRouter
- [~] 4.3.2 Implement evolution triggering logic
- [~] 4.3.3 Implement adaptation validation
- [~] 4.3.4 Implement adaptation persistence to Rust ArrowStorage via PyO3
- [~] 4.3.5 Implement adaptation retrieval and application from Rust ArrowStorage
- [~] 4.3.6 Write integration tests for closed loop

**Acceptance Criteria**:
- High uncertainty triggers appropriate evolution level (REQ-18 AC1)
- Adaptations are validated before persisting (REQ-18 AC2)
- Validated adaptations are stored to Rust ArrowStorage (REQ-18 AC3)
- Similar contexts retrieve learned adaptations (REQ-18 AC4)
- Closed loop is complete and functional (REQ-18 AC5)

---

### Task 4.4: 🧠 Implement L3 Selective Fine-Tuning

**Description**: Implement L3 selective backbone fine-tuning with uncertainty-driven layer unfreezing in Python.

**Subtasks**:
- [~] 4.4.1 Implement SelectiveFinetuner class
- [~] 4.4.2 Implement layer-wise uncertainty analysis
- [~] 4.4.3 Implement progressive layer unfreezing
- [~] 4.4.4 Implement fine-tuning training loop
- [~] 4.4.5 Write unit tests for selective fine-tuning
- [~] 4.4.6 Validate improvement after fine-tuning

**Acceptance Criteria**:
- SelectiveFinetuner unfreezes layers based on uncertainty
- Fine-tuning improves model performance
- Training is more efficient than full fine-tuning
- Unit tests verify correct layer selection

---

## Phase 5: Documentation and Deployment

### Task 5.1: 🧠 Write Comprehensive Documentation

**Description**: Create complete documentation for the unified diffusion system in Python/Markdown.

**Subtasks**:
- [~] 5.1.1 Write API reference documentation
- [~] 5.1.2 Write user guide with examples
- [~] 5.1.3 Write developer guide for extending system (explain Rust Skeleton + Python Brain)
- [~] 5.1.4 Write deployment guide for different tiers
- [~] 5.1.5 Write troubleshooting guide
- [~] 5.1.6 Create tutorial notebooks

**Acceptance Criteria**:
- API reference covers all public interfaces
- User guide includes working examples
- Developer guide explains dual-layer architecture philosophy
- Deployment guide covers edge/local/cloud
- Troubleshooting guide addresses common issues

---

### Task 5.2: 🦴🧠 Performance Optimization

**Description**: Optimize system performance for production deployment (both Rust and Python layers).

**Subtasks**:
- [~] 5.2.1 Profile critical paths and identify bottlenecks in both Rust and Python
- [~] 5.2.2 Optimize memory usage and allocation in Rust components
- [~] 5.2.3 Optimize Python computation with kernel fusion where possible
- [~] 5.2.4 Implement caching for repeated operations
- [~] 5.2.5 Benchmark optimized system (measure Rust vs Python performance)
- [~] 5.2.6 Document optimization techniques

**Acceptance Criteria**:
- All latency targets are met or exceeded
- Memory usage is within acceptable limits
- Benchmark results show improvement
- Optimization techniques are documented
- Rust components provide 10-50x speedup for infrastructure

---

### Task 5.3: 🧠 Final Integration Testing

**Description**: Perform comprehensive integration testing of complete system in Python.

**Subtasks**:
- [~] 5.3.1 Write end-to-end test suite
- [~] 5.3.2 Test all modality combinations
- [~] 5.3.3 Test all evolution levels
- [~] 5.3.4 Test edge/local/cloud deployments
- [~] 5.3.5 Perform stress testing
- [~] 5.3.6 Document test results

**Acceptance Criteria**:
- All integration tests pass
- System handles edge cases gracefully
- Stress tests show acceptable performance
- Test coverage >80% for critical paths

---

### Task 5.4: 🦴🧠 Production Deployment

**Description**: Deploy unified diffusion system to production environments (both Rust and Python components).

**Subtasks**:
- [~] 5.4.1 Prepare production deployment scripts
- [~] 5.4.2 Build and package Rust components as Python wheels with maturin
- [~] 5.4.3 Deploy to edge devices (Rust ArrowStorage + Python brain)
- [~] 5.4.4 Deploy to local workstations
- [~] 5.4.5 Deploy to cloud servers
- [~] 5.4.6 Set up monitoring and logging
- [~] 5.4.7 Validate production deployments

**Acceptance Criteria**:
- Deployment scripts are automated
- Rust wheels build correctly with maturin
- All tiers are successfully deployed
- Monitoring shows healthy system status
- Production validation tests pass
- Rust components integrate seamlessly via PyO3

---

## Summary

**Total Tasks**: 56 tasks across 5 phases (including Rust workspace setup)
**Estimated Timeline**: 10-12 weeks
**Critical Path**: Phase 0 (Rust + Python setup) → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

**Architecture Philosophy**: 
- 🦴 **Rust Skeleton**: Stable, high-performance infrastructure (ArrowStorage, ArrowQuant, VectorSearch, FastTokenizer)
- 🧠 **Python Brain**: Flexible, evolution-friendly learning layer (DiffusionCore, EvolutionRouter, training)

**Key Milestones**:
- Week 2: Rust infrastructure + Python module migration complete (Phase 0)
- Week 4: Text generation working with Rust backend (Phase 1)
- Week 6: Unified score network with memory guidance (Phase 2)
- Week 9: Multimodal generation working (Phase 3)
- Week 11: Advanced features complete (Phase 4)
- Week 12: Production deployment (Phase 5)

**Performance Expectations**:
- Rust ArrowStorage: 10-50x speedup for vector search
- Rust ArrowQuant: 5-10x speedup for quantization
- Rust FastTokenizer: 10-100x speedup for tokenization
- Overall system: 50-70% latency reduction with Rust infrastructure

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
- [x] 0.1.1 Initialize Rust workspace with `cargo new --lib ai-os-rust`
- [x] 0.1.2 Create Cargo workspace members: arrow_storage, arrow_quant, vector_search, fast_tokenizer
- [x] 0.1.3 Configure PyO3 dependencies in Cargo.toml for Python bindings
- [x] 0.1.4 Configure maturin for building Python wheels
- [x] 0.1.5 Set up Rust project structure (src/, tests/, benches/)
- [x] 0.1.6 Create .cargo/config.toml with optimization flags

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
- [x] 0.2.3 Create support directories (config/, utils/, tests/, scripts/, models/)
- [x] 0.2.4 Initialize Python package structure with __init__.py files
- [x] 0.2.5 Create requirements.txt with essential dependencies (torch, numpy, transformers, pyyaml)
- [x] 0.2.6 Create README.md with dual-layer architecture overview

**Acceptance Criteria**:
- Directory structure matches design specification
- All __init__.py files are in place
- requirements.txt includes only 5 core packages: torch, numpy, pyarrow, transformers, pyyaml
- README explains Rust Skeleton + Python Brain philosophy

---

### Task 0.3: 🦴 Implement ArrowStorage in Rust

**Description**: Build high-performance ArrowStorage with vector retrieval in Rust using arrow-rs.

**Subtasks**:
- [x] 0.3.1 Implement ArrowStorage struct with arrow-rs RecordBatch
- [x] 0.3.2 Implement vector search using simsimd for SIMD-accelerated similarity
- [x] 0.3.3 Implement schema definitions (SCHEMA_REGISTRY)
- [x] 0.3.4 Create PyO3 Python bindings for ArrowStorage
- [x] 0.3.5 Write Rust unit tests for storage operations
- [x] 0.3.6 Write Python integration tests calling Rust bindings

**Acceptance Criteria**:
- ArrowStorage can perform vector search with 10-50x speedup vs Python
- Schema registry is accessible from both Rust and Python
- PyO3 bindings work seamlessly from Python
- Integration tests pass for store/retrieve operations

---

### Task 0.4: 🦴 Implement ArrowQuant in Rust

**Description**: Build quantization engine in Rust using ndarray and rayon for parallel processing.

**Subtasks**:
- [x] 0.4.1 Implement INT2/INT4 quantization algorithms in Rust
- [x] 0.4.2 Implement dequantization with SIMD optimizations
- [x] 0.4.3 Implement zero-copy weight loading from Parquet
- [x] 0.4.4 Create PyO3 Python bindings for ArrowQuant
- [x] 0.4.5 Write Rust benchmarks for quantization performance
- [x] 0.4.6 Write Python integration tests

**Acceptance Criteria**:
- Quantization achieves 5-10x speedup vs Python implementation
- Zero-copy loading works correctly
- PyO3 bindings expose quantization functions to Python
- Benchmarks show expected performance gains

---

### Task 0.5: 🦴 Implement FastTokenizer in Rust

**Description**: Build fast tokenizer using Hugging Face tokenizers crate.

**Subtasks**:
- [x] 0.5.1 Wrap tokenizers crate with custom interface
- [x] 0.5.2 Implement batch tokenization with parallel processing
- [x] 0.5.3 Support BPE, WordPiece, and Unigram tokenizers
- [x] 0.5.4 Create PyO3 Python bindings for FastTokenizer
- [x] 0.5.5 Write Rust unit tests for tokenization
- [x] 0.5.6 Benchmark tokenization speed vs Python

**Acceptance Criteria**:
- Tokenization achieves 10-100x speedup vs Python
- Supports all major tokenizer types
- PyO3 bindings work seamlessly from Python
- Unit tests verify correctness

---

### Task 0.6: 🧠 Migrate Python Inference Module

**Description**: Migrate essential inference components from llm_compression/inference/ to new project.

**Subtasks**:
- [x] 0.6.1 Copy and adapt ArrowEngine to support diffusion mode routing
- [x] 0.6.2 Copy WeightLoader V2 with Parquet V2 support (will use Rust ArrowQuant)
- [x] 0.6.3 Integrate Rust FastTokenizer bindings for text/code tokenization
- [x] 0.6.4 Copy InferenceCore as base for SharedTransformer
- [x] 0.6.5 Update imports and dependencies
- [x] 0.6.6 Write unit tests for migrated components

**Acceptance Criteria**:
- All inference components compile without errors
- Unit tests pass for WeightLoader and FastTokenizer integration
- ArrowEngine can route between "ar" and "diffusion" modes
- Rust components are properly integrated via PyO3

---

### Task 0.7: 🧠 Migrate Python Evolution Module

**Description**: Migrate LoRA and skill management components for L2 evolution level.

**Subtasks**:
- [x] 0.7.1 Copy LoRATrainer for L2 fine-tuning
- [x] 0.7.2 Copy SkillFactory for skill card management
- [x] 0.7.3 Copy WeightMapProbe for weight inspection
- [x] 0.7.4 Update imports and dependencies
- [x] 0.7.5 Write unit tests for LoRA operations

**Acceptance Criteria**:
- LoRATrainer can create and apply LoRA adaptations
- SkillFactory can load/save skill cards
- Unit tests pass for evolution components

---

### Task 0.8: 🧠 Migrate Configuration and Utilities

**Description**: Migrate configuration management and utility modules.

**Subtasks**:
- [x] 0.8.1 Copy config.py with configuration management
- [x] 0.8.2 Copy logger.py for logging setup
- [x] 0.8.3 Copy errors.py with exception hierarchy
- [x] 0.8.4 Copy embedding_provider.py for embedding generation
- [x] 0.8.5 Create diffusion_config.py for diffusion-specific settings
- [x] 0.8.6 Write configuration validation tests

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
- [x] 1.1.1 Implement base NoiseScheduler class
- [x] 1.1.2 Implement cosine schedule for continuous modalities
- [x] 1.1.3 Implement mask schedule for discrete modalities
- [x] 1.1.4 Implement timestep sampling (uniform and non-uniform)
- [x] 1.1.5 Implement add_noise() for forward process
- [x] 1.1.6 Write unit tests for all schedule types

**Acceptance Criteria**:
- NoiseScheduler supports both discrete and continuous modes
- Cosine schedule produces correct σ(t) values
- Mask schedule produces correct β(t) values
- Unit tests achieve >90% coverage

---

### Task 1.2: 🧠 Implement DiscreteSampler

**Description**: Implement discrete sampler for text/code generation using mask-based denoising in Python.

**Subtasks**:
- [x] 1.2.1 Implement DiscreteSampler class
- [x] 1.2.2 Implement step() method for unmask operations
- [x] 1.2.3 Implement confidence-based token selection
- [x] 1.2.4 Implement infilling support (preserve prefix/suffix)
- [x] 1.2.5 Write unit tests for sampling operations
- [x] 1.2.6 Write property-based tests for sampling invariants

**Acceptance Criteria**:
- DiscreteSampler can perform iterative unmasking
- Infilling preserves provided context tokens
- Property tests verify sampling correctness
- Unit tests achieve >85% coverage

---

### Task 1.3: 🧠 Implement TextProjectionHead

**Description**: Implement text projection head for discrete text generation in Python.

**Subtasks**:
- [x] 1.3.1 Implement TextProjectionHead class
- [x] 1.3.2 Implement encode() method (tokens → hidden space)
- [x] 1.3.3 Implement decode() method (hidden space → logits)
- [x] 1.3.4 Implement embedding layer and output projection
- [x] 1.3.5 Write unit tests for forward/backward pass
- [x] 1.3.6 Validate parameter count <10M

**Acceptance Criteria**:
- TextProjectionHead has <10M parameters
- encode() and decode() are inverse operations
- Unit tests verify correct tensor shapes
- Gradient flow works correctly

---

### Task 1.4: 🧠 Implement DiffusionCore (Text-Only)

**Description**: Implement core diffusion inference engine for text generation in Python.

**Subtasks**:
- [x] 1.4.1 Implement DiffusionCore class
- [x] 1.4.2 Implement generate() method for text modality
- [x] 1.4.3 Implement noise initialization for masked sequences
- [x] 1.4.4 Implement denoising loop with DiscreteSampler
- [x] 1.4.5 Integrate with NoiseScheduler
- [x] 1.4.6 Write integration tests for end-to-end generation

**Acceptance Criteria**:
- DiffusionCore can generate coherent text
- Denoising loop completes in expected steps
- Integration tests pass for various prompts
- Memory usage is within acceptable limits

---

### Task 1.5: 🧠 Convert MDLM Weights to Parquet V2

**Description**: Convert open-source MDLM model weights to Parquet V2 format with ArrowQuant (uses Rust ArrowQuant backend).

**Subtasks**:
- [x] 1.5.1 Download MDLM checkpoint from HuggingFace
- [x] 1.5.2 Implement weight conversion script in Python
- [x] 1.5.3 Apply Rust ArrowQuant INT2 quantization via PyO3 bindings
- [x] 1.5.4 Save to Parquet V2 format
- [x] 1.5.5 Create metadata.json with model config
- [x] 1.5.6 Validate converted weights load correctly

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
- [x] 1.6.1 Add diffuse() method to ArrowEngine
- [x] 1.6.2 Implement mode routing (ar vs diffusion)
- [x] 1.6.3 Integrate DiffusionCore for text generation
- [x] 1.6.4 Add parameter validation and error handling
- [x] 1.6.5 Write API integration tests
- [x] 1.6.6 Update API documentation

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


---

## Phase 1.5: ArrowQuant V2 - Diffusion Model Quantization Enhancement

### Task 1.9: 🦴 Implement Per-Channel Quantization in Rust

**Description**: Enhance ArrowQuant with per-channel quantization to handle spatial variance in diffusion models.

**Priority**: HIGH (Short-term, 2 weeks)

**Subtasks**:
- [ ] 1.9.1 Implement per-channel scale/zero-point computation in Rust
- [ ] 1.9.2 Implement per-channel quantization with axis parameter
- [ ] 1.9.3 Implement per-channel dequantization
- [ ] 1.9.4 Update Parquet schema to store per-channel params
- [ ] 1.9.5 Write Rust unit tests for per-channel quantization
- [ ] 1.9.6 Write Python integration tests
- [ ] 1.9.7 Benchmark performance vs per-tensor quantization

**Acceptance Criteria**:
- Per-channel quantization achieves <2% quality degradation vs FP32
- Supports both per-output-channel and per-input-channel modes
- Maintains 5-10x speedup vs Python implementation
- PyO3 bindings work seamlessly
- Unit tests achieve >90% coverage

**References**:
- Q-DiT paper: https://arxiv.org/abs/2406.17343
- Per-channel quantization is standard in SOTA diffusion quantization

---

### Task 1.10: 🧠 Implement Dynamic Activation Quantization

**Description**: Implement time-aware dynamic activation quantization for diffusion models in Python.

**Priority**: HIGH (Short-term, 2 weeks)

**Subtasks**:
- [ ] 1.10.1 Implement DynamicActivationQuant class
- [ ] 1.10.2 Implement timestep-based calibration
- [ ] 1.10.3 Implement time-grouping strategy (group similar timesteps)
- [ ] 1.10.4 Integrate with Rust ArrowQuant backend
- [ ] 1.10.5 Write unit tests for dynamic quantization
- [ ] 1.10.6 Validate on Dream 7B model

**Acceptance Criteria**:
- Dynamic quantization reduces activation quantization error by >30%
- Supports configurable number of timestep groups (default: 10)
- Calibration completes in <5 minutes on sample dataset
- Integration with Rust backend maintains performance
- Unit tests achieve >85% coverage

**References**:
- Q-DiT dynamic activation quantization
- Time-aware quantization (arXiv 2025)

---

### Task 1.11: 🧠 Integrate Q-DiT Evolutionary Search

**Description**: Integrate Q-DiT's evolutionary search algorithm for automatic quantization granularity allocation.

**Priority**: HIGH (Short-term, 3 weeks)

**Subtasks**:
- [ ] 1.11.1 Clone and study Q-DiT repository (https://github.com/Juanerx/Q-DiT)
- [ ] 1.11.2 Implement EvolutionarySearchQuantizer class
- [ ] 1.11.3 Implement population initialization and fitness evaluation
- [ ] 1.11.4 Implement mutation and selection operators
- [ ] 1.11.5 Integrate with ArrowQuant backend
- [ ] 1.11.6 Implement FID-based fitness function
- [ ] 1.11.7 Write integration tests
- [ ] 1.11.8 Document Q-DiT integration guide

**Acceptance Criteria**:
- Evolutionary search finds optimal group sizes in <2 hours
- Achieves W4A8 quantization with <1% FID degradation
- Supports both image and text diffusion models
- Integration guide includes usage examples
- Validation on Dream 7B and MDLM models

**References**:
- Q-DiT GitHub: https://github.com/Juanerx/Q-DiT
- Q-DiT paper (CVPR 2025): https://arxiv.org/abs/2406.17343

---

### Task 1.12: 🧠 Implement Hybrid Quantization Mode

**Description**: Implement hybrid quantization mode supporting both offline and online quantization.

**Priority**: MEDIUM (Short-term, 1 week)

**Subtasks**:
- [ ] 1.12.1 Design hybrid quantization architecture
- [ ] 1.12.2 Implement on-demand quantizer loading
- [ ] 1.12.3 Implement quantization mode detection (offline/online/hybrid)
- [ ] 1.12.4 Update ArrowEngine to support hybrid mode
- [ ] 1.12.5 Write unit tests for mode switching
- [ ] 1.12.6 Document usage patterns for each mode

**Acceptance Criteria**:
- Hybrid mode loads quantizer only when needed
- Offline mode has <200ms startup time
- Online mode supports dynamic LoRA/ControlNet quantization
- Mode switching is transparent to users
- Documentation includes deployment recommendations

---

### Task 1.13: 🧠 Validate Quantization on Dream 7B

**Description**: Validate enhanced quantization on Dream 7B discrete diffusion model.

**Priority**: HIGH (Short-term, 1 week)

**Subtasks**:
- [ ] 1.13.1 Download Dream 7B model from HuggingFace
- [ ] 1.13.2 Apply per-channel quantization to Dream 7B
- [ ] 1.13.3 Apply dynamic activation quantization
- [ ] 1.13.4 Run Q-DiT evolutionary search
- [ ] 1.13.5 Measure perplexity degradation
- [ ] 1.13.6 Measure generation quality (human evaluation)
- [ ] 1.13.7 Measure inference latency
- [ ] 1.13.8 Document validation results

**Acceptance Criteria**:
- W4A8 quantization: <3% perplexity degradation
- W4A4 quantization: <5% perplexity degradation
- Inference latency <500ms for 4-step generation (CPU)
- Model size: 28GB → <2GB (INT4) or <1GB (INT2)
- Validation report documents all metrics

**References**:
- Dream 7B: https://huggingface.co/dream-org/Dream-v0-7B

---

## Phase 2.5: Multimodal Projection Heads

### Task 2.8: 🧠 Implement ImageProjectionHead

**Description**: Implement image projection head for continuous image generation in Python (PyTorch).

**Priority**: MEDIUM (Mid-term, 2 weeks)

**Subtasks**:
- [ ] 2.8.1 Implement ImageProjectionHead class
- [ ] 2.8.2 Implement encode() for patch embeddings
- [ ] 2.8.3 Implement decode() for latent predictions
- [ ] 2.8.4 Integrate VAE encoder/decoder
- [ ] 2.8.5 Write unit tests for image head
- [ ] 2.8.6 Validate parameter count <10M
- [ ] 2.8.7 Test with SharedTransformer integration

**Acceptance Criteria**:
- ImageProjectionHead has <10M parameters
- VAE encoder/decoder work correctly
- Latent space has correct dimensionality (e.g., 64x64x4 for 512x512 images)
- Unit tests verify tensor shapes
- Integration with SharedTransformer works seamlessly

---

### Task 2.9: 🧠 Implement AudioProjectionHead

**Description**: Implement audio projection head for waveform/latent audio generation in Python (PyTorch).

**Priority**: MEDIUM (Mid-term, 2 weeks)

**Subtasks**:
- [ ] 2.9.1 Implement AudioProjectionHead class
- [ ] 2.9.2 Implement encode() for audio features
- [ ] 2.9.3 Implement decode() for waveform generation
- [ ] 2.9.4 Support 16kHz+ sampling rate
- [ ] 2.9.5 Write unit tests for audio head
- [ ] 2.9.6 Validate parameter count <10M
- [ ] 2.9.7 Test with SharedTransformer integration

**Acceptance Criteria**:
- AudioProjectionHead has <10M parameters
- Generated audio has ≥16kHz sampling rate (REQ-3 AC2)
- Waveform generation works correctly
- Unit tests verify audio quality
- Integration with SharedTransformer works seamlessly

---

### Task 2.10: 🧠 Train Image Modality via L1 ControlNet

**Description**: Train image generation capability using L1 ControlNet strategy.

**Priority**: MEDIUM (Mid-term, 1-2 weeks)

**Subtasks**:
- [ ] 2.10.1 Prepare image-text paired dataset (e.g., LAION subset)
- [ ] 2.10.2 Implement ControlNet architecture for image modality
- [ ] 2.10.3 Freeze SharedTransformer, train ControlNet only
- [ ] 2.10.4 Implement training loop with FID evaluation
- [ ] 2.10.5 Validate parameter efficiency (~10% of base model)
- [ ] 2.10.6 Measure generation quality (FID score)
- [ ] 2.10.7 Document training process and hyperparameters

**Acceptance Criteria**:
- ControlNet has <10% of base model parameters (REQ-8 AC3)
- Training completes in 1-2 days on single GPU
- Generated images have acceptable FID score
- Total model size: 7B + 18M params (+0.26%)
- Training script is reproducible

---

### Task 2.11: 🧠 Train Audio Modality via L2 LoRA

**Description**: Train audio generation capability using L2 LoRA strategy.

**Priority**: MEDIUM (Mid-term, 1 week)

**Subtasks**:
- [ ] 2.11.1 Prepare audio-text paired dataset (e.g., MusicCaps)
- [ ] 2.11.2 Implement LoRA architecture for audio modality
- [ ] 2.11.3 Freeze SharedTransformer, train LoRA only
- [ ] 2.11.4 Implement training loop with audio quality metrics
- [ ] 2.11.5 Validate parameter efficiency (~1% of base model)
- [ ] 2.11.6 Measure generation quality (FAD score)
- [ ] 2.11.7 Document training process and hyperparameters

**Acceptance Criteria**:
- LoRA has ~1% of base model parameters (REQ-9 AC3)
- Training completes in hours on single GPU
- Generated audio has acceptable FAD score
- Total model size: 7B + 6.7M params (+0.01%)
- Training script is reproducible

---

### Task 2.12: 🧠 Implement Multimodal Parallel Generation

**Description**: Implement simultaneous generation of multiple modalities from single forward pass.

**Priority**: MEDIUM (Mid-term, 1 week)

**Subtasks**:
- [ ] 2.12.1 Update UnifiedScoreNetwork for parallel head execution
- [ ] 2.12.2 Implement temporal synchronization for outputs
- [ ] 2.12.3 Update ArrowEngine.diffuse() to accept multiple modalities
- [ ] 2.12.4 Write integration tests for parallel generation
- [ ] 2.12.5 Validate output synchronization
- [ ] 2.12.6 Benchmark parallel vs sequential generation

**Acceptance Criteria**:
- Single forward pass produces all modality outputs (REQ-16 AC1)
- Outputs are temporally synchronized (REQ-16 AC3)
- Parallel generation is faster than sequential
- Integration tests verify correct behavior
- Benchmark shows expected speedup

---

### Task 2.13: 🧠 Implement L0 Score Composition

**Description**: Implement L0 score composition for zero-training multimodal mixing.

**Priority**: LOW (Mid-term, 1 week)

**Subtasks**:
- [ ] 2.13.1 Implement ScoreComposer class
- [ ] 2.13.2 Implement weighted score mixing
- [ ] 2.13.3 Support external model integration
- [ ] 2.13.4 Implement real-time composition without training
- [ ] 2.13.5 Write unit tests for score composition
- [ ] 2.13.6 Document usage examples

**Acceptance Criteria**:
- Score composition works without any training
- Supports mixing 2+ models in real-time
- Weighted mixing produces coherent outputs
- Unit tests verify correct score fusion
- Documentation includes practical examples

---

## Phase 3.5: Quantization-Aware Multimodal Evolution

### Task 3.8: 🧠 Implement Quantization-Aware Evolution

**Description**: Integrate quantization with multimodal evolution strategies.

**Priority**: LOW (Long-term, 2 weeks)

**Subtasks**:
- [ ] 3.8.1 Design quantization strategy per evolution level
- [ ] 3.8.2 Implement online quantization for L1/L2 adapters
- [ ] 3.8.3 Implement offline re-quantization for L3/L4
- [ ] 3.8.4 Integrate Q-DiT for fine-tuned models
- [ ] 3.8.5 Write integration tests
- [ ] 3.8.6 Document quantization-evolution workflow

**Acceptance Criteria**:
- L1/L2 adapters are quantized online (<100ms overhead)
- L3/L4 models are re-quantized offline with Q-DiT
- Quantization maintains quality across evolution levels
- Integration tests verify correct behavior
- Documentation includes best practices

---

### Task 3.9: 🧠 Validate Multimodal Quantization

**Description**: Validate quantization effectiveness across all modalities.

**Priority**: LOW (Long-term, 1 week)

**Subtasks**:
- [ ] 3.9.1 Quantize text modality (Dream 7B)
- [ ] 3.9.2 Quantize image modality (ControlNet)
- [ ] 3.9.3 Quantize audio modality (LoRA)
- [ ] 3.9.4 Measure quality degradation per modality
- [ ] 3.9.5 Measure inference latency per modality
- [ ] 3.9.6 Measure memory footprint
- [ ] 3.9.7 Document validation results

**Acceptance Criteria**:
- All modalities maintain <5% quality degradation with INT4
- Inference latency meets targets (text <500ms, image <2s, audio <1s)
- Total memory footprint <2GB for all modalities
- Validation report documents all metrics
- Recommendations for deployment configurations

---

## Updated Timeline and Milestones

### Short-Term (Weeks 1-4): ArrowQuant V2 + Q-DiT Integration

**Focus**: Enhance quantization for diffusion models

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1-2 | Task 1.9, 1.10 | Per-channel + Dynamic activation quantization |
| 3-4 | Task 1.11, 1.12, 1.13 | Q-DiT integration + Hybrid mode + Dream 7B validation |

**Milestone**: ArrowQuant V2 with SOTA diffusion quantization ready

---

### Mid-Term (Weeks 5-8): Multimodal Projection Heads

**Focus**: Add image and audio generation capabilities

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 5-6 | Task 2.8, 2.10 | ImageProjectionHead + L1 ControlNet training |
| 7-8 | Task 2.9, 2.11, 2.12 | AudioProjectionHead + L2 LoRA training + Parallel generation |

**Milestone**: Multimodal generation (text + image + audio) working

---

### Long-Term (Weeks 9-12): Advanced Features + Production

**Focus**: Quantization-aware evolution + Production deployment

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 9-10 | Task 3.8, 3.9 | Quantization-aware evolution + Validation |
| 11-12 | Task 5.2, 5.3, 5.4 | Performance optimization + Integration testing + Deployment |

**Milestone**: Production-ready multimodal diffusion system

---

## Updated Performance Expectations

### Quantization Performance (After ArrowQuant V2)

| Metric | Current (Basic) | After Q-DiT Integration | Target |
|--------|----------------|------------------------|--------|
| **Compression** | 16x (FP32→INT2) | 16x (FP32→INT2) | 16x |
| **Speedup** | 5-10x | 10-20x | 10-20x |
| **Quality (W4A8)** | ~5-10% degradation | <1% degradation | <1% |
| **Quality (W4A4)** | ~10-15% degradation | <3% degradation | <3% |

### Multimodal Performance

| Modality | Model Size | Latency (CPU) | Quality Metric |
|----------|-----------|---------------|----------------|
| **Text** | 1.75GB (INT2) | <500ms | Perplexity <1.2x baseline |
| **Image** | +4.5MB (head) | <2s | FID <10 |
| **Audio** | +1.7MB (head) | <1s | FAD <5 |
| **All (Parallel)** | 1.76GB total | <2.5s | All metrics met |

### Deployment Targets (Updated)

| Device Tier | Model Size | Quantization | Modalities | Latency |
|-------------|-----------|--------------|------------|---------|
| **Edge** (2-4GB RAM) | <35MB | INT2 + Q-DiT | Text, Audio | <500ms |
| **Local** (8+GB RAM) | <200MB | INT4 + Q-DiT | Text, Audio, Image | <2s |
| **Cloud** (32+GB RAM) | <2GB | INT2/INT4 + Q-DiT | All | <1s |

---

## Summary of Changes

### New Tasks Added

**Phase 1.5 (Short-term)**:
- Task 1.9: Per-Channel Quantization (Rust)
- Task 1.10: Dynamic Activation Quantization (Python)
- Task 1.11: Q-DiT Evolutionary Search Integration
- Task 1.12: Hybrid Quantization Mode
- Task 1.13: Dream 7B Quantization Validation

**Phase 2.5 (Mid-term)**:
- Task 2.8: ImageProjectionHead Implementation
- Task 2.9: AudioProjectionHead Implementation
- Task 2.10: Image Modality Training (L1 ControlNet)
- Task 2.11: Audio Modality Training (L2 LoRA)
- Task 2.12: Multimodal Parallel Generation
- Task 2.13: L0 Score Composition

**Phase 3.5 (Long-term)**:
- Task 3.8: Quantization-Aware Evolution
- Task 3.9: Multimodal Quantization Validation

### Priority Adjustments

**Moved to Short-term** (as requested):
- Q-DiT integration (Task 1.11)
- Per-channel quantization (Task 1.9)
- Dynamic activation quantization (Task 1.10)
- Hybrid quantization mode (Task 1.12)
- Dream 7B validation (Task 1.13)

**Moved to Mid-term** (as requested):
- Multimodal Projection Heads (Tasks 2.8, 2.9)
- Modality training (Tasks 2.10, 2.11)
- Parallel generation (Task 2.12)

### Total Task Count

- **Original**: 56 tasks
- **Added**: 13 new tasks
- **Total**: 69 tasks across 5 phases + 3 new sub-phases

### Estimated Timeline

- **Original**: 10-12 weeks
- **Updated**: 12-14 weeks (accounting for new quantization and multimodal tasks)
- **Critical Path**: Phase 0 → Phase 1 → Phase 1.5 → Phase 2 → Phase 2.5 → Phase 3 → Phase 3.5 → Phase 4 → Phase 5


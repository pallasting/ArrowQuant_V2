# Requirements Document: Unified Diffusion Architecture

## Overview

The Unified Diffusion Architecture replaces autoregressive (AR) models with thermodynamic diffusion models to achieve unified multimodal generation (text, code, image, audio) through a single shared Score Network. The system integrates with existing ArrowEngine infrastructure, supports memory-guided generation using ArrowStorage, and implements a 5-level progressive self-evolution mechanism with edge deployment capabilities.

## Glossary

- **Score_Network**: Unified neural network that learns the gradient of log-probability distributions (score function) for denoising across all modalities
- **DiffusionCore**: Core inference engine managing iterative denoising process for diffusion-based generation
- **ArrowEngine**: Existing inference routing layer managing model loading and execution
- **ArrowStorage**: Storage system using Hopfield networks for vector-based memory retrieval
- **ArrowQuant**: Quantization system supporting INT2/INT4 weight compression with zero-copy loading
- **MemoryConditioner**: Component converting ArrowStorage retrieval results into diffusion condition vectors
- **NoiseScheduler**: Component managing noise levels across diffusion timesteps for discrete and continuous modalities
- **DiscreteSampler**: Sampler for discrete modalities (text, code) using mask-based denoising
- **ContinuousSampler**: Sampler for continuous modalities (image, audio) using Gaussian noise denoising
- **ControlNet**: Lightweight auxiliary network injecting structural or behavioral constraints without modifying base weights
- **LoRA**: Low-Rank Adaptation for efficient model fine-tuning with minimal parameter updates
- **EBM**: Energy-Based Model for constraint validation and guidance
- **Consistency_Distillation**: Training technique compressing multi-step diffusion to 1-4 steps while maintaining quality
- **MDLM**: Masked Diffusion Language Model for discrete text generation
- **SEDD**: Score Entropy Discrete Diffusion for discrete token generation
- **DiT**: Diffusion Transformer architecture for image generation
- **VAE**: Variational Autoencoder for encoding/decoding images to/from latent space
- **Modality**: Type of data being generated (text, code, image, audio)
- **Projection_Head**: Modality-specific lightweight decoder projecting shared representations to modality space

## Requirements

### REQ-1: Discrete Diffusion Text and Code Generation

**User Story:** As a developer, I want to generate text and code using discrete diffusion models so that I can leverage bidirectional context and support infilling capabilities beyond left-to-right generation.

#### Acceptance Criteria

1. When receiving a text generation request, DiffusionCore shall initialize a fully masked token sequence
2. When NoiseScheduler performs forward noising on text, the system shall apply token masking according to the noise schedule
3. When DiscreteSampler performs denoising steps, the system shall unmask tokens based on model confidence scores
4. When generating text with infilling context, the system shall preserve provided prefix and suffix tokens while generating middle content
5. When text generation completes, the system shall return a coherent text sequence with perplexity within 20% of equivalent AR baseline
6. When requesting code generation, the system shall use the same discrete diffusion process as text generation
7. When applying 4-step consistency distillation, the system shall complete text generation in <500ms on CPU with INT2 quantized 350M parameter model

### REQ-2: Continuous Diffusion Image Generation

**User Story:** As a user, I want to generate images from text descriptions using diffusion models so that I can create visual content based on my memories and preferences.

#### Acceptance Criteria

1. When receiving an image generation request, DiffusionCore shall initialize Gaussian noise in VAE latent space
2. When VAE encoder processes images, the system shall produce latent representations with reduced dimensionality
3. When ContinuousSampler performs denoising steps, the system shall apply score-based updates to latent representations
4. When image generation completes, VAE decoder shall reconstruct pixel-space images from denoised latent representations
5. When generating 512x512 images with models under 600M parameters, the system shall complete generation in <30s on CPU
6. When applying INT4 quantization to image models, the system shall maintain model size under 200MB
7. When requesting memory-guided image generation, the system shall incorporate ArrowStorage retrieval results as conditioning

### REQ-3: Continuous Diffusion Audio Generation

**User Story:** As a user, I want to generate speech and audio from text using diffusion models so that I can create natural-sounding voice output for virtual avatars.

#### Acceptance Criteria

1. When receiving text-to-speech requests, DiffusionCore shall initialize noise in audio waveform or latent space
2. When audio projection head decodes audio, the system shall produce waveforms with minimum 16kHz sampling rate
3. When requesting TTS generation, the system shall complete end-to-end generation in <2s
4. When requesting zero-shot voice cloning with 5s reference audio, the system shall generate speech matching reference voice characteristics
5. When audio generation completes, the system shall output audio with natural prosody and intelligibility

### REQ-4: Unified Score Network Architecture

**User Story:** As a system architect, I want all modalities to share a single Transformer backbone so that I can minimize memory footprint and enable cross-modal knowledge transfer.

#### Acceptance Criteria

1. Score_Network shall contain a SharedTransformer component processing all modalities
2. When computing model parameters, SharedTransformer shall account for at least 90% of total parameters
3. When processing input for any modality, Score_Network shall apply modality-specific embeddings to distinguish modality types
4. When processing input at timestep t, Score_Network shall apply sinusoidal time embeddings to encode denoising step
5. When adding a new modality, the system shall only require training a new Projection_Head with <10M parameters
6. When generating multiple modalities simultaneously, SharedTransformer shall perform a single forward pass producing hidden states for all Projection_Heads
7. Score_Network shall support independent Projection_Heads for text, code, image, and audio modalities

### REQ-5: Memory-Guided Diffusion Conditioning

**User Story:** As a user, I want my personal memories to guide content generation so that generated outputs reflect my experiences and preferences.

#### Acceptance Criteria

1. When generation request includes memory guidance, MemoryConditioner shall query ArrowStorage with user prompt
2. When ArrowStorage returns retrieval results, MemoryConditioner shall extract embedding vectors from top-K memories
3. When extracting memory embeddings, MemoryConditioner shall project them to condition dimension
4. When Score_Network performs denoising, the system shall inject condition vectors via cross-attention mechanism
5. When memory-guided generation completes, the system shall produce outputs semantically related to retrieved memories with >80% relevance rate
6. When applying memory conditioning, the system shall complete condition projection in <10ms

### REQ-6: Uncertainty-Aware Self-Evolution Triggering

**User Story:** As a system designer, I want the model to detect when it is uncertain so that I can trigger self-evolution mechanisms only when needed.

#### Acceptance Criteria

1. When Score_Network performs denoising steps, uncertainty estimator shall compute denoising residual norm
2. When computing uncertainty, uncertainty estimator shall normalize residual by expected noise level at timestep t
3. When uncertainty metric exceeds threshold 1.5, the system shall trigger self-evolution mechanism
4. When uncertainty is below threshold, the system shall continue normal generation without evolution
5. When evaluated against human-annotated uncertain scenarios, uncertainty estimator shall achieve >0.7 correlation
6. When measuring false trigger rate, the system shall maintain <5% rate of incorrect evolution triggers

### REQ-7: Consistency Distillation Acceleration

**User Story:** As a developer deploying to edge devices, I want fast generation with minimal quality loss so that the system can run in real-time on resource-constrained hardware.

#### Acceptance Criteria

1. When applying consistency distillation training, the system shall compress 50-step diffusion process to 1-4 steps
2. When generating with 4-step distilled model, the system shall maintain at least 90% quality of 50-step model
3. When measuring text generation latency on CPU with INT2 quantized 350M parameters, the system shall complete generation in <500ms
4. When applying consistency distillation, the system shall preserve ability to use variable step counts for quality-speed tradeoffs

### REQ-8: ControlNet Structural Constraints

**User Story:** As a developer, I want to inject behavioral and structural constraints into generation so that outputs follow specific patterns without retraining the base model.

#### Acceptance Criteria

1. When loading ControlNet, the system shall support zero-initialized residual connections to Score_Network
2. When multiple ControlNets are active, EvolutionRouter shall combine their outputs via weighted sum
3. When applying ControlNet, the system shall maintain ControlNet parameter count below 10% of base model parameters
4. When CoT-ControlNet is active, the system shall enforce chain-of-thought reasoning structure in text generation
5. When ToolSchema-ControlNet is active, the system shall enforce JSON schema compliance for tool call outputs
6. When storing ControlNet weights, the system shall use Parquet V2 format with ArrowQuant support

### REQ-9: Progressive 5-Level Self-Evolution

**User Story:** As a system architect, I want a progressive evolution mechanism from zero-training to full fine-tuning so that the system can adapt efficiently based on uncertainty level.

#### Acceptance Criteria

1. EvolutionRouter shall implement Level 0 score composition via weighted sum of multiple score predictions
2. EvolutionRouter shall implement Level 1 ControlNet injection with ~10% parameter training for structural preferences
3. EvolutionRouter shall implement Level 2 LoRA fine-tuning with ~1% parameter training for domain knowledge
4. EvolutionRouter shall implement Level 3 selective backbone fine-tuning with uncertainty-driven layer unfreezing
5. EvolutionRouter shall implement Level 4 full model fine-tuning for long-term consolidation
6. When uncertainty is detected, EvolutionRouter shall select appropriate evolution level based on uncertainty magnitude
7. When evolution completes at any level, the system shall validate improvement before persisting adaptation

### REQ-10: Energy-Based Model Constraint Fusion

**User Story:** As a developer, I want to enforce hard constraints on generation so that outputs satisfy physical or logical requirements.

#### Acceptance Criteria

1. When EBM is registered, EvolutionRouter shall compute energy gradients for constraint enforcement
2. When combining scores with EBM constraints, the system shall apply formula: final_score = diffusion_score - eta * energy_gradient
3. When EBM constraints are active, the system shall support multiple simultaneous energy models with independent weights
4. When EBM validation fails, the system shall reject generated samples and retry with stronger constraint weights

### REQ-11: Unified Noise Scheduling

**User Story:** As a developer, I want a noise scheduler handling both discrete and continuous modalities so that the system has consistent behavior across generation types.

#### Acceptance Criteria

1. NoiseScheduler shall support both discrete masking schedules and continuous Gaussian noise schedules
2. When operating in discrete mode, NoiseScheduler shall compute mask probability as function of timestep
3. When operating in continuous mode, NoiseScheduler shall compute noise standard deviation as function of timestep
4. NoiseScheduler shall support cosine, linear, and custom schedule functions
5. When generating inference timesteps, NoiseScheduler shall support uniform and non-uniform sampling strategies
6. When adding noise during training, NoiseScheduler shall apply appropriate noise type based on modality

### REQ-12: ArrowEngine Integration

**User Story:** As a user, I want seamless access to diffusion generation through existing ArrowEngine API so that I can use both AR and diffusion models without changing my workflow.

#### Acceptance Criteria

1. ArrowEngine shall maintain backward compatibility with existing encode() and generate() methods
2. When mode="diffusion" is specified, ArrowEngine shall route requests to DiffusionCore instead of InferenceCore
3. ArrowEngine shall expose new diffuse() method accepting prompt, modality, num_steps, and guidance_scale parameters
4. When diffuse() is called with memory_guided=True, the system shall automatically query ArrowStorage for conditioning
5. When diffuse() is called with modality="text", the system shall return decoded text string
6. When diffuse() is called with modality="image", the system shall return pixel-space numpy array or tensor
7. When diffuse() is called with modality="audio", the system shall return waveform numpy array

### REQ-13: Parquet V2 Storage Format

**User Story:** As a system architect, I want all diffusion model weights stored in Parquet V2 format so that I can leverage zero-copy loading and ArrowQuant compression.

#### Acceptance Criteria

1. When storing Score_Network weights, the system shall use Parquet V2 file format
2. When storing Projection_Head weights, the system shall use Parquet V2 file format
3. When storing ControlNet weights, the system shall use Parquet V2 file format
4. When storing VAE weights, the system shall use Parquet V2 file format
5. The system shall support ArrowQuant INT2 and INT4 quantization for all stored weights
6. When loading weights, the system shall support zero-copy memory mapping
7. When loading weights, the system shall support lazy dequantization on access

### REQ-14: Tiered Deployment Support

**User Story:** As a deployment engineer, I want to deploy appropriate model sizes to different hardware tiers so that the system runs efficiently from mobile devices to cloud servers.

#### Acceptance Criteria

1. When deploying to edge devices with 2-4GB RAM, the system shall support models with <100M parameters under INT2 quantization with total size <35MB
2. When deploying to local workstations with 8+GB RAM, the system shall support models with <600M parameters under INT4 quantization with total size <200MB
3. When deploying to cloud GPU servers, the system shall support models up to 3B parameters
4. When deploying to edge devices, the system shall support at minimum text and audio modalities
5. When deploying to local workstations, the system shall support text, audio, and image modalities
6. When deploying to cloud servers, the system shall support all modalities including unified Score_Network

### REQ-15: Model Metadata and Configuration

**User Story:** As a developer, I want diffusion models to have comprehensive metadata so that the system can automatically configure itself based on model capabilities.

#### Acceptance Criteria

1. When storing diffusion model, the system shall include metadata.json file with model_type="unified_diffusion"
2. metadata.json shall specify diffusion_config including hidden_dim, num_layers, num_heads, and intermediate_dim
3. metadata.json shall specify scheduler configuration including type, num_train_steps, and num_inference_steps
4. metadata.json shall list supported_modalities as array
5. metadata.json shall indicate whether consistency_distilled is true or false
6. metadata.json shall specify quantization method and bit_width
7. When loading model, the system shall parse metadata.json to configure DiffusionCore correctly

### REQ-16: Multimodal Parallel Generation

**User Story:** As a user interacting with a virtual avatar, I want synchronized text, audio, and visual outputs so that the avatar responds naturally without cascading delays.

#### Acceptance Criteria

1. When multimodal generation is requested, Score_Network shall perform single forward pass for all modalities
2. When SharedTransformer produces hidden states, the system shall route them to all active Projection_Heads simultaneously
3. When generating text, audio, and avatar parameters concurrently, the system shall ensure temporal synchronization
4. When multimodal generation completes, the system shall return all outputs with aligned timing information

### REQ-17: Backward Compatibility with AR Models

**User Story:** As a system operator, I want to maintain existing AR model functionality so that I can gradually transition to diffusion models without breaking existing workflows.

#### Acceptance Criteria

1. When mode="ar" is specified, ArrowEngine shall route to existing InferenceCore
2. The system shall support loading both AR models and diffusion models simultaneously
3. When AR model is active, the system shall continue supporting existing LoRA Router functionality
4. The system shall not allow mixing AR model LoRAs with diffusion model LoRAs
5. When switching between AR and diffusion modes, the system shall maintain consistent API behavior for common parameters

### REQ-18: Self-Evolution Closed Loop

**User Story:** As a system architect, I want a complete feedback loop from memory to generation to evolution so that the system continuously improves based on usage.

#### Acceptance Criteria

1. When generation completes with high uncertainty, the system shall trigger appropriate evolution level
2. When evolution produces adaptation, the system shall validate improvement before persisting
3. When adaptation is validated, the system shall store it to ArrowStorage for future retrieval
4. When similar context is encountered, the system shall retrieve and apply learned adaptations
5. The system shall maintain closed loop: ArrowStorage → MemoryConditioner → Score_Network → UncertaintyEstimator → EvolutionRouter → ArrowStorage

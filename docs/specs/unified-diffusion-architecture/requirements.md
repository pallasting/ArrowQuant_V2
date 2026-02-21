# Requirements Document: Unified Diffusion Architecture

## Introduction

The Unified Diffusion Architecture replaces autoregressive (AR) models with thermodynamic diffusion models to enable unified multi-modal generation (text, code, image, audio) through a single shared Score Network. This system integrates with the existing ArrowEngine infrastructure, supports memory-guided generation using ArrowStorage, and implements a 5-level progressive self-evolution mechanism with edge deployment capabilities.

## Glossary

- **Score_Network**: The unified neural network that learns the gradient of the log probability distribution (score function) for denoising across all modalities
- **DiffusionCore**: The core inference engine that manages the iterative denoising process for diffusion-based generation
- **ArrowEngine**: The existing inference routing layer that manages model loading and execution
- **ArrowStorage**: The memory storage system using Hopfield networks for vector-based memory retrieval
- **ArrowQuant**: The quantization system supporting INT2/INT4 weight compression with zero-copy loading
- **MemoryConditioner**: Component that converts ArrowStorage retrieval results into diffusion conditioning vectors
- **NoiseScheduler**: Component that manages noise levels across diffusion timesteps for both discrete and continuous modalities
- **DiscreteSampler**: Sampler for discrete modalities (text, code) using mask-based denoising
- **ContinuousSampler**: Sampler for continuous modalities (image, audio) using Gaussian noise denoising
- **ControlNet**: Lightweight auxiliary network that injects structural or behavioral constraints without modifying base weights
- **LoRA**: Low-Rank Adaptation for efficient model fine-tuning with minimal parameter updates
- **EBM**: Energy-Based Model used for constraint validation and guidance
- **Consistency_Distillation**: Training technique to compress multi-step diffusion into 1-4 steps while preserving quality
- **MDLM**: Masked Diffusion Language Model for discrete text generation
- **SEDD**: Score Entropy Discrete Diffusion for discrete token generation
- **DiT**: Diffusion Transformer architecture for image generation
- **VAE**: Variational Autoencoder for encoding/decoding images to/from latent space
- **Modality**: The type of data being generated (text, code, image, audio)
- **Projection_Head**: Modality-specific lightweight decoder that projects shared representations to modality space

## Requirements

### Requirement 1: Discrete Diffusion Text and Code Generation

**User Story:** As a developer, I want to generate text and code using discrete diffusion models, so that I can leverage bidirectional context and support infilling capabilities beyond left-to-right generation.

#### Acceptance Criteria

1. WHEN a text generation request is received, THE DiffusionCore SHALL initialize a fully masked token sequence
2. WHEN the NoiseScheduler performs forward noising on text, THE System SHALL apply token masking according to the noise schedule
3. WHEN the DiscreteSampler performs a denoising step, THE System SHALL unmask tokens based on model confidence scores
4. WHEN generating text with infilling context, THE System SHALL preserve provided prefix and suffix tokens while generating middle content
5. WHEN text generation completes, THE System SHALL return a coherent text sequence with perplexity within 20% of equivalent AR baseline
6. WHEN code generation is requested, THE System SHALL use the same discrete diffusion process as text generation
7. WHEN 4-step Consistency Distillation is applied, THE System SHALL complete text generation in under 500ms on CPU with 350M parameter model at INT2 quantization

### Requirement 2: Continuous Diffusion Image Generation

**User Story:** As a user, I want to generate images from text descriptions using diffusion models, so that I can create visual content guided by my memories and preferences.

#### Acceptance Criteria

1. WHEN an image generation request is received, THE DiffusionCore SHALL initialize Gaussian noise in the VAE latent space
2. WHEN the VAE Encoder processes an image, THE System SHALL produce a latent representation with dimensionality reduction
3. WHEN the ContinuousSampler performs a denoising step, THE System SHALL apply score-based updates to the latent representation
4. WHEN image generation completes, THE VAE_Decoder SHALL reconstruct a pixel-space image from the denoised latent
5. WHEN generating a 512x512 image with a model under 600M parameters, THE System SHALL complete generation in under 30 seconds on CPU
6. WHEN INT4 quantization is applied to the image model, THE System SHALL maintain model size under 200MB
7. WHEN memory-guided image generation is requested, THE System SHALL incorporate ArrowStorage retrieval results as conditioning

### Requirement 3: Continuous Diffusion Audio Generation

**User Story:** As a user, I want to generate speech and audio from text using diffusion models, so that I can create natural-sounding voice output for virtual embodiment.

#### Acceptance Criteria

1. WHEN a text-to-speech request is received, THE DiffusionCore SHALL initialize noise in the audio waveform or latent space
2. WHEN the AudioProjectionHead decodes audio, THE System SHALL produce waveforms at minimum 16kHz sampling rate
3. WHEN TTS generation is requested, THE System SHALL complete end-to-end generation in under 2 seconds
4. WHEN zero-shot voice cloning is requested with a 5-second reference audio, THE System SHALL generate speech matching the reference voice characteristics
5. WHEN audio generation completes, THE System SHALL output audio with natural prosody and intelligibility

### Requirement 4: Unified Score Network Architecture

**User Story:** As a system architect, I want all modalities to share a single Transformer backbone, so that I can minimize memory footprint and enable cross-modal knowledge transfer.

#### Acceptance Criteria

1. THE Score_Network SHALL contain a SharedTransformer component that processes all modalities
2. WHEN computing model parameters, THE SharedTransformer SHALL account for at least 90% of total parameters
3. WHEN processing input for any modality, THE Score_Network SHALL apply modality-specific embeddings to distinguish modality type
4. WHEN processing input at timestep t, THE Score_Network SHALL apply sinusoidal time embeddings to encode the denoising step
5. WHEN adding a new modality, THE System SHALL require only training a new Projection_Head with less than 10M parameters
6. WHEN generating multiple modalities simultaneously, THE SharedTransformer SHALL perform a single forward pass producing hidden states for all Projection_Heads
7. THE Score_Network SHALL support separate Projection_Heads for text, code, image, and audio modalities

### Requirement 5: Memory-Guided Diffusion Conditioning

**User Story:** As a user, I want my personal memories to guide content generation, so that generated outputs reflect my experiences and preferences.

#### Acceptance Criteria

1. WHEN a generation request includes memory guidance, THE MemoryConditioner SHALL query ArrowStorage with the user prompt
2. WHEN ArrowStorage returns retrieval results, THE MemoryConditioner SHALL extract embedding vectors from top-K memories
3. WHEN memory embeddings are extracted, THE MemoryConditioner SHALL project them to the conditioning dimension
4. WHEN the Score_Network performs denoising, THE System SHALL inject conditioning vectors via cross-attention mechanism
5. WHEN memory-guided generation completes, THE System SHALL produce outputs semantically related to retrieved memories with over 80% relevance rate
6. WHEN memory conditioning is applied, THE System SHALL complete conditioning projection in under 10ms

### Requirement 6: Uncertainty-Aware Self-Evolution Triggering

**User Story:** As a system designer, I want the model to detect when it is uncertain, so that self-evolution mechanisms can be triggered only when needed.

#### Acceptance Criteria

1. WHEN the Score_Network performs a denoising step, THE UncertaintyEstimator SHALL compute the denoising residual norm
2. WHEN computing uncertainty, THE UncertaintyEstimator SHALL normalize the residual by the expected noise level at timestep t
3. WHEN the uncertainty metric exceeds a threshold of 1.5, THE System SHALL trigger self-evolution mechanisms
4. WHEN uncertainty is below threshold, THE System SHALL proceed with normal generation without evolution
5. WHEN evaluated against human-annotated uncertain scenarios, THE UncertaintyEstimator SHALL achieve correlation above 0.7
6. WHEN measuring false trigger rate, THE System SHALL maintain false evolution triggers below 5%

### Requirement 7: Consistency Distillation for Acceleration

**User Story:** As a developer deploying to edge devices, I want fast generation with minimal quality loss, so that the system can run in real-time on resource-constrained hardware.

#### Acceptance Criteria

1. WHEN Consistency_Distillation training is applied, THE System SHALL compress a 50-step diffusion process into 1-4 steps
2. WHEN generating with 4-step distilled model, THE System SHALL maintain at least 90% of the quality of the 50-step model
3. WHEN measuring text generation latency with 350M parameters at INT2 on CPU, THE System SHALL complete generation in under 500ms
4. WHEN Consistency_Distillation is applied, THE System SHALL preserve the ability to use variable step counts for quality-speed tradeoffs

### Requirement 8: ControlNet Structural Constraints

**User Story:** As a developer, I want to inject behavioral and structural constraints into generation, so that outputs follow specific patterns without retraining the base model.

#### Acceptance Criteria

1. WHEN a ControlNet is loaded, THE System SHALL support zero-initialized residual connections to the Score_Network
2. WHEN multiple ControlNets are active, THE EvolutionRouter SHALL combine their outputs with weighted summation
3. WHEN a ControlNet is applied, THE System SHALL maintain ControlNet parameter count below 10% of base model parameters
4. WHEN CoT-ControlNet is active, THE System SHALL enforce chain-of-thought reasoning structure in text generation
5. WHEN ToolSchema-ControlNet is active, THE System SHALL enforce JSON schema compliance for tool calling outputs
6. WHEN ControlNet weights are stored, THE System SHALL use Parquet V2 format with ArrowQuant support

### Requirement 9: Progressive 5-Level Self-Evolution

**User Story:** As a system architect, I want a gradual evolution mechanism from zero-training to full fine-tuning, so that the system can adapt efficiently based on uncertainty levels.

#### Acceptance Criteria

1. THE EvolutionRouter SHALL implement Level 0 score combination through weighted summation of multiple score predictions
2. THE EvolutionRouter SHALL implement Level 1 ControlNet injection for structural preferences with approximately 10% parameter training
3. THE EvolutionRouter SHALL implement Level 2 LoRA fine-tuning for domain knowledge with approximately 1% parameter training
4. THE EvolutionRouter SHALL implement Level 3 selective backbone fine-tuning with uncertainty-driven layer unfreezing
5. THE EvolutionRouter SHALL implement Level 4 full model fine-tuning for long-term consolidation
6. WHEN uncertainty is detected, THE EvolutionRouter SHALL select the appropriate evolution level based on uncertainty magnitude
7. WHEN evolution completes at any level, THE System SHALL validate improvements before persisting adaptations

### Requirement 10: Energy-Based Model Constraint Fusion

**User Story:** As a developer, I want to enforce hard constraints on generation, so that outputs satisfy physical or logical requirements.

#### Acceptance Criteria

1. WHEN an EBM is registered, THE EvolutionRouter SHALL compute energy gradients for constraint enforcement
2. WHEN combining scores with EBM constraints, THE System SHALL apply the formula: final_score = diffusion_score - eta * energy_gradient
3. WHEN EBM constraints are active, THE System SHALL support multiple simultaneous energy models with independent weights
4. WHEN EBM validation fails, THE System SHALL reject generated samples and retry with stronger constraint weighting

### Requirement 11: Unified Noise Scheduling

**User Story:** As a developer, I want a single noise scheduler that handles both discrete and continuous modalities, so that the system has consistent behavior across generation types.

#### Acceptance Criteria

1. THE NoiseScheduler SHALL support both discrete masking schedules and continuous Gaussian noise schedules
2. WHEN operating in discrete mode, THE NoiseScheduler SHALL compute mask probabilities as a function of timestep
3. WHEN operating in continuous mode, THE NoiseScheduler SHALL compute noise standard deviation as a function of timestep
4. THE NoiseScheduler SHALL support cosine, linear, and custom scheduling functions
5. WHEN generating inference timesteps, THE NoiseScheduler SHALL support uniform and non-uniform sampling strategies
6. WHEN adding noise during training, THE NoiseScheduler SHALL apply the appropriate noise type based on modality

### Requirement 12: ArrowEngine Integration

**User Story:** As a user, I want seamless access to diffusion generation through the existing ArrowEngine API, so that I can use both AR and diffusion models without changing my workflow.

#### Acceptance Criteria

1. THE ArrowEngine SHALL maintain backward compatibility with existing encode() and generate() methods
2. WHEN mode="diffusion" is specified, THE ArrowEngine SHALL route requests to DiffusionCore instead of InferenceCore
3. THE ArrowEngine SHALL expose a new diffuse() method accepting prompt, modality, num_steps, and guidance_scale parameters
4. WHEN diffuse() is called with memory_guided=True, THE System SHALL automatically query ArrowStorage for conditioning
5. WHEN diffuse() is called with modality="text", THE System SHALL return decoded text strings
6. WHEN diffuse() is called with modality="image", THE System SHALL return pixel-space numpy arrays or tensors
7. WHEN diffuse() is called with modality="audio", THE System SHALL return waveform numpy arrays

### Requirement 13: Parquet V2 Storage Format

**User Story:** As a system architect, I want all diffusion model weights stored in Parquet V2 format, so that I can leverage zero-copy loading and ArrowQuant compression.

#### Acceptance Criteria

1. WHEN storing Score_Network weights, THE System SHALL use Parquet V2 file format
2. WHEN storing Projection_Head weights, THE System SHALL use Parquet V2 file format
3. WHEN storing ControlNet weights, THE System SHALL use Parquet V2 file format
4. WHEN storing VAE weights, THE System SHALL use Parquet V2 file format
5. THE System SHALL support ArrowQuant INT2 and INT4 quantization for all stored weights
6. WHEN loading weights, THE System SHALL support zero-copy memory mapping
7. WHEN loading weights, THE System SHALL support lazy dequantization on access

### Requirement 14: Tiered Deployment Support

**User Story:** As a deployment engineer, I want to deploy appropriate model sizes to different hardware tiers, so that the system runs efficiently from mobile devices to cloud servers.

#### Acceptance Criteria

1. WHEN deploying to edge devices with 2-4GB RAM, THE System SHALL support models under 100M parameters at INT2 quantization with total size under 35MB
2. WHEN deploying to local workstations with 8+GB RAM, THE System SHALL support models under 600M parameters at INT4 quantization with total size under 200MB
3. WHEN deploying to cloud GPU servers, THE System SHALL support models up to 3B parameters
4. WHEN deploying to edge devices, THE System SHALL support text and audio modalities as minimum requirements
5. WHEN deploying to local workstations, THE System SHALL support text, audio, and image modalities
6. WHEN deploying to cloud servers, THE System SHALL support all modalities including unified Score_Network

### Requirement 15: Model Metadata and Configuration

**User Story:** As a developer, I want comprehensive metadata for diffusion models, so that the system can automatically configure itself based on model capabilities.

#### Acceptance Criteria

1. WHEN a diffusion model is stored, THE System SHALL include a metadata.json file with model_type="unified_diffusion"
2. THE metadata.json SHALL specify diffusion_config including hidden_dim, num_layers, num_heads, and intermediate_dim
3. THE metadata.json SHALL specify scheduler configuration including type, num_train_steps, and num_inference_steps
4. THE metadata.json SHALL list supported_modalities as an array
5. THE metadata.json SHALL indicate whether consistency_distilled is true or false
6. THE metadata.json SHALL specify quantization method and bit_width
7. WHEN loading a model, THE System SHALL parse metadata.json to configure DiffusionCore appropriately

### Requirement 16: Multi-Modal Parallel Generation

**User Story:** As a user interacting with a virtual embodiment, I want synchronized text, audio, and visual outputs, so that the avatar responds naturally without cascading delays.

#### Acceptance Criteria

1. WHEN multi-modal generation is requested, THE Score_Network SHALL perform a single forward pass for all modalities
2. WHEN the SharedTransformer produces hidden states, THE System SHALL route them to all active Projection_Heads simultaneously
3. WHEN generating text, audio, and avatar parameters together, THE System SHALL ensure temporal synchronization
4. WHEN multi-modal generation completes, THE System SHALL return all outputs with aligned timing information

### Requirement 17: Backward Compatibility with AR Models

**User Story:** As a system operator, I want to maintain existing AR model functionality, so that I can gradually transition to diffusion models without breaking existing workflows.

#### Acceptance Criteria

1. WHEN mode="ar" is specified, THE ArrowEngine SHALL route to the existing InferenceCore
2. THE System SHALL support loading both AR models and diffusion models simultaneously
3. WHEN AR models are active, THE System SHALL continue to support existing LoRA Router functionality
4. THE System SHALL NOT allow mixing AR model LoRAs with diffusion model LoRAs
5. WHEN switching between AR and diffusion modes, THE System SHALL maintain consistent API behavior for common parameters

### Requirement 18: Self-Evolution Closed Loop

**User Story:** As a system architect, I want a complete feedback loop from memory to generation to evolution, so that the system continuously improves based on usage.

#### Acceptance Criteria

1. WHEN generation completes with high uncertainty, THE System SHALL trigger appropriate evolution level
2. WHEN evolution produces adaptations, THE System SHALL validate improvements before persistence
3. WHEN adaptations are validated, THE System SHALL store them to ArrowStorage for future retrieval
4. WHEN similar contexts are encountered, THE System SHALL retrieve and apply learned adaptations
5. THE System SHALL maintain a closed loop: ArrowStorage → MemoryConditioner → Score_Network → UncertaintyEstimator → EvolutionRouter → ArrowStorage

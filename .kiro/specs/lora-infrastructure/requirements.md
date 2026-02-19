# Requirements Document: LoRA Infrastructure

## Introduction

The LoRA (Low-Rank Adaptation) Infrastructure implements a zero-copy, Arrow-native system for dynamic model adaptation in the AI-OS memory system. This infrastructure enables efficient loading, injection, and hot-swapping of LoRA adapters without model retraining, supporting the self-evolving intelligence capabilities of ArrowEngine.

The system leverages PyArrow for zero-copy weight loading, achieving sub-second adapter injection with minimal memory overhead. LoRA adapters are stored in Arrow IPC format, enabling memory-mapped loading and seamless integration with the existing ArrowEngine inference pipeline.

## Glossary

- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning technique that adds trainable low-rank matrices to frozen model weights
- **LoRACard**: Data structure representing a complete LoRA adapter with weights, metadata, and configuration
- **LoRALinear**: Wrapper layer that injects LoRA weights into existing Linear layers
- **LoRAManager**: Lifecycle manager for loading, applying, and removing LoRA adapters
- **LoRARouter**: Semantic routing system that selects relevant adapters based on user intent
- **Arrow IPC**: Inter-Process Communication format for zero-copy data serialization
- **InferenceCore**: The complete Transformer implementation that powers ArrowEngine
- **Hot-swapping**: Dynamic replacement of adapters without model reloading
- **Rank**: Dimensionality of low-rank decomposition (typically 4-64)
- **Alpha**: Scaling factor for LoRA weights (typically equals rank)

## Requirements

### Requirement 1: Arrow-Native LoRA Format

**User Story:** As a system architect, I want a standardized Arrow-based format for LoRA adapters, so that adapters can be loaded with zero-copy operations and minimal overhead.

#### Acceptance Criteria

1. THE LoRACard SHALL store adapter name, rank, alpha, and target modules as metadata
2. THE LoRACard SHALL store weight matrices A and B for each target module
3. THE LoRACard SHALL support arbitrary metadata fields (version, author, description)
4. THE LoRAFormat SHALL serialize LoRACard to Arrow IPC format with custom metadata
5. THE LoRAFormat SHALL deserialize LoRACard from Arrow IPC using memory-mapped files
6. WHEN loading a LoRACard, THE LoRAFormat SHALL use zero-copy operations where possible
7. THE LoRAFormat SHALL preserve weight dtypes (float32, float16) during serialization

### Requirement 2: LoRA Layer Injection

**User Story:** As a developer, I want to inject LoRA weights into existing Linear layers, so that I can adapt model behavior without modifying base weights.

#### Acceptance Criteria

1. THE LoRALinear SHALL wrap an existing nn.Linear layer without copying base weights
2. THE LoRALinear SHALL implement forward pass as: output = base_layer(x) + (x @ A @ B) * (alpha / rank)
3. THE LoRALinear SHALL support loading pre-trained LoRA weights (A and B matrices)
4. THE LoRALinear SHALL support training mode for fine-tuning LoRA weights
5. THE LoRALinear SHALL maintain the same input/output dimensions as the base layer
6. WHEN computing forward pass, THE LoRALinear SHALL add LoRA contribution to base output
7. THE LoRALinear SHALL support gradient computation for LoRA weights only (base frozen)

### Requirement 3: LoRA Manager Lifecycle

**User Story:** As a system operator, I want to manage LoRA adapter lifecycle, so that I can dynamically load, apply, and remove adapters at runtime.

#### Acceptance Criteria

1. THE LoRAManager SHALL load LoRACard from disk using LoRAFormat
2. THE LoRAManager SHALL inject LoRA weights into target modules of InferenceCore
3. THE LoRAManager SHALL track active adapters and injected layers
4. THE LoRAManager SHALL support applying multiple adapters to different layers
5. THE LoRAManager SHALL support removing adapters and restoring original layers
6. WHEN applying a LoRACard, THE LoRAManager SHALL inject into all matching target modules
7. WHEN removing a LoRACard, THE LoRAManager SHALL restore original nn.Linear layers
8. THE LoRAManager SHALL log injection statistics (number of layers modified)

### Requirement 4: Semantic LoRA Routing

**User Story:** As an AI system, I want to automatically select relevant LoRA adapters based on user intent, so that the system adapts to different tasks without manual intervention.

#### Acceptance Criteria

1. THE LoRARouter SHALL register LoRACard with semantic descriptions
2. THE LoRARouter SHALL compute embedding vectors for adapter descriptions
3. THE LoRARouter SHALL compute embedding vectors for user queries
4. THE LoRARouter SHALL select top-k adapters based on cosine similarity
5. THE LoRARouter SHALL support similarity threshold filtering
6. WHEN selecting adapters, THE LoRARouter SHALL return adapters above threshold
7. THE LoRARouter SHALL support registering virtual candidates without loading weights
8. THE LoRARouter SHALL normalize embeddings for cosine similarity computation

### Requirement 5: ArrowEngine Integration

**User Story:** As a developer, I want seamless integration with ArrowEngine, so that LoRA adapters work with existing inference pipelines.

#### Acceptance Criteria

1. THE ArrowEngine SHALL support encode_with_lora method for adapter-aware encoding
2. THE ArrowEngine SHALL initialize LoRAManager with InferenceCore instance
3. THE ArrowEngine SHALL support loading and applying LoRA adapters by path
4. THE ArrowEngine SHALL maintain backward compatibility with non-LoRA workflows
5. WHEN using LoRA adapters, THE ArrowEngine SHALL produce embeddings with adapter modifications
6. WHEN no adapters are active, THE ArrowEngine SHALL behave identically to base model

### Requirement 6: Performance Targets

**User Story:** As a performance engineer, I want efficient LoRA operations, so that adapter injection and inference have minimal overhead.

#### Acceptance Criteria

1. WHEN loading a LoRACard, THE System SHALL complete in less than 100ms
2. WHEN applying a LoRACard, THE System SHALL inject into all layers in less than 500ms
3. WHEN computing forward pass with LoRA, THE System SHALL add less than 10% latency overhead
4. THE System SHALL use memory-mapped loading to avoid copying weight data
5. THE System SHALL maintain memory usage below 50MB per active adapter
6. WHEN removing a LoRACard, THE System SHALL restore layers in less than 100ms

### Requirement 7: Error Handling and Validation

**User Story:** As a developer, I want robust error handling, so that the system provides clear feedback when operations fail.

#### Acceptance Criteria

1. WHEN a LoRACard file is missing, THE System SHALL raise a descriptive error
2. WHEN LoRA weights have mismatched dimensions, THE System SHALL raise a descriptive error
3. WHEN target modules are not found, THE System SHALL log a warning and continue
4. WHEN applying an already-active adapter, THE System SHALL log a warning and skip
5. WHEN removing a non-existent adapter, THE System SHALL return silently
6. THE System SHALL validate LoRA rank and alpha parameters during loading
7. THE System SHALL validate weight matrix shapes match target layer dimensions

### Requirement 8: Multi-LoRA Support (Future)

**User Story:** As a researcher, I want to apply multiple LoRA adapters simultaneously, so that I can combine multiple task-specific adaptations.

#### Acceptance Criteria

1. THE System SHALL support applying multiple LoRA adapters to the same layer
2. THE System SHALL compute forward pass as: output = base(x) + sum(LoRA_i(x))
3. THE System SHALL track multiple adapters per layer
4. THE System SHALL support removing individual adapters from multi-LoRA layers
5. WHEN multiple adapters target the same module, THE System SHALL stack them
6. THE System SHALL maintain performance with up to 3 simultaneous adapters

### Requirement 9: Testing and Validation

**User Story:** As a quality engineer, I want comprehensive tests, so that I can verify the system works correctly.

#### Acceptance Criteria

1. THE System SHALL provide unit tests for LoRAFormat serialization/deserialization
2. THE System SHALL provide unit tests for LoRALinear forward pass computation
3. THE System SHALL provide unit tests for LoRAManager injection and removal
4. THE System SHALL provide integration tests for end-to-end LoRA workflow
5. THE System SHALL validate LoRA output matches expected mathematical formula
6. THE System SHALL test error handling for invalid inputs and edge cases
7. WHEN all tests run, THE System SHALL achieve 100% pass rate

### Requirement 10: Documentation

**User Story:** As a developer, I want comprehensive documentation, so that I can understand and use the LoRA infrastructure.

#### Acceptance Criteria

1. THE System SHALL provide API documentation for all public classes and methods
2. THE System SHALL provide examples for creating and saving LoRACard
3. THE System SHALL provide examples for loading and applying adapters
4. THE System SHALL provide examples for semantic routing
5. THE System SHALL document the Arrow IPC format specification
6. THE System SHALL document performance characteristics and limitations

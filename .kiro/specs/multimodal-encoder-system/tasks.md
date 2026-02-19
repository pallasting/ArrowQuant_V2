# Implementation Plan: Multimodal Encoder System

## Overview

This implementation plan extends ArrowEngine with vision and audio encoding capabilities, building on the existing 90% complete BERT-based text encoder. The approach emphasizes code reuse (InferenceCore for all Transformer layers), zero-copy Arrow architecture, and incremental validation at each step.

## Tasks

- [x] 1. Set up multimodal infrastructure and preprocessing
  - Create directory structure for multimodal components
  - Implement Arrow-native image preprocessing with zero-copy operations
  - Implement Arrow-native audio preprocessing with mel-spectrogram computation
  - Set up test data fixtures (diverse images and audio clips)
  - _Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 5.3_

- [ ]* 1.1 Write property test for image preprocessing
  - **Property 8: Image Preprocessing Correctness**
  - **Validates: Requirements 4.3**

- [ ]* 1.2 Write property test for mel-spectrogram computation
  - **Property 9: Mel-Spectrogram Correctness**
  - **Validates: Requirements 5.3**

- [-] 2. Implement Vision Encoder (CLIP ViT)
  - [x] 2.1 Implement PatchEmbedding module for 16x16 patch extraction
    - Conv2d-based patch embedding
    - Handle 224x224 RGB images
    - Output (batch, 196, 768) patch embeddings
    - _Requirements: 1.1_
  
  - [x] 2.2 Implement VisionEncoder class
    - Initialize patch embedding, CLS token, position embeddings
    - Integrate with InferenceCore for 12 Transformer layers
    - Implement CLS token pooling
    - Add pre/post LayerNorm
    - _Requirements: 1.2, 1.3, 1.4_
  
  - [x] 2.3 Implement weight loading for vision encoder
    - Load patch embedding weights from Parquet
    - Load CLS token and position embeddings
    - Load LayerNorm weights
    - Integrate with existing WeightLoader
    - _Requirements: 9.3_
  
  - [ ]* 2.4 Write property test for vision encoder output structure
    - **Property 1: Vision Encoder Output Structure**
    - **Validates: Requirements 1.1, 1.4**
  
  - [ ]* 2.5 Write unit tests for vision encoder edge cases
    - Test batch processing
    - Test normalization
    - Test device placement (CPU/GPU)
    - _Requirements: 1.5_

- [x] 3. Implement Audio Encoder (Whisper)
  - [x] 3.1 Implement MelSpectrogramProcessor
    - Pre-compute and cache mel filter banks
    - Implement STFT computation
    - Apply mel filters and log scaling
    - _Requirements: 2.1_
  
  - [x] 3.2 Implement AudioEncoder class
    - Initialize Conv1d layers for audio embedding
    - Add position embeddings
    - Integrate with InferenceCore for Transformer layers
    - Implement mean pooling over time dimension
    - _Requirements: 2.2, 2.3, 2.4_
  
  - [x] 3.3 Implement weight loading for audio encoder
    - Load Conv1d weights from Parquet
    - Load position embeddings
    - Load LayerNorm weights
    - _Requirements: 9.3_
  
  - [ ]* 3.4 Write property test for audio encoder output structure
    - **Property 3: Audio Encoder Output Structure**
    - **Validates: Requirements 2.1, 2.4**
  
  - [ ]* 3.5 Write unit tests for audio encoder edge cases
    - Test variable-length audio
    - Test batch processing
    - Test normalization
    - _Requirements: 2.5_

- [x] 4. Checkpoint - Ensure basic encoders work
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement CLIP Engine (Dual-Encoder)
  - [x] 5.1 Implement CLIPEngine class
    - Initialize text encoder (reuse ArrowEngine)
    - Initialize vision encoder (VisionEncoder)
    - Add projection layers (384→512 for text, 768→512 for vision)
    - Load logit_scale parameter
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  
  - [x] 5.2 Implement cross-modal similarity computation
    - Implement contrastive similarity with temperature scaling
    - Add batch similarity computation
    - Implement find_best_matches for retrieval
    - _Requirements: 3.5_
  
  - [ ]* 5.3 Write property test for CLIP projection dimensions
    - **Property 5: CLIP Projection Dimensions**
    - **Validates: Requirements 3.3, 3.4**
  
  - [ ]* 5.4 Write property test for CLIP contrastive alignment
    - **Property 7: CLIP Contrastive Alignment**
    - **Validates: Requirements 3.5**
  
  - [ ]* 5.5 Write unit tests for CLIP engine
    - Test text-image retrieval
    - Test batch processing
    - Test similarity computation
    - _Requirements: 3.6_

- [x] 6. Implement model conversion tools
  - [x] 6.1 Extend ModelConverter for CLIP support (Phase 1, Task 1.1)
    - Added _convert_clip() method to ModelConverter
    - Implemented _load_clip_model() for HuggingFace integration
    - Implemented _extract_clip_weights() for vision encoder extraction
    - Implemented _map_clip_keys() for key mapping (currently no-op)
    - Implemented _generate_clip_metadata() for CLIP-specific metadata
    - Integrated with existing convert() method routing
    - Created comprehensive test suite (9 tests, all passing)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [x] 6.2 Extend ModelConverter for Whisper support (Phase 1, Task 1.2)
    - Added _convert_whisper() method to ModelConverter
    - Implemented _load_whisper_model() for HuggingFace integration
    - Implemented _extract_whisper_weights() for encoder-only extraction
    - Implemented _map_whisper_keys() for key mapping (embed_positions → position_embedding)
    - Implemented _generate_whisper_metadata() for Whisper-specific metadata
    - Integrated with existing convert() method routing
    - Created comprehensive test suite (10 tests, all passing)
    - _Requirements: 7.1, 7.2, 7.3_
  
  - [x] 6.2.1 Add model type auto-detection (Phase 1, Task 1.3)
    - Implemented _detect_model_type() method
    - Detects CLIP, Whisper, and BERT models from name patterns
    - Falls back to config inspection when name is ambiguous
    - Returns "unknown" for unsupported models
    - Created comprehensive test suite (15 tests, all passing)
  
  - [x] 6.2.2 Update convert() for auto-detection (Phase 1, Task 1.4)
    - Updated convert() method to support model_type="auto"
    - Routes to appropriate converter based on detected type
    - Maintains backward compatibility with explicit type specification
    - Clear error messages for unknown/unsupported types
  
  - [x] 6.2.3 Standardize on Zstandard compression (Phase 1, Task 1.5)
    - Changed default compression from "lz4" to "zstd"
    - Added compression_level parameter (default: 3)
    - Updated _convert_to_arrow() to support compression levels
    - All tests updated and passing
  
  - [x] 6.2.4 Create unified CLI script (Phase 2, Task 2.1)
    - Created scripts/convert_model.py with comprehensive CLI interface
    - Supports auto-detection and all model types
    - Beautiful output formatting with banners and summaries
    - Comprehensive help messages and examples
    - Error handling with troubleshooting tips
  
  - [x] 6.2.5 Update documentation (Phase 2, Task 2.2)
    - Updated docs/QUICKSTART_MULTIMODAL.md with unified converter section
    - Documented all command-line options and examples
    - Marked legacy scripts as deprecated
    - Provided clear migration guide
  
  - [x] 6.2.6 Deprecate standalone scripts (Phase 2, Task 2.3)
    - Added deprecation warnings to convert_clip_to_parquet.py
    - Added deprecation warnings to convert_whisper_to_parquet.py
    - Scripts remain functional for backward compatibility
    - Clear removal timeline (v2.0.0)
  
  - [x] 6.2.7 Remove legacy code (Phase 2, Task 2.4)
    - Deleted llm_compression/tools/convert_clip.py
    - Verified no imports or references
    - Functionality preserved in ModelConverter

  - [x] 6.2 Extend ModelConverter for Whisper support (Phase 1, Task 1.2)
    - Added _convert_whisper() method to ModelConverter
    - Implemented _load_whisper_model() for HuggingFace integration
    - Implemented _extract_whisper_weights() for encoder-only extraction
    - Implemented _map_whisper_keys() for key mapping (embed_positions → position_embedding)
    - Implemented _generate_whisper_metadata() for Whisper-specific metadata
    - Integrated with existing convert() method routing
    - Created comprehensive test suite (10 tests, all passing)
    - _Requirements: 7.1, 7.2, 7.3_
  
  - [x] 6.3 Add conversion validation
    - Compare embeddings between original and converted models
    - Report compression ratio and file sizes
    - _Requirements: 6.6, 6.7, 7.4, 7.5_
  
  - [ ]* 6.4 Write property test for CLIP conversion round-trip
    - **Property 10: Model Conversion Round-Trip (CLIP)**
    - **Validates: Requirements 6.6**
  
  - [ ]* 6.5 Write property test for Whisper conversion round-trip
    - **Property 11: Model Conversion Round-Trip (Whisper)**
    - **Validates: Requirements 7.4**

- [x] 7. Checkpoint - Ensure conversion tools work
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement precision validation against HuggingFace
  - [x] 8.1 Create diverse test dataset
    - Collect 20+ diverse images (different scenes, objects, styles)
    - Collect 20+ diverse audio clips (speech, music, different speakers)
    - Create corresponding text descriptions
    - _Requirements: 8.1_
  
  - [x] 8.2 Implement precision validation for vision encoder
    - Load both ArrowEngine and HuggingFace CLIP vision encoders
    - Encode test images with both
    - Compute cosine similarities
    - Verify average similarity > 0.95
    - _Requirements: 1.6, 8.6_
  
  - [x] 8.3 Implement precision validation for audio encoder
    - Load both ArrowEngine and HuggingFace Whisper encoders
    - Encode test audio with both
    - Compute cosine similarities
    - Verify average similarity > 0.95
    - _Requirements: 2.6, 8.6_
  
  - [x] 8.4 Implement precision validation for CLIP engine
    - Load both ArrowEngine and HuggingFace CLIP
    - Compute text-image similarities with both
    - Compute Pearson correlation
    - Verify correlation > 0.95
    - _Requirements: 3.6, 8.6_
  
  - [ ]* 8.5 Write property test for vision encoder precision
    - **Property 2: Vision Encoder Precision**
    - **Validates: Requirements 1.6**
  
  - [ ]* 8.6 Write property test for audio encoder precision
    - **Property 4: Audio Encoder Precision**
    - **Validates: Requirements 2.6**
  
  - [ ]* 8.7 Write property test for CLIP similarity correlation
    - **Property 6: CLIP Similarity Correlation**
    - **Validates: Requirements 3.6**

- [ ] 9. Implement performance benchmarking
  - [x] 9.1 Create benchmark suite
    - Measure model loading time
    - Measure single encoding latency (image and audio)
    - Measure batch throughput
    - Measure memory usage
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [x] 9.2 Run benchmarks and validate targets
    - Vision encoder: <500ms load, <100ms encode, 150+ img/s, <1GB memory
    - Audio encoder: <500ms load, <200ms encode, 50+ audio/s, <500MB memory
    - Generate benchmark report
    - _Requirements: 8.7, 10.3, 10.4_
  
  - [ ]* 9.3 Write performance tests
    - Test vision encoder latency
    - Test audio encoder latency
    - Test batch throughput
    - _Requirements: 1.5, 2.5, 4.5, 5.5_

- [x] 10. Implement error handling and validation
  - [x] 10.1 Add input validation
    - Validate image dimensions and format
    - Validate audio sample rate and format
    - Provide descriptive error messages
    - _Requirements: 11.1, 11.2_
  
  - [x] 10.2 Add model loading error handling
    - Check model file existence
    - Validate weight integrity
    - Handle corrupted weights gracefully
    - _Requirements: 11.3_
  
  - [x] 10.3 Add precision warnings
    - Log precision metrics during validation
    - Warn if precision below threshold
    - Raise error if precision critically low
    - _Requirements: 11.4_
  
  - [x]* 10.4 Write unit tests for error handling
    - Test invalid image inputs
    - Test invalid audio inputs
    - Test missing model files
    - Test corrupted weights
    - _Requirements: 11.1, 11.2, 11.3, 11.5_

- [x] 11. Integrate with existing ArrowEngine infrastructure
  - [x] 11.1 Extend EmbeddingProvider protocol
    - Add encode_image method
    - Add encode_audio method
    - Add encode_multimodal method
    - Maintain backward compatibility with text-only interface
    - _Requirements: 9.5_
  
  - [x] 11.2 Implement MultimodalEmbeddingProvider
    - Support text, vision, and audio encoders
    - Support CLIP engine for cross-modal retrieval
    - Lazy initialization of encoders
    - _Requirements: 9.1, 9.2, 9.5_
  
  - [x] 11.3 Extend storage layer for multimodal embeddings
    - Add vision embedding schema
    - Add audio embedding schema
    - Add CLIP embedding schema
    - Implement multimodal storage class
    - _Requirements: 9.4_
  
  - [ ]* 11.4 Write integration tests
    - Test end-to-end image encoding and storage
    - Test end-to-end audio encoding and storage
    - Test cross-modal retrieval
    - Test backward compatibility with text-only workflows
    - _Requirements: 9.6_

- [x] 12. Create documentation and examples
  - [x] 12.1 Write API documentation
    - Document VisionEncoder class and methods
    - Document AudioEncoder class and methods
    - Document CLIPEngine class and methods
    - Document preprocessing utilities
    - _Requirements: 12.1_
  
  - [x] 12.2 Create usage examples
    - Example: Encode images with VisionEncoder
    - Example: Encode audio with AudioEncoder
    - Example: Cross-modal retrieval with CLIPEngine
    - Example: Batch processing for high throughput
    - _Requirements: 12.2, 12.3, 12.4_
  
  - [x] 12.3 Write quickstart guide
    - Installation instructions
    - Model conversion walkthrough
    - Basic usage examples
    - Performance optimization tips
    - _Requirements: 12.5, 12.6_

- [x] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based and unit tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties with 100+ iterations
- Unit tests validate specific examples and edge cases
- Integration tests verify component interactions and backward compatibility
- The implementation reuses InferenceCore for all Transformer computations, minimizing new code
- Zero-copy Arrow architecture is maintained throughout for optimal performance

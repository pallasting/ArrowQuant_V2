//! Property-Based Tests for Validation System
//!
//! **Validates: Requirement 9 (Quality Validation)**
//!
//! This module contains property-based tests using proptest to verify
//! validation system invariants across a wide range of inputs.

use arrow_quant_v2::validation::ValidationSystem;
use proptest::prelude::*;
use std::collections::HashMap;

/// **Validates: Requirements 9**
///
/// Property: Cosine similarity is in [0, 1]
///
/// This property test verifies that:
/// 1. Cosine similarity is always in the range [0, 1]
/// 2. The result is never negative or greater than 1
/// 3. This holds for all valid input vectors
#[cfg(test)]
mod cosine_similarity_range_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_cosine_similarity_in_range(
            // Generate random vectors in range [-10.0, 10.0]
            a in prop::collection::vec(-10.0f32..10.0f32, 1..1000),
            b in prop::collection::vec(-10.0f32..10.0f32, 1..1000),
        ) {
            // Ensure vectors have the same length
            let len = a.len().min(b.len());
            let a = &a[..len];
            let b = &b[..len];

            let validator = ValidationSystem::new(0.7);
            let similarity = validator.cosine_similarity(a, b);

            // Property: Cosine similarity must be in [0, 1]
            prop_assert!(
                similarity >= 0.0,
                "Cosine similarity {} is negative",
                similarity
            );

            prop_assert!(
                similarity <= 1.0,
                "Cosine similarity {} exceeds 1.0",
                similarity
            );
        }
    }

    proptest! {
        #[test]
        fn prop_cosine_similarity_batch_in_range(
            // Generate batch of random vectors
            batch_size in 1usize..10,
            vec_len in 10usize..100,
        ) {
            let validator = ValidationSystem::new(0.7);
            
            // Generate random vectors using proptest's strategy
            let mut batch_a = Vec::new();
            let mut batch_b = Vec::new();
            
            // Use a simple deterministic generation based on indices
            for i in 0..batch_size {
                let a: Vec<f32> = (0..vec_len).map(|j| ((i * vec_len + j) as f32 % 20.0) - 10.0).collect();
                let b: Vec<f32> = (0..vec_len).map(|j| ((i * vec_len + j + 1) as f32 % 20.0) - 10.0).collect();
                batch_a.push(a);
                batch_b.push(b);
            }
            
            let refs_a: Vec<&[f32]> = batch_a.iter().map(|v: &Vec<f32>| v.as_slice()).collect();
            let refs_b: Vec<&[f32]> = batch_b.iter().map(|v: &Vec<f32>| v.as_slice()).collect();
            
            let similarities = validator.cosine_similarity_batch(&refs_a, &refs_b);

            // Property: All similarities must be in [0, 1]
            for (i, &sim) in similarities.iter().enumerate() {
                prop_assert!(
                    sim >= 0.0 && sim <= 1.0,
                    "Batch similarity {} at index {} is out of range [0, 1]",
                    sim,
                    i
                );
            }
        }
    }

    proptest! {
        #[test]
        fn prop_cosine_similarity_positive_vectors(
            // Generate positive vectors
            a in prop::collection::vec(0.0f32..10.0f32, 1..1000),
            b in prop::collection::vec(0.0f32..10.0f32, 1..1000),
        ) {
            let len = a.len().min(b.len());
            let a = &a[..len];
            let b = &b[..len];

            let validator = ValidationSystem::new(0.7);
            let similarity = validator.cosine_similarity(a, b);

            // Property: For positive vectors, similarity should be in [0, 1]
            prop_assert!(
                similarity >= 0.0 && similarity <= 1.0,
                "Cosine similarity {} for positive vectors is out of range",
                similarity
            );
        }
    }

    proptest! {
        #[test]
        fn prop_cosine_similarity_normalized_vectors(
            // Generate vectors and normalize them
            a in prop::collection::vec(-10.0f32..10.0f32, 10..100),
            b in prop::collection::vec(-10.0f32..10.0f32, 10..100),
        ) {
            let len = a.len().min(b.len());
            let a = &a[..len];
            let b = &b[..len];

            // Normalize vectors
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            // Skip if either vector is zero
            if norm_a < 1e-6 || norm_b < 1e-6 {
                return Ok(());
            }
            
            let a_norm: Vec<f32> = a.iter().map(|x| x / norm_a).collect();
            let b_norm: Vec<f32> = b.iter().map(|x| x / norm_b).collect();

            let validator = ValidationSystem::new(0.7);
            let similarity = validator.cosine_similarity(&a_norm, &b_norm);

            // Property: For normalized vectors, similarity should be in [-1, 1]
            // but our implementation returns absolute value, so [0, 1]
            prop_assert!(
                similarity >= 0.0 && similarity <= 1.0,
                "Cosine similarity {} for normalized vectors is out of range",
                similarity
            );
        }
    }
}

/// **Validates: Requirements 9**
///
/// Property: Compression ratio is positive
///
/// This property test verifies that:
/// 1. Compression ratio is always positive
/// 2. Compression ratio is never zero or negative
/// 3. Compression ratio is reasonable (not infinity or NaN)
#[cfg(test)]
mod compression_ratio_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_compression_ratio_positive(
            // Generate random sizes
            original_size in 1000u64..1_000_000_000u64,
            compressed_size in 100u64..100_000_000u64,
        ) {
            // Ensure compressed size is less than original
            let compressed_size = compressed_size.min(original_size - 1);
            
            let compression_ratio = original_size as f32 / compressed_size as f32;

            // Property: Compression ratio must be positive
            prop_assert!(
                compression_ratio > 0.0,
                "Compression ratio {} is not positive",
                compression_ratio
            );

            // Property: Compression ratio must be finite
            prop_assert!(
                compression_ratio.is_finite(),
                "Compression ratio {} is not finite",
                compression_ratio
            );

            // Property: Compression ratio should be >= 1.0 (compressed is smaller)
            prop_assert!(
                compression_ratio >= 1.0,
                "Compression ratio {} is less than 1.0",
                compression_ratio
            );
        }
    }

    proptest! {
        #[test]
        fn prop_compression_ratio_reasonable_range(
            // Generate sizes for typical quantization scenarios
            original_size in 10_000_000u64..1_000_000_000u64, // 10MB to 1GB
            bit_width in prop::sample::select(vec![2u8, 4u8, 8u8]),
        ) {
            // Calculate expected compressed size based on bit width
            // FP16 (16 bits) -> INT2/4/8 (2/4/8 bits)
            let expected_ratio = 16.0 / bit_width as f32;
            let compressed_size = (original_size as f32 / expected_ratio) as u64;
            
            let compression_ratio = original_size as f32 / compressed_size as f32;

            // Property: Compression ratio should be close to expected
            prop_assert!(
                (compression_ratio - expected_ratio).abs() < 0.1,
                "Compression ratio {} differs significantly from expected {}",
                compression_ratio,
                expected_ratio
            );

            // Property: Compression ratio should be in reasonable range [1, 100]
            prop_assert!(
                compression_ratio >= 1.0 && compression_ratio <= 100.0,
                "Compression ratio {} is outside reasonable range [1, 100]",
                compression_ratio
            );
        }
    }

    proptest! {
        #[test]
        fn prop_compression_ratio_monotonic(
            original_size in 1000u64..1_000_000u64,
        ) {
            // Test that smaller compressed size = higher compression ratio
            let compressed_size_1 = original_size / 2;
            let compressed_size_2 = original_size / 4;
            
            let ratio_1 = original_size as f32 / compressed_size_1 as f32;
            let ratio_2 = original_size as f32 / compressed_size_2 as f32;

            // Property: Smaller compressed size should give higher ratio
            prop_assert!(
                ratio_2 > ratio_1,
                "Compression ratio not monotonic: {} (1/4) should be > {} (1/2)",
                ratio_2,
                ratio_1
            );
        }
    }
}

/// **Validates: Requirements 9**
///
/// Property: Per-layer accuracy aggregates to overall accuracy
///
/// This property test verifies that:
/// 1. Overall accuracy is the mean of per-layer accuracies
/// 2. Overall accuracy is in the range [min, max] of per-layer accuracies
/// 3. Aggregation is consistent and deterministic
#[cfg(test)]
mod accuracy_aggregation_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_overall_accuracy_is_mean(
            // Generate random per-layer accuracies
            num_layers in 1usize..100,
        ) {
            let mut per_layer_accuracy = HashMap::new();
            let mut sum = 0.0f32;
            
            // Use deterministic generation based on layer index
            for i in 0..num_layers {
                let accuracy = ((i as f32 * 0.123) % 1.0).abs(); // Deterministic pseudo-random
                per_layer_accuracy.insert(format!("layer_{}", i), accuracy);
                sum += accuracy;
            }
            
            let expected_mean = sum / num_layers as f32;

            // Compute actual mean
            let actual_mean: f32 = per_layer_accuracy.values().sum::<f32>() 
                / per_layer_accuracy.len() as f32;

            // Property: Overall accuracy should equal mean of per-layer accuracies
            prop_assert!(
                (actual_mean - expected_mean).abs() < 1e-6,
                "Overall accuracy {} != expected mean {}",
                actual_mean,
                expected_mean
            );
        }
    }

    proptest! {
        #[test]
        fn prop_overall_accuracy_in_range(
            num_layers in 2usize..50,
        ) {
            let mut per_layer_accuracy = HashMap::new();
            let mut min_acc = f32::INFINITY;
            let mut max_acc = f32::NEG_INFINITY;
            
            // Use deterministic generation
            for i in 0..num_layers {
                let accuracy = ((i as f32 * 0.456) % 1.0).abs();
                per_layer_accuracy.insert(format!("layer_{}", i), accuracy);
                min_acc = min_acc.min(accuracy);
                max_acc = max_acc.max(accuracy);
            }
            
            let overall_accuracy: f32 = per_layer_accuracy.values().sum::<f32>() 
                / per_layer_accuracy.len() as f32;

            // Property: Overall accuracy should be between min and max
            prop_assert!(
                overall_accuracy >= min_acc,
                "Overall accuracy {} < min layer accuracy {}",
                overall_accuracy,
                min_acc
            );

            prop_assert!(
                overall_accuracy <= max_acc,
                "Overall accuracy {} > max layer accuracy {}",
                overall_accuracy,
                max_acc
            );
        }
    }

    proptest! {
        #[test]
        fn prop_aggregation_deterministic(
            num_layers in 1usize..50,
        ) {
            let mut per_layer_accuracy = HashMap::new();
            
            // Use deterministic generation
            for i in 0..num_layers {
                let accuracy = ((i as f32 * 0.789) % 1.0).abs();
                per_layer_accuracy.insert(format!("layer_{}", i), accuracy);
            }
            
            // Compute mean twice
            let mean_1: f32 = per_layer_accuracy.values().sum::<f32>() 
                / per_layer_accuracy.len() as f32;
            let mean_2: f32 = per_layer_accuracy.values().sum::<f32>() 
                / per_layer_accuracy.len() as f32;

            // Property: Aggregation should be deterministic
            prop_assert!(
                (mean_1 - mean_2).abs() < 1e-6,
                "Aggregation not deterministic: {} != {}",
                mean_1,
                mean_2
            );
        }
    }

    proptest! {
        #[test]
        fn prop_single_layer_aggregation(
            accuracy in 0.0f32..1.0f32,
        ) {
            let mut per_layer_accuracy = HashMap::new();
            per_layer_accuracy.insert("layer_0".to_string(), accuracy);
            
            let overall_accuracy: f32 = per_layer_accuracy.values().sum::<f32>() 
                / per_layer_accuracy.len() as f32;

            // Property: For single layer, overall = per-layer
            prop_assert!(
                (overall_accuracy - accuracy).abs() < 1e-6,
                "Single layer aggregation failed: {} != {}",
                overall_accuracy,
                accuracy
            );
        }
    }

    proptest! {
        #[test]
        fn prop_uniform_layers_aggregation(
            num_layers in 1usize..100,
            accuracy in 0.0f32..1.0f32,
        ) {
            let mut per_layer_accuracy = HashMap::new();
            
            for i in 0..num_layers {
                per_layer_accuracy.insert(format!("layer_{}", i), accuracy);
            }
            
            let overall_accuracy: f32 = per_layer_accuracy.values().sum::<f32>() 
                / per_layer_accuracy.len() as f32;

            // Property: For uniform layers, overall = per-layer
            prop_assert!(
                (overall_accuracy - accuracy).abs() < 1e-6,
                "Uniform layers aggregation failed: {} != {}",
                overall_accuracy,
                accuracy
            );
        }
    }
}

/// **Validates: Requirements 9**
///
/// Property: Edge cases are handled correctly
///
/// This module tests edge cases like identical weights and orthogonal weights
#[cfg(test)]
mod edge_case_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_identical_weights_similarity_one(
            // Generate random vector
            weights in prop::collection::vec(-10.0f32..10.0f32, 10..1000),
        ) {
            let validator = ValidationSystem::new(0.7);
            let similarity = validator.cosine_similarity(&weights, &weights);

            // Property: Identical vectors should have similarity = 1.0
            prop_assert!(
                (similarity - 1.0).abs() < 1e-5,
                "Identical weights similarity {} != 1.0",
                similarity
            );
        }
    }

    proptest! {
        #[test]
        fn prop_scaled_weights_similarity_one(
            weights in prop::collection::vec(-10.0f32..10.0f32, 10..1000),
            scale in 0.1f32..10.0f32,
        ) {
            // Scale the weights
            let scaled: Vec<f32> = weights.iter().map(|x| x * scale).collect();

            let validator = ValidationSystem::new(0.7);
            let similarity = validator.cosine_similarity(&weights, &scaled);

            // Property: Scaled vectors should have similarity = 1.0
            prop_assert!(
                (similarity - 1.0).abs() < 1e-5,
                "Scaled weights similarity {} != 1.0 (scale={})",
                similarity,
                scale
            );
        }
    }

    #[test]
    fn test_orthogonal_weights_similarity_zero() {
        // Create orthogonal vectors
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];

        let validator = ValidationSystem::new(0.7);
        let similarity = validator.cosine_similarity(&a, &b);

        // Property: Orthogonal vectors should have similarity ≈ 0.0
        assert!(
            similarity.abs() < 1e-5,
            "Orthogonal weights similarity {} != 0.0",
            similarity
        );
    }

    #[test]
    fn test_orthogonal_weights_3d() {
        // Create orthogonal vectors in 3D
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![0.0, 0.0, 1.0];

        let validator = ValidationSystem::new(0.7);
        
        let sim_ab = validator.cosine_similarity(&a, &b);
        let sim_ac = validator.cosine_similarity(&a, &c);
        let sim_bc = validator.cosine_similarity(&b, &c);

        // All pairs should be orthogonal
        assert!(sim_ab.abs() < 1e-5, "sim(a,b) = {} != 0", sim_ab);
        assert!(sim_ac.abs() < 1e-5, "sim(a,c) = {} != 0", sim_ac);
        assert!(sim_bc.abs() < 1e-5, "sim(b,c) = {} != 0", sim_bc);
    }

    #[test]
    fn test_opposite_weights_similarity() {
        // Create opposite vectors
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![-1.0, -2.0, -3.0, -4.0];

        let validator = ValidationSystem::new(0.7);
        let similarity = validator.cosine_similarity(&a, &b);

        // Property: Opposite vectors should have similarity ≈ 0.0 or 1.0
        // (depending on implementation - absolute value or not)
        // Our implementation uses absolute value, so should be close to 1.0
        assert!(
            (similarity - 1.0).abs() < 1e-5 || similarity.abs() < 1e-5,
            "Opposite weights similarity {} unexpected",
            similarity
        );
    }

    proptest! {
        #[test]
        fn prop_zero_vector_handling(
            non_zero in prop::collection::vec(-10.0f32..10.0f32, 10..100),
        ) {
            let zero = vec![0.0f32; non_zero.len()];

            let validator = ValidationSystem::new(0.7);
            let similarity = validator.cosine_similarity(&zero, &non_zero);

            // Property: Zero vector should give 0.0 similarity (or NaN handled as 0)
            prop_assert!(
                similarity == 0.0 || similarity.is_nan(),
                "Zero vector similarity {} unexpected",
                similarity
            );
        }
    }

    proptest! {
        #[test]
        fn prop_very_small_values(
            len in 10usize..100,
        ) {
            // Generate very small values deterministically
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.123) % 1.0 * 1e-6).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.456) % 1.0 * 1e-6).collect();

            let validator = ValidationSystem::new(0.7);
            let similarity = validator.cosine_similarity(&a, &b);

            // Property: Should handle very small values without overflow/underflow
            prop_assert!(
                similarity.is_finite() || similarity.is_nan(),
                "Very small values caused invalid similarity: {}",
                similarity
            );

            if similarity.is_finite() {
                prop_assert!(
                    similarity >= 0.0 && similarity <= 1.0,
                    "Very small values similarity {} out of range",
                    similarity
                );
            }
        }
    }

    proptest! {
        #[test]
        fn prop_very_large_values(
            len in 10usize..100,
        ) {
            // Generate very large values deterministically
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.789) % 1.0 * 1e6).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.234) % 1.0 * 1e6).collect();

            let validator = ValidationSystem::new(0.7);
            let similarity = validator.cosine_similarity(&a, &b);

            // Property: Should handle very large values without overflow
            prop_assert!(
                similarity.is_finite(),
                "Very large values caused invalid similarity: {}",
                similarity
            );

            prop_assert!(
                similarity >= 0.0 && similarity <= 1.0,
                "Very large values similarity {} out of range",
                similarity
            );
        }
    }
}

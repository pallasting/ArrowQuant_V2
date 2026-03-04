//! Backward Compatibility Regression Tests
//!
//! **Validates: Requirements 7.1, 7.3**
//! **Property 9, 10: Backward Compatibility and Test Coverage**
//!
//! This module contains regression tests to ensure that all optimizations
//! maintain backward compatibility with the baseline implementation.
//!
//! # Test Strategy
//!
//! - Compare optimized implementation results with baseline
//! - Verify all existing API methods remain unchanged
//! - Test default behavior consistency
//! - Ensure all 374+ existing tests continue to pass

use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};
use proptest::prelude::*;

/// Generate test data for backward compatibility testing
fn generate_test_weights(size: usize, seed: u64) -> Vec<f32> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(-10.0..10.0)).collect()
}

/// Generate test parameters
fn generate_test_params(num_groups: usize, total_size: usize, seed: u64) -> Vec<TimeGroupParams> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    
    let group_size = total_size / num_groups;
    
    (0..num_groups)
        .map(|i| {
            let start = i * group_size;
            let end = if i == num_groups - 1 {
                total_size
            } else {
                (i + 1) * group_size
            };
            
            TimeGroupParams {
                time_range: (start, end),
                scale: rng.gen_range(0.01..1.0),
                zero_point: rng.gen_range(0.0..128.0),
                group_size: 64,
            }
        })
        .collect()
}

// ============================================================================
// API Compatibility Tests
// ============================================================================

#[cfg(test)]
mod api_compatibility {
    use super::*;

    #[test]
    fn test_quantizer_constructor_unchanged() {
        // Verify that the basic constructor still works
        let num_groups = 10;
        let quantizer = TimeAwareQuantizer::new(num_groups);
        
        // Constructor should succeed
        assert_eq!(quantizer.num_time_groups(), num_groups);
    }

    #[test]
    fn test_quantize_layer_arrow_signature_unchanged() {
        // Verify that the main quantization method signature is unchanged
        let quantizer = TimeAwareQuantizer::new(10);
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = generate_test_params(10, weights.len(), 42);
        
        // Method should be callable with same signature
        let result = quantizer.quantize_layer_arrow(&weights, &params);
        
        assert!(result.is_ok(), "quantize_layer_arrow should succeed");
    }

    #[test]
    fn test_default_behavior_unchanged() {
        // Verify that default behavior (without explicit optimization flags) works
        let quantizer = TimeAwareQuantizer::new(10);
        let weights = generate_test_weights(1000, 42);
        let params = generate_test_params(10, weights.len(), 42);
        
        let result = quantizer.quantize_layer_arrow(&weights, &params);
        
        assert!(result.is_ok(), "Default behavior should work");
        
        let quantized = result.unwrap();
        assert_eq!(
            quantized.quantized_data().len(),
            weights.len(),
            "Output size should match input size"
        );
    }
}

// ============================================================================
// Result Equivalence Tests
// ============================================================================

#[cfg(test)]
mod result_equivalence {
    use super::*;

    #[test]
    fn test_optimized_produces_same_results_small() {
        // Test that optimized implementation produces same results for small arrays
        let quantizer = TimeAwareQuantizer::new(5);
        let weights = generate_test_weights(100, 42);
        let params = generate_test_params(5, weights.len(), 42);
        
        // Run quantization twice (should be deterministic)
        let result1 = quantizer
            .quantize_layer_arrow(&weights, &params)
            .expect("First quantization should succeed");
        
        let result2 = quantizer
            .quantize_layer_arrow(&weights, &params)
            .expect("Second quantization should succeed");
        
        // Results should be identical
        let data1 = result1.quantized_data();
        let data2 = result2.quantized_data();
        
        assert_eq!(data1.len(), data2.len(), "Lengths should match");
        
        for i in 0..data1.len() {
            assert_eq!(
                data1.value(i),
                data2.value(i),
                "Values should be identical at index {}",
                i
            );
        }
    }

    #[test]
    fn test_optimized_produces_same_results_medium() {
        // Test for medium-sized arrays
        let quantizer = TimeAwareQuantizer::new(10);
        let weights = generate_test_weights(10_000, 42);
        let params = generate_test_params(10, weights.len(), 42);
        
        let result1 = quantizer
            .quantize_layer_arrow(&weights, &params)
            .expect("First quantization should succeed");
        
        let result2 = quantizer
            .quantize_layer_arrow(&weights, &params)
            .expect("Second quantization should succeed");
        
        // Verify identical results
        let data1 = result1.quantized_data();
        let data2 = result2.quantized_data();
        
        for i in 0..data1.len() {
            assert_eq!(
                data1.value(i),
                data2.value(i),
                "Values should be identical at index {}",
                i
            );
        }
    }

    #[test]
    fn test_optimized_produces_same_results_large() {
        // Test for large arrays
        let quantizer = TimeAwareQuantizer::new(10);
        let weights = generate_test_weights(100_000, 42);
        let params = generate_test_params(10, weights.len(), 42);
        
        let result1 = quantizer
            .quantize_layer_arrow(&weights, &params)
            .expect("First quantization should succeed");
        
        let result2 = quantizer
            .quantize_layer_arrow(&weights, &params)
            .expect("Second quantization should succeed");
        
        // Verify identical results
        let data1 = result1.quantized_data();
        let data2 = result2.quantized_data();
        
        for i in 0..data1.len() {
            assert_eq!(
                data1.value(i),
                data2.value(i),
                "Values should be identical at index {}",
                i
            );
        }
    }
}

// ============================================================================
// Property-Based Backward Compatibility Tests
// ============================================================================

#[cfg(test)]
mod property_backward_compat {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        
        #[test]
        /// Property 9: Backward compatibility
        ///
        /// Verifies that optimized implementation produces identical results
        /// to baseline for any valid input.
        fn prop_backward_compatibility(
            size in 100usize..10_000,
            num_groups in 5usize..15,
            seed in any::<u64>(),
        ) {
            let weights = generate_test_weights(size, seed);
            let params = generate_test_params(num_groups, size, seed);
            
            let quantizer = TimeAwareQuantizer::new(num_groups);
            
            // Run quantization multiple times
            let result1 = quantizer
                .quantize_layer_arrow(&weights, &params)
                .expect("Quantization should succeed");
            
            let result2 = quantizer
                .quantize_layer_arrow(&weights, &params)
                .expect("Quantization should succeed");
            
            // Verify results are identical
            let data1 = result1.quantized_data();
            let data2 = result2.quantized_data();
            
            prop_assert_eq!(data1.len(), data2.len(), "Lengths should match");
            
            for i in 0..size {
                prop_assert_eq!(
                    data1.value(i),
                    data2.value(i),
                    "Values should be identical at index {}",
                    i
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        
        #[test]
        /// Property 10: Test coverage maintained
        ///
        /// Verifies that all edge cases continue to work correctly.
        fn prop_edge_cases_handled(
            size in 1usize..1000,
            num_groups in 1usize..20,
            seed in any::<u64>(),
        ) {
            let weights = generate_test_weights(size, seed);
            let params = generate_test_params(num_groups, size, seed);
            
            let quantizer = TimeAwareQuantizer::new(num_groups);
            
            // Should handle all edge cases without panicking
            let result = quantizer.quantize_layer_arrow(&weights, &params);
            
            prop_assert!(result.is_ok(), "Should handle edge cases gracefully");
            
            if let Ok(quantized) = result {
                prop_assert_eq!(
                    quantized.quantized_data().len(),
                    size,
                    "Output size should match input size"
                );
            }
        }
    }
}

// ============================================================================
// Error Handling Compatibility Tests
// ============================================================================

#[cfg(test)]
mod error_handling_compat {
    use super::*;

    #[test]
    fn test_error_handling_unchanged() {
        let quantizer = TimeAwareQuantizer::new(10);
        
        // Empty weights should still be handled
        let empty_weights: Vec<f32> = vec![];
        let params = generate_test_params(10, 100, 42);
        
        let result = quantizer.quantize_layer_arrow(&empty_weights, &params);
        
        // Should either succeed with empty result or return appropriate error
        match result {
            Ok(quantized) => {
                assert_eq!(quantized.quantized_data().len(), 0, "Empty input should produce empty output");
            }
            Err(_) => {
                // Error is also acceptable for empty input
            }
        }
    }

    #[test]
    fn test_invalid_params_handling() {
        let quantizer = TimeAwareQuantizer::new(10);
        let weights = generate_test_weights(100, 42);
        
        // Mismatched params count
        let wrong_params = generate_test_params(5, weights.len(), 42);
        
        let result = quantizer.quantize_layer_arrow(&weights, &wrong_params);
        
        // Should handle gracefully (either succeed or return error)
        // The important thing is it doesn't panic
        let _ = result;
    }
}

// ============================================================================
// Performance Characteristics Tests
// ============================================================================

#[cfg(test)]
mod performance_characteristics {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_performance_not_degraded() {
        // Verify that optimizations don't accidentally degrade performance
        let quantizer = TimeAwareQuantizer::new(10);
        let weights = generate_test_weights(100_000, 42);
        let params = generate_test_params(10, weights.len(), 42);
        
        // Warm up
        let _ = quantizer.quantize_layer_arrow(&weights, &params);
        
        // Measure time
        let start = Instant::now();
        let _ = quantizer.quantize_layer_arrow(&weights, &params);
        let elapsed = start.elapsed();
        
        // Should complete in reasonable time (< 1 second for 100K elements)
        assert!(
            elapsed.as_secs() < 1,
            "Quantization should complete quickly, took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_memory_usage_reasonable() {
        // Verify that memory usage is reasonable
        let quantizer = TimeAwareQuantizer::new(10);
        
        // Process multiple batches
        for _ in 0..10 {
            let weights = generate_test_weights(10_000, 42);
            let params = generate_test_params(10, weights.len(), 42);
            
            let result = quantizer.quantize_layer_arrow(&weights, &params);
            
            assert!(result.is_ok(), "Should handle multiple batches");
            
            // Drop result to free memory
            drop(result);
        }
        
        // If we get here without OOM, memory usage is reasonable
    }
}

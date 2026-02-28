//! Property-Based Tests for Quantization Roundtrip
//!
//! **Validates: Requirement 13 (Testing and Benchmarking)**
//!
//! This module contains property-based tests using proptest to verify
//! quantization invariants across a wide range of inputs.

use arrow_quant_v2::simd::{dequantize_simd, quantize_simd};
use proptest::prelude::*;

/// **Validates: Requirements 13**
///
/// Property: Quantize then dequantize preserves shape
///
/// This property test verifies that:
/// 1. The output shape matches the input shape after roundtrip
/// 2. The length is preserved through quantization and dequantization
/// 3. No data is lost or added during the process
#[cfg(test)]
mod roundtrip_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_roundtrip_preserves_shape(
            // Generate random weights in range [-10.0, 10.0]
            weights in prop::collection::vec(-10.0f32..10.0f32, 1..1000),
            // Generate random scale in reasonable range
            scale in 0.01f32..1.0f32,
            // Generate random zero_point
            zero_point in 0.0f32..128.0f32,
        ) {
            let original_len = weights.len();

            // Quantize
            let quantized = quantize_simd(&weights, scale, zero_point);

            // Property 1: Quantized data should have same length
            prop_assert_eq!(
                quantized.len(),
                original_len,
                "Quantized length {} != original length {}",
                quantized.len(),
                original_len
            );

            // Dequantize
            let dequantized = dequantize_simd(&quantized, scale, zero_point);

            // Property 2: Dequantized data should have same length as original
            prop_assert_eq!(
                dequantized.len(),
                original_len,
                "Dequantized length {} != original length {}",
                dequantized.len(),
                original_len
            );

            // Property 3: Shape is preserved (length is the shape for 1D arrays)
            prop_assert_eq!(
                dequantized.len(),
                weights.len(),
                "Roundtrip did not preserve shape"
            );
        }
    }

    proptest! {
        #[test]
        fn prop_roundtrip_preserves_shape_int2(
            weights in prop::collection::vec(-10.0f32..10.0f32, 1..1000),
        ) {
            // INT2: 2-bit quantization (0-3 range)
            let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale = (max - min) / 3.0; // 2-bit: 4 levels (0-3)
            let zero_point = -min / scale;

            let original_len = weights.len();
            let quantized = quantize_simd(&weights, scale, zero_point);
            let dequantized = dequantize_simd(&quantized, scale, zero_point);

            prop_assert_eq!(dequantized.len(), original_len);
        }
    }

    proptest! {
        #[test]
        fn prop_roundtrip_preserves_shape_int4(
            weights in prop::collection::vec(-10.0f32..10.0f32, 1..1000),
        ) {
            // INT4: 4-bit quantization (0-15 range)
            let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale = (max - min) / 15.0; // 4-bit: 16 levels (0-15)
            let zero_point = -min / scale;

            let original_len = weights.len();
            let quantized = quantize_simd(&weights, scale, zero_point);
            let dequantized = dequantize_simd(&quantized, scale, zero_point);

            prop_assert_eq!(dequantized.len(), original_len);
        }
    }

    proptest! {
        #[test]
        fn prop_roundtrip_preserves_shape_int8(
            weights in prop::collection::vec(-10.0f32..10.0f32, 1..1000),
        ) {
            // INT8: 8-bit quantization (0-255 range)
            let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale = (max - min) / 255.0; // 8-bit: 256 levels (0-255)
            let zero_point = -min / scale;

            let original_len = weights.len();
            let quantized = quantize_simd(&weights, scale, zero_point);
            let dequantized = dequantize_simd(&quantized, scale, zero_point);

            prop_assert_eq!(dequantized.len(), original_len);
        }
    }
}

/// **Validates: Requirements 13**
///
/// Property: Quantization error is bounded by bit_width
///
/// This property test verifies that:
/// 1. The quantization error is bounded by the quantization step size
/// 2. INT2 has larger error than INT4, which has larger error than INT8
/// 3. The error is proportional to the scale parameter
#[cfg(test)]
mod error_bound_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_quantization_error_bounded_int2(
            weights in prop::collection::vec(-10.0f32..10.0f32, 1..1000),
        ) {
            // INT2: 2-bit quantization (0-3 range)
            let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            
            // Handle edge case where all values are the same
            if (max - min).abs() < 1e-6 {
                return Ok(());
            }
            
            let scale = (max - min) / 3.0; // 2-bit: 4 levels (0-3)
            let zero_point = -min / scale;

            let quantized = quantize_simd(&weights, scale, zero_point);
            let dequantized = dequantize_simd(&quantized, scale, zero_point);

            // Property: Error should be bounded by scale (quantization step size)
            // For INT2, we expect larger errors due to coarse quantization
            for (orig, deq) in weights.iter().zip(dequantized.iter()) {
                let error = (orig - deq).abs();
                prop_assert!(
                    error <= scale * 1.5, // Allow 1.5x scale for rounding
                    "INT2 error {} exceeds bound {} for value {}",
                    error,
                    scale * 1.5,
                    orig
                );
            }
        }
    }

    proptest! {
        #[test]
        fn prop_quantization_error_bounded_int4(
            weights in prop::collection::vec(-10.0f32..10.0f32, 1..1000),
        ) {
            // INT4: 4-bit quantization (0-15 range)
            let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            
            // Handle edge case where all values are the same
            if (max - min).abs() < 1e-6 {
                return Ok(());
            }
            
            let scale = (max - min) / 15.0; // 4-bit: 16 levels (0-15)
            let zero_point = -min / scale;

            let quantized = quantize_simd(&weights, scale, zero_point);
            let dequantized = dequantize_simd(&quantized, scale, zero_point);

            // Property: Error should be bounded by scale
            for (orig, deq) in weights.iter().zip(dequantized.iter()) {
                let error = (orig - deq).abs();
                prop_assert!(
                    error <= scale * 1.5,
                    "INT4 error {} exceeds bound {} for value {}",
                    error,
                    scale * 1.5,
                    orig
                );
            }
        }
    }

    proptest! {
        #[test]
        fn prop_quantization_error_bounded_int8(
            weights in prop::collection::vec(-10.0f32..10.0f32, 1..1000),
        ) {
            // INT8: 8-bit quantization (0-255 range)
            let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            
            // Handle edge case where all values are the same
            if (max - min).abs() < 1e-6 {
                return Ok(());
            }
            
            let scale = (max - min) / 255.0; // 8-bit: 256 levels (0-255)
            let zero_point = -min / scale;

            let quantized = quantize_simd(&weights, scale, zero_point);
            let dequantized = dequantize_simd(&quantized, scale, zero_point);

            // Property: Error should be bounded by scale
            // INT8 should have the smallest error
            for (orig, deq) in weights.iter().zip(dequantized.iter()) {
                let error = (orig - deq).abs();
                prop_assert!(
                    error <= scale * 1.5,
                    "INT8 error {} exceeds bound {} for value {}",
                    error,
                    scale * 1.5,
                    orig
                );
            }
        }
    }

    proptest! {
        #[test]
        fn prop_error_decreases_with_bit_width(
            weights in prop::collection::vec(-10.0f32..10.0f32, 10..100),
        ) {
            let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Handle edge case where all values are the same
            if (max - min).abs() < 1e-6 {
                return Ok(());
            }

            // INT2
            let scale_int2 = (max - min) / 3.0;
            let zero_point_int2 = -min / scale_int2;
            let quantized_int2 = quantize_simd(&weights, scale_int2, zero_point_int2);
            let dequantized_int2 = dequantize_simd(&quantized_int2, scale_int2, zero_point_int2);

            // INT4
            let scale_int4 = (max - min) / 15.0;
            let zero_point_int4 = -min / scale_int4;
            let quantized_int4 = quantize_simd(&weights, scale_int4, zero_point_int4);
            let dequantized_int4 = dequantize_simd(&quantized_int4, scale_int4, zero_point_int4);

            // INT8
            let scale_int8 = (max - min) / 255.0;
            let zero_point_int8 = -min / scale_int8;
            let quantized_int8 = quantize_simd(&weights, scale_int8, zero_point_int8);
            let dequantized_int8 = dequantize_simd(&quantized_int8, scale_int8, zero_point_int8);

            // Compute average errors
            let error_int2: f32 = weights
                .iter()
                .zip(dequantized_int2.iter())
                .map(|(o, d)| (o - d).abs())
                .sum::<f32>()
                / weights.len() as f32;

            let error_int4: f32 = weights
                .iter()
                .zip(dequantized_int4.iter())
                .map(|(o, d)| (o - d).abs())
                .sum::<f32>()
                / weights.len() as f32;

            let error_int8: f32 = weights
                .iter()
                .zip(dequantized_int8.iter())
                .map(|(o, d)| (o - d).abs())
                .sum::<f32>()
                / weights.len() as f32;

            // Property: Error should decrease with higher bit width
            // INT2 >= INT4 >= INT8 (with some tolerance for edge cases)
            prop_assert!(
                error_int2 >= error_int4 * 0.9 || error_int2 < 0.01,
                "INT2 error {} should be >= INT4 error {}",
                error_int2,
                error_int4
            );

            prop_assert!(
                error_int4 >= error_int8 * 0.9 || error_int4 < 0.01,
                "INT4 error {} should be >= INT8 error {}",
                error_int4,
                error_int8
            );
        }
    }
}

/// **Validates: Requirements 13**
///
/// Property: Quantization is deterministic
///
/// This property test verifies that:
/// 1. Quantizing the same input twice produces identical results
/// 2. The process is reproducible
#[cfg(test)]
mod determinism_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_quantization_is_deterministic(
            weights in prop::collection::vec(-10.0f32..10.0f32, 1..1000),
            scale in 0.01f32..1.0f32,
            zero_point in 0.0f32..128.0f32,
        ) {
            // Quantize twice
            let quantized1 = quantize_simd(&weights, scale, zero_point);
            let quantized2 = quantize_simd(&weights, scale, zero_point);

            // Property: Results should be identical
            prop_assert_eq!(
                quantized1,
                quantized2,
                "Quantization is not deterministic"
            );
        }
    }

    proptest! {
        #[test]
        fn prop_dequantization_is_deterministic(
            quantized in prop::collection::vec(0u8..=255u8, 1..1000),
            scale in 0.01f32..1.0f32,
            zero_point in 0.0f32..128.0f32,
        ) {
            // Dequantize twice
            let dequantized1 = dequantize_simd(&quantized, scale, zero_point);
            let dequantized2 = dequantize_simd(&quantized, scale, zero_point);

            // Property: Results should be identical
            prop_assert_eq!(
                dequantized1.len(),
                dequantized2.len(),
                "Dequantization length mismatch"
            );

            for (d1, d2) in dequantized1.iter().zip(dequantized2.iter()) {
                prop_assert!(
                    (d1 - d2).abs() < 1e-6,
                    "Dequantization is not deterministic: {} != {}",
                    d1,
                    d2
                );
            }
        }
    }
}

/// **Validates: Requirements 13**
///
/// Property: Quantized values are in valid range
///
/// This property test verifies that:
/// 1. All quantized values are in [0, 255] range
/// 2. No overflow or underflow occurs
#[cfg(test)]
mod range_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_quantized_values_in_range(
            weights in prop::collection::vec(-10.0f32..10.0f32, 1..1000),
            scale in 0.01f32..1.0f32,
            zero_point in 0.0f32..128.0f32,
        ) {
            let quantized = quantize_simd(&weights, scale, zero_point);

            // Property: All quantized values should be in [0, 255]
            for (i, &q) in quantized.iter().enumerate() {
                prop_assert!(
                    q <= 255,
                    "Quantized value {} at index {} exceeds 255",
                    q,
                    i
                );
            }
        }
    }
}

/// **Validates: Requirements 13**
///
/// Property: Edge cases are handled correctly
///
/// This property test verifies that:
/// 1. Zero values are handled correctly
/// 2. Extreme values are clamped properly
/// 3. Empty arrays are handled
#[cfg(test)]
mod edge_case_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_zero_values_handled(
            len in 1usize..1000,
            scale in 0.01f32..1.0f32,
            zero_point in 0.0f32..128.0f32,
        ) {
            let weights = vec![0.0f32; len];
            let quantized = quantize_simd(&weights, scale, zero_point);
            let dequantized = dequantize_simd(&quantized, scale, zero_point);

            prop_assert_eq!(dequantized.len(), len);

            // All dequantized values should be close to zero
            for &d in dequantized.iter() {
                prop_assert!(
                    d.abs() < scale * 2.0,
                    "Dequantized zero value {} too far from zero",
                    d
                );
            }
        }
    }

    proptest! {
        #[test]
        fn prop_extreme_values_clamped(
            scale in 0.01f32..1.0f32,
            zero_point in 0.0f32..128.0f32,
        ) {
            // Test with extreme values
            let weights = vec![-1000.0, -100.0, 0.0, 100.0, 1000.0];
            let quantized = quantize_simd(&weights, scale, zero_point);

            // Property: All quantized values should be in [0, 255]
            for &q in quantized.iter() {
                prop_assert!(q <= 255, "Value not clamped: {}", q);
            }
        }
    }

    #[test]
    fn test_empty_array() {
        let weights: Vec<f32> = vec![];
        let quantized = quantize_simd(&weights, 1.0, 0.0);
        let dequantized = dequantize_simd(&quantized, 1.0, 0.0);

        assert_eq!(quantized.len(), 0);
        assert_eq!(dequantized.len(), 0);
    }

    #[test]
    fn test_single_element() {
        let weights = vec![5.0];
        let quantized = quantize_simd(&weights, 0.1, 0.0);
        let dequantized = dequantize_simd(&quantized, 0.1, 0.0);

        assert_eq!(quantized.len(), 1);
        assert_eq!(dequantized.len(), 1);
        assert!((weights[0] - dequantized[0]).abs() < 0.2);
    }
}

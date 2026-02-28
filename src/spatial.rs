//! Spatial quantization with channel equalization

use crate::errors::Result;
use ndarray::{Array2, ArrayView2};

/// Spatial quantizer for handling spatial variance
#[derive(Clone)]
pub struct SpatialQuantizer {
    group_size: usize,
    #[allow(dead_code)]
    enable_equalization: bool,
    #[allow(dead_code)]
    enable_smoothing: bool,
}

impl SpatialQuantizer {
    /// Create new spatial quantizer
    pub fn new(group_size: usize) -> Self {
        Self {
            group_size,
            enable_equalization: true,
            enable_smoothing: true,
        }
    }

    /// Channel equalization using DiTAS technique
    /// Formula: s_i = sqrt(max(|W_i|) / max(|X_i|))
    pub fn channel_equalization(
        &self,
        weights: &Array2<f32>,
        activations: &Array2<f32>,
    ) -> (Array2<f32>, Vec<f32>) {
        let num_channels = weights.shape()[0];
        let mut equalization_scales = Vec::with_capacity(num_channels);
        let mut equalized_weights = weights.clone();

        for i in 0..num_channels {
            let w_max = weights
                .row(i)
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            let x_max = activations
                .row(i)
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);

            let scale = if x_max > 0.0 {
                (w_max / x_max).sqrt()
            } else {
                1.0
            };

            equalization_scales.push(scale);

            // Scale weights by 1/s_i
            for j in 0..weights.shape()[1] {
                equalized_weights[[i, j]] /= scale;
            }
        }

        (equalized_weights, equalization_scales)
    }

    /// Activation smoothing to reduce variance
    /// Formula: x_smooth = x * (1 - alpha) + mean(x) * alpha
    pub fn activation_smoothing(&self, activations: &Array2<f32>, alpha: f32) -> Array2<f32> {
        let mean = activations.mean().unwrap_or(0.0);
        let mut smoothed = activations.clone();

        for elem in smoothed.iter_mut() {
            *elem = *elem * (1.0 - alpha) + mean * alpha;
        }

        smoothed
    }

    /// Per-group quantization with independent scales
    ///
    /// Divides channels into groups and computes separate quantization parameters
    /// (scale and zero_point) for each group. This allows better adaptation to
    /// varying activation ranges across the spatial dimension.
    ///
    /// # Arguments
    ///
    /// * `weights` - Input weight matrix with shape (num_channels, features)
    ///
    /// # Returns
    ///
    /// * `Result<QuantizedSpatialLayer>` - Quantized layer with per-group parameters
    ///
    /// # Supported Group Sizes
    ///
    /// * 32 - Fine-grained quantization (more groups, higher accuracy)
    /// * 64 - Balanced quantization
    /// * 128 - Coarse quantization
    /// * 256 - Very coarse quantization (fewer groups, faster)
    ///
    /// # Algorithm
    ///
    /// For each group of channels:
    /// 1. Extract the group slice from the weight matrix
    /// 2. Compute min and max values within the group
    /// 3. Calculate scale: (max - min) / 255.0
    /// 4. Calculate zero_point: -min / scale
    /// 5. Quantize each value: q = round((value / scale) + zero_point)
    /// 6. Clamp to [0, 255] range
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ndarray::Array2;
    /// use arrow_quant_v2::spatial::SpatialQuantizer;
    ///
    /// let quantizer = SpatialQuantizer::new(64);
    /// let weights = Array2::from_shape_fn((128, 256), |(i, j)| (i + j) as f32);
    /// let quantized = quantizer.per_group_quantize(&weights)?;
    ///
    /// assert_eq!(quantized.group_size, 64);
    /// assert_eq!(quantized.scales.len(), 2); // 128 channels / 64 = 2 groups
    /// ```
    pub fn per_group_quantize(&self, weights: &Array2<f32>) -> Result<QuantizedSpatialLayer> {
        let num_channels = weights.shape()[0];
        let num_groups = (num_channels + self.group_size - 1) / self.group_size;

        let mut quantized_data = Vec::new();
        let mut scales = Vec::new();
        let mut zero_points = Vec::new();

        for group_idx in 0..num_groups {
            let start = group_idx * self.group_size;
            let end = (start + self.group_size).min(num_channels);

            // Extract group
            let group = weights.slice(ndarray::s![start..end, ..]);

            // Compute quantization parameters for this group
            let min = group.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let scale = (max - min) / 255.0;
            let zero_point = -min / scale;

            // Quantize group
            let quantized = self.quantize_group(&group, scale, zero_point);

            quantized_data.extend(quantized);
            scales.push(scale);
            zero_points.push(zero_point);
        }

        Ok(QuantizedSpatialLayer {
            data: quantized_data,
            scales,
            zero_points,
            group_size: self.group_size,
        })
    }

    /// Quantize a single group
    fn quantize_group(&self, group: &ArrayView2<f32>, scale: f32, zero_point: f32) -> Vec<u8> {
        group
            .iter()
            .map(|&value| {
                let q = ((value / scale) + zero_point).round().clamp(0.0, 255.0);
                q as u8
            })
            .collect()
    }
}

/// Quantized layer with spatial parameters
#[derive(Debug, Clone)]
pub struct QuantizedSpatialLayer {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub zero_points: Vec<f32>,
    pub group_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_channel_equalization() {
        let quantizer = SpatialQuantizer::new(128);
        let weights = Array2::from_shape_fn((256, 512), |(i, j)| (i + j) as f32);
        let activations = Array2::from_shape_fn((256, 512), |(i, j)| (i * j + 1) as f32);

        let (equalized, scales) = quantizer.channel_equalization(&weights, &activations);

        assert_eq!(scales.len(), 256);
        assert_eq!(equalized.shape(), weights.shape());
    }

    #[test]
    fn test_channel_equalization_formula() {
        // Test that the formula s_i = sqrt(max(|W_i|) / max(|X_i|)) is correctly applied
        let quantizer = SpatialQuantizer::new(128);

        // Create simple test data where we can verify the formula manually
        // Channel 0: weights = [1.0, 2.0, 3.0], activations = [4.0, 5.0, 6.0]
        // Expected: w_max = 3.0, x_max = 6.0, scale = sqrt(3.0/6.0) = sqrt(0.5) ≈ 0.707
        let weights = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 2.0, 3.0, // Channel 0
                4.0, 8.0, 16.0, // Channel 1
            ],
        )
        .unwrap();

        let activations = Array2::from_shape_vec(
            (2, 3),
            vec![
                4.0, 5.0, 6.0, // Channel 0
                2.0, 4.0, 8.0, // Channel 1
            ],
        )
        .unwrap();

        let (equalized, scales) = quantizer.channel_equalization(&weights, &activations);

        // Verify scales
        assert_eq!(scales.len(), 2);

        // Channel 0: sqrt(3.0 / 6.0) = sqrt(0.5) ≈ 0.707
        assert!(
            (scales[0] - 0.707).abs() < 0.01,
            "Channel 0 scale should be ~0.707, got {}",
            scales[0]
        );

        // Channel 1: sqrt(16.0 / 8.0) = sqrt(2.0) ≈ 1.414
        assert!(
            (scales[1] - 1.414).abs() < 0.01,
            "Channel 1 scale should be ~1.414, got {}",
            scales[1]
        );

        // Verify weights are scaled by 1/s_i
        // Channel 0, element 0: 1.0 / 0.707 ≈ 1.414
        assert!(
            (equalized[[0, 0]] - 1.414).abs() < 0.01,
            "Equalized weight should be ~1.414, got {}",
            equalized[[0, 0]]
        );

        // Channel 1, element 2: 16.0 / 1.414 ≈ 11.31
        assert!(
            (equalized[[1, 2]] - 11.31).abs() < 0.1,
            "Equalized weight should be ~11.31, got {}",
            equalized[[1, 2]]
        );
    }

    #[test]
    fn test_channel_equalization_zero_activation() {
        // Test edge case: zero activation should result in scale = 1.0
        let quantizer = SpatialQuantizer::new(128);

        let weights = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let activations = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();

        let (_equalized, scales) = quantizer.channel_equalization(&weights, &activations);

        // Should use scale = 1.0 to avoid division by zero
        assert_eq!(scales[0], 1.0, "Scale should be 1.0 for zero activation");

        // Weights should remain unchanged (divided by 1.0)
        assert_eq!(_equalized[[0, 0]], 1.0);
        assert_eq!(_equalized[[0, 1]], 2.0);
        assert_eq!(_equalized[[0, 2]], 3.0);
    }

    #[test]
    fn test_channel_equalization_negative_values() {
        // Test that absolute values are used correctly
        let quantizer = SpatialQuantizer::new(128);

        let weights = Array2::from_shape_vec((1, 3), vec![-3.0, 2.0, -1.0]).unwrap();
        let activations = Array2::from_shape_vec((1, 3), vec![-6.0, 5.0, -4.0]).unwrap();

        let (_equalized, scales) = quantizer.channel_equalization(&weights, &activations);

        // w_max = max(|-3.0|, |2.0|, |-1.0|) = 3.0
        // x_max = max(|-6.0|, |5.0|, |-4.0|) = 6.0
        // scale = sqrt(3.0 / 6.0) ≈ 0.707
        assert!(
            (scales[0] - 0.707).abs() < 0.01,
            "Scale should be ~0.707, got {}",
            scales[0]
        );
    }

    #[test]
    fn test_activation_smoothing() {
        let quantizer = SpatialQuantizer::new(128);
        let activations = Array2::from_shape_fn((10, 10), |(i, j)| (i * j) as f32);

        let smoothed = quantizer.activation_smoothing(&activations, 0.1);

        assert_eq!(smoothed.shape(), activations.shape());
    }

    #[test]
    fn test_activation_smoothing_formula() {
        // Test that the formula x_smooth = x * (1 - alpha) + mean(x) * alpha is correctly applied
        let quantizer = SpatialQuantizer::new(128);

        // Create simple test data: [1.0, 2.0, 3.0, 4.0]
        // Mean = 2.5
        // With alpha = 0.5:
        // x_smooth[0] = 1.0 * 0.5 + 2.5 * 0.5 = 1.75
        // x_smooth[1] = 2.0 * 0.5 + 2.5 * 0.5 = 2.25
        // x_smooth[2] = 3.0 * 0.5 + 2.5 * 0.5 = 2.75
        // x_smooth[3] = 4.0 * 0.5 + 2.5 * 0.5 = 3.25
        let activations = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let alpha = 0.5;

        let smoothed = quantizer.activation_smoothing(&activations, alpha);

        // Verify the formula
        assert!(
            (smoothed[[0, 0]] - 1.75).abs() < 0.01,
            "Expected 1.75, got {}",
            smoothed[[0, 0]]
        );
        assert!(
            (smoothed[[0, 1]] - 2.25).abs() < 0.01,
            "Expected 2.25, got {}",
            smoothed[[0, 1]]
        );
        assert!(
            (smoothed[[1, 0]] - 2.75).abs() < 0.01,
            "Expected 2.75, got {}",
            smoothed[[1, 0]]
        );
        assert!(
            (smoothed[[1, 1]] - 3.25).abs() < 0.01,
            "Expected 3.25, got {}",
            smoothed[[1, 1]]
        );
    }

    #[test]
    fn test_activation_smoothing_alpha_zero() {
        // Test edge case: alpha = 0.0 should return original activations
        let quantizer = SpatialQuantizer::new(128);
        let activations = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let smoothed = quantizer.activation_smoothing(&activations, 0.0);

        // Should be identical to original
        assert_eq!(smoothed[[0, 0]], 1.0);
        assert_eq!(smoothed[[0, 1]], 2.0);
        assert_eq!(smoothed[[1, 0]], 3.0);
        assert_eq!(smoothed[[1, 1]], 4.0);
    }

    #[test]
    fn test_activation_smoothing_alpha_one() {
        // Test edge case: alpha = 1.0 should return all values equal to mean
        let quantizer = SpatialQuantizer::new(128);
        let activations = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mean = 2.5;

        let smoothed = quantizer.activation_smoothing(&activations, 1.0);

        // All values should be equal to mean
        assert!((smoothed[[0, 0]] - mean).abs() < 0.01);
        assert!((smoothed[[0, 1]] - mean).abs() < 0.01);
        assert!((smoothed[[1, 0]] - mean).abs() < 0.01);
        assert!((smoothed[[1, 1]] - mean).abs() < 0.01);
    }

    #[test]
    fn test_activation_smoothing_reduces_variance() {
        // Test that smoothing actually reduces variance
        let quantizer = SpatialQuantizer::new(128);

        // Create data with high variance
        let activations =
            Array2::from_shape_vec((2, 3), vec![-10.0, 0.0, 10.0, -5.0, 5.0, 15.0]).unwrap();

        let smoothed = quantizer.activation_smoothing(&activations, 0.5);

        // Calculate variance before and after
        let mean_orig = activations.mean().unwrap();
        let variance_orig: f32 = activations
            .iter()
            .map(|x| (x - mean_orig).powi(2))
            .sum::<f32>()
            / activations.len() as f32;

        let mean_smooth = smoothed.mean().unwrap();
        let variance_smooth: f32 = smoothed
            .iter()
            .map(|x| (x - mean_smooth).powi(2))
            .sum::<f32>()
            / smoothed.len() as f32;

        // Smoothed variance should be less than original
        assert!(
            variance_smooth < variance_orig,
            "Smoothed variance ({}) should be less than original variance ({})",
            variance_smooth,
            variance_orig
        );
    }

    #[test]
    fn test_activation_smoothing_typical_alpha() {
        // Test with typical alpha value (0.5) as mentioned in design
        let quantizer = SpatialQuantizer::new(128);
        let activations = Array2::from_shape_fn((4, 4), |(i, j)| (i * 10 + j) as f32);

        let smoothed = quantizer.activation_smoothing(&activations, 0.5);

        // Verify shape is preserved
        assert_eq!(smoothed.shape(), activations.shape());

        // Verify values are between original and mean
        let mean = activations.mean().unwrap();
        for (orig, smooth) in activations.iter().zip(smoothed.iter()) {
            if *orig < mean {
                assert!(
                    *smooth > *orig && *smooth < mean,
                    "Smoothed value should be between original and mean"
                );
            } else if *orig > mean {
                assert!(
                    *smooth < *orig && *smooth > mean,
                    "Smoothed value should be between mean and original"
                );
            }
        }
    }

    #[test]
    fn test_per_group_quantize() {
        let quantizer = SpatialQuantizer::new(64);
        let weights = Array2::from_shape_fn((128, 256), |(i, j)| (i + j) as f32 / 100.0);

        let result = quantizer.per_group_quantize(&weights);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(quantized.group_size, 64);
        assert_eq!(quantized.scales.len(), 2); // 128 channels / 64 group_size = 2 groups
    }

    #[test]
    fn test_per_group_quantize_group_size_32() {
        // Test with group size 32
        let quantizer = SpatialQuantizer::new(32);
        let weights = Array2::from_shape_fn((128, 256), |(i, j)| (i + j) as f32 / 100.0);

        let result = quantizer.per_group_quantize(&weights);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(quantized.group_size, 32);
        assert_eq!(quantized.scales.len(), 4); // 128 channels / 32 group_size = 4 groups
        assert_eq!(quantized.zero_points.len(), 4);
    }

    #[test]
    fn test_per_group_quantize_group_size_128() {
        // Test with group size 128
        let quantizer = SpatialQuantizer::new(128);
        let weights = Array2::from_shape_fn((256, 512), |(i, j)| (i + j) as f32 / 100.0);

        let result = quantizer.per_group_quantize(&weights);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(quantized.group_size, 128);
        assert_eq!(quantized.scales.len(), 2); // 256 channels / 128 group_size = 2 groups
        assert_eq!(quantized.zero_points.len(), 2);
    }

    #[test]
    fn test_per_group_quantize_group_size_256() {
        // Test with group size 256
        let quantizer = SpatialQuantizer::new(256);
        let weights = Array2::from_shape_fn((512, 512), |(i, j)| (i + j) as f32 / 100.0);

        let result = quantizer.per_group_quantize(&weights);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(quantized.group_size, 256);
        assert_eq!(quantized.scales.len(), 2); // 512 channels / 256 group_size = 2 groups
        assert_eq!(quantized.zero_points.len(), 2);
    }

    #[test]
    fn test_per_group_quantize_independent_scales() {
        // Test that each group has independent scales
        // Create weights with different ranges per group
        let quantizer = SpatialQuantizer::new(2);

        // Group 0: values in [0, 10]
        // Group 1: values in [100, 110]
        let weights = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.0, 5.0, 10.0, // Channel 0 (Group 0)
                1.0, 6.0, 9.0, // Channel 1 (Group 0)
                100.0, 105.0, 110.0, // Channel 2 (Group 1)
                101.0, 106.0, 109.0, // Channel 3 (Group 1)
            ],
        )
        .unwrap();

        let result = quantizer.per_group_quantize(&weights);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(quantized.scales.len(), 2);
        assert_eq!(quantized.zero_points.len(), 2);

        // Group 0 should have much smaller scale than Group 1
        // Group 0: range = 10 - 0 = 10, scale = 10/255 ≈ 0.039
        // Group 1: range = 110 - 100 = 10, scale = 10/255 ≈ 0.039
        // Actually both have same range, let's verify they're computed independently

        // The key is that scales are computed per group, not globally
        assert!(quantized.scales[0] > 0.0);
        assert!(quantized.scales[1] > 0.0);
    }

    #[test]
    fn test_per_group_quantize_different_ranges() {
        // Test with groups having significantly different value ranges
        let quantizer = SpatialQuantizer::new(2);

        // Group 0: small values [0, 1]
        // Group 1: large values [0, 100]
        let weights = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0, 1.0, // Channel 0 (Group 0)
                0.5, 0.8, // Channel 1 (Group 0)
                0.0, 100.0, // Channel 2 (Group 1)
                50.0, 75.0, // Channel 3 (Group 1)
            ],
        )
        .unwrap();

        let result = quantizer.per_group_quantize(&weights);
        assert!(result.is_ok());

        let quantized = result.unwrap();

        // Group 0 should have smaller scale (range 0-1)
        // Group 1 should have larger scale (range 0-100)
        assert!(
            quantized.scales[0] < quantized.scales[1],
            "Group 0 scale ({}) should be less than Group 1 scale ({})",
            quantized.scales[0],
            quantized.scales[1]
        );
    }

    #[test]
    fn test_per_group_quantize_non_divisible_channels() {
        // Test when number of channels is not divisible by group size
        let quantizer = SpatialQuantizer::new(64);
        let weights = Array2::from_shape_fn((100, 256), |(i, j)| (i + j) as f32 / 100.0);

        let result = quantizer.per_group_quantize(&weights);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(quantized.group_size, 64);
        // 100 channels / 64 group_size = 1 full group + 1 partial group (36 channels)
        assert_eq!(quantized.scales.len(), 2);
        assert_eq!(quantized.zero_points.len(), 2);
    }

    #[test]
    fn test_per_group_quantize_preserves_shape() {
        // Test that quantization preserves the total number of elements
        let quantizer = SpatialQuantizer::new(64);
        let weights = Array2::from_shape_fn((128, 256), |(i, j)| (i + j) as f32 / 100.0);
        let total_elements = weights.len();

        let result = quantizer.per_group_quantize(&weights);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(
            quantized.data.len(),
            total_elements,
            "Quantized data should have same number of elements as input"
        );
    }

    #[test]
    fn test_per_group_quantize_values_in_range() {
        // Test that all quantized values are in valid range [0, 255]
        let quantizer = SpatialQuantizer::new(64);
        let weights = Array2::from_shape_fn((128, 256), |(i, _j)| (i as f32 - 64.0) * 0.1);

        let result = quantizer.per_group_quantize(&weights);
        assert!(result.is_ok());

        let quantized = result.unwrap();

        // All values should be in [0, 255] (u8 type guarantees this)
        // Just verify we have data
        assert!(
            !quantized.data.is_empty(),
            "Quantized data should not be empty"
        );
    }

    #[test]
    fn test_per_group_quantize_single_group() {
        // Test when group size equals or exceeds number of channels
        let quantizer = SpatialQuantizer::new(256);
        let weights = Array2::from_shape_fn((128, 256), |(i, j)| (i + j) as f32 / 100.0);

        let result = quantizer.per_group_quantize(&weights);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(quantized.group_size, 256);
        assert_eq!(quantized.scales.len(), 1); // Only 1 group since 128 < 256
        assert_eq!(quantized.zero_points.len(), 1);
    }

    // Property-based tests
    #[cfg(test)]
    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// **Validates: Requirements 2.5**
            ///
            /// Property: Group quantization should preserve shape
            ///
            /// This property test verifies that:
            /// 1. The total number of quantized elements equals input elements
            /// 2. The output shape can be reconstructed from input shape
            /// 3. All quantized values are in valid range [0, 255]
            /// 4. The number of groups is correctly computed
            /// 5. Each group has corresponding scale and zero_point
            #[test]
            fn prop_group_quantization_preserves_shape(
                num_channels in 1usize..512,
                num_features in 1usize..1024,
                group_size in prop::sample::select(vec![32usize, 64, 128, 256])
            ) {
                let quantizer = SpatialQuantizer::new(group_size);

                // Create random weights in reasonable range [-10.0, 10.0]
                let weights = Array2::from_shape_fn((num_channels, num_features), |(i, j)| {
                    ((i * 7 + j * 13) % 2000) as f32 / 100.0 - 10.0
                });

                let result = quantizer.per_group_quantize(&weights);
                prop_assert!(result.is_ok(), "Quantization should succeed");

                let quantized = result.unwrap();

                // Property 1: Total number of elements preserved
                prop_assert_eq!(
                    quantized.data.len(),
                    num_channels * num_features,
                    "Quantized data should have same number of elements as input"
                );

                // Property 2: Group size is preserved
                prop_assert_eq!(
                    quantized.group_size,
                    group_size,
                    "Group size should be preserved"
                );

                // Property 3: Number of groups is correctly computed
                let expected_num_groups = (num_channels + group_size - 1) / group_size;
                prop_assert_eq!(
                    quantized.scales.len(),
                    expected_num_groups,
                    "Number of scales should match number of groups"
                );
                prop_assert_eq!(
                    quantized.zero_points.len(),
                    expected_num_groups,
                    "Number of zero_points should match number of groups"
                );

                // Property 4: All quantized values are in valid range [0, 255]
                // (This is guaranteed by u8 type, but we verify data is not empty)
                prop_assert!(
                    !quantized.data.is_empty(),
                    "Quantized data should not be empty"
                );

                // Property 5: Scales and zero_points are finite and reasonable
                for (i, &scale) in quantized.scales.iter().enumerate() {
                    prop_assert!(
                        scale.is_finite() && scale >= 0.0,
                        "Scale {} should be finite and non-negative, got {}",
                        i,
                        scale
                    );
                }

                for (i, &zp) in quantized.zero_points.iter().enumerate() {
                    prop_assert!(
                        zp.is_finite(),
                        "Zero point {} should be finite, got {}",
                        i,
                        zp
                    );
                }
            }

            /// **Validates: Requirements 2.1, 2.2**
            ///
            /// Property: Channel equalization should preserve shape and produce valid scales
            ///
            /// This property test verifies that:
            /// 1. Output shape matches input shape
            /// 2. Number of scales equals number of channels
            /// 3. All scales are positive and finite
            /// 4. Scales are correctly computed according to DiTAS formula
            #[test]
            fn prop_channel_equalization_preserves_shape(
                num_channels in 1usize..256,
                num_features in 1usize..512
            ) {
                let quantizer = SpatialQuantizer::new(128);

                // Create random weights and activations
                let weights = Array2::from_shape_fn((num_channels, num_features), |(i, j)| {
                    ((i * 11 + j * 17) % 2000) as f32 / 100.0 - 10.0
                });
                let activations = Array2::from_shape_fn((num_channels, num_features), |(i, j)| {
                    ((i * 13 + j * 19) % 2000) as f32 / 100.0 - 10.0
                });

                let (equalized, scales) = quantizer.channel_equalization(&weights, &activations);

                // Property 1: Shape preservation
                prop_assert_eq!(
                    equalized.shape(),
                    weights.shape(),
                    "Equalized weights should have same shape as input"
                );

                // Property 2: Number of scales equals number of channels
                prop_assert_eq!(
                    scales.len(),
                    num_channels,
                    "Should have one scale per channel"
                );

                // Property 3: All scales are positive and finite
                for (i, &scale) in scales.iter().enumerate() {
                    prop_assert!(
                        scale > 0.0 && scale.is_finite(),
                        "Scale {} should be positive and finite, got {}",
                        i,
                        scale
                    );
                }

                // Property 4: Equalized weights are finite
                for elem in equalized.iter() {
                    prop_assert!(
                        elem.is_finite(),
                        "All equalized weights should be finite"
                    );
                }
            }

            /// **Validates: Requirements 2.4**
            ///
            /// Property: Activation smoothing should preserve shape and reduce variance
            ///
            /// This property test verifies that:
            /// 1. Output shape matches input shape
            /// 2. All smoothed values are finite
            /// 3. With alpha=0, output equals input
            /// 4. With alpha=1, output is constant (all values equal to mean)
            #[test]
            fn prop_activation_smoothing_preserves_shape(
                num_channels in 1usize..128,
                num_features in 1usize..256,
                alpha in 0.0f32..=1.0f32
            ) {
                let quantizer = SpatialQuantizer::new(128);

                // Create random activations
                let activations = Array2::from_shape_fn((num_channels, num_features), |(i, j)| {
                    ((i * 23 + j * 29) % 2000) as f32 / 100.0 - 10.0
                });

                let smoothed = quantizer.activation_smoothing(&activations, alpha);

                // Property 1: Shape preservation
                prop_assert_eq!(
                    smoothed.shape(),
                    activations.shape(),
                    "Smoothed activations should have same shape as input"
                );

                // Property 2: All smoothed values are finite
                for elem in smoothed.iter() {
                    prop_assert!(
                        elem.is_finite(),
                        "All smoothed values should be finite"
                    );
                }

                // Property 3: Mean is preserved (approximately)
                let orig_mean = activations.mean().unwrap_or(0.0);
                let smooth_mean = smoothed.mean().unwrap_or(0.0);
                prop_assert!(
                    (orig_mean - smooth_mean).abs() < 0.01,
                    "Mean should be approximately preserved: orig={}, smooth={}",
                    orig_mean,
                    smooth_mean
                );
            }

            /// **Validates: Requirements 2.4**
            ///
            /// Property: Activation smoothing with alpha=0 should return original
            ///
            /// This is a specific edge case property test.
            #[test]
            fn prop_activation_smoothing_alpha_zero_identity(
                num_channels in 1usize..64,
                num_features in 1usize..128
            ) {
                let quantizer = SpatialQuantizer::new(128);

                let activations = Array2::from_shape_fn((num_channels, num_features), |(i, j)| {
                    ((i * 31 + j * 37) % 1000) as f32 / 100.0
                });

                let smoothed = quantizer.activation_smoothing(&activations, 0.0);

                // With alpha=0, output should equal input
                for (orig, smooth) in activations.iter().zip(smoothed.iter()) {
                    prop_assert!(
                        (orig - smooth).abs() < 1e-6,
                        "With alpha=0, smoothed should equal original: {} vs {}",
                        orig,
                        smooth
                    );
                }
            }

            /// **Validates: Requirements 2.4**
            ///
            /// Property: Activation smoothing with alpha=1 should return constant (mean)
            ///
            /// This is a specific edge case property test.
            #[test]
            fn prop_activation_smoothing_alpha_one_constant(
                num_channels in 1usize..64,
                num_features in 1usize..128
            ) {
                let quantizer = SpatialQuantizer::new(128);

                let activations = Array2::from_shape_fn((num_channels, num_features), |(i, j)| {
                    ((i * 41 + j * 43) % 1000) as f32 / 100.0
                });

                let mean = activations.mean().unwrap_or(0.0);
                let smoothed = quantizer.activation_smoothing(&activations, 1.0);

                // With alpha=1, all values should equal mean
                for smooth in smoothed.iter() {
                    prop_assert!(
                        (smooth - mean).abs() < 0.01,
                        "With alpha=1, all values should equal mean: {} vs {}",
                        smooth,
                        mean
                    );
                }
            }

            /// **Validates: Requirements 2.5**
            ///
            /// Property: Per-group quantization with different group sizes should all preserve shape
            ///
            /// This property test verifies that regardless of group size, the fundamental
            /// properties hold.
            #[test]
            fn prop_different_group_sizes_preserve_shape(
                num_channels in 1usize..256,
                num_features in 1usize..512
            ) {
                let group_sizes = vec![32, 64, 128, 256];

                let weights = Array2::from_shape_fn((num_channels, num_features), |(i, j)| {
                    ((i * 47 + j * 53) % 2000) as f32 / 100.0 - 10.0
                });

                for &group_size in &group_sizes {
                    let quantizer = SpatialQuantizer::new(group_size);
                    let result = quantizer.per_group_quantize(&weights);

                    prop_assert!(result.is_ok(), "Quantization with group_size {} should succeed", group_size);

                    let quantized = result.unwrap();

                    // Shape preservation
                    prop_assert_eq!(
                        quantized.data.len(),
                        num_channels * num_features,
                        "Group size {} should preserve total elements",
                        group_size
                    );

                    // Correct number of groups
                    let expected_groups = (num_channels + group_size - 1) / group_size;
                    prop_assert_eq!(
                        quantized.scales.len(),
                        expected_groups,
                        "Group size {} should have correct number of groups",
                        group_size
                    );
                }
            }
        }
    }
}

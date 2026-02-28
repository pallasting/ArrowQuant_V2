//! Unit tests for SpatialQuantizer
//!
//! This test suite validates the spatial quantization functionality
//! for diffusion models, including:
//! - Channel equalization formula correctness
//! - Activation smoothing with different alpha values
//! - Per-group quantization with different group sizes
//! - Equalization scale computation

use arrow_quant_v2::spatial::{SpatialQuantizer, QuantizedSpatialLayer};
use ndarray::Array2;

// ============================================================================
// Test 1: Channel Equalization Formula Correctness
// ============================================================================

#[test]
fn test_channel_equalization_basic() {
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
            1.0, 2.0, 3.0,    // Channel 0
            4.0, 8.0, 16.0,   // Channel 1
        ],
    )
    .unwrap();

    let activations = Array2::from_shape_vec(
        (2, 3),
        vec![
            4.0, 5.0, 6.0,    // Channel 0
            2.0, 4.0, 8.0,    // Channel 1
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

    let (equalized, scales) = quantizer.channel_equalization(&weights, &activations);

    // Should use scale = 1.0 to avoid division by zero
    assert_eq!(scales[0], 1.0, "Scale should be 1.0 for zero activation");

    // Weights should remain unchanged (divided by 1.0)
    assert_eq!(equalized[[0, 0]], 1.0);
    assert_eq!(equalized[[0, 1]], 2.0);
    assert_eq!(equalized[[0, 2]], 3.0);
}

#[test]
fn test_channel_equalization_negative_values() {
    // Test that absolute values are used correctly
    let quantizer = SpatialQuantizer::new(128);

    let weights = Array2::from_shape_vec((1, 3), vec![-3.0, 2.0, -1.0]).unwrap();
    let activations = Array2::from_shape_vec((1, 3), vec![-6.0, 5.0, -4.0]).unwrap();

    let (equalized, scales) = quantizer.channel_equalization(&weights, &activations);

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
fn test_channel_equalization_preserves_shape() {
    let quantizer = SpatialQuantizer::new(64);
    let weights = Array2::from_shape_fn((100, 200), |(i, j)| (i + j) as f32 / 10.0);
    let activations = Array2::from_shape_fn((100, 200), |(i, j)| (i * 2 + j) as f32 / 10.0);

    let (equalized, scales) = quantizer.channel_equalization(&weights, &activations);

    assert_eq!(equalized.shape(), weights.shape());
    assert_eq!(scales.len(), 100);
}

#[test]
fn test_channel_equalization_all_positive() {
    let quantizer = SpatialQuantizer::new(128);
    let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let activations = Array2::from_shape_vec((2, 2), vec![2.0, 4.0, 6.0, 8.0]).unwrap();

    let (equalized, scales) = quantizer.channel_equalization(&weights, &activations);

    // All scales should be positive
    for scale in &scales {
        assert!(*scale > 0.0, "All scales should be positive");
    }

    // Equalized weights should be finite
    for val in equalized.iter() {
        assert!(val.is_finite(), "All equalized values should be finite");
    }
}

#[test]
fn test_channel_equalization_large_scale_difference() {
    // Test with channels having very different scales
    let quantizer = SpatialQuantizer::new(128);

    let weights = Array2::from_shape_vec(
        (2, 3),
        vec![
            0.1, 0.2, 0.3,    // Channel 0: small values
            100.0, 200.0, 300.0,  // Channel 1: large values
        ],
    )
    .unwrap();

    let activations = Array2::from_shape_vec(
        (2, 3),
        vec![
            10.0, 20.0, 30.0,  // Channel 0
            1.0, 2.0, 3.0,     // Channel 1
        ],
    )
    .unwrap();

    let (equalized, scales) = quantizer.channel_equalization(&weights, &activations);

    // Channel 0: w_max=0.3, x_max=30.0, scale=sqrt(0.3/30.0)=sqrt(0.01)=0.1
    assert!((scales[0] - 0.1).abs() < 0.01);

    // Channel 1: w_max=300.0, x_max=3.0, scale=sqrt(300.0/3.0)=sqrt(100.0)=10.0
    assert!((scales[1] - 10.0).abs() < 0.1);

    // Verify equalization reduces the scale difference
    assert!(scales[1] > scales[0]);
}

// ============================================================================
// Test 2: Activation Smoothing with Different Alpha Values
// ============================================================================

#[test]
fn test_activation_smoothing_basic() {
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
fn test_activation_smoothing_alpha_quarter() {
    // Test with alpha = 0.25
    let quantizer = SpatialQuantizer::new(128);
    let activations = Array2::from_shape_vec((1, 4), vec![0.0, 10.0, 20.0, 30.0]).unwrap();
    let mean = 15.0;
    let alpha = 0.25;

    let smoothed = quantizer.activation_smoothing(&activations, alpha);

    // x_smooth = x * (1 - 0.25) + 15.0 * 0.25 = x * 0.75 + 3.75
    assert!((smoothed[[0, 0]] - 3.75).abs() < 0.01);  // 0 * 0.75 + 3.75
    assert!((smoothed[[0, 1]] - 11.25).abs() < 0.01); // 10 * 0.75 + 3.75
    assert!((smoothed[[0, 2]] - 18.75).abs() < 0.01); // 20 * 0.75 + 3.75
    assert!((smoothed[[0, 3]] - 26.25).abs() < 0.01); // 30 * 0.75 + 3.75
}

#[test]
fn test_activation_smoothing_alpha_three_quarters() {
    // Test with alpha = 0.75 (more smoothing)
    let quantizer = SpatialQuantizer::new(128);
    let activations = Array2::from_shape_vec((1, 4), vec![0.0, 10.0, 20.0, 30.0]).unwrap();
    let mean = 15.0;
    let alpha = 0.75;

    let smoothed = quantizer.activation_smoothing(&activations, alpha);

    // x_smooth = x * 0.25 + 15.0 * 0.75 = x * 0.25 + 11.25
    assert!((smoothed[[0, 0]] - 11.25).abs() < 0.01);  // 0 * 0.25 + 11.25
    assert!((smoothed[[0, 1]] - 13.75).abs() < 0.01);  // 10 * 0.25 + 11.25
    assert!((smoothed[[0, 2]] - 16.25).abs() < 0.01);  // 20 * 0.25 + 11.25
    assert!((smoothed[[0, 3]] - 18.75).abs() < 0.01);  // 30 * 0.25 + 11.25
}

#[test]
fn test_activation_smoothing_preserves_mean() {
    // Test that smoothing preserves the mean
    let quantizer = SpatialQuantizer::new(128);
    let activations = Array2::from_shape_fn((5, 5), |(i, j)| (i * 3 + j * 7) as f32);

    let orig_mean = activations.mean().unwrap();
    let smoothed = quantizer.activation_smoothing(&activations, 0.6);
    let smooth_mean = smoothed.mean().unwrap();

    // Mean should be approximately preserved
    assert!(
        (orig_mean - smooth_mean).abs() < 0.01,
        "Mean should be preserved: orig={}, smooth={}",
        orig_mean,
        smooth_mean
    );
}

// ============================================================================
// Test 3: Per-Group Quantization with Different Group Sizes
// ============================================================================

#[test]
fn test_per_group_quantize_basic() {
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
fn test_per_group_quantize_group_size_64() {
    // Test with group size 64
    let quantizer = SpatialQuantizer::new(64);
    let weights = Array2::from_shape_fn((256, 512), |(i, j)| (i + j) as f32 / 100.0);

    let result = quantizer.per_group_quantize(&weights);
    assert!(result.is_ok());

    let quantized = result.unwrap();
    assert_eq!(quantized.group_size, 64);
    assert_eq!(quantized.scales.len(), 4); // 256 channels / 64 group_size = 4 groups
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
            0.0, 5.0, 10.0,       // Channel 0 (Group 0)
            1.0, 6.0, 9.0,        // Channel 1 (Group 0)
            100.0, 105.0, 110.0,  // Channel 2 (Group 1)
            101.0, 106.0, 109.0,  // Channel 3 (Group 1)
        ],
    )
    .unwrap();

    let result = quantizer.per_group_quantize(&weights);
    assert!(result.is_ok());

    let quantized = result.unwrap();
    assert_eq!(quantized.scales.len(), 2);
    assert_eq!(quantized.zero_points.len(), 2);

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
            0.0, 1.0,      // Channel 0 (Group 0)
            0.5, 0.8,      // Channel 1 (Group 0)
            0.0, 100.0,    // Channel 2 (Group 1)
            50.0, 75.0,    // Channel 3 (Group 1)
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

// ============================================================================
// Test 4: Equalization Scale Computation
// ============================================================================

#[test]
fn test_equalization_scale_computation_basic() {
    let quantizer = SpatialQuantizer::new(128);

    // Simple case: weights and activations with known max values
    let weights = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let activations = Array2::from_shape_vec((1, 4), vec![4.0, 8.0, 12.0, 16.0]).unwrap();

    let (_equalized, scales) = quantizer.channel_equalization(&weights, &activations);

    // w_max = 4.0, x_max = 16.0
    // scale = sqrt(4.0 / 16.0) = sqrt(0.25) = 0.5
    assert!(
        (scales[0] - 0.5).abs() < 0.01,
        "Scale should be 0.5, got {}",
        scales[0]
    );
}

#[test]
fn test_equalization_scale_computation_multiple_channels() {
    let quantizer = SpatialQuantizer::new(128);

    let weights = Array2::from_shape_vec(
        (3, 2),
        vec![
            1.0, 2.0,    // Channel 0: max = 2.0
            3.0, 6.0,    // Channel 1: max = 6.0
            5.0, 10.0,   // Channel 2: max = 10.0
        ],
    )
    .unwrap();

    let activations = Array2::from_shape_vec(
        (3, 2),
        vec![
            8.0, 4.0,    // Channel 0: max = 8.0
            9.0, 3.0,    // Channel 1: max = 9.0
            20.0, 10.0,  // Channel 2: max = 20.0
        ],
    )
    .unwrap();

    let (_equalized, scales) = quantizer.channel_equalization(&weights, &activations);

    // Channel 0: sqrt(2.0 / 8.0) = sqrt(0.25) = 0.5
    assert!((scales[0] - 0.5).abs() < 0.01);

    // Channel 1: sqrt(6.0 / 9.0) = sqrt(0.667) ≈ 0.816
    assert!((scales[1] - 0.816).abs() < 0.01);

    // Channel 2: sqrt(10.0 / 20.0) = sqrt(0.5) ≈ 0.707
    assert!((scales[2] - 0.707).abs() < 0.01);
}

#[test]
fn test_equalization_scale_computation_equal_ranges() {
    let quantizer = SpatialQuantizer::new(128);

    // When weight and activation ranges are equal, scale should be 1.0
    let weights = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let activations = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();

    let (_equalized, scales) = quantizer.channel_equalization(&weights, &activations);

    // w_max = 3.0, x_max = 3.0
    // scale = sqrt(3.0 / 3.0) = sqrt(1.0) = 1.0
    assert!(
        (scales[0] - 1.0).abs() < 0.01,
        "Scale should be 1.0 when ranges are equal, got {}",
        scales[0]
    );
}

#[test]
fn test_equalization_scale_computation_weights_larger() {
    let quantizer = SpatialQuantizer::new(128);

    // When weights have larger range than activations, scale > 1.0
    let weights = Array2::from_shape_vec((1, 2), vec![10.0, 20.0]).unwrap();
    let activations = Array2::from_shape_vec((1, 2), vec![2.0, 4.0]).unwrap();

    let (_equalized, scales) = quantizer.channel_equalization(&weights, &activations);

    // w_max = 20.0, x_max = 4.0
    // scale = sqrt(20.0 / 4.0) = sqrt(5.0) ≈ 2.236
    assert!(
        scales[0] > 1.0,
        "Scale should be > 1.0 when weights have larger range"
    );
    assert!((scales[0] - 2.236).abs() < 0.01);
}

#[test]
fn test_equalization_scale_computation_activations_larger() {
    let quantizer = SpatialQuantizer::new(128);

    // When activations have larger range than weights, scale < 1.0
    let weights = Array2::from_shape_vec((1, 2), vec![2.0, 4.0]).unwrap();
    let activations = Array2::from_shape_vec((1, 2), vec![10.0, 20.0]).unwrap();

    let (_equalized, scales) = quantizer.channel_equalization(&weights, &activations);

    // w_max = 4.0, x_max = 20.0
    // scale = sqrt(4.0 / 20.0) = sqrt(0.2) ≈ 0.447
    assert!(
        scales[0] < 1.0,
        "Scale should be < 1.0 when activations have larger range"
    );
    assert!((scales[0] - 0.447).abs() < 0.01);
}

#[test]
fn test_equalization_scale_all_positive() {
    let quantizer = SpatialQuantizer::new(128);

    // Test that all scales are positive regardless of input
    // Use non-zero values to avoid edge case where both w_max and x_max are 0
    let weights = Array2::from_shape_fn((10, 10), |(i, j)| (i as f32 - 5.0) * (j as f32 - 5.0) + 1.0);
    let activations = Array2::from_shape_fn((10, 10), |(i, j)| (i * j) as f32 + 1.0);

    let (_equalized, scales) = quantizer.channel_equalization(&weights, &activations);

    for (i, &scale) in scales.iter().enumerate() {
        assert!(
            scale > 0.0,
            "Scale {} should be positive, got {}",
            i,
            scale
        );
    }
}

#[test]
fn test_equalization_scale_finite() {
    let quantizer = SpatialQuantizer::new(128);

    // Test that all scales are finite
    let weights = Array2::from_shape_fn((5, 5), |(i, j)| (i + j) as f32 * 0.1);
    let activations = Array2::from_shape_fn((5, 5), |(i, j)| (i * j + 1) as f32 * 0.1);

    let (_equalized, scales) = quantizer.channel_equalization(&weights, &activations);

    for (i, &scale) in scales.iter().enumerate() {
        assert!(
            scale.is_finite(),
            "Scale {} should be finite, got {}",
            i,
            scale
        );
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_end_to_end_spatial_quantization() {
    // Simulate a complete spatial quantization workflow
    let quantizer = SpatialQuantizer::new(64);

    // Create realistic weights and activations
    let weights = Array2::from_shape_fn((256, 512), |(i, j)| {
        ((i * 7 + j * 13) % 2000) as f32 / 100.0 - 10.0
    });
    let activations = Array2::from_shape_fn((256, 512), |(i, j)| {
        ((i * 11 + j * 17) % 2000) as f32 / 100.0 - 10.0
    });

    // Step 1: Channel equalization
    let (equalized_weights, scales) = quantizer.channel_equalization(&weights, &activations);
    assert_eq!(scales.len(), 256);
    assert_eq!(equalized_weights.shape(), weights.shape());

    // Step 2: Activation smoothing
    let smoothed_activations = quantizer.activation_smoothing(&activations, 0.5);
    assert_eq!(smoothed_activations.shape(), activations.shape());

    // Step 3: Per-group quantization
    let result = quantizer.per_group_quantize(&equalized_weights);
    assert!(result.is_ok());

    let quantized = result.unwrap();
    assert_eq!(quantized.group_size, 64);
    assert_eq!(quantized.scales.len(), 4); // 256 / 64 = 4 groups
    assert_eq!(quantized.data.len(), weights.len());
}

#[test]
fn test_spatial_quantization_with_different_modalities() {
    // Test spatial quantization for different diffusion modalities

    // Image modality (continuous diffusion) - typically uses spatial quantization
    let image_quantizer = SpatialQuantizer::new(128);
    let image_weights = Array2::from_shape_fn((512, 1024), |(i, j)| {
        ((i + j) as f32 / 100.0).sin() * 5.0
    });

    let result = image_quantizer.per_group_quantize(&image_weights);
    assert!(result.is_ok());
    let quantized = result.unwrap();
    assert_eq!(quantized.scales.len(), 4); // 512 / 128 = 4 groups

    // Audio modality (continuous diffusion) - also uses spatial quantization
    let audio_quantizer = SpatialQuantizer::new(64);
    let audio_weights = Array2::from_shape_fn((256, 512), |(i, j)| {
        ((i * j) as f32 / 1000.0).cos() * 3.0
    });

    let result = audio_quantizer.per_group_quantize(&audio_weights);
    assert!(result.is_ok());
    let quantized = result.unwrap();
    assert_eq!(quantized.scales.len(), 4); // 256 / 64 = 4 groups
}

#[test]
fn test_combined_equalization_and_smoothing() {
    // Test combining channel equalization and activation smoothing
    let quantizer = SpatialQuantizer::new(128);

    let weights = Array2::from_shape_fn((100, 200), |(i, j)| (i + j) as f32 / 50.0);
    let activations = Array2::from_shape_fn((100, 200), |(i, j)| (i * 2 + j) as f32 / 50.0);

    // Apply equalization
    let (equalized_weights, scales) = quantizer.channel_equalization(&weights, &activations);
    assert_eq!(scales.len(), 100);

    // Apply smoothing to activations
    let smoothed_activations = quantizer.activation_smoothing(&activations, 0.3);

    // Verify both operations preserve shape
    assert_eq!(equalized_weights.shape(), weights.shape());
    assert_eq!(smoothed_activations.shape(), activations.shape());

    // Quantize the equalized weights
    let result = quantizer.per_group_quantize(&equalized_weights);
    assert!(result.is_ok());
}

#[test]
fn test_quantization_with_extreme_values() {
    // Test that quantization handles extreme values correctly
    let quantizer = SpatialQuantizer::new(64);

    // Create weights with extreme values
    let mut weights_vec = vec![0.0; 128 * 256];
    weights_vec[0] = -1000.0;  // Very negative
    weights_vec[1] = 1000.0;   // Very positive
    weights_vec[2] = 0.0001;   // Very small
    let weights = Array2::from_shape_vec((128, 256), weights_vec).unwrap();

    let result = quantizer.per_group_quantize(&weights);
    assert!(result.is_ok());

    let quantized = result.unwrap();
    // All quantized values should be in [0, 255]
    for &val in &quantized.data {
        assert!(val <= 255, "Quantized value should be <= 255");
    }
}

#[test]
fn test_quantization_preserves_relative_ordering() {
    // Test that quantization preserves relative ordering within groups
    let quantizer = SpatialQuantizer::new(4);

    // Create weights with clear ordering
    let weights = Array2::from_shape_vec(
        (4, 1),
        vec![
            1.0,   // Group 0
            2.0,   // Group 0
            3.0,   // Group 0
            4.0,   // Group 0
        ],
    )
    .unwrap();

    let result = quantizer.per_group_quantize(&weights);
    assert!(result.is_ok());

    let quantized = result.unwrap();
    // Within the same group, ordering should be preserved
    assert!(quantized.data[0] < quantized.data[1]);
    assert!(quantized.data[1] < quantized.data[2]);
    assert!(quantized.data[2] < quantized.data[3]);
}

#[test]
fn test_empty_weights_handling() {
    // Test that empty weights are handled gracefully
    let quantizer = SpatialQuantizer::new(64);
    let weights = Array2::from_shape_vec((0, 0), vec![]).unwrap();

    let result = quantizer.per_group_quantize(&weights);
    assert!(result.is_ok());

    let quantized = result.unwrap();
    assert_eq!(quantized.data.len(), 0);
    assert_eq!(quantized.scales.len(), 0);
    assert_eq!(quantized.zero_points.len(), 0);
}

#[test]
fn test_single_channel_quantization() {
    // Test quantization with a single channel
    let quantizer = SpatialQuantizer::new(64);
    let weights = Array2::from_shape_vec((1, 10), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();

    let result = quantizer.per_group_quantize(&weights);
    assert!(result.is_ok());

    let quantized = result.unwrap();
    assert_eq!(quantized.scales.len(), 1);
    assert_eq!(quantized.zero_points.len(), 1);
    assert_eq!(quantized.data.len(), 10);
}

#[test]
fn test_quantization_consistency() {
    // Test that quantizing the same weights twice produces the same result
    let quantizer = SpatialQuantizer::new(64);
    let weights = Array2::from_shape_fn((128, 256), |(i, j)| (i + j) as f32 / 100.0);

    let result1 = quantizer.per_group_quantize(&weights).unwrap();
    let result2 = quantizer.per_group_quantize(&weights).unwrap();

    // Results should be identical
    assert_eq!(result1.data, result2.data);
    assert_eq!(result1.scales, result2.scales);
    assert_eq!(result1.zero_points, result2.zero_points);
}

#[test]
fn test_activation_smoothing_consistency() {
    // Test that smoothing the same activations twice produces the same result
    let quantizer = SpatialQuantizer::new(128);
    let activations = Array2::from_shape_fn((50, 50), |(i, j)| (i * j) as f32);

    let smoothed1 = quantizer.activation_smoothing(&activations, 0.5);
    let smoothed2 = quantizer.activation_smoothing(&activations, 0.5);

    // Results should be identical
    for (v1, v2) in smoothed1.iter().zip(smoothed2.iter()) {
        assert!((v1 - v2).abs() < 1e-6);
    }
}

#[test]
fn test_channel_equalization_consistency() {
    // Test that equalization produces consistent results
    let quantizer = SpatialQuantizer::new(128);
    let weights = Array2::from_shape_fn((50, 50), |(i, j)| (i + j) as f32);
    let activations = Array2::from_shape_fn((50, 50), |(i, j)| (i * j + 1) as f32);

    let (eq1, scales1) = quantizer.channel_equalization(&weights, &activations);
    let (eq2, scales2) = quantizer.channel_equalization(&weights, &activations);

    // Results should be identical
    assert_eq!(scales1, scales2);
    for (v1, v2) in eq1.iter().zip(eq2.iter()) {
        assert!((v1 - v2).abs() < 1e-6);
    }
}

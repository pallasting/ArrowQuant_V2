//! Unit tests for TimeAwareQuantizer
//!
//! This test suite validates the time-aware quantization functionality
//! for diffusion models, including:
//! - Timestep grouping with different group counts
//! - Parameter computation per group
//! - Quantization correctness with time-aware params
//! - Adaptive group size selection

use arrow_quant_v2::time_aware::{ActivationStats, TimeAwareQuantizer};

// ============================================================================
// Test 1: Timestep Grouping with Different Group Counts
// ============================================================================

#[test]
fn test_timestep_grouping_10_groups() {
    let mut quantizer = TimeAwareQuantizer::new(10);
    quantizer.group_timesteps(1000);

    let boundaries = quantizer.time_group_boundaries();

    // Should have 11 boundaries (10 groups + 1)
    assert_eq!(boundaries.len(), 11);
    
    // First boundary should be 0
    assert_eq!(boundaries[0], 0);
    
    // Last boundary should be 1000
    assert_eq!(boundaries[10], 1000);

    // Verify exact boundaries for 10 groups over 1000 timesteps
    let expected = vec![0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
    assert_eq!(boundaries, expected);
}

#[test]
fn test_timestep_grouping_5_groups() {
    let mut quantizer = TimeAwareQuantizer::new(5);
    quantizer.group_timesteps(1000);

    let boundaries = quantizer.time_group_boundaries();
    assert_eq!(boundaries.len(), 6);
    assert_eq!(
        boundaries,
        vec![0, 200, 400, 600, 800, 1000]
    );
}

#[test]
fn test_timestep_grouping_20_groups() {
    let mut quantizer = TimeAwareQuantizer::new(20);
    quantizer.group_timesteps(1000);

    let boundaries = quantizer.time_group_boundaries();
    assert_eq!(boundaries.len(), 21);
    assert_eq!(boundaries[0], 0);
    assert_eq!(boundaries[1], 50);
    assert_eq!(boundaries[10], 500);
    assert_eq!(boundaries[20], 1000);
}

#[test]
fn test_timestep_grouping_single_group() {
    let mut quantizer = TimeAwareQuantizer::new(1);
    quantizer.group_timesteps(1000);

    let boundaries = quantizer.time_group_boundaries();
    assert_eq!(boundaries.len(), 2);
    assert_eq!(boundaries, vec![0, 1000]);
}

#[test]
fn test_timestep_grouping_non_standard_timesteps() {
    // Test with 500 timesteps
    let mut quantizer = TimeAwareQuantizer::new(10);
    quantizer.group_timesteps(500);
    
    let boundaries = quantizer.time_group_boundaries();
    assert_eq!(boundaries.len(), 11);
    assert_eq!(
        boundaries,
        vec![0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    );

    // Test with 250 timesteps
    let mut quantizer = TimeAwareQuantizer::new(5);
    quantizer.group_timesteps(250);
    
    let boundaries = quantizer.time_group_boundaries();
    assert_eq!(boundaries.len(), 6);
    assert_eq!(
        boundaries,
        vec![0, 50, 100, 150, 200, 250]
    );
}

#[test]
fn test_timestep_grouping_edge_case_fewer_timesteps_than_groups() {
    // When num_timesteps < num_groups, should adjust to num_timesteps groups
    let mut quantizer = TimeAwareQuantizer::new(100);
    quantizer.group_timesteps(10);

    let boundaries = quantizer.time_group_boundaries();
    // Should create 10 groups (one per timestep)
    assert_eq!(boundaries.len(), 11);
    assert_eq!(
        boundaries,
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
}

#[test]
fn test_timestep_grouping_boundaries_are_monotonic() {
    let mut quantizer = TimeAwareQuantizer::new(15);
    quantizer.group_timesteps(1000);

    let boundaries = quantizer.time_group_boundaries();
    // Verify boundaries are strictly increasing
    for i in 0..boundaries.len() - 1 {
        assert!(
            boundaries[i] < boundaries[i + 1],
            "Boundaries must be strictly increasing"
        );
    }
}

// ============================================================================
// Test 2: Parameter Computation Per Group
// ============================================================================

#[test]
fn test_compute_params_basic() {
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100);

    let stats = ActivationStats {
        mean: vec![0.0; 100],
        std: vec![1.0; 100],
        min: vec![-2.0; 100],
        max: vec![2.0; 100],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Should have 2 parameter sets
    assert_eq!(params.len(), 2);

    // Verify time ranges
    assert_eq!(params[0].time_range, (0, 50));
    assert_eq!(params[1].time_range, (50, 100));

    // Verify scales are computed correctly
    // Range = max - min = 2.0 - (-2.0) = 4.0
    // Scale = 4.0 / 255.0 ≈ 0.0157
    for param in &params {
        assert!((param.scale - 4.0 / 255.0).abs() < 1e-6);
    }

    // Verify zero_points are computed correctly
    // zero_point = -min / scale = -(-2.0) / (4.0/255.0) = 2.0 * 255.0 / 4.0 = 127.5
    for param in &params {
        assert!((param.zero_point - 127.5).abs() < 1e-3);
    }
}

#[test]
fn test_compute_params_different_ranges() {
    let mut quantizer = TimeAwareQuantizer::new(3);
    quantizer.group_timesteps(300);

    // Create stats with different ranges per group
    let mut min_vals = vec![-1.0; 100];
    min_vals.extend(vec![-5.0; 100]);
    min_vals.extend(vec![-10.0; 100]);

    let mut max_vals = vec![1.0; 100];
    max_vals.extend(vec![5.0; 100]);
    max_vals.extend(vec![10.0; 100]);

    let stats = ActivationStats {
        mean: vec![0.0; 300],
        std: vec![1.0; 300],
        min: min_vals,
        max: max_vals,
    };

    let params = quantizer.compute_params_per_group(&stats);

    assert_eq!(params.len(), 3);

    // Group 0: range = 1.0 - (-1.0) = 2.0
    assert!((params[0].scale - 2.0 / 255.0).abs() < 1e-6);

    // Group 1: range = 5.0 - (-5.0) = 10.0
    assert!((params[1].scale - 10.0 / 255.0).abs() < 1e-6);

    // Group 2: range = 10.0 - (-10.0) = 20.0
    assert!((params[2].scale - 20.0 / 255.0).abs() < 1e-6);
}

#[test]
fn test_compute_params_handles_zero_range() {
    let mut quantizer = TimeAwareQuantizer::new(1);
    quantizer.group_timesteps(10);

    // All values are the same (zero range)
    let stats = ActivationStats {
        mean: vec![5.0; 10],
        std: vec![0.0; 10],
        min: vec![5.0; 10],
        max: vec![5.0; 10],
    };

    let params = quantizer.compute_params_per_group(&stats);

    assert_eq!(params.len(), 1);

    // Should use epsilon scale to avoid division by zero
    assert!(params[0].scale > 0.0);
    assert!(params[0].scale < 1e-5);
}

#[test]
fn test_compute_params_positive_values_only() {
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100);

    let stats = ActivationStats {
        mean: vec![5.0; 100],
        std: vec![1.0; 100],
        min: vec![0.0; 100],
        max: vec![10.0; 100],
    };

    let params = quantizer.compute_params_per_group(&stats);

    assert_eq!(params.len(), 2);

    // Range = 10.0 - 0.0 = 10.0
    // Scale = 10.0 / 255.0
    for param in &params {
        assert!((param.scale - 10.0 / 255.0).abs() < 1e-6);
    }

    // zero_point = -0.0 / scale = 0.0
    for param in &params {
        assert!(param.zero_point.abs() < 1e-6);
    }
}

#[test]
fn test_compute_params_negative_values_only() {
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100);

    let stats = ActivationStats {
        mean: vec![-5.0; 100],
        std: vec![1.0; 100],
        min: vec![-10.0; 100],
        max: vec![0.0; 100],
    };

    let params = quantizer.compute_params_per_group(&stats);

    assert_eq!(params.len(), 2);

    // Range = 0.0 - (-10.0) = 10.0
    // Scale = 10.0 / 255.0
    for param in &params {
        assert!((param.scale - 10.0 / 255.0).abs() < 1e-6);
    }

    // zero_point = -(-10.0) / scale = 10.0 / (10.0/255.0) = 255.0
    for param in &params {
        assert!((param.zero_point - 255.0).abs() < 1e-3);
    }
}

// ============================================================================
// Test 3: Adaptive Group Size Selection
// ============================================================================

#[test]
fn test_adaptive_group_size_early_vs_late() {
    let mut quantizer = TimeAwareQuantizer::new(10);
    quantizer.group_timesteps(1000);

    let stats = ActivationStats {
        mean: vec![0.0; 1000],
        std: vec![1.0; 1000],
        min: vec![-1.0; 1000],
        max: vec![1.0; 1000],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Early groups (first half) should have larger group size (256)
    for i in 0..5 {
        assert_eq!(
            params[i].group_size, 256,
            "Early group {} should have group_size=256",
            i
        );
    }

    // Late groups (second half) should have smaller group size (64)
    for i in 5..10 {
        assert_eq!(
            params[i].group_size, 64,
            "Late group {} should have group_size=64",
            i
        );
    }
}

#[test]
fn test_adaptive_group_size_odd_number_of_groups() {
    let mut quantizer = TimeAwareQuantizer::new(5);
    quantizer.group_timesteps(500);

    let stats = ActivationStats {
        mean: vec![0.0; 500],
        std: vec![1.0; 500],
        min: vec![-1.0; 500],
        max: vec![1.0; 500],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // First 2 groups (indices 0, 1) should be 256
    assert_eq!(params[0].group_size, 256);
    assert_eq!(params[1].group_size, 256);

    // Last 3 groups (indices 2, 3, 4) should be 64
    assert_eq!(params[2].group_size, 64);
    assert_eq!(params[3].group_size, 64);
    assert_eq!(params[4].group_size, 64);
}

#[test]
fn test_adaptive_group_size_single_group() {
    let mut quantizer = TimeAwareQuantizer::new(1);
    quantizer.group_timesteps(100);

    let stats = ActivationStats {
        mean: vec![0.0; 100],
        std: vec![1.0; 100],
        min: vec![-1.0; 100],
        max: vec![1.0; 100],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Single group: i=0, num_time_groups=1, so 0 < 1/2 = 0 < 0 is false, so should be 64
    assert_eq!(params[0].group_size, 64);
}

#[test]
fn test_adaptive_group_size_two_groups() {
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(200);

    let stats = ActivationStats {
        mean: vec![0.0; 200],
        std: vec![1.0; 200],
        min: vec![-1.0; 200],
        max: vec![1.0; 200],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // First group (index 0): 0 < 2/2 = 1, so 256
    assert_eq!(params[0].group_size, 256);

    // Second group (index 1): 1 >= 2/2 = 1, so 64
    assert_eq!(params[1].group_size, 64);
}

// ============================================================================
// Test 4: Quantization Correctness with Time-Aware Params
// ============================================================================

#[test]
fn test_quantize_layer_basic() {
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100);

    let stats = ActivationStats {
        mean: vec![0.0; 100],
        std: vec![1.0; 100],
        min: vec![-2.0; 100],
        max: vec![2.0; 100],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Create test weights
    let weights = vec![0.0, 1.0, -1.0, 2.0, -2.0];

    // Quantize layer
    let result = quantizer.quantize_layer(&weights, &params).unwrap();

    // Verify output structure
    assert_eq!(result.scales.len(), 2); // 2 time groups
    assert_eq!(result.zero_points.len(), 2);
    assert_eq!(result.time_group_params.len(), 2);
    
    // Data should be quantized for all groups
    assert_eq!(result.data.len(), weights.len() * 2);
}

#[test]
fn test_quantize_layer_stores_params_correctly() {
    let mut quantizer = TimeAwareQuantizer::new(3);
    quantizer.group_timesteps(300);

    let stats = ActivationStats {
        mean: vec![0.0; 300],
        std: vec![1.0; 300],
        min: vec![-1.0; 300],
        max: vec![1.0; 300],
    };

    let params = quantizer.compute_params_per_group(&stats);
    let weights = vec![0.5; 100];

    let result = quantizer.quantize_layer(&weights, &params).unwrap();

    // Verify time_group_params are stored correctly
    assert_eq!(result.time_group_params.len(), 3);
    assert_eq!(result.time_group_params[0].time_range, (0, 100));
    assert_eq!(result.time_group_params[1].time_range, (100, 200));
    assert_eq!(result.time_group_params[2].time_range, (200, 300));

    // Verify adaptive group sizes
    assert_eq!(result.time_group_params[0].group_size, 256); // Early
    assert_eq!(result.time_group_params[1].group_size, 64); // Late
    assert_eq!(result.time_group_params[2].group_size, 64); // Late
}

#[test]
fn test_quantize_layer_applies_correct_params() {
    let mut quantizer = TimeAwareQuantizer::new(1);
    quantizer.group_timesteps(10);

    let stats = ActivationStats {
        mean: vec![0.0; 10],
        std: vec![1.0; 10],
        min: vec![0.0; 10],
        max: vec![10.0; 10],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Test weights that should quantize predictably
    let weights = vec![0.0, 5.0, 10.0];

    let result = quantizer.quantize_layer(&weights, &params).unwrap();

    // With min=0, max=10, scale should be 10/255 ≈ 0.0392
    // zero_point should be 0
    // So: 0.0 -> 0, 5.0 -> ~127, 10.0 -> 255
    assert_eq!(result.data[0], 0);
    assert!(result.data[1] > 120 && result.data[1] < 135); // ~127
    assert_eq!(result.data[2], 255);
}

#[test]
fn test_quantize_layer_with_negative_values() {
    let mut quantizer = TimeAwareQuantizer::new(1);
    quantizer.group_timesteps(10);

    let stats = ActivationStats {
        mean: vec![0.0; 10],
        std: vec![1.0; 10],
        min: vec![-5.0; 10],
        max: vec![5.0; 10],
    };

    let params = quantizer.compute_params_per_group(&stats);
    let weights = vec![-5.0, 0.0, 5.0];

    let result = quantizer.quantize_layer(&weights, &params).unwrap();

    // Min value should map to ~0, max to ~255, 0 to ~127
    assert!(result.data[0] < 10); // -5.0 -> near 0
    assert!(result.data[1] > 120 && result.data[1] < 135); // 0.0 -> near 127
    assert!(result.data[2] > 245); // 5.0 -> near 255
}

#[test]
fn test_quantize_layer_multiple_groups() {
    let mut quantizer = TimeAwareQuantizer::new(5);
    quantizer.group_timesteps(500);

    let stats = ActivationStats {
        mean: vec![0.0; 500],
        std: vec![1.0; 500],
        min: vec![-1.0; 500],
        max: vec![1.0; 500],
    };

    let params = quantizer.compute_params_per_group(&stats);
    let weights = vec![0.0; 50];

    let result = quantizer.quantize_layer(&weights, &params).unwrap();

    // Should have 5 sets of parameters
    assert_eq!(result.scales.len(), 5);
    assert_eq!(result.zero_points.len(), 5);
    assert_eq!(result.time_group_params.len(), 5);

    // Data should be quantized for all groups
    assert_eq!(result.data.len(), weights.len() * 5);
}

#[test]
fn test_quantize_layer_clamping() {
    let mut quantizer = TimeAwareQuantizer::new(1);
    quantizer.group_timesteps(10);

    let stats = ActivationStats {
        mean: vec![0.0; 10],
        std: vec![1.0; 10],
        min: vec![-1.0; 10],
        max: vec![1.0; 10],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Test extreme values that should be clamped
    let weights = vec![1000.0, -1000.0, 0.0];

    let result = quantizer.quantize_layer(&weights, &params).unwrap();

    // Extreme values should be clamped to [0, 255]
    assert_eq!(result.data[0], 255); // 1000.0 clamped to 255
    assert_eq!(result.data[1], 0); // -1000.0 clamped to 0
}

#[test]
fn test_quantize_layer_preserves_relative_ordering() {
    let mut quantizer = TimeAwareQuantizer::new(1);
    quantizer.group_timesteps(10);

    let stats = ActivationStats {
        mean: vec![0.0; 10],
        std: vec![1.0; 10],
        min: vec![0.0; 10],
        max: vec![100.0; 10],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Test that quantization preserves ordering
    let weights = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    let result = quantizer.quantize_layer(&weights, &params).unwrap();

    // Verify ordering is preserved
    for i in 0..result.data.len() - 1 {
        assert!(
            result.data[i] <= result.data[i + 1],
            "Quantization should preserve ordering"
        );
    }
}

#[test]
fn test_quantize_layer_empty_weights() {
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100);

    let stats = ActivationStats {
        mean: vec![0.0; 100],
        std: vec![1.0; 100],
        min: vec![-1.0; 100],
        max: vec![1.0; 100],
    };

    let params = quantizer.compute_params_per_group(&stats);
    let weights: Vec<f32> = vec![];

    let result = quantizer.quantize_layer(&weights, &params).unwrap();

    // Should handle empty weights gracefully
    assert_eq!(result.data.len(), 0);
    assert_eq!(result.scales.len(), 2);
    assert_eq!(result.zero_points.len(), 2);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_end_to_end_quantization_workflow() {
    // Simulate a complete quantization workflow
    let mut quantizer = TimeAwareQuantizer::new(10);
    quantizer.group_timesteps(1000);

    // Create realistic activation stats
    let stats = ActivationStats {
        mean: vec![0.0; 1000],
        std: vec![1.0; 1000],
        min: vec![-3.0; 1000],
        max: vec![3.0; 1000],
    };

    // Compute parameters
    let params = quantizer.compute_params_per_group(&stats);

    // Verify we have correct number of params
    assert_eq!(params.len(), 10);

    // Create realistic weights
    let weights: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0) * 6.0 - 3.0).collect();

    // Quantize
    let result = quantizer.quantize_layer(&weights, &params).unwrap();

    // Verify structure
    assert_eq!(result.scales.len(), 10);
    assert_eq!(result.zero_points.len(), 10);
    assert_eq!(result.time_group_params.len(), 10);
    assert_eq!(result.data.len(), weights.len() * 10);

    // Verify adaptive group sizes
    for i in 0..5 {
        assert_eq!(result.time_group_params[i].group_size, 256);
    }
    for i in 5..10 {
        assert_eq!(result.time_group_params[i].group_size, 64);
    }
}

#[test]
fn test_different_modalities_simulation() {
    // Test quantization for different diffusion modalities
    
    // Text modality (discrete diffusion) - typically 1000 timesteps
    let mut text_quantizer = TimeAwareQuantizer::new(10);
    text_quantizer.group_timesteps(1000);
    
    let text_stats = ActivationStats {
        mean: vec![0.0; 1000],
        std: vec![0.5; 1000],
        min: vec![-2.0; 1000],
        max: vec![2.0; 1000],
    };
    
    let text_params = text_quantizer.compute_params_per_group(&text_stats);
    assert_eq!(text_params.len(), 10);
    
    // Image modality (continuous diffusion) - typically 1000 timesteps
    let mut image_quantizer = TimeAwareQuantizer::new(20);
    image_quantizer.group_timesteps(1000);
    
    let image_stats = ActivationStats {
        mean: vec![0.0; 1000],
        std: vec![1.5; 1000],
        min: vec![-5.0; 1000],
        max: vec![5.0; 1000],
    };
    
    let image_params = image_quantizer.compute_params_per_group(&image_stats);
    assert_eq!(image_params.len(), 20);
    
    // Verify different configurations produce different results
    assert_ne!(text_params.len(), image_params.len());
}

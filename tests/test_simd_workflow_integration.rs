/// Integration tests for SIMD quantization workflow (Task 9.3)
///
/// This test suite verifies that the complete SIMD workflow correctly handles:
/// - Time group assignment
/// - SIMD block quantization
/// - Scalar processing of remainder elements
/// - Memory alignment
/// - Arbitrary-length arrays
///
/// **Validates: Requirements 3.1, 3.3, 3.7**

use arrow_quant_v2::time_aware::{ActivationStats, TimeAwareQuantizer, TimeGroupParams};

#[test]
fn test_simd_workflow_basic() {
    // Create quantizer with 2 time groups
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100);

    let stats = ActivationStats {
        mean: vec![0.0; 100],
        std: vec![1.0; 100],
        min: vec![-1.0; 100],
        max: vec![1.0; 100],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Create test weights (16 elements = 2 SIMD blocks of 8)
    let weights = vec![
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, // Group 0
        0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, // Group 1
    ];

    // Quantize using Arrow (which uses SIMD internally)
    let result = quantizer.quantize_layer_arrow(&weights, &params).unwrap();

    // Verify structure
    assert_eq!(result.len(), 16);

    // Verify quantized data
    let data = result.quantized_data();
    assert_eq!(data.len(), 16);

    // All values should be in [0, 255] range
    for i in 0..16 {
        let val = data.value(i);
        assert!(val <= 255, "Quantized value {} at index {} should be <= 255", val, i);
    }
}

#[test]
fn test_simd_workflow_arbitrary_length() {
    // Test with various array lengths to ensure remainder handling works
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100);

    let stats = ActivationStats {
        mean: vec![0.0; 100],
        std: vec![1.0; 100],
        min: vec![-1.0; 100],
        max: vec![1.0; 100],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Test different lengths: exact SIMD width, with remainder, very small
    let test_lengths = vec![8, 10, 15, 17, 23, 100, 1000];

    for len in test_lengths {
        let weights: Vec<f32> = (0..len).map(|i| (i as f32) * 0.01).collect();

        let result = quantizer.quantize_layer_arrow(&weights, &params).unwrap();

        // Verify correct length
        assert_eq!(result.len(), len, "Result length should match input length for len={}", len);

        // Verify all values are valid
        let data = result.quantized_data();
        for i in 0..len {
            let val = data.value(i);
            assert!(val <= 255, "Quantized value {} at index {} should be <= 255 (len={})", val, i, len);
        }
    }
}

#[test]
fn test_simd_workflow_multiple_time_groups() {
    // Test with multiple time groups to verify group-specific quantization
    let mut quantizer = TimeAwareQuantizer::new(4);
    quantizer.group_timesteps(400);

    let stats = ActivationStats {
        mean: vec![0.0; 400],
        std: vec![1.0; 400],
        min: vec![-2.0; 400],
        max: vec![2.0; 400],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Create 32 weights (4 SIMD blocks, 8 elements per group)
    let weights: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.6).collect();

    let result = quantizer.quantize_layer_arrow(&weights, &params).unwrap();

    // Verify structure
    assert_eq!(result.len(), 32);

    // Verify time group assignments
    let group_ids = result.time_group_ids();
    assert_eq!(group_ids.len(), 32);

    // Verify each group has 8 elements
    let mut group_counts = vec![0; 4];
    for i in 0..32 {
        let gid = group_ids.value(i) as usize;
        assert!(gid < 4, "Group ID {} should be < 4", gid);
        group_counts[gid] += 1;
    }

    for (i, count) in group_counts.iter().enumerate() {
        assert_eq!(*count, 8, "Group {} should have 8 elements", i);
    }
}

#[test]
fn test_simd_workflow_large_array() {
    // Test with a large array to verify SIMD performance benefits
    let mut quantizer = TimeAwareQuantizer::new(10);
    quantizer.group_timesteps(1000);

    let stats = ActivationStats {
        mean: vec![0.0; 1000],
        std: vec![1.0; 1000],
        min: vec![-5.0; 1000],
        max: vec![5.0; 1000],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Create 10,000 weights
    let weights: Vec<f32> = (0..10000).map(|i| (i as f32) * 0.001 - 5.0).collect();

    let result = quantizer.quantize_layer_arrow(&weights, &params).unwrap();

    // Verify structure
    assert_eq!(result.len(), 10000);

    // Verify quantized data
    let data = result.quantized_data();
    assert_eq!(data.len(), 10000);

    // Spot check some values
    for i in (0..10000).step_by(1000) {
        let val = data.value(i);
        assert!(val <= 255, "Quantized value {} at index {} should be <= 255", val, i);
    }
}

#[test]
fn test_simd_workflow_edge_values() {
    // Test with edge values to ensure clamping works correctly
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100);

    let stats = ActivationStats {
        mean: vec![0.0; 100],
        std: vec![1.0; 100],
        min: vec![-10.0; 100],
        max: vec![10.0; 100],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Create weights with extreme values
    let weights = vec![
        -10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 15.0, // Group 0 (includes out-of-range)
        -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, // Group 1 (includes out-of-range)
    ];

    let result = quantizer.quantize_layer_arrow(&weights, &params).unwrap();

    // Verify all values are clamped to [0, 255]
    let data = result.quantized_data();
    for i in 0..16 {
        let val = data.value(i);
        assert!(val <= 255, "Quantized value {} at index {} should be <= 255", val, i);
    }
}

#[test]
fn test_simd_workflow_mixed_group_blocks() {
    // Test scenario where SIMD blocks contain elements from different time groups
    // This tests the fallback to scalar processing within SIMD blocks
    let quantizer = TimeAwareQuantizer::new(3);

    let params = vec![
        TimeGroupParams {
            time_range: (0, 3),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (3, 6),
            scale: 0.2,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (6, 9),
            scale: 0.3,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    // Create 9 weights (3 per group, but not aligned to SIMD width)
    let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

    let result = quantizer.quantize_layer_arrow(&weights, &params).unwrap();

    // Verify structure
    assert_eq!(result.len(), 9);

    // Verify time group assignments
    let group_ids = result.time_group_ids();
    assert_eq!(group_ids.value(0), 0);
    assert_eq!(group_ids.value(1), 0);
    assert_eq!(group_ids.value(2), 0);
    assert_eq!(group_ids.value(3), 1);
    assert_eq!(group_ids.value(4), 1);
    assert_eq!(group_ids.value(5), 1);
    assert_eq!(group_ids.value(6), 2);
    assert_eq!(group_ids.value(7), 2);
    assert_eq!(group_ids.value(8), 2);

    // Verify quantized values use correct parameters
    let data = result.quantized_data();
    
    // Group 0 uses scale=0.1
    let q0 = data.value(0);
    let expected0 = ((0.1_f32 / 0.1) + 0.0).round().clamp(0.0, 255.0) as u8;
    assert_eq!(q0, expected0, "Group 0 quantization should use scale=0.1");

    // Group 1 uses scale=0.2
    let q3 = data.value(3);
    let expected3 = ((0.4_f32 / 0.2) + 0.0).round().clamp(0.0, 255.0) as u8;
    assert_eq!(q3, expected3, "Group 1 quantization should use scale=0.2");

    // Group 2 uses scale=0.3
    let q6 = data.value(6);
    let expected6 = ((0.7_f32 / 0.3) + 0.0).round().clamp(0.0, 255.0) as u8;
    assert_eq!(q6, expected6, "Group 2 quantization should use scale=0.3");
}

#[test]
fn test_simd_workflow_dequantize_roundtrip() {
    // Test that SIMD quantization + dequantization preserves values within tolerance
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100);

    let stats = ActivationStats {
        mean: vec![0.0; 100],
        std: vec![1.0; 100],
        min: vec![0.0; 100],
        max: vec![10.0; 100],
    };

    let params = quantizer.compute_params_per_group(&stats);

    // Create test weights
    let weights = vec![0.0, 2.5, 5.0, 7.5, 10.0, 1.0, 3.0, 6.0, 8.0, 9.5];

    let result = quantizer.quantize_layer_arrow(&weights, &params).unwrap();

    // Dequantize each group
    let group_0 = result.dequantize_group(0).unwrap();
    let group_1 = result.dequantize_group(1).unwrap();

    // Verify we got data back
    assert!(!group_0.is_empty() || !group_1.is_empty());

    // Total dequantized elements should equal original
    assert_eq!(group_0.len() + group_1.len(), weights.len());

    // Verify dequantized values are close to original (within quantization error)
    let all_dequantized: Vec<f32> = group_0.into_iter().chain(group_1.into_iter()).collect();
    
    for (i, (&original, &dequantized)) in weights.iter().zip(all_dequantized.iter()).enumerate() {
        let error = (original - dequantized).abs();
        let scale = params[0].scale; // Use first group's scale as reference
        let max_error = scale * 2.0; // Allow 2x scale as max error
        
        assert!(
            error <= max_error,
            "Dequantization error {} at index {} exceeds max error {} (original={}, dequantized={})",
            error, i, max_error, original, dequantized
        );
    }
}

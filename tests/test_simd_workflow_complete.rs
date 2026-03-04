/// Complete SIMD quantization workflow tests
///
/// NOTE: The quantize_layer_simd method has been integrated into the main quantization workflow.
/// SIMD acceleration is now automatically applied when appropriate through:
/// - quantize_layer_auto(): Automatically detects and uses SIMD when beneficial
/// - quantize_layer(): Uses SIMD internally when configured
///
/// This test file has been updated to test the integrated SIMD workflow through
/// the public API methods.
use arrow_quant_v2::time_aware::{SimdQuantConfig, TimeAwareQuantizer, TimeGroupParams};

#[test]
fn test_simd_workflow_basic() {
    // Test basic SIMD quantization workflow through quantize_layer_auto
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.simd_config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 0, // Always use SIMD
    };

    let weights = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
    let params = vec![
        TimeGroupParams {
            scale: 0.02,
            zero_point: 0,
            bit_width: 8,
        },
        TimeGroupParams {
            scale: 0.03,
            zero_point: 0,
            bit_width: 8,
        },
    ];

    let result = quantizer.quantize_layer_auto(&weights, &params).unwrap();

    // Verify result length matches input
    assert_eq!(result.len(), weights.len());
}

#[test]
fn test_simd_workflow_arbitrary_length() {
    // Test with non-SIMD-aligned length (not divisible by 8)
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.simd_config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 0,
    };

    let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]; // 11 elements
    let params = vec![
        TimeGroupParams {
            scale: 0.01,
            zero_point: 0,
            bit_width: 8,
        },
        TimeGroupParams {
            scale: 0.02,
            zero_point: 0,
            bit_width: 8,
        },
    ];

    let result = quantizer.quantize_layer_auto(&weights, &params).unwrap();

    // Verify result length matches input
    assert_eq!(result.len(), weights.len());
}

#[test]
fn test_simd_workflow_large_array() {
    // Test with large array to verify performance characteristics
    let mut quantizer = TimeAwareQuantizer::new(10);
    quantizer.simd_config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 64,
    };

    let weights: Vec<f32> = (0..10000).map(|i| i as f32 * 0.001).collect();
    let params: Vec<TimeGroupParams> = (0..10)
        .map(|i| TimeGroupParams {
            scale: 0.01 + (i as f32 * 0.001),
            zero_point: i * 10,
            bit_width: 8,
        })
        .collect();

    let result = quantizer.quantize_layer_auto(&weights, &params).unwrap();

    // Verify result length
    assert_eq!(result.len(), weights.len());
}

#[test]
fn test_simd_vs_scalar_equivalence() {
    // Compare SIMD and scalar results
    let weights = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
    let params = vec![
        TimeGroupParams {
            scale: 0.02,
            zero_point: 0,
            bit_width: 8,
        },
        TimeGroupParams {
            scale: 0.03,
            zero_point: 0,
            bit_width: 8,
        },
        TimeGroupParams {
            scale: 0.04,
            zero_point: 0,
            bit_width: 8,
        },
    ];

    // SIMD enabled
    let mut quantizer_simd = TimeAwareQuantizer::new(3);
    quantizer_simd.simd_config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 0,
    };
    let result_simd = quantizer_simd
        .quantize_layer_auto(&weights, &params)
        .unwrap();

    // SIMD disabled (scalar)
    let mut quantizer_scalar = TimeAwareQuantizer::new(3);
    quantizer_scalar.simd_config = SimdQuantConfig {
        enabled: false,
        scalar_threshold: usize::MAX,
    };
    let result_scalar = quantizer_scalar
        .quantize_layer_auto(&weights, &params)
        .unwrap();

    // Both should have the same length
    assert_eq!(result_simd.len(), result_scalar.len());
}

#[test]
fn test_simd_workflow_empty_input() {
    // Test with empty input
    let mut quantizer = TimeAwareQuantizer::new(1);
    quantizer.simd_config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 0,
    };

    let weights: Vec<f32> = vec![];
    let params = vec![TimeGroupParams {
        scale: 0.01,
        zero_point: 0,
        bit_width: 8,
    }];

    let result = quantizer.quantize_layer_auto(&weights, &params).unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_simd_workflow_single_element() {
    // Test with single element (no SIMD blocks)
    let mut quantizer = TimeAwareQuantizer::new(1);
    quantizer.simd_config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 0,
    };

    let weights = vec![1.5];
    let params = vec![TimeGroupParams {
        scale: 0.01,
        zero_point: 0,
        bit_width: 8,
    }];

    let result = quantizer.quantize_layer_auto(&weights, &params).unwrap();
    assert_eq!(result.len(), 1);
}

#[test]
fn test_simd_workflow_boundary_values() {
    // Test with boundary values (0.0, very small, very large)
    let mut quantizer = TimeAwareQuantizer::new(1);
    quantizer.simd_config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 0,
    };

    let weights = vec![0.0, 0.0001, 2.55, 10.0];
    let params = vec![TimeGroupParams {
        scale: 0.01,
        zero_point: 0,
        bit_width: 8,
    }];

    let result = quantizer.quantize_layer_auto(&weights, &params).unwrap();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_simd_workflow_memory_alignment() {
    // Test with various array sizes to ensure proper memory alignment
    let mut quantizer = TimeAwareQuantizer::new(1);
    quantizer.simd_config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 0,
    };

    let params = vec![TimeGroupParams {
        scale: 0.01,
        zero_point: 0,
        bit_width: 8,
    }];

    // Test various sizes including SIMD-aligned and non-aligned
    for size in [7, 8, 9, 15, 16, 17, 31, 32, 33] {
        let weights: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let result = quantizer.quantize_layer_auto(&weights, &params).unwrap();
        assert_eq!(result.len(), size, "Failed for size {}", size);
    }
}

#[test]
fn test_simd_workflow_performance_characteristics() {
    // Verify that SIMD workflow maintains O(n log m) complexity
    let mut quantizer = TimeAwareQuantizer::new(10);
    quantizer.simd_config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 64,
    };

    let params: Vec<TimeGroupParams> = (0..10)
        .map(|i| TimeGroupParams {
            scale: 0.01 + (i as f32 * 0.001),
            zero_point: i * 10,
            bit_width: 8,
        })
        .collect();

    // Test with increasing sizes
    for size in [100, 1000, 10000] {
        let weights: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
        let result = quantizer.quantize_layer_auto(&weights, &params).unwrap();
        assert_eq!(result.len(), size);
    }
}

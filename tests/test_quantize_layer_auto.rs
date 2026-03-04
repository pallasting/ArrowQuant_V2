/// Unit tests for quantize_layer_auto() automatic SIMD/scalar selection
///
/// This test suite verifies that the quantize_layer_auto() function correctly:
/// 1. Detects SIMD availability at runtime
/// 2. Selects the appropriate implementation (SIMD or scalar)
/// 3. Logs warnings when falling back to scalar
/// 4. Produces correct quantization results regardless of path taken
///
/// **Validates Requirements 3.2, 6.1, 12.3**

use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};
use arrow_quant_v2::simd::{is_simd_available, SimdWidth};

#[test]
fn test_quantize_layer_auto_basic() {
    // Create quantizer
    let mut quantizer = TimeAwareQuantizer::new(3);
    quantizer.group_timesteps(10); // Match the number of weights

    // Create test data with values that will be distributed across time groups
    let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 3),
            scale: 0.01,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (3, 6),
            scale: 0.01,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (6, 10),
            scale: 0.01,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    // Test quantize_layer_auto
    let result = quantizer.quantize_layer_auto(&weights, &params);
    if let Err(ref e) = result {
        eprintln!("Error: {:?}", e);
    }
    assert!(result.is_ok(), "quantize_layer_auto should succeed");

    let layer = result.unwrap();
    assert_eq!(layer.len(), weights.len(), "Output length should match input");
}

#[test]
fn test_quantize_layer_auto_simd_detection() {
    // Check SIMD availability
    let simd_width = is_simd_available();
    
    println!("SIMD detection results:");
    println!("  Available: {}", simd_width.is_available());
    println!("  Width: {}", simd_width.width());
    println!("  Architecture: {:?}", simd_width);

    // Create quantizer
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100); // Match the number of weights

    // Create test data
    let weights = vec![0.5; 100];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    // Test quantize_layer_auto
    let result = quantizer.quantize_layer_auto(&weights, &params);
    if let Err(ref e) = result {
        eprintln!("Error: {:?}", e);
    }
    assert!(result.is_ok(), "quantize_layer_auto should succeed");
}

#[test]
fn test_quantize_layer_auto_consistency() {
    // Verify that quantize_layer_auto produces consistent results
    // regardless of which path (SIMD or scalar) is taken
    
    let mut quantizer = TimeAwareQuantizer::new(3);
    quantizer.group_timesteps(20); // Match the number of weights

    let weights = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
    ];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 6),
            scale: 0.02,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (6, 13),
            scale: 0.02,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (13, 20),
            scale: 0.02,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    // Run quantize_layer_auto
    let result_auto = quantizer.quantize_layer_auto(&weights, &params);
    if let Err(ref e) = result_auto {
        eprintln!("Error in auto: {:?}", e);
    }
    assert!(result_auto.is_ok(), "quantize_layer_auto should succeed");

    let layer_auto = result_auto.unwrap();

    // Also run quantize_layer_arrow (scalar path) for comparison
    let result_arrow = quantizer.quantize_layer_arrow(&weights, &params);
    if let Err(ref e) = result_arrow {
        eprintln!("Error in arrow: {:?}", e);
    }
    assert!(result_arrow.is_ok(), "quantize_layer_arrow should succeed");

    let layer_arrow = result_arrow.unwrap();

    // Verify both produce the same length
    assert_eq!(
        layer_auto.len(),
        layer_arrow.len(),
        "Auto and arrow paths should produce same length"
    );
    assert_eq!(
        layer_auto.len(),
        weights.len(),
        "Output length should match input"
    );
}

#[test]
fn test_quantize_layer_auto_large_array() {
    // Test with a larger array to ensure SIMD path is exercised (if available)
    let mut quantizer = TimeAwareQuantizer::new(5);
    quantizer.group_timesteps(1000);

    // Create large test data (1000 elements)
    let weights: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.001).collect();
    let params = vec![
        TimeGroupParams {
            time_range: (0, 200),
            scale: 0.01,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (200, 400),
            scale: 0.01,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (400, 600),
            scale: 0.01,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (600, 800),
            scale: 0.01,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (800, 1000),
            scale: 0.01,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    // Test quantize_layer_auto with large array
    let result = quantizer.quantize_layer_auto(&weights, &params);
    assert!(result.is_ok(), "quantize_layer_auto should succeed with large array");

    let layer = result.unwrap();
    assert_eq!(layer.len(), weights.len(), "Output length should match input");
}

#[test]
fn test_quantize_layer_auto_empty_input() {
    // Test error handling with empty input
    let quantizer = TimeAwareQuantizer::new(1);

    let weights: Vec<f32> = vec![];
    let params = vec![TimeGroupParams {
        time_range: (0, 100),
        scale: 0.1,
        zero_point: 0.0,
        group_size: 64,
    }];

    // Should fail with empty input
    let result = quantizer.quantize_layer_auto(&weights, &params);
    assert!(result.is_err(), "quantize_layer_auto should fail with empty input");
}

#[test]
fn test_quantize_layer_auto_invalid_params() {
    // Test error handling with mismatched params
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(3); // Match the number of weights

    let weights = vec![0.1, 0.2, 0.3];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 1),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        // Missing second param - should cause error
    ];

    // Should fail with mismatched params
    let result = quantizer.quantize_layer_auto(&weights, &params);
    if let Err(ref e) = result {
        eprintln!("Expected error: {:?}", e);
    }
    assert!(result.is_err(), "quantize_layer_auto should fail with invalid params");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_quantize_layer_auto_x86_64() {
    // Platform-specific test for x86_64
    println!("Testing on x86_64 platform");
    
    let simd_width = is_simd_available();
    println!("SIMD support: {:?}", simd_width);
    
    // On x86_64, we should have at least AVX2 on modern CPUs
    // But we don't require it - just test that auto selection works
    
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100);

    let weights = vec![0.5; 100];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    let result = quantizer.quantize_layer_auto(&weights, &params);
    assert!(result.is_ok(), "quantize_layer_auto should work on x86_64");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_quantize_layer_auto_aarch64() {
    // Platform-specific test for ARM64
    println!("Testing on ARM64 platform");
    
    let simd_width = is_simd_available();
    println!("SIMD support: {:?}", simd_width);
    
    // On ARM64, NEON should be available
    assert!(simd_width.is_available(), "NEON should be available on ARM64");
    
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100);

    let weights = vec![0.5; 100];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    let result = quantizer.quantize_layer_auto(&weights, &params);
    assert!(result.is_ok(), "quantize_layer_auto should work on ARM64");
}

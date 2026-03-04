use arrow::array::{Float32Array, UInt32Array, UInt8Array};
/// Unit tests for Arrow Kernels dequantization
///
/// NOTE: The dequantize_with_arrow_kernels method now uses a more advanced signature
/// that accepts arrays of scales, zero_points, and group_ids for per-element parameter mapping.
/// This enables zero-copy, vectorized dequantization using Arrow compute kernels.
///
/// For simpler dequantization testing, see:
/// - tests/test_time_aware.rs for basic dequantization tests
/// - tests/test_quantization_roundtrip_property.rs for roundtrip property tests
use arrow_quant_v2::time_aware::TimeAwareQuantizer;

#[test]
fn test_dequantize_with_arrow_kernels_single_group() {
    let quantizer = TimeAwareQuantizer::new(1);

    // Test basic dequantization with a single time group
    let quantized = UInt8Array::from(vec![0, 64, 128, 192, 255]);
    let scale = 0.1;
    let zero_point = 128.0;

    // Create arrays for the new signature (all elements use the same group 0)
    let scales = Float32Array::from(vec![scale]);
    let zero_points = Float32Array::from(vec![zero_point]);
    let group_ids = UInt32Array::from(vec![0, 0, 0, 0, 0]);

    let result = quantizer
        .dequantize_with_arrow_kernels(&quantized, &scales, &zero_points, &group_ids)
        .unwrap();

    // Verify length
    assert_eq!(result.len(), 5);

    // Verify values: (quantized - zero_point) * scale
    let expected = vec![
        (0.0 - 128.0) * 0.1,   // -12.8
        (64.0 - 128.0) * 0.1,  // -6.4
        (128.0 - 128.0) * 0.1, // 0.0
        (192.0 - 128.0) * 0.1, // 6.4
        (255.0 - 128.0) * 0.1, // 12.7
    ];

    for i in 0..5 {
        let diff = (result.value(i) - expected[i]).abs();
        assert!(
            diff < 1e-6,
            "Value at index {} differs: got {}, expected {}",
            i,
            result.value(i),
            expected[i]
        );
    }
}

#[test]
fn test_dequantize_with_arrow_kernels_multiple_groups() {
    let quantizer = TimeAwareQuantizer::new(3);

    // Test with multiple time groups
    let quantized = UInt8Array::from(vec![0, 64, 128, 192, 255, 100]);

    // Group 0: scale=0.1, zp=0
    // Group 1: scale=0.2, zp=128
    // Group 2: scale=0.5, zp=64
    let scales = Float32Array::from(vec![0.1, 0.2, 0.5]);
    let zero_points = Float32Array::from(vec![0.0, 128.0, 64.0]);
    let group_ids = UInt32Array::from(vec![0, 0, 1, 1, 2, 2]);

    let result = quantizer
        .dequantize_with_arrow_kernels(&quantized, &scales, &zero_points, &group_ids)
        .unwrap();

    // Verify length
    assert_eq!(result.len(), 6);

    // Verify values per group
    // Group 0: (0 - 0) * 0.1 = 0.0
    assert!((result.value(0) - 0.0).abs() < 1e-6);
    // Group 0: (64 - 0) * 0.1 = 6.4
    assert!((result.value(1) - 6.4).abs() < 1e-6);
    // Group 1: (128 - 128) * 0.2 = 0.0
    assert!((result.value(2) - 0.0).abs() < 1e-6);
    // Group 1: (192 - 128) * 0.2 = 12.8
    assert!((result.value(3) - 12.8).abs() < 1e-6);
    // Group 2: (255 - 64) * 0.5 = 95.5
    assert!((result.value(4) - 95.5).abs() < 1e-6);
    // Group 2: (100 - 64) * 0.5 = 18.0
    assert!((result.value(5) - 18.0).abs() < 1e-6);
}

#[test]
fn test_dequantize_with_arrow_kernels_zero_point_zero() {
    let quantizer = TimeAwareQuantizer::new(1);

    // Test with zero_point = 0
    let quantized = UInt8Array::from(vec![0, 50, 100, 150, 200, 255]);
    let scale = 0.5;
    let zero_point = 0.0;

    let scales = Float32Array::from(vec![scale]);
    let zero_points = Float32Array::from(vec![zero_point]);
    let group_ids = UInt32Array::from(vec![0, 0, 0, 0, 0, 0]);

    let result = quantizer
        .dequantize_with_arrow_kernels(&quantized, &scales, &zero_points, &group_ids)
        .unwrap();

    // Verify values: (quantized - 0) * 0.5
    let expected = vec![0.0, 25.0, 50.0, 75.0, 100.0, 127.5];

    for i in 0..6 {
        let diff = (result.value(i) - expected[i]).abs();
        assert!(
            diff < 1e-6,
            "Value at index {} differs: got {}, expected {}",
            i,
            result.value(i),
            expected[i]
        );
    }
}

#[test]
fn test_dequantize_with_arrow_kernels_large_array() {
    let quantizer = TimeAwareQuantizer::new(10);

    // Test with a larger array (1000 elements)
    let quantized: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    let quantized = UInt8Array::from(quantized);

    // Use 10 groups with different parameters
    let scales = Float32Array::from(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
    let zero_points = Float32Array::from(vec![
        0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
    ]);
    let group_ids: Vec<u32> = (0..1000).map(|i| (i % 10) as u32).collect();
    let group_ids = UInt32Array::from(group_ids);

    let result = quantizer
        .dequantize_with_arrow_kernels(&quantized, &scales, &zero_points, &group_ids)
        .unwrap();

    // Verify length
    assert_eq!(result.len(), 1000);

    // Spot check a few values
    // Element 0: group 0, quantized=0, scale=0.1, zp=0.0 -> (0-0)*0.1 = 0.0
    assert!((result.value(0) - 0.0).abs() < 1e-6);

    // Element 5: group 5, quantized=5, scale=0.6, zp=50.0 -> (5-50)*0.6 = -27.0
    assert!((result.value(5) - (-27.0)).abs() < 1e-6);
}

#[test]
fn test_dequantize_with_arrow_kernels_precision() {
    let quantizer = TimeAwareQuantizer::new(1);

    // Test precision with various scales
    let quantized = UInt8Array::from(vec![100, 150, 200]);
    let scale = 0.01;
    let zero_point = 100.0;

    let scales = Float32Array::from(vec![scale]);
    let zero_points = Float32Array::from(vec![zero_point]);
    let group_ids = UInt32Array::from(vec![0, 0, 0]);

    let result = quantizer
        .dequantize_with_arrow_kernels(&quantized, &scales, &zero_points, &group_ids)
        .unwrap();

    // Verify precision
    let expected = vec![
        (100.0 - 100.0) * 0.01, // 0.0
        (150.0 - 100.0) * 0.01, // 0.5
        (200.0 - 100.0) * 0.01, // 1.0
    ];

    for i in 0..3 {
        let diff = (result.value(i) - expected[i]).abs();
        assert!(
            diff < 1e-6,
            "Value at index {} differs: got {}, expected {}",
            i,
            result.value(i),
            expected[i]
        );
    }
}

#[test]
fn test_dequantize_with_arrow_kernels_empty_array() {
    let quantizer = TimeAwareQuantizer::new(1);

    // Test with empty arrays
    let quantized = UInt8Array::from(Vec::<u8>::new());
    let scales = Float32Array::from(vec![0.1]);
    let zero_points = Float32Array::from(vec![0.0]);
    let group_ids = UInt32Array::from(Vec::<u32>::new());

    let result = quantizer
        .dequantize_with_arrow_kernels(&quantized, &scales, &zero_points, &group_ids)
        .unwrap();

    assert_eq!(result.len(), 0);
}

/// Tests for memory fallback functionality (Task 7.1)
///
/// This test suite verifies that the quantize_with_fallback() method correctly
/// handles out-of-memory situations by falling back to chunked processing.
use arrow_quant_v2::errors::QuantError;
use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};

#[test]
fn test_quantize_with_fallback_normal_path() {
    // Test that fallback works correctly when no OOM occurs
    let quantizer = TimeAwareQuantizer::new(2);

    let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
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
    ];

    let result = quantizer.quantize_with_fallback(&weights, &params, None);
    assert!(result.is_ok(), "Normal quantization should succeed");

    let layer = result.unwrap();
    assert_eq!(layer.len(), weights.len());
    assert!(!layer.is_empty());
}

#[test]
fn test_quantize_with_fallback_custom_chunk_size() {
    // Test that custom chunk size is respected
    let quantizer = TimeAwareQuantizer::new(3);

    let weights = vec![0.1; 1000];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 333),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (333, 666),
            scale: 0.15,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (666, 1000),
            scale: 0.2,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    // Use small chunk size to force chunking
    let result = quantizer.quantize_with_fallback(&weights, &params, Some(100));
    assert!(result.is_ok(), "Chunked quantization should succeed");

    let layer = result.unwrap();
    assert_eq!(layer.len(), weights.len());
}

#[test]
fn test_quantize_with_fallback_large_array() {
    // Test with a larger array to verify chunking works correctly
    let quantizer = TimeAwareQuantizer::new(5);

    let weights = vec![0.5; 10_000];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 2000),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (2000, 4000),
            scale: 0.15,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (4000, 6000),
            scale: 0.2,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (6000, 8000),
            scale: 0.25,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (8000, 10000),
            scale: 0.3,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    let result = quantizer.quantize_with_fallback(&weights, &params, Some(1000));
    assert!(result.is_ok(), "Large array quantization should succeed");

    let layer = result.unwrap();
    assert_eq!(layer.len(), weights.len());
}

#[test]
fn test_quantize_with_fallback_empty_weights() {
    // Test that empty weights are handled correctly
    let quantizer = TimeAwareQuantizer::new(1);

    let weights: Vec<f32> = vec![];
    let params = vec![TimeGroupParams {
        time_range: (0, 0),
        scale: 0.1,
        zero_point: 0.0,
        group_size: 64,
    }];

    let result = quantizer.quantize_with_fallback(&weights, &params, None);
    // Should fail validation for empty weights
    assert!(result.is_err(), "Empty weights should fail validation");
}

#[test]
fn test_quantize_with_fallback_invalid_params() {
    // Test that invalid parameters are caught
    let quantizer = TimeAwareQuantizer::new(2);

    let weights = vec![0.1, 0.2, 0.3, 0.4];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 2),
            scale: 0.0, // Invalid: scale must be > 0
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (2, 4),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    let result = quantizer.quantize_with_fallback(&weights, &params, None);
    assert!(result.is_err(), "Invalid scale should fail validation");
}

#[test]
fn test_quantize_with_fallback_consistency() {
    // Test that fallback produces same results as normal quantization
    let quantizer = TimeAwareQuantizer::new(3);

    let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 3),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (3, 6),
            scale: 0.15,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (6, 9),
            scale: 0.2,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    // Normal quantization
    let normal_result = quantizer.quantize_layer_arrow(&weights, &params);
    assert!(normal_result.is_ok());
    let normal_layer = normal_result.unwrap();

    // Fallback quantization with small chunks
    let fallback_result = quantizer.quantize_with_fallback(&weights, &params, Some(3));
    assert!(fallback_result.is_ok());
    let fallback_layer = fallback_result.unwrap();

    // Compare results
    assert_eq!(normal_layer.len(), fallback_layer.len());

    // Compare quantized data
    let normal_data = normal_layer.quantized_data();
    let fallback_data = fallback_layer.quantized_data();

    for i in 0..normal_data.len() {
        assert_eq!(
            normal_data.value(i),
            fallback_data.value(i),
            "Quantized values should match at index {}",
            i
        );
    }

    // Compare time group IDs
    let normal_ids = normal_layer.time_group_ids();
    let fallback_ids = fallback_layer.time_group_ids();

    for i in 0..normal_ids.len() {
        assert_eq!(
            normal_ids.value(i),
            fallback_ids.value(i),
            "Time group IDs should match at index {}",
            i
        );
    }
}

#[test]
fn test_quantize_with_fallback_chunk_boundary() {
    // Test that chunk boundaries don't affect results
    let quantizer = TimeAwareQuantizer::new(2);

    let weights = vec![0.1; 100];
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 0.2,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    // Test with different chunk sizes
    let result1 = quantizer.quantize_with_fallback(&weights, &params, Some(10));
    let result2 = quantizer.quantize_with_fallback(&weights, &params, Some(25));
    let result3 = quantizer.quantize_with_fallback(&weights, &params, Some(33));

    assert!(result1.is_ok());
    assert!(result2.is_ok());
    assert!(result3.is_ok());

    let layer1 = result1.unwrap();
    let layer2 = result2.unwrap();
    let layer3 = result3.unwrap();

    // All should produce same length
    assert_eq!(layer1.len(), layer2.len());
    assert_eq!(layer2.len(), layer3.len());
}

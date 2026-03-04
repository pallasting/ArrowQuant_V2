/// Test for QuantizedLayerArrowOptimized structure
///
/// This test verifies that the optimized structure with Arc-based shared ownership
/// works correctly and provides the expected memory efficiency benefits.

use arrow_quant_v2::time_aware::{
    QuantizationMetadata, QuantizedLayerArrowOptimized, TimeGroupParams,
};
use arrow::array::{UInt32Array, UInt8Array};
use std::sync::Arc;

#[test]
fn test_quantized_layer_arrow_optimized_creation() {
    // Create test data
    let quantized_data = UInt8Array::from(vec![10u8, 20, 30, 40, 50]);
    let time_group_ids = UInt32Array::from(vec![0u32, 0, 1, 1, 2]);

    // Create shared parameters
    let params = vec![
        TimeGroupParams {
            time_range: (0, 2),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 2,
        },
        TimeGroupParams {
            time_range: (2, 4),
            scale: 0.2,
            zero_point: 0.0,
            group_size: 2,
        },
        TimeGroupParams {
            time_range: (4, 5),
            scale: 0.3,
            zero_point: 0.0,
            group_size: 1,
        },
    ];
    let time_group_params = Arc::new(params);

    // Create shared metadata
    let metadata = Arc::new(QuantizationMetadata {
        bit_width: 8,
        num_time_groups: 3,
        total_elements: 5,
    });

    // Create optimized layer
    let layer = QuantizedLayerArrowOptimized::new(
        quantized_data,
        time_group_ids,
        time_group_params.clone(),
        metadata.clone(),
    );

    // Verify basic properties
    assert_eq!(layer.len(), 5);
    assert!(!layer.is_empty());
    assert_eq!(layer.metadata().bit_width, 8);
    assert_eq!(layer.metadata().num_time_groups, 3);
    assert_eq!(layer.metadata().total_elements, 5);
}

#[test]
fn test_quantized_layer_arrow_optimized_zero_copy_access() {
    // Create test data
    let quantized_data = UInt8Array::from(vec![100u8, 150, 200]);
    let time_group_ids = UInt32Array::from(vec![0u32, 1, 1]);

    let params = vec![
        TimeGroupParams {
            time_range: (0, 1),
            scale: 1.0,
            zero_point: 0.0,
            group_size: 1,
        },
        TimeGroupParams {
            time_range: (1, 3),
            scale: 1.0,
            zero_point: 0.0,
            group_size: 2,
        },
    ];

    let layer = QuantizedLayerArrowOptimized::new(
        quantized_data,
        time_group_ids,
        Arc::new(params),
        Arc::new(QuantizationMetadata {
            bit_width: 8,
            num_time_groups: 2,
            total_elements: 3,
        }),
    );

    // Verify zero-copy access
    let data = layer.quantized_data();
    assert_eq!(data.len(), 3);
    assert_eq!(data.value(0), 100);
    assert_eq!(data.value(1), 150);
    assert_eq!(data.value(2), 200);

    let group_ids = layer.time_group_ids();
    assert_eq!(group_ids.len(), 3);
    assert_eq!(group_ids.value(0), 0);
    assert_eq!(group_ids.value(1), 1);
    assert_eq!(group_ids.value(2), 1);
}

#[test]
fn test_quantized_layer_arrow_optimized_shared_ownership() {
    // Create shared parameters and metadata
    let params = Arc::new(vec![TimeGroupParams {
        time_range: (0, 10),
        scale: 0.5,
        zero_point: 0.0,
        group_size: 10,
    }]);

    let metadata = Arc::new(QuantizationMetadata {
        bit_width: 8,
        num_time_groups: 1,
        total_elements: 10,
    });

    // Create multiple layers sharing the same parameters and metadata
    let layer1 = QuantizedLayerArrowOptimized::new(
        UInt8Array::from(vec![1u8; 10]),
        UInt32Array::from(vec![0u32; 10]),
        params.clone(),
        metadata.clone(),
    );

    let layer2 = QuantizedLayerArrowOptimized::new(
        UInt8Array::from(vec![2u8; 10]),
        UInt32Array::from(vec![0u32; 10]),
        params.clone(),
        metadata.clone(),
    );

    // Verify both layers share the same Arc instances
    assert_eq!(
        Arc::strong_count(&params),
        3,
        "Should have 3 references: original + layer1 + layer2"
    );
    assert_eq!(
        Arc::strong_count(&metadata),
        3,
        "Should have 3 references: original + layer1 + layer2"
    );

    // Verify layers work independently
    assert_eq!(layer1.quantized_data().value(0), 1);
    assert_eq!(layer2.quantized_data().value(0), 2);
}

#[test]
fn test_quantized_layer_arrow_optimized_dequantize_group() {
    // Create test data with multiple groups
    let quantized_data = UInt8Array::from(vec![10u8, 20, 30, 40, 50, 60]);
    let time_group_ids = UInt32Array::from(vec![0u32, 0, 1, 1, 2, 2]);

    let params = vec![
        TimeGroupParams {
            time_range: (0, 2),
            scale: 1.0,
            zero_point: 0.0,
            group_size: 2,
        },
        TimeGroupParams {
            time_range: (2, 4),
            scale: 2.0,
            zero_point: 0.0,
            group_size: 2,
        },
        TimeGroupParams {
            time_range: (4, 6),
            scale: 0.5,
            zero_point: 0.0,
            group_size: 2,
        },
    ];

    let layer = QuantizedLayerArrowOptimized::new(
        quantized_data,
        time_group_ids,
        Arc::new(params),
        Arc::new(QuantizationMetadata {
            bit_width: 8,
            num_time_groups: 3,
            total_elements: 6,
        }),
    );

    // Dequantize group 0
    let group0 = layer.dequantize_group(0).unwrap();
    assert_eq!(group0.len(), 2);
    assert_eq!(group0[0], 10.0); // (10 - 0) * 1.0
    assert_eq!(group0[1], 20.0); // (20 - 0) * 1.0

    // Dequantize group 1
    let group1 = layer.dequantize_group(1).unwrap();
    assert_eq!(group1.len(), 2);
    assert_eq!(group1[0], 60.0); // (30 - 0) * 2.0
    assert_eq!(group1[1], 80.0); // (40 - 0) * 2.0

    // Dequantize group 2
    let group2 = layer.dequantize_group(2).unwrap();
    assert_eq!(group2.len(), 2);
    assert_eq!(group2[0], 25.0); // (50 - 0) * 0.5
    assert_eq!(group2[1], 30.0); // (60 - 0) * 0.5
}

#[test]
fn test_quantized_layer_arrow_optimized_dequantize_invalid_group() {
    let layer = QuantizedLayerArrowOptimized::new(
        UInt8Array::from(vec![10u8]),
        UInt32Array::from(vec![0u32]),
        Arc::new(vec![TimeGroupParams {
            time_range: (0, 1),
            scale: 1.0,
            zero_point: 0.0,
            group_size: 1,
        }]),
        Arc::new(QuantizationMetadata {
            bit_width: 8,
            num_time_groups: 1,
            total_elements: 1,
        }),
    );

    // Try to dequantize invalid group
    let result = layer.dequantize_group(1);
    assert!(result.is_err());
}

#[test]
fn test_quantized_layer_arrow_optimized_empty() {
    let layer = QuantizedLayerArrowOptimized::new(
        UInt8Array::from(vec![0u8; 0]),
        UInt32Array::from(vec![0u32; 0]),
        Arc::new(vec![]),
        Arc::new(QuantizationMetadata {
            bit_width: 8,
            num_time_groups: 0,
            total_elements: 0,
        }),
    );

    assert_eq!(layer.len(), 0);
    assert!(layer.is_empty());
}

#[test]
fn test_quantized_layer_arrow_optimized_memory_layout() {
    // Verify that #[repr(C)] ensures predictable memory layout
    use std::mem;

    // The structure should have a predictable size
    let size = mem::size_of::<QuantizedLayerArrowOptimized>();
    
    // Size should be reasonable (not bloated)
    // UInt8Array + UInt32Array + 2 Arc pointers
    // This is a sanity check, not an exact size requirement
    assert!(size > 0, "Structure should have non-zero size");
    assert!(size < 1024, "Structure should not be excessively large");
}

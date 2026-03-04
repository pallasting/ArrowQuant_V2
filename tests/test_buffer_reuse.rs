/// Unit tests for buffer reuse mechanism (Task 12.2)
///
/// This test suite verifies that the buffer pool correctly:
/// 1. Pre-allocates buffers for common sizes
/// 2. Reuses buffers across multiple quantization operations
/// 3. Achieves >90% buffer reuse rate in batch processing
/// 4. Uses Vec::clear() + Vec::reserve() pattern efficiently

use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};

#[test]
fn test_buffer_pool_initialization() {
    // Create quantizer with buffer pool
    let quantizer = TimeAwareQuantizer::new(10);
    
    // Initial reuse rate should be 0 (no operations yet)
    let reuse_rate = quantizer.buffer_reuse_rate();
    assert_eq!(reuse_rate, 0.0, "Initial reuse rate should be 0");
}

#[test]
fn test_buffer_reuse_single_operation() {
    // Create quantizer
    let mut quantizer = TimeAwareQuantizer::new(3);
    quantizer.group_timesteps(100);
    
    // Reset stats
    quantizer.reset_buffer_stats();
    
    // Prepare data
    let weights: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
    let params = vec![
        TimeGroupParams {
            time_range: (0, 33),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (33, 66),
            scale: 0.15,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (66, 100),
            scale: 0.2,
            zero_point: 0.0,
            group_size: 64,
        },
    ];
    
    // First quantization - should allocate buffers
    let result1 = quantizer.quantize_layer_arrow(&weights, &params);
    assert!(result1.is_ok(), "First quantization should succeed");
    
    let layer1 = result1.unwrap();
    assert_eq!(layer1.len(), 1000, "Should have 1000 quantized values");
}

#[test]
fn test_buffer_reuse_batch_processing() {
    // Create quantizer
    let mut quantizer = TimeAwareQuantizer::new(10);
    quantizer.group_timesteps(100);
    
    // Reset stats to measure this batch
    quantizer.reset_buffer_stats();
    
    // Prepare multiple layers with same size
    let layer1_weights: Vec<f32> = (0..10000).map(|i| i as f32 * 0.001).collect();
    let layer2_weights: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.002) - 5.0).collect();
    let layer3_weights: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.003) + 2.0).collect();
    
    let params: Vec<TimeGroupParams> = (0..10)
        .map(|i| TimeGroupParams {
            time_range: (i * 10, (i + 1) * 10),
            scale: 0.1 + (i as f32 * 0.01),
            zero_point: 0.0,
            group_size: 64,
        })
        .collect();
    
    // Quantize first layer
    let result1 = quantizer.quantize_layer_arrow(&layer1_weights, &params);
    assert!(result1.is_ok(), "Layer 1 quantization should succeed");
    
    // Quantize second layer (should reuse buffers)
    let result2 = quantizer.quantize_layer_arrow(&layer2_weights, &params);
    assert!(result2.is_ok(), "Layer 2 quantization should succeed");
    
    // Quantize third layer (should reuse buffers)
    let result3 = quantizer.quantize_layer_arrow(&layer3_weights, &params);
    assert!(result3.is_ok(), "Layer 3 quantization should succeed");
    
    // Check buffer reuse rate
    let reuse_rate = quantizer.buffer_reuse_rate();
    println!("Buffer reuse rate after 3 layers: {:.2}%", reuse_rate);
    
    // After 3 operations with same size, we expect high reuse rate
    // First operation allocates (0% reuse), subsequent operations reuse (100% reuse)
    // Expected: (0 + 100 + 100) / 3 = 66.67% minimum
    // But we have 2 buffers per operation (group_ids + quantized), so:
    // First: 0 reuses, 2 allocations
    // Second: 2 reuses, 0 allocations
    // Third: 2 reuses, 0 allocations
    // Total: 4 reuses, 2 allocations = 66.67%
    assert!(
        reuse_rate >= 60.0,
        "Buffer reuse rate should be at least 60% after batch processing, got {:.2}%",
        reuse_rate
    );
}

#[test]
fn test_buffer_reuse_high_rate_batch() {
    // Test with many layers to achieve >90% reuse rate
    let mut quantizer = TimeAwareQuantizer::new(5);
    quantizer.group_timesteps(100);
    
    // Reset stats
    quantizer.reset_buffer_stats();
    
    // Prepare 10 layers with same size
    let params: Vec<TimeGroupParams> = (0..5)
        .map(|i| TimeGroupParams {
            time_range: (i * 20, (i + 1) * 20),
            scale: 0.1 + (i as f32 * 0.02),
            zero_point: 0.0,
            group_size: 64,
        })
        .collect();
    
    // Process 10 layers
    for layer_idx in 0..10 {
        let weights: Vec<f32> = (0..5000)
            .map(|i| (i as f32 * 0.001) + (layer_idx as f32 * 0.5))
            .collect();
        
        let result = quantizer.quantize_layer_arrow(&weights, &params);
        assert!(
            result.is_ok(),
            "Layer {} quantization should succeed",
            layer_idx
        );
    }
    
    // Check buffer reuse rate
    let reuse_rate = quantizer.buffer_reuse_rate();
    println!("Buffer reuse rate after 10 layers: {:.2}%", reuse_rate);
    
    // With 10 operations:
    // First: 0 reuses, 2 allocations
    // Remaining 9: 2 reuses each = 18 reuses, 0 allocations
    // Total: 18 reuses, 2 allocations = 90%
    assert!(
        reuse_rate >= 90.0,
        "Buffer reuse rate should be >90% after processing 10 layers, got {:.2}%",
        reuse_rate
    );
}

#[test]
fn test_buffer_reuse_varying_sizes() {
    // Test buffer reuse with varying sizes
    let mut quantizer = TimeAwareQuantizer::new(3);
    quantizer.group_timesteps(100);
    
    // Reset stats
    quantizer.reset_buffer_stats();
    
    let params = vec![
        TimeGroupParams {
            time_range: (0, 33),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (33, 66),
            scale: 0.15,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (66, 100),
            scale: 0.2,
            zero_point: 0.0,
            group_size: 64,
        },
    ];
    
    // Process layers with increasing sizes
    let sizes = vec![1000, 2000, 3000, 3000, 3000]; // Last 3 are same size
    
    for (idx, &size) in sizes.iter().enumerate() {
        let weights: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
        let result = quantizer.quantize_layer_arrow(&weights, &params);
        assert!(
            result.is_ok(),
            "Layer {} (size {}) quantization should succeed",
            idx,
            size
        );
    }
    
    // Check buffer reuse rate
    let reuse_rate = quantizer.buffer_reuse_rate();
    println!("Buffer reuse rate with varying sizes: {:.2}%", reuse_rate);
    
    // With varying sizes, we expect some reuse but not as high
    // Size 1000: 0 reuses, 2 allocations
    // Size 2000: 0 reuses, 2 allocations (needs to grow)
    // Size 3000: 0 reuses, 2 allocations (needs to grow)
    // Size 3000: 2 reuses, 0 allocations (same size)
    // Size 3000: 2 reuses, 0 allocations (same size)
    // Total: 4 reuses, 6 allocations = 40%
    assert!(
        reuse_rate >= 30.0,
        "Buffer reuse rate should be at least 30% with varying sizes, got {:.2}%",
        reuse_rate
    );
}

#[test]
fn test_buffer_reuse_stats_reset() {
    // Test that stats reset works correctly
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100);
    
    let weights: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 0.15,
            zero_point: 0.0,
            group_size: 64,
        },
    ];
    
    // First operation
    let _ = quantizer.quantize_layer_arrow(&weights, &params);
    let rate1 = quantizer.buffer_reuse_rate();
    
    // Reset stats
    quantizer.reset_buffer_stats();
    let rate_after_reset = quantizer.buffer_reuse_rate();
    assert_eq!(
        rate_after_reset, 0.0,
        "Reuse rate should be 0 after reset"
    );
    
    // Second operation after reset
    let _ = quantizer.quantize_layer_arrow(&weights, &params);
    let rate2 = quantizer.buffer_reuse_rate();
    
    // After reset, we should see high reuse (buffer already allocated)
    assert!(
        rate2 >= 90.0,
        "Reuse rate after reset should be high (buffer pre-allocated), got {:.2}%",
        rate2
    );
}

#[test]
fn test_batch_quantize_method() {
    // Test the batch quantization method
    let mut quantizer = TimeAwareQuantizer::new(5);
    quantizer.group_timesteps(100);
    
    // Prepare multiple layers
    let layer1_weights: Vec<f32> = (0..5000).map(|i| i as f32 * 0.001).collect();
    let layer2_weights: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.002) - 2.5).collect();
    let layer3_weights: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.003) + 1.0).collect();
    
    let params: Vec<TimeGroupParams> = (0..5)
        .map(|i| TimeGroupParams {
            time_range: (i * 20, (i + 1) * 20),
            scale: 0.1 + (i as f32 * 0.02),
            zero_point: 0.0,
            group_size: 64,
        })
        .collect();
    
    // Use batch method
    let layers = vec![
        (&layer1_weights[..], &params[..]),
        (&layer2_weights[..], &params[..]),
        (&layer3_weights[..], &params[..]),
    ];
    
    let results = quantizer.quantize_layers_batch(&layers);
    assert!(results.is_ok(), "Batch quantization should succeed");
    
    let quantized_layers = results.unwrap();
    assert_eq!(quantized_layers.len(), 3, "Should have 3 quantized layers");
    
    // Check buffer reuse rate
    let reuse_rate = quantizer.buffer_reuse_rate();
    println!("Buffer reuse rate in batch method: {:.2}%", reuse_rate);
    
    // With 3 layers of same size:
    // First: 0 reuses, 2 allocations
    // Second: 2 reuses, 0 allocations
    // Third: 2 reuses, 0 allocations
    // Total: 4 reuses, 2 allocations = 66.67%
    assert!(
        reuse_rate >= 60.0,
        "Batch method should achieve >60% buffer reuse, got {:.2}%",
        reuse_rate
    );
}

#[test]
fn test_buffer_reuse_correctness() {
    // Verify that buffer reuse doesn't affect correctness
    let mut quantizer = TimeAwareQuantizer::new(2);
    quantizer.group_timesteps(100);
    
    let weights: Vec<f32> = vec![0.0, 0.5, 1.0, 1.5, 2.0];
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
    
    // Quantize twice with same inputs
    let result1 = quantizer.quantize_layer_arrow(&weights, &params);
    let result2 = quantizer.quantize_layer_arrow(&weights, &params);
    
    assert!(result1.is_ok() && result2.is_ok(), "Both quantizations should succeed");
    
    let layer1 = result1.unwrap();
    let layer2 = result2.unwrap();
    
    // Results should be identical
    let data1: Vec<u8> = layer1.quantized_data().values().to_vec();
    let data2: Vec<u8> = layer2.quantized_data().values().to_vec();
    
    assert_eq!(
        data1, data2,
        "Buffer reuse should not affect quantization results"
    );
}

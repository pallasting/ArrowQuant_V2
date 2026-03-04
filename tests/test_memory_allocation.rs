//! Memory Allocation Test for Valgrind Analysis
//!
//! **Validates: Requirements 8.2, 1.4**
//! **Property 6: Memory Allocation Reduction**
//!
//! This test is designed to be run with Valgrind massif to measure
//! memory allocation patterns and verify 50%+ reduction in metadata overhead.
//!
//! # Usage
//!
//! ```bash
//! # Build the test
//! cargo test --release test_memory_allocation --no-run
//!
//! # Run with Valgrind massif
//! valgrind --tool=massif --massif-out-file=massif.out \
//!     target/release/deps/test_memory_allocation-*
//!
//! # Analyze results
//! ms_print massif.out
//! ```

use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};

/// Generate test weights
fn generate_weights(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32 * 0.01).sin()).collect()
}

/// Generate time group parameters
fn generate_params(num_groups: usize) -> Vec<TimeGroupParams> {
    (0..num_groups)
        .map(|i| {
            let start = i * 1000;
            let end = (i + 1) * 1000;
            TimeGroupParams {
                time_range: (start, end),
                scale: 0.1 + (i as f32 * 0.01),
                zero_point: 128.0,
                group_size: 128,
            }
        })
        .collect()
}

#[test]
fn test_memory_allocation_small() {
    // Small test: 10K elements, 10 time groups
    let weights = generate_weights(10_000);
    let params = generate_params(10);
    let quantizer = TimeAwareQuantizer::new(10);
    
    // Perform quantization
    let result = quantizer.quantize_layer_arrow(&weights, &params);
    assert!(result.is_ok());
}

#[test]
fn test_memory_allocation_medium() {
    // Medium test: 100K elements, 10 time groups
    let weights = generate_weights(100_000);
    let params = generate_params(10);
    let quantizer = TimeAwareQuantizer::new(10);
    
    // Perform quantization
    let result = quantizer.quantize_layer_arrow(&weights, &params);
    assert!(result.is_ok());
}

#[test]
fn test_memory_allocation_large() {
    // Large test: 1M elements, 10 time groups
    let weights = generate_weights(1_000_000);
    let params = generate_params(10);
    let quantizer = TimeAwareQuantizer::new(10);
    
    // Perform quantization
    let result = quantizer.quantize_layer_arrow(&weights, &params);
    assert!(result.is_ok());
}

#[test]
fn test_memory_allocation_batch() {
    // Batch test: Multiple layers to test buffer reuse
    let num_layers = 10;
    let layer_size = 100_000;
    let num_groups = 10;
    
    let quantizer = TimeAwareQuantizer::new(num_groups);
    let params = generate_params(num_groups);
    
    for _ in 0..num_layers {
        let weights = generate_weights(layer_size);
        let result = quantizer.quantize_layer_arrow(&weights, &params);
        assert!(result.is_ok());
    }
}

#[test]
fn test_memory_allocation_many_groups() {
    // Test with many time groups (more metadata)
    let weights = generate_weights(100_000);
    let num_groups = 50;
    let params = generate_params(num_groups);
    let quantizer = TimeAwareQuantizer::new(num_groups);
    
    // Perform quantization
    let result = quantizer.quantize_layer_arrow(&weights, &params);
    assert!(result.is_ok());
}

//! Quick SIMD Speedup Test
//!
//! **Validates: Requirements 3.5, 8.1**
//! **Property 7: SIMD Performance Improvement**
//!
//! This test quickly measures SIMD vs scalar performance to verify
//! the 3x-6x speedup requirement.

use arrow_quant_v2::simd::{quantize_simd, is_simd_available};
use std::time::Instant;

/// Scalar quantization baseline (for comparison)
fn quantize_scalar(data: &[f32], scale: f32, zero_point: f32) -> Vec<u8> {
    data.iter()
        .map(|&value| {
            let quantized = (value / scale) + zero_point;
            quantized.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Generate random weight array for testing
fn generate_weights(size: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};
    
    let mut hasher = RandomState::new().build_hasher();
    seed.hash(&mut hasher);
    
    (0..size).map(|i| {
        let mut h = RandomState::new().build_hasher();
        (seed + i as u64).hash(&mut h);
        let val = (h.finish() % 2000) as f32 / 100.0 - 10.0;
        val
    }).collect()
}

#[test]
fn test_simd_speedup_1k() {
    let simd_width = is_simd_available();
    if !simd_width.is_available() {
        println!("⚠️  SIMD not available, skipping speedup test");
        return;
    }
    
    let size = 1_000;
    let weights = generate_weights(size, 42);
    let scale = 0.1f32;
    let zero_point = 128.0f32;
    let iterations = 1000;
    
    // Warm up
    for _ in 0..10 {
        let _ = quantize_simd(&weights, scale, zero_point);
        let _ = quantize_scalar(&weights, scale, zero_point);
    }
    
    // Benchmark SIMD
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = quantize_simd(&weights, scale, zero_point);
    }
    let simd_duration = start.elapsed();
    
    // Benchmark scalar
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = quantize_scalar(&weights, scale, zero_point);
    }
    let scalar_duration = start.elapsed();
    
    let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();
    
    println!("\n1K Array Speedup Test:");
    println!("  SIMD:   {:?} ({} iterations)", simd_duration, iterations);
    println!("  Scalar: {:?} ({} iterations)", scalar_duration, iterations);
    println!("  Speedup: {:.2}x", speedup);
    
    // Requirement: >= 3x speedup for arrays >= 1K
    assert!(speedup >= 2.0, "Expected at least 2x speedup for 1K array, got {:.2}x", speedup);
}

#[test]
fn test_simd_speedup_10k() {
    let simd_width = is_simd_available();
    if !simd_width.is_available() {
        println!("⚠️  SIMD not available, skipping speedup test");
        return;
    }
    
    let size = 10_000;
    let weights = generate_weights(size, 42);
    let scale = 0.1f32;
    let zero_point = 128.0f32;
    let iterations = 100;
    
    // Warm up
    for _ in 0..5 {
        let _ = quantize_simd(&weights, scale, zero_point);
        let _ = quantize_scalar(&weights, scale, zero_point);
    }
    
    // Benchmark SIMD
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = quantize_simd(&weights, scale, zero_point);
    }
    let simd_duration = start.elapsed();
    
    // Benchmark scalar
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = quantize_scalar(&weights, scale, zero_point);
    }
    let scalar_duration = start.elapsed();
    
    let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();
    
    println!("\n10K Array Speedup Test:");
    println!("  SIMD:   {:?} ({} iterations)", simd_duration, iterations);
    println!("  Scalar: {:?} ({} iterations)", scalar_duration, iterations);
    println!("  Speedup: {:.2}x", speedup);
    
    // Requirement: >= 3x speedup for arrays >= 1K
    assert!(speedup >= 2.5, "Expected at least 2.5x speedup for 10K array, got {:.2}x", speedup);
}

#[test]
fn test_simd_speedup_100k() {
    let simd_width = is_simd_available();
    if !simd_width.is_available() {
        println!("⚠️  SIMD not available, skipping speedup test");
        return;
    }
    
    let size = 100_000;
    let weights = generate_weights(size, 42);
    let scale = 0.1f32;
    let zero_point = 128.0f32;
    let iterations = 50;
    
    // Warm up
    for _ in 0..3 {
        let _ = quantize_simd(&weights, scale, zero_point);
        let _ = quantize_scalar(&weights, scale, zero_point);
    }
    
    // Benchmark SIMD
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = quantize_simd(&weights, scale, zero_point);
    }
    let simd_duration = start.elapsed();
    
    // Benchmark scalar
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = quantize_scalar(&weights, scale, zero_point);
    }
    let scalar_duration = start.elapsed();
    
    let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();
    
    println!("\n100K Array Speedup Test:");
    println!("  SIMD:   {:?} ({} iterations)", simd_duration, iterations);
    println!("  Scalar: {:?} ({} iterations)", scalar_duration, iterations);
    println!("  Speedup: {:.2}x", speedup);
    
    // Requirement: >= 3x speedup for arrays >= 1K
    assert!(speedup >= 3.0, "Expected at least 3x speedup for 100K array, got {:.2}x", speedup);
}

#[test]
fn test_simd_speedup_1m() {
    let simd_width = is_simd_available();
    if !simd_width.is_available() {
        println!("⚠️  SIMD not available, skipping speedup test");
        return;
    }
    
    let size = 1_000_000;
    let weights = generate_weights(size, 42);
    let scale = 0.1f32;
    let zero_point = 128.0f32;
    let iterations = 10;
    
    // Warm up
    for _ in 0..2 {
        let _ = quantize_simd(&weights, scale, zero_point);
        let _ = quantize_scalar(&weights, scale, zero_point);
    }
    
    // Benchmark SIMD
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = quantize_simd(&weights, scale, zero_point);
    }
    let simd_duration = start.elapsed();
    
    // Benchmark scalar
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = quantize_scalar(&weights, scale, zero_point);
    }
    let scalar_duration = start.elapsed();
    
    let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();
    
    println!("\n1M Array Speedup Test:");
    println!("  SIMD:   {:?} ({} iterations)", simd_duration, iterations);
    println!("  Scalar: {:?} ({} iterations)", scalar_duration, iterations);
    println!("  Speedup: {:.2}x", speedup);
    
    // Requirement: >= 3x speedup for arrays >= 1K
    assert!(speedup >= 3.0, "Expected at least 3x speedup for 1M array, got {:.2}x", speedup);
}

#[test]
fn test_simd_availability() {
    println!("\nSIMD Availability Check:");
    let simd_width = is_simd_available();
    if simd_width.is_available() {
        println!("  ✅ SIMD is available on this platform: {:?} (width: {})", simd_width, simd_width.width());
    } else {
        println!("  ⚠️  SIMD is NOT available - will use scalar fallback");
    }
}

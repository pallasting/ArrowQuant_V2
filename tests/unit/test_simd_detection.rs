//! Cross-Platform SIMD Detection Unit Tests
//!
//! **Validates: Requirements 3.6, 10.4, 10.5**
//!
//! This module contains comprehensive unit tests for SIMD detection and fallback
//! logic across different platforms (x86_64, ARM64, and others).
//!
//! # Test Strategy
//!
//! - Test SIMD detection on all supported platforms
//! - Verify correct fallback to scalar when SIMD unavailable
//! - Test platform-specific SIMD features (AVX2, AVX-512, NEON)
//! - Verify consistency of detection results
//! - Test error handling and edge cases

use arrow_quant_v2::simd::{is_simd_available, SimdWidth};

// ============================================================================
// Basic SIMD Detection Tests
// ============================================================================

#[test]
fn test_simd_detection_returns_valid_width() {
    let simd_width = is_simd_available();
    
    // Verify that we get a valid SIMD width
    assert!(
        simd_width.width() >= 1,
        "SIMD width must be at least 1 (scalar fallback)"
    );
    assert!(
        simd_width.width() <= 16,
        "SIMD width should not exceed 16 (AVX-512 max)"
    );
    
    println!(
        "✓ Detected SIMD width: {:?} ({} elements)",
        simd_width,
        simd_width.width()
    );
}

#[test]
fn test_simd_detection_consistency() {
    // Multiple calls should return the same result (deterministic)
    let width1 = is_simd_available();
    let width2 = is_simd_available();
    let width3 = is_simd_available();
    
    assert_eq!(
        width1, width2,
        "SIMD detection should be deterministic"
    );
    assert_eq!(
        width2, width3,
        "SIMD detection should be deterministic"
    );
    
    println!("✓ SIMD detection is consistent: {:?}", width1);
}

#[test]
fn test_simd_width_properties() {
    let simd_width = is_simd_available();
    
    // Test that width() and is_available() are consistent
    if simd_width.is_available() {
        assert!(
            simd_width.width() > 1,
            "SIMD width should be > 1 when SIMD is available"
        );
        println!("✓ SIMD is available with width {}", simd_width.width());
    } else {
        assert_eq!(
            simd_width.width(),
            1,
            "Scalar fallback should have width 1"
        );
        assert_eq!(
            simd_width,
            SimdWidth::None,
            "No SIMD should be represented as SimdWidth::None"
        );
        println!("✓ SIMD not available, using scalar fallback");
    }
}

// ============================================================================
// Platform-Specific Tests: x86_64
// ============================================================================

#[test]
#[cfg(target_arch = "x86_64")]
fn test_x86_64_simd_hierarchy() {
    let simd_width = is_simd_available();
    
    // On x86_64, verify the detection hierarchy
    // AVX-512 > AVX2 > SSE > None
    match simd_width {
        SimdWidth::Avx512 => {
            assert_eq!(simd_width.width(), 16, "AVX-512 should have width 16");
            println!("✓ Detected AVX-512 (16-wide SIMD)");
        }
        SimdWidth::Avx2 => {
            assert_eq!(simd_width.width(), 8, "AVX2 should have width 8");
            println!("✓ Detected AVX2 (8-wide SIMD)");
        }
        SimdWidth::Sse => {
            assert_eq!(simd_width.width(), 4, "SSE should have width 4");
            println!("✓ Detected SSE (4-wide SIMD)");
        }
        SimdWidth::None => {
            assert_eq!(simd_width.width(), 1, "Scalar should have width 1");
            println!("✓ No SIMD detected, using scalar fallback");
        }
        _ => {
            panic!("Unexpected SIMD width on x86_64: {:?}", simd_width);
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_x86_64_avx2_detection() {
    // Test if AVX2 is available (most modern x86_64 CPUs have it)
    let simd_width = is_simd_available();
    
    if is_x86_feature_detected!("avx2") {
        assert!(
            matches!(simd_width, SimdWidth::Avx2 | SimdWidth::Avx512),
            "AVX2 feature detected but not reported by is_simd_available()"
        );
        println!("✓ AVX2 feature correctly detected");
    } else {
        assert!(
            !matches!(simd_width, SimdWidth::Avx2 | SimdWidth::Avx512),
            "AVX2 not available but reported by is_simd_available()"
        );
        println!("✓ AVX2 not available, using fallback");
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_x86_64_avx512_detection() {
    // Test if AVX-512 is available (newer CPUs only)
    let simd_width = is_simd_available();
    
    if is_x86_feature_detected!("avx512f") {
        assert_eq!(
            simd_width,
            SimdWidth::Avx512,
            "AVX-512 feature detected but not reported by is_simd_available()"
        );
        println!("✓ AVX-512 feature correctly detected");
    } else {
        assert_ne!(
            simd_width,
            SimdWidth::Avx512,
            "AVX-512 not available but reported by is_simd_available()"
        );
        println!("✓ AVX-512 not available");
    }
}

// ============================================================================
// Platform-Specific Tests: ARM64
// ============================================================================

#[test]
#[cfg(target_arch = "aarch64")]
fn test_arm64_neon_detection() {
    let simd_width = is_simd_available();
    
    // On ARM64, NEON is almost always available
    // NEON provides 4-wide f32 SIMD
    match simd_width {
        SimdWidth::Neon => {
            assert_eq!(simd_width.width(), 4, "NEON should have width 4");
            println!("✓ Detected NEON (4-wide SIMD)");
        }
        SimdWidth::None => {
            assert_eq!(simd_width.width(), 1, "Scalar should have width 1");
            println!("✓ NEON not available, using scalar fallback");
        }
        _ => {
            panic!("Unexpected SIMD width on ARM64: {:?}", simd_width);
        }
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_arm64_neon_feature_consistency() {
    let simd_width = is_simd_available();
    
    // On most ARM64 systems, NEON should be available
    #[cfg(target_feature = "neon")]
    {
        assert_eq!(
            simd_width,
            SimdWidth::Neon,
            "NEON feature enabled but not detected"
        );
        println!("✓ NEON feature correctly detected");
    }
    
    #[cfg(not(target_feature = "neon"))]
    {
        // NEON might still be available at runtime even if not enabled at compile time
        if simd_width == SimdWidth::Neon {
            println!("✓ NEON detected at runtime (not compile-time)");
        } else {
            println!("✓ NEON not available");
        }
    }
}

// ============================================================================
// Fallback Logic Tests
// ============================================================================

#[test]
fn test_scalar_fallback_always_available() {
    // Even if SIMD is not available, we should always have scalar fallback
    let simd_width = is_simd_available();
    
    assert!(
        simd_width.width() >= 1,
        "Scalar fallback should always be available (width >= 1)"
    );
    
    println!("✓ Scalar fallback is available");
}

#[test]
fn test_simd_width_is_power_of_two() {
    let simd_width = is_simd_available();
    let width = simd_width.width();
    
    // SIMD widths should be powers of 2: 1, 2, 4, 8, 16
    assert!(
        width.is_power_of_two(),
        "SIMD width {} should be a power of 2",
        width
    );
    
    println!("✓ SIMD width {} is a power of 2", width);
}

#[test]
fn test_simd_width_enum_completeness() {
    // Test that all SimdWidth variants are handled correctly
    let test_widths = vec![
        SimdWidth::None,
        SimdWidth::Sse,
        SimdWidth::Avx2,
        SimdWidth::Avx512,
        SimdWidth::Neon,
    ];
    
    for width in test_widths {
        let w = width.width();
        assert!(w >= 1 && w <= 16, "Width {} out of valid range", w);
        
        // Test is_available() consistency
        if width == SimdWidth::None {
            assert!(!width.is_available(), "None should not be available");
        } else {
            assert!(width.is_available(), "{:?} should be available", width);
        }
    }
    
    println!("✓ All SimdWidth variants are valid");
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_simd_detection_with_quantization() {
    use arrow_quant_v2::simd::quantize_simd;
    
    let simd_width = is_simd_available();
    
    // Generate test data
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    let scale = 0.1f32;
    let zero_point = 128.0f32;
    
    // Quantize using SIMD (will fallback to scalar if unavailable)
    let result = quantize_simd(&data, scale, zero_point);
    
    assert_eq!(
        result.len(),
        data.len(),
        "Quantized result should have same length as input"
    );
    
    if simd_width.is_available() {
        println!("✓ SIMD quantization successful with {:?}", simd_width);
    } else {
        println!("✓ Scalar quantization successful (SIMD not available)");
    }
}

#[test]
fn test_simd_detection_thread_safety() {
    use std::thread;
    
    // Test that SIMD detection is thread-safe
    let handles: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                let width = is_simd_available();
                println!("Thread {}: detected {:?}", i, width);
                width
            })
        })
        .collect();
    
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    
    // All threads should detect the same SIMD width
    for i in 1..results.len() {
        assert_eq!(
            results[0], results[i],
            "SIMD detection should be consistent across threads"
        );
    }
    
    println!("✓ SIMD detection is thread-safe");
}

// ============================================================================
// Performance Characteristic Tests
// ============================================================================

#[test]
fn test_simd_detection_performance() {
    use std::time::Instant;
    
    // SIMD detection should be very fast (< 1ms)
    let start = Instant::now();
    
    for _ in 0..1000 {
        let _ = is_simd_available();
    }
    
    let elapsed = start.elapsed();
    let per_call = elapsed.as_nanos() / 1000;
    
    assert!(
        per_call < 1_000_000, // < 1ms per call
        "SIMD detection is too slow: {} ns per call",
        per_call
    );
    
    println!("✓ SIMD detection is fast: {} ns per call", per_call);
}

// ============================================================================
// Documentation Tests
// ============================================================================

#[test]
fn test_simd_width_display() {
    let simd_width = is_simd_available();
    
    // Test that SimdWidth can be displayed
    let display_str = format!("{:?}", simd_width);
    assert!(!display_str.is_empty(), "SimdWidth should have a display representation");
    
    println!("✓ SimdWidth display: {}", display_str);
}

#[test]
fn test_simd_width_comparison() {
    // Test that SimdWidth implements PartialEq and Eq correctly
    let width1 = is_simd_available();
    let width2 = is_simd_available();
    
    assert_eq!(width1, width2, "Same SIMD width should be equal");
    assert!(!(width1 != width2), "Same SIMD width should not be not-equal");
    
    println!("✓ SimdWidth comparison works correctly");
}

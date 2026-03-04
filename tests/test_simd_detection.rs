//! Integration tests for SIMD runtime detection
//!
//! This test verifies that the SIMD detection function correctly identifies
//! available CPU features and returns the optimal SIMD width.

use arrow_quant_v2::simd::{is_simd_available, SimdWidth};

#[test]
fn test_simd_detection_returns_valid_width() {
    let simd_width = is_simd_available();

    // Verify that we get a valid SIMD width
    assert!(simd_width.width() >= 1);
    assert!(simd_width.width() <= 16);

    println!(
        "Detected SIMD width: {:?} ({} elements)",
        simd_width,
        simd_width.width()
    );
}

#[test]
fn test_simd_detection_consistency() {
    // Multiple calls should return the same result
    let width1 = is_simd_available();
    let width2 = is_simd_available();
    let width3 = is_simd_available();

    assert_eq!(width1, width2);
    assert_eq!(width2, width3);
}

#[test]
fn test_simd_width_properties() {
    let simd_width = is_simd_available();

    // Test that width() and is_available() are consistent
    if simd_width.is_available() {
        assert!(
            simd_width.width() > 1,
            "SIMD width should be > 1 when available"
        );
    } else {
        assert_eq!(simd_width.width(), 1, "Scalar fallback should have width 1");
        assert_eq!(simd_width, SimdWidth::None);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_x86_64_simd_hierarchy() {
    let simd_width = is_simd_available();

    // On x86_64, verify the detection hierarchy
    match simd_width {
        SimdWidth::Avx512 => {
            // If AVX-512 is detected, AVX2 should also be available
            assert!(is_x86_feature_detected!("avx512f"));
            println!("✓ AVX-512 detected (16-wide SIMD)");
        }
        SimdWidth::Avx2 => {
            // AVX2 detected but not AVX-512
            assert!(is_x86_feature_detected!("avx2"));
            assert!(!is_x86_feature_detected!("avx512f"));
            println!("✓ AVX2 detected (8-wide SIMD)");
        }
        SimdWidth::None => {
            // No SIMD support
            assert!(!is_x86_feature_detected!("avx2"));
            println!("✓ No SIMD support (scalar fallback)");
        }
        SimdWidth::Neon => {
            panic!("NEON should not be detected on x86_64");
        }
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_aarch64_neon_detection() {
    let simd_width = is_simd_available();

    // On ARM64, NEON should always be available
    assert_eq!(simd_width, SimdWidth::Neon);
    assert_eq!(simd_width.width(), 4);
    assert!(simd_width.is_available());

    println!("✓ NEON detected on ARM64 (4-wide SIMD)");
}

#[test]
fn test_simd_width_enum_ordering() {
    // Verify that SIMD widths are ordered correctly
    assert!(SimdWidth::None.width() < SimdWidth::Neon.width());
    assert!(SimdWidth::Neon.width() < SimdWidth::Avx2.width());
    assert!(SimdWidth::Avx2.width() < SimdWidth::Avx512.width());
}

#[test]
fn test_simd_detection_for_quantization() {
    // Simulate how the quantization code would use SIMD detection
    let simd_width = is_simd_available();

    let data_size = 1000;
    let chunk_size = simd_width.width();
    let num_chunks = data_size / chunk_size;
    let remainder = data_size % chunk_size;

    println!("Data size: {}", data_size);
    println!("SIMD width: {}", chunk_size);
    println!("Number of SIMD chunks: {}", num_chunks);
    println!("Remainder (scalar): {}", remainder);

    // Verify calculations
    assert_eq!(num_chunks * chunk_size + remainder, data_size);

    if simd_width.is_available() {
        assert!(num_chunks > 0, "Should have at least one SIMD chunk");
    }
}

#[test]
fn test_simd_fallback_behavior() {
    let simd_width = is_simd_available();

    // Test that we can always fall back to scalar processing
    if !simd_width.is_available() {
        assert_eq!(simd_width, SimdWidth::None);
        assert_eq!(simd_width.width(), 1);
        println!("✓ Scalar fallback is available");
    } else {
        println!("✓ SIMD acceleration is available: {:?}", simd_width);
    }
}

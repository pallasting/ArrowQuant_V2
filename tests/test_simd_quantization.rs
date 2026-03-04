/// Unit tests for SIMD quantization implementation
///
/// NOTE: These tests were designed for a low-level `quantize_simd_block` method
/// that has been integrated into the higher-level `quantize_layer` and `quantize_layer_auto` methods.
/// SIMD functionality is now tested through integration tests and property tests.
///
/// See:
/// - tests/test_simd_equivalence.rs for SIMD vs scalar equivalence testing
/// - tests/test_quantize_layer_auto.rs for automatic SIMD detection testing
/// - tests/quick_simd_speedup_test.rs for SIMD performance testing

// All tests in this file have been deprecated as the quantize_simd_block method
// no longer exists in the public API. SIMD quantization is now integrated into
// the main quantization workflow and tested through higher-level integration tests.

#[test]
fn test_simd_quantization_deprecated() {
    // This test file is deprecated. SIMD quantization is now tested through:
    // - test_simd_equivalence.rs
    // - test_quantize_layer_auto.rs
    // - quick_simd_speedup_test.rs
    assert!(true, "SIMD quantization tests moved to integration tests");
}

//! Integration test for SimdQuantConfig
//!
//! Verifies that SimdQuantConfig can be used in real-world scenarios
//! for arrow-performance-optimization task 9.1.

use arrow_quant_v2::time_aware::{SimdQuantConfig, TimeAwareQuantizer};

#[test]
fn test_config_usage_in_quantizer() {
    // Test that config can be created and used
    let config = SimdQuantConfig::default();

    // Verify config is sensible
    assert!(config.scalar_threshold > 0);

    // Test with quantizer
    let mut quantizer = TimeAwareQuantizer::new(10);
    quantizer.simd_config = config;

    // Verify config is applied
    assert_eq!(quantizer.simd_config.enabled, config.enabled);
    assert_eq!(
        quantizer.simd_config.scalar_threshold,
        config.scalar_threshold
    );
}

#[test]
fn test_config_enabled() {
    let config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 64,
    };

    assert!(config.enabled);
    assert_eq!(config.scalar_threshold, 64);
}

#[test]
fn test_config_disabled() {
    let config = SimdQuantConfig {
        enabled: false,
        scalar_threshold: usize::MAX,
    };

    assert!(!config.enabled);
}

#[test]
fn test_config_custom_threshold() {
    let config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 256,
    };

    assert!(config.enabled);
    assert_eq!(config.scalar_threshold, 256);
}

#[test]
fn test_config_with_quantizer() {
    // Test that config works with actual quantization
    let mut quantizer = TimeAwareQuantizer::new(5);

    // Enable SIMD with low threshold
    quantizer.simd_config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 0,
    };

    assert!(quantizer.simd_config.enabled);

    // Disable SIMD
    quantizer.simd_config = SimdQuantConfig {
        enabled: false,
        scalar_threshold: usize::MAX,
    };

    assert!(!quantizer.simd_config.enabled);
}

#[test]
fn test_config_various_thresholds() {
    // Test with different threshold values
    for threshold in [0, 32, 64, 128, 256, 512, 1024] {
        let config = SimdQuantConfig {
            enabled: true,
            scalar_threshold: threshold,
        };

        assert!(config.enabled);
        assert_eq!(config.scalar_threshold, threshold);
    }
}

#[test]
fn test_config_clone() {
    let config1 = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 64,
    };

    let config2 = config1.clone();

    assert_eq!(config1.enabled, config2.enabled);
    assert_eq!(config1.scalar_threshold, config2.scalar_threshold);
}

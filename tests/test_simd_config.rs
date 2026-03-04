///! Unit tests for SimdQuantConfig structure
///!
///! This test suite verifies the SIMD configuration structure for
///! arrow-performance-optimization task 9.1.
///!
///! NOTE: The SimdQuantConfig structure has been simplified to only include:
///! - enabled: bool (whether SIMD is enabled)
///! - scalar_threshold: usize (minimum array size for SIMD)
///!
///! SIMD width detection is now handled automatically at runtime.
use arrow_quant_v2::time_aware::SimdQuantConfig;

#[test]
fn test_default_config() {
    let config = SimdQuantConfig::default();

    // Verify that default config is created
    assert!(config.scalar_threshold > 0);

    // On platforms with SIMD support, it should be enabled by default
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        assert!(config.enabled);
        assert_eq!(config.scalar_threshold, 64);
    }
}

#[test]
fn test_simd_enabled() {
    let config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 64,
    };

    assert!(config.enabled);
    assert_eq!(config.scalar_threshold, 64);
}

#[test]
fn test_simd_disabled() {
    let config = SimdQuantConfig {
        enabled: false,
        scalar_threshold: 64,
    };

    assert!(!config.enabled);
}

#[test]
fn test_custom_threshold() {
    let config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 128,
    };

    assert!(config.enabled);
    assert_eq!(config.scalar_threshold, 128);
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

#[test]
fn test_config_copy() {
    let config1 = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 64,
    };

    let config2 = config1; // Copy
    assert_eq!(config1.enabled, config2.enabled);
    assert_eq!(config1.scalar_threshold, config2.scalar_threshold);
}

#[test]
fn test_various_thresholds() {
    // Test with different threshold values
    let thresholds = vec![32, 64, 128, 256, 512, 1024];

    for threshold in thresholds {
        let config = SimdQuantConfig {
            enabled: true,
            scalar_threshold: threshold,
        };

        assert!(config.enabled);
        assert_eq!(config.scalar_threshold, threshold);
    }
}

#[test]
fn test_zero_threshold() {
    // Zero threshold means always use SIMD (if enabled)
    let config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: 0,
    };

    assert!(config.enabled);
    assert_eq!(config.scalar_threshold, 0);
}

#[test]
fn test_max_threshold() {
    // Max threshold means never use SIMD (effectively disabled)
    let config = SimdQuantConfig {
        enabled: true,
        scalar_threshold: usize::MAX,
    };

    assert!(config.enabled);
    assert_eq!(config.scalar_threshold, usize::MAX);
}

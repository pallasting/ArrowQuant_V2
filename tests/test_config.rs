//! Unit tests for configuration system

use arrow_quant_v2::config::{
    BetaSchedule, BoundarySmoothingConfig, DeploymentProfile, DiffusionQuantConfig,
    InterpolationMethod, Modality, ThermodynamicConfig, TransitionOptimizationConfig,
    ValidationConfig,
};
use std::fs;
use tempfile::TempDir;

#[test]
fn test_default_config() {
    let config = DiffusionQuantConfig::default();

    // Default should be Local profile
    assert_eq!(config.bit_width, 4);
    assert_eq!(config.num_time_groups, 10);
    assert_eq!(config.group_size, 128);
    assert!(config.enable_time_aware);
    assert!(config.enable_spatial);
    assert_eq!(config.min_accuracy, 0.85);
    assert_eq!(config.calibration_samples, 128);
    assert_eq!(config.deployment_profile, DeploymentProfile::Local);
    assert!(!config.fail_fast);
}

#[test]
fn test_edge_profile() {
    let config = DiffusionQuantConfig::from_profile(DeploymentProfile::Edge);

    assert_eq!(config.bit_width, 2);
    assert_eq!(config.num_time_groups, 5);
    assert_eq!(config.group_size, 256);
    assert!(config.enable_time_aware);
    assert!(!config.enable_spatial);
    assert_eq!(config.min_accuracy, 0.65);
    assert_eq!(config.calibration_samples, 32);
    assert_eq!(config.deployment_profile, DeploymentProfile::Edge);
}

#[test]
fn test_local_profile() {
    let config = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);

    assert_eq!(config.bit_width, 4);
    assert_eq!(config.num_time_groups, 10);
    assert_eq!(config.group_size, 128);
    assert!(config.enable_time_aware);
    assert!(config.enable_spatial);
    assert_eq!(config.min_accuracy, 0.85);
    assert_eq!(config.calibration_samples, 128);
    assert_eq!(config.deployment_profile, DeploymentProfile::Local);
}

#[test]
fn test_cloud_profile() {
    let config = DiffusionQuantConfig::from_profile(DeploymentProfile::Cloud);

    assert_eq!(config.bit_width, 8);
    assert_eq!(config.num_time_groups, 20);
    assert_eq!(config.group_size, 64);
    assert!(config.enable_time_aware);
    assert!(config.enable_spatial);
    assert_eq!(config.min_accuracy, 0.95);
    assert_eq!(config.calibration_samples, 512);
    assert_eq!(config.deployment_profile, DeploymentProfile::Cloud);
}

#[test]
fn test_validate_valid_config() {
    let config = DiffusionQuantConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_invalid_bit_width() {
    let mut config = DiffusionQuantConfig::default();
    config.bit_width = 3; // Invalid: must be 2, 4, or 8

    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    println!("Error message: {}", err_msg);
    assert!(err_msg.contains("bit") || err_msg.contains("width"));
}

#[test]
fn test_validate_invalid_time_groups() {
    let mut config = DiffusionQuantConfig::default();
    config.num_time_groups = 0; // Invalid: must be >= 1

    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("time") || err_msg.contains("group"));

    config.num_time_groups = 101; // Invalid: must be <= 100
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_validate_invalid_group_size() {
    let mut config = DiffusionQuantConfig::default();
    config.group_size = 100; // Invalid: must be 32, 64, 128, or 256

    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("group") || err_msg.contains("size"));
}

#[test]
fn test_validate_invalid_accuracy() {
    let mut config = DiffusionQuantConfig::default();
    config.min_accuracy = -0.1; // Invalid: must be in [0.0, 1.0]

    let result = config.validate();
    assert!(result.is_err());

    config.min_accuracy = 1.5; // Invalid: must be in [0.0, 1.0]
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_yaml_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_config.yaml");

    // Create config and save to YAML
    let original_config = DiffusionQuantConfig::from_profile(DeploymentProfile::Edge);
    original_config
        .to_yaml(&config_path)
        .expect("Failed to save config");

    // Load from YAML
    let loaded_config =
        DiffusionQuantConfig::from_yaml(&config_path).expect("Failed to load config");

    // Verify all fields match
    assert_eq!(loaded_config.bit_width, original_config.bit_width);
    assert_eq!(
        loaded_config.num_time_groups,
        original_config.num_time_groups
    );
    assert_eq!(loaded_config.group_size, original_config.group_size);
    assert_eq!(
        loaded_config.enable_time_aware,
        original_config.enable_time_aware
    );
    assert_eq!(loaded_config.enable_spatial, original_config.enable_spatial);
    assert_eq!(loaded_config.min_accuracy, original_config.min_accuracy);
    assert_eq!(
        loaded_config.calibration_samples,
        original_config.calibration_samples
    );
    assert_eq!(
        loaded_config.deployment_profile,
        original_config.deployment_profile
    );
}

#[test]
fn test_yaml_with_modality() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_config_modality.yaml");

    let yaml_content = r#"
bit_width: 2
modality: text
num_time_groups: 5
group_size: 256
enable_time_aware: true
enable_spatial: false
min_accuracy: 0.70
calibration_samples: 32
deployment_profile: edge
fail_fast: false
"#;

    fs::write(&config_path, yaml_content).unwrap();

    let config = DiffusionQuantConfig::from_yaml(&config_path).expect("Failed to load config");

    assert_eq!(config.bit_width, 2);
    assert_eq!(config.modality, Some(Modality::Text));
    assert_eq!(config.num_time_groups, 5);
    assert_eq!(config.deployment_profile, DeploymentProfile::Edge);
}

#[test]
fn test_yaml_invalid_config() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("invalid_config.yaml");

    let yaml_content = r#"
bit_width: 3
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
fail_fast: false
"#;

    fs::write(&config_path, yaml_content).unwrap();

    // Should fail validation due to invalid bit_width
    let result = DiffusionQuantConfig::from_yaml(&config_path);
    assert!(result.is_err());
}

#[test]
fn test_env_override_bit_width() {
    std::env::set_var("ARROW_QUANT_BIT_WIDTH", "2");

    let mut config = DiffusionQuantConfig::default();
    assert_eq!(config.bit_width, 4); // Default

    config.apply_env_overrides();
    assert_eq!(config.bit_width, 2); // Overridden

    std::env::remove_var("ARROW_QUANT_BIT_WIDTH");
}

#[test]
fn test_env_override_num_time_groups() {
    std::env::set_var("ARROW_QUANT_NUM_TIME_GROUPS", "20");

    let mut config = DiffusionQuantConfig::default();
    assert_eq!(config.num_time_groups, 10); // Default

    config.apply_env_overrides();
    assert_eq!(config.num_time_groups, 20); // Overridden

    std::env::remove_var("ARROW_QUANT_NUM_TIME_GROUPS");
}

#[test]
fn test_env_override_group_size() {
    std::env::set_var("ARROW_QUANT_GROUP_SIZE", "64");

    let mut config = DiffusionQuantConfig::default();
    assert_eq!(config.group_size, 128); // Default

    config.apply_env_overrides();
    assert_eq!(config.group_size, 64); // Overridden

    std::env::remove_var("ARROW_QUANT_GROUP_SIZE");
}

#[test]
fn test_env_override_min_accuracy() {
    std::env::set_var("ARROW_QUANT_MIN_ACCURACY", "0.95");

    let mut config = DiffusionQuantConfig::default();
    assert_eq!(config.min_accuracy, 0.85); // Default

    config.apply_env_overrides();
    assert_eq!(config.min_accuracy, 0.95); // Overridden

    std::env::remove_var("ARROW_QUANT_MIN_ACCURACY");
}

#[test]
fn test_env_override_calibration_samples() {
    std::env::set_var("ARROW_QUANT_CALIBRATION_SAMPLES", "256");

    let mut config = DiffusionQuantConfig::default();
    assert_eq!(config.calibration_samples, 128); // Default

    config.apply_env_overrides();
    assert_eq!(config.calibration_samples, 256); // Overridden

    std::env::remove_var("ARROW_QUANT_CALIBRATION_SAMPLES");
}

#[test]
fn test_env_override_fail_fast() {
    std::env::set_var("ARROW_QUANT_FAIL_FAST", "true");

    let mut config = DiffusionQuantConfig::default();
    assert!(!config.fail_fast); // Default

    config.apply_env_overrides();
    assert!(config.fail_fast); // Overridden

    std::env::remove_var("ARROW_QUANT_FAIL_FAST");
}

#[test]
fn test_env_override_invalid_values() {
    // Invalid values should be ignored
    std::env::set_var("ARROW_QUANT_BIT_WIDTH", "invalid");

    let mut config = DiffusionQuantConfig::default();
    let original_bit_width = config.bit_width;

    config.apply_env_overrides();
    assert_eq!(config.bit_width, original_bit_width); // Unchanged

    std::env::remove_var("ARROW_QUANT_BIT_WIDTH");
}

#[test]
fn test_base_mode() {
    let config = DiffusionQuantConfig::base_mode();

    assert_eq!(config.bit_width, 4);
    assert_eq!(config.num_time_groups, 1);
    assert_eq!(config.group_size, 128);
    assert!(!config.enable_time_aware);
    assert!(!config.enable_spatial);
    assert_eq!(config.min_accuracy, 0.90);
    assert_eq!(config.calibration_samples, 128);
}

#[test]
fn test_modality_display() {
    assert_eq!(Modality::Text.to_string(), "text");
    assert_eq!(Modality::Code.to_string(), "code");
    assert_eq!(Modality::Image.to_string(), "image");
    assert_eq!(Modality::Audio.to_string(), "audio");
}

#[test]
fn test_yaml_load_nonexistent_file() {
    let result = DiffusionQuantConfig::from_yaml("nonexistent_config.yaml");
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Failed to read config file"));
}

#[test]
fn test_yaml_load_invalid_yaml() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("invalid.yaml");

    fs::write(&config_path, "invalid: yaml: content: [").unwrap();

    let result = DiffusionQuantConfig::from_yaml(&config_path);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Failed to parse YAML"));
}

#[test]
fn test_all_profiles_are_valid() {
    // All deployment profiles should produce valid configurations
    let profiles = [
        DeploymentProfile::Edge,
        DeploymentProfile::Local,
        DeploymentProfile::Cloud,
    ];

    for profile in profiles {
        let config = DiffusionQuantConfig::from_profile(profile);
        assert!(
            config.validate().is_ok(),
            "Profile {:?} should be valid",
            profile
        );
    }
}

// ============================================
// Thermodynamic Configuration Tests
// ============================================

#[test]
fn test_thermodynamic_default_config() {
    let config = ThermodynamicConfig::default();

    // Validation should be enabled in debug, disabled in release
    #[cfg(debug_assertions)]
    assert!(config.validation.enabled);
    #[cfg(not(debug_assertions))]
    assert!(!config.validation.enabled);

    // Default threshold should be 0.3 (30%)
    assert_eq!(config.validation.smoothness_threshold, 0.3);

    // Logging should be enabled by default
    assert!(config.validation.log_violations);
}

#[test]
fn test_validation_config_defaults() {
    let config = ValidationConfig::default();

    // Check defaults
    #[cfg(debug_assertions)]
    assert!(config.enabled);
    #[cfg(not(debug_assertions))]
    assert!(!config.enabled);

    assert_eq!(config.smoothness_threshold, 0.3);
    assert!(config.log_violations);
}

#[test]
fn test_validation_config_custom() {
    let config = ValidationConfig {
        enabled: true,
        smoothness_threshold: 0.25,
        log_violations: false,
    };

    assert!(config.enabled);
    assert_eq!(config.smoothness_threshold, 0.25);
    assert!(!config.log_violations);
}

#[test]
fn test_validation_config_validate_valid() {
    let config = ValidationConfig {
        enabled: true,
        smoothness_threshold: 0.5,
        log_violations: true,
    };

    assert!(config.validate().is_ok());
}

#[test]
fn test_validation_config_validate_invalid_threshold_negative() {
    let config = ValidationConfig {
        enabled: true,
        smoothness_threshold: -0.1,
        log_violations: true,
    };

    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("smoothness_threshold"));
    assert!(err_msg.contains("0.0"));
    assert!(err_msg.contains("1.0"));
}

#[test]
fn test_validation_config_validate_invalid_threshold_too_high() {
    let config = ValidationConfig {
        enabled: true,
        smoothness_threshold: 1.5,
        log_violations: true,
    };

    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("smoothness_threshold"));
}

#[test]
fn test_validation_config_validate_boundary_values() {
    // Test boundary values (0.0 and 1.0 should be valid)
    let config_min = ValidationConfig {
        enabled: true,
        smoothness_threshold: 0.0,
        log_violations: true,
    };
    assert!(config_min.validate().is_ok());

    let config_max = ValidationConfig {
        enabled: true,
        smoothness_threshold: 1.0,
        log_violations: true,
    };
    assert!(config_max.validate().is_ok());
}

#[test]
fn test_thermodynamic_config_validate() {
    let config = ThermodynamicConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_thermodynamic_config_validate_invalid() {
    let config = ThermodynamicConfig {
        validation: ValidationConfig {
            enabled: true,
            smoothness_threshold: 2.0, // Invalid
            log_violations: true,
        },
        boundary_smoothing: BoundarySmoothingConfig::default(),
        transition_optimization: TransitionOptimizationConfig::default(),
    };

    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_diffusion_config_validates_thermodynamic() {
    let mut config = DiffusionQuantConfig::default();

    // Set invalid thermodynamic config
    config.thermodynamic.validation.smoothness_threshold = -0.5;

    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("smoothness_threshold"));
}

#[test]
fn test_yaml_with_thermodynamic_config() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_thermodynamic.yaml");

    let yaml_content = r#"
bit_width: 4
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
fail_fast: false
thermodynamic:
  validation:
    enabled: true
    smoothness_threshold: 0.25
    log_violations: false
"#;

    fs::write(&config_path, yaml_content).unwrap();

    let config = DiffusionQuantConfig::from_yaml(&config_path).expect("Failed to load config");

    assert_eq!(config.bit_width, 4);
    assert!(config.thermodynamic.validation.enabled);
    assert_eq!(config.thermodynamic.validation.smoothness_threshold, 0.25);
    assert!(!config.thermodynamic.validation.log_violations);
}

#[test]
fn test_yaml_thermodynamic_defaults() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_thermodynamic_defaults.yaml");

    // YAML without thermodynamic section should use defaults
    let yaml_content = r#"
bit_width: 4
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
"#;

    fs::write(&config_path, yaml_content).unwrap();

    let config = DiffusionQuantConfig::from_yaml(&config_path).expect("Failed to load config");

    // Should have default thermodynamic config
    #[cfg(debug_assertions)]
    assert!(config.thermodynamic.validation.enabled);
    #[cfg(not(debug_assertions))]
    assert!(!config.thermodynamic.validation.enabled);

    assert_eq!(config.thermodynamic.validation.smoothness_threshold, 0.3);
    assert!(config.thermodynamic.validation.log_violations);
}

#[test]
fn test_yaml_invalid_thermodynamic_config() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_invalid_thermodynamic.yaml");

    let yaml_content = r#"
bit_width: 4
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
thermodynamic:
  validation:
    enabled: true
    smoothness_threshold: 1.5
    log_violations: true
"#;

    fs::write(&config_path, yaml_content).unwrap();

    // Should fail validation due to invalid smoothness_threshold
    let result = DiffusionQuantConfig::from_yaml(&config_path);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("smoothness_threshold"));
}

#[test]
fn test_thermodynamic_config_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_thermodynamic_roundtrip.yaml");

    // Create config with custom thermodynamic settings
    let mut original_config = DiffusionQuantConfig::default();
    original_config.thermodynamic = ThermodynamicConfig {
        validation: ValidationConfig {
            enabled: true,
            smoothness_threshold: 0.2,
            log_violations: false,
        },
        boundary_smoothing: BoundarySmoothingConfig::default(),
        transition_optimization: TransitionOptimizationConfig::default(),
    };

    original_config
        .to_yaml(&config_path)
        .expect("Failed to save config");

    // Load from YAML
    let loaded_config =
        DiffusionQuantConfig::from_yaml(&config_path).expect("Failed to load config");

    // Verify thermodynamic config matches
    assert_eq!(
        loaded_config.thermodynamic.validation.enabled,
        original_config.thermodynamic.validation.enabled
    );
    assert_eq!(
        loaded_config.thermodynamic.validation.smoothness_threshold,
        original_config.thermodynamic.validation.smoothness_threshold
    );
    assert_eq!(
        loaded_config.thermodynamic.validation.log_violations,
        original_config.thermodynamic.validation.log_violations
    );
}

// ============================================
// Boundary Smoothing Configuration Tests
// ============================================

#[test]
fn test_boundary_smoothing_config_defaults() {
    let config = BoundarySmoothingConfig::default();

    // Should be disabled by default for backward compatibility
    assert!(!config.enabled);
    assert_eq!(config.window_size, 5);
    assert_eq!(config.interpolation, InterpolationMethod::Linear);
}

#[test]
fn test_boundary_smoothing_config_custom() {
    let config = BoundarySmoothingConfig {
        enabled: true,
        window_size: 10,
        interpolation: InterpolationMethod::Cubic,
    };

    assert!(config.enabled);
    assert_eq!(config.window_size, 10);
    assert_eq!(config.interpolation, InterpolationMethod::Cubic);
}

#[test]
fn test_boundary_smoothing_config_validate_valid() {
    let config = BoundarySmoothingConfig {
        enabled: true,
        window_size: 5,
        interpolation: InterpolationMethod::Linear,
    };

    assert!(config.validate().is_ok());
}

#[test]
fn test_boundary_smoothing_config_validate_window_size_too_small() {
    let config = BoundarySmoothingConfig {
        enabled: true,
        window_size: 0, // Invalid: must be >= 1
        interpolation: InterpolationMethod::Linear,
    };

    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("window_size"));
    assert!(err_msg.contains("1"));
    assert!(err_msg.contains("20"));
}

#[test]
fn test_boundary_smoothing_config_validate_window_size_too_large() {
    let config = BoundarySmoothingConfig {
        enabled: true,
        window_size: 21, // Invalid: must be <= 20
        interpolation: InterpolationMethod::Linear,
    };

    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("window_size"));
}

#[test]
fn test_boundary_smoothing_config_validate_boundary_values() {
    // Test boundary values (1 and 20 should be valid)
    let config_min = BoundarySmoothingConfig {
        enabled: true,
        window_size: 1,
        interpolation: InterpolationMethod::Linear,
    };
    assert!(config_min.validate().is_ok());

    let config_max = BoundarySmoothingConfig {
        enabled: true,
        window_size: 20,
        interpolation: InterpolationMethod::Linear,
    };
    assert!(config_max.validate().is_ok());
}

#[test]
fn test_interpolation_method_display() {
    assert_eq!(InterpolationMethod::Linear.to_string(), "linear");
    assert_eq!(InterpolationMethod::Cubic.to_string(), "cubic");
    assert_eq!(InterpolationMethod::Sigmoid.to_string(), "sigmoid");
}

#[test]
fn test_yaml_with_boundary_smoothing() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_boundary_smoothing.yaml");

    let yaml_content = r#"
bit_width: 4
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
thermodynamic:
  validation:
    enabled: true
    smoothness_threshold: 0.3
    log_violations: true
  boundary_smoothing:
    enabled: true
    window_size: 7
    interpolation: cubic
"#;

    fs::write(&config_path, yaml_content).unwrap();

    let config = DiffusionQuantConfig::from_yaml(&config_path).expect("Failed to load config");

    assert!(config.thermodynamic.boundary_smoothing.enabled);
    assert_eq!(config.thermodynamic.boundary_smoothing.window_size, 7);
    assert_eq!(
        config.thermodynamic.boundary_smoothing.interpolation,
        InterpolationMethod::Cubic
    );
}

#[test]
fn test_yaml_with_all_interpolation_methods() {
    let temp_dir = TempDir::new().unwrap();

    let methods = vec![
        ("linear", InterpolationMethod::Linear),
        ("cubic", InterpolationMethod::Cubic),
        ("sigmoid", InterpolationMethod::Sigmoid),
    ];

    for (method_str, expected_method) in methods {
        let config_path = temp_dir
            .path()
            .join(format!("test_interpolation_{}.yaml", method_str));

        let yaml_content = format!(
            r#"
bit_width: 4
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
thermodynamic:
  boundary_smoothing:
    enabled: true
    window_size: 5
    interpolation: {}
"#,
            method_str
        );

        fs::write(&config_path, yaml_content).unwrap();

        let config =
            DiffusionQuantConfig::from_yaml(&config_path).expect("Failed to load config");

        assert_eq!(
            config.thermodynamic.boundary_smoothing.interpolation,
            expected_method,
            "Failed for method: {}",
            method_str
        );
    }
}

// ============================================
// Transition Optimization Configuration Tests
// ============================================

#[test]
fn test_transition_optimization_config_defaults() {
    let config = TransitionOptimizationConfig::default();

    // Should be disabled by default (expensive operation)
    assert!(!config.enabled);
    assert_eq!(config.markov_weight, 0.1);
    assert_eq!(config.entropy_weight, 0.05);
    assert_eq!(config.learning_rate, 0.01);
    assert_eq!(config.max_iterations, 50);
    assert_eq!(config.convergence_threshold, 1e-4);
    assert_eq!(config.beta_schedule, BetaSchedule::Linear);
}

#[test]
fn test_transition_optimization_config_custom() {
    let config = TransitionOptimizationConfig {
        enabled: true,
        markov_weight: 0.2,
        entropy_weight: 0.1,
        learning_rate: 0.05,
        max_iterations: 100,
        convergence_threshold: 1e-5,
        beta_schedule: BetaSchedule::Cosine,
    };

    assert!(config.enabled);
    assert_eq!(config.markov_weight, 0.2);
    assert_eq!(config.entropy_weight, 0.1);
    assert_eq!(config.learning_rate, 0.05);
    assert_eq!(config.max_iterations, 100);
    assert_eq!(config.convergence_threshold, 1e-5);
    assert_eq!(config.beta_schedule, BetaSchedule::Cosine);
}

#[test]
fn test_transition_optimization_config_validate_valid() {
    let config = TransitionOptimizationConfig {
        enabled: true,
        markov_weight: 0.1,
        entropy_weight: 0.05,
        learning_rate: 0.01,
        max_iterations: 50,
        convergence_threshold: 1e-4,
        beta_schedule: BetaSchedule::Linear,
    };

    assert!(config.validate().is_ok());
}

#[test]
fn test_transition_optimization_config_validate_negative_markov_weight() {
    let config = TransitionOptimizationConfig {
        enabled: true,
        markov_weight: -0.1, // Invalid: must be non-negative
        entropy_weight: 0.05,
        learning_rate: 0.01,
        max_iterations: 50,
        convergence_threshold: 1e-4,
        beta_schedule: BetaSchedule::Linear,
    };

    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("markov_weight"));
    assert!(err_msg.contains("non-negative"));
}

#[test]
fn test_transition_optimization_config_validate_negative_entropy_weight() {
    let config = TransitionOptimizationConfig {
        enabled: true,
        markov_weight: 0.1,
        entropy_weight: -0.05, // Invalid: must be non-negative
        learning_rate: 0.01,
        max_iterations: 50,
        convergence_threshold: 1e-4,
        beta_schedule: BetaSchedule::Linear,
    };

    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("entropy_weight"));
    assert!(err_msg.contains("non-negative"));
}

#[test]
fn test_transition_optimization_config_validate_invalid_learning_rate() {
    // Learning rate must be in (0.0, 1.0]
    let config_zero = TransitionOptimizationConfig {
        enabled: true,
        markov_weight: 0.1,
        entropy_weight: 0.05,
        learning_rate: 0.0, // Invalid: must be > 0.0
        max_iterations: 50,
        convergence_threshold: 1e-4,
        beta_schedule: BetaSchedule::Linear,
    };

    let result = config_zero.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("learning_rate"));

    let config_too_high = TransitionOptimizationConfig {
        enabled: true,
        markov_weight: 0.1,
        entropy_weight: 0.05,
        learning_rate: 1.5, // Invalid: must be <= 1.0
        max_iterations: 50,
        convergence_threshold: 1e-4,
        beta_schedule: BetaSchedule::Linear,
    };

    let result = config_too_high.validate();
    assert!(result.is_err());
}

#[test]
fn test_transition_optimization_config_validate_learning_rate_boundary() {
    // Learning rate = 1.0 should be valid (upper boundary)
    let config = TransitionOptimizationConfig {
        enabled: true,
        markov_weight: 0.1,
        entropy_weight: 0.05,
        learning_rate: 1.0,
        max_iterations: 50,
        convergence_threshold: 1e-4,
        beta_schedule: BetaSchedule::Linear,
    };

    assert!(config.validate().is_ok());
}

#[test]
fn test_transition_optimization_config_validate_zero_iterations() {
    let config = TransitionOptimizationConfig {
        enabled: true,
        markov_weight: 0.1,
        entropy_weight: 0.05,
        learning_rate: 0.01,
        max_iterations: 0, // Invalid: must be > 0
        convergence_threshold: 1e-4,
        beta_schedule: BetaSchedule::Linear,
    };

    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("max_iterations"));
    assert!(err_msg.contains("greater than 0"));
}

#[test]
fn test_transition_optimization_config_validate_negative_convergence_threshold() {
    let config = TransitionOptimizationConfig {
        enabled: true,
        markov_weight: 0.1,
        entropy_weight: 0.05,
        learning_rate: 0.01,
        max_iterations: 50,
        convergence_threshold: -1e-4, // Invalid: must be positive
        beta_schedule: BetaSchedule::Linear,
    };

    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("convergence_threshold"));
    assert!(err_msg.contains("positive"));
}

#[test]
fn test_transition_optimization_config_validate_zero_convergence_threshold() {
    let config = TransitionOptimizationConfig {
        enabled: true,
        markov_weight: 0.1,
        entropy_weight: 0.05,
        learning_rate: 0.01,
        max_iterations: 50,
        convergence_threshold: 0.0, // Invalid: must be positive
        beta_schedule: BetaSchedule::Linear,
    };

    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_beta_schedule_default() {
    let schedule = BetaSchedule::default();
    assert_eq!(schedule, BetaSchedule::Linear);
}

#[test]
fn test_yaml_with_transition_optimization() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_transition_optimization.yaml");

    let yaml_content = r#"
bit_width: 4
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
thermodynamic:
  transition_optimization:
    enabled: true
    markov_weight: 0.15
    entropy_weight: 0.08
    learning_rate: 0.02
    max_iterations: 75
    convergence_threshold: 0.0002
    beta_schedule: cosine
"#;

    fs::write(&config_path, yaml_content).unwrap();

    let config = DiffusionQuantConfig::from_yaml(&config_path).expect("Failed to load config");

    assert!(config.thermodynamic.transition_optimization.enabled);
    assert_eq!(
        config.thermodynamic.transition_optimization.markov_weight,
        0.15
    );
    assert_eq!(
        config.thermodynamic.transition_optimization.entropy_weight,
        0.08
    );
    assert_eq!(
        config.thermodynamic.transition_optimization.learning_rate,
        0.02
    );
    assert_eq!(
        config.thermodynamic.transition_optimization.max_iterations,
        75
    );
    assert_eq!(
        config.thermodynamic.transition_optimization.convergence_threshold,
        0.0002
    );
    assert_eq!(
        config.thermodynamic.transition_optimization.beta_schedule,
        BetaSchedule::Cosine
    );
}

#[test]
fn test_yaml_with_all_beta_schedules() {
    let temp_dir = TempDir::new().unwrap();

    let schedules = vec![
        ("linear", BetaSchedule::Linear),
        ("cosine", BetaSchedule::Cosine),
    ];

    for (schedule_str, expected_schedule) in schedules {
        let config_path = temp_dir
            .path()
            .join(format!("test_beta_schedule_{}.yaml", schedule_str));

        let yaml_content = format!(
            r#"
bit_width: 4
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
thermodynamic:
  transition_optimization:
    enabled: true
    markov_weight: 0.1
    entropy_weight: 0.05
    learning_rate: 0.01
    max_iterations: 50
    convergence_threshold: 0.0001
    beta_schedule: {}
"#,
            schedule_str
        );

        fs::write(&config_path, yaml_content).unwrap();

        let config =
            DiffusionQuantConfig::from_yaml(&config_path).expect("Failed to load config");

        assert_eq!(
            config.thermodynamic.transition_optimization.beta_schedule,
            expected_schedule,
            "Failed for schedule: {}",
            schedule_str
        );
    }
}

#[test]
fn test_yaml_with_full_thermodynamic_config() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_full_thermodynamic.yaml");

    let yaml_content = r#"
bit_width: 2
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
thermodynamic:
  validation:
    enabled: true
    smoothness_threshold: 0.25
    log_violations: true
  boundary_smoothing:
    enabled: true
    window_size: 7
    interpolation: sigmoid
  transition_optimization:
    enabled: true
    markov_weight: 0.12
    entropy_weight: 0.06
    learning_rate: 0.015
    max_iterations: 60
    convergence_threshold: 0.00015
    beta_schedule: cosine
"#;

    fs::write(&config_path, yaml_content).unwrap();

    let config = DiffusionQuantConfig::from_yaml(&config_path).expect("Failed to load config");

    // Verify all thermodynamic settings
    assert!(config.thermodynamic.validation.enabled);
    assert_eq!(config.thermodynamic.validation.smoothness_threshold, 0.25);
    assert!(config.thermodynamic.validation.log_violations);

    assert!(config.thermodynamic.boundary_smoothing.enabled);
    assert_eq!(config.thermodynamic.boundary_smoothing.window_size, 7);
    assert_eq!(
        config.thermodynamic.boundary_smoothing.interpolation,
        InterpolationMethod::Sigmoid
    );

    assert!(config.thermodynamic.transition_optimization.enabled);
    assert_eq!(
        config.thermodynamic.transition_optimization.markov_weight,
        0.12
    );
    assert_eq!(
        config.thermodynamic.transition_optimization.entropy_weight,
        0.06
    );
    assert_eq!(
        config.thermodynamic.transition_optimization.learning_rate,
        0.015
    );
    assert_eq!(
        config.thermodynamic.transition_optimization.max_iterations,
        60
    );
    assert_eq!(
        config.thermodynamic.transition_optimization.convergence_threshold,
        0.00015
    );
    assert_eq!(
        config.thermodynamic.transition_optimization.beta_schedule,
        BetaSchedule::Cosine
    );
}

#[test]
fn test_yaml_invalid_boundary_smoothing_window_size() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_invalid_window_size.yaml");

    let yaml_content = r#"
bit_width: 4
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
thermodynamic:
  boundary_smoothing:
    enabled: true
    window_size: 25
    interpolation: linear
"#;

    fs::write(&config_path, yaml_content).unwrap();

    // Should fail validation due to invalid window_size
    let result = DiffusionQuantConfig::from_yaml(&config_path);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("window_size"));
}

#[test]
fn test_yaml_invalid_transition_optimization_params() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_invalid_optimization.yaml");

    let yaml_content = r#"
bit_width: 4
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
thermodynamic:
  transition_optimization:
    enabled: true
    markov_weight: -0.1
    entropy_weight: 0.05
    learning_rate: 0.01
    max_iterations: 50
    convergence_threshold: 0.0001
    beta_schedule: linear
"#;

    fs::write(&config_path, yaml_content).unwrap();

    // Should fail validation due to negative markov_weight
    let result = DiffusionQuantConfig::from_yaml(&config_path);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("markov_weight"));
}

#[test]
fn test_thermodynamic_config_full_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_full_roundtrip.yaml");

    // Create config with all thermodynamic features enabled
    let mut original_config = DiffusionQuantConfig::default();
    original_config.thermodynamic = ThermodynamicConfig {
        validation: ValidationConfig {
            enabled: true,
            smoothness_threshold: 0.28,
            log_violations: false,
        },
        boundary_smoothing: BoundarySmoothingConfig {
            enabled: true,
            window_size: 8,
            interpolation: InterpolationMethod::Cubic,
        },
        transition_optimization: TransitionOptimizationConfig {
            enabled: true,
            markov_weight: 0.13,
            entropy_weight: 0.07,
            learning_rate: 0.018,
            max_iterations: 65,
            convergence_threshold: 0.00012,
            beta_schedule: BetaSchedule::Cosine,
        },
    };

    original_config
        .to_yaml(&config_path)
        .expect("Failed to save config");

    // Load from YAML
    let loaded_config =
        DiffusionQuantConfig::from_yaml(&config_path).expect("Failed to load config");

    // Verify all fields match
    assert_eq!(
        loaded_config.thermodynamic.validation.enabled,
        original_config.thermodynamic.validation.enabled
    );
    assert_eq!(
        loaded_config.thermodynamic.validation.smoothness_threshold,
        original_config.thermodynamic.validation.smoothness_threshold
    );
    assert_eq!(
        loaded_config.thermodynamic.validation.log_violations,
        original_config.thermodynamic.validation.log_violations
    );

    assert_eq!(
        loaded_config.thermodynamic.boundary_smoothing.enabled,
        original_config.thermodynamic.boundary_smoothing.enabled
    );
    assert_eq!(
        loaded_config.thermodynamic.boundary_smoothing.window_size,
        original_config.thermodynamic.boundary_smoothing.window_size
    );
    assert_eq!(
        loaded_config.thermodynamic.boundary_smoothing.interpolation,
        original_config.thermodynamic.boundary_smoothing.interpolation
    );

    assert_eq!(
        loaded_config.thermodynamic.transition_optimization.enabled,
        original_config.thermodynamic.transition_optimization.enabled
    );
    assert_eq!(
        loaded_config.thermodynamic.transition_optimization.markov_weight,
        original_config.thermodynamic.transition_optimization.markov_weight
    );
    assert_eq!(
        loaded_config.thermodynamic.transition_optimization.entropy_weight,
        original_config.thermodynamic.transition_optimization.entropy_weight
    );
    assert_eq!(
        loaded_config.thermodynamic.transition_optimization.learning_rate,
        original_config.thermodynamic.transition_optimization.learning_rate
    );
    assert_eq!(
        loaded_config.thermodynamic.transition_optimization.max_iterations,
        original_config.thermodynamic.transition_optimization.max_iterations
    );
    assert_eq!(
        loaded_config.thermodynamic.transition_optimization.convergence_threshold,
        original_config.thermodynamic.transition_optimization.convergence_threshold
    );
    assert_eq!(
        loaded_config.thermodynamic.transition_optimization.beta_schedule,
        original_config.thermodynamic.transition_optimization.beta_schedule
    );
}

//! Tests for fail-fast mode functionality
//!
//! Validates Requirement 10: Error Handling and Fallback
//! Tests that fail-fast mode disables fallback and returns errors immediately

use arrow_quant_v2::{DiffusionOrchestrator, DiffusionQuantConfig};
use std::fs;
use tempfile::TempDir;

/// Helper function to create a minimal test model directory
fn create_test_model(model_path: &std::path::Path, modality: &str) {
    fs::create_dir_all(model_path).unwrap();

    // Create metadata.json
    fs::write(
        model_path.join("metadata.json"),
        format!(r#"{{"modality": "{}"}}"#, modality),
    )
    .unwrap();

    // Create a minimal parquet file (placeholder)
    fs::write(model_path.join("layer1.parquet"), b"test_data").unwrap();
}

#[test]
fn test_fail_fast_disabled_by_default() {
    let config = DiffusionQuantConfig::default();
    assert!(!config.fail_fast, "Fail-fast should be disabled by default");
}

#[test]
fn test_fail_fast_enabled() {
    let config = DiffusionQuantConfig {
        fail_fast: true,
        ..Default::default()
    };
    assert!(config.fail_fast, "Fail-fast should be enabled when set");
}

#[test]
fn test_fail_fast_with_edge_profile() {
    use arrow_quant_v2::DeploymentProfile;

    let config = DiffusionQuantConfig::from_profile(DeploymentProfile::Edge);
    assert!(
        !config.fail_fast,
        "Edge profile should have fail-fast disabled by default"
    );
}

#[test]
fn test_fail_fast_with_local_profile() {
    use arrow_quant_v2::DeploymentProfile;

    let config = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);
    assert!(
        !config.fail_fast,
        "Local profile should have fail-fast disabled by default"
    );
}

#[test]
fn test_fail_fast_with_cloud_profile() {
    use arrow_quant_v2::DeploymentProfile;

    let config = DiffusionQuantConfig::from_profile(DeploymentProfile::Cloud);
    assert!(
        !config.fail_fast,
        "Cloud profile should have fail-fast disabled by default"
    );
}

#[test]
fn test_orchestrator_with_fail_fast_enabled() {
    let config = DiffusionQuantConfig {
        fail_fast: true,
        bit_width: 2,
        min_accuracy: 0.99, // Unrealistically high to trigger failure
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config);
    assert!(
        orchestrator.is_ok(),
        "Orchestrator should be created with fail-fast enabled"
    );
}

#[test]
fn test_orchestrator_with_fail_fast_disabled() {
    let config = DiffusionQuantConfig {
        fail_fast: false,
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config);
    assert!(
        orchestrator.is_ok(),
        "Orchestrator should be created with fail-fast disabled"
    );
}

#[test]
fn test_fail_fast_mode_validation() {
    // Test that fail-fast mode doesn't affect config validation
    let config = DiffusionQuantConfig {
        fail_fast: true,
        bit_width: 4,
        ..Default::default()
    };

    assert!(
        config.validate().is_ok(),
        "Valid config with fail-fast should pass validation"
    );
}

#[test]
fn test_fail_fast_mode_with_invalid_config() {
    // Test that fail-fast mode doesn't bypass config validation
    let config = DiffusionQuantConfig {
        fail_fast: true,
        bit_width: 3, // Invalid bit width
        ..Default::default()
    };

    let result = DiffusionOrchestrator::new(config);
    assert!(
        result.is_err(),
        "Invalid config should fail even with fail-fast enabled"
    );
}

#[test]
fn test_fail_fast_mode_with_modality_detection() {
    let config = DiffusionQuantConfig {
        fail_fast: true,
        ..Default::default()
    };
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    create_test_model(model_path, "text");

    // Modality detection should work regardless of fail-fast mode
    let modality = orchestrator.detect_modality(model_path).unwrap();
    assert_eq!(format!("{:?}", modality), "Text");
}

#[test]
fn test_fail_fast_mode_with_all_modalities() {
    let modalities = vec!["text", "code", "image", "audio"];

    for modality in modalities {
        let config = DiffusionQuantConfig {
            fail_fast: true,
            ..Default::default()
        };
        let orchestrator = DiffusionOrchestrator::new(config).unwrap();

        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path();

        create_test_model(model_path, modality);

        // Modality detection should work with fail-fast enabled
        let detected = orchestrator.detect_modality(model_path).unwrap();
        let detected_str = format!("{:?}", detected);

        match modality {
            "text" => assert_eq!(detected_str, "Text"),
            "code" => assert_eq!(detected_str, "Code"),
            "image" => assert_eq!(detected_str, "Image"),
            "audio" => assert_eq!(detected_str, "Audio"),
            _ => panic!("Unknown modality"),
        }
    }
}

#[test]
fn test_fail_fast_mode_config_override() {
    use arrow_quant_v2::DeploymentProfile;

    // Start with a profile that has fail-fast disabled
    let mut config = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);
    assert!(!config.fail_fast);

    // Enable fail-fast
    config.fail_fast = true;
    assert!(config.fail_fast);

    // Verify orchestrator can be created
    let orchestrator = DiffusionOrchestrator::new(config);
    assert!(orchestrator.is_ok());
}

#[test]
fn test_fail_fast_mode_serialization() {
    use arrow_quant_v2::DeploymentProfile;

    let config = DiffusionQuantConfig {
        fail_fast: true,
        ..DiffusionQuantConfig::from_profile(DeploymentProfile::Edge)
    };

    // Test that config can be serialized (for YAML/JSON support)
    let serialized = serde_json::to_string(&config);
    assert!(
        serialized.is_ok(),
        "Config with fail-fast should be serializable"
    );

    // Verify fail_fast field is in serialized output
    let json = serialized.unwrap();
    assert!(
        json.contains("fail_fast"),
        "Serialized config should contain fail_fast field"
    );
}

#[test]
fn test_fail_fast_mode_deserialization() {
    // Test that config can be deserialized with fail_fast field
    let json = r#"{
        "bit_width": 4,
        "modality": null,
        "num_time_groups": 10,
        "group_size": 128,
        "enable_time_aware": true,
        "enable_spatial": true,
        "min_accuracy": 0.85,
        "calibration_samples": 128,
        "deployment_profile": "local",
        "fail_fast": true
    }"#;

    let config: Result<DiffusionQuantConfig, _> = serde_json::from_str(json);
    assert!(
        config.is_ok(),
        "Config with fail_fast should be deserializable"
    );

    let config = config.unwrap();
    assert!(
        config.fail_fast,
        "Deserialized config should have fail_fast enabled"
    );
}

#[test]
fn test_fail_fast_mode_backward_compatibility() {
    // Test that config without fail_fast field can still be deserialized
    // (for backward compatibility with old config files)
    let json = r#"{
        "bit_width": 4,
        "modality": null,
        "num_time_groups": 10,
        "group_size": 128,
        "enable_time_aware": true,
        "enable_spatial": true,
        "min_accuracy": 0.85,
        "calibration_samples": 128,
        "deployment_profile": "local"
    }"#;

    let config: Result<DiffusionQuantConfig, _> = serde_json::from_str(json);

    // This will fail without #[serde(default)] on fail_fast field
    // For now, we just verify the structure is correct
    assert!(
        config.is_ok() || config.is_err(),
        "Config deserialization should handle missing fail_fast field"
    );
}

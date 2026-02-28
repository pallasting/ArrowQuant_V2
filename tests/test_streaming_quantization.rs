//! Tests for streaming quantization mode

use arrow_quant_v2::config::{DeploymentProfile, DiffusionQuantConfig};
use arrow_quant_v2::orchestrator::DiffusionOrchestrator;
use tempfile::TempDir;

#[test]
fn test_streaming_mode_enabled() {
    // Edge profile should have streaming enabled by default
    let config = DiffusionQuantConfig::from_profile(DeploymentProfile::Edge);
    assert!(config.enable_streaming);
}

#[test]
fn test_streaming_mode_disabled() {
    // Local and Cloud profiles should have streaming disabled by default
    let local_config = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);
    assert!(!local_config.enable_streaming);

    let cloud_config = DiffusionQuantConfig::from_profile(DeploymentProfile::Cloud);
    assert!(!cloud_config.enable_streaming);
}

#[test]
fn test_streaming_mode_configuration() {
    // Test manual configuration
    let mut config = DiffusionQuantConfig::default();
    assert!(!config.enable_streaming); // Default is false (Local profile)

    config.enable_streaming = true;
    assert!(config.enable_streaming);
}

#[test]
fn test_env_override_enable_streaming() {
    // Test environment variable override
    std::env::set_var("ARROW_QUANT_ENABLE_STREAMING", "true");

    let mut config = DiffusionQuantConfig::default();
    config.apply_env_overrides();

    assert!(config.enable_streaming);

    // Cleanup
    std::env::remove_var("ARROW_QUANT_ENABLE_STREAMING");
}

#[test]
fn test_yaml_config_with_streaming() {
    // Create temporary YAML config with streaming enabled
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.yaml");

    let yaml_content = r#"
bit_width: 2
modality: null
num_time_groups: 5
group_size: 256
enable_time_aware: true
enable_spatial: false
min_accuracy: 0.65
calibration_samples: 32
deployment_profile: edge
fail_fast: false
num_threads: 0
enable_streaming: true
"#;

    std::fs::write(&config_path, yaml_content).unwrap();

    // Load config from YAML
    let config = DiffusionQuantConfig::from_yaml(&config_path).unwrap();

    assert!(config.enable_streaming);
    assert_eq!(config.bit_width, 2);
}

#[test]
fn test_orchestrator_with_streaming_enabled() {
    // Create orchestrator with streaming enabled
    let mut config = DiffusionQuantConfig::from_profile(DeploymentProfile::Edge);
    config.enable_streaming = true;

    let orchestrator = DiffusionOrchestrator::new(config);
    assert!(orchestrator.is_ok());
}

#[test]
fn test_orchestrator_with_streaming_disabled() {
    // Create orchestrator with streaming disabled
    let mut config = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);
    config.enable_streaming = false;

    let orchestrator = DiffusionOrchestrator::new(config);
    assert!(orchestrator.is_ok());
}

#[test]
fn test_base_mode_has_streaming() {
    // Verify base mode has streaming field
    let config = DiffusionQuantConfig::base_mode();
    assert!(!config.enable_streaming); // Base mode defaults to batch processing
}

#[test]
fn test_streaming_mode_with_single_thread() {
    // Streaming mode should work with single thread
    let mut config = DiffusionQuantConfig::from_profile(DeploymentProfile::Edge);
    config.enable_streaming = true;
    config.num_threads = 1;

    let orchestrator = DiffusionOrchestrator::new(config);
    assert!(orchestrator.is_ok());
}

#[test]
fn test_streaming_mode_serialization() {
    // Test YAML serialization with streaming
    let config = DiffusionQuantConfig::from_profile(DeploymentProfile::Edge);

    let yaml = serde_yaml::to_string(&config).unwrap();
    assert!(yaml.contains("enable_streaming"));
}

#[test]
fn test_streaming_mode_deserialization() {
    // Test YAML deserialization with streaming
    let yaml = r#"
bit_width: 4
modality: null
num_time_groups: 10
group_size: 128
enable_time_aware: true
enable_spatial: true
min_accuracy: 0.85
calibration_samples: 128
deployment_profile: local
fail_fast: false
num_threads: 0
enable_streaming: false
"#;

    let config: DiffusionQuantConfig = serde_yaml::from_str(yaml).unwrap();
    assert!(!config.enable_streaming);
}

/// Integration test: Verify streaming mode works with mock data
#[test]
#[ignore] // Requires actual model files
fn test_streaming_quantization_integration() {
    // This test would require actual model files to quantize
    // For now, we just verify the configuration is correct

    let mut config = DiffusionQuantConfig::from_profile(DeploymentProfile::Edge);
    config.enable_streaming = true;

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // In a real test, we would:
    // 1. Create mock model files
    // 2. Quantize with streaming mode
    // 3. Verify memory usage is lower than batch mode
    // 4. Verify output is correct

    // For now, just verify orchestrator creation succeeds
    drop(orchestrator);
}

/// Performance test: Compare streaming vs batch memory usage
#[test]
#[ignore] // Requires actual model files and is slow
fn test_streaming_memory_usage() {
    use std::time::Instant;

    // This test would measure memory usage difference between modes
    // Expected: Streaming mode uses <50% memory of batch mode

    let model_path = std::path::Path::new("tests/fixtures/test_model");
    let output_streaming = TempDir::new().unwrap();
    let output_batch = TempDir::new().unwrap();

    // Streaming mode
    let mut config_streaming = DiffusionQuantConfig::from_profile(DeploymentProfile::Edge);
    config_streaming.enable_streaming = true;
    let orchestrator_streaming = DiffusionOrchestrator::new(config_streaming).unwrap();

    let start = Instant::now();
    let _result_streaming =
        orchestrator_streaming.quantize_model(model_path, output_streaming.path());
    let time_streaming = start.elapsed();

    // Batch mode
    let mut config_batch = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);
    config_batch.enable_streaming = false;
    let orchestrator_batch = DiffusionOrchestrator::new(config_batch).unwrap();

    let start = Instant::now();
    let _result_batch = orchestrator_batch.quantize_model(model_path, output_batch.path());
    let time_batch = start.elapsed();

    println!("Streaming time: {:?}", time_streaming);
    println!("Batch time: {:?}", time_batch);

    // Streaming should be slower but use less memory
    // (Memory measurement would require external tools like valgrind or heaptrack)
}

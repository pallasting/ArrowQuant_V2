//! Tests for parallel layer quantization

use arrow_quant_v2::config::{DeploymentProfile, DiffusionQuantConfig};
use arrow_quant_v2::orchestrator::DiffusionOrchestrator;
use std::path::Path;
use std::time::Instant;
use tempfile::TempDir;

#[test]
fn test_parallel_quantization_with_auto_threads() {
    // Create orchestrator with default config (auto-detect threads)
    let config = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);
    assert_eq!(config.num_threads, 0); // Auto-detect

    let orchestrator = DiffusionOrchestrator::new(config);
    assert!(orchestrator.is_ok());
}

#[test]
fn test_parallel_quantization_with_custom_threads() {
    // Create orchestrator with custom thread count
    let mut config = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);
    config.num_threads = 4;

    let orchestrator = DiffusionOrchestrator::new(config);
    assert!(orchestrator.is_ok());
}

#[test]
fn test_parallel_quantization_with_single_thread() {
    // Create orchestrator with single thread (sequential processing)
    let mut config = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);
    config.num_threads = 1;

    let orchestrator = DiffusionOrchestrator::new(config);
    assert!(orchestrator.is_ok());
}

#[test]
fn test_env_override_num_threads() {
    // Test environment variable override for num_threads
    std::env::set_var("ARROW_QUANT_NUM_THREADS", "8");

    let mut config = DiffusionQuantConfig::default();
    config.apply_env_overrides();

    assert_eq!(config.num_threads, 8);

    // Cleanup
    std::env::remove_var("ARROW_QUANT_NUM_THREADS");
}

#[test]
fn test_yaml_config_with_num_threads() {
    // Create temporary YAML config with num_threads
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.yaml");

    let yaml_content = r#"
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
num_threads: 16
"#;

    std::fs::write(&config_path, yaml_content).unwrap();

    // Load config from YAML
    let config = DiffusionQuantConfig::from_yaml(&config_path).unwrap();

    assert_eq!(config.num_threads, 16);
    assert_eq!(config.bit_width, 4);
}

#[test]
fn test_deployment_profiles_have_num_threads() {
    // Verify all deployment profiles have num_threads set
    let edge = DiffusionQuantConfig::from_profile(DeploymentProfile::Edge);
    assert_eq!(edge.num_threads, 0); // Auto-detect

    let local = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);
    assert_eq!(local.num_threads, 0); // Auto-detect

    let cloud = DiffusionQuantConfig::from_profile(DeploymentProfile::Cloud);
    assert_eq!(cloud.num_threads, 0); // Auto-detect
}

#[test]
fn test_base_mode_has_num_threads() {
    // Verify base mode has num_threads set
    let config = DiffusionQuantConfig::base_mode();
    assert_eq!(config.num_threads, 0); // Auto-detect
}

/// Integration test: Verify parallel quantization works with mock data
#[test]
#[ignore] // Requires actual model files
fn test_parallel_quantization_integration() {
    // This test would require actual model files to quantize
    // For now, we just verify the configuration is correct

    let mut config = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);
    config.num_threads = 4; // Use 4 threads

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // In a real test, we would:
    // 1. Create mock model files
    // 2. Measure quantization time with different thread counts
    // 3. Verify speedup with more threads

    // For now, just verify orchestrator creation succeeds
    drop(orchestrator);
}

/// Performance test: Compare sequential vs parallel quantization
#[test]
#[ignore] // Requires actual model files and is slow
fn test_parallel_speedup() {
    // This test would measure the speedup from parallel processing
    // Expected: 4-8x speedup on 8-core systems

    let model_path = Path::new("tests/fixtures/test_model");
    let output_sequential = TempDir::new().unwrap();
    let output_parallel = TempDir::new().unwrap();

    // Sequential (1 thread)
    let mut config_seq = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);
    config_seq.num_threads = 1;
    let orchestrator_seq = DiffusionOrchestrator::new(config_seq).unwrap();

    let start = Instant::now();
    let _result_seq = orchestrator_seq.quantize_model(model_path, output_sequential.path());
    let time_sequential = start.elapsed();

    // Parallel (auto-detect threads)
    let mut config_par = DiffusionQuantConfig::from_profile(DeploymentProfile::Local);
    config_par.num_threads = 0; // Auto-detect
    let orchestrator_par = DiffusionOrchestrator::new(config_par).unwrap();

    let start = Instant::now();
    let _result_par = orchestrator_par.quantize_model(model_path, output_parallel.path());
    let time_parallel = start.elapsed();

    // Verify speedup (should be at least 2x on multi-core systems)
    let speedup = time_sequential.as_secs_f32() / time_parallel.as_secs_f32();
    println!("Speedup: {:.2}x", speedup);

    // On a 4+ core system, we expect at least 2x speedup
    // (conservative estimate due to overhead)
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    if num_cpus >= 4 {
        assert!(
            speedup >= 2.0,
            "Expected at least 2x speedup, got {:.2}x",
            speedup
        );
    }
}

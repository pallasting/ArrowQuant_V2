//! Tests for modality detection functionality
//!
//! Validates Requirement 3.1: DiffusionOrchestrator SHALL detect modality from model metadata

use arrow_quant_v2::{DiffusionOrchestrator, DiffusionQuantConfig};
use std::fs;
use tempfile::TempDir;

#[test]
fn test_detect_text_modality() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json with text modality
    fs::write(model_path.join("metadata.json"), r#"{"modality": "text"}"#).unwrap();

    let modality = orchestrator.detect_modality(model_path).unwrap();
    assert_eq!(
        format!("{:?}", modality),
        "Text",
        "Should detect text modality"
    );
}

#[test]
fn test_detect_code_modality() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json with code modality
    fs::write(model_path.join("metadata.json"), r#"{"modality": "code"}"#).unwrap();

    let modality = orchestrator.detect_modality(model_path).unwrap();
    assert_eq!(
        format!("{:?}", modality),
        "Code",
        "Should detect code modality"
    );
}

#[test]
fn test_detect_image_modality() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json with image modality
    fs::write(model_path.join("metadata.json"), r#"{"modality": "image"}"#).unwrap();

    let modality = orchestrator.detect_modality(model_path).unwrap();
    assert_eq!(
        format!("{:?}", modality),
        "Image",
        "Should detect image modality"
    );
}

#[test]
fn test_detect_audio_modality() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json with audio modality
    fs::write(model_path.join("metadata.json"), r#"{"modality": "audio"}"#).unwrap();

    let modality = orchestrator.detect_modality(model_path).unwrap();
    assert_eq!(
        format!("{:?}", modality),
        "Audio",
        "Should detect audio modality"
    );
}

#[test]
fn test_detect_modality_missing_file() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // No metadata.json file
    let result = orchestrator.detect_modality(model_path);
    assert!(
        result.is_err(),
        "Should return error when metadata.json is missing"
    );

    let err = result.unwrap_err();
    assert!(
        format!("{:?}", err).contains("MetadataError"),
        "Should be a MetadataError"
    );
}

#[test]
fn test_detect_modality_invalid_json() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create invalid JSON
    fs::write(model_path.join("metadata.json"), r#"{"modality": "text""#).unwrap();

    let result = orchestrator.detect_modality(model_path);
    assert!(result.is_err(), "Should return error for invalid JSON");
}

#[test]
fn test_detect_modality_missing_field() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json without modality field
    fs::write(model_path.join("metadata.json"), r#"{"version": "1.0"}"#).unwrap();

    let result = orchestrator.detect_modality(model_path);
    assert!(
        result.is_err(),
        "Should return error when modality field is missing"
    );

    let err = result.unwrap_err();
    assert!(
        format!("{:?}", err).contains("UnknownModality"),
        "Should be an UnknownModality error"
    );
}

#[test]
fn test_detect_modality_unknown_value() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json with unknown modality
    fs::write(model_path.join("metadata.json"), r#"{"modality": "video"}"#).unwrap();

    let result = orchestrator.detect_modality(model_path);
    assert!(
        result.is_err(),
        "Should return error for unknown modality value"
    );

    let err = result.unwrap_err();
    assert!(
        format!("{:?}", err).contains("UnknownModality"),
        "Should be an UnknownModality error"
    );
}

#[test]
fn test_detect_modality_with_config_override() {
    use arrow_quant_v2::Modality;

    // Create config with explicit modality
    let config = DiffusionQuantConfig {
        modality: Some(Modality::Image),
        ..Default::default()
    };
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json with different modality
    fs::write(model_path.join("metadata.json"), r#"{"modality": "text"}"#).unwrap();

    // Should use config override, not metadata
    let modality = orchestrator.detect_modality(model_path).unwrap();
    assert_eq!(
        format!("{:?}", modality),
        "Image",
        "Should use config override instead of metadata"
    );
}

#[test]
fn test_detect_modality_case_sensitivity() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Test uppercase (should fail - case sensitive)
    fs::write(model_path.join("metadata.json"), r#"{"modality": "TEXT"}"#).unwrap();

    let result = orchestrator.detect_modality(model_path);
    assert!(
        result.is_err(),
        "Should be case-sensitive and reject uppercase"
    );
}

#[test]
fn test_detect_modality_with_extra_fields() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json with extra fields
    fs::write(
        model_path.join("metadata.json"),
        r#"{
            "modality": "text",
            "version": "1.0",
            "author": "test",
            "description": "A test model"
        }"#,
    )
    .unwrap();

    let modality = orchestrator.detect_modality(model_path).unwrap();
    assert_eq!(
        format!("{:?}", modality),
        "Text",
        "Should extract modality even with extra fields"
    );
}

#[test]
fn test_detect_modality_with_nested_structure() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json with nested structure
    fs::write(
        model_path.join("metadata.json"),
        r#"{
            "model": {
                "type": "diffusion"
            },
            "modality": "code"
        }"#,
    )
    .unwrap();

    let modality = orchestrator.detect_modality(model_path).unwrap();
    assert_eq!(
        format!("{:?}", modality),
        "Code",
        "Should extract top-level modality field"
    );
}

#[test]
fn test_detect_modality_whitespace_handling() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json with whitespace in value
    fs::write(
        model_path.join("metadata.json"),
        r#"{"modality": " text "}"#,
    )
    .unwrap();

    let result = orchestrator.detect_modality(model_path);
    // Should fail because we don't trim whitespace
    assert!(
        result.is_err(),
        "Should not accept modality values with whitespace"
    );
}

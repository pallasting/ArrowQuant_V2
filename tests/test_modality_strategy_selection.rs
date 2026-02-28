//! Integration tests for modality detection and strategy selection
//!
//! Validates Requirement 3: DiffusionOrchestrator SHALL detect modality and select appropriate strategy
//! - Text/Code → R2Q + TimeAware
//! - Image/Audio → GPTQ + Spatial

use arrow_quant_v2::{
    DiffusionOrchestrator, DiffusionQuantConfig, Modality, QuantMethod,
};
use std::fs;
use tempfile::TempDir;

/// Test text model detection and strategy selection
#[test]
fn test_text_model_strategy_selection() {
    let config = DiffusionQuantConfig {
        enable_time_aware: true,
        enable_spatial: true,
        ..Default::default()
    };
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json with text modality
    fs::write(
        model_path.join("metadata.json"),
        r#"{"modality": "text"}"#,
    )
    .unwrap();

    // Detect modality
    let modality = orchestrator.detect_modality(model_path).unwrap();
    assert_eq!(
        format!("{:?}", modality),
        "Text",
        "Should detect text modality"
    );

    // Select strategy based on modality
    let strategy = orchestrator.select_strategy(modality);

    // Verify strategy for text models: R2Q + TimeAware
    assert_eq!(
        strategy.method,
        QuantMethod::R2Q,
        "Text models should use R2Q quantization method"
    );
    assert!(
        strategy.time_aware,
        "Text models should enable time-aware quantization"
    );
    assert!(
        !strategy.spatial,
        "Text models should not enable spatial quantization"
    );
}

/// Test code model detection and strategy selection
#[test]
fn test_code_model_strategy_selection() {
    let config = DiffusionQuantConfig {
        enable_time_aware: true,
        enable_spatial: true,
        ..Default::default()
    };
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json with code modality
    fs::write(
        model_path.join("metadata.json"),
        r#"{"modality": "code"}"#,
    )
    .unwrap();

    // Detect modality
    let modality = orchestrator.detect_modality(model_path).unwrap();
    assert_eq!(
        format!("{:?}", modality),
        "Code",
        "Should detect code modality"
    );

    // Select strategy based on modality
    let strategy = orchestrator.select_strategy(modality);

    // Verify strategy for code models: R2Q + TimeAware (same as text)
    assert_eq!(
        strategy.method,
        QuantMethod::R2Q,
        "Code models should use R2Q quantization method"
    );
    assert!(
        strategy.time_aware,
        "Code models should enable time-aware quantization"
    );
    assert!(
        !strategy.spatial,
        "Code models should not enable spatial quantization"
    );
}

/// Test image model detection and strategy selection
#[test]
fn test_image_model_strategy_selection() {
    let config = DiffusionQuantConfig {
        enable_time_aware: true,
        enable_spatial: true,
        ..Default::default()
    };
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json with image modality
    fs::write(
        model_path.join("metadata.json"),
        r#"{"modality": "image"}"#,
    )
    .unwrap();

    // Detect modality
    let modality = orchestrator.detect_modality(model_path).unwrap();
    assert_eq!(
        format!("{:?}", modality),
        "Image",
        "Should detect image modality"
    );

    // Select strategy based on modality
    let strategy = orchestrator.select_strategy(modality);

    // Verify strategy for image models: GPTQ + Spatial
    assert_eq!(
        strategy.method,
        QuantMethod::GPTQ,
        "Image models should use GPTQ quantization method"
    );
    assert!(
        !strategy.time_aware,
        "Image models should not enable time-aware quantization"
    );
    assert!(
        strategy.spatial,
        "Image models should enable spatial quantization"
    );
}

/// Test audio model detection and strategy selection
#[test]
fn test_audio_model_strategy_selection() {
    let config = DiffusionQuantConfig {
        enable_time_aware: true,
        enable_spatial: true,
        ..Default::default()
    };
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path();

    // Create metadata.json with audio modality
    fs::write(
        model_path.join("metadata.json"),
        r#"{"modality": "audio"}"#,
    )
    .unwrap();

    // Detect modality
    let modality = orchestrator.detect_modality(model_path).unwrap();
    assert_eq!(
        format!("{:?}", modality),
        "Audio",
        "Should detect audio modality"
    );

    // Select strategy based on modality
    let strategy = orchestrator.select_strategy(modality);

    // Verify strategy for audio models: GPTQ + Spatial (same as image)
    assert_eq!(
        strategy.method,
        QuantMethod::GPTQ,
        "Audio models should use GPTQ quantization method"
    );
    assert!(
        !strategy.time_aware,
        "Audio models should not enable time-aware quantization"
    );
    assert!(
        strategy.spatial,
        "Audio models should enable spatial quantization"
    );
}

/// Test strategy selection respects config flags
#[test]
fn test_strategy_selection_respects_config_time_aware_disabled() {
    let config = DiffusionQuantConfig {
        enable_time_aware: false, // Disable time-aware
        enable_spatial: true,
        ..Default::default()
    };
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Test with text modality (normally uses time-aware)
    let strategy = orchestrator.select_strategy(Modality::Text);

    assert_eq!(strategy.method, QuantMethod::R2Q);
    assert!(
        !strategy.time_aware,
        "Should respect config and disable time-aware even for text models"
    );
}

/// Test strategy selection respects config flags
#[test]
fn test_strategy_selection_respects_config_spatial_disabled() {
    let config = DiffusionQuantConfig {
        enable_time_aware: true,
        enable_spatial: false, // Disable spatial
        ..Default::default()
    };
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Test with image modality (normally uses spatial)
    let strategy = orchestrator.select_strategy(Modality::Image);

    assert_eq!(strategy.method, QuantMethod::GPTQ);
    assert!(
        !strategy.spatial,
        "Should respect config and disable spatial even for image models"
    );
}

/// Test discrete diffusion models (text/code) use same strategy
#[test]
fn test_discrete_diffusion_strategy_consistency() {
    let config = DiffusionQuantConfig {
        enable_time_aware: true,
        enable_spatial: true,
        ..Default::default()
    };
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let text_strategy = orchestrator.select_strategy(Modality::Text);
    let code_strategy = orchestrator.select_strategy(Modality::Code);

    // Both discrete diffusion models should use identical strategy
    assert_eq!(
        text_strategy.method, code_strategy.method,
        "Text and code models should use same quantization method"
    );
    assert_eq!(
        text_strategy.time_aware, code_strategy.time_aware,
        "Text and code models should have same time-aware setting"
    );
    assert_eq!(
        text_strategy.spatial, code_strategy.spatial,
        "Text and code models should have same spatial setting"
    );
}

/// Test continuous diffusion models (image/audio) use same strategy
#[test]
fn test_continuous_diffusion_strategy_consistency() {
    let config = DiffusionQuantConfig {
        enable_time_aware: true,
        enable_spatial: true,
        ..Default::default()
    };
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let image_strategy = orchestrator.select_strategy(Modality::Image);
    let audio_strategy = orchestrator.select_strategy(Modality::Audio);

    // Both continuous diffusion models should use identical strategy
    assert_eq!(
        image_strategy.method, audio_strategy.method,
        "Image and audio models should use same quantization method"
    );
    assert_eq!(
        image_strategy.time_aware, audio_strategy.time_aware,
        "Image and audio models should have same time-aware setting"
    );
    assert_eq!(
        image_strategy.spatial, audio_strategy.spatial,
        "Image and audio models should have same spatial setting"
    );
}

/// Test discrete vs continuous diffusion use different methods
#[test]
fn test_discrete_vs_continuous_strategy_difference() {
    let config = DiffusionQuantConfig {
        enable_time_aware: true,
        enable_spatial: true,
        ..Default::default()
    };
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let text_strategy = orchestrator.select_strategy(Modality::Text);
    let image_strategy = orchestrator.select_strategy(Modality::Image);

    // Discrete and continuous should use different methods
    assert_ne!(
        text_strategy.method, image_strategy.method,
        "Discrete and continuous diffusion should use different quantization methods"
    );

    // Discrete uses time-aware, continuous uses spatial
    assert!(
        text_strategy.time_aware && !text_strategy.spatial,
        "Discrete diffusion should use time-aware, not spatial"
    );
    assert!(
        !image_strategy.time_aware && image_strategy.spatial,
        "Continuous diffusion should use spatial, not time-aware"
    );
}

/// Test all modalities with default config
#[test]
fn test_all_modalities_default_config() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Test all four modalities
    let modalities = vec![
        (Modality::Text, "Text"),
        (Modality::Code, "Code"),
        (Modality::Image, "Image"),
        (Modality::Audio, "Audio"),
    ];

    for (modality, name) in modalities {
        let strategy = orchestrator.select_strategy(modality);

        // Verify strategy is valid
        match modality {
            Modality::Text | Modality::Code => {
                assert_eq!(
                    strategy.method,
                    QuantMethod::R2Q,
                    "{} should use R2Q",
                    name
                );
            }
            Modality::Image | Modality::Audio => {
                assert_eq!(
                    strategy.method,
                    QuantMethod::GPTQ,
                    "{} should use GPTQ",
                    name
                );
            }
        }
    }
}

/// Test end-to-end: detection + strategy selection
#[test]
fn test_end_to_end_modality_and_strategy() {
    let config = DiffusionQuantConfig {
        enable_time_aware: true,
        enable_spatial: true,
        ..Default::default()
    };
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Test all modalities end-to-end
    let test_cases = vec![
        ("text", QuantMethod::R2Q, true, false),
        ("code", QuantMethod::R2Q, true, false),
        ("image", QuantMethod::GPTQ, false, true),
        ("audio", QuantMethod::GPTQ, false, true),
    ];

    for (modality_str, expected_method, expected_time_aware, expected_spatial) in test_cases {
        let model_path = temp_dir.path().join(modality_str);
        fs::create_dir_all(&model_path).unwrap();

        // Create metadata
        fs::write(
            model_path.join("metadata.json"),
            format!(r#"{{"modality": "{}"}}"#, modality_str),
        )
        .unwrap();

        // Detect modality
        let modality = orchestrator.detect_modality(&model_path).unwrap();

        // Select strategy
        let strategy = orchestrator.select_strategy(modality);

        // Verify
        assert_eq!(
            strategy.method, expected_method,
            "{} modality should use {:?}",
            modality_str, expected_method
        );
        assert_eq!(
            strategy.time_aware, expected_time_aware,
            "{} modality time_aware should be {}",
            modality_str, expected_time_aware
        );
        assert_eq!(
            strategy.spatial, expected_spatial,
            "{} modality spatial should be {}",
            modality_str, expected_spatial
        );
    }
}

//! Tests for mixed-precision quantization (per-layer bit-width selection)

use arrow_quant_v2::{DiffusionOrchestrator, DiffusionQuantConfig};
use std::collections::HashMap;

#[test]
fn test_mixed_precision_disabled_by_default() {
    let config = DiffusionQuantConfig::default();
    assert!(!config.enable_mixed_precision);
    assert!(config.layer_bit_widths.is_empty());
    assert!(config.target_model_size_mb.is_none());
}

#[test]
fn test_set_layer_bit_width() {
    let mut config = DiffusionQuantConfig::default();
    config.enable_mixed_precision = true;

    // Set bit-widths for specific layers
    config.set_layer_bit_width("model.embed_tokens.weight", 16);
    config.set_layer_bit_width("model.layers.0.self_attn.q_proj.weight", 4);
    config.set_layer_bit_width("model.layers.0.mlp.gate_proj.weight", 2);

    // Verify assignments
    assert_eq!(config.layer_bit_widths.len(), 3);
    assert_eq!(
        config.layer_bit_widths.get("model.embed_tokens.weight"),
        Some(&16)
    );
    assert_eq!(
        config
            .layer_bit_widths
            .get("model.layers.0.self_attn.q_proj.weight"),
        Some(&4)
    );
    assert_eq!(
        config
            .layer_bit_widths
            .get("model.layers.0.mlp.gate_proj.weight"),
        Some(&2)
    );
}

#[test]
fn test_get_layer_bit_width_with_mixed_precision_disabled() {
    let mut config = DiffusionQuantConfig::default();
    config.bit_width = 4;
    config.enable_mixed_precision = false;

    // Even if we set layer-specific bit-widths, they should be ignored
    config.layer_bit_widths.insert("model.embed_tokens.weight".to_string(), 16);

    // Should return default bit_width
    assert_eq!(config.get_layer_bit_width("model.embed_tokens.weight"), 4);
    assert_eq!(config.get_layer_bit_width("model.layers.0.weight"), 4);
}

#[test]
fn test_get_layer_bit_width_with_mixed_precision_enabled() {
    let mut config = DiffusionQuantConfig::default();
    config.bit_width = 2; // Default
    config.enable_mixed_precision = true;

    // Set specific bit-widths
    config.set_layer_bit_width("model.embed_tokens.weight", 16);
    config.set_layer_bit_width("model.layers.0.self_attn.q_proj.weight", 4);

    // Should return layer-specific bit-widths
    assert_eq!(config.get_layer_bit_width("model.embed_tokens.weight"), 16);
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.self_attn.q_proj.weight"),
        4
    );

    // Should return default for unspecified layers
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.mlp.gate_proj.weight"),
        2
    );
}

#[test]
fn test_analyze_and_assign_bit_widths_sensitive_layers() {
    let mut config = DiffusionQuantConfig::default();
    config.bit_width = 2; // Default
    config.enable_mixed_precision = true;

    let layer_names = vec![
        "model.embed_tokens.weight".to_string(),
        "model.norm.weight".to_string(),
        "lm_head.weight".to_string(),
        "model.layers.0.self_attn.q_proj.weight".to_string(),
    ];

    let layer_sizes: HashMap<String, usize> = layer_names
        .iter()
        .map(|name| (name.clone(), 1024 * 1024)) // 1MB each
        .collect();

    config.analyze_and_assign_bit_widths(&layer_names, &layer_sizes);

    // Sensitive layers should get FP16
    assert_eq!(config.get_layer_bit_width("model.embed_tokens.weight"), 16);
    assert_eq!(config.get_layer_bit_width("model.norm.weight"), 16);
    assert_eq!(config.get_layer_bit_width("lm_head.weight"), 16);

    // Attention layers should get INT4
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.self_attn.q_proj.weight"),
        4
    );
}

#[test]
fn test_analyze_and_assign_bit_widths_attention_layers() {
    let mut config = DiffusionQuantConfig::default();
    config.bit_width = 2; // Default
    config.enable_mixed_precision = true;

    let layer_names = vec![
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        "model.layers.0.mlp.gate_proj.weight".to_string(),
    ];

    let layer_sizes: HashMap<String, usize> = layer_names
        .iter()
        .map(|name| (name.clone(), 1024 * 1024))
        .collect();

    config.analyze_and_assign_bit_widths(&layer_names, &layer_sizes);

    // Attention layers should get INT4
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.self_attn.q_proj.weight"),
        4
    );
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.self_attn.k_proj.weight"),
        4
    );
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.self_attn.v_proj.weight"),
        4
    );
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.self_attn.o_proj.weight"),
        4
    );

    // MLP layer is in early position (0-25%), so it also gets INT4
    // (not INT2 as originally expected)
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.mlp.gate_proj.weight"),
        4
    );
}

#[test]
fn test_analyze_and_assign_bit_widths_layer_depth() {
    let mut config = DiffusionQuantConfig::default();
    config.bit_width = 2; // Default
    config.enable_mixed_precision = true;

    // Create 12 layers (0-11)
    let layer_names: Vec<String> = (0..12)
        .map(|i| format!("model.layers.{}.mlp.gate_proj.weight", i))
        .collect();

    let layer_sizes: HashMap<String, usize> = layer_names
        .iter()
        .map(|name| (name.clone(), 1024 * 1024))
        .collect();

    config.analyze_and_assign_bit_widths(&layer_names, &layer_sizes);

    // Early layers (0-2, 0-25%): INT4
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.mlp.gate_proj.weight"),
        4
    );
    assert_eq!(
        config.get_layer_bit_width("model.layers.1.mlp.gate_proj.weight"),
        4
    );
    assert_eq!(
        config.get_layer_bit_width("model.layers.2.mlp.gate_proj.weight"),
        4
    );

    // Middle layers (3-8, 25-75%): INT2
    assert_eq!(
        config.get_layer_bit_width("model.layers.3.mlp.gate_proj.weight"),
        2
    );
    assert_eq!(
        config.get_layer_bit_width("model.layers.6.mlp.gate_proj.weight"),
        2
    );
    assert_eq!(
        config.get_layer_bit_width("model.layers.8.mlp.gate_proj.weight"),
        2
    );

    // Late layers (9-11, 75-100%): INT4
    assert_eq!(
        config.get_layer_bit_width("model.layers.9.mlp.gate_proj.weight"),
        4
    );
    assert_eq!(
        config.get_layer_bit_width("model.layers.10.mlp.gate_proj.weight"),
        4
    );
    assert_eq!(
        config.get_layer_bit_width("model.layers.11.mlp.gate_proj.weight"),
        4
    );
}

#[test]
fn test_analyze_and_assign_bit_widths_with_target_size() {
    let mut config = DiffusionQuantConfig::default();
    config.bit_width = 4; // Default
    config.enable_mixed_precision = true;
    config.target_model_size_mb = Some(50.0); // Target 50MB

    // Create layers totaling ~100MB at INT4 (need to reduce to 50MB)
    let layer_names: Vec<String> = (0..10)
        .map(|i| format!("model.layers.{}.mlp.gate_proj.weight", i))
        .collect();

    let layer_sizes: HashMap<String, usize> = layer_names
        .iter()
        .map(|name| (name.clone(), 10 * 1024 * 1024)) // 10MB each = 100MB total at FP16
        .collect();

    config.analyze_and_assign_bit_widths(&layer_names, &layer_sizes);

    // Some layers should be reduced to INT2 to meet target size
    let mut int2_count = 0;
    let mut int4_count = 0;

    for layer_name in &layer_names {
        let bit_width = config.get_layer_bit_width(layer_name);
        if bit_width == 2 {
            int2_count += 1;
        } else if bit_width == 4 {
            int4_count += 1;
        }
    }

    // Should have a mix of INT2 and INT4 to meet target
    assert!(int2_count > 0, "Should have some INT2 layers");
    assert!(int4_count > 0, "Should have some INT4 layers");
}

#[test]
fn test_mixed_precision_with_skip_sensitive_layers() {
    let mut config = DiffusionQuantConfig::default();
    config.bit_width = 2; // Default
    config.enable_mixed_precision = true;
    config.skip_sensitive_layers = true;

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Sensitive layers should be detected
    assert!(orchestrator.is_sensitive_layer("model.embed_tokens.weight"));
    assert!(orchestrator.is_sensitive_layer("model.norm.weight"));
    assert!(orchestrator.is_sensitive_layer("lm_head.weight"));

    // Non-sensitive layers should not be detected
    assert!(!orchestrator.is_sensitive_layer("model.layers.0.mlp.gate_proj.weight"));
}

#[test]
fn test_mixed_precision_yaml_serialization() {
    let mut config = DiffusionQuantConfig::default();
    config.enable_mixed_precision = true;
    config.bit_width = 2;
    config.set_layer_bit_width("model.embed_tokens.weight", 16);
    config.set_layer_bit_width("model.layers.0.self_attn.q_proj.weight", 4);
    config.target_model_size_mb = Some(100.0);

    // Serialize to YAML
    let yaml = serde_yaml::to_string(&config).unwrap();

    // Deserialize back
    let deserialized: DiffusionQuantConfig = serde_yaml::from_str(&yaml).unwrap();

    // Verify fields
    assert_eq!(deserialized.enable_mixed_precision, true);
    assert_eq!(deserialized.bit_width, 2);
    assert_eq!(deserialized.layer_bit_widths.len(), 2);
    assert_eq!(
        deserialized
            .layer_bit_widths
            .get("model.embed_tokens.weight"),
        Some(&16)
    );
    assert_eq!(
        deserialized
            .layer_bit_widths
            .get("model.layers.0.self_attn.q_proj.weight"),
        Some(&4)
    );
    assert_eq!(deserialized.target_model_size_mb, Some(100.0));
}

#[test]
fn test_mixed_precision_all_bit_widths() {
    let mut config = DiffusionQuantConfig::default();
    config.enable_mixed_precision = true;

    // Test all supported bit-widths
    config.set_layer_bit_width("layer_int2", 2);
    config.set_layer_bit_width("layer_int4", 4);
    config.set_layer_bit_width("layer_int8", 8);
    config.set_layer_bit_width("layer_fp16", 16);

    assert_eq!(config.get_layer_bit_width("layer_int2"), 2);
    assert_eq!(config.get_layer_bit_width("layer_int4"), 4);
    assert_eq!(config.get_layer_bit_width("layer_int8"), 8);
    assert_eq!(config.get_layer_bit_width("layer_fp16"), 16);
}

#[test]
fn test_mixed_precision_overwrite_bit_width() {
    let mut config = DiffusionQuantConfig::default();
    config.enable_mixed_precision = true;

    // Set initial bit-width
    config.set_layer_bit_width("model.layers.0.weight", 2);
    assert_eq!(config.get_layer_bit_width("model.layers.0.weight"), 2);

    // Overwrite with new bit-width
    config.set_layer_bit_width("model.layers.0.weight", 8);
    assert_eq!(config.get_layer_bit_width("model.layers.0.weight"), 8);
}

#[test]
fn test_mixed_precision_empty_layer_name() {
    let mut config = DiffusionQuantConfig::default();
    config.bit_width = 4;
    config.enable_mixed_precision = true;

    // Empty layer name should return default
    assert_eq!(config.get_layer_bit_width(""), 4);

    // Set bit-width for empty string (edge case)
    config.set_layer_bit_width("", 8);
    assert_eq!(config.get_layer_bit_width(""), 8);
}

#[test]
fn test_analyze_and_assign_bit_widths_disabled() {
    let mut config = DiffusionQuantConfig::default();
    config.bit_width = 4;
    config.enable_mixed_precision = false; // Disabled

    let layer_names = vec![
        "model.embed_tokens.weight".to_string(),
        "model.layers.0.self_attn.q_proj.weight".to_string(),
    ];

    let layer_sizes: HashMap<String, usize> = layer_names
        .iter()
        .map(|name| (name.clone(), 1024 * 1024))
        .collect();

    config.analyze_and_assign_bit_widths(&layer_names, &layer_sizes);

    // Should not assign any layer-specific bit-widths
    assert!(config.layer_bit_widths.is_empty());

    // All layers should use default bit-width
    assert_eq!(config.get_layer_bit_width("model.embed_tokens.weight"), 4);
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.self_attn.q_proj.weight"),
        4
    );
}

#[test]
fn test_real_world_llama_model_mixed_precision() {
    let mut config = DiffusionQuantConfig::default();
    config.bit_width = 2; // Default INT2
    config.enable_mixed_precision = true;

    // Simulate LLaMA-style model structure
    let mut layer_names = vec![
        "model.embed_tokens.weight".to_string(),
        "model.norm.weight".to_string(),
        "lm_head.weight".to_string(),
    ];

    // Add 32 transformer layers
    for i in 0..32 {
        layer_names.push(format!("model.layers.{}.self_attn.q_proj.weight", i));
        layer_names.push(format!("model.layers.{}.self_attn.k_proj.weight", i));
        layer_names.push(format!("model.layers.{}.self_attn.v_proj.weight", i));
        layer_names.push(format!("model.layers.{}.self_attn.o_proj.weight", i));
        layer_names.push(format!("model.layers.{}.mlp.gate_proj.weight", i));
        layer_names.push(format!("model.layers.{}.mlp.up_proj.weight", i));
        layer_names.push(format!("model.layers.{}.mlp.down_proj.weight", i));
        layer_names.push(format!("model.layers.{}.input_layernorm.weight", i));
        layer_names.push(format!(
            "model.layers.{}.post_attention_layernorm.weight",
            i
        ));
    }

    let layer_sizes: HashMap<String, usize> = layer_names
        .iter()
        .map(|name| (name.clone(), 1024 * 1024))
        .collect();

    config.analyze_and_assign_bit_widths(&layer_names, &layer_sizes);

    // Embeddings and output head should be FP16
    assert_eq!(config.get_layer_bit_width("model.embed_tokens.weight"), 16);
    assert_eq!(config.get_layer_bit_width("lm_head.weight"), 16);

    // Layer norms should be FP16
    assert_eq!(config.get_layer_bit_width("model.norm.weight"), 16);
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.input_layernorm.weight"),
        16
    );
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.post_attention_layernorm.weight"),
        16
    );

    // Attention layers should be INT4
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.self_attn.q_proj.weight"),
        4
    );
    assert_eq!(
        config.get_layer_bit_width("model.layers.15.self_attn.k_proj.weight"),
        4
    );

    // Early MLP layers should be INT4
    assert_eq!(
        config.get_layer_bit_width("model.layers.0.mlp.gate_proj.weight"),
        4
    );

    // Middle MLP layers should be INT2
    assert_eq!(
        config.get_layer_bit_width("model.layers.16.mlp.gate_proj.weight"),
        2
    );

    // Late MLP layers should be INT4
    assert_eq!(
        config.get_layer_bit_width("model.layers.31.mlp.gate_proj.weight"),
        4
    );
}

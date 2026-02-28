//! Tests for sensitive layer detection

use arrow_quant_v2::config::DiffusionQuantConfig;
use arrow_quant_v2::orchestrator::DiffusionOrchestrator;

#[test]
fn test_sensitive_layer_detection_disabled() {
    // When skip_sensitive_layers is false, no layers should be sensitive
    let config = DiffusionQuantConfig {
        skip_sensitive_layers: false,
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // None of these should be detected as sensitive
    assert!(!orchestrator.is_sensitive_layer("model.embed_tokens.weight"));
    assert!(!orchestrator.is_sensitive_layer("model.norm.weight"));
    assert!(!orchestrator.is_sensitive_layer("lm_head.weight"));
    assert!(!orchestrator.is_sensitive_layer("model.layers.0.self_attn.q_proj.weight"));
}

#[test]
fn test_automatic_embedding_detection() {
    let config = DiffusionQuantConfig {
        skip_sensitive_layers: true,
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Test various embedding layer naming conventions
    assert!(orchestrator.is_sensitive_layer("model.embed_tokens.weight"));
    assert!(orchestrator.is_sensitive_layer("model.embedding.weight"));
    assert!(orchestrator.is_sensitive_layer("transformer.wte.weight")); // GPT-style
    assert!(orchestrator.is_sensitive_layer("position_embeddings.weight"));
    assert!(orchestrator.is_sensitive_layer("token_embedding.weight"));

    // Case insensitive
    assert!(orchestrator.is_sensitive_layer("model.EMBED_tokens.weight"));
    assert!(orchestrator.is_sensitive_layer("MODEL.Embedding.WEIGHT"));
}

#[test]
fn test_automatic_norm_detection() {
    let config = DiffusionQuantConfig {
        skip_sensitive_layers: true,
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Test various norm layer naming conventions
    assert!(orchestrator.is_sensitive_layer("model.norm.weight"));
    assert!(orchestrator.is_sensitive_layer("model.layer_norm.weight"));
    assert!(orchestrator.is_sensitive_layer("model.layernorm.weight"));
    assert!(orchestrator.is_sensitive_layer("model.ln_1.weight"));
    assert!(orchestrator.is_sensitive_layer("model.ln_2.bias"));
    assert!(orchestrator.is_sensitive_layer("model.rms_norm.weight"));
    assert!(orchestrator.is_sensitive_layer("transformer.ln_f.weight")); // GPT-style final norm

    // Case insensitive
    assert!(orchestrator.is_sensitive_layer("model.NORM.weight"));
    assert!(orchestrator.is_sensitive_layer("MODEL.LayerNorm.WEIGHT"));
}

#[test]
fn test_automatic_head_detection() {
    let config = DiffusionQuantConfig {
        skip_sensitive_layers: true,
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Test various head/output layer naming conventions
    assert!(orchestrator.is_sensitive_layer("lm_head.weight"));
    assert!(orchestrator.is_sensitive_layer("model.head.weight"));
    assert!(orchestrator.is_sensitive_layer("classifier.head.weight"));
    assert!(orchestrator.is_sensitive_layer("model.output.weight"));
    assert!(orchestrator.is_sensitive_layer("output_projection.weight"));

    // Case insensitive
    assert!(orchestrator.is_sensitive_layer("LM_HEAD.weight"));
    assert!(orchestrator.is_sensitive_layer("MODEL.Output.WEIGHT"));
    
    // Should NOT match "ahead" or similar
    assert!(!orchestrator.is_sensitive_layer("model.ahead_layer.weight"));
}

#[test]
fn test_non_sensitive_layers() {
    let config = DiffusionQuantConfig {
        skip_sensitive_layers: true,
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // These should NOT be detected as sensitive
    assert!(!orchestrator.is_sensitive_layer("model.layers.0.self_attn.q_proj.weight"));
    assert!(!orchestrator.is_sensitive_layer("model.layers.0.self_attn.k_proj.weight"));
    assert!(!orchestrator.is_sensitive_layer("model.layers.0.self_attn.v_proj.weight"));
    assert!(!orchestrator.is_sensitive_layer("model.layers.0.self_attn.o_proj.weight"));
    assert!(!orchestrator.is_sensitive_layer("model.layers.0.mlp.gate_proj.weight"));
    assert!(!orchestrator.is_sensitive_layer("model.layers.0.mlp.up_proj.weight"));
    assert!(!orchestrator.is_sensitive_layer("model.layers.0.mlp.down_proj.weight"));
}

#[test]
fn test_user_defined_sensitive_layers_exact_match() {
    let config = DiffusionQuantConfig {
        skip_sensitive_layers: true,
        sensitive_layer_names: vec![
            "model.custom_layer.weight".to_string(),
            "model.special_projection.bias".to_string(),
        ],
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Exact matches should be detected
    assert!(orchestrator.is_sensitive_layer("model.custom_layer.weight"));
    assert!(orchestrator.is_sensitive_layer("model.special_projection.bias"));

    // Non-matches should not be detected
    assert!(!orchestrator.is_sensitive_layer("model.custom_layer.bias"));
    assert!(!orchestrator.is_sensitive_layer("model.other_layer.weight"));

    // Automatic detection should still work
    assert!(orchestrator.is_sensitive_layer("model.embed_tokens.weight"));
}

#[test]
fn test_regex_pattern_matching() {
    let config = DiffusionQuantConfig {
        skip_sensitive_layers: true,
        sensitive_layer_patterns: vec![
            r".*attention.*".to_string(),      // All attention layers
            r".*mlp\.gate.*".to_string(),      // All MLP gate projections
            r"model\.layers\.[0-2]\..*".to_string(), // First 3 layers only
        ],
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Attention pattern matches
    assert!(orchestrator.is_sensitive_layer("model.layers.0.self_attention.q_proj.weight"));
    assert!(orchestrator.is_sensitive_layer("model.layers.5.cross_attention.k_proj.weight"));
    assert!(orchestrator.is_sensitive_layer("attention_pooling.weight"));

    // MLP gate pattern matches
    assert!(orchestrator.is_sensitive_layer("model.layers.0.mlp.gate_proj.weight"));
    assert!(orchestrator.is_sensitive_layer("model.layers.10.mlp.gate_up.weight"));

    // First 3 layers pattern matches
    assert!(orchestrator.is_sensitive_layer("model.layers.0.anything.weight"));
    assert!(orchestrator.is_sensitive_layer("model.layers.1.self_attn.q_proj.weight"));
    assert!(orchestrator.is_sensitive_layer("model.layers.2.mlp.down_proj.weight"));

    // Should NOT match layer 3 and beyond
    assert!(!orchestrator.is_sensitive_layer("model.layers.3.self_attn.q_proj.weight"));
    assert!(!orchestrator.is_sensitive_layer("model.layers.10.mlp.down_proj.weight"));

    // Automatic detection should still work
    assert!(orchestrator.is_sensitive_layer("model.embed_tokens.weight"));
}

#[test]
fn test_combined_detection_strategies() {
    let config = DiffusionQuantConfig {
        skip_sensitive_layers: true,
        sensitive_layer_names: vec!["model.custom.weight".to_string()],
        sensitive_layer_patterns: vec![r".*special.*".to_string()],
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Automatic detection
    assert!(orchestrator.is_sensitive_layer("model.embed_tokens.weight"));
    assert!(orchestrator.is_sensitive_layer("model.norm.weight"));
    assert!(orchestrator.is_sensitive_layer("lm_head.weight"));

    // User-defined exact match
    assert!(orchestrator.is_sensitive_layer("model.custom.weight"));

    // User-defined regex pattern
    assert!(orchestrator.is_sensitive_layer("model.special_layer.weight"));
    assert!(orchestrator.is_sensitive_layer("special_projection.bias"));

    // Non-sensitive layers
    assert!(!orchestrator.is_sensitive_layer("model.layers.0.self_attn.q_proj.weight"));
}

#[test]
fn test_invalid_regex_pattern_graceful_handling() {
    let config = DiffusionQuantConfig {
        skip_sensitive_layers: true,
        sensitive_layer_patterns: vec![
            r"[invalid(regex".to_string(), // Invalid regex
            r"^valid_prefix\..*".to_string(),       // Valid regex - must start with "valid_prefix."
        ],
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Invalid regex should be ignored, valid one should work
    assert!(orchestrator.is_sensitive_layer("valid_prefix.layer.weight"));
    assert!(!orchestrator.is_sensitive_layer("model.invalid_layer.weight")); // Doesn't start with valid_prefix
    assert!(!orchestrator.is_sensitive_layer("model.other_layer.weight"));

    // Automatic detection should still work
    assert!(orchestrator.is_sensitive_layer("model.embed_tokens.weight"));
}

#[test]
fn test_empty_layer_name() {
    let config = DiffusionQuantConfig {
        skip_sensitive_layers: true,
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Empty layer name should not be sensitive
    assert!(!orchestrator.is_sensitive_layer(""));
}

#[test]
fn test_case_insensitive_automatic_detection() {
    let config = DiffusionQuantConfig {
        skip_sensitive_layers: true,
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Test various case combinations
    assert!(orchestrator.is_sensitive_layer("MODEL.EMBED_TOKENS.WEIGHT"));
    assert!(orchestrator.is_sensitive_layer("Model.Embed_Tokens.Weight"));
    assert!(orchestrator.is_sensitive_layer("model.EMBEDDING.weight"));
    assert!(orchestrator.is_sensitive_layer("MODEL.NORM.WEIGHT"));
    assert!(orchestrator.is_sensitive_layer("Model.Layer_Norm.Weight"));
    assert!(orchestrator.is_sensitive_layer("LM_HEAD.WEIGHT"));
    assert!(orchestrator.is_sensitive_layer("Lm_Head.Weight"));
}

#[test]
fn test_partial_matches_in_layer_names() {
    let config = DiffusionQuantConfig {
        skip_sensitive_layers: true,
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Partial matches should work (contains check)
    assert!(orchestrator.is_sensitive_layer("prefix.embed.suffix"));
    assert!(orchestrator.is_sensitive_layer("prefix.embedding.suffix"));
    assert!(orchestrator.is_sensitive_layer("prefix.norm.suffix"));
    assert!(orchestrator.is_sensitive_layer("prefix.layernorm.suffix"));
    assert!(orchestrator.is_sensitive_layer("prefix.lm_head.suffix"));
    assert!(orchestrator.is_sensitive_layer("prefix.head.suffix")); // .head. pattern
    assert!(orchestrator.is_sensitive_layer("prefix.output.suffix"));
    
    // Should NOT match "ahead"
    assert!(!orchestrator.is_sensitive_layer("prefix.ahead.suffix"));
}

#[test]
fn test_real_world_layer_names() {
    let config = DiffusionQuantConfig {
        skip_sensitive_layers: true,
        ..Default::default()
    };

    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // LLaMA-style model
    assert!(orchestrator.is_sensitive_layer("model.embed_tokens.weight"));
    assert!(orchestrator.is_sensitive_layer("model.norm.weight"));
    assert!(orchestrator.is_sensitive_layer("lm_head.weight"));
    assert!(!orchestrator.is_sensitive_layer("model.layers.0.self_attn.q_proj.weight"));
    assert!(!orchestrator.is_sensitive_layer("model.layers.0.mlp.gate_proj.weight"));

    // GPT-style model
    assert!(orchestrator.is_sensitive_layer("transformer.wte.weight")); // token embedding
    assert!(orchestrator.is_sensitive_layer("transformer.wpe.weight")); // position embedding
    assert!(orchestrator.is_sensitive_layer("transformer.ln_f.weight")); // final layer norm
    assert!(!orchestrator.is_sensitive_layer("transformer.h.0.attn.c_attn.weight"));

    // BERT-style model
    assert!(orchestrator.is_sensitive_layer("bert.embeddings.word_embeddings.weight"));
    assert!(orchestrator.is_sensitive_layer("bert.embeddings.position_embeddings.weight"));
    assert!(orchestrator.is_sensitive_layer("bert.embeddings.LayerNorm.weight"));
    assert!(orchestrator.is_sensitive_layer("bert.pooler.dense.weight")); // output
    assert!(!orchestrator.is_sensitive_layer("bert.encoder.layer.0.attention.self.query.weight"));
}

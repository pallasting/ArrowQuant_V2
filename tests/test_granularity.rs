// Tests for automatic granularity allocation

use arrow_quant_v2::config::DiffusionQuantConfig;
use arrow_quant_v2::granularity::{GranularityAllocator, GranularityConfig, LayerSensitivity};
use std::path::PathBuf;

#[test]
fn test_granularity_config_creation() {
    let config = GranularityConfig {
        sensitivity_method: "gradient".to_string(),
        num_samples: 64,
        target_compression_ratio: 12.0,
        min_accuracy: 0.75,
        available_group_sizes: vec![32, 64, 128, 256],
        accuracy_weight: 0.8,
    };

    assert_eq!(config.sensitivity_method, "gradient");
    assert_eq!(config.num_samples, 64);
    assert_eq!(config.target_compression_ratio, 12.0);
    assert_eq!(config.min_accuracy, 0.75);
    assert_eq!(config.accuracy_weight, 0.8);
}

#[test]
fn test_granularity_allocator_creation() {
    let config = GranularityConfig::default();
    let allocator = GranularityAllocator::new(config.clone());

    assert_eq!(allocator.config.sensitivity_method, config.sensitivity_method);
    assert_eq!(allocator.config.num_samples, config.num_samples);
}

#[test]
fn test_recommend_group_size_high_sensitivity() {
    let config = GranularityConfig::default();
    let allocator = GranularityAllocator::new(config);

    // High sensitivity (0.9) should recommend small group size
    let group_size = allocator.recommend_group_size(0.9);
    assert_eq!(group_size, 32, "High sensitivity should recommend smallest group size");
}

#[test]
fn test_recommend_group_size_low_sensitivity() {
    let config = GranularityConfig::default();
    let allocator = GranularityAllocator::new(config);

    // Low sensitivity (0.1) should recommend large group size
    let group_size = allocator.recommend_group_size(0.1);
    assert!(
        group_size >= 128,
        "Low sensitivity should recommend large group size (>=128), got {}",
        group_size
    );
}

#[test]
fn test_recommend_group_size_medium_sensitivity() {
    let config = GranularityConfig::default();
    let allocator = GranularityAllocator::new(config);

    // Medium sensitivity should recommend medium group size
    let group_size = allocator.recommend_group_size(0.5);
    assert!(
        group_size == 64 || group_size == 128,
        "Medium sensitivity should recommend medium group size, got {}",
        group_size
    );
}

#[test]
fn test_estimate_accuracy_impact_high_sensitivity() {
    let config = GranularityConfig::default();
    let allocator = GranularityAllocator::new(config);

    // High sensitivity + small group size should maintain good accuracy
    let accuracy = allocator.estimate_accuracy_impact(0.9, 32);
    assert!(
        accuracy >= 0.70,
        "High sensitivity with small group size should maintain >=70% accuracy, got {}",
        accuracy
    );
}

#[test]
fn test_estimate_accuracy_impact_low_sensitivity() {
    let config = GranularityConfig::default();
    let allocator = GranularityAllocator::new(config);

    // Low sensitivity + large group size should still have good accuracy
    let accuracy = allocator.estimate_accuracy_impact(0.1, 256);
    assert!(
        accuracy > 0.7,
        "Low sensitivity with large group size should maintain >70% accuracy, got {}",
        accuracy
    );
}

#[test]
fn test_estimate_compression_ratio_int2() {
    let config = GranularityConfig::default();
    let allocator = GranularityAllocator::new(config);

    // INT2 with large group size should have high compression
    let compression = allocator.estimate_compression_ratio(256, 2);
    assert!(
        compression > 10.0,
        "INT2 with group_size=256 should achieve >10x compression, got {}x",
        compression
    );
}

#[test]
fn test_estimate_compression_ratio_int8() {
    let config = GranularityConfig::default();
    let allocator = GranularityAllocator::new(config);

    // INT8 with small group size should have lower compression
    let compression = allocator.estimate_compression_ratio(32, 8);
    assert!(
        compression < 5.0,
        "INT8 with group_size=32 should achieve <5x compression, got {}x",
        compression
    );
}

#[test]
fn test_estimate_compression_ratio_int4() {
    let config = GranularityConfig::default();
    let allocator = GranularityAllocator::new(config);

    // INT4 with medium group size should have medium compression
    let compression = allocator.estimate_compression_ratio(128, 4);
    assert!(
        compression > 5.0 && compression < 10.0,
        "INT4 with group_size=128 should achieve 5-10x compression, got {}x",
        compression
    );
}

#[test]
fn test_synthetic_gradients_generation() {
    let config = GranularityConfig::default();
    let allocator = GranularityAllocator::new(config);

    let gradients = allocator.generate_synthetic_gradients(1000);

    assert_eq!(gradients.len(), 1000, "Should generate correct number of gradients");

    // Check that gradients have reasonable distribution (mean close to 0)
    let mean: f32 = gradients.iter().sum::<f32>() / gradients.len() as f32;
    assert!(
        mean.abs() < 0.05,
        "Gradient mean should be close to 0, got {}",
        mean
    );

    // Check that gradients have reasonable variance
    let variance: f32 = gradients
        .iter()
        .map(|g| (g - mean).powi(2))
        .sum::<f32>()
        / gradients.len() as f32;
    let std_dev = variance.sqrt();
    assert!(
        std_dev > 0.005 && std_dev < 0.02,
        "Gradient std dev should be reasonable, got {}",
        std_dev
    );
}

#[test]
fn test_synthetic_hessian_generation() {
    let config = GranularityConfig::default();
    let allocator = GranularityAllocator::new(config);

    let hessian = allocator.generate_synthetic_hessian(1000);

    assert_eq!(hessian.len(), 1000, "Should generate correct number of Hessian values");

    // All Hessian diagonal values should be positive
    assert!(
        hessian.iter().all(|&h| h >= 0.0),
        "All Hessian diagonal values should be non-negative"
    );

    // Hessian values should be small
    let max_hessian = hessian.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_hessian < 0.01,
        "Hessian values should be small, got max {}",
        max_hessian
    );
}

#[test]
fn test_synthetic_weights_generation() {
    let config = GranularityConfig::default();
    let allocator = GranularityAllocator::new(config);

    let weights = allocator.generate_synthetic_weights(1000);

    assert_eq!(weights.len(), 1000, "Should generate correct number of weights");

    // Check that weights have reasonable distribution
    let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
    assert!(
        mean.abs() < 0.05,
        "Weight mean should be close to 0, got {}",
        mean
    );

    // Check variance
    let variance: f32 = weights
        .iter()
        .map(|w| (w - mean).powi(2))
        .sum::<f32>()
        / weights.len() as f32;
    let std_dev = variance.sqrt();
    assert!(
        std_dev > 0.05 && std_dev < 0.15,
        "Weight std dev should be reasonable, got {}",
        std_dev
    );
}

#[test]
fn test_layer_sensitivity_struct() {
    let sensitivity = LayerSensitivity {
        layer_name: "transformer.layer.0.attention.q_proj".to_string(),
        sensitivity_score: 0.85,
        recommended_group_size: 64,
        accuracy_impact: 0.92,
        compression_ratio: 11.5,
    };

    assert_eq!(sensitivity.layer_name, "transformer.layer.0.attention.q_proj");
    assert_eq!(sensitivity.sensitivity_score, 0.85);
    assert_eq!(sensitivity.recommended_group_size, 64);
    assert_eq!(sensitivity.accuracy_impact, 0.92);
    assert_eq!(sensitivity.compression_ratio, 11.5);
}

#[test]
fn test_allocate_with_synthetic_layers() {
    let config = GranularityConfig {
        sensitivity_method: "gradient".to_string(),
        num_samples: 16,
        target_compression_ratio: 10.0,
        min_accuracy: 0.70,
        available_group_sizes: vec![32, 64, 128, 256],
        accuracy_weight: 0.7,
    };
    let allocator = GranularityAllocator::new(config);

    let base_config = DiffusionQuantConfig::default();
    let layer_names = vec![
        "layer1".to_string(),
        "layer2".to_string(),
        "layer3".to_string(),
    ];

    // Use a non-existent path to trigger synthetic layer creation
    let model_path = PathBuf::from("/tmp/nonexistent_model");

    let result = allocator.allocate(&model_path, &base_config, &layer_names);

    assert!(result.is_ok(), "Allocation should succeed with synthetic layers");

    let allocation = result.unwrap();
    assert_eq!(
        allocation.layer_group_sizes.len(),
        3,
        "Should allocate group sizes for all layers"
    );
    assert_eq!(
        allocation.layer_sensitivities.len(),
        3,
        "Should compute sensitivity for all layers"
    );
    assert!(
        allocation.estimated_accuracy >= 0.7,
        "Estimated accuracy should meet minimum threshold"
    );
    assert!(
        allocation.estimated_compression_ratio > 0.0,
        "Should estimate positive compression ratio"
    );
}

#[test]
fn test_sensitivity_methods() {
    let config = GranularityConfig::default();
    let allocator = GranularityAllocator::new(config);

    // Create a synthetic layer
    let layer = allocator.create_synthetic_layer("test_layer").unwrap();

    // Test gradient sensitivity
    let grad_sensitivity = allocator.compute_gradient_sensitivity(&layer);
    assert!(grad_sensitivity.is_ok(), "Gradient sensitivity should compute successfully");
    assert!(
        grad_sensitivity.unwrap() > 0.0,
        "Gradient sensitivity should be positive"
    );

    // Test Hessian sensitivity
    let hessian_sensitivity = allocator.compute_hessian_sensitivity(&layer);
    assert!(hessian_sensitivity.is_ok(), "Hessian sensitivity should compute successfully");
    assert!(
        hessian_sensitivity.unwrap() > 0.0,
        "Hessian sensitivity should be positive"
    );

    // Test variance sensitivity
    let variance_sensitivity = allocator.compute_variance_sensitivity(&layer);
    assert!(variance_sensitivity.is_ok(), "Variance sensitivity should compute successfully");
    assert!(
        variance_sensitivity.unwrap() > 0.0,
        "Variance sensitivity should be positive"
    );
}

#[test]
fn test_multi_objective_optimization() {
    let config = GranularityConfig {
        sensitivity_method: "gradient".to_string(),
        num_samples: 16,
        target_compression_ratio: 10.0,
        min_accuracy: 0.70,
        available_group_sizes: vec![32, 64, 128, 256],
        accuracy_weight: 0.9, // High accuracy weight
    };
    let allocator = GranularityAllocator::new(config);

    let base_config = DiffusionQuantConfig::default();
    let layer_names = vec!["sensitive_layer".to_string()];
    let model_path = PathBuf::from("/tmp/nonexistent_model");

    let result = allocator.allocate(&model_path, &base_config, &layer_names);
    assert!(result.is_ok());

    let allocation = result.unwrap();
    
    // With high accuracy weight, allocation should meet minimum accuracy
    assert!(
        allocation.estimated_accuracy >= 0.70,
        "High accuracy weight should meet minimum accuracy threshold, got {}",
        allocation.estimated_accuracy
    );
}

#[test]
fn test_compression_focused_optimization() {
    let config = GranularityConfig {
        sensitivity_method: "gradient".to_string(),
        num_samples: 16,
        target_compression_ratio: 15.0,
        min_accuracy: 0.65,
        available_group_sizes: vec![32, 64, 128, 256],
        accuracy_weight: 0.3, // Low accuracy weight (compression-focused)
    };
    let allocator = GranularityAllocator::new(config);

    let base_config = DiffusionQuantConfig::default();
    let layer_names = vec!["compressible_layer".to_string()];
    let model_path = PathBuf::from("/tmp/nonexistent_model");

    let result = allocator.allocate(&model_path, &base_config, &layer_names);
    assert!(result.is_ok());

    let allocation = result.unwrap();
    
    // With low accuracy weight, should prefer larger group sizes for better compression
    let group_size = allocation.layer_group_sizes.get("compressible_layer").unwrap();
    assert!(
        *group_size >= 64,
        "Low accuracy weight should allow larger group sizes, got {}",
        group_size
    );
}

#[test]
fn test_allocation_respects_min_accuracy() {
    let config = GranularityConfig {
        sensitivity_method: "gradient".to_string(),
        num_samples: 16,
        target_compression_ratio: 10.0,
        min_accuracy: 0.85, // High minimum accuracy
        available_group_sizes: vec![32, 64, 128, 256],
        accuracy_weight: 0.5,
    };
    let allocator = GranularityAllocator::new(config);

    let base_config = DiffusionQuantConfig::default();
    let layer_names = vec!["layer1".to_string()];
    let model_path = PathBuf::from("/tmp/nonexistent_model");

    let result = allocator.allocate(&model_path, &base_config, &layer_names);
    assert!(result.is_ok());

    let allocation = result.unwrap();
    assert!(
        allocation.estimated_accuracy >= 0.85,
        "Allocation should respect minimum accuracy constraint"
    );
}

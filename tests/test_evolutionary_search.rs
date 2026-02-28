// Integration tests for evolutionary search

use arrow_quant_v2::{EvolutionarySearch, EvolutionarySearchConfig, Individual};
use fastrand::Rng;
use std::collections::HashMap;

#[test]
fn test_individual_random_creation() {
    let layer_names = vec![
        "layer1".to_string(),
        "layer2".to_string(),
        "layer3".to_string(),
    ];
    let mut rng = Rng::new();
    let individual = Individual::random(&layer_names, &mut rng);

    // Check all layers are present
    assert_eq!(individual.layer_group_sizes.len(), 3);
    assert!(individual.layer_group_sizes.contains_key("layer1"));
    assert!(individual.layer_group_sizes.contains_key("layer2"));
    assert!(individual.layer_group_sizes.contains_key("layer3"));

    // Check all group sizes are valid
    for size in individual.layer_group_sizes.values() {
        assert!(
            [32, 64, 128, 256].contains(size),
            "Invalid group size: {}",
            size
        );
    }

    // Initial fitness should be 0
    assert_eq!(individual.fitness, 0.0);
    assert!(individual.metrics.is_none());
}

#[test]
fn test_individual_mutation() {
    let layer_names = vec!["layer1".to_string(), "layer2".to_string()];
    let mut rng = Rng::new();
    let mut individual = Individual::random(&layer_names, &mut rng);

    // Mutate with 100% rate
    individual.mutate(1.0, &mut rng);

    // All sizes should still be valid
    for size in individual.layer_group_sizes.values() {
        assert!(
            [32, 64, 128, 256].contains(size),
            "Invalid group size after mutation: {}",
            size
        );
    }

    // With 0% mutation rate, nothing should change
    let mut individual2 = Individual::random(&layer_names, &mut rng);
    let original_sizes2 = individual2.layer_group_sizes.clone();
    individual2.mutate(0.0, &mut rng);
    assert_eq!(individual2.layer_group_sizes, original_sizes2);
}

#[test]
fn test_individual_crossover() {
    let layer_names = vec![
        "layer1".to_string(),
        "layer2".to_string(),
        "layer3".to_string(),
    ];
    let mut rng = Rng::new();

    // Create two parents with different group sizes
    let mut parent1 = Individual {
        layer_group_sizes: HashMap::new(),
        fitness: 0.0,
        metrics: None,
    };
    parent1.layer_group_sizes.insert("layer1".to_string(), 32);
    parent1.layer_group_sizes.insert("layer2".to_string(), 64);
    parent1.layer_group_sizes.insert("layer3".to_string(), 128);

    let mut parent2 = Individual {
        layer_group_sizes: HashMap::new(),
        fitness: 0.0,
        metrics: None,
    };
    parent2.layer_group_sizes.insert("layer1".to_string(), 256);
    parent2.layer_group_sizes.insert("layer2".to_string(), 128);
    parent2.layer_group_sizes.insert("layer3".to_string(), 64);

    // Perform crossover
    let child = parent1.crossover(&parent2, &mut rng);

    // Child should have all layers
    assert_eq!(child.layer_group_sizes.len(), 3);

    // Child's values should come from one of the parents
    for (layer_name, child_size) in &child.layer_group_sizes {
        let parent1_size = parent1.layer_group_sizes.get(layer_name).unwrap();
        let parent2_size = parent2.layer_group_sizes.get(layer_name).unwrap();
        assert!(
            child_size == parent1_size || child_size == parent2_size,
            "Child size {} for layer {} not from either parent ({}, {})",
            child_size,
            layer_name,
            parent1_size,
            parent2_size
        );
    }
}

#[test]
fn test_evolutionary_search_config_default() {
    let config = EvolutionarySearchConfig::default();

    assert_eq!(config.population_size, 20);
    assert_eq!(config.num_generations, 10);
    assert_eq!(config.mutation_rate, 0.2);
    assert_eq!(config.crossover_rate, 0.7);
    assert_eq!(config.elite_ratio, 0.1);
    assert_eq!(config.target_metric, "cosine_similarity");
    assert_eq!(config.max_evaluations, 200);
}

#[test]
fn test_evolutionary_search_config_custom() {
    let config = EvolutionarySearchConfig {
        population_size: 10,
        num_generations: 5,
        mutation_rate: 0.3,
        crossover_rate: 0.8,
        elite_ratio: 0.2,
        target_metric: "balanced".to_string(),
        max_evaluations: 50,
    };

    assert_eq!(config.population_size, 10);
    assert_eq!(config.num_generations, 5);
    assert_eq!(config.mutation_rate, 0.3);
    assert_eq!(config.crossover_rate, 0.8);
    assert_eq!(config.elite_ratio, 0.2);
    assert_eq!(config.target_metric, "balanced");
    assert_eq!(config.max_evaluations, 50);
}

#[test]
fn test_evolutionary_search_initialization() {
    let config = EvolutionarySearchConfig {
        population_size: 5,
        num_generations: 2,
        mutation_rate: 0.2,
        crossover_rate: 0.7,
        elite_ratio: 0.2,
        target_metric: "cosine_similarity".to_string(),
        max_evaluations: 10,
    };

    let search = EvolutionarySearch::new(config.clone());

    // Verify search is initialized with correct config
    assert_eq!(search.config.population_size, 5);
    assert_eq!(search.config.num_generations, 2);
}

#[test]
fn test_serialization() {
    use serde_json;

    let config = EvolutionarySearchConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: EvolutionarySearchConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.population_size, deserialized.population_size);
    assert_eq!(config.num_generations, deserialized.num_generations);
    assert_eq!(config.mutation_rate, deserialized.mutation_rate);
    assert_eq!(config.target_metric, deserialized.target_metric);
}

#[test]
fn test_individual_serialization() {
    use serde_json;

    let mut layer_group_sizes = HashMap::new();
    layer_group_sizes.insert("layer1".to_string(), 64);
    layer_group_sizes.insert("layer2".to_string(), 128);

    let individual = Individual {
        layer_group_sizes,
        fitness: 0.85,
        metrics: None,
    };

    let json = serde_json::to_string(&individual).unwrap();
    let deserialized: Individual = serde_json::from_str(&json).unwrap();

    assert_eq!(individual.fitness, deserialized.fitness);
    assert_eq!(
        individual.layer_group_sizes.len(),
        deserialized.layer_group_sizes.len()
    );
}

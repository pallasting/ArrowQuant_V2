// Evolutionary search for optimal layer-wise group sizes
// Uses genetic algorithm to find optimal quantization configurations

use crate::config::DiffusionQuantConfig;
use crate::errors::QuantError;
use crate::orchestrator::{DiffusionOrchestrator, QuantizationResult};
use fastrand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Configuration for evolutionary search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionarySearchConfig {
    /// Population size for genetic algorithm
    pub population_size: usize,
    /// Number of generations to evolve
    pub num_generations: usize,
    /// Mutation rate (0.0 to 1.0)
    pub mutation_rate: f32,
    /// Crossover rate (0.0 to 1.0)
    pub crossover_rate: f32,
    /// Elite selection ratio (top % to preserve)
    pub elite_ratio: f32,
    /// Target metric to optimize ("fid", "accuracy", "cosine_similarity")
    pub target_metric: String,
    /// Maximum search budget (number of evaluations)
    pub max_evaluations: usize,
}

impl Default for EvolutionarySearchConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            num_generations: 10,
            mutation_rate: 0.2,
            crossover_rate: 0.7,
            elite_ratio: 0.1,
            target_metric: "cosine_similarity".to_string(),
            max_evaluations: 200,
        }
    }
}

/// Individual in the population (chromosome)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    /// Layer-wise group sizes (layer_name -> group_size)
    pub layer_group_sizes: HashMap<String, usize>,
    /// Fitness score (higher is better)
    pub fitness: f32,
    /// Quantization result metrics
    pub metrics: Option<QuantizationResult>,
}

impl Individual {
    /// Create a new random individual
    pub fn random(layer_names: &[String], rng: &mut Rng) -> Self {
        let valid_group_sizes = vec![32, 64, 128, 256];
        let mut layer_group_sizes = HashMap::new();
        
        for layer_name in layer_names {
            let idx = rng.usize(0..valid_group_sizes.len());
            layer_group_sizes.insert(layer_name.clone(), valid_group_sizes[idx]);
        }
        
        Self {
            layer_group_sizes,
            fitness: 0.0,
            metrics: None,
        }
    }
    
    /// Mutate the individual
    pub fn mutate(&mut self, mutation_rate: f32, rng: &mut Rng) {
        let valid_group_sizes = vec![32, 64, 128, 256];
        
        for (_layer_name, group_size) in self.layer_group_sizes.iter_mut() {
            if rng.f32() < mutation_rate {
                let idx = rng.usize(0..valid_group_sizes.len());
                *group_size = valid_group_sizes[idx];
            }
        }
    }
    
    /// Crossover with another individual
    pub fn crossover(&self, other: &Individual, rng: &mut Rng) -> Individual {
        let mut child_layer_group_sizes = HashMap::new();
        
        for (layer_name, group_size) in &self.layer_group_sizes {
            // 50% chance to inherit from each parent
            if rng.bool() {
                child_layer_group_sizes.insert(layer_name.clone(), *group_size);
            } else if let Some(other_size) = other.layer_group_sizes.get(layer_name) {
                child_layer_group_sizes.insert(layer_name.clone(), *other_size);
            } else {
                child_layer_group_sizes.insert(layer_name.clone(), *group_size);
            }
        }
        
        Individual {
            layer_group_sizes: child_layer_group_sizes,
            fitness: 0.0,
            metrics: None,
        }
    }
}

/// Evolutionary search optimizer
pub struct EvolutionarySearch {
    pub config: EvolutionarySearchConfig,
    rng: Rng,
    evaluations: usize,
}

impl EvolutionarySearch {
    /// Create a new evolutionary search optimizer
    pub fn new(config: EvolutionarySearchConfig) -> Self {
        Self {
            config,
            rng: Rng::new(),
            evaluations: 0,
        }
    }
    
    /// Run evolutionary search to find optimal layer-wise group sizes
    pub fn search(
        &mut self,
        model_path: &Path,
        output_base_path: &Path,
        base_config: &DiffusionQuantConfig,
        layer_names: &[String],
    ) -> Result<SearchResult, QuantError> {
        log::info!("Starting evolutionary search with {} generations, population size {}",
                   self.config.num_generations, self.config.population_size);
        
        // Initialize population
        let mut population = self.initialize_population(layer_names);
        
        // Track best individual across all generations
        let mut best_individual: Option<Individual> = None;
        let mut generation_history = Vec::new();
        
        // Evolve for specified number of generations
        for generation in 0..self.config.num_generations {
            log::info!("Generation {}/{}", generation + 1, self.config.num_generations);
            
            // Evaluate fitness for all individuals
            self.evaluate_population(
                &mut population,
                model_path,
                output_base_path,
                base_config,
                generation,
            )?;
            
            // Check if we've exceeded evaluation budget
            if self.evaluations >= self.config.max_evaluations {
                log::warn!("Reached maximum evaluation budget: {}", self.config.max_evaluations);
                break;
            }
            
            // Sort by fitness (descending)
            population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
            
            // Track best individual
            if let Some(current_best) = population.first() {
                if best_individual.is_none() || current_best.fitness > best_individual.as_ref().unwrap().fitness {
                    best_individual = Some(current_best.clone());
                    log::info!("New best fitness: {:.4}", current_best.fitness);
                }
            }
            
            // Record generation statistics
            let gen_stats = GenerationStats {
                generation,
                best_fitness: population[0].fitness,
                avg_fitness: population.iter().map(|i| i.fitness).sum::<f32>() / population.len() as f32,
                worst_fitness: population.last().unwrap().fitness,
            };
            generation_history.push(gen_stats);
            
            // Create next generation
            population = self.create_next_generation(&population);
        }
        
        // Return best result
        let best = best_individual.ok_or_else(|| {
            QuantError::EvolutionarySearchError("No valid individuals found".to_string())
        })?;
        
        log::info!("Evolutionary search complete. Best fitness: {:.4}", best.fitness);
        log::info!("Total evaluations: {}", self.evaluations);
        
        Ok(SearchResult {
            best_individual: best,
            generation_history,
            total_evaluations: self.evaluations,
        })
    }
    
    /// Initialize random population
    fn initialize_population(&mut self, layer_names: &[String]) -> Vec<Individual> {
        (0..self.config.population_size)
            .map(|_| Individual::random(layer_names, &mut self.rng))
            .collect()
    }
    
    /// Evaluate fitness for all individuals in population
    fn evaluate_population(
        &mut self,
        population: &mut [Individual],
        model_path: &Path,
        output_base_path: &Path,
        base_config: &DiffusionQuantConfig,
        generation: usize,
    ) -> Result<(), QuantError> {
        let pop_len = population.len();
        for (idx, individual) in population.iter_mut().enumerate() {
            // Skip if already evaluated
            if individual.metrics.is_some() {
                continue;
            }
            
            // Check evaluation budget
            if self.evaluations >= self.config.max_evaluations {
                break;
            }
            
            log::debug!("Evaluating individual {}/{} in generation {}", 
                       idx + 1, pop_len, generation);
            
            // Create config with layer-wise group sizes
            let mut config = base_config.clone();
            
            // Apply layer-wise group sizes (simplified: use average for now)
            let avg_group_size = individual.layer_group_sizes.values().sum::<usize>() 
                / individual.layer_group_sizes.len().max(1);
            config.group_size = avg_group_size;
            
            // Create output path for this individual
            let output_path = output_base_path.join(format!("gen{}_ind{}", generation, idx));
            
            // Quantize model with this configuration
            match DiffusionOrchestrator::new(config) {
                Ok(orchestrator) => {
                    match orchestrator.quantize_model(model_path, &output_path) {
                        Ok(result) => {
                            // Compute fitness based on target metric
                            individual.fitness = self.compute_fitness(&result);
                            individual.metrics = Some(result);
                            self.evaluations += 1;
                        }
                        Err(e) => {
                            log::warn!("Quantization failed for individual {}: {}", idx, e);
                            individual.fitness = 0.0;
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to create orchestrator for individual {}: {}", idx, e);
                    individual.fitness = 0.0;
                }
            }
        }
        
        Ok(())
    }
    
    /// Compute fitness score from quantization result
    fn compute_fitness(&self, result: &QuantizationResult) -> f32 {
        match self.config.target_metric.as_str() {
            "cosine_similarity" => result.cosine_similarity,
            "accuracy" => result.cosine_similarity, // Alias
            "compression_ratio" => {
                // Normalize compression ratio to 0-1 range
                (result.compression_ratio / 20.0).min(1.0)
            }
            "balanced" => {
                // Balance accuracy and compression
                let accuracy_score = result.cosine_similarity;
                let compression_score = (result.compression_ratio / 20.0).min(1.0);
                (accuracy_score + compression_score) / 2.0
            }
            _ => result.cosine_similarity,
        }
    }
    
    /// Create next generation using selection, crossover, and mutation
    fn create_next_generation(&mut self, population: &[Individual]) -> Vec<Individual> {
        let mut next_generation = Vec::new();
        
        // Elite selection: preserve top individuals
        let elite_count = (population.len() as f32 * self.config.elite_ratio).ceil() as usize;
        for i in 0..elite_count.min(population.len()) {
            next_generation.push(population[i].clone());
        }
        
        // Fill rest with crossover and mutation
        while next_generation.len() < self.config.population_size {
            // Tournament selection
            let parent1 = self.tournament_select(population);
            let parent2 = self.tournament_select(population);
            
            // Crossover
            let mut child = if self.rng.f32() < self.config.crossover_rate {
                parent1.crossover(parent2, &mut self.rng)
            } else {
                parent1.clone()
            };
            
            // Mutation
            child.mutate(self.config.mutation_rate, &mut self.rng);
            
            next_generation.push(child);
        }
        
        next_generation
    }
    
    /// Tournament selection: select best from random subset
    fn tournament_select<'a>(&mut self, population: &'a [Individual]) -> &'a Individual {
        let tournament_size = 3;
        let mut best: Option<&Individual> = None;
        
        for _ in 0..tournament_size {
            let idx = self.rng.usize(0..population.len());
            let candidate = &population[idx];
            
            if best.is_none() || candidate.fitness > best.unwrap().fitness {
                best = Some(candidate);
            }
        }
        
        best.unwrap()
    }
}

/// Result of evolutionary search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Best individual found
    pub best_individual: Individual,
    /// History of generation statistics
    pub generation_history: Vec<GenerationStats>,
    /// Total number of evaluations performed
    pub total_evaluations: usize,
}

/// Statistics for a generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub worst_fitness: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_individual_creation() {
        let layer_names = vec!["layer1".to_string(), "layer2".to_string()];
        let mut rng = Rng::new();
        let individual = Individual::random(&layer_names, &mut rng);
        
        assert_eq!(individual.layer_group_sizes.len(), 2);
        assert!(individual.layer_group_sizes.contains_key("layer1"));
        assert!(individual.layer_group_sizes.contains_key("layer2"));
        
        // Check valid group sizes
        for size in individual.layer_group_sizes.values() {
            assert!([32, 64, 128, 256].contains(size));
        }
    }
    
    #[test]
    fn test_mutation() {
        let layer_names = vec!["layer1".to_string()];
        let mut rng = Rng::new();
        let mut individual = Individual::random(&layer_names, &mut rng);
        let original_size = *individual.layer_group_sizes.get("layer1").unwrap();
        
        // Mutate with 100% rate
        individual.mutate(1.0, &mut rng);
        
        // Size should be valid
        let new_size = *individual.layer_group_sizes.get("layer1").unwrap();
        assert!([32, 64, 128, 256].contains(&new_size));
    }
    
    #[test]
    fn test_crossover() {
        let layer_names = vec!["layer1".to_string(), "layer2".to_string()];
        let mut rng = Rng::new();
        let parent1 = Individual::random(&layer_names, &mut rng);
        let parent2 = Individual::random(&layer_names, &mut rng);
        
        let child = parent1.crossover(&parent2, &mut rng);
        
        assert_eq!(child.layer_group_sizes.len(), 2);
        
        // Child should have valid group sizes
        for size in child.layer_group_sizes.values() {
            assert!([32, 64, 128, 256].contains(size));
        }
    }
    
    #[test]
    fn test_evolutionary_search_config() {
        let config = EvolutionarySearchConfig::default();
        
        assert_eq!(config.population_size, 20);
        assert_eq!(config.num_generations, 10);
        assert_eq!(config.mutation_rate, 0.2);
        assert_eq!(config.crossover_rate, 0.7);
        assert_eq!(config.elite_ratio, 0.1);
        assert_eq!(config.target_metric, "cosine_similarity");
        assert_eq!(config.max_evaluations, 200);
    }
}

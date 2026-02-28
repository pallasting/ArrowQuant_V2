//! Time-aware quantization for diffusion models

use crate::errors::Result;
use crate::config::ThermodynamicConfig;
use crate::thermodynamic::{MarkovValidator, ThermodynamicMetrics, BoundarySmoother};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// Parameters for a single time group
///
/// Stores quantization parameters for a specific time range in diffusion models.
/// Each time group has its own scale, zero_point, and group_size to handle
/// temporal variance in activation distributions.
///
/// # Fields
///
/// - `time_range`: Start and end timestep indices for this group
/// - `scale`: Quantization scale factor
/// - `zero_point`: Quantization zero point offset
/// - `group_size`: Number of elements per quantization group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeGroupParams {
    pub time_range: (usize, usize),
    pub scale: f32,
    pub zero_point: f32,
    pub group_size: usize,
}

/// Activation statistics for time-aware quantization
#[derive(Debug, Clone)]
pub struct ActivationStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

impl ActivationStats {
    /// Slice statistics for a time range
    pub fn slice(&self, start: usize, end: usize) -> GroupStats {
        let slice_min = self.min[start..end]
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let slice_max = self.max[start..end]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        GroupStats {
            min: slice_min,
            max: slice_max,
        }
    }
}

/// Statistics for a time group
#[derive(Debug, Clone)]
pub struct GroupStats {
    pub min: f32,
    pub max: f32,
}

/// Time-aware quantizer for handling temporal variance
#[derive(Clone)]
pub struct TimeAwareQuantizer {
    num_time_groups: usize,
    time_group_boundaries: Vec<usize>,
    thermodynamic_config: Option<ThermodynamicConfig>,
    /// Metrics from last quantization (if thermodynamic validation enabled)
    last_metrics: Arc<Mutex<Option<ThermodynamicMetrics>>>,
}

impl TimeAwareQuantizer {
    /// Create new time-aware quantizer
    pub fn new(num_time_groups: usize) -> Self {
        Self {
            num_time_groups,
            time_group_boundaries: Vec::new(),
            thermodynamic_config: None,
            last_metrics: Arc::new(Mutex::new(None)),
        }
    }

    /// Create new time-aware quantizer with thermodynamic configuration
    pub fn with_thermodynamic_config(num_time_groups: usize, config: ThermodynamicConfig) -> Self {
        Self {
            num_time_groups,
            time_group_boundaries: Vec::new(),
            thermodynamic_config: Some(config),
            last_metrics: Arc::new(Mutex::new(None)),
        }
    }

    /// Get the time group boundaries
    pub fn time_group_boundaries(&self) -> &[usize] {
        &self.time_group_boundaries
    }

    /// Group timesteps into time groups
    pub fn group_timesteps(&mut self, num_timesteps: usize) {
        let mut boundaries = vec![0];

        // Handle edge case: if num_timesteps < num_groups, adjust num_groups
        let effective_groups = self.num_time_groups.min(num_timesteps);

        for i in 1..effective_groups {
            let boundary = (i * num_timesteps) / effective_groups;
            boundaries.push(boundary);
        }
        boundaries.push(num_timesteps);

        self.time_group_boundaries = boundaries;
        // Update num_time_groups to reflect actual groups created
        self.num_time_groups = effective_groups;
    }

    /// Compute quantization parameters for each time group
    pub fn compute_params_per_group(
        &self,
        activation_stats: &ActivationStats,
    ) -> Vec<TimeGroupParams> {
        let mut params = Vec::new();

        for i in 0..self.num_time_groups {
            let start = self.time_group_boundaries[i];
            let end = self.time_group_boundaries[i + 1];

            // Compute statistics for this time group
            let group_stats = activation_stats.slice(start, end);

            // Compute quantization parameters
            // Handle edge case: if min == max, use a small epsilon to avoid division by zero
            let range = group_stats.max - group_stats.min;
            let scale = if range > 0.0 {
                range / 255.0
            } else {
                // If all values are the same, use a small scale
                1e-6
            };
            let zero_point = -group_stats.min / scale;

            // Adaptive group size based on timestep
            // Early timesteps (high noise): coarser quantization
            // Late timesteps (low noise): finer quantization
            let group_size = if i < self.num_time_groups / 2 {
                256 // Early timesteps
            } else {
                64 // Late timesteps
            };

            params.push(TimeGroupParams {
                time_range: (start, end),
                scale,
                zero_point,
                group_size,
            });
        }

        params
    }

    /// Quantize layer with time-group-specific parameters
    pub fn quantize_layer(
        &self,
        weights: &[f32],
        time_group_params: &[TimeGroupParams],
    ) -> Result<QuantizedLayer> {
        // Start with the original parameters
        let mut params = time_group_params.to_vec();
        
        // Phase 1: Validate Markov properties (if enabled)
        if let Some(ref config) = self.thermodynamic_config {
            if config.validation.enabled {
                let validator = MarkovValidator::new(config.validation.smoothness_threshold)
                    .with_logging(config.validation.log_violations);
                
                let validation_result = validator.validate(&params);
                
                // Store metrics for later retrieval
                let metrics = ThermodynamicMetrics::from_validation(&validation_result);
                if let Ok(mut last_metrics) = self.last_metrics.lock() {
                    *last_metrics = Some(metrics);
                }
                
                // Log validation results
                if config.validation.log_violations {
                    if !validation_result.is_valid {
                        log::warn!(
                            "Markov validation detected {} violations (smoothness score: {:.3})",
                            validation_result.violations.len(),
                            validation_result.smoothness_score
                        );
                    } else {
                        log::info!(
                            "Markov validation passed (smoothness score: {:.3})",
                            validation_result.smoothness_score
                        );
                    }
                }
                
                // Note: Validation does NOT modify quantization behavior
                // It only monitors and logs violations
            }
            
            // Phase 2: Smooth boundaries (if enabled)
            if config.boundary_smoothing.enabled {
                let smoother = BoundarySmoother::new(
                    config.boundary_smoothing.window_size,
                    config.boundary_smoothing.interpolation.into(),
                );
                
                params = smoother.smooth_boundaries(&params);
                
                log::info!(
                    "Applied boundary smoothing (window_size={}, method={})",
                    config.boundary_smoothing.window_size,
                    config.boundary_smoothing.interpolation
                );
            }
            
            // Phase 3: Optimize transitions (if enabled)
            if config.transition_optimization.enabled {
                use crate::thermodynamic::optimizer::{TransitionOptimizer, OptimizerConfig};
                use ndarray::Array2;
                
                // Create optimizer configuration
                let optimizer_config = OptimizerConfig {
                    learning_rate: config.transition_optimization.learning_rate,
                    max_iterations: config.transition_optimization.max_iterations,
                    convergence_threshold: config.transition_optimization.convergence_threshold,
                    gradient_clip: 1.0,
                    markov_weight: config.transition_optimization.markov_weight,
                    entropy_weight: config.transition_optimization.entropy_weight,
                    beta_schedule: config.transition_optimization.beta_schedule.into(),
                };
                
                let optimizer = TransitionOptimizer::new(optimizer_config);
                
                // Convert weights to ndarray Array2
                // Assuming weights are 1D, reshape to 2D (1 x N)
                let weights_array = Array2::from_shape_vec(
                    (1, weights.len()),
                    weights.to_vec(),
                ).map_err(|e| crate::errors::QuantError::QuantizationFailed(
                    format!("Failed to convert weights to array: {}", e)
                ))?;
                
                // Optimize parameters (optimizer is now stateless, no mut needed)
                let opt_result = optimizer.optimize_params(&weights_array, &params)?;
                
                params = opt_result.params;
                
                log::info!(
                    "Applied transition optimization (iterations={}, final_loss={:.6}, converged={})",
                    opt_result.iterations,
                    opt_result.final_loss,
                    opt_result.converged
                );
                
                // Update metrics with optimization info
                if let Ok(mut last_metrics) = self.last_metrics.lock() {
                    if let Some(ref mut metrics) = *last_metrics {
                        metrics.optimization_iterations = opt_result.iterations;
                        metrics.optimization_converged = opt_result.converged;
                        metrics.final_loss = opt_result.final_loss;
                    }
                }
            }
        }
        
        // Quantization logic: Use a single robust quantization instead of replicating per group.
        // We use the first group as a baseline or compute a global one.
        // For now, we use a robust global scale derived from the range of all params to avoid 10x bloat.
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for p in &params {
            let start = p.time_range.0;
            let end = p.time_range.1;
            // Note: We'd ideally want activation-weighted ranges here, but for weights
            // we can just use the union of group ranges.
            min_val = min_val.min(-p.zero_point * p.scale);
            max_val = max_val.max((255.0 - p.zero_point) * p.scale);
        }

        // Compute global parameters for the actual data storage
        let range = max_val - min_val;
        let global_scale = if range > 0.0 { range / 255.0 } else { 1e-6 };
        let global_zero_point = -min_val / global_scale;

        // Perform quantization
        let quantized_data = self.quantize_with_params(
            weights,
            global_scale,
            global_zero_point,
            params[0].group_size,
        );

        // We return the GLOBAL scale/zero_point as the primary ones for the data column
        // The temporal ones are safely tucked away in the metadata (time_group_params)
        Ok(QuantizedLayer {
            data: quantized_data,
            scales: vec![global_scale],
            zero_points: vec![global_zero_point],
            time_group_params: params,
        })
    }

    /// Quantize with specific parameters
    fn quantize_with_params(
        &self,
        weights: &[f32],
        scale: f32,
        zero_point: f32,
        _group_size: usize,
    ) -> Vec<u8> {
        weights
            .iter()
            .map(|&w| {
                let q = ((w / scale) + zero_point).round().clamp(0.0, 255.0);
                q as u8
            })
            .collect()
    }

    /// Get thermodynamic metrics from last quantization
    ///
    /// Returns metrics if thermodynamic validation was enabled and quantization
    /// has been performed at least once.
    ///
    /// # Returns
    ///
    /// `Some(ThermodynamicMetrics)` if metrics are available, `None` otherwise
    pub fn get_thermodynamic_metrics(&self) -> Option<ThermodynamicMetrics> {
        self.last_metrics.lock().ok()?.clone()
    }
}

/// Quantized layer with time-aware parameters
#[derive(Debug, Clone)]
pub struct QuantizedLayer {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub zero_points: Vec<f32>,
    pub time_group_params: Vec<TimeGroupParams>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // Unit Tests
    // ============================================================================

    #[test]
    fn test_timestep_grouping() {
        let mut quantizer = TimeAwareQuantizer::new(10);
        quantizer.group_timesteps(1000);

        assert_eq!(quantizer.time_group_boundaries.len(), 11);
        assert_eq!(quantizer.time_group_boundaries[0], 0);
        assert_eq!(quantizer.time_group_boundaries[10], 1000);

        // Verify exact boundaries for default case (10 groups, 1000 timesteps)
        let expected = vec![0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
        assert_eq!(quantizer.time_group_boundaries, expected);
    }

    #[test]
    fn test_adaptive_group_size() {
        let _quantizer = TimeAwareQuantizer::new(10);
        let stats = ActivationStats {
            mean: vec![0.0; 1000],
            std: vec![1.0; 1000],
            min: vec![-1.0; 1000],
            max: vec![1.0; 1000],
        };

        let mut quantizer_mut = TimeAwareQuantizer::new(10);
        quantizer_mut.group_timesteps(1000);
        let params = quantizer_mut.compute_params_per_group(&stats);

        // Early groups should have larger group size
        assert_eq!(params[0].group_size, 256);
        // Late groups should have smaller group size
        assert_eq!(params[9].group_size, 64);
    }

    #[test]
    fn test_different_group_counts() {
        // Test with 5 groups
        let mut quantizer = TimeAwareQuantizer::new(5);
        quantizer.group_timesteps(1000);
        assert_eq!(quantizer.time_group_boundaries.len(), 6);
        assert_eq!(
            quantizer.time_group_boundaries,
            vec![0, 200, 400, 600, 800, 1000]
        );

        // Test with 20 groups
        let mut quantizer = TimeAwareQuantizer::new(20);
        quantizer.group_timesteps(1000);
        assert_eq!(quantizer.time_group_boundaries.len(), 21);
        assert_eq!(quantizer.time_group_boundaries[0], 0);
        assert_eq!(quantizer.time_group_boundaries[1], 50);
        assert_eq!(quantizer.time_group_boundaries[20], 1000);
    }

    #[test]
    fn test_non_standard_timesteps() {
        // Test with 500 timesteps
        let mut quantizer = TimeAwareQuantizer::new(10);
        quantizer.group_timesteps(500);
        assert_eq!(quantizer.time_group_boundaries.len(), 11);
        assert_eq!(
            quantizer.time_group_boundaries,
            vec![0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        );
    }

    #[test]
    fn test_quantize_layer_basic() {
        let mut quantizer = TimeAwareQuantizer::new(2);
        quantizer.group_timesteps(100);

        // Create simple activation stats
        let stats = ActivationStats {
            mean: vec![0.0; 100],
            std: vec![1.0; 100],
            min: vec![-2.0; 100],
            max: vec![2.0; 100],
        };

        let params = quantizer.compute_params_per_group(&stats);

        // Create test weights
        let weights = vec![0.0, 1.0, -1.0, 2.0, -2.0];

        // Quantize layer
        let result = quantizer.quantize_layer(&weights, &params).unwrap();

        // Verify output structure
        assert_eq!(result.scales.len(), 2); // 2 time groups
        assert_eq!(result.zero_points.len(), 2);
        assert_eq!(result.time_group_params.len(), 2);
        assert!(!result.data.is_empty());
    }

    #[test]
    fn test_quantize_layer_stores_params() {
        let mut quantizer = TimeAwareQuantizer::new(3);
        quantizer.group_timesteps(300);

        let stats = ActivationStats {
            mean: vec![0.0; 300],
            std: vec![1.0; 300],
            min: vec![-1.0; 300],
            max: vec![1.0; 300],
        };

        let params = quantizer.compute_params_per_group(&stats);
        let weights = vec![0.5; 100];

        let result = quantizer.quantize_layer(&weights, &params).unwrap();

        // Verify time_group_params are stored correctly
        assert_eq!(result.time_group_params.len(), 3);
        assert_eq!(result.time_group_params[0].time_range, (0, 100));
        assert_eq!(result.time_group_params[1].time_range, (100, 200));
        assert_eq!(result.time_group_params[2].time_range, (200, 300));

        // Verify adaptive group sizes
        assert_eq!(result.time_group_params[0].group_size, 256); // Early
        assert_eq!(result.time_group_params[1].group_size, 64); // Late
        assert_eq!(result.time_group_params[2].group_size, 64); // Late
    }

    #[test]
    fn test_quantize_layer_applies_correct_params() {
        let mut quantizer = TimeAwareQuantizer::new(1);
        quantizer.group_timesteps(10);

        let stats = ActivationStats {
            mean: vec![0.0; 10],
            std: vec![1.0; 10],
            min: vec![0.0; 10],
            max: vec![10.0; 10],
        };

        let params = quantizer.compute_params_per_group(&stats);

        // Test weights that should quantize predictably
        let weights = vec![0.0, 5.0, 10.0];

        let result = quantizer.quantize_layer(&weights, &params).unwrap();

        // With min=0, max=10, scale should be 10/255 ≈ 0.0392
        // zero_point should be 0
        // So: 0.0 -> 0, 5.0 -> ~127, 10.0 -> 255
        assert_eq!(result.data[0], 0);
        assert!(result.data[1] > 120 && result.data[1] < 135); // ~127
        assert_eq!(result.data[2], 255);
    }

    #[test]
    fn test_quantize_layer_with_negative_values() {
        let mut quantizer = TimeAwareQuantizer::new(1);
        quantizer.group_timesteps(10);

        let stats = ActivationStats {
            mean: vec![0.0; 10],
            std: vec![1.0; 10],
            min: vec![-5.0; 10],
            max: vec![5.0; 10],
        };

        let params = quantizer.compute_params_per_group(&stats);
        let weights = vec![-5.0, 0.0, 5.0];

        let result = quantizer.quantize_layer(&weights, &params).unwrap();

        // Min value should map to ~0, max to ~255
        assert!(result.data[0] < 10); // -5.0 -> near 0
        assert!(result.data[2] > 245); // 5.0 -> near 255
    }

    #[test]
    fn test_quantize_layer_multiple_groups() {
        let mut quantizer = TimeAwareQuantizer::new(5);
        quantizer.group_timesteps(500);

        let stats = ActivationStats {
            mean: vec![0.0; 500],
            std: vec![1.0; 500],
            min: vec![-1.0; 500],
            max: vec![1.0; 500],
        };

        let params = quantizer.compute_params_per_group(&stats);
        let weights = vec![0.0; 50];

        let result = quantizer.quantize_layer(&weights, &params).unwrap();

        // Should have 5 sets of parameters
        assert_eq!(result.scales.len(), 5);
        assert_eq!(result.zero_points.len(), 5);
        assert_eq!(result.time_group_params.len(), 5);

        // Data should NOT be replicated for all groups (1x size)
        assert_eq!(result.data.len(), weights.len());
    }

    #[test]
    fn test_quantize_with_params_clamping() {
        let quantizer = TimeAwareQuantizer::new(1);

        // Test that extreme values are clamped to [0, 255]
        let weights = vec![1000.0, -1000.0, 0.0];
        let scale = 1.0;
        let zero_point = 127.0;

        let result = quantizer.quantize_with_params(&weights, scale, zero_point, 128);

        // Verify clamping behavior
        assert_eq!(result[0], 255); // 1000.0 clamped to 255
        assert_eq!(result[1], 0); // -1000.0 clamped to 0
    }

    // ============================================================================
    // Property-Based Tests
    // ============================================================================

    #[cfg(test)]
    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// **Validates: Requirements 1.1**
            ///
            /// Property: Time groups should cover all timesteps without gaps or overlaps
            ///
            /// This property test verifies that:
            /// 1. Time groups start at 0
            /// 2. Time groups end at num_timesteps
            /// 3. There are no gaps between groups
            /// 4. There are no overlaps between groups
            /// 5. Each timestep belongs to exactly one group
            #[test]
            fn prop_time_groups_cover_all_timesteps(
                num_timesteps in 1usize..10000,
                num_groups in 1usize..100
            ) {
                let mut quantizer = TimeAwareQuantizer::new(num_groups);
                quantizer.group_timesteps(num_timesteps);

                let boundaries = &quantizer.time_group_boundaries;

                // Effective groups is min(num_groups, num_timesteps)
                let effective_groups = num_groups.min(num_timesteps);

                // Property 1: First boundary should be 0
                prop_assert_eq!(boundaries[0], 0, "First boundary must be 0");

                // Property 2: Last boundary should be num_timesteps
                prop_assert_eq!(
                    boundaries[boundaries.len() - 1],
                    num_timesteps,
                    "Last boundary must equal num_timesteps"
                );

                // Property 3: Should have exactly effective_groups + 1 boundaries
                prop_assert_eq!(
                    boundaries.len(),
                    effective_groups + 1,
                    "Should have effective_groups + 1 boundaries"
                );

                // Property 4: Boundaries should be strictly increasing (no gaps or overlaps)
                for i in 0..boundaries.len() - 1 {
                    prop_assert!(
                        boundaries[i] < boundaries[i + 1],
                        "Boundaries must be strictly increasing: {} >= {} at index {}",
                        boundaries[i],
                        boundaries[i + 1],
                        i
                    );
                }

                // Property 5: Each timestep should belong to exactly one group
                // We verify this by checking that consecutive boundaries are adjacent
                // and that the total range equals num_timesteps
                let total_coverage: usize = (0..effective_groups)
                    .map(|i| boundaries[i + 1] - boundaries[i])
                    .sum();

                prop_assert_eq!(
                    total_coverage,
                    num_timesteps,
                    "Total coverage of all groups must equal num_timesteps"
                );

                // Property 6: No group should be empty (each group covers at least 1 timestep)
                for i in 0..effective_groups {
                    let group_size = boundaries[i + 1] - boundaries[i];
                    prop_assert!(
                        group_size > 0,
                        "Group {} is empty (size = 0), boundaries: [{}, {}]",
                        i,
                        boundaries[i],
                        boundaries[i + 1]
                    );
                }
            }

            /// Property: Computed parameters should have valid ranges
            ///
            /// This test verifies that quantization parameters computed from
            /// activation statistics are always valid and within expected ranges.
            #[test]
            fn prop_params_have_valid_ranges(
                num_timesteps in 10usize..1000,
                num_groups in 1usize..50,
                min_val in -100.0f32..0.0f32,
                max_val in 0.0f32..100.0f32
            ) {
                let mut quantizer = TimeAwareQuantizer::new(num_groups);
                quantizer.group_timesteps(num_timesteps);

                let effective_groups = num_groups.min(num_timesteps);

                // Create activation stats with consistent min/max
                let stats = ActivationStats {
                    mean: vec![0.0; num_timesteps],
                    std: vec![1.0; num_timesteps],
                    min: vec![min_val; num_timesteps],
                    max: vec![max_val; num_timesteps],
                };

                let params = quantizer.compute_params_per_group(&stats);

                // Property 1: Should have one param set per effective group
                prop_assert_eq!(params.len(), effective_groups);

                // Property 2: All scales should be positive and finite
                for (i, param) in params.iter().enumerate() {
                    prop_assert!(
                        param.scale > 0.0 && param.scale.is_finite(),
                        "Scale at group {} is invalid: {}",
                        i,
                        param.scale
                    );
                }

                // Property 3: All zero_points should be finite
                for (i, param) in params.iter().enumerate() {
                    prop_assert!(
                        param.zero_point.is_finite(),
                        "Zero point at group {} is invalid: {}",
                        i,
                        param.zero_point
                    );
                }

                // Property 4: Group sizes should be valid (64 or 256)
                for (i, param) in params.iter().enumerate() {
                    prop_assert!(
                        param.group_size == 64 || param.group_size == 256,
                        "Group size at group {} is invalid: {}",
                        i,
                        param.group_size
                    );
                }

                // Property 5: Time ranges should match boundaries
                for (i, param) in params.iter().enumerate() {
                    let expected_start = quantizer.time_group_boundaries[i];
                    let expected_end = quantizer.time_group_boundaries[i + 1];
                    prop_assert_eq!(
                        param.time_range,
                        (expected_start, expected_end),
                        "Time range mismatch at group {}",
                        i
                    );
                }
            }

            /// Property: Quantization should preserve data shape
            ///
            /// This test verifies that quantizing a layer produces output
            /// with the expected structure and dimensions.
            #[test]
            fn prop_quantization_preserves_structure(
                num_groups in 1usize..20,
                num_timesteps in 10usize..500,
                weight_size in 1usize..1000
            ) {
                let mut quantizer = TimeAwareQuantizer::new(num_groups);
                quantizer.group_timesteps(num_timesteps);

                let effective_groups = num_groups.min(num_timesteps);

                let stats = ActivationStats {
                    mean: vec![0.0; num_timesteps],
                    std: vec![1.0; num_timesteps],
                    min: vec![-1.0; num_timesteps],
                    max: vec![1.0; num_timesteps],
                };

                let params = quantizer.compute_params_per_group(&stats);
                let weights = vec![0.5; weight_size];

                let result = quantizer.quantize_layer(&weights, &params).unwrap();

                // Property 1: Should have one scale per effective group
                prop_assert_eq!(result.scales.len(), effective_groups);

                // Property 2: Should have one zero_point per effective group
                prop_assert_eq!(result.zero_points.len(), effective_groups);

                // Property 3: Should have one param set per effective group
                prop_assert_eq!(result.time_group_params.len(), effective_groups);

                // Property 4: Quantized data should have expected size
                // We no longer replicate weights per group, so size = weight_size
                prop_assert_eq!(result.data.len(), weight_size);

                // Property 5: All quantized values should be valid u8 (always true by type)
                // This is guaranteed by the u8 type, but we verify data is not empty
                prop_assert!(!result.data.is_empty(), "Quantized data should not be empty");
            }
        }
    }
}

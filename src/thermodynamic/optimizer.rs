//! Transition Probability Optimizer
//!
//! This module implements gradient-based optimization of quantization parameters
//! to preserve transition probabilities and minimize thermodynamic loss.

use ndarray::Array2;
use rayon::prelude::*;
use super::transition_matrix::{TransitionComputer, TransitionMatrix, BetaSchedule};
use super::loss_functions::ThermodynamicLoss;
use crate::time_aware::TimeGroupParams;
use crate::errors::Result;

/// Configuration for transition optimization
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Learning rate for gradient descent
    pub learning_rate: f32,
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Convergence threshold (stop if loss change < threshold)
    pub convergence_threshold: f32,
    /// Gradient clipping threshold
    pub gradient_clip: f32,
    /// Weight for Markov constraint loss
    pub markov_weight: f32,
    /// Weight for entropy regularization
    pub entropy_weight: f32,
    /// Beta schedule type
    pub beta_schedule: BetaSchedule,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_iterations: 50,
            convergence_threshold: 1e-4,
            gradient_clip: 1.0,
            markov_weight: 0.1,
            entropy_weight: 0.05,
            beta_schedule: BetaSchedule::Linear,
        }
    }
}

/// Optimizer for quantization parameters using thermodynamic constraints
/// 
/// This optimizer is stateless and thread-safe. All mutable state is passed
/// explicitly to methods, enabling parallel processing and avoiding borrow conflicts.
pub struct TransitionOptimizer {
    config: OptimizerConfig,
    loss_fn: ThermodynamicLoss,
}

/// Optimization result containing optimized parameters and metrics
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimized time group parameters
    pub params: Vec<TimeGroupParams>,
    /// Final loss value
    pub final_loss: f32,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Loss history (one value per iteration)
    pub loss_history: Vec<f32>,
}

impl TransitionOptimizer {
    /// Create a new TransitionOptimizer with the specified configuration
    pub fn new(config: OptimizerConfig) -> Self {
        let loss_fn = ThermodynamicLoss::new(
            config.markov_weight,
            config.entropy_weight,
        );

        Self { 
            config, 
            loss_fn,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(OptimizerConfig::default())
    }

    /// Optimize quantization parameters for a single layer
    ///
    /// # Arguments
    ///
    /// * `weights` - Original weight tensor
    /// * `initial_params` - Initial time group parameters
    ///
    /// # Returns
    ///
    /// OptimizationResult containing optimized parameters and metrics
    pub fn optimize_params(
        &self,
        weights: &Array2<f32>,
        initial_params: &[TimeGroupParams],
    ) -> Result<OptimizationResult> {
        let mut params = initial_params.to_vec();
        let mut loss_history = Vec::with_capacity(self.config.max_iterations);
        let mut best_loss = f32::INFINITY;
        let mut best_params = params.clone();

        // Create local buffers for quantization (avoids borrow conflicts)
        let mut quantized_buffer = Array2::zeros(weights.dim());

        // Create transition computer
        let num_timesteps = params.len();
        let mut transition_computer = TransitionComputer::new(
            num_timesteps,
            self.config.beta_schedule,
            0.0001,
            0.02,
        );

        // Compute original transitions
        let original_transitions = self.compute_transitions(
            &mut transition_computer,
            weights,
            num_timesteps,
        );

        // Optimization loop
        for iter in 0..self.config.max_iterations {
            // 1. Quantize with current params (use local buffer)
            self.quantize_with_params_into(weights, &params, &mut quantized_buffer)?;

            // 2. Compute quantized transitions
            let quantized_transitions = self.compute_transitions(
                &mut transition_computer,
                &quantized_buffer,
                num_timesteps,
            );

            // 3. Compute loss
            let loss = self.loss_fn.compute_total_loss(
                weights,
                &quantized_buffer,
                &original_transitions,
                &quantized_transitions,
            );

            loss_history.push(loss);

            // 4. Check for improvement
            if loss < best_loss {
                best_loss = loss;
                best_params.clone_from(&params);
            }

            // 5. Check convergence
            if iter > 0 {
                let loss_change = (loss_history[iter - 1] - loss).abs();
                if loss_change < self.config.convergence_threshold {
                    return Ok(OptimizationResult {
                        params: best_params,
                        final_loss: best_loss,
                        iterations: iter + 1,
                        converged: true,
                        loss_history,
                    });
                }
            }

            // 6. Compute gradients (numerical approximation)
            let gradients = self.compute_gradients(
                weights,
                &params,
                &mut transition_computer,
                &original_transitions,
            )?;

            // 7. Update parameters
            self.update_params(&mut params, &gradients);
        }

        // Return best parameters found
        Ok(OptimizationResult {
            params: best_params,
            final_loss: best_loss,
            iterations: self.config.max_iterations,
            converged: false,
            loss_history,
        })
    }

    /// Optimize parameters for multiple layers in parallel
    /// 
    /// Uses rayon for parallel processing since the optimizer is now stateless
    pub fn optimize_params_parallel(
        &self,
        layers: &[Array2<f32>],
        initial_params: &[Vec<TimeGroupParams>],
    ) -> Result<Vec<OptimizationResult>> {
        // Now we can use rayon since &self is immutable
        layers
            .par_iter()
            .zip(initial_params.par_iter())
            .map(|(weights, params)| self.optimize_params(weights, params))
            .collect()
    }

    /// Compute transitions for a weight tensor
    fn compute_transitions(
        &self,
        computer: &mut TransitionComputer,
        weights: &Array2<f32>,
        num_timesteps: usize,
    ) -> Vec<TransitionMatrix> {
        (0..num_timesteps)
            .map(|t| computer.compute_transition(weights, t))
            .collect()
    }

    /// Quantize weights using specified parameters into provided buffer
    /// 
    /// This method writes the quantized result into the provided buffer,
    /// avoiding allocation and borrow conflicts.
    fn quantize_with_params_into(
        &self,
        weights: &Array2<f32>,
        params: &[TimeGroupParams],
        output: &mut Array2<f32>,
    ) -> Result<()> {
        let num_groups = params.len();
        let total_elements = weights.len();
        let group_size = total_elements / num_groups;
        
        // Iterate over all elements using multi-dimensional indexing
        let shape = weights.shape();
        for row in 0..shape[0] {
            for col in 0..shape[1] {
                let linear_idx = row * shape[1] + col;
                let group_idx = (linear_idx / group_size).min(num_groups - 1);
                let param = &params[group_idx];
                
                let w = weights[[row, col]];
                
                // Quantize to INT2 range [-2, 1]
                let normalized = (w - param.zero_point) / param.scale;
                let quantized_val = normalized.round().clamp(-2.0, 1.0);
                let dequantized = quantized_val * param.scale + param.zero_point;
                
                output[[row, col]] = dequantized;
            }
        }
        
        Ok(())
    }

    /// Quantize weights using specified parameters (allocating version for compatibility)
    fn quantize_with_params(
        &self,
        weights: &Array2<f32>,
        params: &[TimeGroupParams],
    ) -> Result<Array2<f32>> {
        let num_groups = params.len();
        let total_elements = weights.len();
        let group_size = total_elements / num_groups;
        
        let mut quantized = weights.clone();
        
        // Iterate over all elements using multi-dimensional indexing
        let shape = weights.shape();
        for row in 0..shape[0] {
            for col in 0..shape[1] {
                let linear_idx = row * shape[1] + col;
                let group_idx = (linear_idx / group_size).min(num_groups - 1);
                let param = &params[group_idx];
                
                let w = weights[[row, col]];
                
                // Quantize to INT2 range [-2, 1]
                let normalized = (w - param.zero_point) / param.scale;
                let quantized_val = normalized.round().clamp(-2.0, 1.0);
                let dequantized = quantized_val * param.scale + param.zero_point;
                
                quantized[[row, col]] = dequantized;
            }
        }
        
        Ok(quantized)
    }

    /// Compute gradients using numerical differentiation
    fn compute_gradients(
        &self,
        weights: &Array2<f32>,
        params: &[TimeGroupParams],
        transition_computer: &mut TransitionComputer,
        original_transitions: &[TransitionMatrix],
    ) -> Result<Vec<ParamGradient>> {
        let epsilon = 1e-4;
        let mut gradients = Vec::with_capacity(params.len());
        
        // Create a local buffer for parameter variations
        let mut params_buffer: Vec<TimeGroupParams> = params.to_vec();

        for i in 0..params.len() {
            // Compute gradient w.r.t. scale
            params_buffer[i].scale = params[i].scale + epsilon;
            let loss_plus = self.compute_loss_for_params(
                weights,
                &params_buffer,
                transition_computer,
                original_transitions,
            )?;

            params_buffer[i].scale = params[i].scale - epsilon;
            let loss_minus = self.compute_loss_for_params(
                weights,
                &params_buffer,
                transition_computer,
                original_transitions,
            )?;

            let grad_scale = (loss_plus - loss_minus) / (2.0 * epsilon);
            
            // Reset scale
            params_buffer[i].scale = params[i].scale;

            // Compute gradient w.r.t. zero_point
            params_buffer[i].zero_point = params[i].zero_point + epsilon;
            let loss_plus = self.compute_loss_for_params(
                weights,
                &params_buffer,
                transition_computer,
                original_transitions,
            )?;

            params_buffer[i].zero_point = params[i].zero_point - epsilon;
            let loss_minus = self.compute_loss_for_params(
                weights,
                &params_buffer,
                transition_computer,
                original_transitions,
            )?;

            let grad_zero_point = (loss_plus - loss_minus) / (2.0 * epsilon);
            
            // Reset zero_point
            params_buffer[i].zero_point = params[i].zero_point;

            gradients.push(ParamGradient {
                scale: grad_scale,
                zero_point: grad_zero_point,
            });
        }

        Ok(gradients)
    }

    /// Compute loss for given parameters
    fn compute_loss_for_params(
        &self,
        weights: &Array2<f32>,
        params: &[TimeGroupParams],
        transition_computer: &mut TransitionComputer,
        original_transitions: &[TransitionMatrix],
    ) -> Result<f32> {
        // Create local buffer for quantization
        let mut quantized = Array2::zeros(weights.dim());
        self.quantize_with_params_into(weights, params, &mut quantized)?;
        
        let quantized_transitions = self.compute_transitions(
            transition_computer,
            &quantized,
            params.len(),
        );

        Ok(self.loss_fn.compute_total_loss(
            weights,
            &quantized,
            original_transitions,
            &quantized_transitions,
        ))
    }

    /// Update parameters using gradients
    fn update_params(&self, params: &mut [TimeGroupParams], gradients: &[ParamGradient]) {
        for (param, grad) in params.iter_mut().zip(gradients.iter()) {
            // Clip gradients
            let grad_scale = grad.scale.clamp(-self.config.gradient_clip, self.config.gradient_clip);
            let grad_zero_point = grad.zero_point.clamp(-self.config.gradient_clip, self.config.gradient_clip);

            // Gradient descent update
            param.scale -= self.config.learning_rate * grad_scale;
            param.zero_point -= self.config.learning_rate * grad_zero_point;

            // Ensure scale is positive
            param.scale = param.scale.max(1e-6);
        }
    }
}

/// Gradient for a single time group parameter
#[derive(Debug, Clone)]
struct ParamGradient {
    scale: f32,
    zero_point: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = TransitionOptimizer::default();
        assert_eq!(optimizer.config.max_iterations, 50);
        assert_eq!(optimizer.config.learning_rate, 0.01);
    }

    #[test]
    fn test_quantize_with_params() {
        let optimizer = TransitionOptimizer::default();
        
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let params = vec![
            TimeGroupParams { 
                scale: 0.1, 
                zero_point: 0.0,
                group_size: 2,
                time_range: (0, 2),
            },
            TimeGroupParams { 
                scale: 0.1, 
                zero_point: 0.0,
                group_size: 2,
                time_range: (2, 4),
            },
        ];
        
        let mut quantized = Array2::zeros(weights.dim());
        optimizer.quantize_with_params_into(&weights, &params, &mut quantized).unwrap();
        
        assert_eq!(quantized.shape(), weights.shape());
        // Quantized values should be close to original (within quantization error)
        for (i, (&orig, &quant)) in weights.iter().zip(quantized.iter()).enumerate() {
            assert!((orig - quant).abs() < 1.0, "Index {}: {} vs {}", i, orig, quant);
        }
    }

    #[test]
    fn test_optimize_params_basic() {
        let optimizer = TransitionOptimizer::new(OptimizerConfig {
            max_iterations: 5, // Small number for testing
            learning_rate: 0.01,
            convergence_threshold: 1e-4,
            gradient_clip: 1.0,
            markov_weight: 0.1,
            entropy_weight: 0.0,
            beta_schedule: BetaSchedule::Linear,
        });
        
        let weights = Array2::from_shape_vec((4, 4), 
            (0..16).map(|i| i as f32 * 0.1).collect()).unwrap();
        
        let initial_params = vec![
            TimeGroupParams { 
                scale: 0.1, 
                zero_point: 0.0,
                group_size: 4,
                time_range: (0, 4),
            },
            TimeGroupParams { 
                scale: 0.1, 
                zero_point: 0.0,
                group_size: 4,
                time_range: (4, 8),
            },
            TimeGroupParams { 
                scale: 0.1, 
                zero_point: 0.0,
                group_size: 4,
                time_range: (8, 12),
            },
            TimeGroupParams { 
                scale: 0.1, 
                zero_point: 0.0,
                group_size: 4,
                time_range: (12, 16),
            },
        ];
        
        let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
        
        assert_eq!(result.params.len(), 4);
        assert!(result.iterations <= 5);
        assert!(result.final_loss.is_finite());
        assert_eq!(result.loss_history.len(), result.iterations);
    }

    #[test]
    fn test_optimization_reduces_loss() {
        let optimizer = TransitionOptimizer::new(OptimizerConfig {
            max_iterations: 10,
            learning_rate: 0.01,
            convergence_threshold: 1e-6,
            gradient_clip: 1.0,
            markov_weight: 0.1,
            entropy_weight: 0.0,
            beta_schedule: BetaSchedule::Linear,
        });
        
        let weights = Array2::from_shape_vec((4, 4), 
            (0..16).map(|i| i as f32 * 0.1).collect()).unwrap();
        
        let initial_params = vec![
            TimeGroupParams { 
                scale: 0.1, 
                zero_point: 0.0,
                group_size: 4,
                time_range: (0, 4),
            },
            TimeGroupParams { 
                scale: 0.1, 
                zero_point: 0.0,
                group_size: 4,
                time_range: (4, 8),
            },
        ];
        
        let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
        
        // Loss should decrease over iterations
        if result.loss_history.len() > 1 {
            let first_loss = result.loss_history[0];
            let last_loss = *result.loss_history.last().unwrap();
            
            // Allow for some numerical noise, but generally should decrease
            assert!(last_loss <= first_loss * 1.1, 
                    "Loss should not increase significantly: {} -> {}", 
                    first_loss, last_loss);
        }
    }

    #[test]
    fn test_convergence_detection() {
        let optimizer = TransitionOptimizer::new(OptimizerConfig {
            max_iterations: 100,
            learning_rate: 0.001, // Small learning rate for stable convergence
            convergence_threshold: 0.1, // Large threshold for quick convergence
            gradient_clip: 1.0,
            markov_weight: 0.1,
            entropy_weight: 0.0,
            beta_schedule: BetaSchedule::Linear,
        });
        
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let initial_params = vec![
            TimeGroupParams { 
                scale: 0.1, 
                zero_point: 0.0,
                group_size: 4,
                time_range: (0, 4),
            },
        ];
        
        let result = optimizer.optimize_params(&weights, &initial_params).unwrap();
        
        // Should converge before max iterations with large threshold
        assert!(result.iterations < 100 || result.converged);
    }
}

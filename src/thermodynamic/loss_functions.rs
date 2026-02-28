//! Thermodynamic Loss Functions
//!
//! This module implements loss functions for thermodynamic quantization optimization.
//! It combines quantization loss (MSE) with Markov constraint loss (KL divergence).

use ndarray::Array2;
use super::transition_matrix::TransitionMatrix;

/// Thermodynamic loss function combining quantization and Markov constraints
pub struct ThermodynamicLoss {
    /// Weight for Markov constraint loss (default: 0.1)
    pub markov_weight: f32,
    /// Weight for entropy regularization (default: 0.05)
    pub entropy_weight: f32,
}

impl ThermodynamicLoss {
    /// Create a new ThermodynamicLoss with specified weights
    ///
    /// # Arguments
    ///
    /// * `markov_weight` - Weight for Markov constraint loss (typically 0.05-0.2)
    /// * `entropy_weight` - Weight for entropy regularization (typically 0.01-0.1)
    pub fn new(markov_weight: f32, entropy_weight: f32) -> Self {
        Self {
            markov_weight,
            entropy_weight,
        }
    }

    /// Create with default weights
    pub fn default() -> Self {
        Self::new(0.1, 0.05)
    }

    /// Compute total loss combining all components
    ///
    /// # Arguments
    ///
    /// * `original_weights` - Original (unquantized) weights
    /// * `quantized_weights` - Quantized weights
    /// * `original_transitions` - Transition matrices for original weights
    /// * `quantized_transitions` - Transition matrices for quantized weights
    ///
    /// # Returns
    ///
    /// Total loss value (lower is better)
    pub fn compute_total_loss(
        &self,
        original_weights: &Array2<f32>,
        quantized_weights: &Array2<f32>,
        original_transitions: &[TransitionMatrix],
        quantized_transitions: &[TransitionMatrix],
    ) -> f32 {
        // Quantization loss (MSE)
        let quant_loss = self.quantization_loss(original_weights, quantized_weights);

        // Markov constraint loss (KL divergence)
        let markov_loss = self.markov_constraint_loss(
            original_transitions,
            quantized_transitions,
        );

        // Optional entropy regularization
        let entropy_loss = if self.entropy_weight > 0.0 {
            self.entropy_regularization(quantized_weights)
        } else {
            0.0
        };

        // Combine losses
        quant_loss 
            + self.markov_weight * markov_loss 
            + self.entropy_weight * entropy_loss
    }

    /// Compute quantization loss (Mean Squared Error)
    ///
    /// L_quant = MSE(W_original, W_quantized)
    pub fn quantization_loss(
        &self,
        original: &Array2<f32>,
        quantized: &Array2<f32>,
    ) -> f32 {
        assert_eq!(original.shape(), quantized.shape(), 
                   "Original and quantized weights must have same shape");

        let n = original.len() as f32;
        let mse: f32 = original
            .iter()
            .zip(quantized.iter())
            .map(|(&o, &q)| (o - q).powi(2))
            .sum::<f32>() / n;

        mse
    }

    /// Compute Markov constraint loss (KL divergence between Gaussian distributions)
    ///
    /// For Gaussian distributions N(μ₁, σ₁²) and N(μ₂, σ₂²):
    /// KL(N₁ || N₂) = log(σ₂/σ₁) + (σ₁² + (μ₁ - μ₂)²) / (2σ₂²) - 1/2
    pub fn markov_constraint_loss(
        &self,
        original: &[TransitionMatrix],
        quantized: &[TransitionMatrix],
    ) -> f32 {
        assert_eq!(original.len(), quantized.len(),
                   "Original and quantized transitions must have same length");

        if original.is_empty() {
            return 0.0;
        }

        let mut total_kl = 0.0;

        for (orig, quant) in original.iter().zip(quantized.iter()) {
            let kl = self.gaussian_kl_divergence(
                orig.mean,
                orig.std,
                quant.mean,
                quant.std,
            );
            total_kl += kl;
        }

        // Average KL divergence across all timesteps
        total_kl / original.len() as f32
    }

    /// Compute KL divergence between two Gaussian distributions
    ///
    /// KL(N(μ₁, σ₁²) || N(μ₂, σ₂²))
    fn gaussian_kl_divergence(
        &self,
        mean1: f32,
        std1: f32,
        mean2: f32,
        std2: f32,
    ) -> f32 {
        // Add small epsilon for numerical stability
        let eps = 1e-8;
        let std1 = std1.max(eps);
        let std2 = std2.max(eps);

        let var1 = std1.powi(2);
        let var2 = std2.powi(2);
        let mean_diff = mean1 - mean2;

        // KL divergence formula for Gaussians
        let kl = (std2 / std1).ln() 
               + (var1 + mean_diff.powi(2)) / (2.0 * var2) 
               - 0.5;

        kl.max(0.0) // KL divergence is always non-negative
    }

    /// Compute entropy regularization to encourage diverse quantization
    ///
    /// H = -Σ p(x) log p(x)
    ///
    /// We approximate this by computing the negative log of the empirical distribution
    pub fn entropy_regularization(&self, weights: &Array2<f32>) -> f32 {
        // Compute histogram of quantized values
        let mut histogram: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        
        for &w in weights.iter() {
            // Discretize to integer bins
            let bin = (w * 100.0).round() as i32;
            *histogram.entry(bin).or_insert(0) += 1;
        }

        let n = weights.len() as f32;
        let mut entropy = 0.0;

        for &count in histogram.values() {
            if count > 0 {
                let p = count as f32 / n;
                entropy -= p * p.ln();
            }
        }

        // Return negative entropy (we want to maximize entropy, so minimize negative entropy)
        -entropy
    }

    /// Compute gradient of quantization loss w.r.t. quantized weights
    ///
    /// ∂L_quant/∂W_quantized = 2(W_quantized - W_original) / n
    pub fn quantization_loss_gradient(
        &self,
        original: &Array2<f32>,
        quantized: &Array2<f32>,
    ) -> Array2<f32> {
        let n = original.len() as f32;
        (quantized - original) * (2.0 / n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // ============================================================================
    // MSE (Quantization Loss) Tests
    // ============================================================================

    #[test]
    fn test_quantization_loss_basic() {
        let loss_fn = ThermodynamicLoss::default();
        
        let original = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let quantized = Array2::from_shape_vec((2, 2), vec![1.1, 2.1, 2.9, 3.9]).unwrap();
        
        let loss = loss_fn.quantization_loss(&original, &quantized);
        
        // MSE = ((0.1)^2 + (0.1)^2 + (0.1)^2 + (0.1)^2) / 4 = 0.01
        assert!((loss - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_quantization_loss_identical() {
        let loss_fn = ThermodynamicLoss::default();
        
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        
        let loss = loss_fn.quantization_loss(&weights, &weights);
        
        // Loss should be zero for identical weights
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    fn test_quantization_loss_large_error() {
        let loss_fn = ThermodynamicLoss::default();
        
        let original = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let quantized = Array2::from_shape_vec((2, 2), vec![2.0, 3.0, 4.0, 5.0]).unwrap();
        
        let loss = loss_fn.quantization_loss(&original, &quantized);
        
        // MSE = ((1.0)^2 + (1.0)^2 + (1.0)^2 + (1.0)^2) / 4 = 1.0
        assert!((loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantization_loss_negative_values() {
        let loss_fn = ThermodynamicLoss::default();
        
        let original = Array2::from_shape_vec((2, 2), vec![-1.0, -2.0, 1.0, 2.0]).unwrap();
        let quantized = Array2::from_shape_vec((2, 2), vec![-1.1, -1.9, 1.1, 1.9]).unwrap();
        
        let loss = loss_fn.quantization_loss(&original, &quantized);
        
        // MSE = ((0.1)^2 + (0.1)^2 + (0.1)^2 + (0.1)^2) / 4 = 0.01
        assert!((loss - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_quantization_loss_single_element() {
        let loss_fn = ThermodynamicLoss::default();
        
        let original = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
        let quantized = Array2::from_shape_vec((1, 1), vec![5.5]).unwrap();
        
        let loss = loss_fn.quantization_loss(&original, &quantized);
        
        // MSE = (0.5)^2 / 1 = 0.25
        assert!((loss - 0.25).abs() < 1e-6);
    }

    // ============================================================================
    // KL Divergence Tests
    // ============================================================================

    #[test]
    fn test_gaussian_kl_divergence_identical() {
        let loss_fn = ThermodynamicLoss::default();
        
        // KL divergence between identical distributions should be 0
        let kl = loss_fn.gaussian_kl_divergence(0.0, 1.0, 0.0, 1.0);
        assert!(kl.abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_kl_divergence_different_means() {
        let loss_fn = ThermodynamicLoss::default();
        
        // KL divergence should be positive for different means
        let kl = loss_fn.gaussian_kl_divergence(0.0, 1.0, 1.0, 1.0);
        assert!(kl > 0.0);
        
        // Larger mean difference should give larger KL
        let kl_large = loss_fn.gaussian_kl_divergence(0.0, 1.0, 2.0, 1.0);
        assert!(kl_large > kl);
    }

    #[test]
    fn test_gaussian_kl_divergence_different_stds() {
        let loss_fn = ThermodynamicLoss::default();
        
        // KL divergence should be positive for different standard deviations
        let kl = loss_fn.gaussian_kl_divergence(0.0, 1.0, 0.0, 2.0);
        assert!(kl > 0.0);
        assert!(kl.is_finite());
    }

    #[test]
    fn test_gaussian_kl_divergence_symmetry() {
        let loss_fn = ThermodynamicLoss::default();
        
        // KL divergence is NOT symmetric: KL(P||Q) != KL(Q||P)
        let kl_pq = loss_fn.gaussian_kl_divergence(0.0, 1.0, 1.0, 2.0);
        let kl_qp = loss_fn.gaussian_kl_divergence(1.0, 2.0, 0.0, 1.0);
        
        // They should be different
        assert!((kl_pq - kl_qp).abs() > 1e-3);
    }

    #[test]
    fn test_gaussian_kl_divergence_numerical_stability() {
        let loss_fn = ThermodynamicLoss::default();
        
        // Test with very small standard deviations (should use epsilon)
        let kl = loss_fn.gaussian_kl_divergence(0.0, 1e-10, 0.0, 1e-10);
        assert!(kl.is_finite());
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_markov_constraint_loss_basic() {
        let loss_fn = ThermodynamicLoss::default();
        
        let original = vec![
            TransitionMatrix { mean: 0.0, std: 1.0, timestep: 0 },
            TransitionMatrix { mean: 0.1, std: 1.1, timestep: 1 },
        ];
        
        let quantized = vec![
            TransitionMatrix { mean: 0.05, std: 1.05, timestep: 0 },
            TransitionMatrix { mean: 0.15, std: 1.15, timestep: 1 },
        ];
        
        let loss = loss_fn.markov_constraint_loss(&original, &quantized);
        
        // Loss should be positive (distributions are different)
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_markov_constraint_loss_identical() {
        let loss_fn = ThermodynamicLoss::default();
        
        let transitions = vec![
            TransitionMatrix { mean: 0.0, std: 1.0, timestep: 0 },
            TransitionMatrix { mean: 0.1, std: 1.1, timestep: 1 },
        ];
        
        let loss = loss_fn.markov_constraint_loss(&transitions, &transitions);
        
        // Loss should be near zero for identical transitions
        assert!(loss.abs() < 1e-5);
    }

    #[test]
    fn test_markov_constraint_loss_empty() {
        let loss_fn = ThermodynamicLoss::default();
        
        let empty: Vec<TransitionMatrix> = vec![];
        
        let loss = loss_fn.markov_constraint_loss(&empty, &empty);
        
        // Loss should be zero for empty transitions
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_markov_constraint_loss_single_transition() {
        let loss_fn = ThermodynamicLoss::default();
        
        let original = vec![
            TransitionMatrix { mean: 0.0, std: 1.0, timestep: 0 },
        ];
        
        let quantized = vec![
            TransitionMatrix { mean: 0.1, std: 1.1, timestep: 0 },
        ];
        
        let loss = loss_fn.markov_constraint_loss(&original, &quantized);
        
        // Loss should be positive and finite
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_markov_constraint_loss_many_transitions() {
        let loss_fn = ThermodynamicLoss::default();
        
        let original: Vec<TransitionMatrix> = (0..10)
            .map(|i| TransitionMatrix {
                mean: i as f32 * 0.1,
                std: 1.0 + i as f32 * 0.05,
                timestep: i,
            })
            .collect();
        
        let quantized: Vec<TransitionMatrix> = (0..10)
            .map(|i| TransitionMatrix {
                mean: i as f32 * 0.1 + 0.05,
                std: 1.0 + i as f32 * 0.05 + 0.02,
                timestep: i,
            })
            .collect();
        
        let loss = loss_fn.markov_constraint_loss(&original, &quantized);
        
        // Loss should be positive and finite
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    // ============================================================================
    // Entropy Regularization Tests
    // ============================================================================

    #[test]
    fn test_entropy_regularization_uniform_vs_concentrated() {
        let loss_fn = ThermodynamicLoss::default();
        
        // Uniform distribution should have high entropy (low negative entropy)
        let uniform = Array2::from_shape_vec((2, 2), vec![0.0, 0.25, 0.5, 0.75]).unwrap();
        let entropy_uniform = loss_fn.entropy_regularization(&uniform);
        
        // Concentrated distribution should have low entropy (high negative entropy)
        let concentrated = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let entropy_concentrated = loss_fn.entropy_regularization(&concentrated);
        
        // Uniform should have lower negative entropy (higher entropy)
        assert!(entropy_uniform < entropy_concentrated);
    }

    #[test]
    fn test_entropy_regularization_is_finite() {
        let loss_fn = ThermodynamicLoss::default();
        
        let weights = Array2::from_shape_vec((3, 3), vec![
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
            0.7, 0.8, 0.9,
        ]).unwrap();
        
        let entropy = loss_fn.entropy_regularization(&weights);
        
        assert!(entropy.is_finite());
    }

    // ============================================================================
    // Loss Combination Tests (Different Weights)
    // ============================================================================

    #[test]
    fn test_total_loss_no_markov_weight() {
        let loss_fn = ThermodynamicLoss::new(0.0, 0.0); // No Markov or entropy
        
        let original_weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let quantized_weights = Array2::from_shape_vec((2, 2), vec![1.1, 2.1, 2.9, 3.9]).unwrap();
        
        let original_transitions = vec![
            TransitionMatrix { mean: 0.0, std: 1.0, timestep: 0 },
        ];
        let quantized_transitions = vec![
            TransitionMatrix { mean: 0.05, std: 1.05, timestep: 0 },
        ];
        
        let total_loss = loss_fn.compute_total_loss(
            &original_weights,
            &quantized_weights,
            &original_transitions,
            &quantized_transitions,
        );
        
        // Should equal quantization loss only
        let quant_loss = loss_fn.quantization_loss(&original_weights, &quantized_weights);
        assert!((total_loss - quant_loss).abs() < 1e-6);
    }

    #[test]
    fn test_total_loss_with_markov_weight() {
        let loss_fn = ThermodynamicLoss::new(0.1, 0.0); // Markov weight = 0.1
        
        let original_weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let quantized_weights = Array2::from_shape_vec((2, 2), vec![1.1, 2.1, 2.9, 3.9]).unwrap();
        
        let original_transitions = vec![
            TransitionMatrix { mean: 0.0, std: 1.0, timestep: 0 },
        ];
        let quantized_transitions = vec![
            TransitionMatrix { mean: 0.05, std: 1.05, timestep: 0 },
        ];
        
        let total_loss = loss_fn.compute_total_loss(
            &original_weights,
            &quantized_weights,
            &original_transitions,
            &quantized_transitions,
        );
        
        // Should be sum of quantization loss and weighted Markov loss
        let quant_loss = loss_fn.quantization_loss(&original_weights, &quantized_weights);
        let markov_loss = loss_fn.markov_constraint_loss(&original_transitions, &quantized_transitions);
        let expected = quant_loss + 0.1 * markov_loss;
        
        assert!((total_loss - expected).abs() < 1e-5);
    }

    #[test]
    fn test_total_loss_with_entropy_weight() {
        let loss_fn = ThermodynamicLoss::new(0.0, 0.05); // Entropy weight = 0.05
        
        let original_weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let quantized_weights = Array2::from_shape_vec((2, 2), vec![1.1, 2.1, 2.9, 3.9]).unwrap();
        
        let original_transitions = vec![
            TransitionMatrix { mean: 0.0, std: 1.0, timestep: 0 },
        ];
        let quantized_transitions = vec![
            TransitionMatrix { mean: 0.05, std: 1.05, timestep: 0 },
        ];
        
        let total_loss = loss_fn.compute_total_loss(
            &original_weights,
            &quantized_weights,
            &original_transitions,
            &quantized_transitions,
        );
        
        // Should include entropy regularization
        let quant_loss = loss_fn.quantization_loss(&original_weights, &quantized_weights);
        let entropy_loss = loss_fn.entropy_regularization(&quantized_weights);
        let expected = quant_loss + 0.05 * entropy_loss;
        
        assert!((total_loss - expected).abs() < 1e-5);
    }

    #[test]
    fn test_total_loss_with_all_weights() {
        let loss_fn = ThermodynamicLoss::new(0.1, 0.05); // Both weights
        
        let original_weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let quantized_weights = Array2::from_shape_vec((2, 2), vec![1.1, 2.1, 2.9, 3.9]).unwrap();
        
        let original_transitions = vec![
            TransitionMatrix { mean: 0.0, std: 1.0, timestep: 0 },
        ];
        let quantized_transitions = vec![
            TransitionMatrix { mean: 0.05, std: 1.05, timestep: 0 },
        ];
        
        let total_loss = loss_fn.compute_total_loss(
            &original_weights,
            &quantized_weights,
            &original_transitions,
            &quantized_transitions,
        );
        
        // Should include all components
        let quant_loss = loss_fn.quantization_loss(&original_weights, &quantized_weights);
        let markov_loss = loss_fn.markov_constraint_loss(&original_transitions, &quantized_transitions);
        let entropy_loss = loss_fn.entropy_regularization(&quantized_weights);
        let expected = quant_loss + 0.1 * markov_loss + 0.05 * entropy_loss;
        
        assert!((total_loss - expected).abs() < 1e-5);
    }

    #[test]
    fn test_total_loss_different_markov_weights() {
        let original_weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let quantized_weights = Array2::from_shape_vec((2, 2), vec![1.1, 2.1, 2.9, 3.9]).unwrap();
        
        let original_transitions = vec![
            TransitionMatrix { mean: 0.0, std: 1.0, timestep: 0 },
        ];
        let quantized_transitions = vec![
            TransitionMatrix { mean: 0.5, std: 1.5, timestep: 0 },
        ];
        
        // Test with different Markov weights
        let loss_fn_low = ThermodynamicLoss::new(0.05, 0.0);
        let loss_fn_high = ThermodynamicLoss::new(0.2, 0.0);
        
        let loss_low = loss_fn_low.compute_total_loss(
            &original_weights,
            &quantized_weights,
            &original_transitions,
            &quantized_transitions,
        );
        
        let loss_high = loss_fn_high.compute_total_loss(
            &original_weights,
            &quantized_weights,
            &original_transitions,
            &quantized_transitions,
        );
        
        // Higher Markov weight should give higher total loss (since Markov loss is positive)
        assert!(loss_high > loss_low);
    }

    // ============================================================================
    // Gradient Tests
    // ============================================================================

    #[test]
    fn test_quantization_loss_gradient_basic() {
        let loss_fn = ThermodynamicLoss::default();
        
        let original = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let quantized = Array2::from_shape_vec((2, 2), vec![1.1, 2.1, 2.9, 3.9]).unwrap();
        
        let gradient = loss_fn.quantization_loss_gradient(&original, &quantized);
        
        // Gradient should have same shape as input
        assert_eq!(gradient.shape(), original.shape());
        
        // Gradient should point in direction to reduce loss
        // For quantized > original, gradient should be positive
        assert!(gradient[[0, 0]] > 0.0);
        assert!(gradient[[0, 1]] > 0.0);
        
        // For quantized < original, gradient should be negative
        assert!(gradient[[1, 0]] < 0.0);
        assert!(gradient[[1, 1]] < 0.0);
    }

    #[test]
    fn test_quantization_loss_gradient_zero_for_identical() {
        let loss_fn = ThermodynamicLoss::default();
        
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        
        let gradient = loss_fn.quantization_loss_gradient(&weights, &weights);
        
        // Gradient should be zero for identical weights
        for &g in gradient.iter() {
            assert!(g.abs() < 1e-6);
        }
    }
}

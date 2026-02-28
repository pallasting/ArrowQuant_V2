//! Weight Entropy analyzer for Dynamic Bit-Width Allocation
//!
//! Calculates the Shannon Entropy of weight tensors to determine their
//! information density and importance.
//!
//! - Low Entropy -> Highly structured/critical -> Needs higher precision (INT8/FP16)
//! - High Entropy -> Uniform/Noise-like -> Can be aggressively compressed (INT4/INT2)

use std::cmp;

/// Analyzes weight tensors to compute their normalized Shannon Entropy.
pub struct EntropyAnalyzer {
    /// Number of bins for the histogram
    pub num_bins: usize,
}

impl Default for EntropyAnalyzer {
    fn default() -> Self {
        Self { num_bins: 256 }
    }
}

impl EntropyAnalyzer {
    /// Creates a new EntropyAnalyzer with a specific number of bins.
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins: cmp::max(2, num_bins), // At least 2 bins
        }
    }

    /// Computes the normalized Shannon Entropy of the given weights.
    ///
    /// # Arguments
    ///
    /// * `weights` - Flat slice of layer weights
    ///
    /// # Returns
    ///
    /// Normalized entropy in the range `[0.0, 1.0]`.
    /// 1.0 means perfectly uniform distribution (maximum entropy).
    /// 0.0 means all weights are identical (minimum entropy).
    pub fn compute_normalized_entropy(&self, weights: &[f32]) -> f32 {
        if weights.is_empty() {
            return 0.0; // By convention, max structure or empty
        }

        // 1. Find Min and Max
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &w in weights {
            if w < min_val {
                min_val = w;
            }
            if w > max_val {
                max_val = w;
            }
        }

        if min_val >= max_val {
            return 0.0; // Variance is 0, purely structured
        }

        // 2. Build Histogram
        let mut histogram = vec![0usize; self.num_bins];
        let range = max_val - min_val;
        
        for w in weights {
            // Find bin index, clamped to [0, num_bins - 1]
            let mut bin_idx = (((w - min_val) / range) * self.num_bins as f32) as usize;
            if bin_idx >= self.num_bins {
                bin_idx = self.num_bins - 1;
            }
            histogram[bin_idx] += 1;
        }

        // 3. Compute Shannon Entropy: H = -Sum(P(x) * log2(P(x)))
        let mut entropy = 0.0;
        let total = weights.len() as f32;

        for &count in &histogram {
            if count > 0 {
                let p = count as f32 / total;
                entropy -= p * p.log2();
            }
        }

        // 4. Normalize by max possible entropy (log2 of num_bins)
        let max_entropy = (self.num_bins as f32).log2();
        let normalized = entropy / max_entropy;

        normalized.clamp(0.0, 1.0)
    }

    /// Determines the optimal bit-width based on the entropy score.
    ///
    /// Strategy:
    /// - Low entropy (< 0.5): Highly skewed or structured, sensitive to quantization -> INT8
    /// - Normal entropy (0.5..0.7): Moderate distribution -> INT4
    /// - High entropy (> 0.7): Broad, flat distribution (more robust) -> INT2
    pub fn suggest_bit_width(&self, entropy: f32) -> u8 {
        if entropy < 0.5 {
            8
        } else if entropy < 0.7 {
            4
        } else {
            2
        }
    }
}

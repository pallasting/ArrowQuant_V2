//! Transition Probability Computation
//!
//! This module implements transition probability computation for thermodynamic quantization.
//! It computes q(x_t | x_{t-1}) for each timestep using weight statistics as sufficient statistics.

use ndarray::Array2;
use lru::LruCache;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::num::NonZeroUsize;

/// Beta schedule type for diffusion process
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BetaSchedule {
    /// Linear schedule: beta_t = beta_start + (beta_end - beta_start) * t / T
    Linear,
    /// Cosine schedule: beta_t = 1 - alpha_t, where alpha_t follows cosine curve
    Cosine,
}

/// Transition probability matrix represented by Gaussian parameters
#[derive(Debug, Clone)]
pub struct TransitionMatrix {
    /// Mean of the Gaussian transition distribution
    pub mean: f32,
    /// Standard deviation of the Gaussian transition distribution
    pub std: f32,
    /// Timestep index
    pub timestep: usize,
}

/// Computes transition probabilities for diffusion process
pub struct TransitionComputer {
    /// Beta schedule values for each timestep
    beta_schedule: Vec<f32>,
    /// LRU cache for computed transitions
    cache: LruCache<CacheKey, TransitionMatrix>,
    /// CPU feature detection flags
    cpu_features: CpuFeatures,
    /// Cache hit rate metrics
    cache_hits: usize,
    /// Cache miss count
    cache_misses: usize,
}

/// Cache statistics for monitoring performance
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f32,
    /// Current cache size
    pub size: usize,
    /// Maximum cache capacity
    pub capacity: usize,
}

/// CPU feature detection for runtime optimization selection
#[derive(Debug, Clone, Copy)]
struct CpuFeatures {
    has_avx2: bool,
    has_neon: bool,
}

impl CpuFeatures {
    /// Detect available CPU features at runtime
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_avx2: is_x86_feature_detected!("avx2"),
                has_neon: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on aarch64
            Self {
                has_avx2: false,
                has_neon: true,
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                has_avx2: false,
                has_neon: false,
            }
        }
    }
}

/// Cache key for transition matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CacheKey {
    /// Hash of the weight tensor
    layer_hash: u64,
    /// Timestep index
    timestep: usize,
}

impl TransitionComputer {
    /// Create a new TransitionComputer with the specified beta schedule
    ///
    /// # Arguments
    ///
    /// * `num_timesteps` - Number of diffusion timesteps
    /// * `schedule` - Beta schedule type (Linear or Cosine)
    /// * `beta_start` - Starting beta value (default: 0.0001)
    /// * `beta_end` - Ending beta value (default: 0.02)
    pub fn new(
        num_timesteps: usize,
        schedule: BetaSchedule,
        beta_start: f32,
        beta_end: f32,
    ) -> Self {
        let beta_schedule = Self::create_beta_schedule(
            num_timesteps,
            schedule,
            beta_start,
            beta_end,
        );

        // Default cache size: num_timesteps * 10 (reasonable for most models)
        let cache_capacity = Self::compute_default_cache_size(num_timesteps);

        Self {
            beta_schedule,
            cache: LruCache::new(NonZeroUsize::new(cache_capacity).unwrap()),
            cpu_features: CpuFeatures::detect(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Create a new TransitionComputer with custom cache size
    ///
    /// # Arguments
    ///
    /// * `num_timesteps` - Number of diffusion timesteps
    /// * `schedule` - Beta schedule type (Linear or Cosine)
    /// * `beta_start` - Starting beta value
    /// * `beta_end` - Ending beta value
    /// * `cache_capacity` - Maximum number of entries in the LRU cache
    pub fn with_cache_capacity(
        num_timesteps: usize,
        schedule: BetaSchedule,
        beta_start: f32,
        beta_end: f32,
        cache_capacity: usize,
    ) -> Self {
        let beta_schedule = Self::create_beta_schedule(
            num_timesteps,
            schedule,
            beta_start,
            beta_end,
        );

        Self {
            beta_schedule,
            cache: LruCache::new(NonZeroUsize::new(cache_capacity.max(1)).unwrap()),
            cpu_features: CpuFeatures::detect(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Compute optimal cache size based on model characteristics
    ///
    /// The cache size is tuned based on:
    /// - Number of timesteps (more timesteps = larger cache)
    /// - Expected number of unique layers (estimated from typical model sizes)
    ///
    /// Formula: cache_size = num_timesteps * layers_per_timestep
    /// where layers_per_timestep is estimated as 10 for typical models
    ///
    /// For reference:
    /// - Small models (1B params, ~20 layers): 1000 timesteps * 20 = 20,000 entries
    /// - Medium models (7B params, ~32 layers): 1000 timesteps * 32 = 32,000 entries
    /// - Large models (70B params, ~80 layers): 1000 timesteps * 80 = 80,000 entries
    ///
    /// Each cache entry is ~24 bytes (2 f32 + 1 usize), so:
    /// - 20,000 entries ≈ 480 KB
    /// - 32,000 entries ≈ 768 KB
    /// - 80,000 entries ≈ 1.9 MB
    ///
    /// This is well within the <10% memory overhead requirement.
    fn compute_default_cache_size(num_timesteps: usize) -> usize {
        // Default: assume 10 layers per timestep for typical models
        // This can be tuned based on actual model size
        let layers_per_timestep = 10;
        (num_timesteps * layers_per_timestep).max(100)
    }

    /// Tune cache size based on model size
    ///
    /// # Arguments
    ///
    /// * `model_size_params` - Approximate number of parameters in the model
    ///
    /// This method adjusts the cache capacity based on the model size:
    /// - Small models (<2B params): num_timesteps * 20
    /// - Medium models (2B-10B params): num_timesteps * 40
    /// - Large models (>10B params): num_timesteps * 80
    pub fn tune_cache_for_model_size(&mut self, model_size_params: u64) {
        let num_timesteps = self.beta_schedule.len();
        
        let layers_estimate = if model_size_params < 2_000_000_000 {
            20  // Small models: ~20 layers
        } else if model_size_params < 10_000_000_000 {
            40  // Medium models: ~40 layers
        } else {
            80  // Large models: ~80 layers
        };

        let new_capacity = (num_timesteps * layers_estimate).max(100);
        
        // Recreate cache with new capacity
        self.cache = LruCache::new(NonZeroUsize::new(new_capacity).unwrap());
        
        // Reset metrics since we're starting fresh
        self.cache_hits = 0;
        self.cache_misses = 0;
    }

    /// Create beta schedule based on schedule type
    fn create_beta_schedule(
        num_timesteps: usize,
        schedule: BetaSchedule,
        beta_start: f32,
        beta_end: f32,
    ) -> Vec<f32> {
        match schedule {
            BetaSchedule::Linear => {
                (0..num_timesteps)
                    .map(|t| {
                        let alpha = t as f32 / (num_timesteps - 1) as f32;
                        beta_start + (beta_end - beta_start) * alpha
                    })
                    .collect()
            }
            BetaSchedule::Cosine => {
                let s = 0.008; // offset for numerical stability
                (0..num_timesteps)
                    .map(|t| {
                        let t_norm = t as f32 / num_timesteps as f32;
                        let alpha_bar_t = ((t_norm + s) / (1.0 + s) * std::f32::consts::PI / 2.0).cos().powi(2);
                        let alpha_bar_t_prev = if t > 0 {
                            let t_prev = (t - 1) as f32 / num_timesteps as f32;
                            ((t_prev + s) / (1.0 + s) * std::f32::consts::PI / 2.0).cos().powi(2)
                        } else {
                            1.0
                        };
                        (1.0 - alpha_bar_t / alpha_bar_t_prev).clamp(0.0001, 0.9999)
                    })
                    .collect()
            }
        }
    }

    /// Compute transition probability for a given weight tensor and timestep
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight tensor (2D array)
    /// * `timestep` - Timestep index
    ///
    /// # Returns
    ///
    /// TransitionMatrix containing mean and std of the Gaussian transition distribution
    pub fn compute_transition(
        &mut self,
        weights: &Array2<f32>,
        timestep: usize,
    ) -> TransitionMatrix {
        // Check cache first
        let layer_hash = self.hash_weights(weights);
        let cache_key = CacheKey { layer_hash, timestep };

        if let Some(cached) = self.cache.get(&cache_key) {
            self.cache_hits += 1;
            return cached.clone();
        }

        // Cache miss - compute transition
        self.cache_misses += 1;

        let beta_t = self.beta_schedule[timestep];
        let alpha_t = 1.0 - beta_t;

        // Compute weight statistics using SIMD-optimized methods
        let data = weights.as_slice().unwrap();
        let mean = self.compute_mean_simd(data);
        let std = self.compute_std(weights, mean);

        // Apply diffusion forward process
        let transition = TransitionMatrix {
            mean: mean * alpha_t.sqrt(),
            std: (std.powi(2) * alpha_t + beta_t).sqrt(),
            timestep,
        };

        // Cache the result (LRU will evict oldest if at capacity)
        self.cache.put(cache_key, transition.clone());

        transition
    }

    /// Compute standard deviation of weights
    fn compute_std(&self, weights: &Array2<f32>, mean: f32) -> f32 {
        let n = weights.len() as f32;
        if n <= 1.0 {
            return 0.0;
        }

        // Use SIMD-optimized variance computation
        let variance = self.compute_variance_simd(weights.as_slice().unwrap(), mean);
        variance.sqrt()
    }

    /// Compute mean using SIMD acceleration
    #[inline]
    fn compute_mean_simd(&self, data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        // Runtime CPU feature detection
        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_features.has_avx2 {
                return unsafe { self.compute_mean_avx2(data) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.cpu_features.has_neon {
                return unsafe { self.compute_mean_neon(data) };
            }
        }

        // Fallback to scalar
        self.compute_mean_scalar(data)
    }

    /// Compute variance using SIMD acceleration
    #[inline]
    fn compute_variance_simd(&self, data: &[f32], mean: f32) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        // Runtime CPU feature detection
        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_features.has_avx2 {
                return unsafe { self.compute_variance_avx2(data, mean) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.cpu_features.has_neon {
                return unsafe { self.compute_variance_neon(data, mean) };
            }
        }

        // Fallback to scalar
        self.compute_variance_scalar(data, mean)
    }

    // ============================================================================
    // AVX2 Implementation (x86_64)
    // ============================================================================

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn compute_mean_avx2(&self, data: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let mut sum = _mm256_setzero_ps();
        let chunks = data.len() / 8;

        // Process 8 floats at a time
        for i in 0..chunks {
            let offset = i * 8;
            let values = _mm256_loadu_ps(data.as_ptr().add(offset));
            sum = _mm256_add_ps(sum, values);
        }

        // Horizontal sum
        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), sum);
        let mut total = result.iter().sum::<f32>();

        // Handle remainder
        for &value in &data[chunks * 8..] {
            total += value;
        }

        total / data.len() as f32
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn compute_variance_avx2(&self, data: &[f32], mean: f32) -> f32 {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let mean_vec = _mm256_set1_ps(mean);
        let mut sum_sq = _mm256_setzero_ps();
        let chunks = data.len() / 8;

        // Process 8 floats at a time
        for i in 0..chunks {
            let offset = i * 8;
            let values = _mm256_loadu_ps(data.as_ptr().add(offset));
            let diff = _mm256_sub_ps(values, mean_vec);
            let squared = _mm256_mul_ps(diff, diff);
            sum_sq = _mm256_add_ps(sum_sq, squared);
        }

        // Horizontal sum
        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), sum_sq);
        let mut total = result.iter().sum::<f32>();

        // Handle remainder
        for &value in &data[chunks * 8..] {
            let diff = value - mean;
            total += diff * diff;
        }

        total / data.len() as f32
    }

    // ============================================================================
    // NEON Implementation (ARM64)
    // ============================================================================

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn compute_mean_neon(&self, data: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        use std::arch::aarch64::*;

        let mut sum = vdupq_n_f32(0.0);
        let chunks = data.len() / 4;

        // Process 4 floats at a time
        for i in 0..chunks {
            let offset = i * 4;
            let values = vld1q_f32(data.as_ptr().add(offset));
            sum = vaddq_f32(sum, values);
        }

        // Horizontal sum
        let mut result = [0.0f32; 4];
        vst1q_f32(result.as_mut_ptr(), sum);
        let mut total = result.iter().sum::<f32>();

        // Handle remainder
        for &value in &data[chunks * 4..] {
            total += value;
        }

        total / data.len() as f32
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn compute_variance_neon(&self, data: &[f32], mean: f32) -> f32 {
        #[cfg(target_arch = "aarch64")]
        use std::arch::aarch64::*;

        let mean_vec = vdupq_n_f32(mean);
        let mut sum_sq = vdupq_n_f32(0.0);
        let chunks = data.len() / 4;

        // Process 4 floats at a time
        for i in 0..chunks {
            let offset = i * 4;
            let values = vld1q_f32(data.as_ptr().add(offset));
            let diff = vsubq_f32(values, mean_vec);
            let squared = vmulq_f32(diff, diff);
            sum_sq = vaddq_f32(sum_sq, squared);
        }

        // Horizontal sum
        let mut result = [0.0f32; 4];
        vst1q_f32(result.as_mut_ptr(), sum_sq);
        let mut total = result.iter().sum::<f32>();

        // Handle remainder
        for &value in &data[chunks * 4..] {
            let diff = value - mean;
            total += diff * diff;
        }

        total / data.len() as f32
    }

    // ============================================================================
    // Scalar Fallback Implementation
    // ============================================================================

    #[inline]
    fn compute_mean_scalar(&self, data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f32>() / data.len() as f32
    }

    #[inline]
    fn compute_variance_scalar(&self, data: &[f32], mean: f32) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / data.len() as f32
    }

    /// Hash weight tensor for caching
    fn hash_weights(&self, weights: &Array2<f32>) -> u64 {
        let mut hasher = DefaultHasher::new();
        
        // Hash shape
        weights.shape()[0].hash(&mut hasher);
        weights.shape()[1].hash(&mut hasher);
        
        // Sample a few values for hash (for performance)
        let sample_size = 100.min(weights.len());
        let step = weights.len() / sample_size;
        
        for (i, &val) in weights.iter().enumerate() {
            if i % step == 0 {
                val.to_bits().hash(&mut hasher);
            }
        }
        
        hasher.finish()
    }

    /// Clear the transition cache and reset metrics
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }

    /// Get current cache size (number of entries)
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Get cache capacity (maximum number of entries)
    pub fn cache_capacity(&self) -> usize {
        self.cache.cap().get()
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let total_requests = self.cache_hits + self.cache_misses;
        let hit_rate = if total_requests > 0 {
            self.cache_hits as f32 / total_requests as f32
        } else {
            0.0
        };

        CacheStats {
            hits: self.cache_hits,
            misses: self.cache_misses,
            hit_rate,
            size: self.cache.len(),
            capacity: self.cache.cap().get(),
        }
    }

    /// Reset cache metrics without clearing the cache
    pub fn reset_metrics(&mut self) {
        self.cache_hits = 0;
        self.cache_misses = 0;
    }

    /// Check if cache is performing well (hit rate > 70%)
    pub fn is_cache_effective(&self) -> bool {
        let stats = self.cache_stats();
        stats.hit_rate > 0.7 && (stats.hits + stats.misses) > 10
    }

    /// Get cache hit rate as a percentage (0-100)
    pub fn cache_hit_rate_percent(&self) -> f32 {
        self.cache_stats().hit_rate * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_linear_beta_schedule() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        assert_eq!(computer.beta_schedule.len(), 10);
        assert!((computer.beta_schedule[0] - 0.0001).abs() < 1e-6);
        assert!((computer.beta_schedule[9] - 0.02).abs() < 1e-6);
        
        // Check monotonicity
        for i in 0..9 {
            assert!(computer.beta_schedule[i] < computer.beta_schedule[i + 1]);
        }
    }

    #[test]
    fn test_linear_beta_schedule_intermediate_values() {
        let computer = TransitionComputer::new(5, BetaSchedule::Linear, 0.0, 1.0);
        
        // With 5 timesteps from 0.0 to 1.0, we expect: 0.0, 0.25, 0.5, 0.75, 1.0
        assert!((computer.beta_schedule[0] - 0.0).abs() < 1e-6);
        assert!((computer.beta_schedule[1] - 0.25).abs() < 1e-6);
        assert!((computer.beta_schedule[2] - 0.5).abs() < 1e-6);
        assert!((computer.beta_schedule[3] - 0.75).abs() < 1e-6);
        assert!((computer.beta_schedule[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_beta_schedule() {
        let computer = TransitionComputer::new(10, BetaSchedule::Cosine, 0.0001, 0.02);
        
        assert_eq!(computer.beta_schedule.len(), 10);
        
        // Cosine schedule should have smaller values at the beginning
        assert!(computer.beta_schedule[0] < computer.beta_schedule[9]);
        
        // All values should be within valid range
        for &beta in &computer.beta_schedule {
            assert!(beta >= 0.0001);
            assert!(beta <= 0.9999);
        }
    }

    #[test]
    fn test_cosine_beta_schedule_properties() {
        let computer = TransitionComputer::new(100, BetaSchedule::Cosine, 0.0001, 0.02);
        
        // Cosine schedule should be smooth (no extremely large jumps)
        // Note: Cosine schedule can have larger jumps near the end due to its nature
        for i in 0..99 {
            let diff = (computer.beta_schedule[i + 1] - computer.beta_schedule[i]).abs();
            assert!(diff < 0.2, "Very large jump detected at timestep {}: {}", i, diff);
        }
    }

    #[test]
    fn test_compute_transition_with_known_values() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        // Create weights with known statistics: mean=1.0
        let weights = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 2.0]).unwrap();
        
        let transition = computer.compute_transition(&weights, 0);
        
        // At timestep 0, beta_t = 0.0001, alpha_t = 0.9999
        let expected_mean = 1.0 * 0.9999_f32.sqrt();
        
        // Verify the mean is computed correctly
        assert!((transition.mean - expected_mean).abs() < 0.01, 
                "Expected mean ~{}, got {}", expected_mean, transition.mean);
        
        // Verify std is positive and finite
        assert!(transition.std > 0.0, "Std should be positive");
        assert!(transition.std.is_finite(), "Std should be finite");
        assert_eq!(transition.timestep, 0);
    }

    #[test]
    fn test_compute_transition() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        let transition = computer.compute_transition(&weights, 5);
        
        assert_eq!(transition.timestep, 5);
        assert!(transition.mean.is_finite());
        assert!(transition.std.is_finite());
        assert!(transition.std > 0.0);
    }

    #[test]
    fn test_compute_transition_different_timesteps() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        let t0 = computer.compute_transition(&weights, 0);
        let _t5 = computer.compute_transition(&weights, 5);
        let t9 = computer.compute_transition(&weights, 9);
        
        // As timestep increases, beta increases, so std should generally increase
        // and mean should decrease (due to alpha_t decreasing)
        assert!(t0.mean > t9.mean, "Mean should decrease with timestep");
        assert!(t0.std < t9.std, "Std should increase with timestep");
    }

    #[test]
    fn test_compute_transition_zero_weights() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::zeros((4, 4));
        
        let transition = computer.compute_transition(&weights, 5);
        
        assert_eq!(transition.timestep, 5);
        assert!((transition.mean).abs() < 1e-6, "Mean should be near zero");
        assert!(transition.std.is_finite());
        assert!(transition.std >= 0.0);
    }

    #[test]
    fn test_compute_transition_uniform_weights() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        // All weights are the same value
        let weights = Array2::from_elem((4, 4), 5.0);
        
        let transition = computer.compute_transition(&weights, 5);
        
        assert_eq!(transition.timestep, 5);
        assert!(transition.mean.is_finite());
        // For uniform weights, the weight std is 0, but after diffusion process
        // std = sqrt(0 * alpha_t + beta_t) = sqrt(beta_t)
        // At timestep 5, beta_t is small, so std should be relatively small
        assert!(transition.std < 1.0, "Std should be relatively small for uniform weights, got {}", transition.std);
        assert!(transition.std > 0.0, "Std should be positive due to beta_t");
    }

    #[test]
    fn test_compute_transition_single_element() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::from_shape_vec((1, 1), vec![3.14]).unwrap();
        
        let transition = computer.compute_transition(&weights, 5);
        
        assert_eq!(transition.timestep, 5);
        assert!(transition.mean.is_finite());
        assert!(transition.std.is_finite());
        assert!(transition.std >= 0.0);
    }

    #[test]
    fn test_transition_caching() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        // First computation - should be a cache miss
        let t1 = computer.compute_transition(&weights, 5);
        assert_eq!(computer.cache_size(), 1);
        assert_eq!(computer.cache_stats().misses, 1);
        assert_eq!(computer.cache_stats().hits, 0);
        
        // Second computation (should use cache) - cache hit
        let t2 = computer.compute_transition(&weights, 5);
        assert_eq!(computer.cache_size(), 1);
        assert_eq!(computer.cache_stats().hits, 1);
        assert_eq!(computer.cache_stats().misses, 1);
        
        // Results should be identical
        assert_eq!(t1.mean, t2.mean);
        assert_eq!(t1.std, t2.std);
        
        // Different timestep should create new cache entry
        let _t3 = computer.compute_transition(&weights, 6);
        assert_eq!(computer.cache_size(), 2);
        assert_eq!(computer.cache_stats().misses, 2);
    }

    #[test]
    fn test_caching_different_weights() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights1 = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        let weights2 = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.2);
        
        computer.compute_transition(&weights1, 5);
        assert_eq!(computer.cache_size(), 1);
        
        // Different weights should create new cache entry
        computer.compute_transition(&weights2, 5);
        assert_eq!(computer.cache_size(), 2);
    }

    #[test]
    fn test_caching_same_weights_different_shape() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights1 = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        let weights2 = Array2::from_shape_fn((2, 8), |(i, j)| (i * 8 + j) as f32 * 0.1);
        
        computer.compute_transition(&weights1, 5);
        assert_eq!(computer.cache_size(), 1);
        
        // Different shape should create new cache entry even with similar values
        computer.compute_transition(&weights2, 5);
        assert_eq!(computer.cache_size(), 2);
    }

    #[test]
    fn test_clear_cache() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        computer.compute_transition(&weights, 5);
        assert_eq!(computer.cache_size(), 1);
        assert_eq!(computer.cache_stats().misses, 1);
        
        computer.clear_cache();
        assert_eq!(computer.cache_size(), 0);
        assert_eq!(computer.cache_stats().hits, 0);
        assert_eq!(computer.cache_stats().misses, 0);
    }

    #[test]
    fn test_clear_cache_multiple_entries() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        // Add multiple cache entries
        for t in 0..5 {
            computer.compute_transition(&weights, t);
        }
        assert_eq!(computer.cache_size(), 5);
        
        computer.clear_cache();
        assert_eq!(computer.cache_size(), 0);
    }

    #[test]
    fn test_beta_schedule_equality() {
        assert_eq!(BetaSchedule::Linear, BetaSchedule::Linear);
        assert_eq!(BetaSchedule::Cosine, BetaSchedule::Cosine);
        assert_ne!(BetaSchedule::Linear, BetaSchedule::Cosine);
    }

    #[test]
    fn test_transition_matrix_clone() {
        let tm = TransitionMatrix {
            mean: 1.5,
            std: 0.5,
            timestep: 3,
        };
        
        let cloned = tm.clone();
        assert_eq!(tm.mean, cloned.mean);
        assert_eq!(tm.std, cloned.std);
        assert_eq!(tm.timestep, cloned.timestep);
    }

    #[test]
    fn test_compute_std_edge_cases() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        // Single element
        let weights_single = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
        let std_single = computer.compute_std(&weights_single, 5.0);
        assert_eq!(std_single, 0.0, "Std of single element should be 0");
        
        // Empty-like case (1 element)
        let weights_one = Array2::from_shape_vec((1, 1), vec![3.0]).unwrap();
        let std_one = computer.compute_std(&weights_one, 3.0);
        assert_eq!(std_one, 0.0, "Std of one element should be 0");
    }

    #[test]
    fn test_large_timestep_count() {
        let computer = TransitionComputer::new(1000, BetaSchedule::Linear, 0.0001, 0.02);
        
        assert_eq!(computer.beta_schedule.len(), 1000);
        assert!((computer.beta_schedule[0] - 0.0001).abs() < 1e-6);
        assert!((computer.beta_schedule[999] - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_schedule_with_large_timesteps() {
        let computer = TransitionComputer::new(1000, BetaSchedule::Cosine, 0.0001, 0.02);
        
        assert_eq!(computer.beta_schedule.len(), 1000);
        
        // All values should be clamped within valid range
        for &beta in &computer.beta_schedule {
            assert!(beta >= 0.0001, "Beta too small: {}", beta);
            assert!(beta <= 0.9999, "Beta too large: {}", beta);
        }
    }

    // ============================================================================
    // SIMD Tests
    // ============================================================================

    #[test]
    fn test_cpu_features_detection() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        // Just verify that CPU features are detected without panicking
        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, we should detect AVX2 if available
            // This is a runtime check, so we can't assert specific values
            let _ = computer.cpu_features.has_avx2;
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            // On aarch64, NEON should always be available
            assert!(computer.cpu_features.has_neon);
        }
    }

    #[test]
    fn test_compute_mean_scalar() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = computer.compute_mean_scalar(&data);
        
        assert!((mean - 3.0).abs() < 1e-6, "Expected mean 3.0, got {}", mean);
    }

    #[test]
    fn test_compute_mean_scalar_empty() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let data: Vec<f32> = vec![];
        let mean = computer.compute_mean_scalar(&data);
        
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_compute_variance_scalar() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        // Data: [1, 2, 3, 4, 5], mean = 3
        // Variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
        //          = (4 + 1 + 0 + 1 + 4) / 5 = 10 / 5 = 2.0
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let variance = computer.compute_variance_scalar(&data, mean);
        
        assert!((variance - 2.0).abs() < 1e-6, "Expected variance 2.0, got {}", variance);
    }

    #[test]
    fn test_compute_variance_scalar_zero() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        // All same values should have zero variance
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let mean = 5.0;
        let variance = computer.compute_variance_scalar(&data, mean);
        
        assert!((variance).abs() < 1e-6, "Expected variance ~0, got {}", variance);
    }

    #[test]
    fn test_compute_mean_simd_consistency() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        // Test with various sizes to ensure SIMD and scalar paths agree
        for size in [1, 4, 7, 8, 15, 16, 31, 32, 100, 1000] {
            let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
            
            let mean_simd = computer.compute_mean_simd(&data);
            let mean_scalar = computer.compute_mean_scalar(&data);
            
            assert!(
                (mean_simd - mean_scalar).abs() < 1e-4,
                "SIMD and scalar mean differ for size {}: simd={}, scalar={}",
                size, mean_simd, mean_scalar
            );
        }
    }

    #[test]
    fn test_compute_variance_simd_consistency() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        // Test with various sizes to ensure SIMD and scalar paths agree
        for size in [1, 4, 7, 8, 15, 16, 31, 32, 100, 1000] {
            let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
            let mean = computer.compute_mean_scalar(&data);
            
            let var_simd = computer.compute_variance_simd(&data, mean);
            let var_scalar = computer.compute_variance_scalar(&data, mean);
            
            assert!(
                (var_simd - var_scalar).abs() < 1e-4,
                "SIMD and scalar variance differ for size {}: simd={}, scalar={}",
                size, var_simd, var_scalar
            );
        }
    }

    #[test]
    fn test_simd_mean_large_values() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let data: Vec<f32> = vec![1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0];
        let mean_simd = computer.compute_mean_simd(&data);
        let mean_scalar = computer.compute_mean_scalar(&data);
        
        assert!(
            (mean_simd - mean_scalar).abs() < 0.1,
            "SIMD and scalar mean differ for large values: simd={}, scalar={}",
            mean_simd, mean_scalar
        );
    }

    #[test]
    fn test_simd_variance_large_values() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let data: Vec<f32> = vec![1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0];
        let mean = computer.compute_mean_scalar(&data);
        let var_simd = computer.compute_variance_simd(&data, mean);
        let var_scalar = computer.compute_variance_scalar(&data, mean);
        
        assert!(
            (var_simd - var_scalar).abs() < 1.0,
            "SIMD and scalar variance differ for large values: simd={}, scalar={}",
            var_simd, var_scalar
        );
    }

    #[test]
    fn test_simd_mean_negative_values() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let data: Vec<f32> = vec![-5.0, -3.0, -1.0, 1.0, 3.0, 5.0];
        let mean_simd = computer.compute_mean_simd(&data);
        let mean_scalar = computer.compute_mean_scalar(&data);
        
        assert!(
            (mean_simd - mean_scalar).abs() < 1e-4,
            "SIMD and scalar mean differ for negative values: simd={}, scalar={}",
            mean_simd, mean_scalar
        );
        
        // Mean should be 0
        assert!((mean_simd).abs() < 1e-4, "Expected mean ~0, got {}", mean_simd);
    }

    #[test]
    fn test_simd_variance_negative_values() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let data: Vec<f32> = vec![-5.0, -3.0, -1.0, 1.0, 3.0, 5.0];
        let mean = 0.0;
        let var_simd = computer.compute_variance_simd(&data, mean);
        let var_scalar = computer.compute_variance_scalar(&data, mean);
        
        assert!(
            (var_simd - var_scalar).abs() < 1e-4,
            "SIMD and scalar variance differ for negative values: simd={}, scalar={}",
            var_simd, var_scalar
        );
    }

    #[test]
    fn test_transition_with_simd_optimizations() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        // Create a large weight matrix to benefit from SIMD
        let weights = Array2::from_shape_fn((100, 100), |(i, j)| {
            ((i * 100 + j) as f32 * 0.01).sin()
        });
        
        let transition = computer.compute_transition(&weights, 5);
        
        assert_eq!(transition.timestep, 5);
        assert!(transition.mean.is_finite());
        assert!(transition.std.is_finite());
        assert!(transition.std > 0.0);
    }

    #[test]
    fn test_simd_performance_benefit() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        // Create a large dataset to see SIMD benefits
        let data: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.001).sin()).collect();
        
        // Both should produce the same result
        let mean_simd = computer.compute_mean_simd(&data);
        let mean_scalar = computer.compute_mean_scalar(&data);
        
        assert!(
            (mean_simd - mean_scalar).abs() < 1e-3,
            "SIMD and scalar mean differ: simd={}, scalar={}",
            mean_simd, mean_scalar
        );
    }

    #[test]
    fn test_simd_with_unaligned_data() {
        let computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        // Test with sizes that don't align to SIMD boundaries (not multiples of 4 or 8)
        for size in [1, 3, 5, 7, 9, 11, 13, 17, 19, 23, 29, 31] {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            
            let mean_simd = computer.compute_mean_simd(&data);
            let mean_scalar = computer.compute_mean_scalar(&data);
            
            assert!(
                (mean_simd - mean_scalar).abs() < 1e-4,
                "SIMD and scalar mean differ for unaligned size {}: simd={}, scalar={}",
                size, mean_simd, mean_scalar
            );
        }
    }

    // ============================================================================
    // LRU Cache Tests
    // ============================================================================

    #[test]
    fn test_lru_cache_capacity() {
        let computer = TransitionComputer::with_cache_capacity(10, BetaSchedule::Linear, 0.0001, 0.02, 5);
        
        assert_eq!(computer.cache_capacity(), 5);
        assert_eq!(computer.cache_size(), 0);
    }

    #[test]
    fn test_lru_cache_eviction() {
        let mut computer = TransitionComputer::with_cache_capacity(10, BetaSchedule::Linear, 0.0001, 0.02, 3);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        // Fill cache to capacity
        for t in 0..3 {
            computer.compute_transition(&weights, t);
        }
        assert_eq!(computer.cache_size(), 3);
        
        // Add one more - should evict the oldest (timestep 0)
        computer.compute_transition(&weights, 3);
        assert_eq!(computer.cache_size(), 3); // Still at capacity
        
        // Access timestep 0 again - should be a cache miss (was evicted)
        computer.reset_metrics();
        computer.compute_transition(&weights, 0);
        assert_eq!(computer.cache_stats().misses, 1);
        assert_eq!(computer.cache_stats().hits, 0);
    }

    #[test]
    fn test_cache_hit_rate_metrics() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        // First access - miss
        computer.compute_transition(&weights, 5);
        let stats = computer.cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate, 0.0);
        
        // Second access - hit
        computer.compute_transition(&weights, 5);
        let stats = computer.cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
        
        // Third access - hit
        computer.compute_transition(&weights, 5);
        let stats = computer.cache_stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_cache_hit_rate_percent() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        // 1 miss, 2 hits = 66.67% hit rate
        computer.compute_transition(&weights, 5);
        computer.compute_transition(&weights, 5);
        computer.compute_transition(&weights, 5);
        
        let hit_rate_pct = computer.cache_hit_rate_percent();
        assert!((hit_rate_pct - 66.67).abs() < 0.1);
    }

    #[test]
    fn test_reset_metrics() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        // Generate some cache activity
        computer.compute_transition(&weights, 5);
        computer.compute_transition(&weights, 5);
        assert_eq!(computer.cache_stats().hits, 1);
        assert_eq!(computer.cache_stats().misses, 1);
        
        // Reset metrics
        computer.reset_metrics();
        let stats = computer.cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_rate, 0.0);
        
        // Cache should still contain the entry
        assert_eq!(computer.cache_size(), 1);
    }

    #[test]
    fn test_is_cache_effective() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        // Not enough requests yet
        computer.compute_transition(&weights, 5);
        assert!(!computer.is_cache_effective());
        
        // Generate good hit rate (>70%)
        for _ in 0..15 {
            computer.compute_transition(&weights, 5);
        }
        
        // Should be effective now (15 hits, 1 miss = 93.75% hit rate)
        assert!(computer.is_cache_effective());
    }

    #[test]
    fn test_is_cache_not_effective_low_hit_rate() {
        let mut computer = TransitionComputer::new(100, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        // Generate poor hit rate by accessing different timesteps (within bounds)
        for t in 0..20 {
            computer.compute_transition(&weights, t);
        }
        
        // All misses, 0% hit rate - not effective
        assert!(!computer.is_cache_effective());
    }

    #[test]
    fn test_tune_cache_for_small_model() {
        let mut computer = TransitionComputer::new(100, BetaSchedule::Linear, 0.0001, 0.02);
        
        // Small model: 1B parameters
        computer.tune_cache_for_model_size(1_000_000_000);
        
        // Should be 100 timesteps * 20 layers = 2000
        assert_eq!(computer.cache_capacity(), 2000);
    }

    #[test]
    fn test_tune_cache_for_medium_model() {
        let mut computer = TransitionComputer::new(100, BetaSchedule::Linear, 0.0001, 0.02);
        
        // Medium model: 7B parameters
        computer.tune_cache_for_model_size(7_000_000_000);
        
        // Should be 100 timesteps * 40 layers = 4000
        assert_eq!(computer.cache_capacity(), 4000);
    }

    #[test]
    fn test_tune_cache_for_large_model() {
        let mut computer = TransitionComputer::new(100, BetaSchedule::Linear, 0.0001, 0.02);
        
        // Large model: 70B parameters
        computer.tune_cache_for_model_size(70_000_000_000);
        
        // Should be 100 timesteps * 80 layers = 8000
        assert_eq!(computer.cache_capacity(), 8000);
    }

    #[test]
    fn test_tune_cache_resets_metrics() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        // Generate some cache activity
        computer.compute_transition(&weights, 5);
        computer.compute_transition(&weights, 5);
        assert_eq!(computer.cache_stats().hits, 1);
        
        // Tune cache
        computer.tune_cache_for_model_size(1_000_000_000);
        
        // Metrics should be reset
        let stats = computer.cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_default_cache_size_computation() {
        let computer = TransitionComputer::new(100, BetaSchedule::Linear, 0.0001, 0.02);
        
        // Default should be 100 * 10 = 1000
        assert_eq!(computer.cache_capacity(), 1000);
    }

    #[test]
    fn test_cache_stats_structure() {
        let mut computer = TransitionComputer::new(10, BetaSchedule::Linear, 0.0001, 0.02);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        computer.compute_transition(&weights, 5);
        computer.compute_transition(&weights, 5);
        computer.compute_transition(&weights, 6);
        
        let stats = computer.cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert!((stats.hit_rate - 0.3333).abs() < 0.01);
        assert_eq!(stats.size, 2);
        assert!(stats.capacity > 0);
    }

    #[test]
    fn test_lru_eviction_order() {
        let mut computer = TransitionComputer::with_cache_capacity(10, BetaSchedule::Linear, 0.0001, 0.02, 2);
        
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        
        // Add timestep 0 and 1
        computer.compute_transition(&weights, 0);
        computer.compute_transition(&weights, 1);
        assert_eq!(computer.cache_size(), 2);
        
        // Access timestep 0 again (makes it more recently used)
        computer.compute_transition(&weights, 0);
        
        // Add timestep 2 - should evict timestep 1 (least recently used)
        computer.compute_transition(&weights, 2);
        assert_eq!(computer.cache_size(), 2);
        
        // Reset metrics to test which entries are cached
        computer.reset_metrics();
        
        // Timestep 0 should be cached (hit)
        computer.compute_transition(&weights, 0);
        assert_eq!(computer.cache_stats().hits, 1);
        assert_eq!(computer.cache_stats().misses, 0);
        
        // Timestep 2 should be cached (hit)
        computer.compute_transition(&weights, 2);
        assert_eq!(computer.cache_stats().hits, 2);
        assert_eq!(computer.cache_stats().misses, 0);
        
        // Timestep 1 should be evicted (miss)
        computer.compute_transition(&weights, 1);
        assert_eq!(computer.cache_stats().hits, 2);
        assert_eq!(computer.cache_stats().misses, 1);
    }

    #[test]
    fn test_cache_with_zero_capacity_fallback() {
        // Should fallback to minimum capacity of 1
        let computer = TransitionComputer::with_cache_capacity(10, BetaSchedule::Linear, 0.0001, 0.02, 0);
        assert_eq!(computer.cache_capacity(), 1);
    }
}


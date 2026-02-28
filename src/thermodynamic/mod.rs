//! Thermodynamic Constraint Enhancement Module
//!
//! This module implements explicit thermodynamic constraints (Markov chain properties)
//! to enhance quantization quality for both diffusion models and autoregressive models.
//!
//! # Overview
//!
//! The module provides three main components:
//!
//! 1. **Markov Validator** - Validates and monitors Markov smoothness properties
//! 2. **Boundary Smoother** - Smooths parameter transitions at group boundaries
//! 3. **Transition Optimizer** - Optimizes quantization parameters to preserve transition probabilities
//!
//! # Architecture
//!
//! ```text
//! Quantization Pipeline
//!         │
//!         ▼
//! ┌────────────────────┐
//! │ Phase 1: Validate  │  ← MarkovValidator
//! │ (Optional)         │
//! └────────┬───────────┘
//!          │
//!          ▼
//! ┌────────────────────┐
//! │ Phase 2: Smooth    │  ← BoundarySmoother
//! │ (Optional)         │
//! └────────┬───────────┘
//!          │
//!          ▼
//! ┌────────────────────┐
//! │ Phase 3: Optimize  │  ← TransitionOptimizer
//! │ (Optional)         │
//! └────────┬───────────┘
//!          │
//!          ▼
//!    Quantized Model
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use arrow_quant_v2::thermodynamic::{MarkovValidator, BoundarySmoother};
//!
//! // Phase 1: Validate Markov properties
//! let validator = MarkovValidator::new(0.3); // 30% threshold
//! let validation = validator.validate(&time_group_params);
//! println!("Smoothness score: {}", validation.smoothness_score);
//!
//! // Phase 2: Smooth boundaries
//! let smoother = BoundarySmoother::new(5); // window size = 5
//! let smoothed_params = smoother.smooth_boundaries(&time_group_params);
//! ```
//!
//! # Features
//!
//! - **Markov Validation**: Compute smoothness scores and detect violations
//! - **Boundary Smoothing**: Linear, cubic, and sigmoid interpolation methods
//! - **Transition Optimization**: Preserve transition probabilities via gradient descent
//! - **Backward Compatible**: All features are opt-in (disabled by default)
//! - **Performance Optimized**: Caching, parallel processing, SIMD support
//!
//! # References
//!
//! - [Thermodynamic Quantization Analysis](../../../THERMODYNAMIC_QUANTIZATION_ANALYSIS.md)
//! - [Enhancement Roadmap](../../../THERMODYNAMIC_ENHANCEMENT_ROADMAP.md)
//! - [Design Document](.kiro/specs/thermodynamic-enhancement/design.md)

// Phase 1: Markov Validation
pub mod markov_validator;

// Phase 2: Boundary Smoothing
pub mod boundary_smoothing;

// Phase 3: Transition Optimization
pub mod transition_matrix;
pub mod loss_functions;
pub mod optimizer;

// Information Flow
pub mod entropy;

// Re-exports for convenience
pub use markov_validator::{
    MarkovValidator,
    ValidationResult,
    MarkovViolation,
    ViolationSeverity,
};

// Phase 2 re-exports
pub use boundary_smoothing::{BoundarySmoother, InterpolationMethod};

// Phase 3 re-exports
pub use transition_matrix::{TransitionComputer, TransitionMatrix, BetaSchedule};
pub use loss_functions::ThermodynamicLoss;
pub use optimizer::{TransitionOptimizer, OptimizerConfig, OptimizationResult};

// Entropy Analysis
pub use entropy::EntropyAnalyzer;

/// Metrics collected during thermodynamic validation and optimization
#[derive(Debug, Clone, Default)]
pub struct ThermodynamicMetrics {
    /// Overall Markov smoothness score (0-1, higher is better)
    pub smoothness_score: f32,
    /// Per-boundary smoothness scores
    pub boundary_scores: Vec<f32>,
    /// Number of violations detected
    pub violation_count: usize,
    /// List of violations with details
    pub violations: Vec<MarkovViolation>,
    /// Number of optimization iterations performed (Phase 3)
    pub optimization_iterations: usize,
    /// Whether optimization converged (Phase 3)
    pub optimization_converged: bool,
    /// Final loss value after optimization (Phase 3)
    pub final_loss: f32,
}

impl ThermodynamicMetrics {
    /// Create metrics from validation result
    pub fn from_validation(result: &ValidationResult) -> Self {
        Self {
            smoothness_score: result.smoothness_score,
            boundary_scores: result.boundary_scores.clone(),
            violation_count: result.violations.len(),
            violations: result.violations.clone(),
            optimization_iterations: 0,
            optimization_converged: false,
            final_loss: 0.0,
        }
    }

    /// Check if validation passed (no violations)
    pub fn is_valid(&self) -> bool {
        self.violation_count == 0
    }
}


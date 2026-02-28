//! Markov Chain Property Validator
//!
//! This module provides validation and monitoring of Markov smoothness properties
//! in quantization parameters. It detects parameter jumps at group boundaries
//! that can break Markov chain properties and lead to error accumulation.

use crate::time_aware::TimeGroupParams;

/// Severity level of a Markov property violation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationSeverity {
    /// Low severity: parameter jump < 30%
    Low,
    /// Medium severity: parameter jump 30-50%
    Medium,
    /// High severity: parameter jump > 50%
    High,
}

impl std::fmt::Display for ViolationSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViolationSeverity::Low => write!(f, "low"),
            ViolationSeverity::Medium => write!(f, "medium"),
            ViolationSeverity::High => write!(f, "high"),
        }
    }
}

/// A detected Markov property violation at a group boundary
#[derive(Debug, Clone)]
pub struct MarkovViolation {
    /// Index of the boundary where violation occurred
    pub boundary_idx: usize,
    /// Magnitude of scale parameter jump (as fraction)
    pub scale_jump: f32,
    /// Magnitude of zero_point parameter jump (normalized to 0-1)
    pub zero_point_jump: f32,
    /// Severity level of the violation
    pub severity: ViolationSeverity,
}

/// Result of Markov property validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether all parameters pass the smoothness threshold
    pub is_valid: bool,
    /// Overall smoothness score (0-1, higher is better)
    pub smoothness_score: f32,
    /// List of detected violations
    pub violations: Vec<MarkovViolation>,
    /// Per-boundary smoothness scores
    pub boundary_scores: Vec<f32>,
}

/// Validator for Markov chain properties in quantization parameters
///
/// # Example
///
/// ```rust,ignore
/// use arrow_quant_v2::thermodynamic::MarkovValidator;
///
/// let validator = MarkovValidator::new(0.3); // 30% threshold
/// let result = validator.validate(&time_group_params);
///
/// if !result.is_valid {
///     println!("Found {} violations", result.violations.len());
///     for violation in &result.violations {
///         println!("  Boundary {}: {:.1}% jump ({})",
///             violation.boundary_idx,
///             violation.scale_jump * 100.0,
///             violation.severity
///         );
///     }
/// }
/// ```
pub struct MarkovValidator {
    /// Threshold for parameter jumps (as fraction, e.g., 0.3 = 30%)
    smoothness_threshold: f32,
    /// Whether to enable detailed logging
    enable_logging: bool,
}

impl MarkovValidator {
    /// Create a new Markov validator with the given smoothness threshold
    ///
    /// # Arguments
    ///
    /// * `smoothness_threshold` - Maximum allowed parameter jump as fraction (e.g., 0.3 = 30%)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let validator = MarkovValidator::new(0.3); // Allow up to 30% jumps
    /// ```
    pub fn new(smoothness_threshold: f32) -> Self {
        Self {
            smoothness_threshold,
            enable_logging: true,
        }
    }

    /// Create a validator with custom settings
    pub fn with_logging(mut self, enable: bool) -> Self {
        self.enable_logging = enable;
        self
    }

    /// Validate Markov smoothness properties of quantization parameters
    ///
    /// # Arguments
    ///
    /// * `params` - Time group parameters to validate
    ///
    /// # Returns
    ///
    /// A `ValidationResult` containing smoothness score and any violations
    pub fn validate(&self, params: &[TimeGroupParams]) -> ValidationResult {
        if params.len() < 2 {
            // Need at least 2 groups to check boundaries
            return ValidationResult {
                is_valid: true,
                smoothness_score: 1.0,
                violations: Vec::new(),
                boundary_scores: Vec::new(),
            };
        }

        let mut violations = Vec::new();
        let mut boundary_scores = Vec::new();

        // Check each boundary
        for i in 0..params.len() - 1 {
            let left = &params[i];
            let right = &params[i + 1];

            // Compute parameter jumps
            let scale_jump = (right.scale - left.scale).abs() / left.scale;
            let zero_point_jump = (right.zero_point - left.zero_point).abs() / 255.0;

            // Compute boundary smoothness score
            let boundary_score = 1.0 - ((scale_jump + zero_point_jump) / 2.0).min(1.0);
            boundary_scores.push(boundary_score);

            // Check if jump exceeds threshold
            if scale_jump > self.smoothness_threshold {
                let severity = if scale_jump > 0.5 {
                    ViolationSeverity::High
                } else if scale_jump > 0.3 {
                    ViolationSeverity::Medium
                } else {
                    ViolationSeverity::Low
                };

                violations.push(MarkovViolation {
                    boundary_idx: i,
                    scale_jump,
                    zero_point_jump,
                    severity,
                });

                if self.enable_logging {
                    log::warn!(
                        "Markov violation at boundary {}: {:.1}% scale jump ({})",
                        i,
                        scale_jump * 100.0,
                        severity
                    );
                }
            }
        }

        // Compute overall smoothness score
        let smoothness_score = self.compute_smoothness_score(params);

        if self.enable_logging {
            log::info!("Markov smoothness score: {:.3}", smoothness_score);
        }

        ValidationResult {
            is_valid: violations.is_empty(),
            smoothness_score,
            violations,
            boundary_scores,
        }
    }

    /// Compute overall smoothness score (0-1, higher is better)
    ///
    /// The score is computed as 1 minus the normalized total parameter jump.
    /// A score of 1.0 means perfectly smooth (no jumps), while 0.0 means
    /// maximum possible jumps.
    pub fn compute_smoothness_score(&self, params: &[TimeGroupParams]) -> f32 {
        if params.len() < 2 {
            return 1.0;
        }

        let mut total_jump = 0.0;
        let mut max_possible_jump = 0.0;

        for i in 0..params.len() - 1 {
            let left = &params[i];
            let right = &params[i + 1];

            // Compute normalized jumps
            let scale_jump = (right.scale - left.scale).abs() / left.scale;
            let zero_point_jump = (right.zero_point - left.zero_point).abs() / 255.0;

            total_jump += scale_jump + zero_point_jump;
            max_possible_jump += 2.0; // Max jump is 100% for each parameter
        }

        // Normalize to 0-1 range
        1.0 - (total_jump / max_possible_jump).min(1.0)
    }

    /// Get the smoothness threshold
    pub fn threshold(&self) -> f32 {
        self.smoothness_threshold
    }
}

impl Default for MarkovValidator {
    fn default() -> Self {
        Self::new(0.3) // Default 30% threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_params() -> Vec<TimeGroupParams> {
        vec![
            TimeGroupParams {
                time_range: (0, 25),
                scale: 0.05,
                zero_point: 128.0,
                group_size: 256,
            },
            TimeGroupParams {
                time_range: (25, 50),
                scale: 0.08, // 60% jump
                zero_point: 120.0,
                group_size: 128,
            },
            TimeGroupParams {
                time_range: (50, 75),
                scale: 0.09, // 12.5% jump
                zero_point: 115.0,
                group_size: 64,
            },
        ]
    }

    #[test]
    fn test_perfect_smoothness() {
        let params = vec![
            TimeGroupParams {
                time_range: (0, 50),
                scale: 0.05,
                zero_point: 128.0,
                group_size: 256,
            },
            TimeGroupParams {
                time_range: (50, 100),
                scale: 0.05, // No jump
                zero_point: 128.0,
                group_size: 256,
            },
        ];

        let validator = MarkovValidator::new(0.3);
        let result = validator.validate(&params);

        assert!(result.is_valid);
        assert!(result.smoothness_score > 0.99);
        assert_eq!(result.violations.len(), 0);
    }

    #[test]
    fn test_large_jump_detection() {
        let params = create_test_params();
        let validator = MarkovValidator::new(0.3);
        let result = validator.validate(&params);

        assert!(!result.is_valid);
        // The smoothness score should be less than 1.0 due to jumps
        assert!(result.smoothness_score < 1.0);
        assert_eq!(result.violations.len(), 1);

        let violation = &result.violations[0];
        assert_eq!(violation.boundary_idx, 0);
        assert!(violation.scale_jump > 0.5);
        assert_eq!(violation.severity, ViolationSeverity::High);
    }

    #[test]
    fn test_smoothness_score_computation() {
        let params = create_test_params();
        let validator = MarkovValidator::new(0.3);
        let score = validator.compute_smoothness_score(&params);

        // Should be less than 1.0 due to jumps
        assert!(score < 1.0);
        // Should be greater than 0.0 (not worst case)
        assert!(score > 0.0);
    }

    #[test]
    fn test_single_group_no_violations() {
        let params = vec![TimeGroupParams {
            time_range: (0, 100),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        }];

        let validator = MarkovValidator::new(0.3);
        let result = validator.validate(&params);

        assert!(result.is_valid);
        assert_eq!(result.smoothness_score, 1.0);
        assert_eq!(result.violations.len(), 0);
    }

    #[test]
    fn test_violation_severity_levels() {
        // Test low severity (< 30%)
        let params_low = vec![
            TimeGroupParams {
                time_range: (0, 50),
                scale: 0.10,
                zero_point: 128.0,
                group_size: 256,
            },
            TimeGroupParams {
                time_range: (50, 100),
                scale: 0.12, // 20% jump
                zero_point: 128.0,
                group_size: 256,
            },
        ];

        let validator = MarkovValidator::new(0.15); // Lower threshold
        let result = validator.validate(&params_low);
        assert_eq!(result.violations[0].severity, ViolationSeverity::Low);

        // Test medium severity (30-50%)
        let params_medium = vec![
            TimeGroupParams {
                time_range: (0, 50),
                scale: 0.10,
                zero_point: 128.0,
                group_size: 256,
            },
            TimeGroupParams {
                time_range: (50, 100),
                scale: 0.14, // 40% jump
                zero_point: 128.0,
                group_size: 256,
            },
        ];

        let result = validator.validate(&params_medium);
        assert_eq!(result.violations[0].severity, ViolationSeverity::Medium);

        // Test high severity (> 50%)
        let params_high = vec![
            TimeGroupParams {
                time_range: (0, 50),
                scale: 0.10,
                zero_point: 128.0,
                group_size: 256,
            },
            TimeGroupParams {
                time_range: (50, 100),
                scale: 0.16, // 60% jump
                zero_point: 128.0,
                group_size: 256,
            },
        ];

        let result = validator.validate(&params_high);
        assert_eq!(result.violations[0].severity, ViolationSeverity::High);
    }

    #[test]
    fn test_boundary_scores() {
        let params = create_test_params();
        let validator = MarkovValidator::new(0.3);
        let result = validator.validate(&params);

        assert_eq!(result.boundary_scores.len(), 2);
        // First boundary has large jump, should have low score
        assert!(result.boundary_scores[0] < 0.7);
        // Second boundary has small jump, should have high score
        assert!(result.boundary_scores[1] > 0.9);
    }
}

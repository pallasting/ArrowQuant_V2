//! Comprehensive unit tests for MarkovValidator
//!
//! Tests cover:
//! - Smoothness score computation with known inputs
//! - Violation detection at different thresholds
//! - Severity classification
//! - Edge cases (empty params, single group, identical params)
//!
//! Requirements: REQ-3.1.1

use arrow_quant_v2::thermodynamic::{MarkovValidator, ViolationSeverity};
use arrow_quant_v2::time_aware::TimeGroupParams;

/// Helper function to create test parameters with specific scale values
fn create_params_with_scales(scales: Vec<f32>) -> Vec<TimeGroupParams> {
    scales
        .into_iter()
        .enumerate()
        .map(|(i, scale)| TimeGroupParams {
            time_range: (i * 25, (i + 1) * 25),
            scale,
            zero_point: 128.0,
            group_size: 256,
        })
        .collect()
}

/// Helper function to create test parameters with specific zero points
fn create_params_with_zero_points(zero_points: Vec<f32>) -> Vec<TimeGroupParams> {
    zero_points
        .into_iter()
        .enumerate()
        .map(|(i, zp)| TimeGroupParams {
            time_range: (i * 25, (i + 1) * 25),
            scale: 0.05,
            zero_point: zp,
            group_size: 256,
        })
        .collect()
}

// ============================================================================
// Smoothness Score Computation Tests
// ============================================================================

#[test]
fn test_smoothness_score_perfect_smoothness() {
    // All parameters identical - perfect smoothness
    let params = create_params_with_scales(vec![0.05, 0.05, 0.05, 0.05]);
    let validator = MarkovValidator::new(0.3);
    let score = validator.compute_smoothness_score(&params);
    
    assert!(
        (score - 1.0).abs() < 1e-6,
        "Perfect smoothness should give score ~1.0, got {}",
        score
    );
}

#[test]
fn test_smoothness_score_known_input_small_jumps() {
    // Known input: 10% jumps between consecutive groups
    let params = create_params_with_scales(vec![0.10, 0.11, 0.121, 0.1331]);
    let validator = MarkovValidator::new(0.3);
    let score = validator.compute_smoothness_score(&params);
    
    // With 10% jumps, total_jump = 3 * (0.1 + 0.0) = 0.3
    // max_possible_jump = 3 * 2.0 = 6.0
    // score = 1.0 - (0.3 / 6.0) = 0.95
    assert!(
        (score - 0.95).abs() < 0.01,
        "Expected score ~0.95 for 10% jumps, got {}",
        score
    );
}

#[test]
fn test_smoothness_score_known_input_large_jumps() {
    // Known input: 100% jumps (doubling)
    let params = create_params_with_scales(vec![0.05, 0.10, 0.20]);
    let validator = MarkovValidator::new(0.3);
    let score = validator.compute_smoothness_score(&params);
    
    // With 100% jumps, total_jump = 2 * (1.0 + 0.0) = 2.0
    // max_possible_jump = 2 * 2.0 = 4.0
    // score = 1.0 - (2.0 / 4.0) = 0.5
    assert!(
        (score - 0.5).abs() < 0.01,
        "Expected score ~0.5 for 100% jumps, got {}",
        score
    );
}

#[test]
fn test_smoothness_score_with_zero_point_jumps() {
    // Test that zero_point jumps also affect smoothness
    let params = create_params_with_zero_points(vec![128.0, 200.0, 50.0]);
    let validator = MarkovValidator::new(0.3);
    let score = validator.compute_smoothness_score(&params);
    
    // Large zero_point jumps should reduce smoothness score
    assert!(
        score < 0.8,
        "Large zero_point jumps should reduce score below 0.8, got {}",
        score
    );
}

#[test]
fn test_smoothness_score_combined_jumps() {
    // Test with both scale and zero_point jumps
    let params = vec![
        TimeGroupParams {
            time_range: (0, 25),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (25, 50),
            scale: 0.10, // 100% scale jump
            zero_point: 200.0, // ~28% zero_point jump (72/255)
            group_size: 256,
        },
    ];
    
    let validator = MarkovValidator::new(0.3);
    let score = validator.compute_smoothness_score(&params);
    
    // Combined jumps should result in lower score
    assert!(
        score < 0.7,
        "Combined jumps should reduce score below 0.7, got {}",
        score
    );
}

// ============================================================================
// Violation Detection Tests
// ============================================================================

#[test]
fn test_violation_detection_no_violations() {
    // Small jumps below threshold
    let params = create_params_with_scales(vec![0.10, 0.12, 0.14]); // 20%, 16.7% jumps
    let validator = MarkovValidator::new(0.3); // 30% threshold
    let result = validator.validate(&params);
    
    assert!(result.is_valid, "Should be valid with jumps below threshold");
    assert_eq!(result.violations.len(), 0, "Should have no violations");
}

#[test]
fn test_violation_detection_single_violation() {
    // One jump exceeds threshold
    let params = create_params_with_scales(vec![0.10, 0.15, 0.16]); // 50%, 6.7% jumps
    let validator = MarkovValidator::new(0.3); // 30% threshold
    let result = validator.validate(&params);
    
    assert!(!result.is_valid, "Should be invalid with jump above threshold");
    assert_eq!(result.violations.len(), 1, "Should have exactly 1 violation");
    assert_eq!(result.violations[0].boundary_idx, 0, "Violation at first boundary");
}

#[test]
fn test_violation_detection_multiple_violations() {
    // Multiple jumps exceed threshold
    let params = create_params_with_scales(vec![0.10, 0.15, 0.10, 0.15]); // 50%, -33%, 50% jumps
    let validator = MarkovValidator::new(0.3); // 30% threshold
    let result = validator.validate(&params);
    
    assert!(!result.is_valid, "Should be invalid with multiple violations");
    assert_eq!(result.violations.len(), 3, "Should have 3 violations");
}

#[test]
fn test_violation_detection_different_thresholds() {
    let params = create_params_with_scales(vec![0.10, 0.12]); // 20% jump
    
    // With 10% threshold - should violate
    let validator_strict = MarkovValidator::new(0.1);
    let result_strict = validator_strict.validate(&params);
    assert!(!result_strict.is_valid, "Should violate with 10% threshold");
    assert_eq!(result_strict.violations.len(), 1);
    
    // With 30% threshold - should pass
    let validator_loose = MarkovValidator::new(0.3);
    let result_loose = validator_loose.validate(&params);
    assert!(result_loose.is_valid, "Should pass with 30% threshold");
    assert_eq!(result_loose.violations.len(), 0);
}

#[test]
fn test_violation_boundary_index() {
    // Test that boundary indices are correctly reported
    let params = create_params_with_scales(vec![0.10, 0.11, 0.20, 0.21]); // Small, large, small jumps
    let validator = MarkovValidator::new(0.3);
    let result = validator.validate(&params);
    
    assert_eq!(result.violations.len(), 1, "Should have 1 violation");
    assert_eq!(
        result.violations[0].boundary_idx, 1,
        "Violation should be at boundary 1 (between groups 1 and 2)"
    );
}

#[test]
fn test_violation_jump_magnitude() {
    let params = create_params_with_scales(vec![0.10, 0.16]); // 60% jump
    let validator = MarkovValidator::new(0.3);
    let result = validator.validate(&params);
    
    assert_eq!(result.violations.len(), 1);
    let violation = &result.violations[0];
    
    // Check scale_jump is approximately 0.6 (60%)
    assert!(
        (violation.scale_jump - 0.6).abs() < 0.01,
        "Expected scale_jump ~0.6, got {}",
        violation.scale_jump
    );
}

// ============================================================================
// Severity Classification Tests
// ============================================================================

#[test]
fn test_severity_low() {
    // Jump between 0% and 30%
    let params = create_params_with_scales(vec![0.10, 0.12]); // 20% jump
    let validator = MarkovValidator::new(0.15); // Lower threshold to trigger violation
    let result = validator.validate(&params);
    
    assert_eq!(result.violations.len(), 1);
    assert_eq!(
        result.violations[0].severity,
        ViolationSeverity::Low,
        "20% jump should be Low severity"
    );
}

#[test]
fn test_severity_medium() {
    // Jump between 30% and 50%
    let params = create_params_with_scales(vec![0.10, 0.14]); // 40% jump
    let validator = MarkovValidator::new(0.2); // Lower threshold to trigger violation
    let result = validator.validate(&params);
    
    assert_eq!(result.violations.len(), 1);
    assert_eq!(
        result.violations[0].severity,
        ViolationSeverity::Medium,
        "40% jump should be Medium severity"
    );
}

#[test]
fn test_severity_high() {
    // Jump greater than 50%
    let params = create_params_with_scales(vec![0.10, 0.16]); // 60% jump
    let validator = MarkovValidator::new(0.3);
    let result = validator.validate(&params);
    
    assert_eq!(result.violations.len(), 1);
    assert_eq!(
        result.violations[0].severity,
        ViolationSeverity::High,
        "60% jump should be High severity"
    );
}

#[test]
fn test_severity_boundary_cases() {
    // Test exact boundary values
    // Note: severity uses > comparisons, so:
    // - scale_jump <= 0.3: Low
    // - 0.3 < scale_jump <= 0.5: Medium
    // - scale_jump > 0.5: High
    
    // Exactly 30% - should be Low (not > 0.3)
    let params_30 = create_params_with_scales(vec![0.10, 0.13]); // 30% jump
    let validator = MarkovValidator::new(0.2);
    let result_30 = validator.validate(&params_30);
    assert_eq!(result_30.violations[0].severity, ViolationSeverity::Low);
    
    // Around 40% - should be Medium (> 0.3 but not > 0.5)
    let params_40 = create_params_with_scales(vec![0.10, 0.14]); // 40% jump
    let result_40 = validator.validate(&params_40);
    assert_eq!(result_40.violations[0].severity, ViolationSeverity::Medium);
    
    // Over 50% - should be High (> 0.5)
    let params_60 = create_params_with_scales(vec![0.10, 0.16]); // 60% jump
    let result_60 = validator.validate(&params_60);
    assert_eq!(result_60.violations[0].severity, ViolationSeverity::High);
    
    // Just over 30% - should be Medium (> 0.3 but not > 0.5)
    let params_31 = create_params_with_scales(vec![0.10, 0.131]); // 31% jump
    let result_31 = validator.validate(&params_31);
    assert_eq!(result_31.violations[0].severity, ViolationSeverity::Medium);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[test]
fn test_edge_case_empty_params() {
    let params: Vec<TimeGroupParams> = vec![];
    let validator = MarkovValidator::new(0.3);
    let result = validator.validate(&params);
    
    assert!(result.is_valid, "Empty params should be valid");
    assert_eq!(result.smoothness_score, 1.0, "Empty params should have perfect score");
    assert_eq!(result.violations.len(), 0, "Empty params should have no violations");
    assert_eq!(result.boundary_scores.len(), 0, "Empty params should have no boundary scores");
}

#[test]
fn test_edge_case_single_group() {
    let params = vec![TimeGroupParams {
        time_range: (0, 100),
        scale: 0.05,
        zero_point: 128.0,
        group_size: 256,
    }];
    
    let validator = MarkovValidator::new(0.3);
    let result = validator.validate(&params);
    
    assert!(result.is_valid, "Single group should be valid");
    assert_eq!(result.smoothness_score, 1.0, "Single group should have perfect score");
    assert_eq!(result.violations.len(), 0, "Single group should have no violations");
    assert_eq!(result.boundary_scores.len(), 0, "Single group should have no boundary scores");
}

#[test]
fn test_edge_case_identical_params() {
    // All parameters exactly identical
    let params = vec![
        TimeGroupParams {
            time_range: (0, 25),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (25, 50),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (50, 75),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        },
    ];
    
    let validator = MarkovValidator::new(0.3);
    let result = validator.validate(&params);
    
    assert!(result.is_valid, "Identical params should be valid");
    assert!(
        (result.smoothness_score - 1.0).abs() < 1e-6,
        "Identical params should have perfect score"
    );
    assert_eq!(result.violations.len(), 0, "Identical params should have no violations");
    assert_eq!(result.boundary_scores.len(), 2, "Should have 2 boundary scores");
    
    // All boundary scores should be perfect
    for score in &result.boundary_scores {
        assert!(
            (score - 1.0).abs() < 1e-6,
            "All boundary scores should be ~1.0, got {}",
            score
        );
    }
}

#[test]
fn test_edge_case_very_small_scale() {
    // Test with very small scale values (near zero)
    let params = create_params_with_scales(vec![0.001, 0.002]); // 100% jump but small absolute values
    let validator = MarkovValidator::new(0.3);
    let result = validator.validate(&params);
    
    // Should still detect the 100% relative jump
    assert!(!result.is_valid, "Should detect large relative jump even with small scales");
    assert_eq!(result.violations.len(), 1);
}

#[test]
fn test_edge_case_zero_scale() {
    // Test with zero scale (edge case that might cause division by zero)
    let params = vec![
        TimeGroupParams {
            time_range: (0, 25),
            scale: 0.0,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (25, 50),
            scale: 0.05,
            zero_point: 128.0,
            group_size: 256,
        },
    ];
    
    let validator = MarkovValidator::new(0.3);
    // Should not panic - implementation should handle this gracefully
    let result = validator.validate(&params);
    
    // With zero scale, the jump calculation will be infinite or NaN
    // The implementation should handle this case
    assert!(result.violations.len() > 0 || !result.is_valid);
}

#[test]
fn test_edge_case_negative_jump() {
    // Test with decreasing scale (negative jump direction)
    let params = create_params_with_scales(vec![0.20, 0.10]); // -50% jump (50% decrease)
    let validator = MarkovValidator::new(0.3);
    let result = validator.validate(&params);
    
    // Should detect violation regardless of jump direction (uses abs())
    assert!(!result.is_valid, "Should detect large decrease as violation");
    assert_eq!(result.violations.len(), 1);
    assert!(
        (result.violations[0].scale_jump - 0.5).abs() < 0.01,
        "Should report absolute jump magnitude"
    );
}

// ============================================================================
// Boundary Scores Tests
// ============================================================================

#[test]
fn test_boundary_scores_count() {
    let params = create_params_with_scales(vec![0.10, 0.11, 0.12, 0.13]);
    let validator = MarkovValidator::new(0.3);
    let result = validator.validate(&params);
    
    // With 4 groups, should have 3 boundaries
    assert_eq!(
        result.boundary_scores.len(),
        3,
        "Should have n-1 boundary scores for n groups"
    );
}

#[test]
fn test_boundary_scores_values() {
    let params = vec![
        TimeGroupParams {
            time_range: (0, 25),
            scale: 0.10,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (25, 50),
            scale: 0.10, // No jump - perfect boundary
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (50, 75),
            scale: 0.20, // 100% jump - poor boundary
            zero_point: 128.0,
            group_size: 256,
        },
    ];
    
    let validator = MarkovValidator::new(0.3);
    let result = validator.validate(&params);
    
    assert_eq!(result.boundary_scores.len(), 2);
    
    // First boundary (no jump) should have high score
    assert!(
        result.boundary_scores[0] > 0.99,
        "Perfect boundary should have score > 0.99, got {}",
        result.boundary_scores[0]
    );
    
    // Second boundary (large jump) should have low score
    assert!(
        result.boundary_scores[1] < 0.6,
        "Poor boundary should have score < 0.6, got {}",
        result.boundary_scores[1]
    );
}

#[test]
fn test_boundary_scores_range() {
    // Test that boundary scores are always in [0, 1] range
    let params = create_params_with_scales(vec![0.01, 0.10, 0.05, 0.20]); // Various jumps
    let validator = MarkovValidator::new(0.3);
    let result = validator.validate(&params);
    
    for (i, score) in result.boundary_scores.iter().enumerate() {
        assert!(
            *score >= 0.0 && *score <= 1.0,
            "Boundary score {} should be in [0, 1], got {}",
            i,
            score
        );
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_default_threshold() {
    let validator = MarkovValidator::default();
    assert_eq!(validator.threshold(), 0.3, "Default threshold should be 0.3 (30%)");
}

#[test]
fn test_custom_threshold() {
    let validator = MarkovValidator::new(0.5);
    assert_eq!(validator.threshold(), 0.5, "Custom threshold should be preserved");
}

#[test]
fn test_logging_configuration() {
    let validator = MarkovValidator::new(0.3).with_logging(false);
    
    // Test that validator works with logging disabled
    let params = create_params_with_scales(vec![0.10, 0.20]);
    let result = validator.validate(&params);
    
    // Should still detect violations, just not log them
    assert!(!result.is_valid);
    assert_eq!(result.violations.len(), 1);
}

// ============================================================================
// Integration-style Tests
// ============================================================================

#[test]
fn test_realistic_time_aware_params() {
    // Simulate realistic time-aware quantization parameters
    let params = vec![
        TimeGroupParams {
            time_range: (0, 250),
            scale: 0.045,
            zero_point: 125.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (250, 500),
            scale: 0.052,
            zero_point: 130.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (500, 750),
            scale: 0.048,
            zero_point: 128.0,
            group_size: 256,
        },
        TimeGroupParams {
            time_range: (750, 1000),
            scale: 0.050,
            zero_point: 127.0,
            group_size: 256,
        },
    ];
    
    let validator = MarkovValidator::new(0.3);
    let result = validator.validate(&params);
    
    // These realistic params should have good smoothness
    assert!(result.is_valid, "Realistic params should be valid");
    assert!(
        result.smoothness_score > 0.85,
        "Realistic params should have high smoothness score, got {}",
        result.smoothness_score
    );
}

#[test]
fn test_worst_case_alternating_params() {
    // Worst case: alternating between very different values
    let params = create_params_with_scales(vec![0.01, 0.10, 0.01, 0.10, 0.01]);
    let validator = MarkovValidator::new(0.3);
    let result = validator.validate(&params);
    
    assert!(!result.is_valid, "Alternating params should be invalid");
    assert!(
        result.smoothness_score < 0.3,
        "Alternating params should have very low smoothness score, got {}",
        result.smoothness_score
    );
    assert_eq!(result.violations.len(), 4, "Should have violations at all boundaries");
}

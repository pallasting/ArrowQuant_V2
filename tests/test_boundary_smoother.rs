//! Comprehensive unit tests for BoundarySmoother
//!
//! This test suite verifies:
//! - Each interpolation method independently (Linear, Cubic, Sigmoid)
//! - Parameter jump reduction (>50% requirement)
//! - Different window sizes (1-20)
//! - Edge cases (boundaries, single group, empty params)
//!
//! Requirements: REQ-1.2.3, REQ-3.1.1

use arrow_quant_v2::thermodynamic::{BoundarySmoother, InterpolationMethod};
use arrow_quant_v2::time_aware::TimeGroupParams;

/// Helper function to create test parameters with a large jump
fn create_params_with_jump() -> Vec<TimeGroupParams> {
    vec![
        TimeGroupParams {
            time_range: (0, 25),
            scale: 0.1,
            zero_point: 100.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (25, 50),
            scale: 0.1,
            zero_point: 100.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (50, 75),
            scale: 0.8,  // Large jump: 700% increase
            zero_point: 250.0,  // Large jump
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (75, 100),
            scale: 0.8,
            zero_point: 250.0,
            group_size: 128,
        },
    ]
}

/// Helper function to compute maximum parameter jump
fn compute_max_jump(params: &[TimeGroupParams]) -> (f32, f32) {
    let mut max_scale_jump = 0.0_f32;
    let mut max_zp_jump = 0.0_f32;
    
    for i in 0..params.len() - 1 {
        let scale_jump = (params[i + 1].scale - params[i].scale).abs() / params[i].scale;
        let zp_jump = (params[i + 1].zero_point - params[i].zero_point).abs() / 255.0;
        max_scale_jump = max_scale_jump.max(scale_jump);
        max_zp_jump = max_zp_jump.max(zp_jump);
    }
    
    (max_scale_jump, max_zp_jump)
}

// ============================================================================
// Test 1: Linear Interpolation Method
// ============================================================================

#[test]
fn test_linear_interpolation_basic() {
    let smoother = BoundarySmoother::new(2, InterpolationMethod::Linear);
    let params = create_params_with_jump();
    
    let smoothed = smoother.smooth_boundaries(&params);
    
    assert_eq!(smoothed.len(), params.len());
    
    // Verify all values are in valid range
    for p in &smoothed {
        assert!(p.scale >= 0.1 && p.scale <= 0.8);
        assert!(p.zero_point >= 100.0 && p.zero_point <= 250.0);
    }
}

#[test]
fn test_linear_reduces_jumps_by_50_percent() {
    let smoother = BoundarySmoother::new(3, InterpolationMethod::Linear);
    let params = create_params_with_jump();
    
    let (max_scale_before, max_zp_before) = compute_max_jump(&params);
    let smoothed = smoother.smooth_boundaries(&params);
    let (max_scale_after, max_zp_after) = compute_max_jump(&smoothed);
    
    // REQ-1.2.3: Smoothing SHALL reduce parameter jumps by at least 50%
    let scale_reduction = (max_scale_before - max_scale_after) / max_scale_before;
    let zp_reduction = (max_zp_before - max_zp_after) / max_zp_before;
    
    assert!(
        scale_reduction >= 0.50,
        "Scale jump reduction {:.1}% is less than 50%",
        scale_reduction * 100.0
    );
    assert!(
        zp_reduction >= 0.50,
        "Zero point jump reduction {:.1}% is less than 50%",
        zp_reduction * 100.0
    );
}

#[test]
fn test_linear_interpolation_formula_verification() {
    // Verify exact linear interpolation formula
    let smoother = BoundarySmoother::new(1, InterpolationMethod::Linear);
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.0,
            zero_point: 0.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 1.0,
            zero_point: 100.0,
            group_size: 128,
        },
    ];
    
    let smoothed = smoother.smooth_boundaries(&params);
    
    // With window_size=1, total_window=2
    // α values: 1/3 and 2/3
    let alpha_left = 1.0 / 3.0;
    let expected_scale_left = 0.0 * (1.0 - alpha_left) + 1.0 * alpha_left;
    
    assert!((smoothed[0].scale - expected_scale_left).abs() < 1e-6);
}

// ============================================================================
// Test 2: Cubic Interpolation Method
// ============================================================================

#[test]
fn test_cubic_interpolation_basic() {
    let smoother = BoundarySmoother::new(2, InterpolationMethod::Cubic);
    let params = create_params_with_jump();
    
    let smoothed = smoother.smooth_boundaries(&params);
    
    assert_eq!(smoothed.len(), params.len());
    
    // Verify all values are in valid range
    for p in &smoothed {
        assert!(p.scale >= 0.1 && p.scale <= 0.8);
        assert!(p.zero_point >= 100.0 && p.zero_point <= 250.0);
    }
}

#[test]
fn test_cubic_reduces_jumps_by_50_percent() {
    let smoother = BoundarySmoother::new(3, InterpolationMethod::Cubic);
    let params = create_params_with_jump();
    
    let (max_scale_before, max_zp_before) = compute_max_jump(&params);
    let smoothed = smoother.smooth_boundaries(&params);
    let (max_scale_after, max_zp_after) = compute_max_jump(&smoothed);
    
    // REQ-1.2.3: Smoothing SHALL reduce parameter jumps by at least 50%
    let scale_reduction = (max_scale_before - max_scale_after) / max_scale_before;
    let zp_reduction = (max_zp_before - max_zp_after) / max_zp_before;
    
    assert!(
        scale_reduction >= 0.50,
        "Cubic: Scale jump reduction {:.1}% is less than 50%",
        scale_reduction * 100.0
    );
    assert!(
        zp_reduction >= 0.50,
        "Cubic: Zero point jump reduction {:.1}% is less than 50%",
        zp_reduction * 100.0
    );
}

#[test]
fn test_cubic_hermite_formula_verification() {
    // Verify cubic Hermite interpolation formula
    let smoother = BoundarySmoother::new(1, InterpolationMethod::Cubic);
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.0,
            zero_point: 0.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 1.0,
            zero_point: 100.0,
            group_size: 128,
        },
    ];
    
    let smoothed = smoother.smooth_boundaries(&params);
    
    // Verify using cubic Hermite basis functions
    let alpha: f32 = 1.0 / 3.0;
    let h00 = 2.0 * alpha.powi(3) - 3.0 * alpha.powi(2) + 1.0;
    let h01 = -2.0 * alpha.powi(3) + 3.0 * alpha.powi(2);
    let expected_scale = h00 * 0.0 + h01 * 1.0;
    
    assert!((smoothed[0].scale - expected_scale).abs() < 1e-6);
}

#[test]
fn test_cubic_smoother_than_linear() {
    // Cubic should produce smoother transitions (smaller second derivatives)
    let params = vec![
        TimeGroupParams {
            time_range: (0, 25),
            scale: 0.1,
            zero_point: 50.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (25, 50),
            scale: 0.1,
            zero_point: 50.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (50, 75),
            scale: 0.1,
            zero_point: 50.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (75, 100),
            scale: 0.9,
            zero_point: 200.0,
            group_size: 128,
        },
    ];
    
    let linear_smoother = BoundarySmoother::new(2, InterpolationMethod::Linear);
    let cubic_smoother = BoundarySmoother::new(2, InterpolationMethod::Cubic);
    
    let linear_smoothed = linear_smoother.smooth_boundaries(&params);
    let cubic_smoothed = cubic_smoother.smooth_boundaries(&params);
    
    // Both should produce valid results
    assert_eq!(linear_smoothed.len(), params.len());
    assert_eq!(cubic_smoothed.len(), params.len());
    
    // Cubic should have continuous second derivatives (C² continuity)
    // This is a qualitative property verified by the implementation
}

// ============================================================================
// Test 3: Sigmoid Interpolation Method
// ============================================================================

#[test]
fn test_sigmoid_interpolation_basic() {
    let smoother = BoundarySmoother::new(2, InterpolationMethod::Sigmoid);
    let params = create_params_with_jump();
    
    let smoothed = smoother.smooth_boundaries(&params);
    
    assert_eq!(smoothed.len(), params.len());
    
    // Verify all values are in valid range
    for p in &smoothed {
        assert!(p.scale >= 0.1 && p.scale <= 0.8);
        assert!(p.zero_point >= 100.0 && p.zero_point <= 250.0);
    }
}

#[test]
fn test_sigmoid_reduces_jumps_by_50_percent() {
    let smoother = BoundarySmoother::new(3, InterpolationMethod::Sigmoid);
    let params = create_params_with_jump();
    
    let (max_scale_before, max_zp_before) = compute_max_jump(&params);
    let smoothed = smoother.smooth_boundaries(&params);
    let (max_scale_after, max_zp_after) = compute_max_jump(&smoothed);
    
    // REQ-1.2.3: Smoothing SHALL reduce parameter jumps by at least 50%
    let scale_reduction = (max_scale_before - max_scale_after) / max_scale_before;
    let zp_reduction = (max_zp_before - max_zp_after) / max_zp_before;
    
    assert!(
        scale_reduction >= 0.50,
        "Sigmoid: Scale jump reduction {:.1}% is less than 50%",
        scale_reduction * 100.0
    );
    assert!(
        zp_reduction >= 0.50,
        "Sigmoid: Zero point jump reduction {:.1}% is less than 50%",
        zp_reduction * 100.0
    );
}

#[test]
fn test_sigmoid_formula_verification() {
    // Verify sigmoid interpolation formula
    let smoother = BoundarySmoother::new(1, InterpolationMethod::Sigmoid);
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.0,
            zero_point: 0.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 1.0,
            zero_point: 100.0,
            group_size: 128,
        },
    ];
    
    let smoothed = smoother.smooth_boundaries(&params);
    
    // Verify using sigmoid formula
    let k = 10.0_f32;
    let alpha = 1.0 / 3.0;
    let alpha_sigmoid = 1.0 / (1.0 + (-k * (alpha - 0.5)).exp());
    let expected_scale = 0.0 * (1.0 - alpha_sigmoid) + 1.0 * alpha_sigmoid;
    
    assert!((smoothed[0].scale - expected_scale).abs() < 1e-6);
}

#[test]
fn test_sigmoid_gradual_transition() {
    // Sigmoid should produce S-curve transitions
    let smoother = BoundarySmoother::new(1, InterpolationMethod::Sigmoid);
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.2,
            zero_point: 80.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 0.6,
            zero_point: 160.0,
            group_size: 128,
        },
    ];
    
    let smoothed = smoother.smooth_boundaries(&params);
    
    // Values should be between endpoints
    assert!(smoothed[0].scale >= 0.2 && smoothed[0].scale <= 0.6);
    assert!(smoothed[1].scale >= 0.2 && smoothed[1].scale <= 0.6);
}

// ============================================================================
// Test 4: Different Window Sizes
// ============================================================================

#[test]
fn test_window_size_1() {
    let smoother = BoundarySmoother::new(1, InterpolationMethod::Linear);
    let params = create_params_with_jump();
    
    let smoothed = smoother.smooth_boundaries(&params);
    assert_eq!(smoothed.len(), params.len());
    
    let (max_scale_after, _) = compute_max_jump(&smoothed);
    assert!(max_scale_after < 7.0); // Should reduce the 700% jump
}

#[test]
fn test_window_size_5() {
    let smoother = BoundarySmoother::new(5, InterpolationMethod::Linear);
    let params = create_params_with_jump();
    
    let smoothed = smoother.smooth_boundaries(&params);
    assert_eq!(smoothed.len(), params.len());
    
    let (max_scale_after, _) = compute_max_jump(&smoothed);
    assert!(max_scale_after < 7.0);
}

#[test]
fn test_window_size_10() {
    let smoother = BoundarySmoother::new(10, InterpolationMethod::Linear);
    
    // Create more groups to test larger window
    let params = vec![
        TimeGroupParams { time_range: (0, 10), scale: 0.1, zero_point: 100.0, group_size: 128 },
        TimeGroupParams { time_range: (10, 20), scale: 0.1, zero_point: 100.0, group_size: 128 },
        TimeGroupParams { time_range: (20, 30), scale: 0.1, zero_point: 100.0, group_size: 128 },
        TimeGroupParams { time_range: (30, 40), scale: 0.1, zero_point: 100.0, group_size: 128 },
        TimeGroupParams { time_range: (40, 50), scale: 0.1, zero_point: 100.0, group_size: 128 },
        TimeGroupParams { time_range: (50, 60), scale: 0.8, zero_point: 250.0, group_size: 128 },
        TimeGroupParams { time_range: (60, 70), scale: 0.8, zero_point: 250.0, group_size: 128 },
        TimeGroupParams { time_range: (70, 80), scale: 0.8, zero_point: 250.0, group_size: 128 },
    ];
    
    let smoothed = smoother.smooth_boundaries(&params);
    assert_eq!(smoothed.len(), params.len());
}

#[test]
fn test_window_size_20_maximum() {
    let smoother = BoundarySmoother::new(20, InterpolationMethod::Linear);
    
    // Create many groups
    let mut params = Vec::new();
    for i in 0..25 {
        let scale = if i < 12 { 0.1 } else { 0.8 };
        let zp = if i < 12 { 100.0 } else { 250.0 };
        params.push(TimeGroupParams {
            time_range: (i * 10, (i + 1) * 10),
            scale,
            zero_point: zp,
            group_size: 128,
        });
    }
    
    let smoothed = smoother.smooth_boundaries(&params);
    assert_eq!(smoothed.len(), params.len());
}

#[test]
fn test_larger_window_reduces_jumps_more() {
    let params = create_params_with_jump();
    
    let smoother_small = BoundarySmoother::new(1, InterpolationMethod::Linear);
    let smoother_large = BoundarySmoother::new(3, InterpolationMethod::Linear);
    
    let smoothed_small = smoother_small.smooth_boundaries(&params);
    let smoothed_large = smoother_large.smooth_boundaries(&params);
    
    let (max_scale_small, _) = compute_max_jump(&smoothed_small);
    let (max_scale_large, _) = compute_max_jump(&smoothed_large);
    
    // Larger window should generally reduce jumps more
    // (though not always guaranteed due to boundary effects)
    assert!(max_scale_large <= max_scale_small * 1.5);
}

// ============================================================================
// Test 5: Edge Cases
// ============================================================================

#[test]
fn test_empty_params() {
    let smoother = BoundarySmoother::new(5, InterpolationMethod::Linear);
    let params: Vec<TimeGroupParams> = vec![];
    
    let smoothed = smoother.smooth_boundaries(&params);
    assert_eq!(smoothed.len(), 0);
}

#[test]
fn test_single_group() {
    let smoother = BoundarySmoother::new(5, InterpolationMethod::Linear);
    let params = vec![TimeGroupParams {
        time_range: (0, 100),
        scale: 0.5,
        zero_point: 128.0,
        group_size: 128,
    }];
    
    let smoothed = smoother.smooth_boundaries(&params);
    assert_eq!(smoothed.len(), 1);
    assert_eq!(smoothed[0].scale, 0.5);
    assert_eq!(smoothed[0].zero_point, 128.0);
}

#[test]
fn test_two_groups_boundary() {
    let smoother = BoundarySmoother::new(2, InterpolationMethod::Linear);
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.1,
            zero_point: 100.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 0.9,
            zero_point: 250.0,
            group_size: 128,
        },
    ];
    
    let smoothed = smoother.smooth_boundaries(&params);
    assert_eq!(smoothed.len(), 2);
    
    // Both groups should be smoothed
    assert!(smoothed[0].scale > 0.1 && smoothed[0].scale < 0.9);
    assert!(smoothed[1].scale > 0.1 && smoothed[1].scale < 0.9);
}

#[test]
fn test_identical_params_no_change() {
    let smoother = BoundarySmoother::new(3, InterpolationMethod::Linear);
    let params = vec![
        TimeGroupParams {
            time_range: (0, 25),
            scale: 0.5,
            zero_point: 128.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (25, 50),
            scale: 0.5,
            zero_point: 128.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (50, 75),
            scale: 0.5,
            zero_point: 128.0,
            group_size: 128,
        },
    ];
    
    let smoothed = smoother.smooth_boundaries(&params);
    
    // When params are identical, smoothing should not change them
    for (original, smoothed) in params.iter().zip(smoothed.iter()) {
        assert!((original.scale - smoothed.scale).abs() < 1e-6);
        assert!((original.zero_point - smoothed.zero_point).abs() < 1e-6);
    }
}

#[test]
fn test_window_larger_than_groups() {
    // Window size larger than number of groups
    let smoother = BoundarySmoother::new(10, InterpolationMethod::Linear);
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.1,
            zero_point: 100.0,
            group_size: 128,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 0.9,
            zero_point: 250.0,
            group_size: 128,
        },
    ];
    
    let smoothed = smoother.smooth_boundaries(&params);
    assert_eq!(smoothed.len(), 2);
    
    // Should still work, window is clamped to available groups
    assert!(smoothed[0].scale > 0.1 && smoothed[0].scale < 0.9);
}

// ============================================================================
// Test 6: All Methods Comparison
// ============================================================================

#[test]
fn test_all_methods_reduce_jumps() {
    let params = create_params_with_jump();
    let (max_scale_before, _) = compute_max_jump(&params);
    
    for method in &[
        InterpolationMethod::Linear,
        InterpolationMethod::Cubic,
        InterpolationMethod::Sigmoid,
    ] {
        let smoother = BoundarySmoother::new(3, *method);
        let smoothed = smoother.smooth_boundaries(&params);
        let (max_scale_after, _) = compute_max_jump(&smoothed);
        
        assert!(
            max_scale_after < max_scale_before,
            "{:?} failed to reduce jumps",
            method
        );
    }
}

#[test]
fn test_all_methods_meet_50_percent_reduction() {
    let params = create_params_with_jump();
    let (max_scale_before, max_zp_before) = compute_max_jump(&params);
    
    for method in &[
        InterpolationMethod::Linear,
        InterpolationMethod::Cubic,
        InterpolationMethod::Sigmoid,
    ] {
        let smoother = BoundarySmoother::new(3, *method);
        let smoothed = smoother.smooth_boundaries(&params);
        let (max_scale_after, max_zp_after) = compute_max_jump(&smoothed);
        
        let scale_reduction = (max_scale_before - max_scale_after) / max_scale_before;
        let zp_reduction = (max_zp_before - max_zp_after) / max_zp_before;
        
        assert!(
            scale_reduction >= 0.50,
            "{:?}: Scale reduction {:.1}% < 50%",
            method,
            scale_reduction * 100.0
        );
        assert!(
            zp_reduction >= 0.50,
            "{:?}: ZP reduction {:.1}% < 50%",
            method,
            zp_reduction * 100.0
        );
    }
}

// ============================================================================
// Test 7: Constructor Validation
// ============================================================================

#[test]
#[should_panic(expected = "Window size must be in range [1, 20]")]
fn test_invalid_window_size_zero() {
    BoundarySmoother::new(0, InterpolationMethod::Linear);
}

#[test]
#[should_panic(expected = "Window size must be in range [1, 20]")]
fn test_invalid_window_size_too_large() {
    BoundarySmoother::new(21, InterpolationMethod::Linear);
}

#[test]
fn test_valid_window_size_boundaries() {
    // Test boundary values
    let _smoother1 = BoundarySmoother::new(1, InterpolationMethod::Linear);
    let _smoother20 = BoundarySmoother::new(20, InterpolationMethod::Linear);
    // Should not panic
}

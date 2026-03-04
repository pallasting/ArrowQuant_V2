//! Unit tests for TimeGroupBoundaries structure
//!
//! Tests the pre-computation and binary search functionality of time group boundaries.

use arrow_quant_v2::time_aware::{TimeGroupBoundaries, TimeGroupParams};

#[test]
fn test_precompute_boundaries_basic() {
    // Create 3 time groups with different scales
    let params = vec![
        TimeGroupParams {
            time_range: (0, 33),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (33, 66),
            scale: 0.15,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (66, 100),
            scale: 0.2,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    let boundaries = TimeGroupBoundaries::precompute_boundaries(&params);

    // Verify number of groups
    assert_eq!(boundaries.num_groups(), 3);

    // Verify boundaries are sorted
    assert!(boundaries.is_sorted());

    // Verify boundaries are computed correctly
    // Upper bound = (255.0 - zero_point) * scale
    let expected_boundaries = vec![
        (255.0 - 0.0) * 0.1,  // 25.5
        (255.0 - 0.0) * 0.15, // 38.25
        (255.0 - 0.0) * 0.2,  // 51.0
    ];

    let actual_boundaries = boundaries.boundaries();
    assert_eq!(actual_boundaries.len(), 3);

    // Check that boundaries match expected values (sorted)
    let mut sorted_expected = expected_boundaries.clone();
    sorted_expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for (i, &expected) in sorted_expected.iter().enumerate() {
        assert!(
            (actual_boundaries[i] - expected).abs() < 1e-6,
            "Boundary {} mismatch: expected {}, got {}",
            i,
            expected,
            actual_boundaries[i]
        );
    }
}

#[test]
fn test_precompute_boundaries_with_zero_point() {
    // Create time groups with non-zero zero_points
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.1,
            zero_point: 10.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 0.2,
            zero_point: 20.0,
            group_size: 64,
        },
    ];

    let boundaries = TimeGroupBoundaries::precompute_boundaries(&params);

    assert_eq!(boundaries.num_groups(), 2);
    assert!(boundaries.is_sorted());

    // Upper bound = (255.0 - zero_point) * scale
    let expected_boundaries = vec![
        (255.0 - 10.0) * 0.1, // 24.5
        (255.0 - 20.0) * 0.2, // 47.0
    ];

    let actual_boundaries = boundaries.boundaries();
    let mut sorted_expected = expected_boundaries.clone();
    sorted_expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for (i, &expected) in sorted_expected.iter().enumerate() {
        assert!(
            (actual_boundaries[i] - expected).abs() < 1e-6,
            "Boundary {} mismatch: expected {}, got {}",
            i,
            expected,
            actual_boundaries[i]
        );
    }
}

#[test]
fn test_find_group_basic() {
    let params = vec![
        TimeGroupParams {
            time_range: (0, 33),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (33, 66),
            scale: 0.2,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (66, 100),
            scale: 0.3,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    let boundaries = TimeGroupBoundaries::precompute_boundaries(&params);

    // Test values in different ranges
    // Boundaries are: [25.5, 51.0, 76.5] (sorted)

    // Value less than first boundary
    let group_id = boundaries.find_group(10.0);
    assert!(group_id < 3, "Group ID should be valid");

    // Value between boundaries
    let group_id = boundaries.find_group(40.0);
    assert!(group_id < 3, "Group ID should be valid");

    // Value greater than last boundary
    let group_id = boundaries.find_group(100.0);
    assert!(group_id < 3, "Group ID should be valid");
}

#[test]
fn test_find_group_edge_cases() {
    let params = vec![
        TimeGroupParams {
            time_range: (0, 50),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (50, 100),
            scale: 0.2,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    let boundaries = TimeGroupBoundaries::precompute_boundaries(&params);

    // Test exact boundary values
    let boundary_value = boundaries.boundaries()[0];
    let group_id = boundaries.find_group(boundary_value);
    assert!(group_id < 2, "Group ID should be valid");

    // Test negative values
    let group_id = boundaries.find_group(-10.0);
    assert!(group_id < 2, "Group ID should be valid");

    // Test very large values
    let group_id = boundaries.find_group(1000.0);
    assert_eq!(group_id, 1, "Large values should map to last group");
}

#[test]
fn test_is_sorted() {
    let params = vec![
        TimeGroupParams {
            time_range: (0, 33),
            scale: 0.3,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (33, 66),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (66, 100),
            scale: 0.2,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    let boundaries = TimeGroupBoundaries::precompute_boundaries(&params);

    // Boundaries should always be sorted after precompute_boundaries
    assert!(boundaries.is_sorted());
}

#[test]
fn test_single_group() {
    let params = vec![TimeGroupParams {
        time_range: (0, 100),
        scale: 0.1,
        zero_point: 0.0,
        group_size: 64,
    }];

    let boundaries = TimeGroupBoundaries::precompute_boundaries(&params);

    assert_eq!(boundaries.num_groups(), 1);
    assert!(boundaries.is_sorted());

    // All values should map to group 0
    assert_eq!(boundaries.find_group(0.0), 0);
    assert_eq!(boundaries.find_group(10.0), 0);
    assert_eq!(boundaries.find_group(100.0), 0);
}

#[test]
fn test_boundaries_coverage() {
    // Test that boundaries cover all time groups
    let params = vec![
        TimeGroupParams {
            time_range: (0, 25),
            scale: 0.1,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (25, 50),
            scale: 0.15,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (50, 75),
            scale: 0.2,
            zero_point: 0.0,
            group_size: 64,
        },
        TimeGroupParams {
            time_range: (75, 100),
            scale: 0.25,
            zero_point: 0.0,
            group_size: 64,
        },
    ];

    let boundaries = TimeGroupBoundaries::precompute_boundaries(&params);

    assert_eq!(boundaries.num_groups(), 4);
    assert_eq!(boundaries.boundaries().len(), 4);
    assert!(boundaries.is_sorted());

    // Verify all groups are accessible
    for i in 0..4 {
        let test_value = (i as f32) * 10.0;
        let group_id = boundaries.find_group(test_value);
        assert!(
            group_id < 4,
            "Group ID {} should be less than 4 for value {}",
            group_id,
            test_value
        );
    }
}

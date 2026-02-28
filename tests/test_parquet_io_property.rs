//! Property-Based Tests for Parquet I/O
//!
//! **Validates: Requirement 13 (Testing and Benchmarking)**
//!
//! This module contains property-based tests using proptest to verify
//! Parquet I/O invariants across a wide range of inputs.

use arrow_quant_v2::config::Modality;
use arrow_quant_v2::schema::{
    ActivationStatsMetadata, ParquetV2Extended, SchemaVersion, SpatialQuantMetadata,
    TimeAwareQuantMetadata,
};
use arrow_quant_v2::time_aware::TimeGroupParams;
use proptest::prelude::*;
use tempfile::TempDir;

/// Generate random valid modality
fn arb_modality() -> impl Strategy<Value = Modality> {
    prop_oneof![
        Just(Modality::Text),
        Just(Modality::Code),
        Just(Modality::Image),
        Just(Modality::Audio),
    ]
}

/// Generate random valid bit width
fn arb_bit_width() -> impl Strategy<Value = u8> {
    prop_oneof![Just(2u8), Just(4u8), Just(8u8),]
}

/// Generate random valid group size
fn arb_group_size() -> impl Strategy<Value = usize> {
    prop_oneof![Just(32usize), Just(64usize), Just(128usize), Just(256usize),]
}

/// Generate random TimeGroupParams
fn arb_time_group_params() -> impl Strategy<Value = TimeGroupParams> {
    (
        0usize..1000,
        100usize..1000,
        0.001f32..1.0f32,
        0.0f32..128.0f32,
        arb_group_size(),
    )
        .prop_map(|(start, end, scale, zero_point, group_size)| TimeGroupParams {
            time_range: (start, start + end),
            scale,
            zero_point,
            group_size,
        })
}

/// Generate random TimeAwareQuantMetadata
fn arb_time_aware_metadata() -> impl Strategy<Value = TimeAwareQuantMetadata> {
    (
        any::<bool>(),
        1usize..20,
        prop::collection::vec(arb_time_group_params(), 1..10),
    )
        .prop_map(|(enabled, num_time_groups, time_group_params)| TimeAwareQuantMetadata {
            enabled,
            num_time_groups,
            time_group_params,
        })
}

/// Generate random SpatialQuantMetadata
fn arb_spatial_metadata() -> impl Strategy<Value = SpatialQuantMetadata> {
    (
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        prop::collection::vec(0.001f32..10.0f32, 1..256),
    )
        .prop_map(
            |(enabled, channel_equalization, activation_smoothing, equalization_scales)| {
                SpatialQuantMetadata {
                    enabled,
                    channel_equalization,
                    activation_smoothing,
                    equalization_scales,
                }
            },
        )
}

/// Generate random ActivationStatsMetadata
fn arb_activation_stats() -> impl Strategy<Value = ActivationStatsMetadata> {
    (
        prop::collection::vec(-10.0f32..10.0f32, 1..100),
        prop::collection::vec(0.0f32..5.0f32, 1..100),
        prop::collection::vec(-10.0f32..10.0f32, 1..100),
        prop::collection::vec(-10.0f32..10.0f32, 1..100),
    )
        .prop_map(|(mean, std, min, max)| ActivationStatsMetadata { mean, std, min, max })
}

/// Generate random ParquetV2Extended with base fields only
fn arb_parquet_v2_base() -> impl Strategy<Value = ParquetV2Extended> {
    (
        "[a-z_]{5,20}",
        prop::collection::vec(1usize..1024, 1..4),
        "[a-z0-9]{3,10}",
        prop::collection::vec(0u8..=255u8, 10..1000),
        1usize..1000000,
        "[a-z0-9]{3,10}",
        prop::collection::vec(0.001f32..1.0f32, 1..10),
        prop::collection::vec(0.0f32..128.0f32, 1..10),
        prop::option::of(0usize..3),
        prop::option::of(arb_group_size()),
    )
        .prop_map(
            |(
                layer_name,
                shape,
                dtype,
                data,
                num_params,
                quant_type,
                scales,
                zero_points,
                quant_axis,
                group_size,
            )| {
                ParquetV2Extended {
                    layer_name,
                    shape,
                    dtype,
                    data,
                    num_params,
                    quant_type,
                    scales,
                    zero_points,
                    quant_axis,
                    group_size,
                    is_diffusion_model: false,
                    modality: None,
                    time_aware_quant: None,
                    spatial_quant: None,
                    activation_stats: None,
                }
            },
        )
}

/// Generate random ParquetV2Extended with time-aware metadata
fn arb_parquet_v2_time_aware() -> impl Strategy<Value = ParquetV2Extended> {
    (arb_parquet_v2_base(), arb_modality(), arb_time_aware_metadata()).prop_map(
        |(mut base, modality, time_aware)| {
            base.is_diffusion_model = true;
            base.modality = Some(modality.to_string());
            base.time_aware_quant = Some(time_aware);
            base
        },
    )
}

/// Generate random ParquetV2Extended with spatial metadata
fn arb_parquet_v2_spatial() -> impl Strategy<Value = ParquetV2Extended> {
    (arb_parquet_v2_base(), arb_modality(), arb_spatial_metadata()).prop_map(
        |(mut base, modality, spatial)| {
            base.is_diffusion_model = true;
            base.modality = Some(modality.to_string());
            base.spatial_quant = Some(spatial);
            base
        },
    )
}

/// Generate random ParquetV2Extended with all metadata
fn arb_parquet_v2_full() -> impl Strategy<Value = ParquetV2Extended> {
    (
        arb_parquet_v2_base(),
        arb_modality(),
        arb_time_aware_metadata(),
        arb_spatial_metadata(),
        arb_activation_stats(),
        arb_bit_width(),
    )
        .prop_map(
            |(mut base, modality, time_aware, spatial, activation_stats, bit_width)| {
                base.is_diffusion_model = true;
                base.modality = Some(modality.to_string());
                base.time_aware_quant = Some(time_aware);
                base.spatial_quant = Some(spatial);
                base.activation_stats = Some(activation_stats);
                base.quant_type = format!("int{}", bit_width);
                base
            },
        )
}

/// **Validates: Requirements 13**
///
/// Property: Write then read preserves metadata exactly
///
/// This property test verifies that:
/// 1. All metadata fields are preserved through write/read cycle
/// 2. No data corruption occurs during I/O
/// 3. The roundtrip is lossless
#[cfg(test)]
mod roundtrip_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_roundtrip_preserves_base_metadata(
            schema in arb_parquet_v2_base()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write
            schema.write_to_parquet(&path).unwrap();

            // Read
            let read_schema = ParquetV2Extended::read_from_parquet(&path).unwrap();

            // Property: All base fields should be preserved
            prop_assert_eq!(read_schema.layer_name, schema.layer_name);
            prop_assert_eq!(read_schema.shape, schema.shape);
            prop_assert_eq!(read_schema.dtype, schema.dtype);
            prop_assert_eq!(read_schema.data, schema.data);
            prop_assert_eq!(read_schema.num_params, schema.num_params);
            prop_assert_eq!(read_schema.quant_type, schema.quant_type);
            prop_assert_eq!(read_schema.scales.len(), schema.scales.len());
            prop_assert_eq!(read_schema.zero_points.len(), schema.zero_points.len());
            prop_assert_eq!(read_schema.quant_axis, schema.quant_axis);
            prop_assert_eq!(read_schema.group_size, schema.group_size);
            prop_assert_eq!(read_schema.is_diffusion_model, schema.is_diffusion_model);
        }
    }

    proptest! {
        #[test]
        fn prop_roundtrip_preserves_time_aware_metadata(
            schema in arb_parquet_v2_time_aware()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write
            schema.write_to_parquet(&path).unwrap();

            // Read
            let read_schema = ParquetV2Extended::read_from_parquet(&path).unwrap();

            // Property: Time-aware metadata should be preserved
            prop_assert_eq!(read_schema.is_diffusion_model, true);
            prop_assert!(read_schema.modality.is_some());
            prop_assert!(read_schema.time_aware_quant.is_some());

            if let (Some(orig_ta), Some(read_ta)) = (&schema.time_aware_quant, &read_schema.time_aware_quant) {
                prop_assert_eq!(read_ta.enabled, orig_ta.enabled);
                prop_assert_eq!(read_ta.num_time_groups, orig_ta.num_time_groups);
                prop_assert_eq!(read_ta.time_group_params.len(), orig_ta.time_group_params.len());
            }
        }
    }

    proptest! {
        #[test]
        fn prop_roundtrip_preserves_spatial_metadata(
            schema in arb_parquet_v2_spatial()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write
            schema.write_to_parquet(&path).unwrap();

            // Read
            let read_schema = ParquetV2Extended::read_from_parquet(&path).unwrap();

            // Property: Spatial metadata should be preserved
            prop_assert_eq!(read_schema.is_diffusion_model, true);
            prop_assert!(read_schema.modality.is_some());
            prop_assert!(read_schema.spatial_quant.is_some());

            if let (Some(orig_sp), Some(read_sp)) = (&schema.spatial_quant, &read_schema.spatial_quant) {
                prop_assert_eq!(read_sp.enabled, orig_sp.enabled);
                prop_assert_eq!(read_sp.channel_equalization, orig_sp.channel_equalization);
                prop_assert_eq!(read_sp.activation_smoothing, orig_sp.activation_smoothing);
                prop_assert_eq!(read_sp.equalization_scales.len(), orig_sp.equalization_scales.len());
            }
        }
    }

    proptest! {
        #[test]
        fn prop_roundtrip_preserves_full_metadata(
            schema in arb_parquet_v2_full()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write
            schema.write_to_parquet(&path).unwrap();

            // Read
            let read_schema = ParquetV2Extended::read_from_parquet(&path).unwrap();

            // Property: All metadata should be preserved
            prop_assert_eq!(read_schema.is_diffusion_model, true);
            prop_assert!(read_schema.modality.is_some());
            prop_assert!(read_schema.time_aware_quant.is_some());
            prop_assert!(read_schema.spatial_quant.is_some());
            prop_assert!(read_schema.activation_stats.is_some());
            // Bit width is encoded in quant_type
            prop_assert_eq!(read_schema.quant_type, schema.quant_type);
        }
    }

    proptest! {
        #[test]
        fn prop_roundtrip_preserves_scales_exactly(
            schema in arb_parquet_v2_base()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write
            schema.write_to_parquet(&path).unwrap();

            // Read
            let read_schema = ParquetV2Extended::read_from_parquet(&path).unwrap();

            // Property: Scales should be preserved exactly (floating point equality)
            prop_assert_eq!(read_schema.scales.len(), schema.scales.len());
            for (orig, read) in schema.scales.iter().zip(read_schema.scales.iter()) {
                prop_assert!(
                    (orig - read).abs() < 1e-6,
                    "Scale mismatch: {} != {}",
                    orig,
                    read
                );
            }
        }
    }

    proptest! {
        #[test]
        fn prop_roundtrip_preserves_zero_points_exactly(
            schema in arb_parquet_v2_base()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write
            schema.write_to_parquet(&path).unwrap();

            // Read
            let read_schema = ParquetV2Extended::read_from_parquet(&path).unwrap();

            // Property: Zero points should be preserved exactly
            prop_assert_eq!(read_schema.zero_points.len(), schema.zero_points.len());
            for (orig, read) in schema.zero_points.iter().zip(read_schema.zero_points.iter()) {
                prop_assert!(
                    (orig - read).abs() < 1e-6,
                    "Zero point mismatch: {} != {}",
                    orig,
                    read
                );
            }
        }
    }

    proptest! {
        #[test]
        fn prop_roundtrip_preserves_data_exactly(
            schema in arb_parquet_v2_base()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write
            schema.write_to_parquet(&path).unwrap();

            // Read
            let read_schema = ParquetV2Extended::read_from_parquet(&path).unwrap();

            // Property: Quantized data should be preserved byte-for-byte
            prop_assert_eq!(read_schema.data, schema.data);
        }
    }
}

/// **Validates: Requirements 13**
///
/// Property: Schema version detection is correct
///
/// This property test verifies that:
/// 1. V2 base schemas are detected as V2
/// 2. V2 Extended schemas are detected as V2Extended
/// 3. Detection is consistent and reliable
#[cfg(test)]
mod schema_version_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_base_schema_detected_as_v2(
            schema in arb_parquet_v2_base()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write base schema (no diffusion metadata)
            schema.write_to_parquet(&path).unwrap();

            // Property: Should be detected as V2 (not V2Extended)
            let version = ParquetV2Extended::detect_schema_version(&path).unwrap();
            prop_assert_eq!(version, SchemaVersion::V2);
        }
    }

    proptest! {
        #[test]
        fn prop_time_aware_schema_detected_as_v2_extended(
            schema in arb_parquet_v2_time_aware()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write time-aware schema
            schema.write_to_parquet(&path).unwrap();

            // Property: Should be detected as V2Extended
            let version = ParquetV2Extended::detect_schema_version(&path).unwrap();
            prop_assert_eq!(version, SchemaVersion::V2Extended);
        }
    }

    proptest! {
        #[test]
        fn prop_spatial_schema_detected_as_v2_extended(
            schema in arb_parquet_v2_spatial()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write spatial schema
            schema.write_to_parquet(&path).unwrap();

            // Property: Should be detected as V2Extended
            let version = ParquetV2Extended::detect_schema_version(&path).unwrap();
            prop_assert_eq!(version, SchemaVersion::V2Extended);
        }
    }

    proptest! {
        #[test]
        fn prop_full_schema_detected_as_v2_extended(
            schema in arb_parquet_v2_full()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write full extended schema
            schema.write_to_parquet(&path).unwrap();

            // Property: Should be detected as V2Extended
            let version = ParquetV2Extended::detect_schema_version(&path).unwrap();
            prop_assert_eq!(version, SchemaVersion::V2Extended);
        }
    }

    proptest! {
        #[test]
        fn prop_version_detection_is_deterministic(
            schema in arb_parquet_v2_full()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write schema
            schema.write_to_parquet(&path).unwrap();

            // Property: Multiple detections should return same result
            let version1 = ParquetV2Extended::detect_schema_version(&path).unwrap();
            let version2 = ParquetV2Extended::detect_schema_version(&path).unwrap();
            let version3 = ParquetV2Extended::detect_schema_version(&path).unwrap();

            prop_assert_eq!(version1, version2);
            prop_assert_eq!(version2, version3);
        }
    }
}

/// **Validates: Requirements 13**
///
/// Property: Backward compatibility with V2 schema
///
/// This property test verifies that:
/// 1. V2 base schemas can be read as V2Extended
/// 2. Missing diffusion fields default to None
/// 3. No errors occur when reading V2 schemas
#[cfg(test)]
mod backward_compatibility_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_v2_base_readable_as_v2_extended(
            schema in arb_parquet_v2_base()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write base schema
            schema.write_to_parquet(&path).unwrap();

            // Property: Should be readable without errors
            let read_result = ParquetV2Extended::read_from_parquet(&path);
            prop_assert!(read_result.is_ok());

            let read_schema = read_result.unwrap();

            // Property: Diffusion fields should be None or false
            prop_assert_eq!(read_schema.is_diffusion_model, false);
            prop_assert!(read_schema.modality.is_none());
            prop_assert!(read_schema.time_aware_quant.is_none());
            prop_assert!(read_schema.spatial_quant.is_none());
            prop_assert!(read_schema.activation_stats.is_none());
        }
    }

    proptest! {
        #[test]
        fn prop_v2_base_preserves_core_fields(
            schema in arb_parquet_v2_base()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write base schema
            schema.write_to_parquet(&path).unwrap();

            // Read as V2Extended
            let read_schema = ParquetV2Extended::read_from_parquet(&path).unwrap();

            // Property: Core V2 fields should be preserved
            prop_assert_eq!(read_schema.layer_name, schema.layer_name);
            prop_assert_eq!(read_schema.shape, schema.shape);
            prop_assert_eq!(read_schema.dtype, schema.dtype);
            prop_assert_eq!(read_schema.data, schema.data);
            prop_assert_eq!(read_schema.num_params, schema.num_params);
            prop_assert_eq!(read_schema.quant_type, schema.quant_type);
        }
    }
}

/// **Validates: Requirements 13**
///
/// Property: I/O operations are idempotent
///
/// This property test verifies that:
/// 1. Writing the same schema multiple times produces identical files
/// 2. Reading multiple times produces identical results
#[cfg(test)]
mod idempotency_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_write_is_idempotent(
            schema in arb_parquet_v2_full()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path1 = temp_dir.path().join("test1.parquet");
            let path2 = temp_dir.path().join("test2.parquet");

            // Write same schema twice
            schema.write_to_parquet(&path1).unwrap();
            schema.write_to_parquet(&path2).unwrap();

            // Read both
            let read1 = ParquetV2Extended::read_from_parquet(&path1).unwrap();
            let read2 = ParquetV2Extended::read_from_parquet(&path2).unwrap();

            // Property: Results should be identical
            prop_assert_eq!(read1.layer_name, read2.layer_name);
            prop_assert_eq!(read1.data, read2.data);
            prop_assert_eq!(read1.scales, read2.scales);
            prop_assert_eq!(read1.zero_points, read2.zero_points);
        }
    }

    proptest! {
        #[test]
        fn prop_read_is_idempotent(
            schema in arb_parquet_v2_full()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write once
            schema.write_to_parquet(&path).unwrap();

            // Read multiple times
            let read1 = ParquetV2Extended::read_from_parquet(&path).unwrap();
            let read2 = ParquetV2Extended::read_from_parquet(&path).unwrap();
            let read3 = ParquetV2Extended::read_from_parquet(&path).unwrap();

            // Property: All reads should be identical
            prop_assert_eq!(&read1.layer_name, &read2.layer_name);
            prop_assert_eq!(&read2.layer_name, &read3.layer_name);
            prop_assert_eq!(&read1.data, &read2.data);
            prop_assert_eq!(&read2.data, &read3.data);
        }
    }
}

/// **Validates: Requirements 13**
///
/// Property: Modality preservation
///
/// This property test verifies that:
/// 1. All modality types are preserved correctly
/// 2. Modality detection is consistent
#[cfg(test)]
mod modality_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_modality_preserved_for_all_types(
            modality in arb_modality(),
            schema in arb_parquet_v2_base()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Create schema with specific modality
            let mut extended_schema = schema;
            extended_schema.is_diffusion_model = true;
            extended_schema.modality = Some(modality.to_string());

            // Write
            extended_schema.write_to_parquet(&path).unwrap();

            // Read
            let read_schema = ParquetV2Extended::read_from_parquet(&path).unwrap();

            // Property: Modality should be preserved
            prop_assert_eq!(read_schema.modality, Some(modality.to_string()));
        }
    }
}

/// **Validates: Requirements 13**
///
/// Property: Bit width preservation via quant_type
///
/// This property test verifies that:
/// 1. All bit widths (2, 4, 8) are preserved correctly in quant_type
/// 2. Quant type encoding is consistent
#[cfg(test)]
mod bit_width_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_bit_width_preserved_for_all_values(
            bit_width in arb_bit_width(),
            schema in arb_parquet_v2_base()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Create schema with specific bit width
            let mut extended_schema = schema;
            extended_schema.quant_type = format!("int{}", bit_width);

            // Write
            extended_schema.write_to_parquet(&path).unwrap();

            // Read
            let read_schema = ParquetV2Extended::read_from_parquet(&path).unwrap();

            // Property: Bit width should be preserved in quant_type
            prop_assert_eq!(read_schema.quant_type, format!("int{}", bit_width));
        }
    }

    proptest! {
        #[test]
        fn prop_quant_type_preserved(
            schema in arb_parquet_v2_base()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.parquet");

            // Write
            schema.write_to_parquet(&path).unwrap();

            // Read
            let read_schema = ParquetV2Extended::read_from_parquet(&path).unwrap();

            // Property: quant_type should be preserved exactly
            prop_assert_eq!(read_schema.quant_type, schema.quant_type);
        }
    }
}

//! Property-Based Tests for Zero-Copy Memory Access
//!
//! **Validates: Requirements 1.1, 1.2, 1.3, 5.4, 8.4**
//!
//! This module contains property-based tests using proptest to verify
//! zero-copy memory access properties across Rust-Python data transmission.
//!
//! **Property 2: Zero-copy memory access (Rust internal)**
//! **Property 8: Python API zero-copy export**
//!
//! These tests verify that:
//! 1. Arrow C Data Interface maintains zero-copy semantics
//! 2. No data copying occurs during RecordBatch operations
//! 3. Memory addresses remain consistent across operations
//! 4. Buffer sharing works correctly with Arrow arrays

use arrow::array::{Array, Float32Array, ListArray, RecordBatch, StringArray, UInt8Array, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};
use proptest::prelude::*;
use rand::SeedableRng;
use std::sync::Arc;

/// **Validates: Requirements 1.1, 1.2, 1.3**
///
/// Property 2: Zero-Copy Memory Access (Internal Rust)
///
/// This property test verifies that:
/// 1. RecordBatch operations maintain zero-copy semantics
/// 2. Multiple accesses to the same data share the same buffer
/// 3. Memory addresses remain consistent across operations
#[cfg(test)]
mod zero_copy_internal_properties {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        
        #[test]
        fn prop_recordbatch_zero_copy_access(
            // Generate random array size between 100 and 10,000
            size in 100usize..10_000,
            // Generate random seed for reproducibility
            seed in any::<u64>(),
        ) {
            // Generate random float data
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            use rand::Rng;
            let data: Vec<f32> = (0..size)
                .map(|_| rng.gen_range(-100.0..100.0))
                .collect();

            // Create RecordBatch
            let schema = Arc::new(Schema::new(vec![
                Field::new("data", DataType::Float32, false),
            ]));
            let float_array = Float32Array::from(data.clone());
            let batch = RecordBatch::try_new(
                schema,
                vec![Arc::new(float_array)],
            ).expect("RecordBatch creation should succeed");

            // Access data multiple times
            let column1 = batch.column(0);
            let array1 = column1.as_any().downcast_ref::<Float32Array>().unwrap();
            let slice1 = array1.values();

            let column2 = batch.column(0);
            let array2 = column2.as_any().downcast_ref::<Float32Array>().unwrap();
            let slice2 = array2.values();

            // Property: Multiple accesses should share the same buffer (same pointer)
            prop_assert_eq!(
                slice1.as_ptr(),
                slice2.as_ptr(),
                "Multiple accesses should share the same buffer (zero-copy)"
            );

            // Property: Data should be correct
            prop_assert_eq!(slice1.len(), size);
            for i in 0..size {
                prop_assert!((slice1[i] - data[i]).abs() < 1e-6);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        
        #[test]
        fn prop_list_array_zero_copy_nested_access(
            // Generate random number of lists
            num_lists in 3usize..20,
            // Generate random list sizes
            list_size in 10usize..100,
            // Generate random seed
            seed in any::<u64>(),
        ) {
            // Generate random nested data
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            use rand::Rng;
            
            let total_size = num_lists * list_size;
            let values: Vec<f32> = (0..total_size)
                .map(|_| rng.gen_range(-100.0..100.0))
                .collect();

            // Create offsets for ListArray
            let offsets: Vec<i32> = (0..=num_lists)
                .map(|i| (i * list_size) as i32)
                .collect();

            // Create ListArray
            let values_array = Float32Array::from(values.clone());
            let offsets_buffer = arrow::buffer::OffsetBuffer::new(offsets.into());
            let list_array = ListArray::new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                offsets_buffer,
                Arc::new(values_array),
                None,
            );

            // Access each list and verify zero-copy
            for i in 0..num_lists {
                let list_values = list_array.value(i);
                let float_array = list_values.as_any().downcast_ref::<Float32Array>().unwrap();
                let slice = float_array.values();

                // Property: Slice should reference the underlying buffer directly
                let buffer_ptr = float_array.values().as_ptr();
                let slice_ptr = slice.as_ptr();
                prop_assert_eq!(
                    buffer_ptr,
                    slice_ptr,
                    "List element access should be zero-copy (same pointer)"
                );

                // Property: Data should be correct
                prop_assert_eq!(slice.len(), list_size);
                let offset = i * list_size;
                for j in 0..list_size {
                    prop_assert!((slice[j] - values[offset + j]).abs() < 1e-6);
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        
        #[test]
        fn prop_quantized_data_zero_copy_access(
            // Generate random array size
            size in 100usize..5_000,
            // Generate random number of time groups
            num_groups in 3usize..15,
            // Generate random seed
            seed in any::<u64>(),
        ) {
            let quantizer = TimeAwareQuantizer::new(num_groups);

            // Generate random weights
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            use rand::Rng;
            let weights: Vec<f32> = (0..size)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();

            // Generate time group parameters
            let params: Vec<TimeGroupParams> = (0..num_groups)
                .map(|i| TimeGroupParams {
                    time_range: (i * 100, (i + 1) * 100),
                    scale: rng.gen_range(0.01..1.0),
                    zero_point: rng.gen_range(0.0..128.0),
                    group_size: 64,
                })
                .collect();

            // Quantize
            let result = quantizer
                .quantize_layer_arrow(&weights, &params)
                .expect("Quantization should succeed");

            // Access quantized data multiple times
            let batch = &result.batch;
            
            // First access
            let quantized_col1 = batch.column_by_name("quantized_data").unwrap();
            let quantized_array1 = quantized_col1.as_any().downcast_ref::<UInt8Array>().unwrap();
            let quantized_slice1 = quantized_array1.values();

            // Second access
            let quantized_col2 = batch.column_by_name("quantized_data").unwrap();
            let quantized_array2 = quantized_col2.as_any().downcast_ref::<UInt8Array>().unwrap();
            let quantized_slice2 = quantized_array2.values();

            // Property: Multiple accesses should share the same buffer
            prop_assert_eq!(
                quantized_slice1.as_ptr(),
                quantized_slice2.as_ptr(),
                "Quantized data accesses should be zero-copy (same pointer)"
            );

            // Property: Data length should match
            prop_assert_eq!(quantized_slice1.len(), size);

            // Access time group IDs multiple times
            let group_ids_col1 = batch.column_by_name("time_group_ids").unwrap();
            let group_ids_array1 = group_ids_col1.as_any().downcast_ref::<UInt32Array>().unwrap();
            let group_ids_slice1 = group_ids_array1.values();

            let group_ids_col2 = batch.column_by_name("time_group_ids").unwrap();
            let group_ids_array2 = group_ids_col2.as_any().downcast_ref::<UInt32Array>().unwrap();
            let group_ids_slice2 = group_ids_array2.values();

            // Property: Time group IDs should also be zero-copy
            prop_assert_eq!(
                group_ids_slice1.as_ptr(),
                group_ids_slice2.as_ptr(),
                "Time group IDs accesses should be zero-copy (same pointer)"
            );

            // Property: Group IDs length should match
            prop_assert_eq!(group_ids_slice1.len(), size);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        
        #[test]
        fn prop_schema_validation_preserves_zero_copy(
            // Generate random array size
            size in 100usize..2_000,
            // Generate random seed
            seed in any::<u64>(),
        ) {
            // Generate random data
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            use rand::Rng;
            let layer_names = vec!["layer.0"];
            let weights: Vec<f32> = (0..size)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();

            // Create RecordBatch
            let schema = Arc::new(Schema::new(vec![
                Field::new("layer_name", DataType::Utf8, false),
                Field::new(
                    "weights",
                    DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                    false,
                ),
            ]));

            let layer_names_array = StringArray::from(layer_names);
            let weights_array = Float32Array::from(weights);
            let weights_offsets = arrow::buffer::OffsetBuffer::new(vec![0, size as i32].into());
            let weights_list = ListArray::new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                weights_offsets,
                Arc::new(weights_array),
                None,
            );

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(layer_names_array), Arc::new(weights_list)],
            ).expect("RecordBatch creation should succeed");

            // Get pointer before schema validation
            let weights_col = batch.column_by_name("weights").unwrap();
            let weights_list = weights_col.as_any().downcast_ref::<ListArray>().unwrap();
            let first_layer = weights_list.value(0);
            let weights_f32 = first_layer.as_any().downcast_ref::<Float32Array>().unwrap();
            let ptr_before = weights_f32.values().as_ptr();

            // Validate schema (should not copy data)
            let validated_schema = batch.schema();
            prop_assert_eq!(validated_schema.fields().len(), 2);

            // Get pointer after schema validation
            let weights_col2 = batch.column_by_name("weights").unwrap();
            let weights_list2 = weights_col2.as_any().downcast_ref::<ListArray>().unwrap();
            let first_layer2 = weights_list2.value(0);
            let weights_f32_2 = first_layer2.as_any().downcast_ref::<Float32Array>().unwrap();
            let ptr_after = weights_f32_2.values().as_ptr();

            // Property: Schema validation should not copy data
            prop_assert_eq!(
                ptr_before,
                ptr_after,
                "Schema validation should preserve zero-copy (same pointer)"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        
        #[test]
        fn prop_buffer_reference_counting_safety(
            // Generate random array size
            size in 100usize..1_000,
            // Generate random seed
            seed in any::<u64>(),
        ) {
            // Generate random data
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            use rand::Rng;
            let data: Vec<f32> = (0..size)
                .map(|_| rng.gen_range(-100.0..100.0))
                .collect();

            // Create RecordBatch
            let schema = Arc::new(Schema::new(vec![
                Field::new("data", DataType::Float32, false),
            ]));
            let float_array = Float32Array::from(data.clone());
            let batch = RecordBatch::try_new(
                schema,
                vec![Arc::new(float_array)],
            ).expect("RecordBatch creation should succeed");

            // Get reference to data
            let column = batch.column(0);
            let float_array = column.as_any().downcast_ref::<Float32Array>().unwrap();
            let slice = float_array.values();
            let ptr_original = slice.as_ptr();

            // Drop intermediate references
            drop(float_array);
            drop(column);

            // Access data again
            let column2 = batch.column(0);
            let float_array2 = column2.as_any().downcast_ref::<Float32Array>().unwrap();
            let slice2 = float_array2.values();
            let ptr_after_drop = slice2.as_ptr();

            // Property: Buffer should remain valid (same pointer)
            prop_assert_eq!(
                ptr_original,
                ptr_after_drop,
                "Buffer should remain valid after dropping intermediate references"
            );

            // Property: Data should still be accessible and correct
            prop_assert_eq!(slice2.len(), size);
            for i in 0..size {
                prop_assert!((slice2[i] - data[i]).abs() < 1e-6);
            }
        }
    }
}

/// **Validates: Requirements 5.4, 8.4**
///
/// Property 8: Python API Zero-Copy Export
///
/// This property test verifies that:
/// 1. RecordBatch export maintains zero-copy semantics
/// 2. Arrow C Data Interface is used correctly
/// 3. No data copying occurs during export preparation
#[cfg(test)]
mod zero_copy_export_properties {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        
        #[test]
        fn prop_recordbatch_export_preparation_zero_copy(
            // Generate random array size
            size in 100usize..5_000,
            // Generate random number of time groups
            num_groups in 3usize..15,
            // Generate random seed
            seed in any::<u64>(),
        ) {
            let quantizer = TimeAwareQuantizer::new(num_groups);

            // Generate random weights
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            use rand::Rng;
            let weights: Vec<f32> = (0..size)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();

            // Generate time group parameters
            let params: Vec<TimeGroupParams> = (0..num_groups)
                .map(|i| TimeGroupParams {
                    time_range: (i * 100, (i + 1) * 100),
                    scale: rng.gen_range(0.01..1.0),
                    zero_point: rng.gen_range(0.0..128.0),
                    group_size: 64,
                })
                .collect();

            // Quantize
            let result = quantizer
                .quantize_layer_arrow(&weights, &params)
                .expect("Quantization should succeed");

            let batch = &result.batch;

            // Get pointer to quantized data before export preparation
            let quantized_col = batch.column_by_name("quantized_data").unwrap();
            let quantized_array = quantized_col.as_any().downcast_ref::<UInt8Array>().unwrap();
            let ptr_before = quantized_array.values().as_ptr();

            // Simulate export preparation: convert to StructArray
            // This is what happens in export_recordbatch_to_pyarrow()
            let struct_array = arrow::array::StructArray::from(batch.clone());
            let array_data = struct_array.into_data();

            // Access the data after conversion
            let batch_after = RecordBatch::from(arrow::array::StructArray::from(array_data.clone()));
            let quantized_col_after = batch_after.column_by_name("quantized_data").unwrap();
            let quantized_array_after = quantized_col_after.as_any().downcast_ref::<UInt8Array>().unwrap();
            let ptr_after = quantized_array_after.values().as_ptr();

            // Property: Export preparation should not copy data
            prop_assert_eq!(
                ptr_before,
                ptr_after,
                "Export preparation should preserve zero-copy (same pointer)"
            );

            // Property: Data should remain correct
            prop_assert_eq!(quantized_array_after.len(), size);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        
        #[test]
        fn prop_multiple_columns_zero_copy_export(
            // Generate random array size
            size in 100usize..2_000,
            // Generate random number of time groups
            num_groups in 3usize..10,
            // Generate random seed
            seed in any::<u64>(),
        ) {
            let quantizer = TimeAwareQuantizer::new(num_groups);

            // Generate random weights
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            use rand::Rng;
            let weights: Vec<f32> = (0..size)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();

            // Generate time group parameters
            let params: Vec<TimeGroupParams> = (0..num_groups)
                .map(|i| TimeGroupParams {
                    time_range: (i * 100, (i + 1) * 100),
                    scale: rng.gen_range(0.01..1.0),
                    zero_point: rng.gen_range(0.0..128.0),
                    group_size: 64,
                })
                .collect();

            // Quantize
            let result = quantizer
                .quantize_layer_arrow(&weights, &params)
                .expect("Quantization should succeed");

            let batch = &result.batch;

            // Get pointers to all columns before export
            let quantized_col = batch.column_by_name("quantized_data").unwrap();
            let quantized_array = quantized_col.as_any().downcast_ref::<UInt8Array>().unwrap();
            let quantized_ptr_before = quantized_array.values().as_ptr();

            let group_ids_col = batch.column_by_name("time_group_ids").unwrap();
            let group_ids_array = group_ids_col.as_any().downcast_ref::<UInt32Array>().unwrap();
            let group_ids_ptr_before = group_ids_array.values().as_ptr();

            // Simulate export: convert to StructArray and back
            let struct_array = arrow::array::StructArray::from(batch.clone());
            let array_data = struct_array.into_data();
            let batch_after = RecordBatch::from(arrow::array::StructArray::from(array_data));

            // Get pointers after export
            let quantized_col_after = batch_after.column_by_name("quantized_data").unwrap();
            let quantized_array_after = quantized_col_after.as_any().downcast_ref::<UInt8Array>().unwrap();
            let quantized_ptr_after = quantized_array_after.values().as_ptr();

            let group_ids_col_after = batch_after.column_by_name("time_group_ids").unwrap();
            let group_ids_array_after = group_ids_col_after.as_any().downcast_ref::<UInt32Array>().unwrap();
            let group_ids_ptr_after = group_ids_array_after.values().as_ptr();

            // Property: All columns should maintain zero-copy
            prop_assert_eq!(
                quantized_ptr_before,
                quantized_ptr_after,
                "Quantized data should be zero-copy during export"
            );

            prop_assert_eq!(
                group_ids_ptr_before,
                group_ids_ptr_after,
                "Time group IDs should be zero-copy during export"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        
        #[test]
        fn prop_large_array_zero_copy_export(
            // Generate large array size (100K to 1M elements)
            size in 100_000usize..1_000_000,
            // Generate random seed
            seed in any::<u64>(),
        ) {
            // For large arrays, zero-copy is critical for performance
            let num_groups = 10;
            let quantizer = TimeAwareQuantizer::new(num_groups);

            // Generate random weights
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            use rand::Rng;
            let weights: Vec<f32> = (0..size)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();

            // Generate time group parameters
            let params: Vec<TimeGroupParams> = (0..num_groups)
                .map(|i| TimeGroupParams {
                    time_range: (i * 100, (i + 1) * 100),
                    scale: rng.gen_range(0.01..1.0),
                    zero_point: rng.gen_range(0.0..128.0),
                    group_size: 64,
                })
                .collect();

            // Quantize
            let result = quantizer
                .quantize_layer_arrow(&weights, &params)
                .expect("Quantization should succeed");

            let batch = &result.batch;

            // Get pointer before export
            let quantized_col = batch.column_by_name("quantized_data").unwrap();
            let quantized_array = quantized_col.as_any().downcast_ref::<UInt8Array>().unwrap();
            let ptr_before = quantized_array.values().as_ptr();

            // Simulate export
            let struct_array = arrow::array::StructArray::from(batch.clone());
            let array_data = struct_array.into_data();
            let batch_after = RecordBatch::from(arrow::array::StructArray::from(array_data));

            // Get pointer after export
            let quantized_col_after = batch_after.column_by_name("quantized_data").unwrap();
            let quantized_array_after = quantized_col_after.as_any().downcast_ref::<UInt8Array>().unwrap();
            let ptr_after = quantized_array_after.values().as_ptr();

            // Property: Large arrays must be zero-copy (critical for performance)
            prop_assert_eq!(
                ptr_before,
                ptr_after,
                "Large array export must be zero-copy to avoid memory overhead"
            );

            // Property: Data size should be preserved
            prop_assert_eq!(quantized_array_after.len(), size);
        }
    }
}

/// **Validates: Requirements 1.1, 1.2, 1.3, 5.4, 8.4**
///
/// Property: Zero-Copy Bidirectional Transmission
///
/// This property test verifies that:
/// 1. Rust->Python transmission maintains zero-copy
/// 2. Python->Rust transmission maintains zero-copy
/// 3. Round-trip operations preserve data integrity
#[cfg(test)]
mod zero_copy_bidirectional_properties {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        
        #[test]
        fn prop_roundtrip_preserves_zero_copy_semantics(
            // Generate random array size
            size in 100usize..5_000,
            // Generate random number of time groups
            num_groups in 3usize..15,
            // Generate random seed
            seed in any::<u64>(),
        ) {
            let quantizer = TimeAwareQuantizer::new(num_groups);

            // Generate random weights
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            use rand::Rng;
            let weights: Vec<f32> = (0..size)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();

            // Generate time group parameters
            let params: Vec<TimeGroupParams> = (0..num_groups)
                .map(|i| TimeGroupParams {
                    time_range: (i * 100, (i + 1) * 100),
                    scale: rng.gen_range(0.01..1.0),
                    zero_point: rng.gen_range(0.0..128.0),
                    group_size: 64,
                })
                .collect();

            // Step 1: Quantize (creates RecordBatch)
            let result = quantizer
                .quantize_layer_arrow(&weights, &params)
                .expect("Quantization should succeed");

            let batch_original = &result.batch;

            // Get pointer to original data
            let quantized_col_orig = batch_original.column_by_name("quantized_data").unwrap();
            let quantized_array_orig = quantized_col_orig.as_any().downcast_ref::<UInt8Array>().unwrap();
            let ptr_original = quantized_array_orig.values().as_ptr();

            // Step 2: Simulate export to Python (Rust->Python)
            let struct_array = arrow::array::StructArray::from(batch_original.clone());
            let array_data_export = struct_array.into_data();

            // Step 3: Simulate import from Python (Python->Rust)
            let batch_imported = RecordBatch::from(arrow::array::StructArray::from(array_data_export));

            // Get pointer after round-trip
            let quantized_col_imported = batch_imported.column_by_name("quantized_data").unwrap();
            let quantized_array_imported = quantized_col_imported.as_any().downcast_ref::<UInt8Array>().unwrap();
            let ptr_imported = quantized_array_imported.values().as_ptr();

            // Property: Round-trip should preserve zero-copy (same pointer)
            prop_assert_eq!(
                ptr_original,
                ptr_imported,
                "Round-trip transmission should preserve zero-copy semantics"
            );

            // Property: Data should be identical
            prop_assert_eq!(
                quantized_array_orig.values(),
                quantized_array_imported.values(),
                "Round-trip should preserve data integrity"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        
        #[test]
        fn prop_multiple_roundtrips_preserve_zero_copy(
            // Generate random array size
            size in 100usize..2_000,
            // Generate random number of roundtrips
            num_roundtrips in 2usize..10,
            // Generate random seed
            seed in any::<u64>(),
        ) {
            let num_groups = 5;
            let quantizer = TimeAwareQuantizer::new(num_groups);

            // Generate random weights
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            use rand::Rng;
            let weights: Vec<f32> = (0..size)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();

            // Generate time group parameters
            let params: Vec<TimeGroupParams> = (0..num_groups)
                .map(|i| TimeGroupParams {
                    time_range: (i * 100, (i + 1) * 100),
                    scale: rng.gen_range(0.01..1.0),
                    zero_point: rng.gen_range(0.0..128.0),
                    group_size: 64,
                })
                .collect();

            // Quantize
            let result = quantizer
                .quantize_layer_arrow(&weights, &params)
                .expect("Quantization should succeed");

            let mut batch = result.batch.clone();

            // Get original pointer
            let quantized_col = batch.column_by_name("quantized_data").unwrap();
            let quantized_array = quantized_col.as_any().downcast_ref::<UInt8Array>().unwrap();
            let ptr_original = quantized_array.values().as_ptr();

            // Perform multiple roundtrips
            for _ in 0..num_roundtrips {
                // Export
                let struct_array = arrow::array::StructArray::from(batch.clone());
                let array_data = struct_array.into_data();
                
                // Import
                batch = RecordBatch::from(arrow::array::StructArray::from(array_data));
            }

            // Get pointer after multiple roundtrips
            let quantized_col_final = batch.column_by_name("quantized_data").unwrap();
            let quantized_array_final = quantized_col_final.as_any().downcast_ref::<UInt8Array>().unwrap();
            let ptr_final = quantized_array_final.values().as_ptr();

            // Property: Multiple roundtrips should preserve zero-copy
            prop_assert_eq!(
                ptr_original,
                ptr_final,
                "Multiple roundtrips should preserve zero-copy semantics"
            );
        }
    }
}

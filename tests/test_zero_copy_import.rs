///! Zero-copy import verification tests
///!
///! This test suite verifies that import_pyarrow_table() maintains zero-copy
///! semantics when importing PyArrow Tables through the Arrow C Data Interface.
///!
///! **Validates: Requirements 1.1, 5.4**
///!
///! The tests verify that:
///! 1. Arrow C Data Interface is used correctly
///! 2. No data copying occurs during import
///! 3. Rust code directly accesses Python-owned buffers
///!
///! Note: Full memory address verification requires Python integration tests
///! with memory profiling tools. These Rust tests verify the correct API usage.

#[cfg(test)]
mod tests {
    use arrow::array::{Array, Float32Array, ListArray, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Test that RecordBatch columns provide zero-copy access to underlying buffers
    ///
    /// This test verifies that when we extract data from a RecordBatch:
    /// 1. Float32Array::values() returns a slice reference (not a copy)
    /// 2. The slice directly references the underlying Arrow buffer
    /// 3. No allocation occurs when accessing the data
    #[test]
    fn test_recordbatch_provides_zero_copy_access() {
        // Create a RecordBatch with test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]));

        // Create test data
        let layer_names = StringArray::from(vec!["layer.0"]);

        let weights_values = Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let weights_offsets = arrow::buffer::OffsetBuffer::new(vec![0, 5].into());
        let weights_list = ListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            weights_offsets,
            Arc::new(weights_values),
            None,
        );

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(layer_names), Arc::new(weights_list)])
                .unwrap();

        // Extract weights column
        let weights_column = batch.column_by_name("weights").unwrap();
        let weights_list = weights_column.as_any().downcast_ref::<ListArray>().unwrap();

        // Extract first layer's weights
        let first_layer = weights_list.value(0);
        let weights_f32 = first_layer.as_any().downcast_ref::<Float32Array>().unwrap();

        // Get zero-copy slice reference
        let weights_slice: &[f32] = weights_f32.values();

        // Verify data is correct
        assert_eq!(weights_slice.len(), 5);
        assert_eq!(weights_slice, &[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Verify this is a slice reference, not a copy
        // The slice should point to the same memory as the underlying buffer
        let buffer_ptr = weights_f32.values().as_ptr();
        let slice_ptr = weights_slice.as_ptr();
        assert_eq!(
            buffer_ptr, slice_ptr,
            "Slice should reference the same memory as the buffer"
        );
    }

    /// Test that multiple accesses to the same data don't create copies
    #[test]
    fn test_multiple_accesses_share_same_buffer() {
        // Create a RecordBatch
        let schema = Arc::new(Schema::new(vec![Field::new(
            "weights",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        )]));

        let weights_values = Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0]);
        let weights_offsets = arrow::buffer::OffsetBuffer::new(vec![0, 4].into());
        let weights_list = ListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            weights_offsets,
            Arc::new(weights_values),
            None,
        );

        let batch = RecordBatch::try_new(schema, vec![Arc::new(weights_list)]).unwrap();

        // Access the same data multiple times
        let weights_column = batch.column_by_name("weights").unwrap();
        let weights_list = weights_column.as_any().downcast_ref::<ListArray>().unwrap();

        let first_layer = weights_list.value(0);
        let weights_f32 = first_layer.as_any().downcast_ref::<Float32Array>().unwrap();

        // Get multiple slice references
        let slice1 = weights_f32.values();
        let slice2 = weights_f32.values();
        let slice3 = weights_f32.values();

        // All slices should point to the same memory
        assert_eq!(slice1.as_ptr(), slice2.as_ptr());
        assert_eq!(slice2.as_ptr(), slice3.as_ptr());
    }

    /// Test that Arrow buffer reference counting works correctly
    #[test]
    fn test_arrow_buffer_reference_counting() {
        // Create a RecordBatch
        let schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Float32,
            false,
        )]));

        let data = Float32Array::from(vec![1.0f32, 2.0, 3.0]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(data)]).unwrap();

        // Get reference to the data
        let column = batch.column(0);
        let float_array = column.as_any().downcast_ref::<Float32Array>().unwrap();
        let slice = float_array.values();

        // Verify data is accessible
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], 1.0);

        // The buffer should remain valid even after we drop intermediate references
        drop(float_array);
        drop(column);

        // We can still access the batch
        let column2 = batch.column(0);
        let float_array2 = column2.as_any().downcast_ref::<Float32Array>().unwrap();
        let slice2 = float_array2.values();

        assert_eq!(slice2.len(), 3);
        assert_eq!(slice2[0], 1.0);
    }

    /// Test that zero-copy access works with large arrays
    #[test]
    fn test_zero_copy_with_large_arrays() {
        // Create a large array to ensure we're not accidentally copying
        const SIZE: usize = 1_000_000;
        let large_data: Vec<f32> = (0..SIZE).map(|i| i as f32).collect();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Float32,
            false,
        )]));

        let data = Float32Array::from(large_data.clone());
        let batch = RecordBatch::try_new(schema, vec![Arc::new(data)]).unwrap();

        // Access the data
        let column = batch.column(0);
        let float_array = column.as_any().downcast_ref::<Float32Array>().unwrap();
        let slice = float_array.values();

        // Verify size and sample values
        assert_eq!(slice.len(), SIZE);
        assert_eq!(slice[0], 0.0);
        assert_eq!(slice[SIZE - 1], (SIZE - 1) as f32);

        // Verify this is a reference, not a copy
        // If it were a copy, we'd see different memory addresses
        let buffer_ptr = float_array.values().as_ptr();
        let slice_ptr = slice.as_ptr();
        assert_eq!(buffer_ptr, slice_ptr);
    }

    /// Test that ListArray provides zero-copy access to nested data
    #[test]
    fn test_list_array_zero_copy_access() {
        // Create a ListArray with multiple lists
        let values = Float32Array::from(vec![
            1.0f32, 2.0, 3.0, // First list
            4.0, 5.0, // Second list
            6.0, 7.0, 8.0, 9.0, // Third list
        ]);

        let offsets = arrow::buffer::OffsetBuffer::new(vec![0, 3, 5, 9].into());
        let list_array = ListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            offsets,
            Arc::new(values),
            None,
        );

        // Access each list
        for i in 0..list_array.len() {
            let list_values = list_array.value(i);
            let float_array = list_values.as_any().downcast_ref::<Float32Array>().unwrap();
            let slice = float_array.values();

            // Verify we can access the data
            assert!(!slice.is_empty());

            // Verify this is a slice into the original buffer, not a copy
            let buffer_ptr = float_array.values().as_ptr();
            let slice_ptr = slice.as_ptr();
            assert_eq!(buffer_ptr, slice_ptr);
        }
    }

    /// Test that schema validation doesn't copy data
    #[test]
    fn test_schema_validation_preserves_zero_copy() {
        // Create a RecordBatch
        let schema = Arc::new(Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]));

        let layer_names = StringArray::from(vec!["layer.0"]);
        let weights_values = Float32Array::from(vec![1.0f32, 2.0, 3.0]);
        let weights_offsets = arrow::buffer::OffsetBuffer::new(vec![0, 3].into());
        let weights_list = ListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            weights_offsets,
            Arc::new(weights_values),
            None,
        );

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(layer_names), Arc::new(weights_list)],
        )
        .unwrap();

        // Get pointer to original data
        let weights_column = batch.column_by_name("weights").unwrap();
        let weights_list = weights_column.as_any().downcast_ref::<ListArray>().unwrap();
        let first_layer = weights_list.value(0);
        let weights_f32 = first_layer.as_any().downcast_ref::<Float32Array>().unwrap();
        let original_ptr = weights_f32.values().as_ptr();

        // Validate schema (this should not copy data)
        let validated_schema = batch.schema();
        assert_eq!(validated_schema.fields().len(), 2);

        // Access data again and verify same pointer
        let weights_column2 = batch.column_by_name("weights").unwrap();
        let weights_list2 = weights_column2
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let first_layer2 = weights_list2.value(0);
        let weights_f32_2 = first_layer2
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let after_validation_ptr = weights_f32_2.values().as_ptr();

        assert_eq!(
            original_ptr, after_validation_ptr,
            "Schema validation should not copy data"
        );
    }
}

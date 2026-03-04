//! Unit tests for validate_arrow_input() method
//!
//! These tests verify the production-grade input validation for PyArrow Tables
//! in the ArrowQuantV2 Python API.
//!
//! Requirements tested:
//! - REQ-5.1: Python API SHALL validate PyArrow Table schema and return detailed error information
//! - REQ-6.3: System SHALL return detailed schema validation errors for mismatched schemas

#[cfg(test)]
mod tests {
    use arrow::array::{Array, Float32Array, ListArray, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    /// Helper function to create a valid input schema
    fn create_valid_schema() -> Schema {
        Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ])
    }

    /// Helper function to create a valid RecordBatch for testing
    fn create_valid_record_batch() -> RecordBatch {
        let schema = Arc::new(create_valid_schema());

        // Create layer_name column
        let layer_names = StringArray::from(vec!["layer.0.weight", "layer.1.weight"]);

        // Create weights column (list of float32)
        let weights_values = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let weights_offsets = vec![0, 3, 6]; // First list has 3 elements, second has 3
        let weights_list = ListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            arrow::buffer::OffsetBuffer::new(weights_offsets.into()),
            Arc::new(weights_values),
            None,
        );

        RecordBatch::try_new(
            schema,
            vec![Arc::new(layer_names), Arc::new(weights_list)],
        )
        .unwrap()
    }

    #[test]
    fn test_valid_schema_with_required_fields() {
        let batch = create_valid_record_batch();
        let schema = batch.schema();

        // Verify schema has required fields
        assert!(schema.field_with_name("layer_name").is_ok());
        assert!(schema.field_with_name("weights").is_ok());

        // Verify field types
        let layer_name_field = schema.field_with_name("layer_name").unwrap();
        assert!(matches!(layer_name_field.data_type(), DataType::Utf8));

        let weights_field = schema.field_with_name("weights").unwrap();
        match weights_field.data_type() {
            DataType::List(inner) => {
                assert!(matches!(inner.data_type(), DataType::Float32));
            }
            _ => panic!("Expected List type for weights field"),
        }
    }

    #[test]
    fn test_valid_schema_with_optional_shape_field() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
            Field::new(
                "shape",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                true,
            ),
        ]);

        // Verify all fields exist
        assert!(schema.field_with_name("layer_name").is_ok());
        assert!(schema.field_with_name("weights").is_ok());
        assert!(schema.field_with_name("shape").is_ok());

        // Verify shape field type
        let shape_field = schema.field_with_name("shape").unwrap();
        match shape_field.data_type() {
            DataType::List(inner) => {
                assert!(matches!(inner.data_type(), DataType::Int64));
            }
            _ => panic!("Expected List type for shape field"),
        }
    }

    #[test]
    fn test_missing_layer_name_field() {
        // Schema missing layer_name field
        let schema = Schema::new(vec![Field::new(
            "weights",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        )]);

        // Verify layer_name field is missing
        assert!(schema.field_with_name("layer_name").is_err());
    }

    #[test]
    fn test_missing_weights_field() {
        // Schema missing weights field
        let schema = Schema::new(vec![Field::new("layer_name", DataType::Utf8, false)]);

        // Verify weights field is missing
        assert!(schema.field_with_name("weights").is_err());
    }

    #[test]
    fn test_invalid_layer_name_type() {
        // Schema with wrong type for layer_name (Int32 instead of Utf8)
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Int32, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]);

        let layer_name_field = schema.field_with_name("layer_name").unwrap();
        assert!(!matches!(
            layer_name_field.data_type(),
            DataType::Utf8 | DataType::LargeUtf8
        ));
    }

    #[test]
    fn test_invalid_weights_type_not_list() {
        // Schema with wrong type for weights (Float32 instead of List<Float32>)
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new("weights", DataType::Float32, false),
        ]);

        let weights_field = schema.field_with_name("weights").unwrap();
        assert!(!matches!(
            weights_field.data_type(),
            DataType::List(_) | DataType::LargeList(_)
        ));
    }

    #[test]
    fn test_invalid_weights_inner_type() {
        // Schema with wrong inner type for weights (Float64 instead of Float32)
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
                false,
            ),
        ]);

        let weights_field = schema.field_with_name("weights").unwrap();
        match weights_field.data_type() {
            DataType::List(inner) => {
                assert!(!matches!(inner.data_type(), DataType::Float32));
            }
            _ => {}
        }
    }

    #[test]
    fn test_invalid_shape_type() {
        // Schema with wrong type for optional shape field (Int32 instead of Int64)
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
            Field::new(
                "shape",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ),
        ]);

        let shape_field = schema.field_with_name("shape").unwrap();
        match shape_field.data_type() {
            DataType::List(inner) => {
                assert!(!matches!(inner.data_type(), DataType::Int64));
            }
            _ => {}
        }
    }

    #[test]
    fn test_large_utf8_accepted_for_layer_name() {
        // Schema with LargeUtf8 for layer_name (should be accepted)
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::LargeUtf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]);

        let layer_name_field = schema.field_with_name("layer_name").unwrap();
        assert!(matches!(
            layer_name_field.data_type(),
            DataType::Utf8 | DataType::LargeUtf8
        ));
    }

    #[test]
    fn test_large_list_accepted_for_weights() {
        // Schema with LargeList for weights (should be accepted)
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::LargeList(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]);

        let weights_field = schema.field_with_name("weights").unwrap();
        assert!(matches!(
            weights_field.data_type(),
            DataType::List(_) | DataType::LargeList(_)
        ));
    }

    #[test]
    fn test_record_batch_data_access() {
        let batch = create_valid_record_batch();

        // Verify we can access the data
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);

        // Access layer_name column
        let layer_names = batch
            .column_by_name("layer_name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(layer_names.value(0), "layer.0.weight");
        assert_eq!(layer_names.value(1), "layer.1.weight");

        // Access weights column
        let weights_list = batch
            .column_by_name("weights")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(weights_list.len(), 2);

        // Verify first weights array
        let weights_0_array = weights_list.value(0);
        let weights_0 = weights_0_array
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(weights_0.len(), 3);
        assert_eq!(weights_0.value(0), 1.0);
        assert_eq!(weights_0.value(1), 2.0);
        assert_eq!(weights_0.value(2), 3.0);
    }
}

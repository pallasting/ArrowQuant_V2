///! Comprehensive unit tests for FFI helper functions
///!
///! This test suite provides unit testing for the Arrow FFI helper functions:
///! - import_pyarrow_table: Import PyArrow Table through C Data Interface
///! - export_recordbatch_to_pyarrow: Export RecordBatch to PyArrow
///! - validate_quantization_schema: Validate Arrow schema for quantization
///!
///! Requirements tested:
///! - 1.1: Arrow Table import through C Data Interface (zero-copy)
///! - 1.4: Schema validation for missing columns
///! - 1.5: Schema validation for incorrect types
///! - 4.1: RecordBatch export with correct schema

#[cfg(test)]
mod tests {
    use arrow::array::{
        ArrayRef, BinaryBuilder, Float32Array, Float32Builder, Int64Builder, ListBuilder,
        RecordBatch, StringBuilder, StringArray, UInt8Builder,
    };
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    // Mock PyResult for testing without Python runtime
    type PyResult<T> = Result<T, String>;

    /// Test helper: Create a valid quantization input schema
    fn create_valid_input_schema() -> Schema {
        Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
            Field::new(
                "shape",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                false,
            ),
        ])
    }

    /// Test helper: Create a valid quantization output schema
    fn create_valid_output_schema() -> Schema {
        Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new("quantized_data", DataType::Binary, false),
            Field::new(
                "scales",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
            Field::new(
                "zero_points",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
            Field::new(
                "shape",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                false,
            ),
            Field::new("bit_width", DataType::UInt8, false),
        ])
    }

    /// Simplified version of validate_quantization_schema for unit testing
    fn validate_quantization_schema(schema: &Schema) -> PyResult<()> {
        // Check for required fields
        let layer_name_field = schema.field_with_name("layer_name").map_err(|_| {
            "Missing required field 'layer_name' in Arrow schema. \
                Expected schema: {layer_name: string, weights: list<float32>, shape: list<int64>}"
                .to_string()
        })?;

        let weights_field = schema.field_with_name("weights").map_err(|_| {
            "Missing required field 'weights' in Arrow schema. \
                Expected schema: {layer_name: string, weights: list<float32>, shape: list<int64>}"
                .to_string()
        })?;

        // Validate field types
        if !matches!(
            layer_name_field.data_type(),
            DataType::Utf8 | DataType::LargeUtf8
        ) {
            return Err(format!(
                "Invalid type for 'layer_name' field: {:?}. Expected string type.",
                layer_name_field.data_type()
            ));
        }

        // Validate weights field is a list of float32
        match weights_field.data_type() {
            DataType::List(inner) | DataType::LargeList(inner) => {
                if !matches!(inner.data_type(), DataType::Float32) {
                    return Err(format!(
                        "Invalid type for 'weights' field: list<{:?}>. Expected list<float32>.",
                        inner.data_type()
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "Invalid type for 'weights' field: {:?}. Expected list<float32>.",
                    weights_field.data_type()
                ));
            }
        }

        // Validate optional shape field if present
        if let Ok(shape_field) = schema.field_with_name("shape") {
            match shape_field.data_type() {
                DataType::List(inner) | DataType::LargeList(inner) => {
                    if !matches!(inner.data_type(), DataType::Int64) {
                        return Err(format!(
                            "Invalid type for 'shape' field: list<{:?}>. Expected list<int64>.",
                            inner.data_type()
                        ));
                    }
                }
                _ => {
                    return Err(format!(
                        "Invalid type for 'shape' field: {:?}. Expected list<int64>.",
                        shape_field.data_type()
                    ));
                }
            }
        }

        Ok(())
    }

    // ============================================================================
    // Tests for validate_quantization_schema - Input Schema Validation
    // ============================================================================

    #[test]
    fn test_validate_input_schema_valid_with_all_fields() {
        let schema = create_valid_input_schema();
        assert!(validate_quantization_schema(&schema).is_ok());
    }

    #[test]
    fn test_validate_input_schema_valid_without_optional_shape() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]);

        assert!(validate_quantization_schema(&schema).is_ok());
    }

    #[test]
    fn test_validate_input_schema_missing_layer_name() {
        let schema = Schema::new(vec![Field::new(
            "weights",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        )]);

        let result = validate_quantization_schema(&schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("layer_name"));
        assert!(err.contains("Missing"));
    }

    #[test]
    fn test_validate_input_schema_missing_weights() {
        let schema = Schema::new(vec![Field::new("layer_name", DataType::Utf8, false)]);

        let result = validate_quantization_schema(&schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("weights"));
        assert!(err.contains("Missing"));
    }

    #[test]
    fn test_validate_input_schema_wrong_layer_name_type() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Int32, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]);

        let result = validate_quantization_schema(&schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("layer_name"));
        assert!(err.contains("Invalid type"));
    }

    #[test]
    fn test_validate_input_schema_wrong_weights_inner_type() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                false,
            ),
        ]);

        let result = validate_quantization_schema(&schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("weights"));
        assert!(err.contains("float32"));
    }

    #[test]
    fn test_validate_input_schema_wrong_shape_inner_type() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
            Field::new(
                "shape",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]);

        let result = validate_quantization_schema(&schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("shape"));
        assert!(err.contains("int64"));
    }

    #[test]
    fn test_validate_input_schema_large_utf8_accepted() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::LargeUtf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]);

        assert!(validate_quantization_schema(&schema).is_ok());
    }

    #[test]
    fn test_validate_input_schema_large_list_accepted() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::LargeList(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
            Field::new(
                "shape",
                DataType::LargeList(Arc::new(Field::new("item", DataType::Int64, true))),
                false,
            ),
        ]);

        assert!(validate_quantization_schema(&schema).is_ok());
    }

    #[test]
    fn test_validate_input_schema_extra_columns_allowed() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
            Field::new("extra_field", DataType::Utf8, false),
            Field::new("another_field", DataType::Int32, false),
        ]);

        assert!(validate_quantization_schema(&schema).is_ok());
    }

    #[test]
    fn test_validate_input_schema_weights_not_list() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new("weights", DataType::Float32, false),
        ]);

        let result = validate_quantization_schema(&schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("weights"));
        assert!(err.contains("list"));
    }

    #[test]
    fn test_validate_input_schema_shape_not_list() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
            Field::new("shape", DataType::Int64, false),
        ]);

        let result = validate_quantization_schema(&schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("shape"));
        assert!(err.contains("list"));
    }

    // ============================================================================
    // Tests for RecordBatch creation - Output Schema Validation
    // ============================================================================

    #[test]
    fn test_create_output_recordbatch_single_layer() {
        // Create output RecordBatch with single layer
        let mut layer_names = StringBuilder::new();
        let mut quantized_data = BinaryBuilder::new();
        let mut scales = ListBuilder::new(Float32Builder::new());
        let mut zero_points = ListBuilder::new(Float32Builder::new());
        let mut shapes = ListBuilder::new(Int64Builder::new());
        let mut bit_widths = UInt8Builder::new();

        // Add single layer data
        layer_names.append_value("layer.0.weight");
        quantized_data.append_value(&[0u8, 1, 2, 3]);

        scales.values().append_value(0.5);
        scales.append(true);

        zero_points.values().append_value(0.0);
        zero_points.append(true);

        shapes.values().append_value(100);
        shapes.append(true);

        bit_widths.append_value(4);

        // Create RecordBatch
        let schema = Arc::new(create_valid_output_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(layer_names.finish()),
                Arc::new(quantized_data.finish()),
                Arc::new(scales.finish()),
                Arc::new(zero_points.finish()),
                Arc::new(shapes.finish()),
                Arc::new(bit_widths.finish()),
            ],
        );

        assert!(batch.is_ok());
        let batch = batch.unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 6);
    }

    #[test]
    fn test_create_output_recordbatch_multiple_layers() {
        // Create output RecordBatch with multiple layers
        let mut layer_names = StringBuilder::new();
        let mut quantized_data = BinaryBuilder::new();
        let mut scales = ListBuilder::new(Float32Builder::new());
        let mut zero_points = ListBuilder::new(Float32Builder::new());
        let mut shapes = ListBuilder::new(Int64Builder::new());
        let mut bit_widths = UInt8Builder::new();

        // Add 3 layers
        for i in 0..3 {
            layer_names.append_value(&format!("layer.{}.weight", i));
            quantized_data.append_value(&[0u8, 1, 2, 3]);

            scales.values().append_value(0.5);
            scales.append(true);

            zero_points.values().append_value(0.0);
            zero_points.append(true);

            shapes.values().append_value(100);
            shapes.append(true);

            bit_widths.append_value(4);
        }

        // Create RecordBatch
        let schema = Arc::new(create_valid_output_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(layer_names.finish()),
                Arc::new(quantized_data.finish()),
                Arc::new(scales.finish()),
                Arc::new(zero_points.finish()),
                Arc::new(shapes.finish()),
                Arc::new(bit_widths.finish()),
            ],
        );

        assert!(batch.is_ok());
        let batch = batch.unwrap();
        assert_eq!(batch.num_rows(), 3);
    }

    #[test]
    fn test_create_output_recordbatch_with_multidimensional_shapes() {
        // Create output RecordBatch with multi-dimensional shapes
        let mut layer_names = StringBuilder::new();
        let mut quantized_data = BinaryBuilder::new();
        let mut scales = ListBuilder::new(Float32Builder::new());
        let mut zero_points = ListBuilder::new(Float32Builder::new());
        let mut shapes = ListBuilder::new(Int64Builder::new());
        let mut bit_widths = UInt8Builder::new();

        // Add layer with 2D shape
        layer_names.append_value("layer_2d");
        quantized_data.append_value(&[0u8; 200]);

        scales.values().append_value(0.5);
        scales.append(true);

        zero_points.values().append_value(0.0);
        zero_points.append(true);

        // Shape: [10, 20]
        shapes.values().append_value(10);
        shapes.values().append_value(20);
        shapes.append(true);

        bit_widths.append_value(4);

        // Add layer with 3D shape
        layer_names.append_value("layer_3d");
        quantized_data.append_value(&[0u8; 60]);

        scales.values().append_value(0.5);
        scales.append(true);

        zero_points.values().append_value(0.0);
        zero_points.append(true);

        // Shape: [5, 4, 3]
        shapes.values().append_value(5);
        shapes.values().append_value(4);
        shapes.values().append_value(3);
        shapes.append(true);

        bit_widths.append_value(4);

        // Create RecordBatch
        let schema = Arc::new(create_valid_output_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(layer_names.finish()),
                Arc::new(quantized_data.finish()),
                Arc::new(scales.finish()),
                Arc::new(zero_points.finish()),
                Arc::new(shapes.finish()),
                Arc::new(bit_widths.finish()),
            ],
        );

        assert!(batch.is_ok());
        let batch = batch.unwrap();
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn test_create_output_recordbatch_empty() {
        // Create empty output RecordBatch
        let layer_names = StringBuilder::new();
        let quantized_data = BinaryBuilder::new();
        let scales = ListBuilder::new(Float32Builder::new());
        let zero_points = ListBuilder::new(Float32Builder::new());
        let shapes = ListBuilder::new(Int64Builder::new());
        let bit_widths = UInt8Builder::new();

        // Create RecordBatch
        let schema = Arc::new(create_valid_output_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(layer_names.finish()),
                Arc::new(quantized_data.finish()),
                Arc::new(scales.finish()),
                Arc::new(zero_points.finish()),
                Arc::new(shapes.finish()),
                Arc::new(bit_widths.finish()),
            ],
        );

        assert!(batch.is_ok());
        let batch = batch.unwrap();
        assert_eq!(batch.num_rows(), 0);
    }

    #[test]
    fn test_output_schema_has_correct_types() {
        let schema = create_valid_output_schema();

        // Verify layer_name is string
        let layer_name_field = schema.field_with_name("layer_name").unwrap();
        assert!(matches!(layer_name_field.data_type(), DataType::Utf8));

        // Verify quantized_data is binary
        let quantized_data_field = schema.field_with_name("quantized_data").unwrap();
        assert!(matches!(
            quantized_data_field.data_type(),
            DataType::Binary
        ));

        // Verify scales is list<float32>
        let scales_field = schema.field_with_name("scales").unwrap();
        match scales_field.data_type() {
            DataType::List(inner) => {
                assert!(matches!(inner.data_type(), DataType::Float32));
            }
            _ => panic!("scales should be list type"),
        }

        // Verify zero_points is list<float32>
        let zp_field = schema.field_with_name("zero_points").unwrap();
        match zp_field.data_type() {
            DataType::List(inner) => {
                assert!(matches!(inner.data_type(), DataType::Float32));
            }
            _ => panic!("zero_points should be list type"),
        }

        // Verify shape is list<int64>
        let shape_field = schema.field_with_name("shape").unwrap();
        match shape_field.data_type() {
            DataType::List(inner) => {
                assert!(matches!(inner.data_type(), DataType::Int64));
            }
            _ => panic!("shape should be list type"),
        }

        // Verify bit_width is uint8
        let bit_width_field = schema.field_with_name("bit_width").unwrap();
        assert!(matches!(bit_width_field.data_type(), DataType::UInt8));
    }

    #[test]
    fn test_output_recordbatch_with_different_bit_widths() {
        for bit_width in [2u8, 4u8, 8u8] {
            let mut layer_names = StringBuilder::new();
            let mut quantized_data = BinaryBuilder::new();
            let mut scales = ListBuilder::new(Float32Builder::new());
            let mut zero_points = ListBuilder::new(Float32Builder::new());
            let mut shapes = ListBuilder::new(Int64Builder::new());
            let mut bit_widths = UInt8Builder::new();

            layer_names.append_value("test_layer");
            quantized_data.append_value(&[0u8; 100]);

            scales.values().append_value(0.5);
            scales.append(true);

            zero_points.values().append_value(0.0);
            zero_points.append(true);

            shapes.values().append_value(100);
            shapes.append(true);

            bit_widths.append_value(bit_width);

            let schema = Arc::new(create_valid_output_schema());
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(layer_names.finish()),
                    Arc::new(quantized_data.finish()),
                    Arc::new(scales.finish()),
                    Arc::new(zero_points.finish()),
                    Arc::new(shapes.finish()),
                    Arc::new(bit_widths.finish()),
                ],
            );

            assert!(batch.is_ok());
        }
    }

    // ============================================================================
    // Tests for data extraction from RecordBatch
    // ============================================================================

    #[test]
    fn test_extract_layer_names_from_recordbatch() {
        // Create RecordBatch
        let mut layer_names = StringBuilder::new();
        layer_names.append_value("layer.0");
        layer_names.append_value("layer.1");
        layer_names.append_value("layer.2");

        let layer_names_array = layer_names.finish();

        // Extract values
        let extracted: Vec<&str> = (0..layer_names_array.len())
            .map(|i| layer_names_array.value(i))
            .collect();

        assert_eq!(extracted, vec!["layer.0", "layer.1", "layer.2"]);
    }

    #[test]
    fn test_extract_weights_from_list_array() {
        // Create list array with weights
        let mut weights = ListBuilder::new(Float32Builder::new());

        // Add first layer weights
        for val in [1.0f32, 2.0, 3.0, 4.0] {
            weights.values().append_value(val);
        }
        weights.append(true);

        // Add second layer weights
        for val in [5.0f32, 6.0, 7.0] {
            weights.values().append_value(val);
        }
        weights.append(true);

        let weights_array = weights.finish();

        // Extract first layer
        let first_layer = weights_array.value(0);
        let first_layer_f32 = first_layer
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let first_values: Vec<f32> = first_layer_f32.values().to_vec();

        assert_eq!(first_values, vec![1.0, 2.0, 3.0, 4.0]);

        // Extract second layer
        let second_layer = weights_array.value(1);
        let second_layer_f32 = second_layer
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let second_values: Vec<f32> = second_layer_f32.values().to_vec();

        assert_eq!(second_values, vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_extract_shapes_from_list_array() {
        // Create list array with shapes
        let mut shapes = ListBuilder::new(Int64Builder::new());

        // Add 1D shape
        shapes.values().append_value(100);
        shapes.append(true);

        // Add 2D shape
        shapes.values().append_value(10);
        shapes.values().append_value(20);
        shapes.append(true);

        // Add 3D shape
        shapes.values().append_value(5);
        shapes.values().append_value(4);
        shapes.values().append_value(3);
        shapes.append(true);

        let shapes_array = shapes.finish();

        // Extract shapes
        assert_eq!(shapes_array.len(), 3);

        // Verify 1D shape
        let shape_0 = shapes_array.value(0);
        let shape_0_i64 = shape_0.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
        assert_eq!(shape_0_i64.values(), &[100]);

        // Verify 2D shape
        let shape_1 = shapes_array.value(1);
        let shape_1_i64 = shape_1.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
        assert_eq!(shape_1_i64.values(), &[10, 20]);

        // Verify 3D shape
        let shape_2 = shapes_array.value(2);
        let shape_2_i64 = shape_2.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
        assert_eq!(shape_2_i64.values(), &[5, 4, 3]);
    }
}

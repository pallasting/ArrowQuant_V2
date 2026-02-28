/**
 * Unit tests for Task 2.4: Result Building Phase (Rust-only)
 * 
 * Tests the Arrow RecordBatch building logic without requiring Python runtime.
 * 
 * **Validates: Requirements 4.1, 4.2, 4.5**
 */

#[cfg(test)]
mod test_result_building {
    use arrow::array::{
        Array, BinaryBuilder, Float32Builder, Int64Builder, ListBuilder, 
        StringBuilder, UInt8Builder, RecordBatch
    };
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_result_schema_creation() {
        // Test that we can create the expected result schema
        let result_schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new("quantized_data", DataType::Binary, false),
            Field::new(
                "scales",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
                false,
            ),
            Field::new(
                "zero_points",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
                false,
            ),
            Field::new(
                "shape",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
                false,
            ),
            Field::new("bit_width", DataType::UInt8, false),
        ]);

        // Verify schema has all required fields
        assert_eq!(result_schema.fields().len(), 6);
        assert!(result_schema.field_with_name("layer_name").is_ok());
        assert!(result_schema.field_with_name("quantized_data").is_ok());
        assert!(result_schema.field_with_name("scales").is_ok());
        assert!(result_schema.field_with_name("zero_points").is_ok());
        assert!(result_schema.field_with_name("shape").is_ok());
        assert!(result_schema.field_with_name("bit_width").is_ok());
    }

    #[test]
    fn test_result_builders_creation() {
        // Test that we can create all required builders
        let mut result_layer_names = StringBuilder::new();
        let mut result_quantized_data = BinaryBuilder::new();
        let mut result_scales = ListBuilder::new(Float32Builder::new());
        let mut result_zero_points = ListBuilder::new(Float32Builder::new());
        let mut result_shapes = ListBuilder::new(Int64Builder::new());
        let mut result_bit_widths = UInt8Builder::new();

        // Append test data
        result_layer_names.append_value("test.layer");
        result_quantized_data.append_value(&[1u8, 2, 3, 4]);
        
        // Append scales
        result_scales.values().append_value(0.5);
        result_scales.values().append_value(0.6);
        result_scales.append(true);
        
        // Append zero_points
        result_zero_points.values().append_value(0.0);
        result_zero_points.values().append_value(0.1);
        result_zero_points.append(true);
        
        // Append shape
        result_shapes.values().append_value(128);
        result_shapes.append(true);
        
        result_bit_widths.append_value(4);

        // Finish builders
        let layer_names_array = result_layer_names.finish();
        let quantized_data_array = result_quantized_data.finish();
        let scales_array = result_scales.finish();
        let zero_points_array = result_zero_points.finish();
        let shapes_array = result_shapes.finish();
        let bit_widths_array = result_bit_widths.finish();

        // Verify arrays have correct length
        assert_eq!(layer_names_array.len(), 1);
        assert_eq!(quantized_data_array.len(), 1);
        assert_eq!(scales_array.len(), 1);
        assert_eq!(zero_points_array.len(), 1);
        assert_eq!(shapes_array.len(), 1);
        assert_eq!(bit_widths_array.len(), 1);
    }

    #[test]
    fn test_result_recordbatch_creation() {
        // Test that we can create a RecordBatch with the result schema
        // Create builders
        let mut result_layer_names = StringBuilder::new();
        let mut result_quantized_data = BinaryBuilder::new();
        let mut result_scales = ListBuilder::new(Float32Builder::new());
        let mut result_zero_points = ListBuilder::new(Float32Builder::new());
        let mut result_shapes = ListBuilder::new(Int64Builder::new());
        let mut result_bit_widths = UInt8Builder::new();

        // Append test data for 2 layers
        for i in 0..2 {
            result_layer_names.append_value(&format!("layer.{}", i));
            result_quantized_data.append_value(&[1u8, 2, 3, 4]);
            
            result_scales.values().append_value(0.5);
            result_scales.append(true);
            
            result_zero_points.values().append_value(0.0);
            result_zero_points.append(true);
            
            result_shapes.values().append_value(128);
            result_shapes.append(true);
            
            result_bit_widths.append_value(4);
        }

        // Finish builders to get arrays
        let layer_names_array = Arc::new(result_layer_names.finish());
        let quantized_data_array = Arc::new(result_quantized_data.finish());
        let scales_array = Arc::new(result_scales.finish());
        let zero_points_array = Arc::new(result_zero_points.finish());
        let shapes_array = Arc::new(result_shapes.finish());
        let bit_widths_array = Arc::new(result_bit_widths.finish());
        
        // Create schema from actual array types
        let result_schema = Schema::new(vec![
            Field::new("layer_name", layer_names_array.data_type().clone(), false),
            Field::new("quantized_data", quantized_data_array.data_type().clone(), false),
            Field::new("scales", scales_array.data_type().clone(), false),
            Field::new("zero_points", zero_points_array.data_type().clone(), false),
            Field::new("shape", shapes_array.data_type().clone(), false),
            Field::new("bit_width", bit_widths_array.data_type().clone(), false),
        ]);

        // Create RecordBatch
        let result_batch = RecordBatch::try_new(
            Arc::new(result_schema),
            vec![
                layer_names_array,
                quantized_data_array,
                scales_array,
                zero_points_array,
                shapes_array,
                bit_widths_array,
            ],
        );

        if let Err(e) = &result_batch {
            panic!("Failed to create RecordBatch: {:?}", e);
        }
        let batch = result_batch.unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 6);
    }

    #[test]
    fn test_result_empty_recordbatch() {
        // Test that we can create an empty RecordBatch
        // Create empty builders
        let mut result_layer_names = StringBuilder::new();
        let mut result_quantized_data = BinaryBuilder::new();
        let mut result_scales = ListBuilder::new(Float32Builder::new());
        let mut result_zero_points = ListBuilder::new(Float32Builder::new());
        let mut result_shapes = ListBuilder::new(Int64Builder::new());
        let mut result_bit_widths = UInt8Builder::new();

        // Finish builders to get arrays
        let layer_names_array = Arc::new(result_layer_names.finish());
        let quantized_data_array = Arc::new(result_quantized_data.finish());
        let scales_array = Arc::new(result_scales.finish());
        let zero_points_array = Arc::new(result_zero_points.finish());
        let shapes_array = Arc::new(result_shapes.finish());
        let bit_widths_array = Arc::new(result_bit_widths.finish());
        
        // Create schema from actual array types
        let result_schema = Schema::new(vec![
            Field::new("layer_name", layer_names_array.data_type().clone(), false),
            Field::new("quantized_data", quantized_data_array.data_type().clone(), false),
            Field::new("scales", scales_array.data_type().clone(), false),
            Field::new("zero_points", zero_points_array.data_type().clone(), false),
            Field::new("shape", shapes_array.data_type().clone(), false),
            Field::new("bit_width", bit_widths_array.data_type().clone(), false),
        ]);

        // Create empty RecordBatch
        let result_batch = RecordBatch::try_new(
            Arc::new(result_schema),
            vec![
                layer_names_array,
                quantized_data_array,
                scales_array,
                zero_points_array,
                shapes_array,
                bit_widths_array,
            ],
        );

        if let Err(e) = &result_batch {
            panic!("Failed to create empty RecordBatch: {:?}", e);
        }
        let batch = result_batch.unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 6);
    }

    #[test]
    fn test_result_multiple_scales_zero_points() {
        // Test building result with multiple scales and zero_points per layer
        let mut result_scales = ListBuilder::new(Float32Builder::new());
        let mut result_zero_points = ListBuilder::new(Float32Builder::new());

        // Append multiple values for one layer
        result_scales.values().append_value(0.5);
        result_scales.values().append_value(0.6);
        result_scales.values().append_value(0.7);
        result_scales.append(true);
        
        result_zero_points.values().append_value(0.0);
        result_zero_points.values().append_value(0.1);
        result_zero_points.values().append_value(0.2);
        result_zero_points.append(true);

        let scales_array = result_scales.finish();
        let zero_points_array = result_zero_points.finish();

        // Verify arrays
        assert_eq!(scales_array.len(), 1);
        assert_eq!(zero_points_array.len(), 1);
        
        // Verify inner values count
        assert_eq!(scales_array.value(0).len(), 3);
        assert_eq!(zero_points_array.value(0).len(), 3);
    }

    #[test]
    fn test_result_multidimensional_shape() {
        // Test building result with multidimensional shapes
        let mut result_shapes = ListBuilder::new(Int64Builder::new());

        // Append 2D shape
        result_shapes.values().append_value(64);
        result_shapes.values().append_value(128);
        result_shapes.append(true);

        // Append 3D shape
        result_shapes.values().append_value(32);
        result_shapes.values().append_value(64);
        result_shapes.values().append_value(128);
        result_shapes.append(true);

        let shapes_array = result_shapes.finish();

        // Verify arrays
        assert_eq!(shapes_array.len(), 2);
        assert_eq!(shapes_array.value(0).len(), 2);
        assert_eq!(shapes_array.value(1).len(), 3);
    }

    #[test]
    fn test_result_bit_width_values() {
        // Test that we can store different bit_width values
        let mut result_bit_widths = UInt8Builder::new();

        for bit_width in [2u8, 4, 8] {
            result_bit_widths.append_value(bit_width);
        }

        let bit_widths_array = result_bit_widths.finish();
        assert_eq!(bit_widths_array.len(), 3);
        assert_eq!(bit_widths_array.value(0), 2);
        assert_eq!(bit_widths_array.value(1), 4);
        assert_eq!(bit_widths_array.value(2), 8);
    }
}

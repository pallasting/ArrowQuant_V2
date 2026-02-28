//! Unit tests for validate_quantization_schema function
//! 
//! These tests verify the schema validation logic in isolation without
//! requiring Python bindings or PyArrow integration.

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    // Mock PyResult for testing
    type PyResult<T> = Result<T, String>;

    /// Simplified version of validate_quantization_schema for unit testing
    fn validate_quantization_schema(schema: &Schema) -> PyResult<()> {
        // Check for required fields
        let layer_name_field = schema.field_with_name("layer_name")
            .map_err(|_| {
                "Missing required field 'layer_name' in Arrow schema. \
                Expected schema: {layer_name: string, weights: list<float32>, shape: list<int64>}".to_string()
            })?;
        
        let weights_field = schema.field_with_name("weights")
            .map_err(|_| {
                "Missing required field 'weights' in Arrow schema. \
                Expected schema: {layer_name: string, weights: list<float32>, shape: list<int64>}".to_string()
            })?;
        
        // Validate field types
        if !matches!(layer_name_field.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
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

    #[test]
    fn test_valid_schema_with_all_fields() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new("weights", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), false),
            Field::new("shape", DataType::List(Arc::new(Field::new("item", DataType::Int64, true))), false),
        ]);

        assert!(validate_quantization_schema(&schema).is_ok());
    }

    #[test]
    fn test_valid_schema_without_optional_shape() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new("weights", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), false),
        ]);

        assert!(validate_quantization_schema(&schema).is_ok());
    }

    #[test]
    fn test_missing_layer_name() {
        let schema = Schema::new(vec![
            Field::new("weights", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), false),
        ]);

        let result = validate_quantization_schema(&schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("layer_name"));
        assert!(err.contains("Missing"));
    }

    #[test]
    fn test_missing_weights() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
        ]);

        let result = validate_quantization_schema(&schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("weights"));
        assert!(err.contains("Missing"));
    }

    #[test]
    fn test_wrong_layer_name_type() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Int32, false),
            Field::new("weights", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), false),
        ]);

        let result = validate_quantization_schema(&schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("layer_name"));
        assert!(err.contains("Invalid type"));
    }

    #[test]
    fn test_wrong_weights_inner_type() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new("weights", DataType::List(Arc::new(Field::new("item", DataType::Int64, true))), false),
        ]);

        let result = validate_quantization_schema(&schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("weights"));
        assert!(err.contains("float32"));
    }

    #[test]
    fn test_wrong_shape_inner_type() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new("weights", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), false),
            Field::new("shape", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), false),
        ]);

        let result = validate_quantization_schema(&schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("shape"));
        assert!(err.contains("int64"));
    }

    #[test]
    fn test_large_utf8_accepted() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::LargeUtf8, false),
            Field::new("weights", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), false),
        ]);

        assert!(validate_quantization_schema(&schema).is_ok());
    }

    #[test]
    fn test_large_list_accepted() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new("weights", DataType::LargeList(Arc::new(Field::new("item", DataType::Float32, true))), false),
            Field::new("shape", DataType::LargeList(Arc::new(Field::new("item", DataType::Int64, true))), false),
        ]);

        assert!(validate_quantization_schema(&schema).is_ok());
    }

    #[test]
    fn test_extra_columns_allowed() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new("weights", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), false),
            Field::new("extra_field", DataType::Utf8, false),
            Field::new("another_field", DataType::Int32, false),
        ]);

        assert!(validate_quantization_schema(&schema).is_ok());
    }

    #[test]
    fn test_weights_not_list() {
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
    fn test_shape_not_list() {
        let schema = Schema::new(vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new("weights", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), false),
            Field::new("shape", DataType::Int64, false),
        ]);

        let result = validate_quantization_schema(&schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("shape"));
        assert!(err.contains("list"));
    }
}

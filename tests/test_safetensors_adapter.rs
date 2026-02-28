//! Tests for SafeTensors adapter

use arrow_quant_v2::safetensors_adapter::{SafeTensorsAdapter, SafeTensorsHeader, TensorInfo};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use tempfile::NamedTempFile;

/// Create a test SafeTensors file with sample data
fn create_test_safetensors() -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    
    // Create header
    let mut tensors = HashMap::new();
    tensors.insert(
        "layer.weight".to_string(),
        TensorInfo {
            dtype: "F32".to_string(),
            shape: vec![2, 3],
            data_offsets: (0, 24), // 6 floats * 4 bytes
        },
    );
    tensors.insert(
        "layer.bias".to_string(),
        TensorInfo {
            dtype: "F32".to_string(),
            shape: vec![2],
            data_offsets: (24, 32), // 2 floats * 4 bytes
        },
    );
    
    let mut metadata = HashMap::new();
    metadata.insert("architecture".to_string(), "diffusion-text".to_string());
    metadata.insert("modality".to_string(), "text".to_string());
    
    let header = SafeTensorsHeader {
        tensors,
        metadata: Some(metadata),
    };
    
    let header_json = serde_json::to_vec(&header).unwrap();
    let header_size = header_json.len() as u64;
    
    // Write header size (8 bytes, little-endian)
    file.write_all(&header_size.to_le_bytes()).unwrap();
    
    // Write header JSON
    file.write_all(&header_json).unwrap();
    
    // Write tensor data
    // layer.weight: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for &val in &weight_data {
        file.write_all(&val.to_le_bytes()).unwrap();
    }
    
    // layer.bias: [0.1, 0.2]
    let bias_data: Vec<f32> = vec![0.1, 0.2];
    for &val in &bias_data {
        file.write_all(&val.to_le_bytes()).unwrap();
    }
    
    file.flush().unwrap();
    file
}

#[test]
fn test_load_safetensors() {
    let file = create_test_safetensors();
    let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
    
    // Check tensor names
    let names = adapter.tensor_names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"layer.weight".to_string()));
    assert!(names.contains(&"layer.bias".to_string()));
}

#[test]
fn test_get_metadata() {
    let file = create_test_safetensors();
    let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
    
    let metadata = adapter.get_metadata().unwrap();
    assert_eq!(metadata.get("architecture"), Some(&"diffusion-text".to_string()));
    assert_eq!(metadata.get("modality"), Some(&"text".to_string()));
}

#[test]
fn test_get_tensor_info() {
    let file = create_test_safetensors();
    let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
    
    let info = adapter.get_tensor_info("layer.weight").unwrap();
    assert_eq!(info.dtype, "F32");
    assert_eq!(info.shape, vec![2, 3]);
    assert_eq!(info.data_offsets, (0, 24));
}

#[test]
fn test_get_tensor_f32() {
    let file = create_test_safetensors();
    let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
    
    let tensor = adapter.get_tensor_f32("layer.weight").unwrap();
    assert_eq!(tensor.shape(), &[2, 3]);
    
    let values = tensor.into_raw_vec();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_get_all_tensors_f32() {
    let file = create_test_safetensors();
    let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
    
    let tensors = adapter.get_all_tensors_f32().unwrap();
    assert_eq!(tensors.len(), 2);
    
    assert_eq!(tensors["layer.weight"], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(tensors["layer.bias"], vec![0.1, 0.2]);
}

#[test]
fn test_get_all_tensors_2d() {
    let file = create_test_safetensors();
    let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
    
    let tensors = adapter.get_all_tensors_2d().unwrap();
    assert_eq!(tensors.len(), 2);
    
    // layer.weight should be 2x3
    let weight = &tensors["layer.weight"];
    assert_eq!(weight.shape(), &[2, 3]);
    
    // layer.bias should be reshaped to 2x1
    let bias = &tensors["layer.bias"];
    assert_eq!(bias.shape(), &[2, 1]);
}

#[test]
fn test_detect_modality() {
    let file = create_test_safetensors();
    let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
    
    let modality = adapter.detect_modality();
    assert_eq!(modality, Some("text".to_string()));
}

#[test]
fn test_detect_modality_from_architecture() {
    let mut file = NamedTempFile::new().unwrap();
    
    // Create header with architecture but no explicit modality
    let mut tensors = HashMap::new();
    tensors.insert(
        "test".to_string(),
        TensorInfo {
            dtype: "F32".to_string(),
            shape: vec![1],
            data_offsets: (0, 4),
        },
    );
    
    let mut metadata = HashMap::new();
    metadata.insert("architecture".to_string(), "image-dit".to_string());
    
    let header = SafeTensorsHeader {
        tensors,
        metadata: Some(metadata),
    };
    
    let header_json = serde_json::to_vec(&header).unwrap();
    let header_size = header_json.len() as u64;
    
    file.write_all(&header_size.to_le_bytes()).unwrap();
    file.write_all(&header_json).unwrap();
    file.write_all(&1.0f32.to_le_bytes()).unwrap();
    file.flush().unwrap();
    
    let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
    let modality = adapter.detect_modality();
    assert_eq!(modality, Some("image".to_string()));
}

#[test]
fn test_missing_tensor() {
    let file = create_test_safetensors();
    let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
    
    let result = adapter.get_tensor_f32("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_f16_conversion() {
    let mut file = NamedTempFile::new().unwrap();
    
    // Create header with F16 tensor
    let mut tensors = HashMap::new();
    tensors.insert(
        "f16_tensor".to_string(),
        TensorInfo {
            dtype: "F16".to_string(),
            shape: vec![2],
            data_offsets: (0, 4), // 2 f16 values * 2 bytes
        },
    );
    
    let header = SafeTensorsHeader {
        tensors,
        metadata: None,
    };
    
    let header_json = serde_json::to_vec(&header).unwrap();
    let header_size = header_json.len() as u64;
    
    file.write_all(&header_size.to_le_bytes()).unwrap();
    file.write_all(&header_json).unwrap();
    
    // Write F16 data (1.0 and 2.0 in f16 format)
    let f16_1 = half::f16::from_f32(1.0);
    let f16_2 = half::f16::from_f32(2.0);
    file.write_all(&f16_1.to_bits().to_le_bytes()).unwrap();
    file.write_all(&f16_2.to_bits().to_le_bytes()).unwrap();
    file.flush().unwrap();
    
    let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
    let tensor = adapter.get_tensor_f32("f16_tensor").unwrap();
    
    let values = tensor.into_raw_vec();
    assert!((values[0] - 1.0).abs() < 0.001);
    assert!((values[1] - 2.0).abs() < 0.001);
}

#[test]
fn test_bf16_conversion() {
    let mut file = NamedTempFile::new().unwrap();
    
    // Create header with BF16 tensor
    let mut tensors = HashMap::new();
    tensors.insert(
        "bf16_tensor".to_string(),
        TensorInfo {
            dtype: "BF16".to_string(),
            shape: vec![2],
            data_offsets: (0, 4), // 2 bf16 values * 2 bytes
        },
    );
    
    let header = SafeTensorsHeader {
        tensors,
        metadata: None,
    };
    
    let header_json = serde_json::to_vec(&header).unwrap();
    let header_size = header_json.len() as u64;
    
    file.write_all(&header_size.to_le_bytes()).unwrap();
    file.write_all(&header_json).unwrap();
    
    // Write BF16 data (1.0 and 2.0 in bf16 format)
    // BF16 is just F32 with lower 16 bits truncated
    let bf16_1 = (1.0f32.to_bits() >> 16) as u16;
    let bf16_2 = (2.0f32.to_bits() >> 16) as u16;
    file.write_all(&bf16_1.to_le_bytes()).unwrap();
    file.write_all(&bf16_2.to_le_bytes()).unwrap();
    file.flush().unwrap();
    
    let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
    let tensor = adapter.get_tensor_f32("bf16_tensor").unwrap();
    
    let values = tensor.into_raw_vec();
    assert!((values[0] - 1.0).abs() < 0.01); // BF16 has lower precision
    assert!((values[1] - 2.0).abs() < 0.01);
}

#[test]
fn test_empty_metadata() {
    let mut file = NamedTempFile::new().unwrap();
    
    // Create header without metadata
    let mut tensors = HashMap::new();
    tensors.insert(
        "test".to_string(),
        TensorInfo {
            dtype: "F32".to_string(),
            shape: vec![1],
            data_offsets: (0, 4),
        },
    );
    
    let header = SafeTensorsHeader {
        tensors,
        metadata: None,
    };
    
    let header_json = serde_json::to_vec(&header).unwrap();
    let header_size = header_json.len() as u64;
    
    file.write_all(&header_size.to_le_bytes()).unwrap();
    file.write_all(&header_json).unwrap();
    file.write_all(&1.0f32.to_le_bytes()).unwrap();
    file.flush().unwrap();
    
    let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
    assert!(adapter.get_metadata().is_none());
    assert_eq!(adapter.detect_modality(), None);
}

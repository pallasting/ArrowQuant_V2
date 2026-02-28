//! SafeTensors to Parquet V2 Extended converter
//!
//! This module provides utilities to convert SafeTensors models to Parquet V2 Extended format,
//! enabling direct quantization of SafeTensors models.

use crate::config::Modality;
use crate::errors::{QuantError, Result};
use crate::safetensors_adapter::SafeTensorsAdapter;
use crate::schema::ParquetV2Extended;
use crate::sharded_safetensors::ShardedSafeTensorsAdapter;
use std::path::Path;

/// Convert SafeTensors model to Parquet V2 Extended format
///
/// This function handles both single-file and sharded SafeTensors models,
/// converting them to the Parquet V2 Extended format required by the quantization system.
///
/// # Arguments
///
/// * `safetensors_path` - Path to SafeTensors file, index file, or directory
/// * `output_path` - Output directory for Parquet files
/// * `modality` - Optional modality (auto-detected if None)
///
/// # Returns
///
/// Returns the detected modality
///
/// # Examples
///
/// ```no_run
/// use arrow_quant_v2::safetensors_to_parquet::convert_safetensors_to_parquet;
/// use std::path::Path;
///
/// let modality = convert_safetensors_to_parquet(
///     Path::new("model.safetensors"),
///     Path::new("model_parquet/"),
///     None
/// ).unwrap();
/// ```
pub fn convert_safetensors_to_parquet(
    safetensors_path: &Path,
    output_path: &Path,
    modality: Option<Modality>,
) -> Result<Modality> {
    // Create output directory
    std::fs::create_dir_all(output_path)?;

    // Check if sharded model
    if crate::sharded_safetensors::is_sharded_model(safetensors_path) {
        convert_sharded_safetensors_to_parquet(safetensors_path, output_path, modality)
    } else {
        convert_single_safetensors_to_parquet(safetensors_path, output_path, modality)
    }
}

/// Convert single-file SafeTensors to Parquet
fn convert_single_safetensors_to_parquet(
    safetensors_path: &Path,
    output_path: &Path,
    modality: Option<Modality>,
) -> Result<Modality> {
    eprintln!("Converting single-file SafeTensors to Parquet...");

    // Load SafeTensors adapter
    let adapter = SafeTensorsAdapter::load(safetensors_path)?;

    // Detect modality
    let detected_modality = if let Some(m) = modality {
        m
    } else {
        detect_modality_from_adapter(&adapter)?
    };

    // Get all tensor names
    let tensor_names = adapter.tensor_names();
    eprintln!("Found {} tensors", tensor_names.len());

    // Convert each tensor to Parquet
    for (idx, tensor_name) in tensor_names.iter().enumerate() {
        if idx % 10 == 0 || idx == tensor_names.len() - 1 {
            eprintln!("Progress: {}/{} tensors converted", idx + 1, tensor_names.len());
        }

        // Extract tensor as f32 array
        let tensor_data = adapter.get_tensor_f32(tensor_name)?;
        
        // Write raw bytes to a companion .bin file to bypass Parquet's 2GB limitation
        let flat_data = tensor_data.clone().into_raw_vec();
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                flat_data.as_ptr() as *const u8,
                flat_data.len() * 4,
            ).to_vec()
        };
        let bin_file = output_path.join(format!("{}.bin", sanitize_filename(&tensor_name)));
        std::fs::write(&bin_file, data_bytes)?;

        // Convert to Parquet V2 Extended (schema only)
        let parquet_schema = convert_tensor_to_parquet_v2(
            tensor_name,
            tensor_data,
            detected_modality,
        )?;

        // Write to Parquet file
        let output_file = output_path.join(format!("{}.parquet", sanitize_filename(tensor_name)));
        parquet_schema.write_to_parquet(&output_file)?;
    }

    // Write metadata.json
    write_metadata_file(output_path, detected_modality, &adapter)?;

    eprintln!("Conversion complete: {} tensors", tensor_names.len());

    Ok(detected_modality)
}

/// Convert sharded SafeTensors to Parquet
fn convert_sharded_safetensors_to_parquet(
    safetensors_path: &Path,
    output_path: &Path,
    modality: Option<Modality>,
) -> Result<Modality> {
    eprintln!("Converting sharded SafeTensors to Parquet...");

    // Find index file
    let index_path = if safetensors_path.is_dir() {
        crate::sharded_safetensors::find_index_file(safetensors_path)?
    } else {
        safetensors_path.to_path_buf()
    };

    // Load sharded adapter
    let mut adapter = ShardedSafeTensorsAdapter::load(&index_path)?;

    // Detect modality
    let detected_modality = if let Some(m) = modality {
        m
    } else {
        detect_modality_from_sharded_adapter(&adapter)?
    };

    // Get all tensor names
    let tensor_names = adapter.tensor_names();
    eprintln!("Found {} tensors across {} shards", tensor_names.len(), adapter.num_shards());

    // Convert each tensor to Parquet
    for (idx, tensor_name) in tensor_names.iter().enumerate() {
        if idx % 10 == 0 || idx == tensor_names.len() - 1 {
            eprintln!("Progress: {}/{} tensors converted", idx + 1, tensor_names.len());
        }

        // Extract tensor as f32 array
        let tensor_data = adapter.get_tensor_f32(&tensor_name)?;
        
        // Write raw bytes to a companion .bin file to bypass Parquet's 2GB limitation
        let flat_data = tensor_data.clone().into_raw_vec();
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                flat_data.as_ptr() as *const u8,
                flat_data.len() * 4,
            ).to_vec()
        };
        let bin_file = output_path.join(format!("{}.bin", sanitize_filename(&tensor_name)));
        std::fs::write(&bin_file, data_bytes)?;

        // Convert to Parquet V2 Extended (schema only)
        let parquet_schema = convert_tensor_to_parquet_v2(
            &tensor_name,
            tensor_data,
            detected_modality,
        )?;

        // Write to Parquet file
        let output_file = output_path.join(format!("{}.parquet", sanitize_filename(&tensor_name)));
        parquet_schema.write_to_parquet(&output_file)?;
    }

    // Write metadata.json
    write_sharded_metadata_file(output_path, detected_modality, &adapter)?;

    eprintln!("Conversion complete: {} tensors from {} shards", tensor_names.len(), adapter.num_shards());

    Ok(detected_modality)
}

/// Convert a single tensor to Parquet V2 Extended format
fn convert_tensor_to_parquet_v2(
    tensor_name: &str,
    tensor_data: ndarray::ArrayD<f32>,
    modality: Modality,
) -> Result<ParquetV2Extended> {
    // Get tensor shape
    let shape = tensor_data.shape().to_vec();

    // For now, create a minimal ParquetV2Extended schema
    // The actual quantization will be applied later by the orchestrator
    // We leave data empty because writing FP32 blobs >2GB crashes Parquet.
    let schema = ParquetV2Extended::new_unquantized(
        tensor_name.to_string(),
        shape,
        modality,
        Vec::new(),
    );

    Ok(schema)
}

/// Detect modality from SafeTensors adapter
fn detect_modality_from_adapter(adapter: &SafeTensorsAdapter) -> Result<Modality> {
    if let Some(modality_str) = adapter.detect_modality() {
        parse_modality(&modality_str)
    } else {
        // Default to text if not detected
        eprintln!("Warning: Could not detect modality, defaulting to 'text'");
        Ok(Modality::Text)
    }
}

/// Detect modality from sharded SafeTensors adapter
fn detect_modality_from_sharded_adapter(adapter: &ShardedSafeTensorsAdapter) -> Result<Modality> {
    if let Some(modality_str) = adapter.detect_modality() {
        parse_modality(&modality_str)
    } else {
        // Default to text if not detected
        eprintln!("Warning: Could not detect modality, defaulting to 'text'");
        Ok(Modality::Text)
    }
}

/// Parse modality string to enum
fn parse_modality(modality_str: &str) -> Result<Modality> {
    match modality_str.to_lowercase().as_str() {
        "text" => Ok(Modality::Text),
        "code" => Ok(Modality::Code),
        "image" => Ok(Modality::Image),
        "audio" => Ok(Modality::Audio),
        _ => Err(QuantError::UnknownModality),
    }
}

/// Write metadata.json file
fn write_metadata_file(
    output_path: &Path,
    modality: Modality,
    adapter: &SafeTensorsAdapter,
) -> Result<()> {
    use serde_json::json;

    let metadata = json!({
        "modality": modality.to_string().to_lowercase(),
        "format": "parquet_v2_extended",
        "source": "safetensors",
        "architecture": adapter.detect_modality().unwrap_or_else(|| "unknown".to_string()),
    });

    let metadata_path = output_path.join("metadata.json");
    let metadata_str = serde_json::to_string_pretty(&metadata)?;
    std::fs::write(metadata_path, metadata_str)?;

    Ok(())
}

/// Write metadata.json file for sharded model
fn write_sharded_metadata_file(
    output_path: &Path,
    modality: Modality,
    adapter: &ShardedSafeTensorsAdapter,
) -> Result<()> {
    use serde_json::json;

    let metadata = json!({
        "modality": modality.to_string().to_lowercase(),
        "format": "parquet_v2_extended",
        "source": "safetensors_sharded",
        "num_shards": adapter.num_shards(),
        "architecture": adapter.detect_modality().unwrap_or_else(|| "unknown".to_string()),
    });

    let metadata_path = output_path.join("metadata.json");
    let metadata_str = serde_json::to_string_pretty(&metadata)?;
    std::fs::write(metadata_path, metadata_str)?;

    Ok(())
}

/// Sanitize filename for cross-platform compatibility
pub fn sanitize_filename(name: &str) -> String {
    name.replace('/', "_")
        .replace('\\', "_")
        .replace(':', "_")
        .replace('*', "_")
        .replace('?', "_")
        .replace('"', "_")
        .replace('<', "_")
        .replace('>', "_")
        .replace('|', "_")
        .replace('.', "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("model.layers.0.weight"), "model_layers_0_weight");
        assert_eq!(sanitize_filename("model/layers/0/weight"), "model_layers_0_weight");
        assert_eq!(sanitize_filename("model:layers:0:weight"), "model_layers_0_weight");
    }

    #[test]
    fn test_parse_modality() {
        assert!(matches!(parse_modality("text").unwrap(), Modality::Text));
        assert!(matches!(parse_modality("TEXT").unwrap(), Modality::Text));
        assert!(matches!(parse_modality("code").unwrap(), Modality::Code));
        assert!(matches!(parse_modality("image").unwrap(), Modality::Image));
        assert!(matches!(parse_modality("audio").unwrap(), Modality::Audio));
        assert!(parse_modality("unknown").is_err());
    }
}

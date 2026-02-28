//! SafeTensors Model Input Adapter
//!
//! This module provides conversion utilities to load models from SafeTensors format
//! and convert them to the internal format expected by ArrowQuant V2.
//!
//! SafeTensors is a simple, safe format for storing tensors safely (as opposed to pickle)
//! and fast (zero-copy). This adapter enables ArrowQuant V2 to quantize models stored
//! in SafeTensors format without requiring conversion to Parquet first.

use crate::errors::{QuantError, Result};
use ndarray::{Array2, ArrayD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// SafeTensors file header containing tensor metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorsHeader {
    /// Metadata about each tensor (name -> TensorInfo)
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorInfo>,
    
    /// Optional metadata (model type, architecture, etc.)
    #[serde(rename = "__metadata__", skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

/// Information about a single tensor in SafeTensors format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Data type (e.g., "F32", "F16", "BF16", "I32", "I64")
    pub dtype: String,
    
    /// Shape of the tensor
    pub shape: Vec<usize>,
    
    /// Byte offset in the data section
    pub data_offsets: (usize, usize),
}

/// Supported data types in SafeTensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeTensorsDType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U8,
}

impl SafeTensorsDType {
    /// Parse dtype string from SafeTensors header
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "I32" => Ok(Self::I32),
            "I64" => Ok(Self::I64),
            "U8" => Ok(Self::U8),
            _ => Err(QuantError::Internal(format!(
                "Unsupported SafeTensors dtype: {}. Supported types: F32, F16, BF16, I32, I64, U8",
                s
            ))),
        }
    }
    
    /// Get size in bytes for this dtype
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }
}

/// SafeTensors model loader and adapter
pub struct SafeTensorsAdapter {
    /// Path to SafeTensors file
    path: std::path::PathBuf,
    
    /// Parsed header
    header: SafeTensorsHeader,
    
    /// Raw data buffer (memory-mapped for zero-copy)
    data: Vec<u8>,
}

impl SafeTensorsAdapter {
    /// Load a SafeTensors file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to .safetensors file
    ///
    /// # Returns
    ///
    /// Adapter instance ready for tensor extraction
    ///
    /// # Example
    ///
    /// ```no_run
    /// use arrow_quant_v2::safetensors_adapter::SafeTensorsAdapter;
    ///
    /// let adapter = SafeTensorsAdapter::load("model.safetensors").unwrap();
    /// let weights = adapter.get_all_tensors_f32().unwrap();
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        
        // Open file
        let mut file = File::open(path).map_err(|e| {
            QuantError::Storage(format!("Failed to open SafeTensors file: {}", e))
        })?;
        
        // Read header size (first 8 bytes, little-endian u64)
        let mut header_size_bytes = [0u8; 8];
        file.read_exact(&mut header_size_bytes).map_err(|e| {
            QuantError::Storage(format!("Failed to read header size: {}", e))
        })?;
        let header_size = u64::from_le_bytes(header_size_bytes) as usize;
        
        // Read header JSON
        let mut header_bytes = vec![0u8; header_size];
        file.read_exact(&mut header_bytes).map_err(|e| {
            QuantError::Storage(format!("Failed to read header: {}", e))
        })?;
        
        let header: SafeTensorsHeader = serde_json::from_slice(&header_bytes).map_err(|e| {
            QuantError::Storage(format!("Failed to parse SafeTensors header: {}", e))
        })?;
        
        // Read remaining data
        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(|e| {
            QuantError::Storage(format!("Failed to read tensor data: {}", e))
        })?;
        
        Ok(Self {
            path: path.to_path_buf(),
            header,
            data,
        })
    }
    
    /// Get model metadata (architecture, modality, etc.)
    pub fn get_metadata(&self) -> Option<&HashMap<String, String>> {
        self.header.metadata.as_ref()
    }
    
    /// Get list of all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.header.tensors.keys().cloned().collect()
    }
    
    /// Get tensor info by name
    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.header.tensors.get(name)
    }
    
    /// Extract a single tensor as f32 array
    ///
    /// Automatically converts from F16/BF16/I32/I64 to F32 if needed.
    pub fn get_tensor_f32(&self, name: &str) -> Result<ArrayD<f32>> {
        let info = self.header.tensors.get(name).ok_or_else(|| {
            QuantError::Internal(format!("Tensor not found: {}", name))
        })?;
        
        let dtype = SafeTensorsDType::from_str(&info.dtype)?;
        let (start, end) = info.data_offsets;
        
        // Validate offsets
        if end > self.data.len() {
            return Err(QuantError::Internal(format!(
                "Invalid data offset for tensor {}: end={} > data_len={}",
                name, end, self.data.len()
            )));
        }
        
        let tensor_data = &self.data[start..end];
        
        // Convert to f32 based on dtype
        let values = match dtype {
            SafeTensorsDType::F32 => {
                // Direct copy (already f32)
                self.read_f32_slice(tensor_data)
            }
            SafeTensorsDType::F16 => {
                // Convert f16 to f32
                self.read_f16_to_f32(tensor_data)
            }
            SafeTensorsDType::BF16 => {
                // Convert bf16 to f32
                self.read_bf16_to_f32(tensor_data)
            }
            SafeTensorsDType::I32 => {
                // Convert i32 to f32
                self.read_i32_to_f32(tensor_data)
            }
            SafeTensorsDType::I64 => {
                // Convert i64 to f32
                self.read_i64_to_f32(tensor_data)
            }
            SafeTensorsDType::U8 => {
                // Convert u8 to f32
                self.read_u8_to_f32(tensor_data)
            }
        };
        
        // Create ndarray with correct shape
        let array = ArrayD::from_shape_vec(info.shape.clone(), values).map_err(|e| {
            QuantError::Internal(format!("Failed to create array for {}: {}", name, e))
        })?;
        
        Ok(array)
    }
    
    /// Extract all tensors as f32 HashMap
    ///
    /// Returns a map of tensor_name -> flattened f32 vector.
    /// This is the format expected by ArrowQuant V2 quantization pipeline.
    pub fn get_all_tensors_f32(&self) -> Result<HashMap<String, Vec<f32>>> {
        let mut tensors = HashMap::new();
        
        for name in self.tensor_names() {
            let array = self.get_tensor_f32(&name)?;
            let flattened = array.into_raw_vec();
            tensors.insert(name, flattened);
        }
        
        Ok(tensors)
    }
    
    /// Extract all tensors as 2D arrays (for layer-wise quantization)
    ///
    /// Reshapes tensors to 2D format: [out_features, in_features]
    /// This is required for per-channel and per-group quantization.
    pub fn get_all_tensors_2d(&self) -> Result<HashMap<String, Array2<f32>>> {
        let mut tensors = HashMap::new();
        
        for name in self.tensor_names() {
            let array = self.get_tensor_f32(&name)?;
            
            // Reshape to 2D
            let shape = array.shape();
            let array_2d = if shape.len() == 2 {
                // Already 2D
                array.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
                    QuantError::Internal(format!("Failed to convert to 2D: {}", e))
                })?
            } else {
                // Flatten to 2D: [first_dim, product_of_rest]
                let first_dim = shape[0];
                let rest: usize = shape[1..].iter().product();
                
                let flattened = array.into_raw_vec();
                Array2::from_shape_vec((first_dim, rest), flattened).map_err(|e| {
                    QuantError::Internal(format!("Failed to reshape to 2D: {}", e))
                })?
            };
            
            tensors.insert(name, array_2d);
        }
        
        Ok(tensors)
    }
    
    /// Detect modality from metadata
    ///
    /// Looks for "modality" field in __metadata__ section.
    /// Falls back to heuristics based on model architecture,
    /// then to tensor-name pattern analysis.
    pub fn detect_modality(&self) -> Option<String> {
        if let Some(metadata) = &self.header.metadata {
            // Check explicit modality field
            if let Some(modality) = metadata.get("modality") {
                return Some(modality.clone());
            }
            
            // Heuristics based on architecture
            if let Some(arch) = metadata.get("architecture") {
                let arch_lower = arch.to_lowercase();
                
                if arch_lower.contains("dit") || arch_lower.contains("diffusion") {
                    if arch_lower.contains("text") || arch_lower.contains("mdlm") {
                        return Some("text".to_string());
                    } else if arch_lower.contains("image") || arch_lower.contains("vae") {
                        return Some("image".to_string());
                    } else if arch_lower.contains("audio") || arch_lower.contains("wavegrad") {
                        return Some("audio".to_string());
                    } else if arch_lower.contains("code") {
                        return Some("code".to_string());
                    }
                }
            }
        }
        
        // Fallback: Infer modality from tensor name patterns
        self.detect_modality_from_tensor_names()
    }

    /// Infer modality by scanning tensor names for architecture-specific patterns.
    fn detect_modality_from_tensor_names(&self) -> Option<String> {
        let names = self.tensor_names();
        if names.is_empty() {
            return None;
        }

        // Score each modality based on matching tensor name patterns
        let mut text_score: i32 = 0;
        let mut image_score: i32 = 0;
        let mut audio_score: i32 = 0;

        for name in &names {
            let n = name.to_lowercase();

            // Text / LLM patterns
            if n.contains("embed_tokens") || n.contains("lm_head") || n.contains("self_attn")
                || n.contains("mlp.gate") || n.contains("mlp.up") || n.contains("mlp.down")
                || n.contains("input_layernorm") || n.contains("post_attention_layernorm")
                || n.contains("rotary_emb") || n.contains("wte") || n.contains("wpe")
                || n.contains("transformer.h.") || n.contains("model.layers.")
            {
                text_score += 1;
            }

            // Image / Vision patterns
            if n.contains("conv_in") || n.contains("conv_out") || n.contains("down_blocks")
                || n.contains("up_blocks") || n.contains("mid_block") || n.contains("unet")
                || n.contains("vae.") || n.contains("patch_embed") || n.contains("to_rgb")
            {
                image_score += 1;
            }

            // Audio patterns
            if n.contains("mel") || n.contains("spectrogram") || n.contains("waveform")
                || n.contains("audio_encoder") || n.contains("vocoder")
            {
                audio_score += 1;
            }
        }

        // Pick highest score, with text as tiebreaker (most common)
        if text_score == 0 && image_score == 0 && audio_score == 0 {
            return None;
        }

        if image_score > text_score && image_score > audio_score {
            Some("image".to_string())
        } else if audio_score > text_score && audio_score > image_score {
            Some("audio".to_string())
        } else {
            Some("text".to_string())
        }
    }
    
    // Helper methods for dtype conversion
    
    fn read_f32_slice(&self, data: &[u8]) -> Vec<f32> {
        let count = data.len() / 4;
        let mut values = Vec::with_capacity(count);
        
        for i in 0..count {
            let bytes = [
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ];
            values.push(f32::from_le_bytes(bytes));
        }
        
        values
    }
    
    fn read_f16_to_f32(&self, data: &[u8]) -> Vec<f32> {
        let count = data.len() / 2;
        let mut values = Vec::with_capacity(count);
        
        for i in 0..count {
            let bytes = [data[i * 2], data[i * 2 + 1]];
            let f16_bits = u16::from_le_bytes(bytes);
            
            // Convert f16 to f32 using half crate
            let f16_val = half::f16::from_bits(f16_bits);
            values.push(f16_val.to_f32());
        }
        
        values
    }
    
    fn read_bf16_to_f32(&self, data: &[u8]) -> Vec<f32> {
        let count = data.len() / 2;
        let mut values = Vec::with_capacity(count);
        
        for i in 0..count {
            let bytes = [data[i * 2], data[i * 2 + 1]];
            let bf16_bits = u16::from_le_bytes(bytes);
            
            // Convert bf16 to f32: bf16 is just f32 with lower 16 bits truncated
            let f32_bits = (bf16_bits as u32) << 16;
            values.push(f32::from_bits(f32_bits));
        }
        
        values
    }
    
    fn read_i32_to_f32(&self, data: &[u8]) -> Vec<f32> {
        let count = data.len() / 4;
        let mut values = Vec::with_capacity(count);
        
        for i in 0..count {
            let bytes = [
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ];
            let i32_val = i32::from_le_bytes(bytes);
            values.push(i32_val as f32);
        }
        
        values
    }
    
    fn read_i64_to_f32(&self, data: &[u8]) -> Vec<f32> {
        let count = data.len() / 8;
        let mut values = Vec::with_capacity(count);
        
        for i in 0..count {
            let bytes = [
                data[i * 8],
                data[i * 8 + 1],
                data[i * 8 + 2],
                data[i * 8 + 3],
                data[i * 8 + 4],
                data[i * 8 + 5],
                data[i * 8 + 6],
                data[i * 8 + 7],
            ];
            let i64_val = i64::from_le_bytes(bytes);
            values.push(i64_val as f32);
        }
        
        values
    }
    
    fn read_u8_to_f32(&self, data: &[u8]) -> Vec<f32> {
        data.iter().map(|&x| x as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    fn create_test_safetensors() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        
        // Create header
        let header = SafeTensorsHeader {
            tensors: {
                let mut map = HashMap::new();
                map.insert(
                    "layer.weight".to_string(),
                    TensorInfo {
                        dtype: "F32".to_string(),
                        shape: vec![2, 3],
                        data_offsets: (0, 24), // 6 floats * 4 bytes
                    },
                );
                map
            },
            metadata: Some({
                let mut map = HashMap::new();
                map.insert("architecture".to_string(), "diffusion-text".to_string());
                map
            }),
        };
        
        let header_json = serde_json::to_vec(&header).unwrap();
        let header_size = header_json.len() as u64;
        
        // Write header size
        file.write_all(&header_size.to_le_bytes()).unwrap();
        
        // Write header
        file.write_all(&header_json).unwrap();
        
        // Write tensor data (6 f32 values)
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for &val in &data {
            file.write_all(&val.to_le_bytes()).unwrap();
        }
        
        file.flush().unwrap();
        file
    }
    
    #[test]
    fn test_load_safetensors() {
        let file = create_test_safetensors();
        let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
        
        assert_eq!(adapter.tensor_names().len(), 1);
        assert!(adapter.tensor_names().contains(&"layer.weight".to_string()));
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
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors["layer.weight"], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
    
    #[test]
    fn test_detect_modality() {
        let file = create_test_safetensors();
        let adapter = SafeTensorsAdapter::load(file.path()).unwrap();
        
        let modality = adapter.detect_modality();
        assert_eq!(modality, Some("text".to_string()));
    }
}

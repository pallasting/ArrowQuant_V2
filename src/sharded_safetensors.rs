//! Sharded SafeTensors Support
//!
//! This module provides support for loading sharded SafeTensors models,
//! which are split across multiple files for large models (>5GB).
//!
//! Sharded format:
//! - model.safetensors.index.json: Index file with shard mapping
//! - model-00001-of-00005.safetensors: Shard files
//!
//! Example index.json:
//! ```json
//! {
//!   "metadata": {
//!     "total_size": 28000000000
//!   },
//!   "weight_map": {
//!     "layer.0.weight": "model-00001-of-00005.safetensors",
//!     "layer.1.weight": "model-00002-of-00005.safetensors"
//!   }
//! }
//! ```

use crate::errors::{QuantError, Result};
use crate::safetensors_adapter::SafeTensorsAdapter;
use ndarray::{Array2, ArrayD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Sharded SafeTensors index metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardedIndex {
    /// Metadata about the sharded model
    #[serde(default)]
    pub metadata: ShardMetadata,
    
    /// Mapping of tensor names to shard files
    pub weight_map: HashMap<String, String>,
}

/// Metadata for sharded model
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardMetadata {
    /// Total size in bytes
    #[serde(default)]
    pub total_size: Option<u64>,
    
    /// Model architecture
    #[serde(default)]
    pub architecture: Option<String>,
    
    /// Model modality
    #[serde(default)]
    pub modality: Option<String>,
}

/// Sharded SafeTensors loader
///
/// Loads models split across multiple SafeTensors files.
pub struct ShardedSafeTensorsAdapter {
    /// Base directory containing shard files
    base_dir: PathBuf,
    
    /// Parsed index
    index: ShardedIndex,
    
    /// Cache of loaded shard adapters
    shard_cache: HashMap<String, SafeTensorsAdapter>,
}

impl ShardedSafeTensorsAdapter {
    /// Load a sharded SafeTensors model from index file
    ///
    /// # Arguments
    ///
    /// * `index_path` - Path to .safetensors.index.json file
    ///
    /// # Returns
    ///
    /// Adapter instance ready for tensor extraction
    ///
    /// # Example
    ///
    /// ```no_run
    /// use arrow_quant_v2::sharded_safetensors::ShardedSafeTensorsAdapter;
    ///
    /// let adapter = ShardedSafeTensorsAdapter::load(
    ///     "model.safetensors.index.json"
    /// ).unwrap();
    ///
    /// let weights = adapter.get_all_tensors_f32().unwrap();
    /// ```
    pub fn load<P: AsRef<Path>>(index_path: P) -> Result<Self> {
        let index_path = index_path.as_ref();
        
        // Read index file
        let index_content = std::fs::read_to_string(index_path).map_err(|e| {
            QuantError::Storage(format!("Failed to read index file: {}", e))
        })?;
        
        let index: ShardedIndex = serde_json::from_str(&index_content).map_err(|e| {
            QuantError::Storage(format!("Failed to parse index JSON: {}", e))
        })?;
        
        // Get base directory
        let base_dir = index_path.parent()
            .ok_or_else(|| QuantError::Storage("Invalid index path".to_string()))?
            .to_path_buf();
        
        Ok(Self {
            base_dir,
            index,
            shard_cache: HashMap::new(),
        })
    }
    
    /// Get list of all tensor names across all shards
    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.index.weight_map.keys().cloned().collect();
        // Sort by shard name so that tensors in the same shard are loaded consecutively
        names.sort_by(|a, b| {
            let shard_a = self.index.weight_map.get(a).unwrap();
            let shard_b = self.index.weight_map.get(b).unwrap();
            shard_a.cmp(shard_b).then_with(|| a.cmp(b))
        });
        names
    }
    
    /// Get shard file for a specific tensor
    pub fn get_shard_for_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.index.weight_map.get(tensor_name).map(|s| s.as_str())
    }
    
    /// Load a shard adapter (with caching of 1 shard at a time)
    fn load_shard(&mut self, shard_name: &str) -> Result<&SafeTensorsAdapter> {
        // Check cache first
        if !self.shard_cache.contains_key(shard_name) {
            // Free up memory from previous shards before loading a new one
            // This ensures we only keep one shard (~5GB) in memory at a time
            self.shard_cache.clear();
            
            let shard_path = self.base_dir.join(shard_name);
            let adapter = SafeTensorsAdapter::load(&shard_path)?;
            self.shard_cache.insert(shard_name.to_string(), adapter);
        }
        
        Ok(self.shard_cache.get(shard_name).unwrap())
    }
    
    /// Extract a single tensor as f32 array
    pub fn get_tensor_f32(&mut self, name: &str) -> Result<ArrayD<f32>> {
        // Find which shard contains this tensor
        let shard_name = self.get_shard_for_tensor(name)
            .ok_or_else(|| QuantError::Internal(format!("Tensor not found: {}", name)))?
            .to_string(); // Clone the string to avoid borrow issues
        
        // Load shard and extract tensor
        let shard = self.load_shard(&shard_name)?;
        shard.get_tensor_f32(name)
    }
    
    /// Extract all tensors as f32 HashMap
    ///
    /// Loads shards on-demand to minimize memory usage.
    pub fn get_all_tensors_f32(&mut self) -> Result<HashMap<String, Vec<f32>>> {
        let mut tensors = HashMap::new();
        
        for name in self.tensor_names() {
            let array = self.get_tensor_f32(&name)?;
            let flattened = array.into_raw_vec();
            tensors.insert(name, flattened);
        }
        
        Ok(tensors)
    }
    
    /// Extract all tensors as 2D arrays
    pub fn get_all_tensors_2d(&mut self) -> Result<HashMap<String, Array2<f32>>> {
        let mut tensors = HashMap::new();
        
        for name in self.tensor_names() {
            let array = self.get_tensor_f32(&name)?;
            
            // Reshape to 2D
            let shape = array.shape();
            let array_2d = if shape.len() == 2 {
                array.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
                    QuantError::Internal(format!("Failed to convert to 2D: {}", e))
                })?
            } else {
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
    
    /// Detect modality from index metadata, with tensor-name fallback
    pub fn detect_modality(&self) -> Option<String> {
        // Check explicit modality field
        if let Some(ref modality) = self.index.metadata.modality {
            return Some(modality.clone());
        }
        
        // Heuristics based on architecture
        if let Some(ref arch) = self.index.metadata.architecture {
            let arch_lower = arch.to_lowercase();
            
            if arch_lower.contains("text") || arch_lower.contains("mdlm") {
                return Some("text".to_string());
            } else if arch_lower.contains("image") || arch_lower.contains("dit") {
                return Some("image".to_string());
            } else if arch_lower.contains("audio") {
                return Some("audio".to_string());
            } else if arch_lower.contains("code") {
                return Some("code".to_string());
            }
        }
        
        // Fallback: Infer from weight_map tensor names
        let names = self.tensor_names();
        if names.is_empty() {
            return None;
        }

        let mut text_score: i32 = 0;
        let mut image_score: i32 = 0;
        let mut audio_score: i32 = 0;

        for name in &names {
            let n = name.to_lowercase();

            if n.contains("embed_tokens") || n.contains("lm_head") || n.contains("self_attn")
                || n.contains("mlp.gate") || n.contains("mlp.up") || n.contains("mlp.down")
                || n.contains("input_layernorm") || n.contains("post_attention_layernorm")
                || n.contains("rotary_emb") || n.contains("wte") || n.contains("wpe")
                || n.contains("transformer.h.") || n.contains("model.layers.")
            {
                text_score += 1;
            }

            if n.contains("conv_in") || n.contains("conv_out") || n.contains("down_blocks")
                || n.contains("up_blocks") || n.contains("mid_block") || n.contains("unet")
                || n.contains("vae.") || n.contains("patch_embed") || n.contains("to_rgb")
            {
                image_score += 1;
            }

            if n.contains("mel") || n.contains("spectrogram") || n.contains("waveform")
                || n.contains("audio_encoder") || n.contains("vocoder")
            {
                audio_score += 1;
            }
        }

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
    
    /// Get total model size in bytes
    pub fn get_total_size(&self) -> Option<u64> {
        self.index.metadata.total_size
    }
    
    /// Get number of shards
    pub fn num_shards(&self) -> usize {
        self.index.weight_map.values()
            .collect::<std::collections::HashSet<_>>()
            .len()
    }
    
    /// Get list of all shard files
    pub fn shard_files(&self) -> Vec<String> {
        self.index.weight_map.values()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }
    
    /// Clear shard cache to free memory
    pub fn clear_cache(&mut self) {
        self.shard_cache.clear();
    }
    
    /// Get memory usage of cached shards (approximate)
    pub fn cache_memory_usage(&self) -> usize {
        // Rough estimate: each cached shard uses ~100MB on average
        self.shard_cache.len() * 100 * 1024 * 1024
    }
}

/// Helper function to detect if a path is a sharded model
pub fn is_sharded_model<P: AsRef<Path>>(path: P) -> bool {
    let path = path.as_ref();
    
    // Check if it's an index file
    if path.is_file() {
        if let Some(name) = path.file_name() {
            let name_str = name.to_string_lossy();
            return name_str.ends_with(".safetensors.index.json");
        }
    }
    
    // Check if directory contains index file
    if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.ends_with(".safetensors.index.json") {
                    return true;
                }
            }
        }
    }
    
    false
}

/// Find index file in a directory
pub fn find_index_file<P: AsRef<Path>>(dir: P) -> Result<PathBuf> {
    let dir = dir.as_ref();
    
    for entry in std::fs::read_dir(dir).map_err(|e| {
        QuantError::Storage(format!("Failed to read directory: {}", e))
    })? {
        let entry = entry.map_err(|e| {
            QuantError::Storage(format!("Failed to read entry: {}", e))
        })?;
        
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        
        if name_str.ends_with(".safetensors.index.json") {
            return Ok(entry.path());
        }
    }
    
    Err(QuantError::Storage(
        "No index file found in directory".to_string()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;
    
    fn create_test_sharded_model() -> TempDir {
        let dir = TempDir::new().unwrap();
        
        // Create index file
        let index = ShardedIndex {
            metadata: ShardMetadata {
                total_size: Some(1000),
                architecture: Some("diffusion-text".to_string()),
                modality: Some("text".to_string()),
            },
            weight_map: {
                let mut map = HashMap::new();
                map.insert("layer.0.weight".to_string(), "shard-00001.safetensors".to_string());
                map.insert("layer.1.weight".to_string(), "shard-00002.safetensors".to_string());
                map
            },
        };
        
        let index_json = serde_json::to_string_pretty(&index).unwrap();
        let index_path = dir.path().join("model.safetensors.index.json");
        let mut file = File::create(&index_path).unwrap();
        file.write_all(index_json.as_bytes()).unwrap();
        
        // Create shard files (minimal valid SafeTensors)
        for shard_name in ["shard-00001.safetensors", "shard-00002.safetensors"] {
            let shard_path = dir.path().join(shard_name);
            let mut file = File::create(&shard_path).unwrap();
            
            // Write minimal SafeTensors header
            let header = serde_json::json!({});
            let header_bytes = serde_json::to_vec(&header).unwrap();
            let header_size = header_bytes.len() as u64;
            
            file.write_all(&header_size.to_le_bytes()).unwrap();
            file.write_all(&header_bytes).unwrap();
        }
        
        dir
    }
    
    #[test]
    fn test_load_sharded_index() {
        let dir = create_test_sharded_model();
        let index_path = dir.path().join("model.safetensors.index.json");
        
        let adapter = ShardedSafeTensorsAdapter::load(&index_path).unwrap();
        
        assert_eq!(adapter.tensor_names().len(), 2);
        assert_eq!(adapter.num_shards(), 2);
    }
    
    #[test]
    fn test_get_shard_for_tensor() {
        let dir = create_test_sharded_model();
        let index_path = dir.path().join("model.safetensors.index.json");
        
        let adapter = ShardedSafeTensorsAdapter::load(&index_path).unwrap();
        
        assert_eq!(
            adapter.get_shard_for_tensor("layer.0.weight"),
            Some("shard-00001.safetensors")
        );
        assert_eq!(
            adapter.get_shard_for_tensor("layer.1.weight"),
            Some("shard-00002.safetensors")
        );
    }
    
    #[test]
    fn test_detect_modality() {
        let dir = create_test_sharded_model();
        let index_path = dir.path().join("model.safetensors.index.json");
        
        let adapter = ShardedSafeTensorsAdapter::load(&index_path).unwrap();
        
        assert_eq!(adapter.detect_modality(), Some("text".to_string()));
    }
    
    #[test]
    fn test_is_sharded_model() {
        let dir = create_test_sharded_model();
        let index_path = dir.path().join("model.safetensors.index.json");
        
        assert!(is_sharded_model(&index_path));
        assert!(is_sharded_model(dir.path()));
    }
    
    #[test]
    fn test_find_index_file() {
        let dir = create_test_sharded_model();
        
        let index_path = find_index_file(dir.path()).unwrap();
        assert!(index_path.exists());
        assert!(index_path.to_string_lossy().ends_with(".safetensors.index.json"));
    }
    
    #[test]
    fn test_shard_files() {
        let dir = create_test_sharded_model();
        let index_path = dir.path().join("model.safetensors.index.json");
        
        let adapter = ShardedSafeTensorsAdapter::load(&index_path).unwrap();
        let shards = adapter.shard_files();
        
        assert_eq!(shards.len(), 2);
        assert!(shards.contains(&"shard-00001.safetensors".to_string()));
        assert!(shards.contains(&"shard-00002.safetensors".to_string()));
    }
}

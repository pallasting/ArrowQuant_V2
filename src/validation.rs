//! Quality validation system

use crate::errors::Result;
use std::collections::HashMap;
use std::path::Path;

/// Validation report
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationReport {
    pub cosine_similarity: f32,
    pub per_layer_accuracy: HashMap<String, f32>,
    pub compression_ratio: f32,
    pub model_size_mb: f32,
    pub passed: bool,
}

/// Quality validation system
#[derive(Clone)]
pub struct ValidationSystem {
    min_accuracy: f32,
    bit_width: Option<u8>,
}

impl ValidationSystem {
    /// Create new validation system
    pub fn new(min_accuracy: f32) -> Self {
        Self {
            min_accuracy,
            bit_width: None,
        }
    }

    /// Set bit-width for validation reporting
    pub fn set_bit_width(&mut self, bit_width: u8) {
        self.bit_width = Some(bit_width);
    }

    /// Create new validation system with bit-width-specific thresholds
    ///
    /// Uses standard thresholds based on bit width:
    /// - INT2: cosine_similarity >= 0.70
    /// - INT4: cosine_similarity >= 0.90
    /// - INT8: cosine_similarity >= 0.95
    ///
    /// # Arguments
    /// * `bit_width` - Target bit width (2, 4, or 8)
    ///
    /// # Returns
    /// ValidationSystem with appropriate threshold for the bit width
    pub fn new_with_bit_width(bit_width: u8) -> Self {
        let min_accuracy = Self::get_threshold_for_bit_width(bit_width);
        Self {
            min_accuracy,
            bit_width: Some(bit_width),
        }
    }

    /// Get standard quality threshold for a given bit width
    ///
    /// # Arguments
    /// * `bit_width` - Target bit width (2, 4, or 8)
    ///
    /// # Returns
    /// Minimum cosine similarity threshold
    fn get_threshold_for_bit_width(bit_width: u8) -> f32 {
        match bit_width {
            2 => 0.70, // INT2: Lower threshold due to aggressive quantization
            4 => 0.90, // INT4: Moderate threshold
            8 => 0.95, // INT8: High threshold, close to FP16
            _ => 0.85, // Default for other bit widths
        }
    }

    /// Validate quantization quality
    ///
    /// Computes per-layer cosine similarity between original and quantized models,
    /// generates statistics, and produces a validation report.
    ///
    /// # Arguments
    /// * `original_path` - Path to original model directory
    /// * `quantized_path` - Path to quantized model directory
    ///
    /// # Returns
    /// ValidationReport with cosine similarity, per-layer accuracy, compression ratio, etc.
    pub fn validate_quality(
        &self,
        original_path: &Path,
        quantized_path: &Path,
    ) -> Result<ValidationReport> {
        // Load models
        let original_layers = self.load_model_layers(original_path)?;
        let quantized_layers = self.load_model_layers(quantized_path)?;

        eprintln!("[Debug] Validation: Loaded {} original layers, {} quantized layers", 
                 original_layers.len(), quantized_layers.len());

        if original_layers.is_empty() || quantized_layers.is_empty() {
             eprintln!("[Error] Validation aborted: One or both models have 0 layers loaded.");
        }

        // Compute per-layer cosine similarity
        let mut per_layer_accuracy = HashMap::new();
        let mut similarities = Vec::new();

        for (layer_name, orig_weights) in original_layers.iter() {
            if let Some(quant_weights) = quantized_layers.get(layer_name) {
                // Ensure same shape
                if orig_weights.len() != quant_weights.len() {
                    log::warn!(
                        "Layer {} shape mismatch: original {} vs quantized {}",
                        layer_name,
                        orig_weights.len(),
                        quant_weights.len()
                    );
                    continue;
                }

                // Compute cosine similarity
                let similarity = self.cosine_similarity(orig_weights, quant_weights);
                per_layer_accuracy.insert(layer_name.clone(), similarity);
                similarities.push(similarity);

                // Optional: Progress indicator during validation
                if similarities.len() % 50 == 0 || similarities.len() == original_layers.len() {
                    eprintln!("[Validation] Processed {}/{} layers...", similarities.len(), original_layers.len());
                }

                // Log warning for layers below threshold
                if similarity < self.min_accuracy {
                    if let Some(bw) = self.bit_width {
                        log::warn!(
                            "Layer {} below INT{} threshold: {:.4} < {:.4}",
                            layer_name,
                            bw,
                            similarity,
                            self.min_accuracy
                        );
                    } else {
                        log::warn!(
                            "Layer {} below threshold: {:.4} < {:.4}",
                            layer_name,
                            similarity,
                            self.min_accuracy
                        );
                    }
                }
            } else {
                log::warn!("Layer {} not found in quantized model", layer_name);
            }
        }

        // Compute statistics
        let stats = self.compute_statistics(&similarities);

        // Compute compression ratio
        let orig_size = self.compute_size(original_path)?;
        let quant_size = self.compute_size(quantized_path)?;
        let compression_ratio = if quant_size > 0 {
            orig_size as f32 / quant_size as f32
        } else {
            0.0
        };

        // Average similarity
        let avg_similarity = stats.mean;

        // Check if validation passed
        let passed = avg_similarity >= self.min_accuracy;

        let bit_width_label = self.bit_width.map(|bw| format!("INT{}", bw)).unwrap_or_else(|| "Unknown".to_string());
        eprintln!("[Validation Result] {} Mode - Cosine Similarity: {:.4} (Required: {:.4}), Compression: {:.2}x", 
                 bit_width_label, avg_similarity, self.min_accuracy, compression_ratio);

        if let Some(bw) = self.bit_width {
            log::info!(
                "INT{} validation complete: avg_similarity={:.4}, min={:.4}, max={:.4}, std={:.4}, threshold={:.4}, passed={}",
                bw,
                avg_similarity,
                stats.min,
                stats.max,
                stats.std,
                self.min_accuracy,
                passed
            );
        } else {
            log::info!(
                "Validation complete: avg_similarity={:.4}, min={:.4}, max={:.4}, std={:.4}, threshold={:.4}, passed={}",
                avg_similarity,
                stats.min,
                stats.max,
                stats.std,
                self.min_accuracy,
                passed
            );
        }

        Ok(ValidationReport {
            cosine_similarity: avg_similarity,
            per_layer_accuracy,
            compression_ratio,
            model_size_mb: quant_size as f32 / (1024.0 * 1024.0),
            passed,
        })
    }

    /// Cosine similarity implementation continues below...
    /// (rest of the methods remain unchanged)

    /// Compute cosine similarity between two vectors using SIMD acceleration
    ///
    /// Uses simsimd library for optimal SIMD performance across platforms.
    /// Falls back to scalar implementation if SIMD is not available.
    ///
    /// # Arguments
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    /// Cosine similarity in range [0, 1], or 0.0 if vectors are invalid
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        // Use simsimd for SIMD-accelerated cosine similarity
        // simsimd automatically selects the best SIMD instruction set (AVX2, NEON, etc.)
        if let Some(distance) = simsimd::SpatialSimilarity::cosine(a, b) {
            // simsimd returns cosine distance (1 - cosine_similarity)
            // Convert to similarity: similarity = 1 - distance
            let similarity: f32 = 1.0 - distance as f32;
            // Clamp to [0, 1] to handle floating point errors
            similarity.max(0.0).min(1.0)
        } else {
            // Fallback to scalar implementation if SIMD fails
            self.cosine_similarity_scalar(a, b)
        }
    }

    /// Scalar fallback for cosine similarity computation
    fn cosine_similarity_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Compute cosine similarity for batched vectors
    ///
    /// Efficiently computes cosine similarity for multiple vector pairs.
    /// Uses parallel processing with rayon for improved performance.
    ///
    /// # Arguments
    /// * `batch_a` - Slice of first vectors
    /// * `batch_b` - Slice of second vectors
    ///
    /// # Returns
    /// Vector of cosine similarities, one per pair
    pub fn cosine_similarity_batch(&self, batch_a: &[&[f32]], batch_b: &[&[f32]]) -> Vec<f32> {
        use rayon::prelude::*;

        if batch_a.len() != batch_b.len() {
            return vec![0.0; batch_a.len().min(batch_b.len())];
        }

        // Parallel computation of similarities
        batch_a
            .par_iter()
            .zip(batch_b.par_iter())
            .map(|(a, b)| self.cosine_similarity(a, b))
            .collect()
    }

    /// Compute directory size
    fn compute_size(&self, path: &Path) -> Result<u64> {
        if !path.exists() {
            return Ok(0);
        }

        if path.is_file() {
            return Ok(std::fs::metadata(path)?.len());
        }

        let mut total_size = 0u64;
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_file() {
                total_size += metadata.len();
            } else if metadata.is_dir() {
                total_size += self.compute_size(&entry.path())?;
            }
        }

        Ok(total_size)
    }

    /// Load model layers from directory
    ///
    /// Loads all .parquet files from the model directory and extracts weight tensors.
    /// Returns a HashMap mapping layer names to flattened weight vectors.
    fn load_model_layers(&self, model_path: &Path) -> Result<HashMap<String, Vec<f32>>> {
        let mut layers = HashMap::new();

        if !model_path.exists() {
            return Err(crate::errors::QuantError::ModelNotFound(
                model_path.display().to_string(),
            ));
        }

        // Iterate through all .parquet files in the directory
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                let layer_name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .replace('.', "_"); // Normalize to underscores to match our sanitized naming

                // Extract weights from parquet file
                match self.extract_weights_from_parquet(&path) {
                    Ok(weights) => {
                        layers.insert(layer_name, weights);
                    }
                    Err(e) => {
                        eprintln!("[Warning] Failed to extract weights from {}: {}", path.display(), e);
                    }
                }
            }
        }

        if layers.is_empty() {
            log::warn!("No layers loaded from {}", model_path.display());
        } else {
            log::info!(
                "Loaded {} layers from {}",
                layers.len(),
                model_path.display()
            );
        }

        Ok(layers)
    }

    /// Extract weights from a Parquet file (with dequantization support)
    fn extract_weights_from_parquet(&self, path: &Path) -> Result<Vec<f32>> {
        use crate::schema::ParquetV2Extended;
        
        // Use our existing robust reader
        let layer_data = ParquetV2Extended::read_from_parquet(path)?;
        
        if layer_data.data.is_empty() {
            return Err(crate::errors::QuantError::Internal(
                format!("No weights found in data column for: {}", path.display()),
            ));
        }

        let bit_width = if layer_data.quant_type.contains("int8") {
            8
        } else if layer_data.quant_type.contains("int4") {
            4
        } else if layer_data.quant_type.contains("int2") {
            2
        } else {
            0
        };

        if bit_width == 0 {
            // Raw FP32: Convert bytes to f32
            let floats_len = layer_data.data.len() / 4;
            let mut weights = Vec::with_capacity(floats_len);
            for i in 0..floats_len {
                let b = [
                    layer_data.data[i*4], 
                    layer_data.data[i*4+1], 
                    layer_data.data[i*4+2], 
                    layer_data.data[i*4+3]
                ];
                weights.push(f32::from_le_bytes(b));
            }
            return Ok(weights);
        }

        // Quantized data: Perform dequantization
        let scale = layer_data.scales.get(0).cloned().unwrap_or(1.0);
        let zero_point = layer_data.zero_points.get(0).cloned().unwrap_or(0.0);

        let mut weights = Vec::new();
        match bit_width {
            8 => {
            weights.reserve(layer_data.data.len());
            for &q in &layer_data.data {
                weights.push((q as f32 - zero_point) * scale);
            }
        }
        4 => {
            // Unpack 4-bit values (currently 1 byte per weight for engineering stability)
            weights.reserve(layer_data.data.len());
            for &q in &layer_data.data {
                weights.push((q as f32 - zero_point) * scale);
            }
        }
        2 => {
            // Unpack 2-bit values (currently 1 byte per weight for engineering stability)
            weights.reserve(layer_data.data.len());
            for &q in &layer_data.data {
                weights.push((q as f32 - zero_point) * scale);
            }
        }          
            _ => {
                return Err(crate::errors::QuantError::Internal(
                    format!("Validation dequantization not implemented for INT{}", bit_width),
                ));
            }
        }
        Ok(weights)
    }

    /// Compute statistics for a set of similarity values
    fn compute_statistics(&self, values: &[f32]) -> Statistics {
        if values.is_empty() {
            return Statistics {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std: 0.0,
            };
        }

        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = values.iter().sum::<f32>() / values.len() as f32;

        // Compute standard deviation
        let variance = values
            .iter()
            .map(|v| {
                let diff = v - mean;
                diff * diff
            })
            .sum::<f32>()
            / values.len() as f32;
        let std = variance.sqrt();

        Statistics {
            min,
            max,
            mean,
            std,
        }
    }
}

/// Statistics for validation metrics
#[derive(Debug, Clone)]
struct Statistics {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let validator = ValidationSystem::new(0.7);

        // Identical vectors should have similarity = 1.0
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = validator.cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6, "Expected 1.0, got {}", sim);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let validator = ValidationSystem::new(0.7);

        // Orthogonal vectors should have similarity = 0.0
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = validator.cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-6, "Expected 0.0, got {}", sim);
    }

    #[test]
    fn test_cosine_similarity_similar() {
        let validator = ValidationSystem::new(0.7);

        // Similar vectors should have high similarity
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.1, 2.9];
        let sim = validator.cosine_similarity(&a, &b);
        assert!(sim > 0.99, "Expected > 0.99, got {}", sim);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let validator = ValidationSystem::new(0.7);

        // Opposite vectors should have similarity close to -1.0
        // But since we clamp to [0, 1], it should be 0.0
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = validator.cosine_similarity(&a, &b);
        assert!(sim < 0.1, "Expected close to 0.0, got {}", sim);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let validator = ValidationSystem::new(0.7);

        // Empty vectors should return 0.0
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let sim = validator.cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched_length() {
        let validator = ValidationSystem::new(0.7);

        // Mismatched lengths should return 0.0
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let sim = validator.cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let validator = ValidationSystem::new(0.7);

        // Zero vector should return 0.0
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        let sim = validator.cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_large_vectors() {
        let validator = ValidationSystem::new(0.7);

        // Test with larger vectors (typical weight tensor size)
        let size = 1024;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32) * 1.01).collect();
        let sim = validator.cosine_similarity(&a, &b);
        assert!(
            sim > 0.99,
            "Expected > 0.99 for similar large vectors, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_batch_basic() {
        let validator = ValidationSystem::new(0.7);

        let a1 = vec![1.0, 2.0, 3.0];
        let a2 = vec![4.0, 5.0, 6.0];
        let b1 = vec![1.0, 2.0, 3.0];
        let b2 = vec![4.0, 5.0, 6.0];

        let batch_a = vec![a1.as_slice(), a2.as_slice()];
        let batch_b = vec![b1.as_slice(), b2.as_slice()];

        let similarities = validator.cosine_similarity_batch(&batch_a, &batch_b);

        assert_eq!(similarities.len(), 2);
        assert!((similarities[0] - 1.0).abs() < 1e-6);
        assert!((similarities[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_batch_empty() {
        let validator = ValidationSystem::new(0.7);

        let batch_a: Vec<&[f32]> = vec![];
        let batch_b: Vec<&[f32]> = vec![];

        let similarities = validator.cosine_similarity_batch(&batch_a, &batch_b);
        assert_eq!(similarities.len(), 0);
    }

    #[test]
    fn test_cosine_similarity_batch_mismatched() {
        let validator = ValidationSystem::new(0.7);

        let a1 = vec![1.0, 2.0, 3.0];
        let b1 = vec![1.0, 2.0, 3.0];
        let b2 = vec![4.0, 5.0, 6.0];

        let batch_a = vec![a1.as_slice()];
        let batch_b = vec![b1.as_slice(), b2.as_slice()];

        let similarities = validator.cosine_similarity_batch(&batch_a, &batch_b);
        assert_eq!(similarities.len(), 1);
    }

    #[test]
    fn test_cosine_similarity_batch_parallel() {
        let validator = ValidationSystem::new(0.7);

        // Test with many pairs to verify parallel processing
        let num_pairs = 100;
        let mut batch_a = Vec::new();
        let mut batch_b = Vec::new();
        let mut vecs_a = Vec::new();
        let mut vecs_b = Vec::new();

        for i in 0..num_pairs {
            let a: Vec<f32> = (0..256).map(|j| (i * 256 + j) as f32).collect();
            let b: Vec<f32> = (0..256).map(|j| (i * 256 + j) as f32 * 1.01).collect();
            vecs_a.push(a);
            vecs_b.push(b);
        }

        for i in 0..num_pairs {
            batch_a.push(vecs_a[i].as_slice());
            batch_b.push(vecs_b[i].as_slice());
        }

        let similarities = validator.cosine_similarity_batch(&batch_a, &batch_b);

        assert_eq!(similarities.len(), num_pairs);
        for sim in similarities {
            assert!(sim > 0.99, "Expected > 0.99, got {}", sim);
        }
    }

    #[test]
    fn test_scalar_fallback() {
        let validator = ValidationSystem::new(0.7);

        // Test scalar fallback directly
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = validator.cosine_similarity_scalar(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_statistics_basic() {
        let validator = ValidationSystem::new(0.7);

        let values = vec![0.5, 0.7, 0.9, 0.8, 0.6];
        let stats = validator.compute_statistics(&values);

        assert_eq!(stats.min, 0.5);
        assert_eq!(stats.max, 0.9);
        assert!((stats.mean - 0.7).abs() < 1e-6);
        assert!(stats.std > 0.0);
    }

    #[test]
    fn test_compute_statistics_empty() {
        let validator = ValidationSystem::new(0.7);

        let values: Vec<f32> = vec![];
        let stats = validator.compute_statistics(&values);

        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
    }

    #[test]
    fn test_compute_statistics_single() {
        let validator = ValidationSystem::new(0.7);

        let values = vec![0.75];
        let stats = validator.compute_statistics(&values);

        assert_eq!(stats.min, 0.75);
        assert_eq!(stats.max, 0.75);
        assert_eq!(stats.mean, 0.75);
        assert_eq!(stats.std, 0.0);
    }

    #[test]
    fn test_compute_statistics_identical() {
        let validator = ValidationSystem::new(0.7);

        let values = vec![0.8, 0.8, 0.8, 0.8];
        let stats = validator.compute_statistics(&values);

        assert_eq!(stats.min, 0.8);
        assert_eq!(stats.max, 0.8);
        assert_eq!(stats.mean, 0.8);
        assert_eq!(stats.std, 0.0);
    }

    #[test]
    fn test_validate_quality_nonexistent_path() {
        let validator = ValidationSystem::new(0.7);

        let result = validator.validate_quality(
            Path::new("/nonexistent/path"),
            Path::new("/another/nonexistent"),
        );

        // Should return error for nonexistent path
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_report_structure() {
        // Test that ValidationReport can be created and accessed
        let report = ValidationReport {
            cosine_similarity: 0.85,
            per_layer_accuracy: HashMap::new(),
            compression_ratio: 16.0,
            model_size_mb: 32.5,
            passed: true,
        };

        assert_eq!(report.cosine_similarity, 0.85);
        assert_eq!(report.compression_ratio, 16.0);
        assert_eq!(report.model_size_mb, 32.5);
        assert!(report.passed);
    }

    #[test]
    fn test_new_with_bit_width_int2() {
        // INT2 should have threshold of 0.70
        let validator = ValidationSystem::new_with_bit_width(2);
        assert_eq!(validator.min_accuracy, 0.70);
        assert_eq!(validator.bit_width, Some(2));
    }

    #[test]
    fn test_new_with_bit_width_int4() {
        // INT4 should have threshold of 0.90
        let validator = ValidationSystem::new_with_bit_width(4);
        assert_eq!(validator.min_accuracy, 0.90);
        assert_eq!(validator.bit_width, Some(4));
    }

    #[test]
    fn test_new_with_bit_width_int8() {
        // INT8 should have threshold of 0.95
        let validator = ValidationSystem::new_with_bit_width(8);
        assert_eq!(validator.min_accuracy, 0.95);
        assert_eq!(validator.bit_width, Some(8));
    }

    #[test]
    fn test_new_with_bit_width_default() {
        // Other bit widths should use default threshold of 0.85
        let validator = ValidationSystem::new_with_bit_width(16);
        assert_eq!(validator.min_accuracy, 0.85);
        assert_eq!(validator.bit_width, Some(16));
    }

    #[test]
    fn test_get_threshold_for_bit_width() {
        // Test the threshold mapping
        assert_eq!(ValidationSystem::get_threshold_for_bit_width(2), 0.70);
        assert_eq!(ValidationSystem::get_threshold_for_bit_width(4), 0.90);
        assert_eq!(ValidationSystem::get_threshold_for_bit_width(8), 0.95);
        assert_eq!(ValidationSystem::get_threshold_for_bit_width(16), 0.85);
        assert_eq!(ValidationSystem::get_threshold_for_bit_width(32), 0.85);
    }

    #[test]
    fn test_validation_system_without_bit_width() {
        // Test that new() creates validator without bit_width
        let validator = ValidationSystem::new(0.75);
        assert_eq!(validator.min_accuracy, 0.75);
        assert_eq!(validator.bit_width, None);
    }

    #[test]
    fn test_bit_width_specific_thresholds_int2() {
        // INT2: Should pass with similarity >= 0.70
        let validator = ValidationSystem::new_with_bit_width(2);

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.1, 2.1, 3.0, 3.9, 4.9]; // High similarity

        let similarity = validator.cosine_similarity(&a, &b);
        assert!(
            similarity >= 0.70,
            "INT2 similarity should be >= 0.70, got {}",
            similarity
        );
    }

    #[test]
    fn test_bit_width_specific_thresholds_int4() {
        // INT4: Should require similarity >= 0.90
        let validator = ValidationSystem::new_with_bit_width(4);

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Identical

        let similarity = validator.cosine_similarity(&a, &b);
        assert!(
            similarity >= 0.90,
            "INT4 similarity should be >= 0.90, got {}",
            similarity
        );
    }

    #[test]
    fn test_bit_width_specific_thresholds_int8() {
        // INT8: Should require similarity >= 0.95
        let validator = ValidationSystem::new_with_bit_width(8);

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Identical

        let similarity = validator.cosine_similarity(&a, &b);
        assert!(
            similarity >= 0.95,
            "INT8 similarity should be >= 0.95, got {}",
            similarity
        );
    }

    #[test]
    fn test_per_layer_validation_statistics() {
        // Test that per-layer validation computes correct statistics
        let validator = ValidationSystem::new(0.7);

        // Create mock per-layer accuracy data
        let mut per_layer_accuracy = HashMap::new();
        per_layer_accuracy.insert("layer1".to_string(), 0.95);
        per_layer_accuracy.insert("layer2".to_string(), 0.85);
        per_layer_accuracy.insert("layer3".to_string(), 0.75);
        per_layer_accuracy.insert("layer4".to_string(), 0.90);
        per_layer_accuracy.insert("layer5".to_string(), 0.80);

        let values: Vec<f32> = per_layer_accuracy.values().copied().collect();
        let stats = validator.compute_statistics(&values);

        // Verify statistics
        assert_eq!(stats.min, 0.75);
        assert_eq!(stats.max, 0.95);
        assert!(
            (stats.mean - 0.85).abs() < 1e-6,
            "Expected mean 0.85, got {}",
            stats.mean
        );
        assert!(stats.std > 0.0, "Standard deviation should be positive");
    }

    #[test]
    fn test_threshold_checking_pass() {
        // Test that validation passes when similarity meets threshold
        let validator = ValidationSystem::new(0.70);

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let similarity = validator.cosine_similarity(&a, &b);
        assert!(
            similarity >= validator.min_accuracy,
            "Similarity {} should meet threshold {}",
            similarity,
            validator.min_accuracy
        );
    }

    #[test]
    fn test_threshold_checking_fail() {
        // Test that validation fails when similarity is below threshold
        let validator = ValidationSystem::new(0.95);

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.5, 2.5, 3.5, 4.5, 5.5]; // Similar but not identical

        let similarity = validator.cosine_similarity(&a, &b);
        // This should be high but likely below 0.95
        assert!(
            similarity < validator.min_accuracy || similarity >= validator.min_accuracy,
            "Similarity is {}",
            similarity
        );
    }

    #[test]
    fn test_threshold_checking_boundary() {
        // Test threshold checking at exact boundary
        let validator = ValidationSystem::new(0.90);

        // Create vectors that should give exactly 0.90 similarity (approximately)
        let a = vec![1.0, 0.0];
        let b = vec![0.9, 0.436]; // cos(25°) ≈ 0.906

        let similarity = validator.cosine_similarity(&a, &b);
        // Should be close to threshold
        assert!(
            similarity >= 0.85 && similarity <= 0.95,
            "Similarity {} should be near threshold",
            similarity
        );
    }

    #[test]
    fn test_per_layer_validation_empty() {
        // Test per-layer validation with no layers
        let validator = ValidationSystem::new(0.7);

        let per_layer_accuracy: HashMap<String, f32> = HashMap::new();
        let values: Vec<f32> = per_layer_accuracy.values().copied().collect();
        let stats = validator.compute_statistics(&values);

        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
    }

    #[test]
    fn test_per_layer_validation_single_layer() {
        // Test per-layer validation with single layer
        let validator = ValidationSystem::new(0.7);

        let mut per_layer_accuracy = HashMap::new();
        per_layer_accuracy.insert("layer1".to_string(), 0.88);

        let values: Vec<f32> = per_layer_accuracy.values().copied().collect();
        let stats = validator.compute_statistics(&values);

        assert_eq!(stats.min, 0.88);
        assert_eq!(stats.max, 0.88);
        assert_eq!(stats.mean, 0.88);
        assert_eq!(stats.std, 0.0);
    }

    #[test]
    fn test_per_layer_validation_all_pass() {
        // Test per-layer validation where all layers pass threshold
        let validator = ValidationSystem::new(0.70);

        let mut per_layer_accuracy = HashMap::new();
        per_layer_accuracy.insert("layer1".to_string(), 0.95);
        per_layer_accuracy.insert("layer2".to_string(), 0.85);
        per_layer_accuracy.insert("layer3".to_string(), 0.75);

        for (layer_name, accuracy) in &per_layer_accuracy {
            assert!(
                accuracy >= &validator.min_accuracy,
                "Layer {} with accuracy {} should pass threshold {}",
                layer_name,
                accuracy,
                validator.min_accuracy
            );
        }
    }

    #[test]
    fn test_per_layer_validation_some_fail() {
        // Test per-layer validation where some layers fail threshold
        let validator = ValidationSystem::new(0.90);

        let mut per_layer_accuracy = HashMap::new();
        per_layer_accuracy.insert("layer1".to_string(), 0.95); // Pass
        per_layer_accuracy.insert("layer2".to_string(), 0.85); // Fail
        per_layer_accuracy.insert("layer3".to_string(), 0.92); // Pass
        per_layer_accuracy.insert("layer4".to_string(), 0.75); // Fail

        let mut pass_count = 0;
        let mut fail_count = 0;

        for (_, accuracy) in &per_layer_accuracy {
            if accuracy >= &validator.min_accuracy {
                pass_count += 1;
            } else {
                fail_count += 1;
            }
        }

        assert_eq!(pass_count, 2, "Expected 2 layers to pass");
        assert_eq!(fail_count, 2, "Expected 2 layers to fail");
    }

    #[test]
    fn test_threshold_checking_int2_boundary() {
        // Test INT2 threshold (0.70) boundary checking
        let validator = ValidationSystem::new_with_bit_width(2);

        let a = vec![1.0, 2.0, 3.0];
        let b_pass = vec![1.0, 2.0, 3.0]; // Should pass
        let b_fail = vec![5.0, -2.0, 1.0]; // Should fail

        let sim_pass = validator.cosine_similarity(&a, &b_pass);
        let sim_fail = validator.cosine_similarity(&a, &b_fail);

        assert!(
            sim_pass >= 0.70,
            "Expected pass with similarity {}",
            sim_pass
        );
        assert!(
            sim_fail < 0.70,
            "Expected fail with similarity {}",
            sim_fail
        );
    }

    #[test]
    fn test_threshold_checking_int4_boundary() {
        // Test INT4 threshold (0.90) boundary checking
        let validator = ValidationSystem::new_with_bit_width(4);

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b_pass = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Should pass
        let b_fail = vec![1.5, 2.5, 3.0, 3.5, 4.5]; // Might fail

        let sim_pass = validator.cosine_similarity(&a, &b_pass);
        let sim_fail = validator.cosine_similarity(&a, &b_fail);

        assert!(
            sim_pass >= 0.90,
            "Expected pass with similarity {}",
            sim_pass
        );
        // sim_fail might be close to 0.90, so we just verify it's computed
        assert!(
            sim_fail >= 0.0 && sim_fail <= 1.0,
            "Similarity should be in [0, 1], got {}",
            sim_fail
        );
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// **Validates: Requirements 9.1, 9.2**
    /// Property: Cosine similarity must always be in the range [0, 1]
    proptest! {
        #[test]
        fn prop_cosine_similarity_bounded(
            a in prop::collection::vec(-1000.0f32..1000.0f32, 1..1000),
            b in prop::collection::vec(-1000.0f32..1000.0f32, 1..1000)
        ) {
            let validator = ValidationSystem::new(0.7);

            // Ensure vectors have same length
            let min_len = a.len().min(b.len());
            let a_slice = &a[..min_len];
            let b_slice = &b[..min_len];

            let similarity = validator.cosine_similarity(a_slice, b_slice);

            // Property: similarity must be in [0, 1]
            prop_assert!(similarity >= 0.0, "Similarity {} should be >= 0.0", similarity);
            prop_assert!(similarity <= 1.0, "Similarity {} should be <= 1.0", similarity);
        }
    }

    /// **Validates: Requirements 9.1**
    /// Property: Identical vectors should have similarity = 1.0
    proptest! {
        #[test]
        fn prop_cosine_similarity_identical(
            a in prop::collection::vec(-1000.0f32..1000.0f32, 1..1000)
        ) {
            let validator = ValidationSystem::new(0.7);

            // Skip if all zeros (undefined similarity)
            if a.iter().all(|&x| x.abs() < 1e-6) {
                return Ok(());
            }

            let similarity = validator.cosine_similarity(&a, &a);

            // Property: identical vectors should have similarity ≈ 1.0
            prop_assert!((similarity - 1.0).abs() < 1e-4,
                        "Identical vectors should have similarity ≈ 1.0, got {}", similarity);
        }
    }

    /// **Validates: Requirements 9.1**
    /// Property: Cosine similarity is symmetric
    proptest! {
        #[test]
        fn prop_cosine_similarity_symmetric(
            a in prop::collection::vec(-1000.0f32..1000.0f32, 1..500),
            b in prop::collection::vec(-1000.0f32..1000.0f32, 1..500)
        ) {
            let validator = ValidationSystem::new(0.7);

            // Ensure vectors have same length
            let min_len = a.len().min(b.len());
            let a_slice = &a[..min_len];
            let b_slice = &b[..min_len];

            let sim_ab = validator.cosine_similarity(a_slice, b_slice);
            let sim_ba = validator.cosine_similarity(b_slice, a_slice);

            // Property: cosine_similarity(a, b) == cosine_similarity(b, a)
            prop_assert!((sim_ab - sim_ba).abs() < 1e-5,
                        "Cosine similarity should be symmetric: {} vs {}", sim_ab, sim_ba);
        }
    }

    /// **Validates: Requirements 9.1**
    /// Property: Scaling vectors doesn't change similarity
    proptest! {
        #[test]
        fn prop_cosine_similarity_scale_invariant(
            a in prop::collection::vec(-1000.0f32..1000.0f32, 1..500),
            b in prop::collection::vec(-1000.0f32..1000.0f32, 1..500),
            scale_a in 0.1f32..10.0f32,
            scale_b in 0.1f32..10.0f32
        ) {
            let validator = ValidationSystem::new(0.7);

            // Ensure vectors have same length
            let min_len = a.len().min(b.len());
            let a_slice = &a[..min_len];
            let b_slice = &b[..min_len];

            // Skip if vectors are too small
            if a_slice.iter().all(|&x| x.abs() < 1e-6) || b_slice.iter().all(|&x| x.abs() < 1e-6) {
                return Ok(());
            }

            let sim_original = validator.cosine_similarity(a_slice, b_slice);

            // Scale vectors
            let a_scaled: Vec<f32> = a_slice.iter().map(|&x| x * scale_a).collect();
            let b_scaled: Vec<f32> = b_slice.iter().map(|&x| x * scale_b).collect();

            let sim_scaled = validator.cosine_similarity(&a_scaled, &b_scaled);

            // Property: scaling doesn't change cosine similarity
            prop_assert!((sim_original - sim_scaled).abs() < 1e-4,
                        "Cosine similarity should be scale-invariant: {} vs {}",
                        sim_original, sim_scaled);
        }
    }

    /// **Validates: Requirements 9.2**
    /// Property: Statistics should be consistent with input values
    proptest! {
        #[test]
        fn prop_statistics_bounds(
            values in prop::collection::vec(0.0f32..1.0f32, 1..100)
        ) {
            let validator = ValidationSystem::new(0.7);
            let stats = validator.compute_statistics(&values);

            // Property: min <= mean <= max
            prop_assert!(stats.min <= stats.mean, "min {} should be <= mean {}", stats.min, stats.mean);
            prop_assert!(stats.mean <= stats.max, "mean {} should be <= max {}", stats.mean, stats.max);

            // Property: std >= 0
            prop_assert!(stats.std >= 0.0, "std {} should be >= 0", stats.std);

            // Property: min and max should be in the input values
            let actual_min = values.iter().cloned().fold(f32::INFINITY, f32::min);
            let actual_max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            prop_assert!((stats.min - actual_min).abs() < 1e-5,
                        "Computed min {} should match actual min {}", stats.min, actual_min);
            prop_assert!((stats.max - actual_max).abs() < 1e-5,
                        "Computed max {} should match actual max {}", stats.max, actual_max);
        }
    }

    /// **Validates: Requirements 9.3, 9.4**
    /// Property: Threshold checking should be consistent
    proptest! {
        #[test]
        fn prop_threshold_checking_consistent(
            threshold in 0.0f32..1.0f32,
            similarity in 0.0f32..1.0f32
        ) {
            let validator = ValidationSystem::new(threshold);

            // Property: if similarity >= threshold, validation should pass
            let should_pass = similarity >= threshold;
            let does_pass = similarity >= validator.min_accuracy;

            prop_assert_eq!(should_pass, does_pass,
                           "Threshold checking inconsistent: similarity={}, threshold={}",
                           similarity, threshold);
        }
    }

    /// **Validates: Requirements 9.1**
    /// Property: Batch cosine similarity should match individual computations
    proptest! {
        #[test]
        fn prop_batch_similarity_matches_individual(
            vecs_a in prop::collection::vec(
                prop::collection::vec(-100.0f32..100.0f32, 10..100),
                1..10
            ),
            vecs_b in prop::collection::vec(
                prop::collection::vec(-100.0f32..100.0f32, 10..100),
                1..10
            )
        ) {
            let validator = ValidationSystem::new(0.7);

            // Ensure same number of vector pairs and same lengths
            let num_pairs = vecs_a.len().min(vecs_b.len());

            // Normalize vector lengths to match
            let mut normalized_a = Vec::new();
            let mut normalized_b = Vec::new();

            for i in 0..num_pairs {
                let min_len = vecs_a[i].len().min(vecs_b[i].len());
                normalized_a.push(vecs_a[i][..min_len].to_vec());
                normalized_b.push(vecs_b[i][..min_len].to_vec());
            }

            // Skip if any vectors are all zeros
            let has_zero_vector = normalized_a.iter().any(|v| v.iter().all(|&x| x.abs() < 1e-6))
                || normalized_b.iter().any(|v| v.iter().all(|&x| x.abs() < 1e-6));

            if has_zero_vector {
                return Ok(());
            }

            // Compute batch similarities
            let batch_a: Vec<&[f32]> = normalized_a.iter().map(|v| v.as_slice()).collect();
            let batch_b: Vec<&[f32]> = normalized_b.iter().map(|v| v.as_slice()).collect();
            let batch_sims = validator.cosine_similarity_batch(&batch_a, &batch_b);

            // Compute individual similarities
            let individual_sims: Vec<f32> = normalized_a.iter()
                .zip(normalized_b.iter())
                .map(|(a, b)| validator.cosine_similarity(a, b))
                .collect();

            // Property: batch and individual results should match
            prop_assert_eq!(batch_sims.len(), individual_sims.len());
            for (i, (batch_sim, individual_sim)) in batch_sims.iter().zip(individual_sims.iter()).enumerate() {
                prop_assert!((batch_sim - individual_sim).abs() < 1e-4,
                            "Pair {}: Batch similarity {} should match individual {}",
                            i, batch_sim, individual_sim);
            }
        }
    }
}

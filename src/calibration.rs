//! Calibration Data Management
//!
//! Supports loading calibration data from multiple formats:
//! - JSONL (JSON Lines)
//! - Parquet
//! - HuggingFace Dataset format
//!
//! Validates Requirement 11: Calibration Data Management

use crate::errors::{QuantError, Result};
use arrow::array::{Array, ArrayRef, Float32Array};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Activation statistics for calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationStats {
    /// Mean values per timestep
    pub mean: Vec<f32>,
    /// Standard deviation per timestep
    pub std: Vec<f32>,
    /// Minimum values per timestep
    pub min: Vec<f32>,
    /// Maximum values per timestep
    pub max: Vec<f32>,
    /// Number of timesteps
    pub num_timesteps: usize,
}

impl ActivationStats {
    /// Create new activation statistics
    pub fn new(num_timesteps: usize) -> Self {
        Self {
            mean: vec![0.0; num_timesteps],
            std: vec![0.0; num_timesteps],
            min: vec![f32::INFINITY; num_timesteps],
            max: vec![f32::NEG_INFINITY; num_timesteps],
            num_timesteps,
        }
    }

    /// Compute statistics from calibration dataset
    pub fn from_dataset(dataset: &CalibrationDataset) -> Result<Self> {
        if dataset.is_empty() {
            return Err(QuantError::Internal(
                "Cannot compute statistics from empty dataset".to_string(),
            ));
        }

        // Determine number of timesteps
        let num_timesteps = dataset
            .get_samples()
            .iter()
            .filter_map(|s| s.timestep)
            .max()
            .unwrap_or(0)
            + 1;

        let mut stats = Self::new(num_timesteps);

        // Group samples by timestep
        let mut timestep_samples: Vec<Vec<&Vec<f32>>> = vec![Vec::new(); num_timesteps];
        for sample in dataset.get_samples() {
            if let Some(timestep) = sample.timestep {
                if timestep < num_timesteps {
                    timestep_samples[timestep].push(&sample.data);
                }
            }
        }

        // Compute statistics per timestep
        for (timestep, samples) in timestep_samples.iter().enumerate() {
            if samples.is_empty() {
                // Use default values for empty timesteps
                stats.mean[timestep] = 0.0;
                stats.std[timestep] = 1.0;
                stats.min[timestep] = -1.0;
                stats.max[timestep] = 1.0;
                continue;
            }

            // Flatten all samples for this timestep
            let mut all_values = Vec::new();
            for sample in samples {
                all_values.extend(*sample);
            }

            if all_values.is_empty() {
                // Use default values for empty data
                stats.mean[timestep] = 0.0;
                stats.std[timestep] = 1.0;
                stats.min[timestep] = -1.0;
                stats.max[timestep] = 1.0;
                continue;
            }

            // Compute mean
            let mean: f32 = all_values.iter().sum::<f32>() / all_values.len() as f32;
            stats.mean[timestep] = if mean.is_finite() { mean } else { 0.0 };

            // Compute std
            let variance: f32 = all_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                / all_values.len() as f32;
            let std = variance.sqrt();
            stats.std[timestep] = if std.is_finite() && std > 0.0 {
                std
            } else {
                1.0
            };

            // Compute min/max
            let min = all_values.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = all_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            stats.min[timestep] = if min.is_finite() { min } else { -1.0 };
            stats.max[timestep] = if max.is_finite() { max } else { 1.0 };
        }

        Ok(stats)
    }
}

/// Calibration cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    /// Cached calibration dataset
    dataset: CalibrationDataset,
    /// Cached activation statistics
    stats: ActivationStats,
    /// Cache metadata
    metadata: CacheMetadata,
}

/// Cache metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheMetadata {
    /// Hash of the source data
    source_hash: u64,
    /// Number of samples
    num_samples: usize,
    /// Sample shape
    sample_shape: Option<Vec<usize>>,
    /// Number of timesteps
    num_timesteps: usize,
    /// Creation timestamp
    created_at: u64,
}

/// Calibration data sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSample {
    /// Input data (flattened tensor)
    pub data: Vec<f32>,
    /// Optional timestep information for time-aware quantization
    pub timestep: Option<usize>,
    /// Optional shape information
    pub shape: Option<Vec<usize>>,
}

/// Calibration dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationDataset {
    /// Samples
    pub samples: Vec<CalibrationSample>,
    /// Number of samples
    pub num_samples: usize,
}

impl CalibrationDataset {
    /// Create a new empty calibration dataset
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            num_samples: 0,
        }
    }

    /// Add a sample to the dataset
    pub fn add_sample(&mut self, sample: CalibrationSample) {
        self.samples.push(sample);
        self.num_samples += 1;
    }

    /// Get a sample by index
    pub fn get_sample(&self, index: usize) -> Option<&CalibrationSample> {
        self.samples.get(index)
    }

    /// Get all samples
    pub fn get_samples(&self) -> &[CalibrationSample] {
        &self.samples
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.num_samples
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.num_samples == 0
    }
}

impl Default for CalibrationDataset {
    fn default() -> Self {
        Self::new()
    }
}

/// Calibration data loader
pub struct CalibrationLoader {
    /// Maximum number of samples to load
    max_samples: usize,
    /// Cache directory
    cache_dir: Option<PathBuf>,
    /// Enable caching
    enable_cache: bool,
}

impl CalibrationLoader {
    /// Create a new calibration loader
    ///
    /// # Arguments
    /// * `max_samples` - Maximum number of samples to load (default: 128)
    pub fn new(max_samples: usize) -> Self {
        Self {
            max_samples,
            cache_dir: None,
            enable_cache: false,
        }
    }

    /// Create a new calibration loader with caching enabled
    ///
    /// # Arguments
    /// * `max_samples` - Maximum number of samples to load
    /// * `cache_dir` - Directory to store cache files
    pub fn with_cache(max_samples: usize, cache_dir: PathBuf) -> Result<Self> {
        // Create cache directory if it doesn't exist
        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir)?;
        }

        Ok(Self {
            max_samples,
            cache_dir: Some(cache_dir),
            enable_cache: true,
        })
    }

    /// Get cache key for a given source
    fn get_cache_key(&self, source_hash: u64) -> String {
        format!("calibration_{:x}.json", source_hash)
    }

    /// Compute hash of source data
    fn compute_source_hash(&self, source: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        self.max_samples.hash(&mut hasher);
        hasher.finish()
    }

    /// Load from cache if available
    fn load_from_cache(&self, source_hash: u64) -> Result<Option<CacheEntry>> {
        if !self.enable_cache {
            return Ok(None);
        }

        let cache_dir = self
            .cache_dir
            .as_ref()
            .ok_or_else(|| QuantError::Internal("Cache directory not set".to_string()))?;

        let cache_key = self.get_cache_key(source_hash);
        let cache_path = cache_dir.join(cache_key);

        if !cache_path.exists() {
            return Ok(None);
        }

        // Read cache file
        let file = File::open(&cache_path)?;
        let entry: CacheEntry =
            serde_json::from_reader(file).map_err(|e| QuantError::SerdeError(e))?;

        // Validate cache entry
        if entry.metadata.source_hash != source_hash {
            return Ok(None);
        }

        Ok(Some(entry))
    }

    /// Save to cache
    fn save_to_cache(
        &self,
        source_hash: u64,
        dataset: &CalibrationDataset,
        stats: &ActivationStats,
        metadata: CacheMetadata,
    ) -> Result<()> {
        if !self.enable_cache {
            return Ok(());
        }

        let cache_dir = self
            .cache_dir
            .as_ref()
            .ok_or_else(|| QuantError::Internal("Cache directory not set".to_string()))?;

        let cache_key = self.get_cache_key(source_hash);
        let cache_path = cache_dir.join(cache_key);

        let entry = CacheEntry {
            dataset: dataset.clone(),
            stats: stats.clone(),
            metadata,
        };

        // Write cache file
        let file = File::create(&cache_path)?;
        serde_json::to_writer_pretty(file, &entry).map_err(|e| QuantError::SerdeError(e))?;

        Ok(())
    }

    /// Clear cache
    pub fn clear_cache(&self) -> Result<()> {
        if !self.enable_cache {
            return Ok(());
        }

        let cache_dir = self
            .cache_dir
            .as_ref()
            .ok_or_else(|| QuantError::Internal("Cache directory not set".to_string()))?;

        if cache_dir.exists() {
            for entry in std::fs::read_dir(cache_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
                    std::fs::remove_file(path)?;
                }
            }
        }

        Ok(())
    }

    /// Generate synthetic noise samples for diffusion models
    ///
    /// When calibration data is not provided, this method generates synthetic
    /// noise samples that simulate typical diffusion model inputs. The samples
    /// follow a Gaussian distribution with mean=0 and std=1.
    ///
    /// This method supports caching to avoid regenerating the same synthetic data.
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples to generate (32-1024)
    /// * `sample_shape` - Shape of each sample (e.g., [1, 512] for text, [4, 64, 64] for image)
    /// * `num_timesteps` - Number of diffusion timesteps (default: 1000)
    ///
    /// # Returns
    /// * `Result<(CalibrationDataset, ActivationStats)>` - Generated synthetic calibration dataset and statistics
    ///
    /// # Example
    /// ```
    /// use arrow_quant_v2::calibration::CalibrationLoader;
    ///
    /// let loader = CalibrationLoader::new(128);
    /// let (dataset, stats) = loader.generate_synthetic_data(128, vec![1, 512], 1000).unwrap();
    /// assert_eq!(dataset.len(), 128);
    /// ```
    pub fn generate_synthetic_data(
        &self,
        num_samples: usize,
        sample_shape: Vec<usize>,
        num_timesteps: usize,
    ) -> Result<(CalibrationDataset, ActivationStats)> {
        // Compute cache key
        let source = format!(
            "synthetic_{}_{:?}_{}",
            num_samples, sample_shape, num_timesteps
        );
        let source_hash = self.compute_source_hash(&source);

        // Try to load from cache
        if let Some(entry) = self.load_from_cache(source_hash)? {
            return Ok((entry.dataset, entry.stats));
        }

        // Generate new data
        let dataset = self.generate_synthetic_data_uncached(
            num_samples,
            sample_shape.clone(),
            num_timesteps,
        )?;

        // Compute statistics
        let stats = ActivationStats::from_dataset(&dataset)?;

        // Save to cache
        let metadata = CacheMetadata {
            source_hash,
            num_samples,
            sample_shape: Some(sample_shape),
            num_timesteps,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        self.save_to_cache(source_hash, &dataset, &stats, metadata)?;

        Ok((dataset, stats))
    }

    /// Generate synthetic data without caching (internal method)
    fn generate_synthetic_data_uncached(
        &self,
        num_samples: usize,
        sample_shape: Vec<usize>,
        num_timesteps: usize,
    ) -> Result<CalibrationDataset> {
        // Validate input parameters
        if num_samples < 32 || num_samples > 1024 {
            return Err(QuantError::Internal(format!(
                "num_samples must be in range [32, 1024], got {}",
                num_samples
            )));
        }

        if sample_shape.is_empty() {
            return Err(QuantError::Internal(
                "sample_shape cannot be empty".to_string(),
            ));
        }

        // Calculate total size per sample
        let sample_size: usize = sample_shape.iter().product();
        if sample_size == 0 {
            return Err(QuantError::Internal(
                "sample_shape dimensions must be positive".to_string(),
            ));
        }

        let mut dataset = CalibrationDataset::new();

        // Generate samples with varying noise levels across timesteps
        for i in 0..num_samples {
            // Distribute samples across timesteps
            let timestep = (i * num_timesteps) / num_samples;

            // Generate Gaussian noise: N(0, 1)
            // Use Box-Muller transform for generating normal distribution
            let data = self.generate_gaussian_noise(sample_size);

            let sample = CalibrationSample {
                data,
                timestep: Some(timestep),
                shape: Some(sample_shape.clone()),
            };

            dataset.add_sample(sample);
        }

        Ok(dataset)
    }

    /// Generate Gaussian noise using Box-Muller transform
    ///
    /// Generates random samples from a standard normal distribution N(0, 1)
    ///
    /// # Arguments
    /// * `size` - Number of samples to generate
    ///
    /// # Returns
    /// * `Vec<f32>` - Vector of Gaussian noise samples
    fn generate_gaussian_noise(&self, size: usize) -> Vec<f32> {
        use std::f32::consts::PI;

        let mut rng = fastrand::Rng::new();
        let mut noise = Vec::with_capacity(size);

        // Generate pairs of uniform random numbers and convert to Gaussian
        for _ in 0..(size / 2) {
            let u1: f32 = rng.f32();
            let u2: f32 = rng.f32();

            // Box-Muller transform
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;

            noise.push(r * theta.cos());
            noise.push(r * theta.sin());
        }

        // Handle odd size
        if size % 2 == 1 {
            let u1: f32 = rng.f32();
            let u2: f32 = rng.f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            noise.push(r * theta.cos());
        }

        noise
    }

    /// Load calibration data from a file
    ///
    /// Automatically detects format based on file extension:
    /// - `.jsonl` → JSONL format
    /// - `.parquet` → Parquet format
    /// - `.arrow` → Arrow IPC format (HuggingFace Dataset)
    ///
    /// This method supports caching to avoid reloading the same data.
    ///
    /// # Arguments
    /// * `path` - Path to calibration data file
    ///
    /// # Returns
    /// * `Result<(CalibrationDataset, ActivationStats)>` - Loaded calibration dataset and statistics
    pub fn load_from_file(&self, path: &Path) -> Result<(CalibrationDataset, ActivationStats)> {
        if !path.exists() {
            return Err(QuantError::ModelNotFound(path.display().to_string()));
        }

        // Compute cache key based on file path and modification time
        let metadata = std::fs::metadata(path)?;
        let modified = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let source = format!("{}_{}", path.display(), modified);
        let source_hash = self.compute_source_hash(&source);

        // Try to load from cache
        if let Some(entry) = self.load_from_cache(source_hash)? {
            return Ok((entry.dataset, entry.stats));
        }

        // Load from file
        let extension = path
            .extension()
            .and_then(|s| s.to_str())
            .ok_or_else(|| QuantError::Internal("Invalid file extension".to_string()))?;

        let dataset = match extension {
            "jsonl" => self.load_jsonl(path)?,
            "parquet" => self.load_parquet(path)?,
            "arrow" => self.load_arrow(path)?,
            _ => {
                return Err(QuantError::Internal(format!(
                    "Unsupported calibration data format: {}",
                    extension
                )))
            }
        };

        // Compute statistics
        let stats = ActivationStats::from_dataset(&dataset)?;

        // Save to cache
        let cache_metadata = CacheMetadata {
            source_hash,
            num_samples: dataset.len(),
            sample_shape: dataset.get_sample(0).and_then(|s| s.shape.clone()),
            num_timesteps: stats.num_timesteps,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        self.save_to_cache(source_hash, &dataset, &stats, cache_metadata)?;

        Ok((dataset, stats))
    }

    /// Load calibration data from JSONL format
    ///
    /// Expected format:
    /// ```json
    /// {"data": [0.1, 0.2, ...], "timestep": 0, "shape": [1, 512]}
    /// {"data": [0.3, 0.4, ...], "timestep": 1, "shape": [1, 512]}
    /// ```
    fn load_jsonl(&self, path: &Path) -> Result<CalibrationDataset> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut dataset = CalibrationDataset::new();

        for (idx, line) in reader.lines().enumerate() {
            if idx >= self.max_samples {
                break;
            }

            let line = line?;
            let sample: CalibrationSample =
                serde_json::from_str(&line).map_err(|e| QuantError::SerdeError(e))?;

            dataset.add_sample(sample);
        }

        if dataset.is_empty() {
            return Err(QuantError::Internal(
                "No calibration samples loaded from JSONL".to_string(),
            ));
        }

        Ok(dataset)
    }

    /// Load calibration data from Parquet format
    ///
    /// Expected schema:
    /// - `data`: List<Float32> - Flattened tensor data
    /// - `timestep`: Int64 (optional) - Timestep information
    /// - `shape`: List<Int64> (optional) - Shape information
    fn load_parquet(&self, path: &Path) -> Result<CalibrationDataset> {
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let mut reader = builder.build()?;

        let mut dataset = CalibrationDataset::new();
        let mut samples_loaded = 0;

        while let Some(batch) = reader.next() {
            let batch = batch?;
            samples_loaded += self.process_parquet_batch(&batch, &mut dataset)?;

            if samples_loaded >= self.max_samples {
                break;
            }
        }

        if dataset.is_empty() {
            return Err(QuantError::Internal(
                "No calibration samples loaded from Parquet".to_string(),
            ));
        }

        Ok(dataset)
    }

    /// Process a single Parquet record batch
    fn process_parquet_batch(
        &self,
        batch: &RecordBatch,
        dataset: &mut CalibrationDataset,
    ) -> Result<usize> {
        let num_rows = batch.num_rows();
        let mut samples_added = 0;

        // Get data column (required)
        let data_col = batch
            .column_by_name("data")
            .ok_or_else(|| QuantError::Internal("Missing 'data' column in Parquet".to_string()))?;

        // Get optional columns
        let timestep_col = batch.column_by_name("timestep");
        let shape_col = batch.column_by_name("shape");

        for row_idx in 0..num_rows {
            if dataset.len() >= self.max_samples {
                break;
            }

            // Extract data
            let data = self.extract_float_list(data_col, row_idx)?;

            // Extract optional timestep
            let timestep = timestep_col.and_then(|col| {
                if let Some(array) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
                    if !array.is_null(row_idx) {
                        return Some(array.value(row_idx) as usize);
                    }
                }
                None
            });

            // Extract optional shape
            let shape = shape_col.and_then(|col| self.extract_int_list(col, row_idx).ok());

            let sample = CalibrationSample {
                data,
                timestep,
                shape,
            };

            dataset.add_sample(sample);
            samples_added += 1;
        }

        Ok(samples_added)
    }

    /// Extract float list from Arrow array
    fn extract_float_list(&self, array: &ArrayRef, row_idx: usize) -> Result<Vec<f32>> {
        // Handle List<Float32>
        if let Some(list_array) = array.as_any().downcast_ref::<arrow::array::ListArray>() {
            if list_array.is_null(row_idx) {
                return Err(QuantError::Internal(
                    "Null data in calibration sample".to_string(),
                ));
            }

            let value_array = list_array.value(row_idx);
            if let Some(float_array) = value_array.as_any().downcast_ref::<Float32Array>() {
                let values: Vec<f32> = (0..float_array.len())
                    .map(|i| float_array.value(i))
                    .collect();
                return Ok(values);
            }
        }

        Err(QuantError::Internal(
            "Invalid data type for 'data' column, expected List<Float32>".to_string(),
        ))
    }

    /// Extract int list from Arrow array
    fn extract_int_list(&self, array: &ArrayRef, row_idx: usize) -> Result<Vec<usize>> {
        // Handle List<Int64>
        if let Some(list_array) = array.as_any().downcast_ref::<arrow::array::ListArray>() {
            if list_array.is_null(row_idx) {
                return Ok(Vec::new());
            }

            let value_array = list_array.value(row_idx);
            if let Some(int_array) = value_array
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
            {
                let values: Vec<usize> = (0..int_array.len())
                    .map(|i| int_array.value(i) as usize)
                    .collect();
                return Ok(values);
            }
        }

        Ok(Vec::new())
    }

    /// Load calibration data from Arrow IPC format (HuggingFace Dataset)
    ///
    /// This is similar to Parquet but uses Arrow IPC format
    fn load_arrow(&self, path: &Path) -> Result<CalibrationDataset> {
        use arrow::ipc::reader::FileReader;

        let file = File::open(path)?;
        let reader = FileReader::try_new(file, None)?;

        let mut dataset = CalibrationDataset::new();

        for batch_result in reader {
            let batch = batch_result?;
            self.process_parquet_batch(&batch, &mut dataset)?;

            if dataset.len() >= self.max_samples {
                break;
            }
        }

        if dataset.is_empty() {
            return Err(QuantError::Internal(
                "No calibration samples loaded from Arrow IPC".to_string(),
            ));
        }

        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_calibration_dataset_basic() {
        let mut dataset = CalibrationDataset::new();
        assert_eq!(dataset.len(), 0);
        assert!(dataset.is_empty());

        let sample = CalibrationSample {
            data: vec![1.0, 2.0, 3.0],
            timestep: Some(0),
            shape: Some(vec![1, 3]),
        };

        dataset.add_sample(sample.clone());
        assert_eq!(dataset.len(), 1);
        assert!(!dataset.is_empty());

        let retrieved = dataset.get_sample(0).unwrap();
        assert_eq!(retrieved.data, sample.data);
        assert_eq!(retrieved.timestep, sample.timestep);
    }

    #[test]
    fn test_load_jsonl() -> Result<()> {
        // Create temporary JSONL file
        let mut temp_file = NamedTempFile::new()?;
        writeln!(
            temp_file,
            r#"{{"data": [0.1, 0.2, 0.3], "timestep": 0, "shape": [1, 3]}}"#
        )?;
        writeln!(
            temp_file,
            r#"{{"data": [0.4, 0.5, 0.6], "timestep": 1, "shape": [1, 3]}}"#
        )?;
        temp_file.flush()?;

        // Rename to .jsonl extension
        let temp_path = temp_file.path();
        let jsonl_path = temp_path.with_extension("jsonl");
        std::fs::copy(temp_path, &jsonl_path)?;

        // Load calibration data
        let loader = CalibrationLoader::new(128);
        let dataset = loader.load_jsonl(&jsonl_path)?;

        assert_eq!(dataset.len(), 2);

        let sample0 = dataset.get_sample(0).unwrap();
        assert_eq!(sample0.data, vec![0.1, 0.2, 0.3]);
        assert_eq!(sample0.timestep, Some(0));

        let sample1 = dataset.get_sample(1).unwrap();
        assert_eq!(sample1.data, vec![0.4, 0.5, 0.6]);
        assert_eq!(sample1.timestep, Some(1));

        // Cleanup
        std::fs::remove_file(&jsonl_path)?;

        Ok(())
    }

    #[test]
    fn test_max_samples_limit() -> Result<()> {
        // Create temporary JSONL file with 5 samples
        let mut temp_file = NamedTempFile::new()?;
        for i in 0..5 {
            writeln!(
                temp_file,
                r#"{{"data": [{}, {}, {}], "timestep": {}}}"#,
                i as f32,
                i as f32 + 0.1,
                i as f32 + 0.2,
                i
            )?;
        }
        temp_file.flush()?;

        let temp_path = temp_file.path();
        let jsonl_path = temp_path.with_extension("jsonl");
        std::fs::copy(temp_path, &jsonl_path)?;

        // Load with max_samples = 3
        let loader = CalibrationLoader::new(3);
        let dataset = loader.load_jsonl(&jsonl_path)?;

        assert_eq!(dataset.len(), 3);

        // Cleanup
        std::fs::remove_file(&jsonl_path)?;

        Ok(())
    }

    #[test]
    fn test_invalid_file_path() {
        let loader = CalibrationLoader::new(128);
        let result = loader.load_from_file(Path::new("nonexistent.jsonl"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_parquet() -> Result<()> {
        use arrow::array::{Float32Array, Int64Array, ListArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        // Create temporary Parquet file
        let temp_file = NamedTempFile::new()?;
        let temp_path = temp_file.path();
        let parquet_path = temp_path.with_extension("parquet");

        // Define schema
        let schema = Schema::new(vec![
            Field::new(
                "data",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
                false,
            ),
            Field::new("timestep", DataType::Int64, true),
            Field::new(
                "shape",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
                true,
            ),
        ]);

        // Create data arrays
        let data_values = Float32Array::from(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let data_offsets = arrow::buffer::OffsetBuffer::new(vec![0, 3, 6].into());
        let data_list = ListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            data_offsets,
            Arc::new(data_values),
            None,
        );

        let timestep_array = Int64Array::from(vec![Some(0), Some(1)]);

        let shape_values = Int64Array::from(vec![1, 3, 1, 3]);
        let shape_offsets = arrow::buffer::OffsetBuffer::new(vec![0, 2, 4].into());
        let shape_list = ListArray::new(
            Arc::new(Field::new("item", DataType::Int64, false)),
            shape_offsets,
            Arc::new(shape_values),
            None,
        );

        // Create record batch
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(data_list),
                Arc::new(timestep_array),
                Arc::new(shape_list),
            ],
        )?;

        // Write to Parquet
        let file = File::create(&parquet_path)?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        // Load calibration data
        let loader = CalibrationLoader::new(128);
        let dataset = loader.load_parquet(&parquet_path)?;

        assert_eq!(dataset.len(), 2);

        let sample0 = dataset.get_sample(0).unwrap();
        assert_eq!(sample0.data, vec![0.1, 0.2, 0.3]);
        assert_eq!(sample0.timestep, Some(0));
        assert_eq!(sample0.shape, Some(vec![1, 3]));

        let sample1 = dataset.get_sample(1).unwrap();
        assert_eq!(sample1.data, vec![0.4, 0.5, 0.6]);
        assert_eq!(sample1.timestep, Some(1));

        // Cleanup
        std::fs::remove_file(&parquet_path)?;

        Ok(())
    }

    #[test]
    fn test_load_from_file_auto_detect() -> Result<()> {
        // Create temporary JSONL file
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, r#"{{"data": [0.1, 0.2, 0.3], "timestep": 0}}"#)?;
        temp_file.flush()?;

        let temp_path = temp_file.path();
        let jsonl_path = temp_path.with_extension("jsonl");
        std::fs::copy(temp_path, &jsonl_path)?;

        // Test auto-detection
        let loader = CalibrationLoader::new(128);
        let (dataset, stats) = loader.load_from_file(&jsonl_path)?;

        assert_eq!(dataset.len(), 1);
        assert!(stats.num_timesteps >= 1);

        // Cleanup
        std::fs::remove_file(&jsonl_path)?;

        Ok(())
    }

    #[test]
    fn test_calibration_caching() -> Result<()> {
        use tempfile::TempDir;

        // Create temporary cache directory
        let cache_dir = TempDir::new()?;
        let loader = CalibrationLoader::with_cache(128, cache_dir.path().to_path_buf())?;

        // Generate synthetic data (should cache)
        let (dataset1, stats1) = loader.generate_synthetic_data(64, vec![1, 256], 1000)?;
        assert_eq!(dataset1.len(), 64);

        // Generate same data again (should load from cache)
        let (dataset2, stats2) = loader.generate_synthetic_data(64, vec![1, 256], 1000)?;
        assert_eq!(dataset2.len(), 64);

        // Verify data is the same
        assert_eq!(dataset1.len(), dataset2.len());
        assert_eq!(stats1.num_timesteps, stats2.num_timesteps);

        // Verify cache file exists
        let cache_files: Vec<_> = std::fs::read_dir(cache_dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        assert_eq!(cache_files.len(), 1);

        Ok(())
    }

    #[test]
    fn test_calibration_cache_clear() -> Result<()> {
        use tempfile::TempDir;

        // Create temporary cache directory
        let cache_dir = TempDir::new()?;
        let loader = CalibrationLoader::with_cache(128, cache_dir.path().to_path_buf())?;

        // Generate synthetic data (should cache)
        let (_dataset, _stats) = loader.generate_synthetic_data(64, vec![1, 256], 1000)?;

        // Verify cache file exists
        let cache_files_before: Vec<_> = std::fs::read_dir(cache_dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        assert_eq!(cache_files_before.len(), 1);

        // Clear cache
        loader.clear_cache()?;

        // Verify cache is empty
        let cache_files_after: Vec<_> = std::fs::read_dir(cache_dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        assert_eq!(cache_files_after.len(), 0);

        Ok(())
    }

    #[test]
    fn test_calibration_cache_file_loading() -> Result<()> {
        use tempfile::TempDir;

        // Create temporary cache directory
        let cache_dir = TempDir::new()?;
        let loader = CalibrationLoader::with_cache(128, cache_dir.path().to_path_buf())?;

        // Create temporary JSONL file
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, r#"{{"data": [0.1, 0.2, 0.3], "timestep": 0}}"#)?;
        writeln!(temp_file, r#"{{"data": [0.4, 0.5, 0.6], "timestep": 1}}"#)?;
        temp_file.flush()?;

        let temp_path = temp_file.path();
        let jsonl_path = temp_path.with_extension("jsonl");
        std::fs::copy(temp_path, &jsonl_path)?;

        // Load from file (should cache)
        let (dataset1, stats1) = loader.load_from_file(&jsonl_path)?;
        assert_eq!(dataset1.len(), 2);

        // Load again (should use cache)
        let (dataset2, stats2) = loader.load_from_file(&jsonl_path)?;
        assert_eq!(dataset2.len(), 2);

        // Verify data is the same
        assert_eq!(stats1.num_timesteps, stats2.num_timesteps);

        // Cleanup
        std::fs::remove_file(&jsonl_path)?;

        Ok(())
    }

    #[test]
    fn test_activation_stats_computation() -> Result<()> {
        let mut dataset = CalibrationDataset::new();

        // Add samples with known values
        for t in 0..10 {
            let data = vec![t as f32; 100];
            dataset.add_sample(CalibrationSample {
                data,
                timestep: Some(t),
                shape: Some(vec![1, 100]),
            });
        }

        let stats = ActivationStats::from_dataset(&dataset)?;

        assert_eq!(stats.num_timesteps, 10);
        assert_eq!(stats.mean.len(), 10);
        assert_eq!(stats.std.len(), 10);
        assert_eq!(stats.min.len(), 10);
        assert_eq!(stats.max.len(), 10);

        // Check statistics for timestep 5
        assert!((stats.mean[5] - 5.0).abs() < 0.01);
        // Note: std defaults to 1.0 when all values are identical (variance = 0)
        assert!((stats.std[5] - 1.0).abs() < 0.01);
        assert!((stats.min[5] - 5.0).abs() < 0.01);
        assert!((stats.max[5] - 5.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_activation_stats_empty_dataset() {
        let dataset = CalibrationDataset::new();
        let result = ActivationStats::from_dataset(&dataset);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_synthetic_data_basic() -> Result<()> {
        let loader = CalibrationLoader::new(128);
        let (dataset, stats) = loader.generate_synthetic_data(128, vec![1, 512], 1000)?;

        assert_eq!(dataset.len(), 128);
        // Note: num_timesteps is based on max timestep + 1, not total timesteps
        // With 128 samples distributed across 1000 timesteps, we get ~128 unique timesteps
        assert!(stats.num_timesteps >= 100 && stats.num_timesteps <= 1000);

        // Check first sample
        let sample = dataset.get_sample(0).unwrap();
        assert_eq!(sample.data.len(), 512);
        assert_eq!(sample.timestep, Some(0));
        assert_eq!(sample.shape, Some(vec![1, 512]));

        // Check last sample
        let last_sample = dataset.get_sample(127).unwrap();
        assert_eq!(last_sample.data.len(), 512);
        assert!(last_sample.timestep.is_some());

        // Check statistics
        assert!(stats.mean.len() >= 100);
        assert!(stats.std.len() >= 100);
        assert!(stats.min.len() >= 100);
        assert!(stats.max.len() >= 100);

        Ok(())
    }

    #[test]
    fn test_generate_synthetic_data_min_samples() -> Result<()> {
        let loader = CalibrationLoader::new(128);
        let (dataset, stats) = loader.generate_synthetic_data(32, vec![1, 256], 1000)?;

        assert_eq!(dataset.len(), 32);
        // With 32 samples, we get ~32 unique timesteps
        assert!(stats.num_timesteps >= 30 && stats.num_timesteps <= 1000);

        Ok(())
    }

    #[test]
    fn test_generate_synthetic_data_max_samples() -> Result<()> {
        let loader = CalibrationLoader::new(1024);
        let (dataset, stats) = loader.generate_synthetic_data(1024, vec![1, 128], 1000)?;

        assert_eq!(dataset.len(), 1024);
        assert_eq!(stats.num_timesteps, 1000);

        Ok(())
    }

    #[test]
    fn test_generate_synthetic_data_invalid_num_samples() {
        let loader = CalibrationLoader::new(128);

        // Too few samples
        let result = loader.generate_synthetic_data(16, vec![1, 512], 1000);
        assert!(result.is_err());

        // Too many samples
        let result = loader.generate_synthetic_data(2048, vec![1, 512], 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_synthetic_data_invalid_shape() {
        let loader = CalibrationLoader::new(128);

        // Empty shape
        let result = loader.generate_synthetic_data(128, vec![], 1000);
        assert!(result.is_err());

        // Zero dimension
        let result = loader.generate_synthetic_data(128, vec![0, 512], 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_synthetic_data_multidimensional() -> Result<()> {
        let loader = CalibrationLoader::new(128);

        // Image-like shape: [4, 64, 64]
        let (dataset, stats) = loader.generate_synthetic_data(64, vec![4, 64, 64], 1000)?;

        assert_eq!(dataset.len(), 64);
        // With 64 samples, we get ~64 unique timesteps
        assert!(stats.num_timesteps >= 60 && stats.num_timesteps <= 1000);

        let sample = dataset.get_sample(0).unwrap();
        assert_eq!(sample.data.len(), 4 * 64 * 64);
        assert_eq!(sample.shape, Some(vec![4, 64, 64]));

        Ok(())
    }

    #[test]
    fn test_generate_synthetic_data_timestep_distribution() -> Result<()> {
        let loader = CalibrationLoader::new(128);
        let num_samples = 100;
        let num_timesteps = 1000;
        let (dataset, stats) =
            loader.generate_synthetic_data(num_samples, vec![1, 128], num_timesteps)?;

        // With 100 samples, we get ~100 unique timesteps
        assert!(stats.num_timesteps >= 90 && stats.num_timesteps <= num_timesteps);

        // Check that timesteps are distributed across the range
        let mut timesteps: Vec<usize> = dataset
            .get_samples()
            .iter()
            .filter_map(|s| s.timestep)
            .collect();

        timesteps.sort();

        // First sample should be at timestep 0
        assert_eq!(timesteps[0], 0);

        // Last sample should be near the end
        let expected_last = num_timesteps - (num_timesteps / num_samples);
        assert!(timesteps[num_samples - 1] >= expected_last - 50);

        // Check that timesteps are roughly evenly distributed
        let expected_step = num_timesteps / num_samples;
        for i in 1..timesteps.len() {
            let actual_step = timesteps[i] - timesteps[i - 1];
            // Allow reasonable tolerance
            assert!(actual_step <= expected_step + 2);
        }

        Ok(())
    }

    #[test]
    fn test_gaussian_noise_properties() -> Result<()> {
        let loader = CalibrationLoader::new(128);
        let (dataset, _stats) = loader.generate_synthetic_data(128, vec![1, 1000], 1000)?;

        // Collect all noise values
        let mut all_values = Vec::new();
        for sample in dataset.get_samples() {
            all_values.extend(&sample.data);
        }

        // Calculate mean and std
        let mean: f32 = all_values.iter().sum::<f32>() / all_values.len() as f32;
        let variance: f32 =
            all_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / all_values.len() as f32;
        let std = variance.sqrt();

        // For N(0, 1), mean should be close to 0 and std close to 1
        // With large sample size, these should be within reasonable bounds
        assert!(mean.abs() < 0.1, "Mean should be close to 0, got {}", mean);
        assert!(
            (std - 1.0).abs() < 0.1,
            "Std should be close to 1, got {}",
            std
        );

        Ok(())
    }

    #[test]
    fn test_generate_synthetic_data_odd_size() -> Result<()> {
        let loader = CalibrationLoader::new(128);

        // Odd sample size to test Box-Muller edge case
        let (dataset, _stats) = loader.generate_synthetic_data(64, vec![1, 513], 1000)?;

        assert_eq!(dataset.len(), 64);

        let sample = dataset.get_sample(0).unwrap();
        assert_eq!(sample.data.len(), 513);

        Ok(())
    }

    // ============================================================================
    // Additional Comprehensive Tests for Task 8.4
    // ============================================================================

    /// Test loading JSONL with missing optional fields
    #[test]
    fn test_load_jsonl_missing_optional_fields() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        // Sample without timestep and shape
        writeln!(temp_file, r#"{{"data": [1.0, 2.0, 3.0]}}"#)?;
        // Sample with only timestep
        writeln!(temp_file, r#"{{"data": [4.0, 5.0, 6.0], "timestep": 5}}"#)?;
        temp_file.flush()?;

        let temp_path = temp_file.path();
        let jsonl_path = temp_path.with_extension("jsonl");
        std::fs::copy(temp_path, &jsonl_path)?;

        let loader = CalibrationLoader::new(128);
        let dataset = loader.load_jsonl(&jsonl_path)?;

        assert_eq!(dataset.len(), 2);

        let sample0 = dataset.get_sample(0).unwrap();
        assert_eq!(sample0.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(sample0.timestep, None);
        assert_eq!(sample0.shape, None);

        let sample1 = dataset.get_sample(1).unwrap();
        assert_eq!(sample1.timestep, Some(5));

        std::fs::remove_file(&jsonl_path)?;
        Ok(())
    }

    /// Test loading JSONL with malformed JSON
    #[test]
    fn test_load_jsonl_malformed() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, r#"{{"data": [1.0, 2.0, 3.0]}}"#)?;
        writeln!(temp_file, r#"{{invalid json}}"#)?; // Malformed line
        temp_file.flush()?;

        let temp_path = temp_file.path();
        let jsonl_path = temp_path.with_extension("jsonl");
        std::fs::copy(temp_path, &jsonl_path)?;

        let loader = CalibrationLoader::new(128);
        let result = loader.load_jsonl(&jsonl_path);

        // Should fail on malformed JSON
        assert!(result.is_err());

        std::fs::remove_file(&jsonl_path)?;
        Ok(())
    }

    /// Test loading JSONL with empty file
    #[test]
    fn test_load_jsonl_empty_file() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let temp_path = temp_file.path();
        let jsonl_path = temp_path.with_extension("jsonl");
        std::fs::copy(temp_path, &jsonl_path)?;

        let loader = CalibrationLoader::new(128);
        let result = loader.load_jsonl(&jsonl_path);

        // Should fail on empty file
        assert!(result.is_err());

        std::fs::remove_file(&jsonl_path)?;
        Ok(())
    }

    /// Test loading JSONL with large data arrays
    #[test]
    fn test_load_jsonl_large_arrays() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;

        // Create a large array (10000 elements)
        let large_array: Vec<f32> = (0..10000).map(|i| i as f32 * 0.001).collect();
        let json_data = serde_json::json!({
            "data": large_array,
            "timestep": 0,
            "shape": vec![1, 10000]
        });
        writeln!(temp_file, "{}", serde_json::to_string(&json_data)?)?;
        temp_file.flush()?;

        let temp_path = temp_file.path();
        let jsonl_path = temp_path.with_extension("jsonl");
        std::fs::copy(temp_path, &jsonl_path)?;

        let loader = CalibrationLoader::new(128);
        let dataset = loader.load_jsonl(&jsonl_path)?;

        assert_eq!(dataset.len(), 1);
        let sample = dataset.get_sample(0).unwrap();
        assert_eq!(sample.data.len(), 10000);
        assert_eq!(sample.data[0], 0.0);
        assert!((sample.data[9999] - 9.999).abs() < 0.001);

        std::fs::remove_file(&jsonl_path)?;
        Ok(())
    }

    /// Test loading from file with unsupported extension
    #[test]
    fn test_load_from_file_unsupported_format() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "some data")?;
        temp_file.flush()?;

        let temp_path = temp_file.path();
        let txt_path = temp_path.with_extension("txt");
        std::fs::copy(temp_path, &txt_path)?;

        let loader = CalibrationLoader::new(128);
        let result = loader.load_from_file(&txt_path);

        // Should fail on unsupported format
        assert!(result.is_err());

        std::fs::remove_file(&txt_path)?;
        Ok(())
    }

    /// Test synthetic data generation with various shapes
    #[test]
    fn test_generate_synthetic_data_various_shapes() -> Result<()> {
        let loader = CalibrationLoader::new(128);

        // Test 1D shape
        let (dataset1d, _) = loader.generate_synthetic_data(64, vec![512], 1000)?;
        assert_eq!(dataset1d.get_sample(0).unwrap().data.len(), 512);

        // Test 2D shape (text-like)
        let (dataset2d, _) = loader.generate_synthetic_data(64, vec![1, 512], 1000)?;
        assert_eq!(dataset2d.get_sample(0).unwrap().data.len(), 512);

        // Test 3D shape (image-like)
        let (dataset3d, _) = loader.generate_synthetic_data(64, vec![4, 32, 32], 1000)?;
        assert_eq!(dataset3d.get_sample(0).unwrap().data.len(), 4 * 32 * 32);

        // Test 4D shape (batch of images)
        let (dataset4d, _) = loader.generate_synthetic_data(64, vec![2, 4, 16, 16], 1000)?;
        assert_eq!(dataset4d.get_sample(0).unwrap().data.len(), 2 * 4 * 16 * 16);

        Ok(())
    }

    /// Test synthetic data generation with different timestep counts
    #[test]
    fn test_generate_synthetic_data_various_timesteps() -> Result<()> {
        let loader = CalibrationLoader::new(128);

        // Test with 50 timesteps
        let (_dataset50, stats50) = loader.generate_synthetic_data(64, vec![1, 256], 50)?;
        // With 64 samples across 50 timesteps, we should get close to 50 unique timesteps
        assert!(stats50.num_timesteps >= 45 && stats50.num_timesteps <= 50);
        assert!(stats50.mean.len() >= 45);

        // Test with 1000 timesteps (default)
        let (_dataset1000, stats1000) = loader.generate_synthetic_data(64, vec![1, 256], 1000)?;
        // With 64 samples across 1000 timesteps, we get ~64 unique timesteps
        assert!(stats1000.num_timesteps >= 60 && stats1000.num_timesteps <= 1000);
        assert!(stats1000.mean.len() >= 60);

        // Test with 5000 timesteps
        let (_dataset5000, stats5000) = loader.generate_synthetic_data(64, vec![1, 256], 5000)?;
        // With 64 samples across 5000 timesteps, we get ~64 unique timesteps
        assert!(stats5000.num_timesteps >= 60 && stats5000.num_timesteps <= 5000);
        assert!(stats5000.mean.len() >= 60);

        Ok(())
    }

    /// Test synthetic data generation with boundary sample counts
    #[test]
    fn test_generate_synthetic_data_boundary_samples() -> Result<()> {
        let loader = CalibrationLoader::new(1024);

        // Minimum valid samples (32)
        let (dataset_min, _) = loader.generate_synthetic_data(32, vec![1, 128], 1000)?;
        assert_eq!(dataset_min.len(), 32);

        // Maximum valid samples (1024)
        let (dataset_max, _) = loader.generate_synthetic_data(1024, vec![1, 128], 1000)?;
        assert_eq!(dataset_max.len(), 1024);

        Ok(())
    }

    /// Test caching with different parameters
    #[test]
    fn test_calibration_caching_different_params() -> Result<()> {
        use tempfile::TempDir;

        let cache_dir = TempDir::new()?;
        let loader = CalibrationLoader::with_cache(128, cache_dir.path().to_path_buf())?;

        // Generate data with different parameters
        let (_dataset1, _) = loader.generate_synthetic_data(64, vec![1, 256], 1000)?;
        let (_dataset2, _) = loader.generate_synthetic_data(64, vec![1, 512], 1000)?; // Different shape
        let (_dataset3, _) = loader.generate_synthetic_data(128, vec![1, 256], 1000)?; // Different count

        // Each should create a separate cache entry
        let cache_files: Vec<_> = std::fs::read_dir(cache_dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();

        // Should have 3 different cache files
        assert_eq!(cache_files.len(), 3);

        Ok(())
    }

    /// Test cache persistence across loader instances
    #[test]
    fn test_calibration_cache_persistence() -> Result<()> {
        use tempfile::TempDir;

        let cache_dir = TempDir::new()?;

        // First loader instance
        {
            let loader1 = CalibrationLoader::with_cache(128, cache_dir.path().to_path_buf())?;
            let (dataset1, _) = loader1.generate_synthetic_data(64, vec![1, 256], 1000)?;
            assert_eq!(dataset1.len(), 64);
        }

        // Second loader instance (should use cached data)
        {
            let loader2 = CalibrationLoader::with_cache(128, cache_dir.path().to_path_buf())?;
            let (dataset2, _) = loader2.generate_synthetic_data(64, vec![1, 256], 1000)?;
            assert_eq!(dataset2.len(), 64);
        }

        // Verify only one cache file exists
        let cache_files: Vec<_> = std::fs::read_dir(cache_dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        assert_eq!(cache_files.len(), 1);

        Ok(())
    }

    /// Test caching disabled by default
    #[test]
    fn test_calibration_no_cache_by_default() -> Result<()> {
        use tempfile::TempDir;

        let cache_dir = TempDir::new()?;

        // Create loader without caching
        let loader = CalibrationLoader::new(128);
        let (dataset, _) = loader.generate_synthetic_data(64, vec![1, 256], 1000)?;
        assert_eq!(dataset.len(), 64);

        // Verify no cache files created
        let cache_files: Vec<_> = std::fs::read_dir(cache_dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        assert_eq!(cache_files.len(), 0);

        Ok(())
    }

    /// Test activation statistics with edge cases
    #[test]
    fn test_activation_stats_edge_cases() -> Result<()> {
        let mut dataset = CalibrationDataset::new();

        // Add samples with extreme values
        dataset.add_sample(CalibrationSample {
            data: vec![f32::MAX, f32::MIN, 0.0],
            timestep: Some(0),
            shape: Some(vec![1, 3]),
        });

        // Add samples with very small values
        dataset.add_sample(CalibrationSample {
            data: vec![1e-10, -1e-10, 0.0],
            timestep: Some(1),
            shape: Some(vec![1, 3]),
        });

        let stats = ActivationStats::from_dataset(&dataset)?;

        assert_eq!(stats.num_timesteps, 2);

        // Check that statistics are finite
        for i in 0..stats.num_timesteps {
            assert!(stats.mean[i].is_finite());
            assert!(stats.std[i].is_finite());
            assert!(stats.min[i].is_finite());
            assert!(stats.max[i].is_finite());
        }

        Ok(())
    }

    /// Test activation statistics with single sample per timestep
    #[test]
    fn test_activation_stats_single_sample_per_timestep() -> Result<()> {
        let mut dataset = CalibrationDataset::new();

        for t in 0..5 {
            dataset.add_sample(CalibrationSample {
                data: vec![t as f32; 10],
                timestep: Some(t),
                shape: Some(vec![1, 10]),
            });
        }

        let stats = ActivationStats::from_dataset(&dataset)?;

        assert_eq!(stats.num_timesteps, 5);

        // With identical values, std defaults to 1.0 (not 0)
        for t in 0..5 {
            assert!((stats.mean[t] - t as f32).abs() < 0.01);
            assert!((stats.std[t] - 1.0).abs() < 0.01); // Default value for zero variance
            assert!((stats.min[t] - t as f32).abs() < 0.01);
            assert!((stats.max[t] - t as f32).abs() < 0.01);
        }

        Ok(())
    }

    /// Test activation statistics with multiple samples per timestep
    #[test]
    fn test_activation_stats_multiple_samples_per_timestep() -> Result<()> {
        let mut dataset = CalibrationDataset::new();

        // Add 3 samples for each timestep with varying values
        for t in 0..3 {
            for i in 0..3 {
                let value = (t * 10 + i) as f32;
                dataset.add_sample(CalibrationSample {
                    data: vec![value; 100],
                    timestep: Some(t),
                    shape: Some(vec![1, 100]),
                });
            }
        }

        let stats = ActivationStats::from_dataset(&dataset)?;

        assert_eq!(stats.num_timesteps, 3);

        // Check that statistics are computed correctly
        // For timestep 0: values are 0, 1, 2 (mean = 1.0)
        assert!((stats.mean[0] - 1.0).abs() < 0.01);

        // For timestep 1: values are 10, 11, 12 (mean = 11.0)
        assert!((stats.mean[1] - 11.0).abs() < 0.01);

        Ok(())
    }

    /// Test activation statistics with missing timesteps
    #[test]
    fn test_activation_stats_missing_timesteps() -> Result<()> {
        let mut dataset = CalibrationDataset::new();

        // Add samples only for timesteps 0, 2, 4 (skip 1, 3)
        for t in [0, 2, 4].iter() {
            dataset.add_sample(CalibrationSample {
                data: vec![*t as f32; 100],
                timestep: Some(*t),
                shape: Some(vec![1, 100]),
            });
        }

        let stats = ActivationStats::from_dataset(&dataset)?;

        assert_eq!(stats.num_timesteps, 5);

        // Check that missing timesteps have default values
        assert!((stats.mean[1] - 0.0).abs() < 0.01); // Missing timestep 1
        assert!((stats.std[1] - 1.0).abs() < 0.01);
        assert!((stats.mean[3] - 0.0).abs() < 0.01); // Missing timestep 3
        assert!((stats.std[3] - 1.0).abs() < 0.01);

        Ok(())
    }

    /// Test Parquet loading with max_samples limit
    #[test]
    fn test_load_parquet_max_samples() -> Result<()> {
        use arrow::array::{Float32Array, Int64Array, ListArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        let temp_file = NamedTempFile::new()?;
        let temp_path = temp_file.path();
        let parquet_path = temp_path.with_extension("parquet");

        // Create schema
        let schema = Schema::new(vec![
            Field::new(
                "data",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
                false,
            ),
            Field::new("timestep", DataType::Int64, true),
        ]);

        // Create 10 samples
        let mut data_values = Vec::new();
        let mut data_offsets = vec![0];
        let mut timestep_values = Vec::new();

        for i in 0..10 {
            data_values.extend(vec![i as f32; 100]);
            data_offsets.push(data_offsets.last().unwrap() + 100);
            timestep_values.push(Some(i as i64));
        }

        let data_array = Float32Array::from(data_values);
        let data_offsets_buf = arrow::buffer::OffsetBuffer::new(data_offsets.into());
        let data_list = ListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            data_offsets_buf,
            Arc::new(data_array),
            None,
        );

        let timestep_array = Int64Array::from(timestep_values);

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(data_list), Arc::new(timestep_array)],
        )?;

        // Write to Parquet
        let file = File::create(&parquet_path)?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        // Load with max_samples = 5
        let loader = CalibrationLoader::new(5);
        let dataset = loader.load_parquet(&parquet_path)?;

        // Should only load 5 samples
        assert_eq!(dataset.len(), 5);

        std::fs::remove_file(&parquet_path)?;
        Ok(())
    }

    /// Test file loading with caching and file modification
    #[test]
    fn test_load_from_file_cache_invalidation() -> Result<()> {
        use std::thread;
        use std::time::Duration;
        use tempfile::TempDir;

        let cache_dir = TempDir::new()?;
        let loader = CalibrationLoader::with_cache(128, cache_dir.path().to_path_buf())?;

        // Create initial JSONL file
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, r#"{{"data": [1.0, 2.0, 3.0], "timestep": 0}}"#)?;
        temp_file.flush()?;

        let temp_path = temp_file.path();
        let jsonl_path = temp_path.with_extension("jsonl");
        std::fs::copy(temp_path, &jsonl_path)?;

        // Load first time (should cache)
        let (dataset1, _) = loader.load_from_file(&jsonl_path)?;
        assert_eq!(dataset1.len(), 1);

        // Wait to ensure different modification time
        thread::sleep(Duration::from_secs(2));

        // Create a new file with different content
        std::fs::remove_file(&jsonl_path)?;
        let mut new_file = File::create(&jsonl_path)?;
        writeln!(new_file, r#"{{"data": [1.0, 2.0, 3.0], "timestep": 0}}"#)?;
        writeln!(new_file, r#"{{"data": [4.0, 5.0, 6.0], "timestep": 1}}"#)?;
        new_file.flush()?;

        // Load again (should detect modification and reload)
        let (dataset2, _) = loader.load_from_file(&jsonl_path)?;
        assert_eq!(dataset2.len(), 2);

        std::fs::remove_file(&jsonl_path)?;
        Ok(())
    }

    /// Test synthetic data generation produces valid Gaussian distribution
    #[test]
    fn test_synthetic_data_gaussian_distribution() -> Result<()> {
        let loader = CalibrationLoader::new(128);
        let (dataset, _) = loader.generate_synthetic_data(128, vec![1, 10000], 1000)?;

        // Collect all values from all samples
        let mut all_values: Vec<f32> = Vec::new();
        for sample in dataset.get_samples() {
            all_values.extend(&sample.data);
        }

        // Filter out any non-finite values
        let finite_values: Vec<f32> = all_values
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();

        if finite_values.is_empty() {
            return Err(QuantError::Internal(
                "No finite values in synthetic data".to_string(),
            ));
        }

        // Calculate statistics
        let n = finite_values.len() as f32;
        let mean: f32 = finite_values.iter().sum::<f32>() / n;
        let variance: f32 = finite_values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / n;
        let std = variance.sqrt();

        // For N(0, 1), mean ≈ 0 and std ≈ 1
        assert!(mean.is_finite(), "Mean should be finite, got {}", mean);
        assert!(std.is_finite(), "Std should be finite, got {}", std);
        assert!(mean.abs() < 0.05, "Mean should be close to 0, got {}", mean);
        assert!(
            (std - 1.0).abs() < 0.05,
            "Std should be close to 1, got {}",
            std
        );

        // Check that values are within reasonable range (-4, 4) for ~99.99% of samples
        let outliers = finite_values.iter().filter(|&x| x.abs() > 4.0).count();
        let outlier_ratio = outliers as f32 / n;
        assert!(
            outlier_ratio < 0.001,
            "Too many outliers: {}",
            outlier_ratio
        );

        Ok(())
    }
}

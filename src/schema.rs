//! Extended Parquet V2 schema for diffusion models

use crate::config::Modality;
use crate::errors::Result;
use crate::spatial::QuantizedSpatialLayer;
use crate::time_aware::{QuantizedLayer, TimeGroupParams};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Schema version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaVersion {
    V2,
    V2Extended,
}

/// Extended Parquet V2 schema with diffusion-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParquetV2Extended {
    // Base Parquet V2 fields
    pub layer_name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data: Vec<u8>,
    pub num_params: usize,
    pub quant_type: String,
    pub scales: Vec<f32>,
    pub zero_points: Vec<f32>,
    pub quant_axis: Option<usize>,
    pub group_size: Option<usize>,

    // Diffusion-specific extensions
    pub is_diffusion_model: bool,
    pub modality: Option<String>,
    pub time_aware_quant: Option<TimeAwareQuantMetadata>,
    pub spatial_quant: Option<SpatialQuantMetadata>,
    pub activation_stats: Option<ActivationStatsMetadata>,
}

/// Time-aware quantization metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeAwareQuantMetadata {
    pub enabled: bool,
    pub num_time_groups: usize,
    pub time_group_params: Vec<TimeGroupParams>,
}

/// Spatial quantization metadata
///
/// Stores metadata about spatial quantization techniques applied to a layer,
/// including channel equalization and activation smoothing.
///
/// # Fields
///
/// - `enabled`: Whether spatial quantization was applied
/// - `channel_equalization`: Whether DiTAS channel equalization was used
/// - `activation_smoothing`: Whether activation smoothing was applied
/// - `equalization_scales`: Per-channel equalization scale factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialQuantMetadata {
    pub enabled: bool,
    pub channel_equalization: bool,
    pub activation_smoothing: bool,
    pub equalization_scales: Vec<f32>,
}

/// Activation statistics metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationStatsMetadata {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

impl ParquetV2Extended {
    /// Create unquantized placeholder for SafeTensors conversion
    ///
    /// Creates a minimal ParquetV2Extended schema for a tensor that hasn't been quantized yet.
    /// This is used during SafeTensors → Parquet conversion, before actual quantization.
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name of the tensor/layer
    /// * `shape` - Shape of the tensor
    /// * `modality` - Model modality
    ///
    /// # Returns
    ///
    /// ParquetV2Extended with placeholder values
    pub fn new_unquantized(
        layer_name: String,
        shape: Vec<usize>,
        modality: Modality,
        data: Vec<u8>,
    ) -> Self {
        let num_params: usize = shape.iter().product();
        
        Self {
            layer_name,
            shape,
            dtype: "float32".to_string(),
            data,
            num_params,
            quant_type: "none".to_string(), // Not yet quantized
            scales: Vec::new(),
            zero_points: Vec::new(),
            quant_axis: None,
            group_size: None,
            is_diffusion_model: true,
            modality: Some(modality.to_string()),
            time_aware_quant: None,
            spatial_quant: None,
            activation_stats: None,
        }
    }

    /// Create from base Parquet V2 (backward compatibility)
    pub fn from_v2_base(
        layer_name: String,
        shape: Vec<usize>,
        dtype: String,
        data: Vec<u8>,
        num_params: usize,
        quant_type: String,
        scales: Vec<f32>,
        zero_points: Vec<f32>,
        quant_axis: Option<usize>,
        group_size: Option<usize>,
    ) -> Self {
        Self {
            layer_name,
            shape,
            dtype,
            data,
            num_params,
            quant_type,
            scales,
            zero_points,
            quant_axis,
            group_size,
            is_diffusion_model: false,
            modality: None,
            time_aware_quant: None,
            spatial_quant: None,
            activation_stats: None,
        }
    }

    /// Create with time-aware quantization
    pub fn with_time_aware(mut self, modality: Modality, quantized_layer: QuantizedLayer) -> Self {
        self.is_diffusion_model = true;
        self.modality = Some(modality.to_string());
        self.data = quantized_layer.data;
        self.scales = quantized_layer.scales;
        self.zero_points = quantized_layer.zero_points;
        self.time_aware_quant = Some(TimeAwareQuantMetadata {
            enabled: true,
            num_time_groups: quantized_layer.time_group_params.len(),
            time_group_params: quantized_layer.time_group_params,
        });
        self
    }

    /// Create with time-aware quantization and custom bit-width
    pub fn with_time_aware_and_bit_width(
        mut self,
        modality: Modality,
        quantized_layer: QuantizedLayer,
        bit_width: u8,
    ) -> Self {
        self.is_diffusion_model = true;
        self.modality = Some(modality.to_string());
        self.data = quantized_layer.data;
        self.scales = quantized_layer.scales;
        self.zero_points = quantized_layer.zero_points;
        self.quant_type = format!("int{}", bit_width);
        self.time_aware_quant = Some(TimeAwareQuantMetadata {
            enabled: true,
            num_time_groups: quantized_layer.time_group_params.len(),
            time_group_params: quantized_layer.time_group_params,
        });
        self
    }

    /// Create with spatial quantization
    pub fn with_spatial(
        mut self,
        modality: Modality,
        quantized_layer: QuantizedSpatialLayer,
        equalization_scales: Vec<f32>,
    ) -> Self {
        self.is_diffusion_model = true;
        self.modality = Some(modality.to_string());
        self.data = quantized_layer.data;
        self.scales = quantized_layer.scales;
        self.zero_points = quantized_layer.zero_points;
        self.group_size = Some(quantized_layer.group_size);
        self.spatial_quant = Some(SpatialQuantMetadata {
            enabled: true,
            channel_equalization: !equalization_scales.is_empty(),
            activation_smoothing: false,
            equalization_scales,
        });
        self
    }

    /// Create with spatial quantization and custom bit-width
    pub fn with_spatial_and_bit_width(
        mut self,
        modality: Modality,
        quantized_layer: QuantizedSpatialLayer,
        equalization_scales: Vec<f32>,
        bit_width: u8,
    ) -> Self {
        self.is_diffusion_model = true;
        self.modality = Some(modality.to_string());
        self.data = quantized_layer.data;
        self.scales = quantized_layer.scales;
        self.zero_points = quantized_layer.zero_points;
        self.group_size = Some(quantized_layer.group_size);
        self.quant_type = format!("int{}", bit_width);
        self.spatial_quant = Some(SpatialQuantMetadata {
            enabled: true,
            channel_equalization: !equalization_scales.is_empty(),
            activation_smoothing: false,
            equalization_scales,
        });
        self
    }

    /// Update bit-width only (for base quantization)
    pub fn with_bit_width(mut self, bit_width: u8) -> Self {
        self.quant_type = format!("int{}", bit_width);
        self
    }

    /// Detect schema version from file
    pub fn detect_schema_version(path: &Path) -> Result<SchemaVersion> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;

        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema();

        // Check for diffusion-specific columns
        let has_diffusion_field = schema
            .fields()
            .iter()
            .any(|field| field.name() == "is_diffusion_model");

        if has_diffusion_field {
            Ok(SchemaVersion::V2Extended)
        } else {
            Ok(SchemaVersion::V2)
        }
    }

    /// Write to Parquet file with extended schema
    pub fn write_to_parquet(&self, path: &Path) -> Result<()> {
        use arrow::array::{ArrayRef, BooleanArray, Float32Array, StringArray, UInt64Array};
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::fs::File;
        use std::sync::Arc;

        // Build Arrow schema - only include diffusion fields if this is a diffusion model
        let mut fields = vec![
            Field::new("layer_name", DataType::Utf8, false),
            Field::new(
                "shape",
                DataType::List(Arc::new(Field::new("item", DataType::UInt64, false))),
                false,
            ),
            Field::new("dtype", DataType::Utf8, false),
            Field::new("data", DataType::LargeBinary, false),
            Field::new("num_params", DataType::UInt64, false),
            Field::new("quant_type", DataType::Utf8, false),
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
            Field::new("quant_axis", DataType::UInt64, true),
            Field::new("group_size", DataType::UInt64, true),
        ];

        // Only add diffusion-specific fields if this is a diffusion model
        if self.is_diffusion_model {
            fields.push(Field::new("is_diffusion_model", DataType::Boolean, false));
            fields.push(Field::new("modality", DataType::Utf8, true));

            // Add metadata fields as JSON strings for simplicity
            if self.time_aware_quant.is_some() {
                fields.push(Field::new("time_aware_quant_json", DataType::Utf8, true));
            }
            if self.spatial_quant.is_some() {
                fields.push(Field::new("spatial_quant_json", DataType::Utf8, true));
            }
            if self.activation_stats.is_some() {
                fields.push(Field::new("activation_stats_json", DataType::Utf8, true));
            }
        }

        let schema = Arc::new(Schema::new(fields));

        // Create arrays for base fields
        let layer_name_array = StringArray::from(vec![self.layer_name.as_str()]);

        // Create shape list array
        let shape_array = {
            let shape_u64: Vec<u64> = self.shape.iter().map(|&x| x as u64).collect();
            let values = UInt64Array::from(shape_u64);
            let offsets = OffsetBuffer::from_lengths(vec![values.len()]);
            arrow::array::ListArray::new(
                Arc::new(Field::new("item", DataType::UInt64, false)),
                offsets,
                Arc::new(values),
                None,
            )
        };

        let dtype_array = StringArray::from(vec![self.dtype.as_str()]);
        let data_array = arrow::array::LargeBinaryArray::from(vec![self.data.as_slice()]);
        let num_params_array = UInt64Array::from(vec![self.num_params as u64]);
        let quant_type_array = StringArray::from(vec![self.quant_type.as_str()]);

        // Create scales list array
        let scales_array = {
            let values = Float32Array::from(self.scales.clone());
            let offsets = OffsetBuffer::from_lengths(vec![values.len()]);
            arrow::array::ListArray::new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                offsets,
                Arc::new(values),
                None,
            )
        };

        // Create zero_points list array
        let zero_points_array = {
            let values = Float32Array::from(self.zero_points.clone());
            let offsets = OffsetBuffer::from_lengths(vec![values.len()]);
            arrow::array::ListArray::new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                offsets,
                Arc::new(values),
                None,
            )
        };

        let quant_axis_array = match self.quant_axis {
            Some(axis) => Arc::new(UInt64Array::from(vec![Some(axis as u64)])) as ArrayRef,
            None => Arc::new(UInt64Array::from(vec![None])) as ArrayRef,
        };

        let group_size_array = match self.group_size {
            Some(size) => Arc::new(UInt64Array::from(vec![Some(size as u64)])) as ArrayRef,
            None => Arc::new(UInt64Array::from(vec![None])) as ArrayRef,
        };

        // Build columns vector
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(layer_name_array),
            Arc::new(shape_array),
            Arc::new(dtype_array),
            Arc::new(data_array),
            Arc::new(num_params_array),
            Arc::new(quant_type_array),
            Arc::new(scales_array),
            Arc::new(zero_points_array),
            quant_axis_array,
            group_size_array,
        ];

        // Only add diffusion fields if this is a diffusion model
        if self.is_diffusion_model {
            let is_diffusion_array = BooleanArray::from(vec![self.is_diffusion_model]);
            let modality_array = match &self.modality {
                Some(m) => Arc::new(StringArray::from(vec![Some(m.as_str())])) as ArrayRef,
                None => Arc::new(StringArray::from(vec![None::<&str>])) as ArrayRef,
            };

            columns.push(Arc::new(is_diffusion_array));
            columns.push(modality_array);

            // Add metadata as JSON strings
            if let Some(ref time_aware) = self.time_aware_quant {
                let json = serde_json::to_string(time_aware)?;
                columns.push(Arc::new(StringArray::from(vec![Some(json.as_str())])));
            }

            if let Some(ref spatial) = self.spatial_quant {
                let json = serde_json::to_string(spatial)?;
                columns.push(Arc::new(StringArray::from(vec![Some(json.as_str())])));
            }

            if let Some(ref stats) = self.activation_stats {
                let json = serde_json::to_string(stats)?;
                columns.push(Arc::new(StringArray::from(vec![Some(json.as_str())])));
            }
        }

        // Create RecordBatch
        let batch = RecordBatch::try_new(schema.clone(), columns)?;

        // Write to Parquet with compression
        let file = File::create(path)?;

        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(
                parquet::basic::ZstdLevel::default(),
            ))
            .set_statistics_enabled(parquet::file::properties::EnabledStatistics::None)
            .build();

        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    /// Read from Parquet file
    pub fn read_from_parquet(path: &Path) -> Result<Self> {
        Self::read_from_parquet_with_options(path, false)
    }

    /// Read from Parquet with zero-copy optimization using memory-mapped files
    pub fn read_from_parquet_zero_copy(path: &Path) -> Result<Self> {
        Self::read_from_parquet_with_options(path, true)
    }

    /// Internal method to read from Parquet with configurable memory mapping
    fn read_from_parquet_with_options(path: &Path, use_mmap: bool) -> Result<Self> {
        use arrow::array::{Array, AsArray};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::file::reader::SerializedFileReader;
        use std::fs::File;

        // Detect schema version
        let version = Self::detect_schema_version(path)?;

        // Open Parquet file with optional memory mapping
        let file = File::open(path)?;

        let builder = if use_mmap {
            // Use memory-mapped file for zero-copy reads
            let file_len = file.metadata()?.len() as usize;
            ParquetRecordBatchReaderBuilder::try_new(file)?.with_batch_size(8192)
        // Larger batch size for better performance
        } else {
            // Standard file I/O
            ParquetRecordBatchReaderBuilder::try_new(file)?
        };

        let mut reader = builder.build()?;

        // Read first batch (we write single-row batches)
        let batch = reader.next().ok_or_else(|| {
            crate::errors::QuantError::Internal("Empty Parquet file".to_string())
        })??;

        // Helper to get string value
        let get_string = |col_name: &str| -> Result<String> {
            let col = batch.column_by_name(col_name).ok_or_else(|| {
                crate::errors::QuantError::Internal(format!("Missing column: {}", col_name))
            })?;
            let string_array = col.as_string::<i32>();
            Ok(string_array.value(0).to_string())
        };

        // Helper to get optional string value
        let get_optional_string = |col_name: &str| -> Result<Option<String>> {
            match batch.column_by_name(col_name) {
                Some(col) => {
                    let string_array = col.as_string::<i32>();
                    if string_array.is_null(0) {
                        Ok(None)
                    } else {
                        Ok(Some(string_array.value(0).to_string()))
                    }
                }
                None => Ok(None),
            }
        };

        // Helper to get u64 value
        let get_u64 = |col_name: &str| -> Result<usize> {
            let col = batch.column_by_name(col_name).ok_or_else(|| {
                crate::errors::QuantError::Internal(format!("Missing column: {}", col_name))
            })?;
            let uint_array = col.as_primitive::<arrow::datatypes::UInt64Type>();
            Ok(uint_array.value(0) as usize)
        };

        // Helper to get optional u64 value
        let get_optional_u64 = |col_name: &str| -> Result<Option<usize>> {
            let col = batch.column_by_name(col_name).ok_or_else(|| {
                crate::errors::QuantError::Internal(format!("Missing column: {}", col_name))
            })?;
            let uint_array = col.as_primitive::<arrow::datatypes::UInt64Type>();
            if uint_array.is_null(0) {
                Ok(None)
            } else {
                Ok(Some(uint_array.value(0) as usize))
            }
        };

        // Helper to get list of u64 values
        let get_u64_list = |col_name: &str| -> Result<Vec<usize>> {
            let col = batch.column_by_name(col_name).ok_or_else(|| {
                crate::errors::QuantError::Internal(format!("Missing column: {}", col_name))
            })?;
            let list_array = col.as_list::<i32>();
            let values = list_array.value(0);
            let uint_array = values.as_primitive::<arrow::datatypes::UInt64Type>();
            Ok(uint_array.values().iter().map(|&x| x as usize).collect())
        };

        // Helper to get list of f32 values
        let get_f32_list = |col_name: &str| -> Result<Vec<f32>> {
            let col = batch.column_by_name(col_name).ok_or_else(|| {
                crate::errors::QuantError::Internal(format!("Missing column: {}", col_name))
            })?;
            let list_array = col.as_list::<i32>();
            let values = list_array.value(0);
            let float_array = values.as_primitive::<arrow::datatypes::Float32Type>();
            Ok(float_array.values().to_vec())
        };

        // Helper to get binary data (zero-copy when possible)
        let get_binary = |col_name: &str| -> Result<Vec<u8>> {
            let col = batch.column_by_name(col_name).ok_or_else(|| {
                crate::errors::QuantError::Internal(format!("Missing column: {}", col_name))
            })?;
            let binary_array = col.as_binary::<i64>();

            // Zero-copy: Get reference to underlying buffer
            // Note: We still need to clone for owned data, but this avoids intermediate allocations
            let value_slice = binary_array.value(0);
            Ok(value_slice.to_vec())
        };

        // Read base fields
        let layer_name = get_string("layer_name")?;
        let shape = get_u64_list("shape")?;
        let dtype = get_string("dtype")?;
        let mut data = get_binary("data")?;
        let num_params = get_u64("num_params")?;
        let quant_type = get_string("quant_type")?;
        let scales = get_f32_list("scales")?;
        let zero_points = get_f32_list("zero_points")?;
        let quant_axis = get_optional_u64("quant_axis")?;
        let group_size = get_optional_u64("group_size")?;

        // Fallback: If Parquet 'data' is empty, check for companion .bin file
        if data.is_empty() {
            let bin_path = path.with_extension("bin");
            if bin_path.exists() {
                use std::io::Read;
                let mut bin_file = std::fs::File::open(&bin_path)?;
                let mut bytes = Vec::new();
                bin_file.read_to_end(&mut bytes)?;
                data = bytes;
            }
        }

        // Read diffusion-specific fields if V2Extended
        let (is_diffusion_model, modality, time_aware_quant, spatial_quant, activation_stats) =
            if version == SchemaVersion::V2Extended {
                // Read is_diffusion_model
                let is_diffusion = batch
                    .column_by_name("is_diffusion_model")
                    .map(|col| {
                        let bool_array = col.as_boolean();
                        bool_array.value(0)
                    })
                    .unwrap_or(false);

                // Read modality
                let modality = get_optional_string("modality")?;

                // Read time_aware_quant metadata
                let time_aware = get_optional_string("time_aware_quant_json")?
                    .and_then(|json| serde_json::from_str::<TimeAwareQuantMetadata>(&json).ok());

                // Read spatial_quant metadata
                let spatial = get_optional_string("spatial_quant_json")?
                    .and_then(|json| serde_json::from_str::<SpatialQuantMetadata>(&json).ok());

                // Read activation_stats metadata
                let stats = get_optional_string("activation_stats_json")?
                    .and_then(|json| serde_json::from_str::<ActivationStatsMetadata>(&json).ok());

                (is_diffusion, modality, time_aware, spatial, stats)
            } else {
                // V2 schema - no diffusion metadata
                (false, None, None, None, None)
            };

        Ok(Self {
            layer_name,
            shape,
            dtype,
            data,
            num_params,
            quant_type,
            scales,
            zero_points,
            quant_axis,
            group_size,
            is_diffusion_model,
            modality,
            time_aware_quant,
            spatial_quant,
            activation_stats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_base_schema_creation() {
        let schema = ParquetV2Extended::from_v2_base(
            "test_layer".to_string(),
            vec![256, 512],
            "int8".to_string(),
            vec![0u8; 1024],
            131072,
            "int8".to_string(),
            vec![1.0],
            vec![0.0],
            Some(0),
            Some(128),
        );

        assert_eq!(schema.layer_name, "test_layer");
        assert!(!schema.is_diffusion_model);
        assert!(schema.modality.is_none());
    }

    #[test]
    fn test_time_aware_extension() {
        let base = ParquetV2Extended::from_v2_base(
            "test_layer".to_string(),
            vec![256, 512],
            "int8".to_string(),
            vec![],
            131072,
            "int8".to_string(),
            vec![],
            vec![],
            Some(0),
            Some(128),
        );

        let quantized = QuantizedLayer {
            data: vec![0u8; 1024],
            scales: vec![1.0, 1.0],
            zero_points: vec![0.0, 0.0],
            time_group_params: vec![],
        };

        let extended = base.with_time_aware(Modality::Text, quantized);

        assert!(extended.is_diffusion_model);
        assert_eq!(extended.modality, Some("text".to_string()));
        assert!(extended.time_aware_quant.is_some());
    }

    #[test]
    fn test_write_base_schema() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_base.parquet");

        let schema = ParquetV2Extended::from_v2_base(
            "test_layer".to_string(),
            vec![256, 512],
            "int8".to_string(),
            vec![1u8, 2, 3, 4],
            131072,
            "int8".to_string(),
            vec![1.0],
            vec![0.0],
            Some(0),
            Some(128),
        );

        // Write to Parquet
        let result = schema.write_to_parquet(&path);
        assert!(
            result.is_ok(),
            "Failed to write Parquet: {:?}",
            result.err()
        );

        // Verify file exists
        assert!(path.exists());

        // Verify file is not empty
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_write_time_aware_schema() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_time_aware.parquet");

        let base = ParquetV2Extended::from_v2_base(
            "transformer.layer.0".to_string(),
            vec![768, 768],
            "int2".to_string(),
            vec![],
            589824,
            "int2".to_string(),
            vec![],
            vec![],
            Some(0),
            Some(64),
        );

        let time_group_params = vec![
            crate::time_aware::TimeGroupParams {
                time_range: (0, 100),
                scale: 0.5,
                zero_point: 0.0,
                group_size: 256,
            },
            crate::time_aware::TimeGroupParams {
                time_range: (100, 200),
                scale: 0.3,
                zero_point: 0.0,
                group_size: 128,
            },
        ];

        let quantized = QuantizedLayer {
            data: vec![0u8; 1024],
            scales: vec![0.5, 0.3],
            zero_points: vec![0.0, 0.0],
            time_group_params: time_group_params.clone(),
        };

        let extended = base.with_time_aware(Modality::Text, quantized);

        // Write to Parquet
        let result = extended.write_to_parquet(&path);
        assert!(
            result.is_ok(),
            "Failed to write Parquet: {:?}",
            result.err()
        );

        // Verify file exists
        assert!(path.exists());

        // Verify file is not empty
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_write_spatial_schema() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_spatial.parquet");

        let base = ParquetV2Extended::from_v2_base(
            "conv.layer.0".to_string(),
            vec![512, 512],
            "int4".to_string(),
            vec![],
            262144,
            "int4".to_string(),
            vec![],
            vec![],
            Some(0),
            Some(128),
        );

        let quantized_spatial = QuantizedSpatialLayer {
            data: vec![0u8; 2048],
            scales: vec![0.8, 0.9, 1.0],
            zero_points: vec![0.0, 0.0, 0.0],
            group_size: 128,
        };

        let equalization_scales = vec![1.1, 0.9, 1.0, 0.95];

        let extended = base.with_spatial(Modality::Image, quantized_spatial, equalization_scales);

        // Write to Parquet
        let result = extended.write_to_parquet(&path);
        assert!(
            result.is_ok(),
            "Failed to write Parquet: {:?}",
            result.err()
        );

        // Verify file exists
        assert!(path.exists());

        // Verify file is not empty
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_write_with_all_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_full.parquet");

        let mut schema = ParquetV2Extended::from_v2_base(
            "full_layer".to_string(),
            vec![1024, 1024],
            "int2".to_string(),
            vec![5u8; 512],
            1048576,
            "int2".to_string(),
            vec![0.5, 0.6],
            vec![0.0, 0.0],
            Some(0),
            Some(64),
        );

        // Add all metadata types
        schema.is_diffusion_model = true;
        schema.modality = Some("code".to_string());
        schema.time_aware_quant = Some(TimeAwareQuantMetadata {
            enabled: true,
            num_time_groups: 10,
            time_group_params: vec![],
        });
        schema.spatial_quant = Some(SpatialQuantMetadata {
            enabled: true,
            channel_equalization: true,
            activation_smoothing: true,
            equalization_scales: vec![1.0, 1.1, 0.9],
        });
        schema.activation_stats = Some(ActivationStatsMetadata {
            mean: vec![0.0, 0.1],
            std: vec![1.0, 1.1],
            min: vec![-2.0, -1.5],
            max: vec![2.0, 1.5],
        });

        // Write to Parquet
        let result = schema.write_to_parquet(&path);
        assert!(
            result.is_ok(),
            "Failed to write Parquet: {:?}",
            result.err()
        );

        // Verify file exists
        assert!(path.exists());

        // Verify file is not empty
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_detect_schema_version_v2() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_v2.parquet");

        // Create a base V2 schema (no diffusion fields)
        let schema = ParquetV2Extended::from_v2_base(
            "test_layer".to_string(),
            vec![256, 512],
            "int8".to_string(),
            vec![1u8, 2, 3, 4],
            131072,
            "int8".to_string(),
            vec![1.0],
            vec![0.0],
            Some(0),
            Some(128),
        );

        schema.write_to_parquet(&path).unwrap();

        // Detect version
        let version = ParquetV2Extended::detect_schema_version(&path).unwrap();
        assert_eq!(version, SchemaVersion::V2);
    }

    #[test]
    fn test_detect_schema_version_v2_extended() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_v2_extended.parquet");

        // Create a V2Extended schema with diffusion fields
        let base = ParquetV2Extended::from_v2_base(
            "test_layer".to_string(),
            vec![256, 512],
            "int2".to_string(),
            vec![],
            131072,
            "int2".to_string(),
            vec![],
            vec![],
            Some(0),
            Some(128),
        );

        let quantized = QuantizedLayer {
            data: vec![0u8; 1024],
            scales: vec![1.0, 1.0],
            zero_points: vec![0.0, 0.0],
            time_group_params: vec![],
        };

        let extended = base.with_time_aware(Modality::Text, quantized);
        extended.write_to_parquet(&path).unwrap();

        // Detect version
        let version = ParquetV2Extended::detect_schema_version(&path).unwrap();
        assert_eq!(version, SchemaVersion::V2Extended);
    }

    #[test]
    fn test_read_base_schema() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_read_base.parquet");

        // Write base schema
        let original = ParquetV2Extended::from_v2_base(
            "test_layer".to_string(),
            vec![256, 512],
            "int8".to_string(),
            vec![1u8, 2, 3, 4, 5],
            131072,
            "int8".to_string(),
            vec![1.0, 2.0],
            vec![0.0, 0.5],
            Some(0),
            Some(128),
        );

        original.write_to_parquet(&path).unwrap();

        // Read back
        let read = ParquetV2Extended::read_from_parquet(&path).unwrap();

        // Verify base fields
        assert_eq!(read.layer_name, "test_layer");
        assert_eq!(read.shape, vec![256, 512]);
        assert_eq!(read.dtype, "int8");
        assert_eq!(read.data, vec![1u8, 2, 3, 4, 5]);
        assert_eq!(read.num_params, 131072);
        assert_eq!(read.quant_type, "int8");
        assert_eq!(read.scales, vec![1.0, 2.0]);
        assert_eq!(read.zero_points, vec![0.0, 0.5]);
        assert_eq!(read.quant_axis, Some(0));
        assert_eq!(read.group_size, Some(128));

        // Verify no diffusion metadata
        assert!(!read.is_diffusion_model);
        assert!(read.modality.is_none());
        assert!(read.time_aware_quant.is_none());
        assert!(read.spatial_quant.is_none());
        assert!(read.activation_stats.is_none());
    }

    #[test]
    fn test_read_time_aware_schema() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_read_time_aware.parquet");

        // Write time-aware schema
        let base = ParquetV2Extended::from_v2_base(
            "transformer.layer.0".to_string(),
            vec![768, 768],
            "int2".to_string(),
            vec![],
            589824,
            "int2".to_string(),
            vec![],
            vec![],
            Some(0),
            Some(64),
        );

        let time_group_params = vec![
            crate::time_aware::TimeGroupParams {
                time_range: (0, 100),
                scale: 0.5,
                zero_point: 0.0,
                group_size: 256,
            },
            crate::time_aware::TimeGroupParams {
                time_range: (100, 200),
                scale: 0.3,
                zero_point: 0.0,
                group_size: 128,
            },
        ];

        let quantized = QuantizedLayer {
            data: vec![0u8; 1024],
            scales: vec![0.5, 0.3],
            zero_points: vec![0.0, 0.0],
            time_group_params: time_group_params.clone(),
        };

        let original = base.with_time_aware(Modality::Text, quantized);
        original.write_to_parquet(&path).unwrap();

        // Read back
        let read = ParquetV2Extended::read_from_parquet(&path).unwrap();

        // Verify diffusion metadata
        assert!(read.is_diffusion_model);
        assert_eq!(read.modality, Some("text".to_string()));
        assert!(read.time_aware_quant.is_some());

        let time_aware = read.time_aware_quant.unwrap();
        assert!(time_aware.enabled);
        assert_eq!(time_aware.num_time_groups, 2);
        assert_eq!(time_aware.time_group_params.len(), 2);
        assert_eq!(time_aware.time_group_params[0].time_range, (0, 100));
        assert_eq!(time_aware.time_group_params[1].time_range, (100, 200));
    }

    #[test]
    fn test_read_spatial_schema() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_read_spatial.parquet");

        // Write spatial schema
        let base = ParquetV2Extended::from_v2_base(
            "conv.layer.0".to_string(),
            vec![512, 512],
            "int4".to_string(),
            vec![],
            262144,
            "int4".to_string(),
            vec![],
            vec![],
            Some(0),
            Some(128),
        );

        let quantized_spatial = QuantizedSpatialLayer {
            data: vec![0u8; 2048],
            scales: vec![0.8, 0.9, 1.0],
            zero_points: vec![0.0, 0.0, 0.0],
            group_size: 128,
        };

        let equalization_scales = vec![1.1, 0.9, 1.0, 0.95];

        let original = base.with_spatial(
            Modality::Image,
            quantized_spatial,
            equalization_scales.clone(),
        );
        original.write_to_parquet(&path).unwrap();

        // Read back
        let read = ParquetV2Extended::read_from_parquet(&path).unwrap();

        // Verify diffusion metadata
        assert!(read.is_diffusion_model);
        assert_eq!(read.modality, Some("image".to_string()));
        assert!(read.spatial_quant.is_some());

        let spatial = read.spatial_quant.unwrap();
        assert!(spatial.enabled);
        assert!(spatial.channel_equalization);
        assert_eq!(spatial.equalization_scales, equalization_scales);
    }

    #[test]
    fn test_read_full_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_read_full.parquet");

        // Write schema with all metadata
        let mut original = ParquetV2Extended::from_v2_base(
            "full_layer".to_string(),
            vec![1024, 1024],
            "int2".to_string(),
            vec![5u8; 512],
            1048576,
            "int2".to_string(),
            vec![0.5, 0.6],
            vec![0.0, 0.0],
            Some(0),
            Some(64),
        );

        original.is_diffusion_model = true;
        original.modality = Some("code".to_string());
        original.time_aware_quant = Some(TimeAwareQuantMetadata {
            enabled: true,
            num_time_groups: 10,
            time_group_params: vec![],
        });
        original.spatial_quant = Some(SpatialQuantMetadata {
            enabled: true,
            channel_equalization: true,
            activation_smoothing: true,
            equalization_scales: vec![1.0, 1.1, 0.9],
        });
        original.activation_stats = Some(ActivationStatsMetadata {
            mean: vec![0.0, 0.1],
            std: vec![1.0, 1.1],
            min: vec![-2.0, -1.5],
            max: vec![2.0, 1.5],
        });

        original.write_to_parquet(&path).unwrap();

        // Read back
        let read = ParquetV2Extended::read_from_parquet(&path).unwrap();

        // Verify all fields
        assert_eq!(read.layer_name, "full_layer");
        assert_eq!(read.shape, vec![1024, 1024]);
        assert_eq!(read.dtype, "int2");
        assert_eq!(read.data.len(), 512);
        assert_eq!(read.num_params, 1048576);
        assert_eq!(read.quant_type, "int2");
        assert_eq!(read.scales, vec![0.5, 0.6]);
        assert_eq!(read.zero_points, vec![0.0, 0.0]);
        assert_eq!(read.quant_axis, Some(0));
        assert_eq!(read.group_size, Some(64));

        // Verify diffusion metadata
        assert!(read.is_diffusion_model);
        assert_eq!(read.modality, Some("code".to_string()));

        assert!(read.time_aware_quant.is_some());
        let time_aware = read.time_aware_quant.unwrap();
        assert!(time_aware.enabled);
        assert_eq!(time_aware.num_time_groups, 10);

        assert!(read.spatial_quant.is_some());
        let spatial = read.spatial_quant.unwrap();
        assert!(spatial.enabled);
        assert!(spatial.channel_equalization);
        assert!(spatial.activation_smoothing);
        assert_eq!(spatial.equalization_scales, vec![1.0, 1.1, 0.9]);

        assert!(read.activation_stats.is_some());
        let stats = read.activation_stats.unwrap();
        assert_eq!(stats.mean, vec![0.0, 0.1]);
        assert_eq!(stats.std, vec![1.0, 1.1]);
        assert_eq!(stats.min, vec![-2.0, -1.5]);
        assert_eq!(stats.max, vec![2.0, 1.5]);
    }

    #[test]
    fn test_roundtrip_base_schema() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_roundtrip_base.parquet");

        let original = ParquetV2Extended::from_v2_base(
            "roundtrip_layer".to_string(),
            vec![128, 256, 512],
            "int4".to_string(),
            vec![10u8, 20, 30, 40, 50],
            16777216,
            "int4".to_string(),
            vec![0.1, 0.2, 0.3],
            vec![0.0, 0.5, 1.0],
            None,
            Some(256),
        );

        // Write and read
        original.write_to_parquet(&path).unwrap();
        let read = ParquetV2Extended::read_from_parquet(&path).unwrap();

        // Verify exact match
        assert_eq!(read.layer_name, original.layer_name);
        assert_eq!(read.shape, original.shape);
        assert_eq!(read.dtype, original.dtype);
        assert_eq!(read.data, original.data);
        assert_eq!(read.num_params, original.num_params);
        assert_eq!(read.quant_type, original.quant_type);
        assert_eq!(read.scales, original.scales);
        assert_eq!(read.zero_points, original.zero_points);
        assert_eq!(read.quant_axis, original.quant_axis);
        assert_eq!(read.group_size, original.group_size);
        assert_eq!(read.is_diffusion_model, original.is_diffusion_model);
    }

    #[test]
    fn test_backward_compatibility() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_backward_compat.parquet");

        // Write a V2 base schema
        let v2_schema = ParquetV2Extended::from_v2_base(
            "compat_layer".to_string(),
            vec![512, 512],
            "int8".to_string(),
            vec![1u8; 100],
            262144,
            "int8".to_string(),
            vec![1.0],
            vec![0.0],
            Some(0),
            Some(128),
        );

        v2_schema.write_to_parquet(&path).unwrap();

        // Read should work and detect V2 schema
        let version = ParquetV2Extended::detect_schema_version(&path).unwrap();
        assert_eq!(version, SchemaVersion::V2);

        let read = ParquetV2Extended::read_from_parquet(&path).unwrap();
        assert!(!read.is_diffusion_model);
        assert!(read.modality.is_none());
        assert!(read.time_aware_quant.is_none());
        assert!(read.spatial_quant.is_none());
    }

    #[test]
    fn test_roundtrip_time_aware_schema() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_roundtrip_time_aware.parquet");

        // Create time-aware schema
        let base = ParquetV2Extended::from_v2_base(
            "roundtrip_time_aware".to_string(),
            vec![1024, 2048],
            "int2".to_string(),
            vec![],
            2097152,
            "int2".to_string(),
            vec![],
            vec![],
            Some(0),
            Some(64),
        );

        let time_group_params = vec![
            crate::time_aware::TimeGroupParams {
                time_range: (0, 250),
                scale: 0.45,
                zero_point: 0.1,
                group_size: 256,
            },
            crate::time_aware::TimeGroupParams {
                time_range: (250, 500),
                scale: 0.35,
                zero_point: 0.05,
                group_size: 128,
            },
            crate::time_aware::TimeGroupParams {
                time_range: (500, 750),
                scale: 0.25,
                zero_point: 0.0,
                group_size: 64,
            },
        ];

        let quantized = QuantizedLayer {
            data: vec![42u8; 2048],
            scales: vec![0.45, 0.35, 0.25],
            zero_points: vec![0.1, 0.05, 0.0],
            time_group_params: time_group_params.clone(),
        };

        let original = base.with_time_aware(Modality::Code, quantized);

        // Write and read
        original.write_to_parquet(&path).unwrap();
        let read = ParquetV2Extended::read_from_parquet(&path).unwrap();

        // Verify exact match
        assert_eq!(read.layer_name, original.layer_name);
        assert_eq!(read.shape, original.shape);
        assert_eq!(read.dtype, original.dtype);
        assert_eq!(read.data, original.data);
        assert_eq!(read.num_params, original.num_params);
        assert_eq!(read.quant_type, original.quant_type);
        assert_eq!(read.scales, original.scales);
        assert_eq!(read.zero_points, original.zero_points);
        assert_eq!(read.quant_axis, original.quant_axis);
        assert_eq!(read.group_size, original.group_size);
        assert_eq!(read.is_diffusion_model, original.is_diffusion_model);
        assert_eq!(read.modality, original.modality);

        // Verify time-aware metadata
        assert!(read.time_aware_quant.is_some());
        let read_time_aware = read.time_aware_quant.unwrap();
        let orig_time_aware = original.time_aware_quant.unwrap();
        assert_eq!(read_time_aware.enabled, orig_time_aware.enabled);
        assert_eq!(
            read_time_aware.num_time_groups,
            orig_time_aware.num_time_groups
        );
        assert_eq!(
            read_time_aware.time_group_params.len(),
            orig_time_aware.time_group_params.len()
        );

        for (read_param, orig_param) in read_time_aware
            .time_group_params
            .iter()
            .zip(orig_time_aware.time_group_params.iter())
        {
            assert_eq!(read_param.time_range, orig_param.time_range);
            assert_eq!(read_param.scale, orig_param.scale);
            assert_eq!(read_param.zero_point, orig_param.zero_point);
            assert_eq!(read_param.group_size, orig_param.group_size);
        }
    }

    #[test]
    fn test_roundtrip_spatial_schema() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_roundtrip_spatial.parquet");

        // Create spatial schema
        let base = ParquetV2Extended::from_v2_base(
            "roundtrip_spatial".to_string(),
            vec![2048, 2048],
            "int4".to_string(),
            vec![],
            4194304,
            "int4".to_string(),
            vec![],
            vec![],
            Some(0),
            Some(128),
        );

        let quantized_spatial = QuantizedSpatialLayer {
            data: vec![99u8; 4096],
            scales: vec![0.75, 0.85, 0.95, 1.05],
            zero_points: vec![0.0, 0.1, 0.2, 0.3],
            group_size: 128,
        };

        let equalization_scales = vec![1.05, 0.95, 1.0, 0.98, 1.02];

        let original = base.with_spatial(
            Modality::Audio,
            quantized_spatial,
            equalization_scales.clone(),
        );

        // Write and read
        original.write_to_parquet(&path).unwrap();
        let read = ParquetV2Extended::read_from_parquet(&path).unwrap();

        // Verify exact match
        assert_eq!(read.layer_name, original.layer_name);
        assert_eq!(read.shape, original.shape);
        assert_eq!(read.dtype, original.dtype);
        assert_eq!(read.data, original.data);
        assert_eq!(read.num_params, original.num_params);
        assert_eq!(read.quant_type, original.quant_type);
        assert_eq!(read.scales, original.scales);
        assert_eq!(read.zero_points, original.zero_points);
        assert_eq!(read.quant_axis, original.quant_axis);
        assert_eq!(read.group_size, original.group_size);
        assert_eq!(read.is_diffusion_model, original.is_diffusion_model);
        assert_eq!(read.modality, original.modality);

        // Verify spatial metadata
        assert!(read.spatial_quant.is_some());
        let read_spatial = read.spatial_quant.unwrap();
        let orig_spatial = original.spatial_quant.unwrap();
        assert_eq!(read_spatial.enabled, orig_spatial.enabled);
        assert_eq!(
            read_spatial.channel_equalization,
            orig_spatial.channel_equalization
        );
        assert_eq!(
            read_spatial.activation_smoothing,
            orig_spatial.activation_smoothing
        );
        assert_eq!(
            read_spatial.equalization_scales,
            orig_spatial.equalization_scales
        );
    }

    #[test]
    fn test_roundtrip_full_extended_schema() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_roundtrip_full_extended.parquet");

        // Create schema with all metadata types
        let mut original = ParquetV2Extended::from_v2_base(
            "roundtrip_full".to_string(),
            vec![512, 1024, 2048],
            "int2".to_string(),
            vec![7u8; 1024],
            1073741824,
            "int2".to_string(),
            vec![0.4, 0.5, 0.6],
            vec![0.0, 0.1, 0.2],
            None,
            Some(64),
        );

        original.is_diffusion_model = true;
        original.modality = Some("text".to_string());

        original.time_aware_quant = Some(TimeAwareQuantMetadata {
            enabled: true,
            num_time_groups: 5,
            time_group_params: vec![crate::time_aware::TimeGroupParams {
                time_range: (0, 200),
                scale: 0.5,
                zero_point: 0.0,
                group_size: 256,
            }],
        });

        original.spatial_quant = Some(SpatialQuantMetadata {
            enabled: true,
            channel_equalization: true,
            activation_smoothing: false,
            equalization_scales: vec![1.0, 1.05, 0.95, 1.02],
        });

        original.activation_stats = Some(ActivationStatsMetadata {
            mean: vec![0.0, 0.05, 0.1],
            std: vec![1.0, 1.05, 1.1],
            min: vec![-3.0, -2.5, -2.0],
            max: vec![3.0, 2.5, 2.0],
        });

        // Write and read
        original.write_to_parquet(&path).unwrap();
        let read = ParquetV2Extended::read_from_parquet(&path).unwrap();

        // Verify all base fields
        assert_eq!(read.layer_name, original.layer_name);
        assert_eq!(read.shape, original.shape);
        assert_eq!(read.dtype, original.dtype);
        assert_eq!(read.data, original.data);
        assert_eq!(read.num_params, original.num_params);
        assert_eq!(read.quant_type, original.quant_type);
        assert_eq!(read.scales, original.scales);
        assert_eq!(read.zero_points, original.zero_points);
        assert_eq!(read.quant_axis, original.quant_axis);
        assert_eq!(read.group_size, original.group_size);
        assert_eq!(read.is_diffusion_model, original.is_diffusion_model);
        assert_eq!(read.modality, original.modality);

        // Verify all metadata types
        assert!(read.time_aware_quant.is_some());
        assert!(read.spatial_quant.is_some());
        assert!(read.activation_stats.is_some());

        let read_time = read.time_aware_quant.unwrap();
        let orig_time = original.time_aware_quant.unwrap();
        assert_eq!(read_time.enabled, orig_time.enabled);
        assert_eq!(read_time.num_time_groups, orig_time.num_time_groups);

        let read_spatial = read.spatial_quant.unwrap();
        let orig_spatial = original.spatial_quant.unwrap();
        assert_eq!(read_spatial.enabled, orig_spatial.enabled);
        assert_eq!(
            read_spatial.channel_equalization,
            orig_spatial.channel_equalization
        );
        assert_eq!(
            read_spatial.equalization_scales,
            orig_spatial.equalization_scales
        );

        let read_stats = read.activation_stats.unwrap();
        let orig_stats = original.activation_stats.unwrap();
        assert_eq!(read_stats.mean, orig_stats.mean);
        assert_eq!(read_stats.std, orig_stats.std);
        assert_eq!(read_stats.min, orig_stats.min);
        assert_eq!(read_stats.max, orig_stats.max);
    }

    #[test]
    fn test_schema_version_detection_edge_cases() {
        let dir = tempdir().unwrap();

        // Test 1: Empty diffusion metadata should still be V2Extended
        let path1 = dir.path().join("test_empty_diffusion.parquet");
        let mut schema1 = ParquetV2Extended::from_v2_base(
            "empty_diffusion".to_string(),
            vec![256, 256],
            "int8".to_string(),
            vec![1u8; 10],
            65536,
            "int8".to_string(),
            vec![1.0],
            vec![0.0],
            Some(0),
            Some(128),
        );
        schema1.is_diffusion_model = true;
        schema1.modality = None;
        schema1.write_to_parquet(&path1).unwrap();

        let version1 = ParquetV2Extended::detect_schema_version(&path1).unwrap();
        assert_eq!(version1, SchemaVersion::V2Extended);

        // Test 2: Multiple modalities
        for modality in &["text", "code", "image", "audio"] {
            let path = dir
                .path()
                .join(format!("test_{}_modality.parquet", modality));
            let base = ParquetV2Extended::from_v2_base(
                format!("{}_layer", modality),
                vec![128, 128],
                "int4".to_string(),
                vec![],
                16384,
                "int4".to_string(),
                vec![],
                vec![],
                Some(0),
                Some(64),
            );

            let quantized = QuantizedLayer {
                data: vec![0u8; 256],
                scales: vec![1.0],
                zero_points: vec![0.0],
                time_group_params: vec![],
            };

            let modality_enum = match *modality {
                "text" => Modality::Text,
                "code" => Modality::Code,
                "image" => Modality::Image,
                "audio" => Modality::Audio,
                _ => unreachable!(),
            };

            let extended = base.with_time_aware(modality_enum, quantized);
            extended.write_to_parquet(&path).unwrap();

            let version = ParquetV2Extended::detect_schema_version(&path).unwrap();
            assert_eq!(version, SchemaVersion::V2Extended);

            let read = ParquetV2Extended::read_from_parquet(&path).unwrap();
            assert!(read.is_diffusion_model);
            assert_eq!(read.modality, Some(modality.to_string()));
        }
    }
}

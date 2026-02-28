//! Error types for ArrowQuant V2

use thiserror::Error;

/// Result type alias for ArrowQuant V2 operations
pub type Result<T> = std::result::Result<T, QuantError>;

/// Error types for quantization operations
#[derive(Error, Debug)]
pub enum QuantError {
    #[error("Invalid bit width: {0}. Must be 2, 4, or 8")]
    InvalidBitWidth(u8),

    #[error("Invalid time groups: {0}. Must be between 1 and 100")]
    InvalidTimeGroups(usize),

    #[error("Invalid group size: {0}. Must be 32, 64, 128, or 256")]
    InvalidGroupSize(usize),

    #[error("Invalid accuracy threshold: {0}. Must be between 0.0 and 1.0")]
    InvalidAccuracy(f32),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Unknown modality in model metadata")]
    UnknownModality,

    #[error("Model path not found: {0}")]
    ModelNotFound(String),

    #[error("Failed to read metadata: {0}")]
    MetadataError(String),

    #[error("Quantization failed: {0}")]
    QuantizationFailed(String),

    #[error("Validation failed: cosine similarity {0} below threshold {1}")]
    ValidationFailed(f32, f32),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Arrow error: {0}")]
    ArrowError(#[from] arrow::error::ArrowError),

    #[error("Parquet error: {0}")]
    ParquetError(#[from] parquet::errors::ParquetError),

    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Evolutionary search error: {0}")]
    EvolutionarySearchError(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

#[cfg(feature = "python")]
impl From<QuantError> for pyo3::PyErr {
    fn from(err: QuantError) -> pyo3::PyErr {
        use pyo3::exceptions::PyRuntimeError;
        PyRuntimeError::new_err(err.to_string())
    }
}

//! ArrowQuant V2 for Diffusion - Rust Quantization Engine
//!
//! High-performance quantization library for diffusion models with:
//! - Time-aware quantization (temporal variance handling)
//! - Spatial quantization (channel equalization, activation smoothing)
//! - Extended Parquet V2 schema with diffusion metadata
//! - PyO3 Python bindings for seamless integration

pub mod buffer_pool;
pub mod calibration;
pub mod config;
pub mod errors;
pub mod evolutionary;
pub mod granularity;
pub mod memory_scheduler;
pub mod orchestrator;
pub mod python;
pub mod python_async;
pub mod safetensors_adapter;
pub mod safetensors_to_parquet;
pub mod schema;
pub mod sharded_safetensors;
pub mod simd;
pub mod spatial;
pub mod thermodynamic;
pub mod time_aware;
pub mod validation;

// Re-export commonly used types
pub use config::{
    BoundarySmoothingConfig, DiffusionQuantConfig, InterpolationMethod, Modality,
    ThermodynamicConfig, ValidationConfig,
};
pub use errors::{QuantError, Result};
pub use orchestrator::DiffusionOrchestrator;
pub use safetensors_adapter::SafeTensorsAdapter;
pub use schema::{ParquetV2Extended, SchemaVersion};
pub use sharded_safetensors::ShardedSafeTensorsAdapter;
pub use thermodynamic::ThermodynamicMetrics;

// PyO3 module definition
use pyo3::prelude::*;

#[pymodule]
fn arrow_quant_v2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register Python classes
    m.add_class::<python::PyDiffusionQuantConfig>()?;
    m.add_class::<python::ArrowQuantV2>()?;
    m.add_class::<python_async::AsyncArrowQuantV2>()?;
    m.add_class::<python::PyShardedSafeTensorsLoader>()?;
    
    // Register Python functions
    m.add_function(wrap_pyfunction!(python::quantize_diffusion_model, m)?)?;
    m.add_function(wrap_pyfunction!(python::load_sharded_safetensors, m)?)?;
    
    Ok(())
}

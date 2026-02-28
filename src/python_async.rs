//! Async PyO3 Python bindings for non-blocking quantization operations

use crate::config::DiffusionQuantConfig;
use crate::orchestrator::{DiffusionOrchestrator, QuantizationResult};
use crate::python::{convert_error, PyDiffusionQuantConfig};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Async progress reporter for Python callbacks
#[derive(Clone)]
struct AsyncProgressReporter {
    callback: Option<Arc<Mutex<PyObject>>>,
    last_report_time: Arc<Mutex<Instant>>,
}

impl AsyncProgressReporter {
    fn new(callback: Option<PyObject>) -> Self {
        Self {
            callback: callback.map(|cb| Arc::new(Mutex::new(cb))),
            last_report_time: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Report progress to Python async callback
    async fn report(&self, message: &str, progress: f32) {
        if let Some(callback) = &self.callback {
            Python::with_gil(|py| {
                if let Ok(cb) = callback.lock() {
                    // Try to call the callback, but don't fail if it errors
                    if let Err(e) = cb.call1(py, (message, progress)) {
                        eprintln!("Async progress callback error (ignored): {}", e);
                    }
                }
            });
        }
    }

    /// Report progress with time throttling (only report every 5 seconds)
    async fn report_throttled(&self, message: &str, progress: f32) {
        if let Ok(mut last_time) = self.last_report_time.lock() {
            let now = Instant::now();
            if now.duration_since(*last_time).as_secs() >= 5 {
                self.report(message, progress).await;
                *last_time = now;
            }
        }
    }
}

/// Async Python wrapper for ArrowQuant V2
#[pyclass(name = "AsyncArrowQuantV2")]
pub struct AsyncArrowQuantV2 {
    config: Option<DiffusionQuantConfig>,
}

#[pymethods]
impl AsyncArrowQuantV2 {
    #[new]
    /// Create a new AsyncArrowQuantV2 instance.
    ///
    /// Returns:
    ///     AsyncArrowQuantV2 instance
    fn new() -> Self {
        Self { config: None }
    }

    /// Asynchronously quantize a diffusion model with diffusion-specific optimizations.
    ///
    /// This method is non-blocking and returns a coroutine that can be awaited in Python.
    ///
    /// Args:
    ///     model_path: Path to input model directory
    ///     output_path: Path to output quantized model directory
    ///     config: Optional DiffusionQuantConfig for quantization parameters
    ///     progress_callback: Optional Python async callback function for progress updates
    ///                       Callback signature: async fn(message: str, progress: float) -> None
    ///                       - message: Human-readable progress message
    ///                       - progress: Float between 0.0 and 1.0 indicating completion
    ///
    /// Returns:
    ///     Coroutine that resolves to a dictionary containing:
    ///         - quantized_path: Path to quantized model
    ///         - compression_ratio: Compression ratio achieved
    ///         - cosine_similarity: Average cosine similarity
    ///         - model_size_mb: Size of quantized model in MB
    ///         - modality: Detected modality (text, code, image, audio)
    ///         - bit_width: Bit width used for quantization
    ///         - quantization_time_s: Time taken for quantization in seconds
    ///
    /// Raises:
    ///     QuantizationError: If quantization fails
    ///     ConfigurationError: If configuration is invalid
    ///
    /// Example:
    ///     ```python
    ///     import asyncio
    ///     from arrow_quant_v2 import AsyncArrowQuantV2, DiffusionQuantConfig
    ///
    ///     async def main():
    ///         quantizer = AsyncArrowQuantV2()
    ///         result = await quantizer.quantize_diffusion_model_async(
    ///             model_path="dream-7b/",
    ///             output_path="dream-7b-int2/",
    ///             config=DiffusionQuantConfig(bit_width=2)
    ///         )
    ///         print(f"Compression ratio: {result['compression_ratio']}")
    ///
    ///     asyncio.run(main())
    ///     ```
    #[pyo3(signature = (model_path, output_path, config=None, progress_callback=None))]
    fn quantize_diffusion_model_async<'py>(
        &mut self,
        py: Python<'py>,
        model_path: String,
        output_path: String,
        config: Option<PyDiffusionQuantConfig>,
        progress_callback: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let config = match config {
            Some(c) => c.inner,
            None => DiffusionQuantConfig::default(),
        };

        // Store config for potential future use
        self.config = Some(config.clone());

        let model_path = PathBuf::from(model_path);
        let output_path = PathBuf::from(output_path);

        // Create async task
        future_into_py(py, async move {
            // Create progress reporter
            let progress_reporter = AsyncProgressReporter::new(progress_callback);

            // Report start
            progress_reporter.report("Starting async quantization...", 0.0).await;

            // Execute quantization in background thread to avoid blocking
            let result = tokio::task::spawn_blocking(move || {
                quantize_model_blocking(&model_path, &output_path, &config)
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Task join error: {}", e))
            })?
            .map_err(convert_error)?;

            // Report completion
            progress_reporter.report("Async quantization complete", 1.0).await;

            // Convert result to Python dict
            Python::with_gil(|py| {
                let mut dict = HashMap::new();
                dict.insert(
                    "quantized_path".to_string(),
                    result.quantized_path.to_str().unwrap().to_object(py),
                );
                dict.insert(
                    "compression_ratio".to_string(),
                    result.compression_ratio.to_object(py),
                );
                dict.insert(
                    "cosine_similarity".to_string(),
                    result.cosine_similarity.to_object(py),
                );
                dict.insert(
                    "model_size_mb".to_string(),
                    result.model_size_mb.to_object(py),
                );
                dict.insert(
                    "modality".to_string(),
                    result.modality.to_string().to_object(py),
                );
                dict.insert("bit_width".to_string(), result.bit_width.to_object(py));
                dict.insert(
                    "quantization_time_s".to_string(),
                    result.quantization_time_s.to_object(py),
                );
                Ok(dict)
            })
        })
    }

    /// Asynchronously quantize multiple diffusion models concurrently.
    ///
    /// This method allows concurrent quantization of multiple models, which can
    /// significantly speed up batch processing on multi-core systems.
    ///
    /// Args:
    ///     models: List of tuples (model_path, output_path, config)
    ///             Each tuple contains:
    ///             - model_path: Path to input model directory
    ///             - output_path: Path to output quantized model directory
    ///             - config: Optional DiffusionQuantConfig (None uses default)
    ///     progress_callback: Optional Python async callback function for progress updates
    ///                       Callback signature: async fn(model_idx: int, message: str, progress: float) -> None
    ///
    /// Returns:
    ///     Coroutine that resolves to a list of dictionaries, one per model
    ///
    /// Raises:
    ///     QuantizationError: If any quantization fails
    ///
    /// Example:
    ///     ```python
    ///     import asyncio
    ///     from arrow_quant_v2 import AsyncArrowQuantV2, DiffusionQuantConfig
    ///
    ///     async def main():
    ///         quantizer = AsyncArrowQuantV2()
    ///         models = [
    ///             ("model1/", "model1-int2/", DiffusionQuantConfig(bit_width=2)),
    ///             ("model2/", "model2-int4/", DiffusionQuantConfig(bit_width=4)),
    ///             ("model3/", "model3-int8/", None),  # Use default config
    ///         ]
    ///         results = await quantizer.quantize_multiple_models_async(models)
    ///         for i, result in enumerate(results):
    ///             print(f"Model {i}: {result['compression_ratio']}x compression")
    ///
    ///     asyncio.run(main())
    ///     ```
    #[pyo3(signature = (models, progress_callback=None))]
    fn quantize_multiple_models_async<'py>(
        &self,
        py: Python<'py>,
        models: Vec<(String, String, Option<PyDiffusionQuantConfig>)>,
        progress_callback: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Clone callback for each task before entering async context
        let callbacks: Vec<Option<PyObject>> = (0..models.len())
            .map(|_| progress_callback.as_ref().map(|cb| cb.clone_ref(py)))
            .collect();
        
        future_into_py(py, async move {
            let num_models = models.len();
            let mut tasks = Vec::new();

            // Create async task for each model
            for (idx, ((model_path, output_path, config), callback)) in 
                models.into_iter().zip(callbacks.into_iter()).enumerate() {
                let config = match config {
                    Some(c) => c.inner,
                    None => DiffusionQuantConfig::default(),
                };

                let model_path = PathBuf::from(model_path);
                let output_path = PathBuf::from(output_path);

                // Spawn concurrent task
                let task = tokio::task::spawn(async move {
                    let progress_reporter = AsyncProgressReporter::new(callback);

                    // Report start for this model
                    progress_reporter
                        .report(
                            &format!("Starting quantization for model {}/{}", idx + 1, num_models),
                            0.0,
                        )
                        .await;

                    // Execute quantization in background thread
                    let result = tokio::task::spawn_blocking(move || {
                        quantize_model_blocking(&model_path, &output_path, &config)
                    })
                    .await
                    .map_err(|e| {
                        crate::errors::QuantError::Internal(format!("Task join error: {}", e))
                    })?;

                    // Report completion for this model
                    progress_reporter
                        .report(
                            &format!("Completed quantization for model {}/{}", idx + 1, num_models),
                            1.0,
                        )
                        .await;

                    result
                });

                tasks.push(task);
            }

            // Wait for all tasks to complete
            let results = futures::future::join_all(tasks).await;

            // Convert results to Python list
            Python::with_gil(|py| {
                let mut py_results = Vec::new();

                for result in results {
                    let result = result
                        .map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "Task join error: {}",
                                e
                            ))
                        })?
                        .map_err(convert_error)?;

                    let mut dict = HashMap::new();
                    dict.insert(
                        "quantized_path".to_string(),
                        result.quantized_path.to_str().unwrap().to_object(py),
                    );
                    dict.insert(
                        "compression_ratio".to_string(),
                        result.compression_ratio.to_object(py),
                    );
                    dict.insert(
                        "cosine_similarity".to_string(),
                        result.cosine_similarity.to_object(py),
                    );
                    dict.insert(
                        "model_size_mb".to_string(),
                        result.model_size_mb.to_object(py),
                    );
                    dict.insert(
                        "modality".to_string(),
                        result.modality.to_string().to_object(py),
                    );
                    dict.insert("bit_width".to_string(), result.bit_width.to_object(py));
                    dict.insert(
                        "quantization_time_s".to_string(),
                        result.quantization_time_s.to_object(py),
                    );

                    py_results.push(dict);
                }

                Ok(py_results)
            })
        })
    }

    /// Asynchronously validate quantization quality.
    ///
    /// Args:
    ///     original_path: Path to original model directory
    ///     quantized_path: Path to quantized model directory
    ///
    /// Returns:
    ///     Coroutine that resolves to a dictionary containing validation results
    ///
    /// Raises:
    ///     ValidationError: If validation fails
    #[pyo3(signature = (original_path, quantized_path))]
    fn validate_quality_async<'py>(
        &self,
        py: Python<'py>,
        original_path: String,
        quantized_path: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let original_path = PathBuf::from(original_path);
        let quantized_path = PathBuf::from(quantized_path);

        future_into_py(py, async move {
            use crate::validation::ValidationSystem;

            // Execute validation in background thread
            let report = tokio::task::spawn_blocking(move || {
                let validator = ValidationSystem::new(0.70);
                validator.validate_quality(&original_path, &quantized_path)
            })
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Task join error: {}", e))
            })?
            .map_err(convert_error)?;

            // Convert result to Python dict
            Python::with_gil(|py| {
                let mut dict = HashMap::new();
                dict.insert(
                    "cosine_similarity".to_string(),
                    report.cosine_similarity.to_object(py),
                );
                dict.insert(
                    "compression_ratio".to_string(),
                    report.compression_ratio.to_object(py),
                );
                dict.insert(
                    "model_size_mb".to_string(),
                    report.model_size_mb.to_object(py),
                );
                dict.insert("passed".to_string(), report.passed.to_object(py));

                // Convert per_layer_accuracy HashMap to Python dict
                let per_layer_dict = pyo3::types::PyDict::new_bound(py);
                for (layer, accuracy) in report.per_layer_accuracy.iter() {
                    per_layer_dict.set_item(layer, accuracy).unwrap();
                }
                dict.insert(
                    "per_layer_accuracy".to_string(),
                    per_layer_dict.to_object(py),
                );

                Ok(dict)
            })
        })
    }
}

/// Blocking quantization function (runs in background thread)
fn quantize_model_blocking(
    model_path: &PathBuf,
    output_path: &PathBuf,
    config: &DiffusionQuantConfig,
) -> crate::errors::Result<QuantizationResult> {
    let orchestrator = DiffusionOrchestrator::new(config.clone())?;
    orchestrator.quantize_model(model_path, output_path)
}

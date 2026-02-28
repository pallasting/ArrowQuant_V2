/// Arrow IPC wrapper methods that use quantize_batch as backend
///
/// These methods provide a PyArrow Table interface while internally using
/// the proven quantize_batch implementation. This avoids Arrow FFI complexity
/// while still providing a convenient Arrow-based API.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

impl crate::python::ArrowQuantV2 {
    /// Quantize weights using PyArrow Table (convenience wrapper).
    ///
    /// This method provides a PyArrow Table interface for quantization by internally
    /// converting to/from the batch API format. It's designed for convenience rather
    /// than maximum performance.
    ///
    /// # Arguments
    ///
    /// * `table` - PyArrow Table with schema:
    ///   - layer_name: string
    ///   - weights: list<float> or list<double>
    ///   - shape: list<int64> (optional)
    /// * `bit_width` - Target bit width (2, 4, or 8). Default: 4
    ///
    /// # Returns
    ///
    /// PyArrow Table with schema:
    /// - layer_name: string
    /// - quantized_data: binary
    /// - scales: list<float32>
    /// - zero_points: list<float32>
    /// - shape: list<int64>
    /// - bit_width: uint8
    ///
    /// # Performance
    ///
    /// This method involves data conversion:
    /// - Input: PyArrow → Python lists → numpy arrays
    /// - Output: Python dict → PyArrow Table
    ///
    /// For maximum performance with large models, consider using `quantize_batch()`
    /// directly with numpy arrays.
    ///
    /// # Example
    ///
    /// ```python
    /// import pyarrow as pa
    /// import numpy as np
    /// from arrow_quant_v2 import ArrowQuantV2
    ///
    /// # Create input table
    /// table = pa.Table.from_pydict({
    ///     "layer_name": ["layer.0.weight", "layer.1.weight"],
    ///     "weights": [
    ///         np.random.randn(1000).astype(np.float32).tolist(),
    ///         np.random.randn(2000).astype(np.float32).tolist(),
    ///     ],
    ///     "shape": [[1000], [2000]],
    /// })
    ///
    /// # Quantize
    /// quantizer = ArrowQuantV2(mode="diffusion")
    /// result_table = quantizer.quantize_arrow(table, bit_width=4)
    /// ```
    #[pyo3(signature = (table, bit_width=None))]
    pub fn quantize_arrow_wrapper(
        &self,
        table: &Bound<'_, PyAny>,
        bit_width: Option<u8>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let bit_width = bit_width.unwrap_or(4);

            // Validate bit width
            if ![2, 4, 8].contains(&bit_width) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Invalid bit_width: {}. Must be 2, 4, or 8", bit_width)
                ));
            }

            // Convert PyArrow Table to Python dict
            // table.to_pydict() returns {'column_name': [values...]}
            let table_dict = table.call_method0("to_pydict")?;
            
            // Extract columns
            let layer_names: Vec<String> = table_dict
                .get_item("layer_name")?
                .extract()?;
            
            let weights_lists: Vec<Vec<f64>> = table_dict
                .get_item("weights")?
                .extract()?;
            
            // Optional shape column
            let shapes_lists: Option<Vec<Vec<i64>>> = table_dict
                .get_item("shape")
                .ok()
                .and_then(|item| item.extract().ok());

            // Import numpy to create arrays
            let np = py.import_bound("numpy")?;
            
            // Build weights dict for quantize_batch
            let weights_dict = PyDict::new_bound(py);
            
            for (i, layer_name) in layer_names.iter().enumerate() {
                // Convert weights list to numpy array
                let weights_list = &weights_lists[i];
                let weights_array = np.call_method1(
                    "array",
                    (weights_list, py.eval_bound("numpy.float32", None, None)?)
                )?;
                
                weights_dict.set_item(layer_name, weights_array)?;
            }

            // Call quantize_batch (the proven implementation)
            let batch_results = self.quantize_batch(&weights_dict, Some(bit_width), None)?;

            // Convert batch results to PyArrow Table format
            let result_layer_names = PyList::empty_bound(py);
            let result_quantized_data = PyList::empty_bound(py);
            let result_scales = PyList::empty_bound(py);
            let result_zero_points = PyList::empty_bound(py);
            let result_shapes = PyList::empty_bound(py);
            let result_bit_widths = PyList::empty_bound(py);

            // Process results in same order as input
            for layer_name in &layer_names {
                if let Some(layer_result) = batch_results.get(layer_name) {
                    let layer_result_dict: &Bound<'_, PyDict> = layer_result.downcast_bound(py)?;
                    
                    result_layer_names.append(layer_name)?;
                    result_quantized_data.append(layer_result_dict.get_item("quantized_data")?)?;
                    result_scales.append(layer_result_dict.get_item("scales")?)?;
                    result_zero_points.append(layer_result_dict.get_item("zero_points")?)?;
                    result_shapes.append(layer_result_dict.get_item("shape")?)?;
                    result_bit_widths.append(layer_result_dict.get_item("bit_width")?)?;
                }
            }

            // Create result dict
            let result_dict = PyDict::new_bound(py);
            result_dict.set_item("layer_name", result_layer_names)?;
            result_dict.set_item("quantized_data", result_quantized_data)?;
            result_dict.set_item("scales", result_scales)?;
            result_dict.set_item("zero_points", result_zero_points)?;
            result_dict.set_item("shape", result_shapes)?;
            result_dict.set_item("bit_width", result_bit_widths)?;

            // Create PyArrow Table from dict
            let pyarrow = py.import_bound("pyarrow")?;
            let result_table = pyarrow.call_method1("table", (result_dict,))?;

            Ok(result_table.to_object(py))
        })
    }

    /// Quantize weights using PyArrow RecordBatch (convenience wrapper).
    ///
    /// Similar to `quantize_arrow_wrapper()` but accepts RecordBatch instead of Table.
    /// Internally converts to Table and uses the same implementation.
    ///
    /// # Example
    ///
    /// ```python
    /// import pyarrow as pa
    /// import numpy as np
    /// from arrow_quant_v2 import ArrowQuantV2
    ///
    /// # Create RecordBatch
    /// batch = pa.RecordBatch.from_pydict({
    ///     "layer_name": ["layer.0.weight"],
    ///     "weights": [np.random.randn(1000).astype(np.float32).tolist()],
    ///     "shape": [[1000]],
    /// })
    ///
    /// # Quantize
    /// quantizer = ArrowQuantV2(mode="diffusion")
    /// result_batch = quantizer.quantize_arrow_batch(batch, bit_width=4)
    /// ```
    #[pyo3(signature = (record_batch, bit_width=None))]
    pub fn quantize_arrow_batch_wrapper(
        &self,
        record_batch: &Bound<'_, PyAny>,
        bit_width: Option<u8>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            // Convert RecordBatch to Table
            let pyarrow = py.import_bound("pyarrow")?;
            let table = pyarrow.call_method1("Table", (
                pyarrow.call_method1("from_batches", ([record_batch],))?,
            ))?;

            // Use quantize_arrow_wrapper
            let result_table = self.quantize_arrow_wrapper(&table, bit_width)?;

            // Convert back to RecordBatch
            let result_table_bound = result_table.bind(py);
            let batches = result_table_bound.call_method0("to_batches")?;
            let batch_list: Vec<PyObject> = batches.extract()?;
            
            if batch_list.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Result table has no batches"
                ));
            }

            Ok(batch_list[0].clone())
        })
    }
}

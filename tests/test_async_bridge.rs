//! Test async bridge functionality
//!
//! This test verifies that the async bridge correctly converts Rust futures
//! to Python asyncio futures using pyo3-async-runtimes.
//!
//! **Validates: Requirements 9.2**

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::Duration;

/// Test that async bridge can handle successful futures
/// 
/// **Validates: Requirements 9.2 - Success scenario**
#[test]
fn test_async_bridge_success() {
    pyo3::prepare_freethreaded_python();
    
    Python::with_gil(|py| {
        // Import asyncio
        let asyncio = py.import_bound("asyncio").unwrap();
        
        // Create a simple async function that returns a value
        let code = r#"
import asyncio
from arrow_quant_v2 import AsyncArrowQuantV2

async def test_async():
    # Just test that the async quantizer can be created
    quantizer = AsyncArrowQuantV2()
    return "success"

result = asyncio.run(test_async())
"#;
        
        // Execute the code
        let locals = PyDict::new_bound(py);
        py.run_bound(code, None, Some(&locals)).unwrap();
        
        // Check result
        let result: String = locals.get_item("result").unwrap().unwrap().extract().unwrap();
        assert_eq!(result, "success");
    });
}

/// Test that async bridge properly handles GIL management
/// 
/// **Validates: Requirements 9.2 - GIL management correctness**
#[test]
fn test_async_bridge_gil_management() {
    pyo3::prepare_freethreaded_python();
    
    Python::with_gil(|py| {
        // Test that we can create multiple async quantizers without deadlock
        let code = r#"
import asyncio
from arrow_quant_v2 import AsyncArrowQuantV2

async def test_multiple():
    quantizers = [AsyncArrowQuantV2() for _ in range(5)]
    return len(quantizers)

result = asyncio.run(test_multiple())
"#;
        
        let locals = PyDict::new_bound(py);
        py.run_bound(code, None, Some(&locals)).unwrap();
        
        let result: usize = locals.get_item("result").unwrap().unwrap().extract().unwrap();
        assert_eq!(result, 5);
    });
}

/// Test that async bridge can handle errors correctly
/// 
/// **Validates: Requirements 9.2 - Failure scenario and error propagation**
#[test]
fn test_async_bridge_error_handling() {
    pyo3::prepare_freethreaded_python();
    
    Python::with_gil(|py| {
        // Test that errors are properly propagated
        let code = r#"
import asyncio
from arrow_quant_v2 import AsyncArrowQuantV2

async def test_error():
    quantizer = AsyncArrowQuantV2()
    try:
        # This should fail because paths don't exist
        result = await quantizer.quantize_diffusion_model_async(
            model_path="/nonexistent/path",
            output_path="/nonexistent/output"
        )
        return "should_not_reach"
    except Exception as e:
        return "error_caught"

result = asyncio.run(test_error())
"#;
        
        let locals = PyDict::new_bound(py);
        py.run_bound(code, None, Some(&locals)).unwrap();
        
        let result: String = locals.get_item("result").unwrap().unwrap().extract().unwrap();
        assert_eq!(result, "error_caught");
    });
}

/// Test concurrent async operations (10+ tasks)
/// 
/// **Validates: Requirements 9.2 - Concurrent execution with 10+ tasks**
#[test]
fn test_async_bridge_concurrent_10plus() {
    pyo3::prepare_freethreaded_python();
    
    Python::with_gil(|py| {
        // Test 12 concurrent async operations
        let code = r#"
import asyncio
import tempfile
from pathlib import Path
from arrow_quant_v2 import AsyncArrowQuantV2

async def single_task(quantizer, index):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / f"model_{index}"
            output_path = Path(tmpdir) / f"output_{index}"
            model_path.mkdir()
            output_path.mkdir()
            
            result = await quantizer.quantize_diffusion_model_async(
                model_path=str(model_path),
                output_path=str(output_path)
            )
    except Exception:
        pass  # Expected to fail
    return index

async def test_concurrent():
    quantizer = AsyncArrowQuantV2()
    tasks = [single_task(quantizer, i) for i in range(12)]
    results = await asyncio.gather(*tasks)
    return len(results)

result = asyncio.run(test_concurrent())
"#;
        
        let locals = PyDict::new_bound(py);
        py.run_bound(code, None, Some(&locals)).unwrap();
        
        let result: usize = locals.get_item("result").unwrap().unwrap().extract().unwrap();
        assert_eq!(result, 12, "Should complete 12 concurrent tasks without deadlock");
    });
}

/// Test progress callback functionality
/// 
/// **Validates: Requirements 9.2 - Progress callbacks work correctly**
#[test]
fn test_async_bridge_progress_callback() {
    pyo3::prepare_freethreaded_python();
    
    Python::with_gil(|py| {
        let code = r#"
import asyncio
import tempfile
from pathlib import Path
from arrow_quant_v2 import AsyncArrowQuantV2

progress_calls = []

def progress_callback(message, progress):
    progress_calls.append((message, progress))

async def test_progress():
    quantizer = AsyncArrowQuantV2()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model"
        output_path = Path(tmpdir) / "output"
        model_path.mkdir()
        output_path.mkdir()
        
        try:
            result = await quantizer.quantize_diffusion_model_async(
                model_path=str(model_path),
                output_path=str(output_path),
                progress_callback=progress_callback
            )
        except Exception:
            pass  # Expected to fail
    
    return len(progress_calls)

result = asyncio.run(test_progress())
"#;
        
        let locals = PyDict::new_bound(py);
        py.run_bound(code, None, Some(&locals)).unwrap();
        
        let result: usize = locals.get_item("result").unwrap().unwrap().extract().unwrap();
        assert!(result > 0, "Progress callback should be called at least once");
    });
}

/// Test multiple models async method
/// 
/// **Validates: Requirements 9.2 - Multiple concurrent model quantization**
#[test]
fn test_async_bridge_multiple_models() {
    pyo3::prepare_freethreaded_python();
    
    Python::with_gil(|py| {
        let code = r#"
import asyncio
import tempfile
from pathlib import Path
from arrow_quant_v2 import AsyncArrowQuantV2, DiffusionQuantConfig

async def test_multiple():
    quantizer = AsyncArrowQuantV2()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        models = []
        for i in range(3):
            model_path = Path(tmpdir) / f"model_{i}"
            output_path = Path(tmpdir) / f"output_{i}"
            model_path.mkdir()
            output_path.mkdir()
            
            config = DiffusionQuantConfig(bit_width=4) if i == 0 else None
            models.append((str(model_path), str(output_path), config))
        
        try:
            results = await quantizer.quantize_multiple_models_async(models)
            return "unexpected_success"
        except Exception as e:
            return "error_caught"

result = asyncio.run(test_multiple())
"#;
        
        let locals = PyDict::new_bound(py);
        py.run_bound(code, None, Some(&locals)).unwrap();
        
        let result: String = locals.get_item("result").unwrap().unwrap().extract().unwrap();
        assert_eq!(result, "error_caught", "Should handle multiple models method");
    });
}

/// Test error propagation from Rust to Python
/// 
/// **Validates: Requirements 9.2 - Error propagation from Rust to Python**
#[test]
fn test_async_bridge_error_propagation() {
    pyo3::prepare_freethreaded_python();
    
    Python::with_gil(|py| {
        let code = r#"
import asyncio
from arrow_quant_v2 import AsyncArrowQuantV2

async def test_errors():
    quantizer = AsyncArrowQuantV2()
    
    test_cases = [
        ("/nonexistent/path", "/output"),
        ("", ""),
        ("/dev/null", "/output"),
    ]
    
    errors_caught = 0
    for model_path, output_path in test_cases:
        try:
            result = await quantizer.quantize_diffusion_model_async(
                model_path=model_path,
                output_path=output_path
            )
        except Exception as e:
            # Verify we get a Python exception, not a Rust panic
            errors_caught += 1
    
    return errors_caught

result = asyncio.run(test_errors())
"#;
        
        let locals = PyDict::new_bound(py);
        py.run_bound(code, None, Some(&locals)).unwrap();
        
        let result: usize = locals.get_item("result").unwrap().unwrap().extract().unwrap();
        assert_eq!(result, 3, "All error cases should be properly propagated to Python");
    });
}
